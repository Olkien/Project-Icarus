import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from analysis.decode_smooth import REFERENCE, RULE_S4_HITS, SCENARIOS, SEEDS
from analysis.s4_parts import episode, finished, row
from analysis.yaw_damp import PID_KD, fly_one, setting_dir
from analysis.yaw_damp import SETTINGS as EARLIER
from env.runlog import run_path
from fly.check_load import REPO_ROOT, Report

OUT_PATH = REPO_ROOT / "results" / "yaw_damp_matched_s0.55.txt"
SETTINGS = (0.05, 0.08, 0.11)
SAT = 0.999999


def paths_for(name: str, label: str, kd: float | None) -> list[Path]:
    ctrl = "pid" if label == "pid" else "fly_real"
    folder = setting_dir(kd) if kd is not None else REFERENCE[(name, label)]
    return [run_path(name, ctrl, s, folder) for s in SEEDS]


def tick_stats(paths: list[Path]) -> tuple[float, float, float]:
    obs_sd, true_sd, sat = [], [], []
    for p in paths:
        ticks = [json.loads(raw) for raw in p.read_text(encoding="utf-8").splitlines() if '"type":"tick"' in raw]
        obs_sd.append(np.std([t["obs"]["yaw_rate"] for t in ticks]))
        true_sd.append(np.std([t["state"]["omega"] for t in ticks]))
        sat.append(np.mean([abs(t["action"]["yaw_torque"]) >= SAT for t in ticks]))
    return float(np.mean(obs_sd)), float(np.mean(true_sd)), float(np.mean(sat))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    start = time.perf_counter()
    jobs = [(n, kd, s, str(run_path(n, "fly_real", s, setting_dir(kd))))
            for kd in SETTINGS for n in SCENARIOS for s in SEEDS]
    jobs = [j for j in jobs if not finished(Path(j[3]))]
    for kd in SETTINGS:
        setting_dir(kd).mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} flights to run, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(fly_one, j) for j in jobs]):
            name, kd, seed, hits = fut.result()
            print(f"{name} kd{kd:g} s{seed}: hits {hits}", flush=True)
    rep = Report()
    rep.say(f"matched-brake test, diagnostic only, seeds 0-4, fly_real plus yaw_damp kd; PID Kd {PID_KD:g}; "
            f"new kd {', '.join(f'{k:g}' for k in SETTINGS)}; earlier kd {', '.join(f'{k:g}' for k in EARLIER)} "
            "read from their runs for context")
    rep.say(f"rule fixed before the run: supported only if for some new kd S4 mean hits >= {RULE_S4_HITS:g}, "
            "S1 5/5, delay-alone 5/5, and 0 lock-ins over the 15 flights of that kd. Nothing adopted.")
    rep.say("time per target = mean over episodes with a hit; closest = smallest true distance to the "
            "active target, mean over episodes that timed out")
    rep.say("variant controller success mean_hits mean_time_per_target n_timed closest_mm n_timeout "
            "lockins hits_per_seed")
    kds = (None, *SETTINGS, *EARLIER)
    for name in SCENARIOS:
        rep.say(row(name, "pid_40", paths_for(name, "pid", None)))
        for kd in kds:
            rep.say(row(name, f"fly_kd{kd or 0:g}", paths_for(name, "fly", kd)))
    rep.say("tick stats: yaw_rate sd (obs, true omega) per episode, then mean over seeds; saturated = "
            "fraction of ticks with |yaw_torque| at the clip limit 1")
    for name in SCENARIOS:
        for label, kd in (("pid_40", None), *((f"fly_kd{kd or 0:g}", kd) for kd in kds)):
            obs_sd, true_sd, sat = tick_stats(paths_for(name, "pid" if label == "pid_40" else "fly", kd))
            rep.say(f"  {name} {label}: yaw_rate sd obs {obs_sd:.3f}, true {true_sd:.3f}, saturated {sat:.3f}")
    supported = False
    for kd in SETTINGS:
        eps = {n: [episode(p) for p in paths_for(n, "fly", kd)] for n in SCENARIOS}
        s4_hits = float(np.mean([e["hits"] for e in eps["s4_turbulent"]]))
        s1_ok = all(e["success"] for e in eps["s1_still"])
        delay_ok = all(e["success"] for e in eps["s4d_alone_delay"])
        lock = sum(e["lockins"] or 0 for es in eps.values() for e in es)
        passed = s4_hits >= RULE_S4_HITS and s1_ok and delay_ok and lock == 0
        supported = supported or passed
        rep.say(f"rule kd{kd:g}: S4 mean hits {s4_hits:.2f}, S1 5/5 {s1_ok}, delay-alone 5/5 {delay_ok}, "
                f"lock-ins {lock}, passes {passed}")
    rep.say(f"damping supported as the missing ingredient by the rule: {supported}")
    rep.say("reference rows (pid_40, fly_kd0) and earlier kd rows read from earlier runs, not rerun; the "
            "kd 0 tick-for-tick check is in results/yaw_damp_s0.55.txt (option unchanged since)")
    rep.say(f"flights run now {len(jobs)}, wall {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
