import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from analysis.decode_smooth import (
    FLY_PARAMS,
    REFERENCE,
    RULE_S4_HITS,
    SCENARIOS,
    SEEDS,
    scenario_of,
    ticks,
)
from analysis.run import load_params, run_one
from analysis.s4_parts import PID_PARAMS, episode, finished, row
from env.runlog import RUNS_DIR, run_path
from env.scenario import load_scenario
from fly.check_load import REPO_ROOT, Report

OUT_PATH = REPO_ROOT / "results" / "yaw_damp_s0.55.txt"
RUN_DIR = RUNS_DIR / "yaw_damp"
PID_KD = float(json.loads(PID_PARAMS.read_text(encoding="utf-8"))["Kd"])
SETTINGS = (0.372, 0.744, 1.488)


def fly_params(kd: float) -> dict:
    return {**load_params(str(FLY_PARAMS)), "yaw_damp": kd}


def setting_dir(kd: float) -> Path:
    return RUN_DIR / f"kd{kd:g}"


def fly_one(job: tuple) -> tuple:
    name, kd, seed, out = job
    m = run_one(scenario_of(name), "fly_real", seed, fly_params(kd), out=Path(out))
    return name, kd, seed, m["targets_hit"]


def paths_for(name: str, label: str, kd: float | None) -> list[Path]:
    ctrl = "pid" if label == "pid" else "fly_real"
    folder = setting_dir(kd) if kd is not None else REFERENCE[(name, label)]
    return [run_path(name, ctrl, s, folder) for s in SEEDS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    start = time.perf_counter()
    rep = Report()
    check = run_path("s1_still", "fly_real", 0, RUN_DIR / "check_kd0")
    check.parent.mkdir(parents=True, exist_ok=True)
    if not finished(check):
        run_one(load_scenario("s1_still"), "fly_real", 0, fly_params(0.0), out=check)
    same = ticks(check) == ticks(run_path("s1_still", "fly_real", 0, RUNS_DIR / "s0.55_tuned"))
    rep.say(f"check: S1 seed 0, yaw_damp 0 vs results/runs/s0.55_tuned, tick lines identical: {same}")
    if not same:
        rep.say("check failed, no flights run")
        rep.save(Path(args.out))
        return
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
    rep.say(f"damping ablation, diagnostic only, seeds 0-4, fly_real {FLY_PARAMS.name} plus yaw_damp kd; "
            f"yaw command = gain_yaw x yaw decode - kd x yaw_rate obs, then clip to [-1, 1]; PID Kd {PID_KD:g}")
    rep.say("the fly gets the same yaw_rate observation and delay as the PID; both send yaw_torque in "
            "[-1, 1], which the sim multiplies by taumax")
    rep.say(f"rule fixed before the run: damping supported only if for some kd S4 mean hits >= "
            f"{RULE_S4_HITS:g}, S1 5/5, and 0 lock-ins over the 15 flights of that kd. Nothing adopted.")
    rep.say("time per target = mean over episodes with a hit; closest = smallest true distance to the "
            "active target, mean over episodes that timed out")
    rep.say("variant controller success mean_hits mean_time_per_target n_timed closest_mm n_timeout "
            "lockins hits_per_seed")
    for name in SCENARIOS:
        rep.say(row(name, "pid_40", paths_for(name, "pid", None)))
        rep.say(row(name, "fly_kd0", paths_for(name, "fly", None)))
        for kd in SETTINGS:
            rep.say(row(name, f"fly_kd{kd:g}", paths_for(name, "fly", kd)))
    supported = False
    for kd in SETTINGS:
        eps = {n: [episode(p) for p in paths_for(n, "fly", kd)] for n in SCENARIOS}
        s4_hits = float(np.mean([e["hits"] for e in eps["s4_turbulent"]]))
        s1_ok = all(e["success"] for e in eps["s1_still"])
        lock = sum(e["lockins"] or 0 for es in eps.values() for e in es)
        passed = s4_hits >= RULE_S4_HITS and s1_ok and lock == 0
        supported = supported or passed
        rep.say(f"rule kd{kd:g}: S4 mean hits {s4_hits:.2f}, S1 5/5 {s1_ok}, lock-ins {lock}, passes {passed}")
    rep.say(f"damping supported as the missing ingredient by the rule: {supported}")
    rep.say("reference rows (pid_40, fly_kd0) read from earlier runs, not rerun")
    rep.say(f"flights run now {len(jobs)}, wall {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
