import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from analysis.lockin_check import TICK_S
from analysis.run import load_params, run_one
from analysis.s4_parts import VARIANTS_PATH, episode, finished, load_variants, row
from controllers.fly import FlyReal
from env.params import CONFIG_DIR
from env.runlog import RUNS_DIR, run_path
from env.scenario import load_scenario
from fly.check_load import REPO_ROOT, Report

OUT_PATH = REPO_ROOT / "results" / "decode_smooth_s0.55.txt"
CHECK_PATH = REPO_ROOT / "results" / "decode_smooth_check_s0.55.txt"
RUN_DIR = RUNS_DIR / "decode_smooth"
FLY_PARAMS = CONFIG_DIR / "fly_real_tuned_s0.55.json"
SETTINGS_MS = (20.0, 40.0, 60.0)
SCENARIOS = ("s1_still", "s4d_alone_delay", "s4_turbulent")
SEEDS = tuple(range(5))
OPEN_SEEDS = tuple(range(2000, 2020))
OPEN_RANGE_MM = 200.0
WARM_TICKS = 100
MEASURE_TICKS = 200
RULE_S4_HITS = 2.5
REFERENCE = {("s1_still", "fly"): RUNS_DIR / "s0.55_tuned", ("s4_turbulent", "fly"): RUNS_DIR / "s0.55_tuned",
             ("s4d_alone_delay", "fly"): RUNS_DIR / "s4_parts", ("s1_still", "pid"): RUNS_DIR / "pid_40",
             ("s4_turbulent", "pid"): RUNS_DIR / "pid_40", ("s4d_alone_delay", "pid"): RUNS_DIR / "s4_parts"}

_ctrl: dict = {}


def fly_params(ms: float) -> dict:
    return {**load_params(str(FLY_PARAMS)), "decode_smooth_ms": ms}


def setting_dir(ms: float) -> Path:
    return RUN_DIR / f"ms{ms:g}"


def scenario_of(name: str) -> dict:
    if name.startswith("s4d_"):
        return next(v for v in load_variants(VARIANTS_PATH) if v["name"] == name)
    return load_scenario(name)


def ticks(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if '"type":"tick"' in line]


def open_loop(job: tuple) -> tuple:
    ms, seed = job
    if ms not in _ctrl:
        _ctrl[ms] = FlyReal(fly_params(ms))
    ctrl = _ctrl[ms]
    ctrl.reset(seed, {})
    obs = {"bearing": 0.0, "range": OPEN_RANGE_MM, "vel_body": [0.0, 0.0], "yaw_rate": 0.0,
           "air_rel": [0.0, 0.0]}
    yaw, decode = [], []
    for k in range(WARM_TICKS + MEASURE_TICKS):
        ctrl.act(obs, k * TICK_S)
        yaw.append(ctrl.yaw_signal)
        decode.append(ctrl.thrust_decode)
    return job, float(np.mean(yaw[WARM_TICKS:])), float(np.mean(decode[WARM_TICKS:]))


def check(workers: int, out: Path) -> bool:
    rep = Report()
    start = time.perf_counter()
    path = run_path("s1_still", "fly_real", 0, RUN_DIR / "check_ms0")
    path.parent.mkdir(parents=True, exist_ok=True)
    run_one(load_scenario("s1_still"), "fly_real", 0, fly_params(0.0), out=path)
    old = ticks(run_path("s1_still", "fly_real", 0, RUNS_DIR / "s0.55_tuned"))
    new = ticks(path)
    same = old == new
    rep.say(f"check 1: S1 seed 0, decode_smooth_ms 0 vs results/runs/s0.55_tuned: {len(new)} ticks new, "
            f"{len(old)} old, tick lines identical: {same}")
    jobs = [(ms, s) for ms in (0.0, *SETTINGS_MS) for s in OPEN_SEEDS]
    with ProcessPoolExecutor(workers) as pool:
        res = {job: (y, d) for job, y, d in pool.map(open_loop, jobs)}
    rep.say(f"check 2: open loop, bearing 0, range {OPEN_RANGE_MM:g} mm, seeds {OPEN_SEEDS[0]}-"
            f"{OPEN_SEEDS[-1]}, mean over {MEASURE_TICKS * TICK_S:g} s after {WARM_TICKS * TICK_S:g} s warm up")
    rep.say("  within noise = |mean diff| below 2 x standard error of the per-seed diff, or below "
            "2 x standard error of the seed means at 0")
    ok = True
    for key, idx in (("yaw", 0), ("thrust decode", 1)):
        base = np.array([res[(0.0, s)][idx] for s in OPEN_SEEDS])
        se0 = base.std(ddof=1) / np.sqrt(len(base))
        rep.say(f"  {key} at 0 ms: mean {base.mean():.4f}, seed se {se0:.4f}")
        for ms in SETTINGS_MS:
            vals = np.array([res[(ms, s)][idx] for s in OPEN_SEEDS])
            diff = vals - base
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            inside = abs(diff.mean()) <= 2 * max(se, 1e-12) or abs(diff.mean()) <= 2 * se0
            ok = ok and inside
            rep.say(f"  {key} at {ms:g} ms: mean {vals.mean():.4f}, diff {diff.mean():+.5f}, diff se "
                    f"{se:.5f}, within noise: {inside}")
    rep.say(f"checks pass: {same and ok}; total {time.perf_counter() - start:.0f} s")
    rep.save(out)
    return same and ok


def fly_one(job: tuple) -> tuple:
    name, ms, seed, out = job
    m = run_one(scenario_of(name), "fly_real", seed, fly_params(ms), out=Path(out))
    return name, ms, seed, m["targets_hit"]


def paths_for(name: str, label: str, ms: float | None) -> list[Path]:
    ctrl = "pid" if label == "pid" else "fly_real"
    folder = setting_dir(ms) if ms is not None else REFERENCE[(name, label)]
    return [run_path(name, ctrl, s, folder) for s in SEEDS]


def flights(workers: int, out: Path) -> None:
    start = time.perf_counter()
    jobs = [(n, ms, s, str(run_path(n, "fly_real", s, setting_dir(ms))))
            for ms in SETTINGS_MS for n in SCENARIOS for s in SEEDS]
    jobs = [j for j in jobs if not finished(Path(j[3]))]
    print(f"{len(jobs)} flights to run, {workers} workers", flush=True)
    for ms in SETTINGS_MS:
        setting_dir(ms).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(fly_one, j) for j in jobs]):
            name, ms, seed, hits = fut.result()
            print(f"{name} ms{ms:g} s{seed}: hits {hits}", flush=True)
    rep = Report()
    rep.say(f"decode smoothing test, seeds 0-4, fly_real {FLY_PARAMS.name} plus decode_smooth_ms; "
            "EMA on the yaw decode (after yaw_zero) and thrust decode, before the gains, state starts at 0")
    rep.say(f"rule fixed before the run: improvement only if S4 mean hits >= {RULE_S4_HITS:g}, S1 5/5, "
            "and 0 lock-ins over all 15 flights of the setting. No setting is adopted.")
    rep.say("time per target = mean over episodes with a hit; closest = smallest true distance to the "
            "active target, mean over episodes that timed out")
    rep.say("variant controller success mean_hits mean_time_per_target n_timed closest_mm n_timeout "
            "lockins hits_per_seed")
    for name in SCENARIOS:
        rep.say(row(name, "pid_40", paths_for(name, "pid", None)))
        rep.say(row(name, "fly_ms0", paths_for(name, "fly", None)))
        for ms in SETTINGS_MS:
            rep.say(row(name, f"fly_ms{ms:g}", paths_for(name, "fly", ms)))
    for ms in SETTINGS_MS:
        eps = {n: [episode(p) for p in paths_for(n, "fly", ms)] for n in SCENARIOS}
        s4_hits = float(np.mean([e["hits"] for e in eps["s4_turbulent"]]))
        s1_ok = all(e["success"] for e in eps["s1_still"])
        lock = sum(e["lockins"] or 0 for es in eps.values() for e in es)
        better = s4_hits >= RULE_S4_HITS and s1_ok and lock == 0
        rep.say(f"rule ms{ms:g}: S4 mean hits {s4_hits:.2f}, S1 5/5 {s1_ok}, lock-ins {lock}, "
                f"improvement {better}")
    rep.say(f"reference rows (pid_40, fly_ms0) read from earlier runs, not rerun: "
            f"{json.dumps({f'{k[0]} {k[1]}': v.name for k, v in REFERENCE.items()})}")
    rep.say(f"flights run now {len(jobs)}, wall {time.perf_counter() - start:.0f} s")
    rep.save(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("check", "flights"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.stage == "check":
        check(args.workers, Path(args.out or CHECK_PATH))
    else:
        flights(args.workers, Path(args.out or OUT_PATH))


if __name__ == "__main__":
    main()
