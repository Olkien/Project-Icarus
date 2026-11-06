import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from analysis.lockin_check import lockin_events, read_spikes
from analysis.run import load_params, run_one
from env.params import CONFIG_DIR
from env.runlog import RUNS_DIR, run_path
from env.scenario import validate_scenario
from fly.check_load import REPO_ROOT, Report

OUT_PATH = REPO_ROOT / "results" / "s4_parts.txt"
VARIANTS_PATH = CONFIG_DIR / "diagnostic" / "s4_parts.json"
RUN_DIR = RUNS_DIR / "s4_parts"
SEEDS = tuple(range(5))
FLY_PARAMS = CONFIG_DIR / "fly_real_tuned_s0.55.json"
PID_PARAMS = CONFIG_DIR / "pid_tuned_40.json"
LABELS = {"fly_real": ("fly_real", FLY_PARAMS), "pid_40": ("pid", PID_PARAMS)}
REFERENCE = {("s1_still", "fly_real"): RUNS_DIR / "s0.55_tuned", ("s1_still", "pid_40"): RUNS_DIR / "pid_40",
             ("s4_turbulent", "fly_real"): RUNS_DIR / "s0.55_tuned", ("s4_turbulent", "pid_40"): RUNS_DIR / "pid_40"}


def file_label(label: str) -> str:
    return LABELS[label][0]


def load_variants(path: Path) -> list[dict]:
    variants = json.loads(path.read_text(encoding="utf-8"))
    for v in variants:
        validate_scenario(v)
    return variants


def finished(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as f:
        return '"summary"' in f.readlines()[-1]


def fly_one(job: tuple) -> tuple:
    scenario, label, seed, out = job
    controller, params_path = LABELS[label]
    start = time.perf_counter()
    m = run_one(scenario, controller, seed, load_params(str(params_path)), out=Path(out))
    return scenario["name"], label, seed, m["targets_hit"], time.perf_counter() - start


def true_range(t: dict) -> float:
    return math.hypot(t["target"]["x"] - t["state"]["x"], t["target"]["y"] - t["state"]["y"])


def episode(path: Path) -> dict:
    rows = [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines()]
    ticks = [r for r in rows if r["type"] == "tick"]
    m = rows[-1]["metrics"]
    timeout = not m["episode_success"] and not m["out_of_bounds"]
    last = ticks[-1]["target"]["i"]
    closest = min(true_range(t) for t in ticks if t["target"]["i"] == last) if timeout else None
    lockins = len(lockin_events(read_spikes(path))) if "spikes_total" in ticks[0]["info"] else None
    return {"success": m["episode_success"], "hits": m["targets_hit"], "tpt": m["time_per_target_s"],
            "timeout": timeout, "closest": closest, "lockins": lockins}


def row(name: str, label: str, paths: list[Path]) -> str:
    eps = [episode(p) for p in paths]
    tpt = [e["tpt"] for e in eps if e["tpt"] is not None]
    closest = [e["closest"] for e in eps if e["closest"] is not None]
    lock = "-" if eps[0]["lockins"] is None else str(sum(e["lockins"] for e in eps))
    return (f"{name} {label} {sum(e['success'] for e in eps)}/{len(eps)} "
            f"{np.mean([e['hits'] for e in eps]):.2f} "
            + (f"{np.mean(tpt):.3f} {len(tpt)}" if tpt else "None 0") + " "
            + (f"{np.mean(closest):.2f} {len(closest)}" if closest else "- 0") + f" {lock} "
            + ",".join(str(e["hits"]) for e in eps))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--variants", default=str(VARIANTS_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    variants = load_variants(Path(args.variants))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(v, label, s, str(run_path(v["name"], file_label(label), s, RUN_DIR)))
            for v in variants for label in ("fly_real", "pid_40") for s in SEEDS]
    jobs = [j for j in jobs if not finished(Path(j[3]))]
    print(f"{len(jobs)} flights to run, {args.workers} workers", flush=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(fly_one, j) for j in jobs]):
            name, label, seed, hits, wall = fut.result()
            print(f"{name} {label} s{seed}: hits {hits}, {wall:.0f} s", flush=True)
    rep = Report()
    rep.say(f"S4 decomposition, seeds 0-4; fly_real uses {FLY_PARAMS.name}, pid uses {PID_PARAMS.name}; "
            "lock-ins by the absolute rule; variants in configs/diagnostic/s4_parts.json")
    rep.say("time per target = mean over episodes with a hit; closest = smallest true distance to the "
            "active target, mean over episodes that timed out")
    rep.say("variant controller success mean_hits mean_time_per_target n_timed closest_mm n_timeout "
            "lockins hits_per_seed")
    for name in ("s1_still", *[v["name"] for v in variants], "s4_turbulent"):
        for label in LABELS:
            if (name, label) in REFERENCE:
                paths = [run_path(name, file_label(label), s, REFERENCE[(name, label)]) for s in SEEDS]
            else:
                paths = [run_path(name, file_label(label), s, RUN_DIR) for s in SEEDS]
            rep.say(row(name, label, paths))
    rep.say(f"reference rows s1_still and s4_turbulent read from {REFERENCE[('s1_still', 'fly_real')].name} "
            f"and {REFERENCE[('s1_still', 'pid_40')].name}, not rerun")
    rep.say(f"flights run now {len(jobs)}, wall {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
