import argparse
import copy
import csv
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from analysis.evaluate import RESULTS_DIR, parse_seeds
from analysis.run import default_params, load_params, run_one
from env.params import load_physics
from env.scenario import load_scenario, validate_scenario

SWEEP_CSV = RESULTS_DIR / "difficulty_v2_sweep.csv"
INT_KEYS = ("length_ticks",)
FIELDS = (
    "mode", "scenario", "settings", "controller", "seeds", "success_rate", "mean_targets_hit",
    "timeouts", "out_of_bounds", "median_time_per_target_s", "median_rms_cross_track_mm",
    "ideal_success_rate", "ideal_timeouts", "ideal_out_of_bounds",
)


def set_lever(scenario: dict, lever: str, value: float) -> dict:
    out = copy.deepcopy(scenario)
    parts = lever.split(".")
    node = out
    for part in parts[:-1]:
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node.setdefault(part, {})
    key = parts[-1]
    node[key] = round(value) if key in INT_KEYS else value
    validate_scenario(out)
    return out


def apply_settings(scenario: dict, settings: dict[str, float]) -> dict:
    for lever, value in settings.items():
        scenario = set_lever(scenario, lever, value)
    return scenario


def ideal_sensors(scenario: dict) -> dict:
    out = copy.deepcopy(scenario)
    out["sensor_delay_scale"] = 0.0
    out["noise_multiplier"] = 0.0
    out.pop("dropout", None)
    validate_scenario(out)
    return out


def one_episode(job: tuple) -> dict:
    scenario, controller, params, seed = job
    return run_one(scenario, controller, seed, params, physics=load_physics(), log=False)


def summarize(rows: list[dict]) -> dict:
    times = [r["time_per_target_s"] for r in rows if r["time_per_target_s"] is not None]
    oob = sum(r["out_of_bounds"] for r in rows)
    success = sum(r["episode_success"] for r in rows)
    return {
        "success_rate": success / len(rows),
        "mean_targets_hit": statistics.fmean(r["targets_hit"] for r in rows),
        "timeouts": len(rows) - success - oob,
        "out_of_bounds": oob,
        "median_time_per_target_s": statistics.median(times) if times else None,
        "median_rms_cross_track_mm": statistics.median(r["rms_cross_track_mm"] for r in rows),
    }


def evaluate_settings(
    base: dict, settings_list: list[dict], controller: str, params: dict | None,
    seeds: list[int], ideal: bool, pool: ProcessPoolExecutor | None, mode: str,
) -> list[dict]:
    scenarios = [apply_settings(base, s) for s in settings_list]
    variants = [scenarios] + ([[ideal_sensors(s) for s in scenarios]] if ideal else [])
    summaries = []
    for group in variants:
        work = [(s, controller, params, seed) for s in group for seed in seeds]
        rows = list(pool.map(one_episode, work, chunksize=8)) if pool else [
            one_episode(w) for w in work]
        n = len(seeds)
        summaries.append([summarize(rows[i * n:(i + 1) * n]) for i in range(len(group))])
    out = []
    for i, settings in enumerate(settings_list):
        row = {"mode": mode, "scenario": base["name"], "settings": json.dumps(settings),
               "controller": controller, "seeds": f"{min(seeds)}-{max(seeds)}:{len(seeds)}",
               **summaries[0][i]}
        if ideal:
            row["ideal_success_rate"] = summaries[1][i]["success_rate"]
            row["ideal_timeouts"] = summaries[1][i]["timeouts"]
            row["ideal_out_of_bounds"] = summaries[1][i]["out_of_bounds"]
        out.append(row)
    return out


def parse_ranges(items: list[str]) -> dict[str, tuple[float, float]]:
    ranges = {}
    for item in items:
        lever, span = item.split("=")
        lo, hi = span.split(":")
        ranges[lever] = (float(lo), float(hi))
    return ranges


def random_settings(
    ranges: dict[str, tuple[float, float]], count: int, seed: int
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    return [{lever: as_setting(lever, float(rng.uniform(lo, hi)))
             for lever, (lo, hi) in ranges.items()} for _ in range(count)]


def as_setting(lever: str, value: float) -> float | int:
    return round(value) if lever.split(".")[-1] in INT_KEYS else value


def append_csv(rows: list[dict], path: Path) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n", extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict]) -> None:
    for r in rows:
        ideal = f"  ideal {r['ideal_success_rate']:.2f}" if "ideal_success_rate" in r else ""
        print(f"{r['scenario']} {r['settings']}: success {r['success_rate']:.2f}  "
              f"timeouts {r['timeouts']}  oob {r['out_of_bounds']}{ideal}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--lever", help="dotted path, e.g. wind.0.speed_mm_s or dropout.p_start")
    parser.add_argument("--values", help="comma separated values for --lever")
    parser.add_argument("--random", type=int, default=0, help="number of random combinations")
    parser.add_argument("--ranges", nargs="*", default=[], help="lever=lo:hi for --random")
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--fixed", default="{}", help="JSON of lever: value held for the run")
    parser.add_argument("--ideal", action="store_true", help="add the ideal-sensor check")
    parser.add_argument("--controller", default="pid")
    parser.add_argument("--params")
    parser.add_argument("--seeds", default="0-49")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", default=str(SWEEP_CSV))
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    if max(seeds) >= 1000:
        parser.error("calibration uses tuning seeds only")
    params = load_params(args.params) if args.params else default_params(args.controller)
    fixed = json.loads(args.fixed)
    if args.lever:
        settings = [{**fixed, args.lever: as_setting(args.lever, float(v))}
                    for v in args.values.split(",")]
        mode = "sweep"
    elif args.random:
        drawn = random_settings(parse_ranges(args.ranges), args.random, args.search_seed)
        settings = [{**fixed, **s} for s in drawn]
        mode = "random"
    else:
        settings = [fixed]
        mode = "check"
    pool = ProcessPoolExecutor(max_workers=args.jobs) if args.jobs > 1 else None
    try:
        rows = evaluate_settings(load_scenario(args.scenario), settings, args.controller, params,
                                 seeds, args.ideal, pool, mode)
    finally:
        if pool:
            pool.shutdown()
    print_rows(rows)
    append_csv(rows, Path(args.out))


if __name__ == "__main__":
    main()
