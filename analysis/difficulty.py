import argparse
import copy
import csv
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from analysis.evaluate import RESULTS_DIR, parse_seeds
from analysis.run import default_params, load_params, run_one
from env.params import load_physics
from env.scenario import load_scenario, validate_scenario

SWEEP_CSV = RESULTS_DIR / "difficulty_sweep.csv"
FIELDS = (
    "scenario", "lever", "value", "fixed", "controller", "seeds", "success_rate",
    "mean_targets_hit", "timeouts", "out_of_bounds", "median_time_per_target_s",
    "median_rms_cross_track_mm",
)


def set_lever(scenario: dict, lever: str, value: float) -> dict:
    out = copy.deepcopy(scenario)
    parts = lever.split(".")
    if parts[0] == "noise_multiplier":
        out["noise_multiplier"] = value
    elif parts[0] == "wind" and len(parts) == 3:
        out["wind"][int(parts[1])][parts[2]] = value
    else:
        raise ValueError(f"unknown lever {lever!r}, use noise_multiplier or wind.<i>.<key>")
    validate_scenario(out)
    return out


def apply_fixed(scenario: dict, fixed: list[str]) -> dict:
    for item in fixed:
        lever, value = item.split("=")
        scenario = set_lever(scenario, lever, float(value))
    return scenario


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


def sweep(
    base: dict, lever: str, values: list[float], controller: str, params: dict | None,
    seeds: list[int], fixed: list[str], pool: ProcessPoolExecutor | None,
) -> list[dict]:
    out = []
    base = apply_fixed(base, fixed)
    for value in values:
        scenario = set_lever(base, lever, value)
        work = [(scenario, controller, params, s) for s in seeds]
        rows = list(pool.map(one_episode, work)) if pool else [one_episode(w) for w in work]
        row = {"scenario": base["name"], "lever": lever, "value": value,
               "fixed": " ".join(fixed), "controller": controller,
               "seeds": f"{min(seeds)}-{max(seeds)}", **summarize(rows)}
        print(f"{base['name']} {lever}={value:g} {row['fixed']}: success {row['success_rate']:.2f}"
              f"  timeouts {row['timeouts']}  oob {row['out_of_bounds']}")
        out.append(row)
    return out


def append_csv(rows: list[dict], path: Path) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--lever", required=True, help="noise_multiplier or wind.<i>.<key>")
    parser.add_argument("--values", required=True, help="comma separated")
    parser.add_argument("--fixed", nargs="*", default=[], help="lever=value held for the sweep")
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
    values = [float(v) for v in args.values.split(",")]
    pool = ProcessPoolExecutor(max_workers=args.jobs) if args.jobs > 1 else None
    try:
        rows = sweep(load_scenario(args.scenario), args.lever, values, args.controller, params,
                     seeds, args.fixed, pool)
    finally:
        if pool:
            pool.shutdown()
    append_csv(rows, Path(args.out))


if __name__ == "__main__":
    main()
