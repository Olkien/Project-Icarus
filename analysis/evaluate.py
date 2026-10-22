import argparse
import csv
import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from analysis.run import load_params, run_one
from env.params import load_physics
from env.runlog import REPO_ROOT, git_commit
from env.scenario import load_scenario

RESULTS_DIR = REPO_ROOT / "results"
TUNING_SCENARIOS = ("s1_still", "s2_crosswind", "s3_gust", "s4_turbulent")


def parse_seeds(text: str) -> list[int]:
    seeds = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds += list(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def one_job(job: tuple) -> dict:
    scenario_name, controller, seed, params, log, commit = job
    scenario = load_scenario(scenario_name)
    metrics = run_one(scenario, controller, seed, params, physics=load_physics(), log=log,
                      commit=commit)
    return {"scenario": scenario["name"], "controller": controller, "seed": seed, **metrics}


def evaluate(
    controller: str,
    params: dict | None,
    scenarios: list[str],
    seeds: list[int],
    jobs: int = 1,
    log: bool = False,
) -> list[dict]:
    commit = git_commit()
    work = [(s, controller, seed, params, log, commit) for s in scenarios for seed in seeds]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            return list(pool.map(one_job, work, chunksize=max(1, len(work) // (jobs * 4))))
    return [one_job(w) for w in work]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for name in dict.fromkeys(r["scenario"] for r in rows):
        group = [r for r in rows if r["scenario"] == name]
        times = [r["time_per_target_s"] for r in group if r["time_per_target_s"] is not None]
        successes = sum(r["episode_success"] for r in group)
        lo, hi = wilson_interval(successes, len(group))
        oob = sum(r["out_of_bounds"] for r in group)
        out.append({
            "scenario": name,
            "runs": len(group),
            "success_rate": successes / len(group),
            "success_ci95_lo": lo,
            "success_ci95_hi": hi,
            "timeouts": len(group) - successes - oob,
            "out_of_bounds": oob,
            "median_time_per_target_s": statistics.median(times) if times else None,
            "median_time_miss6_s": statistics.median(r["time_per_target_miss6_s"] for r in group),
            "median_rms_cross_track_mm": statistics.median(r["rms_cross_track_mm"] for r in group),
            "misses": sum(r["misses"] for r in group),
        })
    return out


def fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def print_table(summary: list[dict]) -> None:
    cols = list(summary[0])
    print("  ".join(f"{c:>14}" for c in cols))
    for row in summary:
        print("  ".join(f"{fmt(row[c]):>14}" for c in cols))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", required=True)
    parser.add_argument("--params")
    parser.add_argument("--scenarios", nargs="+", default=list(TUNING_SCENARIOS))
    parser.add_argument("--seeds", default="1000-1019")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--log", action="store_true", help="also write run files")
    parser.add_argument("--out", help="CSV path")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    rows = evaluate(args.controller, load_params(args.params), args.scenarios,
                    parse_seeds(args.seeds), args.jobs, args.log)
    tag = f"_{args.tag}" if args.tag else ""
    out = Path(args.out) if args.out else RESULTS_DIR / f"eval_{args.controller}{tag}.csv"
    write_csv(rows, out)
    print(out)
    print_table(summarize(rows))


if __name__ == "__main__":
    main()
