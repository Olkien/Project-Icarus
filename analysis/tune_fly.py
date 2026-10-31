import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from analysis.evaluate import RESULTS_DIR
from analysis.lockin_check import lockin_events
from analysis.run import run_one
from controllers import controller_class
from env.metrics import tuning_cost
from env.params import CONFIG_DIR, load_physics
from env.scenario import load_scenario

SPACE_PATH = CONFIG_DIR / "fly_tuning_space.json"


def load_space(path: Path | str) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    bounds = controller_class(cfg["controller"]).param_space()
    unknown = [p for p in cfg["params"] if p not in bounds]
    if unknown:
        raise ValueError(f"not in param_space(): {unknown}")
    cfg["bounds"] = {p: list(bounds[p]) for p in cfg["params"]}
    return cfg


def plan(cfg: dict) -> list[dict]:
    rng = np.random.default_rng(cfg["search_seed"])
    trials = [{}]
    for _ in range(cfg["evals"] - 1):
        trials.append({p: float(rng.uniform(lo, hi)) for p, (lo, hi) in cfg["bounds"].items()})
    return [{**cfg.get("fixed", {}), **t} for t in trials]


def spike_logger(spikes: list[int]):
    def on_tick(record: dict) -> None:
        if "spikes_total" in record["info"]:
            spikes.append(record["info"]["spikes_total"])
    return on_tick


def evaluate(job: tuple) -> dict:
    index, cfg, params, overrides = job
    start = time.perf_counter()
    costs = []
    lockins = 0
    for name in cfg["scenarios"]:
        scenario = {**load_scenario(name), **overrides}
        for seed in cfg["seeds"]:
            spikes: list[int] = []
            metrics = run_one(scenario, cfg["controller"], seed, params, physics=load_physics(),
                              log=False, on_tick=spike_logger(spikes))
            costs.append(tuning_cost(metrics))
            lockins += len(lockin_events(np.array(spikes)))
    return {"eval": index, "cost": statistics.fmean(costs), "episodes": len(costs),
            "lockins": lockins, "wall_s": round(time.perf_counter() - start, 2), "params": json.dumps(params)}


def meta_of(cfg: dict) -> dict:
    return {k: cfg[k] for k in ("controller", "scenarios", "seeds", "search_seed", "bounds")} | {
        "fixed": cfg.get("fixed", {})}


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["eval"] = int(r["eval"])
        r["cost"] = float(r["cost"])
    return rows


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=["eval", "cost", "episodes", "lockins", "wall_s", "params"],
                                lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)


def tune(cfg: dict, out_dir: Path, jobs: int = 1, overrides: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"tune_{cfg['controller']}_log.csv"
    meta_path = out_dir / f"tune_{cfg['controller']}_meta.json"
    meta = meta_of(cfg)
    stale = meta_path.exists() and log_path.exists()
    if stale and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
        raise ValueError(f"{meta_path} was made with other settings; move it to start over")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    trials = plan(cfg)
    done = {r["eval"] for r in read_log(log_path)}
    todo = [(i, cfg, trials[i], overrides or {}) for i in range(len(trials)) if i not in done]
    start = time.perf_counter()
    if jobs > 1 and todo:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(evaluate, job) for job in todo]
            for fut in as_completed(futures):
                report(log_path, fut.result())
    else:
        for job in todo:
            report(log_path, evaluate(job))
    wall = time.perf_counter() - start
    rows = read_log(log_path)
    best = min(rows, key=lambda r: r["cost"])
    episodes = len(todo) * len(cfg["scenarios"]) * len(cfg["seeds"])
    summary = {"evals_done": len(rows), "evals_run_now": len(todo), "best_eval": best["eval"],
               "best_cost": best["cost"], "best_params": json.loads(best["params"]),
               "episodes_per_min": round(60.0 * episodes / wall, 1) if episodes else None}
    (out_dir / f"tune_{cfg['controller']}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def report(log_path: Path, row: dict) -> None:
    append_row(log_path, row)
    print(f"eval {row['eval']:3d}  cost {row['cost']:.4f}  lockins {row['lockins']}  "
          f"{row['wall_s']:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", default=str(SPACE_PATH))
    parser.add_argument("--evals", type=int, help="override the budget in the space file")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--stepper", choices=("numba", "numpy"))
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()
    cfg = load_space(args.space)
    if args.evals:
        cfg["evals"] = args.evals
    if args.scenarios:
        cfg["scenarios"] = args.scenarios
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.stepper:
        cfg["fixed"] = {**cfg.get("fixed", {}), "stepper": args.stepper}
    summary = tune(cfg, Path(args.out_dir), args.jobs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
