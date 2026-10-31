import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from analysis.evaluate import RESULTS_DIR, TUNING_SCENARIOS, write_csv
from analysis.run import load_params, run_one
from controllers import controller_class
from env.metrics import tuning_cost
from env.params import CONFIG_DIR, load_physics
from env.scenario import load_scenario

TUNING_SEEDS = 50
SEEDS_PER_EVAL = 4


def eval_seeds(index: int, fixed: list[int] | None = None) -> list[int]:
    if fixed:
        return fixed
    return [(SEEDS_PER_EVAL * index + i) % TUNING_SEEDS for i in range(SEEDS_PER_EVAL)]


def episode_cost(job: tuple) -> float:
    controller, params, scenario_name, seed = job
    metrics = run_one(load_scenario(scenario_name), controller, seed, params,
                      physics=load_physics(), log=False)
    return tuning_cost(metrics)


def sample_params(space: dict, rng: np.random.Generator) -> dict:
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in space.items()}


def tune(controller: str, base: dict, evals: int, jobs: int, search_seed: int,
         seeds_fixed: list[int] | None = None) -> dict:
    cls = controller_class(controller)
    space = cls.param_space()
    rng = np.random.default_rng(search_seed)
    log = []
    best = None
    pool = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    try:
        for index in range(evals):
            trial = {} if index == 0 else sample_params(space, rng)
            params = {**base, **trial}
            seeds = eval_seeds(index, seeds_fixed)
            work = [(controller, params, s, seed) for s in TUNING_SCENARIOS for seed in seeds]
            costs = list(pool.map(episode_cost, work)) if pool else [episode_cost(w) for w in work]
            cost = statistics.fmean(costs)
            row = {"eval": index, "cost": cost, "seeds": " ".join(map(str, seeds)),
                   **{k: params[k] for k in space}}
            log.append(row)
            if best is None or cost < best["cost"]:
                best = {"eval": index, "cost": cost, "params": params}
            print(f"eval {index:3d}  cost {cost:.4f}  best {best['cost']:.4f} (eval {best['eval']})")
    finally:
        if pool:
            pool.shutdown()
    return {"controller": controller, "evals": evals, "search_seed": search_seed,
            "scenarios": list(TUNING_SCENARIOS), "seeds_per_eval": len(seeds_fixed or range(SEEDS_PER_EVAL)),
            "best": best, "log": log}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", default="pid")
    parser.add_argument("--base", default=str(CONFIG_DIR / "pid_default.json"))
    parser.add_argument("--evals", type=int, default=150)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--log-name")
    parser.add_argument("--out", default=str(CONFIG_DIR / "pid_tuned.json"))
    args = parser.parse_args()

    result = tune(args.controller, load_params(args.base), args.evals, args.jobs,
                  args.search_seed, args.seeds)
    name = args.log_name or f"tune_{args.controller}"
    write_csv(result["log"], RESULTS_DIR / f"{name}_log.csv")
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n",
                                              encoding="utf-8", newline="\n")
    Path(args.out).write_text(json.dumps(result["best"]["params"], indent=2) + "\n",
                              encoding="utf-8", newline="\n")
    print(f"best eval {result['best']['eval']} cost {result['best']['cost']:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
