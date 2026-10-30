import argparse
import json
from pathlib import Path

from controllers import make_controller
from env.engine import Sim, run_episode
from env.metrics import compute_metrics
from env.params import CONFIG_DIR, load_physics, physics_hash
from env.runlog import RUNS_DIR, SCHEMA, git_commit, run_path, write_run
from env.scenario import load_scenario, scenario_hash


def load_params(path: str | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def default_params(controller: str) -> dict | None:
    tuned = CONFIG_DIR / f"{controller}_tuned.json"
    return load_params(str(tuned)) if tuned.exists() else None


def make_header(scenario: dict, controller: str, seed: int, physics: dict, commit: str) -> dict:
    return {
        "type": "header", "schema": SCHEMA, "scenario": scenario["name"],
        "scenario_hash": scenario_hash(scenario), "controller": controller, "seed": seed,
        "mode": scenario["mode"], "physics_hash": physics_hash(physics),
        "dt_s": physics["dt_s"], "tick_s": physics["dt_s"] * physics["tick_steps"],
        "commit": commit,
    }


def run_one(
    scenario: dict,
    controller: str,
    seed: int,
    params: dict | None = None,
    physics: dict | None = None,
    out: Path | None = None,
    log: bool = True,
    commit: str | None = None,
    on_tick=None,
) -> dict:
    physics = physics or load_physics()
    sim = Sim(physics)
    ctrl = make_controller(controller, params)
    result = run_episode(sim, ctrl, seed, scenario, on_tick)
    metrics = compute_metrics(result.ticks, scenario, sim.tick_s, result.act_s_total)
    if log:
        header = make_header(scenario, controller, seed, physics, commit or git_commit())
        if hasattr(ctrl, "header_info"):
            header.update(ctrl.header_info())
        path = out or run_path(scenario["name"], controller, seed)
        write_run(path, header, result.ticks, {"type": "summary", "metrics": metrics})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--params", help="controller params JSON, default <controller>_tuned")
    parser.add_argument("--out", help="run file path, or a folder")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    out = None
    if args.out:
        out = Path(args.out)
        if out.suffix != ".jsonl":
            out = run_path(scenario["name"], args.controller, args.seed, out)
    params = load_params(args.params) if args.params else default_params(args.controller)
    metrics = run_one(scenario, args.controller, args.seed, params,
                      out=out, log=not args.no_log)
    if not args.no_log:
        print(out or run_path(scenario["name"], args.controller, args.seed, RUNS_DIR))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
