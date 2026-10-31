import argparse
import json
import time
from pathlib import Path

import numpy as np

from analysis.lockin_check import lockin_events, read_spikes
from analysis.run import default_params, run_one
from env.params import CONFIG_DIR
from env.runlog import run_path
from env.scenario import load_scenario
from fly.check_load import REPO_ROOT, Report

OUT_PATH = REPO_ROOT / "results" / "eval_s1_s4.txt"
SCENARIOS = ("s1_still", "s2_crosswind", "s3_gust", "s4_turbulent")
CONTROLLERS = ("fly_real", "pid")
SEEDS = tuple(range(5))


def read_metrics(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.loads(f.readlines()[-1])["metrics"]


def fly_runs(controllers: tuple[str, ...]) -> None:
    for controller in controllers:
        params = default_params(controller)
        for name in SCENARIOS:
            scenario = load_scenario(str(CONFIG_DIR / "scenarios" / f"{name}.json"))
            for seed in SEEDS:
                start = time.perf_counter()
                m = run_one(scenario, controller, seed, params)
                print(f"{name} {controller} s{seed}: hits {m['targets_hit']}, "
                      f"{time.perf_counter() - start:.0f} s", flush=True)


def summary(out: Path) -> None:
    rep = Report()
    rep.say("closed-loop S1 to S4, seeds 0-4, serial, no tuning; pid uses configs/pid_tuned.json")
    rep.say("median time per target = median over seeds of the episode time_per_target_s, seeds "
            "with no hit left out; ms per tick is the mean over seeds")
    rep.say("scenario controller success mean_hits median_time_per_target n_timed ms_per_tick "
            "lockin_events")
    rows = []
    for name in SCENARIOS:
        for controller in CONTROLLERS:
            paths = [run_path(name, controller, s) for s in SEEDS]
            metrics = [read_metrics(p) for p in paths]
            times = [m["time_per_target_s"] for m in metrics if m["time_per_target_s"] is not None]
            events = "-"
            found = []
            if controller == "fly_real":
                found = [(s, e) for s, p in zip(SEEDS, paths) for e in lockin_events(read_spikes(p))]
                events = str(len(found))
            median = f"{np.median(times):.3f}" if times else "None"
            rep.say(f"{name} {controller} {sum(m['episode_success'] for m in metrics)}/{len(SEEDS)} "
                    f"{np.mean([m['targets_hit'] for m in metrics]):.2f} {median} {len(times)} "
                    f"{np.mean([m['ctrl_ms_per_tick'] for m in metrics]):.3f} {events}")
            rows += [(name, s, m) for s, m in zip(SEEDS, metrics) if controller == "fly_real"]
            for s, e in found:
                rep.say(f"  seed {s}: start {e['start_s']:.3f} s, duration {e['duration_s']:.3f} s"
                        f"{', still locked at the end' if e['to_end'] else ''}")
    rep.say("fly_real per run: scenario seed hits misses episode_s out_of_bounds ms_per_tick spikes_per_tick")
    for name, s, m in rows:
        rep.say(f"  {name} {s} {m['targets_hit']} {m['misses']} {m['episode_s']:.3f} "
                f"{m['out_of_bounds']} {m['ctrl_ms_per_tick']:.2f} {m['spikes_per_tick']:.2f}")
    rep.save(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--controllers", nargs="+", default=list(CONTROLLERS))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    if not args.summary_only:
        fly_runs(tuple(args.controllers))
    summary(Path(args.out))


if __name__ == "__main__":
    main()
