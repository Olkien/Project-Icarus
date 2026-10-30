import argparse
import json
import math
from pathlib import Path

import numpy as np

from fly.check_load import REPO_ROOT, Report

RUNS_DIR = REPO_ROOT / "results" / "runs"
OUT_PATH = REPO_ROOT / "results" / "band_flights.txt"
CONTROLLERS = ("fly_real", "fly_steer_only")
SEEDS = (0, 1, 2)
CIRCLE_TURNS = 2.0


def summarise(path: Path) -> dict:
    ticks = []
    metrics = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(raw)
        if row["type"] == "tick":
            ticks.append(row)
        elif row["type"] == "summary":
            metrics = row["metrics"]
    first = [t for t in ticks if t["target"]["i"] == ticks[0]["target"]["i"]]
    closest_mm = min(math.hypot(t["state"]["x"] - t["target"]["x"], t["state"]["y"] - t["target"]["y"])
                     for t in first)
    bearing_rad = np.abs([t["obs"]["bearing"] for t in ticks]).mean()
    turns = {}
    for i in sorted({t["target"]["i"] for t in ticks}):
        psi = [t["state"]["psi"] for t in ticks if t["target"]["i"] == i]
        turns[i] = float(np.abs(np.unwrap(psi)[-1] - psi[0])) / (2.0 * math.pi)
    return {"metrics": metrics, "closest_mm": closest_mm, "bearing_rad": float(bearing_rad),
            "turns": turns, "circles": max(turns.values()) >= CIRCLE_TURNS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    rep = Report()
    rep.say(f"S1 closed loop; circles = net heading change on one target of at least {CIRCLE_TURNS:g} turns")
    rep.say("controller seed ms/tick spikes/tick hits time_per_target closest_mm mean|b|_rad oob circles "
            "turns_per_target")
    for name in CONTROLLERS:
        for seed in SEEDS:
            r = summarise(RUNS_DIR / f"s1_still__{name}__s{seed}.jsonl")
            m = r["metrics"]
            turns = " ".join(f"{i}:{v:.1f}" for i, v in r["turns"].items())
            rep.say(f"{name} {seed} {m['ctrl_ms_per_tick']:.2f} {m['spikes_per_tick']:.2f} "
                    f"{m['targets_hit']} {m['time_per_target_s']} {r['closest_mm']:.2f} "
                    f"{r['bearing_rad']:.3f} {m['out_of_bounds']} {r['circles']} {turns}")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
