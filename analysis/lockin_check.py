import argparse
import json
from pathlib import Path

import numpy as np

from fly.check_load import REPO_ROOT, Report

RUNS_DIR = REPO_ROOT / "results" / "runs"
OUT_PATH = REPO_ROOT / "results" / "lockin_check.txt"
TICK_S = 0.005
BASE_TICKS = 400
MIN_TICKS = 200
FACTOR = 3.0


def lockin_events(spikes: np.ndarray) -> list[dict]:
    spikes = np.asarray(spikes, dtype=float)
    events = []
    k = 0
    while k < spikes.size:
        base = np.median(spikes[:BASE_TICKS] if k < BASE_TICKS else spikes[k - BASE_TICKS:k])
        end = k
        while end < spikes.size and spikes[end] > FACTOR * base:
            end += 1
        if end - k >= MIN_TICKS:
            events.append({"start_s": k * TICK_S, "duration_s": (end - k) * TICK_S,
                           "base": float(base), "to_end": end == spikes.size})
            k = end
        else:
            k += 1
    return events


def read_spikes(path: Path) -> np.ndarray | None:
    spikes = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            row = json.loads(raw)
            if row["type"] == "tick":
                if "spikes_total" not in row.get("info", {}):
                    return None
                spikes.append(row["info"]["spikes_total"])
    return np.array(spikes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=str(RUNS_DIR))
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    root = Path(args.runs)
    rep = Report()
    rep.say(f"lock-in: spikes per tick above {FACTOR:g} x median of the {BASE_TICKS * TICK_S:g} s before "
            f"the stretch, for at least {MIN_TICKS * TICK_S:g} s in a row; first {BASE_TICKS * TICK_S:g} s "
            f"use the median of the first {BASE_TICKS} ticks")
    rep.say("run  ticks  median_spikes  events")
    flights = events = 0
    for path in sorted(root.rglob("*__fly*.jsonl")):
        name = path.relative_to(root).as_posix()
        spikes = read_spikes(path)
        if spikes is None:
            rep.say(f"{name}  no spike data, skipped")
            continue
        found = lockin_events(spikes)
        flights += 1
        events += len(found)
        rep.say(f"{name}  {spikes.size}  {np.median(spikes):.0f}  {len(found)}")
        for e in found:
            rep.say(f"  start {e['start_s']:.3f} s  duration {e['duration_s']:.3f} s  base {e['base']:.0f}"
                    f"{'  still locked at the end' if e['to_end'] else ''}")
    rep.say(f"{events} events in {flights} flights")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
