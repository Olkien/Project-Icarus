import argparse
import json
from pathlib import Path

import numpy as np

from fly.check_load import REPO_ROOT, Report

RUNS_DIR = REPO_ROOT / "results" / "runs"
OUT_PATH = REPO_ROOT / "results" / "lockin_check.txt"
TICK_S = 0.005
ABS_SPIKES = 1500.0
ABS_MIN_TICKS = 100
BASE_TICKS = 400
MIN_TICKS = 200
FACTOR = 3.0
RULES = ("absolute", "relative")


def lockin_events(spikes: np.ndarray, rule: str = "absolute") -> list[dict]:
    spikes = np.asarray(spikes, dtype=float)
    if rule == "relative":
        return relative_events(spikes)
    events = []
    k = 0
    while k < spikes.size:
        end = k
        while end < spikes.size and spikes[end] > ABS_SPIKES:
            end += 1
        if end - k >= ABS_MIN_TICKS:
            events.append({"start_s": k * TICK_S, "duration_s": (end - k) * TICK_S,
                           "base": ABS_SPIKES, "to_end": end == spikes.size})
        k = max(end, k + 1)
    return events


def relative_events(spikes: np.ndarray) -> list[dict]:
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
    parser.add_argument("--rule", choices=(*RULES, "both"), default="absolute")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    root = Path(args.runs)
    rep = Report()
    rules = RULES if args.rule == "both" else (args.rule,)
    rep.say(f"absolute rule: spikes per tick above {ABS_SPIKES:g} for at least "
            f"{ABS_MIN_TICKS * TICK_S:g} s in a row")
    rep.say(f"relative rule: spikes per tick above {FACTOR:g} x median of the {BASE_TICKS * TICK_S:g} s "
            f"before the stretch, for at least {MIN_TICKS * TICK_S:g} s in a row; first "
            f"{BASE_TICKS * TICK_S:g} s use the median of the first {BASE_TICKS} ticks")
    rep.say("run  ticks  median_spikes  " + "  ".join(rules))
    flights = 0
    events = dict.fromkeys(rules, 0)
    for path in sorted(root.rglob("*__fly*.jsonl")):
        name = path.relative_to(root).as_posix()
        spikes = read_spikes(path)
        if spikes is None:
            rep.say(f"{name}  no spike data, skipped")
            continue
        found = {rule: lockin_events(spikes, rule) for rule in rules}
        flights += 1
        rep.say(f"{name}  {spikes.size}  {np.median(spikes):.0f}  "
                + "  ".join(str(len(found[rule])) for rule in rules))
        for rule in rules:
            events[rule] += len(found[rule])
            for e in found[rule]:
                rep.say(f"  {rule}: start {e['start_s']:.3f} s  duration {e['duration_s']:.3f} s"
                        f"{'  still locked at the end' if e['to_end'] else ''}")
    for rule in rules:
        rep.say(f"{rule}: {events[rule]} events in {flights} flights")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
