import itertools
import math

MISS_TIME_S = 6.0
RECOVERY_BAND_MM = 15.0
RECOVERY_HOLD_S = 0.1


def cross_track_mm(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        return math.hypot(px - ax, py - ay)
    u = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
    return math.hypot(px - (ax + u * dx), py - (ay + u * dy))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def gust_end_s(scenario: dict) -> float | None:
    ends = [c["t_start_s"] + c["duration_s"] for c in scenario.get("wind", [])
            if c["type"] == "gust"]
    return max(ends) if ends else None


def segments(ticks: list[dict], tick_s: float) -> list[dict]:
    segs = [{"i": 0, "start": 0, "hit": None}]
    for idx, record in enumerate(ticks):
        for _ in range(record["events"].count("hit")):
            segs[-1]["hit"] = idx
            segs[-1]["t_hit"] = (record["k"] + 1) * tick_s
            segs.append({"i": segs[-1]["i"] + 1, "start": idx + 1, "hit": None})
    return [s for s in segs if s["start"] < len(ticks)]


def position(record: dict) -> tuple[float, float]:
    return record["state"]["x"], record["state"]["y"]


def compute_metrics(
    ticks: list[dict], scenario: dict, tick_s: float, act_s_total: float | None = None
) -> dict:
    count = int(scenario["targets"]["count"])
    segs = segments(ticks, tick_s)
    hit_segs = [s for s in segs if s["hit"] is not None]
    starts = {s["i"]: position(ticks[s["start"]]) for s in segs}

    times = []
    efficiencies = []
    for s in hit_segs:
        times.append(s["t_hit"] - ticks[s["start"]]["t"])
        end = min(s["hit"] + 1, len(ticks) - 1)
        path = 0.0
        for a, b in zip(ticks[s["start"]:end], ticks[s["start"] + 1:end + 1]):
            (ax, ay), (bx, by) = position(a), position(b)
            path += math.hypot(bx - ax, by - ay)
        sx, sy = starts[s["i"]]
        target = ticks[s["hit"]]["target"]
        straight = math.hypot(target["x"] - sx, target["y"] - sy)
        if path > 0.0:
            efficiencies.append(straight / path)

    cross = []
    for record in ticks:
        sx, sy = starts[record["target"]["i"]]
        px, py = position(record)
        cross.append(cross_track_mm(px, py, sx, sy, record["target"]["x"], record["target"]["y"]))

    hits = len(hit_segs)
    misses = count - hits
    oob = any("oob" in r["events"] for r in ticks)
    end_s = (ticks[-1]["k"] + 1) * tick_s if ticks else 0.0

    recovery = None
    t_end_gust = gust_end_s(scenario)
    if t_end_gust is not None and end_s > t_end_gust:
        recovery = end_s - t_end_gust
        run_start = None
        for record, ct in zip(ticks, cross):
            if record["t"] < t_end_gust:
                continue
            if ct >= RECOVERY_BAND_MM:
                run_start = None
                continue
            if run_start is None:
                run_start = record["t"]
            if record["t"] - run_start >= RECOVERY_HOLD_S - 1e-9:
                recovery = run_start - t_end_gust
                break

    thrusts = [r["action"]["thrust"] for r in ticks]
    yaws = [r["action"]["yaw_torque"] for r in ticks]
    spikes = [r["info"]["spikes_total"] for r in ticks if "spikes_total" in r["info"]]
    n = len(ticks)
    return {
        "episode_success": hits == count and not oob,
        "targets_hit": hits,
        "misses": misses,
        "time_per_target_s": mean(times),
        "time_per_target_miss6_s": (sum(times) + MISS_TIME_S * misses) / count,
        "path_efficiency": mean(efficiencies),
        "rms_cross_track_mm": math.sqrt(sum(c * c for c in cross) / n) if n else None,
        "max_cross_track_mm": max(cross) if cross else None,
        "recovery_time_s": recovery,
        "effort_thrust": mean(thrusts),
        "effort_yaw": mean([abs(y) for y in yaws]),
        "jitter_yaw": mean([abs(b - a) for a, b in itertools.pairwise(yaws)]),
        "out_of_bounds": oob,
        "episode_s": end_s,
        "ctrl_ms_per_tick": 1000.0 * act_s_total / n if act_s_total is not None and n else None,
        "spikes_per_tick": mean(spikes),
    }


def tuning_cost(metrics: dict) -> float:
    return metrics["time_per_target_miss6_s"] + 0.01 * metrics["rms_cross_track_mm"]
