import argparse
import ctypes
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import beta

from analysis.lockin_check import ABS_MIN_TICKS, ABS_SPIKES, TICK_S, lockin_events, read_spikes
from controllers.fly import FlyReal, FlySteerOnly
from env.params import CONFIG_DIR, load_physics
from env.runlog import run_path
from env.sensors import Sensors, to_obs_dict
from fly.check_load import REPO_ROOT, Report

OUT_PATH = REPO_ROOT / "results" / "soak_representative.txt"
RECORDS_PATH = REPO_ROOT / "results" / "runs" / "soak_representative" / "records.jsonl"
SCALES = (0.6, 0.55, 0.5)
SEEDS = tuple(range(2000, 2050))
FAMILIES = ("A1", "A2", "B")
TRACES = {"A1": ("s3_gust", "fly_real", 4), "A2": ("s1_still", "fly_steer_only", 0)}
AFTER_ONSET_TICKS = 600
B_TICKS = 2000
RANGE_LOW_MM = 300.0
RANGE_HIGH_MM = 350.0
RANGE_STEP_MM = 0.05
RAM_PER_JOB_GB = 1.5
MAX_JOBS = 6

_state: dict = {}


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                *[(n, ctypes.c_ulonglong) for n in ("total", "avail", "tp", "ap", "tv", "av", "ae")]]


def allowed_jobs() -> int:
    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return max(1, min(MAX_JOBS, int(status.avail / 2**30 / RAM_PER_JOB_GB)))


def trace_obs(family: str) -> list[dict]:
    name, controller, seed = TRACES[family]
    path = run_path(name, controller, seed)
    onset = round(lockin_events(read_spikes(path))[0]["start_s"] / TICK_S)
    rows = [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines()]
    return [r["obs"] for r in rows if r["type"] == "tick"][:onset + AFTER_ONSET_TICKS]


def family_b_obs(seed: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    hold_rad = float(rng.uniform(-2.4, -1.2))
    hold_ticks = round(float(rng.uniform(0.8, 1.2)) / TICK_S)
    swing_ticks = round(float(rng.uniform(0.2, 0.5)) / TICK_S)
    end_rad = float(rng.uniform(-0.2, 0.2))
    range_mm = float(rng.uniform(RANGE_LOW_MM, RANGE_HIGH_MM))
    scenario = json.loads((CONFIG_DIR / "scenarios" / "s4_turbulent.json").read_text(encoding="utf-8"))
    sensors = Sensors(load_physics()["sensors"], scenario["noise_multiplier"],
                      np.random.default_rng([seed, 1]), scenario["sensor_delay_scale"],
                      scenario["dropout"])
    obs = []
    for k in range(B_TICKS):
        frac = min(max((k - hold_ticks) / swing_ticks, 0.0), 1.0)
        bearing = hold_rad + (end_rad - hold_rad) * frac
        range_mm += float(rng.normal(0.0, RANGE_STEP_MM))
        range_mm = min(max(range_mm, RANGE_LOW_MM), RANGE_HIGH_MM)
        sample = [bearing, range_mm, 200.0, 0.0, 0.0, 200.0, 0.0]
        if k == 0:
            sensors.reset(sample)
        obs.append(to_obs_dict(sensors.observe(sample)))
    return obs, {"hold_rad": hold_rad, "hold_ticks": hold_ticks, "swing_ticks": swing_ticks,
                 "end_rad": end_rad}


def worker_state() -> dict:
    if not _state:
        _state.update(fly_real=FlyReal(), fly_steer_only=FlySteerOnly(), scale=None,
                      A1=trace_obs("A1"), A2=trace_obs("A2"))
    return _state


def soak_run(job: tuple[float, str, int]) -> dict:
    scale, family, seed = job
    state = worker_state()
    ctrl = state["fly_steer_only" if family == "A2" else "fly_real"]
    if state["scale"] != scale:
        ctrl.net.set_weight_scale(scale)
        state["scale"] = scale
    ctrl.reset(seed, {})
    obs, drawn = family_b_obs(seed) if family == "B" else (state[family], {})
    spikes = np.zeros(len(obs), dtype=np.int64)
    for k, ob in enumerate(obs):
        ctrl.act(ob, k * TICK_S)
        spikes[k] = ctrl.spikes_total
    events = lockin_events(spikes)
    rolling = np.convolve(spikes, np.ones(ABS_MIN_TICKS) / ABS_MIN_TICKS, "valid")
    return {"scale": scale, "family": family, "seed": seed, "ticks": len(obs),
            "lockin": bool(events), "onset_s": events[0]["start_s"] if events else None,
            "max_roll": float(rolling.max()), "mean": float(spikes.mean()),
            "ceiling_frac": float((spikes > ABS_SPIKES).mean()), **drawn}


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    low = float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0
    high = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return low, high


def read_records(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines()]
    return {(r["scale"], r["family"], r["seed"]): r for r in rows}


def rate_row(k: int, n: int) -> str:
    if n == 0:
        return f"{k} {n} - -"
    low, high = clopper_pearson(k, n)
    return f"{k} {n} {low:.4f} {high:.4f}"


def report(records: dict, wall_s: float, out: Path, note: str = "") -> None:
    rep = Report()
    rep.say(f"representative soak; lock-in = above {ABS_SPIKES:g} spikes per tick for "
            f"{ABS_MIN_TICKS * TICK_S:g} s; seeds {SEEDS[0]}-{SEEDS[-1]}")
    if note:
        rep.say(note)
    rep.say("scale family lockins n cp95_low cp95_high mean_onset_s status")
    for scale in SCALES:
        pooled = 0
        for family in FAMILIES:
            rows = [records[(scale, family, s)] for s in SEEDS if (scale, family, s) in records]
            k = sum(r["lockin"] for r in rows)
            pooled += k
            onset = [r["onset_s"] for r in rows if r["lockin"]]
            status = "" if len(rows) == len(SEEDS) else " PARTIAL"
            rep.say(f"{scale:g} {family} {rate_row(k, len(rows))} "
                    + (f"{np.mean(onset):.3f}" if onset else "-") + status)
        n = sum(1 for key in records if key[0] == scale)
        status = "" if n == len(SEEDS) * len(FAMILIES) else " PARTIAL"
        rep.say(f"{scale:g} pooled {rate_row(pooled, n)} -{status}")
    rep.say(f"run wall {wall_s:.0f} s")
    rep.save(out)
    for scale in SCALES:
        detail = Report()
        detail.say(f"soak detail, scale {scale:g}")
        detail.say(f"A1, A2: logged obs of {TRACES['A1']} and {TRACES['A2']} to onset + "
                   f"{AFTER_ONSET_TICKS * TICK_S:g} s, open loop, network seed varies; A1 fly_real, "
                   "A2 fly_steer_only; mapping and gains as in configs")
        detail.say(f"B: fly_real, {B_TICKS * TICK_S:g} s. bearing holds uniform(-2.4, -1.2) rad for "
                   "uniform(0.8, 1.2) s, swings linearly over uniform(0.2, 0.5) s to uniform(-0.2, 0.2) "
                   f"rad, holds. range uniform({RANGE_LOW_MM:g}, {RANGE_HIGH_MM:g}) mm, walk "
                   f"{RANGE_STEP_MM:g} mm sd per tick, clipped. velocity 200, 0 mm/s, yaw rate 0, air "
                   "200, 0. all of it through env Sensors at S4 noise, delay and dropout")
        detail.say("ceiling_frac = share of ticks above the detector level")
        detail.say("family seed ticks lockin onset_s max_0.5s_mean mean_spikes ceiling_frac")
        for key in sorted(k for k in records if k[0] == scale):
            r = records[key]
            detail.say(f"{r['family']} {r['seed']} {r['ticks']} {r['lockin']} {r['onset_s']} "
                       f"{r['max_roll']:.1f} {r['mean']:.1f} {r['ceiling_frac']:.4f}")
        detail.save(out.with_name(f"{out.stem}_s{scale:g}.txt"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--records", default=str(RECORDS_PATH))
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--wall-s", type=float, default=0.0)
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    path = Path(args.records)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = read_records(path)
    if args.report_only:
        report(records, args.wall_s, Path(args.out), args.note)
        return
    jobs = [(scale, family, s) for scale in SCALES for family in FAMILIES for s in SEEDS
            if (scale, family, s) not in records]
    workers = args.workers or allowed_jobs()
    print(f"{len(jobs)} episodes to run, {workers} workers", flush=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool, path.open("a", encoding="utf-8", newline="\n") as f:
        futures = [pool.submit(soak_run, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures)):
            row = fut.result()
            records[(row["scale"], row["family"], row["seed"])] = row
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(jobs)} {time.perf_counter() - start:.0f} s", flush=True)
    report(records, time.perf_counter() - start, Path(args.out))


if __name__ == "__main__":
    main()
