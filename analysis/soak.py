import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from analysis.fly_pool import fly_model
from analysis.lockin_check import TICK_S, lockin_events
from controllers.fly import FlyReal
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report

OUT_PATH = REPO_ROOT / "results" / "soak.txt"
SPIKES_DIR = REPO_ROOT / "results" / "runs" / "soak"
TICKS = 4000
SEEDS = tuple(range(20))
BEARING_STEP_DEG = 2.0
BEARING_MAX_DEG = 180.0
RANGE_NEAR_MM = 20.0
RANGE_FAR_MM = 400.0
SWEEP_TICKS = 400

_ctrl: dict = {}


def bearing_walk_rad(seed: int, ticks: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(ticks)
    b = float(rng.uniform(-BEARING_MAX_DEG, BEARING_MAX_DEG))
    for k in range(ticks):
        b += float(rng.normal(0.0, BEARING_STEP_DEG))
        if abs(b) > BEARING_MAX_DEG:
            b = math.copysign(2.0 * BEARING_MAX_DEG - abs(b), b)
        out[k] = math.radians(b)
    return out


def range_sweep_mm(ticks: int) -> np.ndarray:
    phase = np.arange(ticks) % (2 * SWEEP_TICKS) / SWEEP_TICKS
    return RANGE_FAR_MM - (RANGE_FAR_MM - RANGE_NEAR_MM) * (1.0 - np.abs(phase - 1.0))


def soak_run(job: tuple[float, int]) -> np.ndarray:
    scale, seed = job
    if "ctrl" not in _ctrl:
        _ctrl.update(ctrl=FlyReal(), scale=None)
    ctrl = _ctrl["ctrl"]
    if _ctrl["scale"] != scale:
        ctrl.net.set_weight_scale(scale)
        _ctrl["scale"] = scale
    ctrl.reset(seed, {})
    bearing = bearing_walk_rad(seed, TICKS)
    range_mm = range_sweep_mm(TICKS)
    spikes = np.zeros(TICKS, dtype=np.int64)
    for k in range(TICKS):
        obs = {"bearing": float(bearing[k]), "range": float(range_mm[k]), "vel_body": [200.0, 0.0],
               "yaw_rate": 0.0, "air_rel": [200.0, 0.0]}
        ctrl.act(obs, k * TICK_S)
        spikes[k] = ctrl.spikes_total
    return spikes


def poisson_interval(count: int, exposure_s: float) -> tuple[float, float]:
    low = chi2.ppf(0.025, 2 * count) / 2.0 if count else 0.0
    high = chi2.ppf(0.975, 2 * count + 2) / 2.0
    return 100.0 * low / exposure_s, 100.0 * high / exposure_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--weight-scale", nargs="+", type=float, default=[fly_model()["weight_scale"]])
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    jobs = [(scale, s) for scale in args.weight_scale for s in SEEDS]
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(soak_run, jobs))
    SPIKES_DIR.mkdir(parents=True, exist_ok=True)
    rep = Report()
    rep.say(f"soak test: fly_real on synthetic observations, no simulator, {TICKS} ticks "
            f"({TICKS * TICK_S:g} s), {len(SEEDS)} seeds, {args.workers} workers")
    rep.say(f"bearing: random walk, {BEARING_STEP_DEG:g} deg sd per tick, reflected at "
            f"+-{BEARING_MAX_DEG:g} deg, random start; range: {RANGE_FAR_MM:g} to {RANGE_NEAR_MM:g} mm "
            f"and back, {SWEEP_TICKS * TICK_S:g} s each way")
    for scale in args.weight_scale:
        runs = [(s, spikes) for (sc, s), spikes in zip(jobs, results) if sc == scale]
        np.savez_compressed(SPIKES_DIR / f"soak_scale{scale:g}.npz",
                            **{f"s{s}": spikes for s, spikes in runs})
        rep.say(f"== weight scale {scale:g}")
        events = 0
        locked_runs = 0
        at_risk_s = 0.0
        for s, spikes in runs:
            found = lockin_events(spikes)
            events += len(found)
            locked_runs += bool(found)
            at_risk_s += TICKS * TICK_S - sum(e["duration_s"] for e in found)
            rep.say(f"  seed {s:2d}  median {np.median(spikes):6.0f}  max {spikes.max():5d}  events "
                    f"{len(found)}" + "".join(
                        f"  [start {e['start_s']:.3f} s, {e['duration_s']:.3f} s"
                        f"{', to end' if e['to_end'] else ''}]" for e in found))
        exposure_s = len(runs) * TICKS * TICK_S
        low, high = poisson_interval(events, exposure_s)
        rep.say(f"  runs with a lock-in {locked_runs}/{len(runs)}; events {events}; exposure "
                f"{exposure_s:g} s; {100.0 * events / exposure_s:.3f} events per 100 s; exact 95% "
                f"interval {low:.3f} to {high:.3f}")
        if events:
            low, high = poisson_interval(events, at_risk_s)
            rep.say(f"  counting only unlocked time: exposure {at_risk_s:.3f} s; "
                    f"{100.0 * events / at_risk_s:.3f} events per 100 s; exact 95% interval "
                    f"{low:.3f} to {high:.3f}")
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
