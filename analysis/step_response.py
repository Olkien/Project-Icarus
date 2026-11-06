import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from analysis.lockin_check import TICK_S
from analysis.run import load_params
from controllers.fly import FlyReal
from env.params import CONFIG_DIR
from fly.check_load import REPO_ROOT, Report

OUT_PATH = REPO_ROOT / "results" / "step_response_s0.55.txt"
FLY_PARAMS = CONFIG_DIR / "fly_real_tuned_s0.55.json"
SEEDS = tuple(range(2000, 2020))
PRE_TICKS = 200
POST_TICKS = 200
STEADY_TICKS = 100
FRACTIONS = (0.5, 0.9)
STEPS = {
    "bearing +0.5": ("bearing", (0.0, 300.0), (0.5, 300.0)),
    "bearing -0.5": ("bearing", (0.0, 300.0), (-0.5, 300.0)),
    "range 500 to 150": ("range", (0.0, 500.0), (0.0, 150.0)),
}

_ctrl = None


def obs(bearing: float, range_mm: float) -> dict:
    return {"bearing": bearing, "range": range_mm, "vel_body": [0.0, 0.0], "yaw_rate": 0.0,
            "air_rel": [0.0, 0.0]}


def step_run(job: tuple) -> tuple:
    global _ctrl
    name, seed = job
    if _ctrl is None:
        _ctrl = FlyReal(load_params(str(FLY_PARAMS)))
    _, before, after = STEPS[name]
    _ctrl.reset(seed, {})
    yaw, torque, decode, thrust = [], [], [], []
    for k in range(PRE_TICKS + POST_TICKS):
        action = _ctrl.act(obs(*(before if k < PRE_TICKS else after)), k * TICK_S)
        yaw.append(_ctrl.yaw_signal)
        torque.append(action["yaw_torque"])
        decode.append(_ctrl.thrust_decode)
        thrust.append(action["thrust"])
    return job, {"yaw": np.array(yaw), "torque": np.array(torque), "decode": np.array(decode),
                 "thrust": np.array(thrust)}


def rise_ms(trace: np.ndarray, frac: float) -> float | None:
    base = trace[PRE_TICKS - STEADY_TICKS:PRE_TICKS].mean()
    final = trace[-STEADY_TICKS:].mean()
    change = final - base
    if change == 0:
        return None
    progress = (trace[PRE_TICKS:] - base) / change
    hit = np.flatnonzero(progress >= frac)
    return float((hit[0] + 1) * TICK_S * 1000.0) if hit.size else None


def spread(values: list) -> str:
    v = [x for x in values if x is not None]
    if not v:
        return "none"
    return f"median {np.median(v):.0f}, range {min(v):.0f} to {max(v):.0f}, n {len(v)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    start = time.perf_counter()
    jobs = [(name, s) for name in STEPS for s in SEEDS]
    with ProcessPoolExecutor(args.workers) as pool:
        results = dict(pool.map(step_run, jobs))
    ctrl = FlyReal(load_params(str(FLY_PARAMS)))
    rep = Report()
    rep.say(f"fly step response, open loop, weight scale {ctrl.weight_scale:g}, {FLY_PARAMS.name} (gain_yaw "
            f"{ctrl.gain_yaw:g}, gain_thrust {ctrl.gain_thrust:.4f}), yaw_zero on, network seeds "
            f"{SEEDS[0]}-{SEEDS[-1]}")
    rep.say(f"obs given straight to the controller, no sensor noise, delay or dropout; vel, yaw rate, air "
            f"at 0; {PRE_TICKS * TICK_S:g} s before the step, {POST_TICKS * TICK_S:g} s after")
    rep.say(f"base = mean of the last {STEADY_TICKS * TICK_S:g} s before the step, final = mean of the last "
            f"{STEADY_TICKS * TICK_S:g} s after; rise time = first tick after the step at which "
            "(value - base) / (final - base) reaches the fraction, counted in ticks of 5 ms")
    rep.say(f"the yaw and thrust decodes sum spikes over a {ctrl.steer['window_ms']:g} ms window")
    for name, (kind, _, _) in STEPS.items():
        keys = ("yaw", "torque") if kind == "bearing" else ("decode", "thrust")
        rep.say(f"== {name}")
        for key in keys:
            traces = np.array([results[(name, s)][key] for s in SEEDS])
            mean = traces.mean(axis=0)
            base = traces[:, PRE_TICKS - STEADY_TICKS:PRE_TICKS].mean(axis=1)
            final = traces[:, -STEADY_TICKS:].mean(axis=1)
            sd_pre = traces[:, PRE_TICKS - STEADY_TICKS:PRE_TICKS].std(axis=1, ddof=1)
            sd_post = traces[:, -STEADY_TICKS:].std(axis=1, ddof=1)
            rep.say(f"  {key}: base mean {base.mean():.4f}, final mean {final.mean():.4f} (seed sd "
                    f"{final.std(ddof=1):.4f})")
            for frac in FRACTIONS:
                per_seed = [rise_ms(t, frac) for t in traces]
                rep.say(f"    {100 * frac:g}% rise ms, per seed: {spread(per_seed)}; on the 20-seed mean "
                        f"trace: {rise_ms(mean, frac)}")
            rep.say(f"    steady noise, sd over ticks within a seed, mean over seeds: before "
                    f"{sd_pre.mean():.4f}, after {sd_post.mean():.4f}")
    rep.say(f"total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
