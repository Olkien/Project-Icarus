import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from analysis.fly_pool import NETWORKS, WINDOW_MS, fly_model, run_all, side_ids
from analysis.push_pull import AMPLITUDES_HZ, push_pull_rates
from analysis.steering_test import (
    BEARINGS_DEG,
    DNA02,
    LC10,
    MAPPING_PATH,
    SEEDS,
    evaluate,
    yaw_signal,
)
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import ConnectomeNetwork

OUT_PATH = REPO_ROOT / "results" / "yaw_zero.txt"
CENTRES_HZ = (25.0, 30.0, 35.0, 40.0, 45.0)
SMALL_DEG = (5, 20)
SMALL_LEVELS = (5, 10, 20)
SMALL_MIN = 0.8


def rate_key(bearing_deg: int, c: float, a: float, seed: int) -> tuple:
    rl, rr = push_pull_rates(math.radians(bearing_deg), c, a)
    return round(rl, 6), round(rr, 6), seed


def small_angle(yaws: dict) -> tuple[float, dict]:
    per_level = {}
    for level in SMALL_LEVELS:
        cases = [(b, y) for b in (level, -level) for y in yaws[b]]
        per_level[level] = sum((y > 0) if b > 0 else (y < 0) for b, y in cases) / len(cases)
    return float(np.mean(list(per_level.values()))), per_level


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    scale = fly_model()["weight_scale"]
    net = ConnectomeNetwork(NETWORKS["orig"])
    lc10 = side_ids(net, LC10)
    dna02 = side_ids(net, DNA02)
    pass_bearings = [s * b for b in BEARINGS_DEG for s in (1, -1)]
    bearings = sorted(set(pass_bearings) | {s * b for b in SMALL_DEG for s in (1, -1)})
    grid = [(c, a) for c in CENTRES_HZ for a in AMPLITUDES_HZ]
    needed = {rate_key(0, c, 0.0, s): None for c in CENTRES_HZ for s in SEEDS}
    for c, a in grid:
        for b in bearings:
            for s in SEEDS:
                needed[rate_key(b, c, a, s)] = None
    keys = list(needed)
    jobs = [{"kind": "windows", "net": "orig", "scale": scale, "seed": s,
             "drive": [(lc10["L"], rl), (lc10["R"], rr)], "groups": [dna02["L"], dna02["R"]]}
            for rl, rr, s in keys]
    start = time.perf_counter()
    raw = {k: yaw_signal(w) for k, w in zip(keys, run_all(jobs, args.workers))}
    rep = Report()
    rep.say(f"yaw zero-point calibration: weight scale {scale:g}, spike input, 1 s per case, "
            f"seeds 0-19, {len(jobs)} distinct runs")
    rep.say(f"yaw = (DNa02 L - DNa02 R)/(L + R + 1) per {WINDOW_MS:g} ms window, mean over the run; "
            f"at bearing 0 both sides get c, so a does not change the zero point")
    zero = {}
    rep.say("  c_Hz  yaw_zero  sd")
    for c in CENTRES_HZ:
        y = np.array([raw[rate_key(0, c, 0.0, s)] for s in SEEDS])
        zero[c] = round(float(y.mean()), 4)
        rep.say(f"  {c:4g} {zero[c]:9.4f} {y.std(ddof=1):6.4f}")
    rep.say(f"steering rule on yaw - yaw_zero(c): sign >= 95% at |b| >= 30 deg, Spearman >= 0.8; "
            f"small-angle rule: sign >= {100 * SMALL_MIN:g}% over |b| = 5, 10, 20 deg pooled")
    by_a = {a: True for a in AMPLITUDES_HZ}
    for c, a in grid:
        yaws = {b: [raw[rate_key(b, c, a, s)] - zero[c] for s in SEEDS] for b in bearings}
        ev = evaluate(yaws, pass_bearings)
        small, per_level = small_angle(yaws)
        ok = ev["pass"] and small >= SMALL_MIN
        by_a[a] = by_a[a] and ok
        rep.say(f"== c {c:g} Hz, a {a:g} Hz: {'PASS' if ok else 'fail'}  sign "
                f"{100 * ev['sign']:.1f}%  rho {ev['rho']:.3f}  small {100 * small:.1f}% ("
                + ", ".join(f"{lv} deg {100 * v:.1f}%" for lv, v in per_level.items()) + ")")
        rep.say("  bearing_deg  mean_yaw_cal     sd    correct")
        for b in bearings:
            v = np.array(yaws[b])
            right_sign = (v > 0) if b > 0 else (v < 0)
            rep.say(f"  {b:11d} {v.mean():13.4f} {v.std(ddof=1):6.4f}  {int(right_sign.sum()):2d}/20")
        rep.say("  |b| mean |yaw|: " + ", ".join(
            f"{lv} {m:.4f}" for lv, m in zip(ev["levels"], ev["mean_abs"])))
    for a, ok in by_a.items():
        rep.say(f"a {a:g} Hz: {'PASS at every c' if ok else 'fail'}")
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    mapping["yaw_zero"] = {"c_hz": list(CENTRES_HZ), "yaw": [zero[c] for c in CENTRES_HZ],
                           "use": "subtract from the yaw decode, interpolated in c"}
    MAPPING_PATH.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8", newline="\n")
    rep.say(f"wrote yaw_zero to configs/{MAPPING_PATH.name}")
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
