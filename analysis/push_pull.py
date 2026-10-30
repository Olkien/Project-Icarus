import argparse
import math
import time
from pathlib import Path

import numpy as np

from analysis.fly_pool import NETWORKS, WINDOW_MS, fly_model, run_all, side_ids
from analysis.steering_test import BEARINGS_DEG, DNA02, LC10, SEEDS, evaluate, yaw_signal
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import ConnectomeNetwork

OUT_PATH = REPO_ROOT / "results" / "push_pull_steering.txt"
CENTRES_HZ = (30.0, 35.0, 40.0)
AMPLITUDES_HZ = (20.0, 25.0)
RATE_MAX_HZ = 80.0


def push_pull_rates(bearing_rad: float, c: float, a: float) -> tuple[float, float]:
    push = a * bearing_rad / math.pi
    return (min(max(c + push, 0.0), RATE_MAX_HZ), min(max(c - push, 0.0), RATE_MAX_HZ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    scale = fly_model()["weight_scale"]
    net = ConnectomeNetwork(NETWORKS["orig"])
    lc10 = side_ids(net, LC10)
    dna02 = side_ids(net, DNA02)
    bearings = [s * b for b in BEARINGS_DEG for s in (1, -1)]
    settings = [(c, a) for c in CENTRES_HZ for a in AMPLITUDES_HZ]
    needed = {}
    for c, a in settings:
        for b in bearings:
            rl, rr = push_pull_rates(math.radians(b), c, a)
            for s in SEEDS:
                needed[(round(rl, 6), round(rr, 6), s)] = None
    keys = list(needed)
    jobs = [{"kind": "windows", "net": "orig", "scale": scale, "seed": s,
             "drive": [(lc10["L"], rl), (lc10["R"], rr)], "groups": [dna02["L"], dna02["R"]]}
            for rl, rr, s in keys]
    start = time.perf_counter()
    windows = dict(zip(keys, run_all(jobs, args.workers)))
    rep = Report()
    rep.say(f"push-pull steering test: weight scale {scale:g}, spike input, 1 s per case, "
            f"seeds 0-19, {len(jobs)} distinct runs")
    rep.say(f"mapping: LC10 L = c + a b/pi, LC10 R = c - a b/pi, clipped to 0-{RATE_MAX_HZ:g} Hz; "
            f"yaw = (DNa02 L - DNa02 R)/(L + R + 1) per {WINDOW_MS:g} ms window, mean over the run")
    rep.say("pass: sign correct in >= 95% of cases with |b| >= 30 deg, and Spearman(|b|, mean "
            "|yaw|) >= 0.8; margin = min of the two excesses")
    passed = []
    for c, a in settings:
        yaws = {}
        for b in bearings:
            rl, rr = push_pull_rates(math.radians(b), c, a)
            yaws[b] = [yaw_signal(windows[(round(rl, 6), round(rr, 6), s)]) for s in SEEDS]
        ev = evaluate(yaws, bearings)
        if ev["pass"]:
            passed.append((c, a))
        rep.say(f"== c {c:g} Hz, a {a:g} Hz: {'PASS' if ev['pass'] else 'fail'}  sign "
                f"{100 * ev['sign']:.1f}%  rho {ev['rho']:.3f}  margin {ev['margin']:+.3f}")
        rep.say("  bearing_deg  LC10_L  LC10_R  mean_yaw     sd    correct")
        for b in sorted(bearings):
            y = np.array(yaws[b])
            rl, rr = push_pull_rates(math.radians(b), c, a)
            right_sign = (y > 0) if b > 0 else (y < 0)
            rep.say(f"  {b:11d} {rl:7.2f} {rr:7.2f} {y.mean():9.4f} {y.std(ddof=1):6.4f}  "
                    f"{int(right_sign.sum()):2d}/20")
        rep.say("  |b| mean |yaw|: " + ", ".join(
            f"{lv} {m:.4f}" for lv, m in zip(ev["levels"], ev["mean_abs"])))
    rep.say("passing settings (c, a): " + (", ".join(f"({c:g}, {a:g})" for c, a in passed)
                                          or "none") + "; no setting picked yet")
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
