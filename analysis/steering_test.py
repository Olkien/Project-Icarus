import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analysis.fly_pool import NETWORKS, WINDOW_MS, fly_model, run_all, side_ids
from env.params import CONFIG_DIR
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import ConnectomeNetwork

OUT_PATH = REPO_ROOT / "results" / "steering_test.txt"
MAPPING_PATH = CONFIG_DIR / "fly_mapping_real.json"
R_MIN = (0.0, 10.0)
R_MAX = (20.0, 40.0, 60.0, 80.0)
BEARINGS_DEG = (10, 30, 60, 90, 120, 150, 180)
SEEDS = tuple(range(20))
SIGN_MIN = 0.95
RHO_MIN = 0.8
SIGN_FROM_DEG = 30
LC10 = "type^=LC10"
DNA02 = "type=DNa02"


def lc10_rates(bearing_rad: float, r_min: float, r_max: float) -> tuple[float, float]:
    left = r_min + (r_max - r_min) * max(0.0, bearing_rad) / math.pi
    right = r_min + (r_max - r_min) * max(0.0, -bearing_rad) / math.pi
    return left, right


def yaw_signal(windows: np.ndarray) -> float:
    left, right = windows[:, 0].astype(float), windows[:, 1].astype(float)
    return float(np.mean((left - right) / (left + right + 1.0)))


def evaluate(yaws: dict, bearings: list[int]) -> dict:
    cases = [(b, y) for b in bearings for y in yaws[b] if abs(b) >= SIGN_FROM_DEG]
    correct = sum((y > 0) if b > 0 else (y < 0) for b, y in cases)
    frac = correct / len(cases)
    levels = sorted({abs(b) for b in bearings})
    mean_abs = [np.mean([abs(y) for b in bearings if abs(b) == lv for y in yaws[b]])
                for lv in levels]
    rho = float(spearmanr(levels, mean_abs).statistic)
    if math.isnan(rho):
        rho = 0.0
    return {"sign": frac, "rho": rho, "levels": levels, "mean_abs": mean_abs,
            "pass": frac >= SIGN_MIN and rho >= RHO_MIN,
            "margin": min(frac - SIGN_MIN, rho - RHO_MIN)}


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
    settings = [(lo, hi) for lo in R_MIN for hi in R_MAX]
    needed = {}
    for lo, hi in settings:
        for b in bearings:
            rl, rr = lc10_rates(math.radians(b), lo, hi)
            for s in SEEDS:
                needed[(round(rl, 6), round(rr, 6), s)] = None
    keys = list(needed)
    jobs = [{"kind": "windows", "net": "orig", "scale": scale, "seed": s,
             "drive": [(lc10["L"], rl), (lc10["R"], rr)], "groups": [dna02["L"], dna02["R"]]}
            for rl, rr, s in keys]
    start = time.perf_counter()
    windows = dict(zip(keys, run_all(jobs, args.workers)))
    rep = Report()
    rep.say(f"open-loop steering test (T3): weight scale {scale:g}, spike input, 1 s per case, "
            f"seeds 0-19, {len(jobs)} distinct runs")
    rep.say("mapping: LC10 L = r_min + (r_max - r_min) max(0, b)/pi, LC10 R = r_min + "
            "(r_max - r_min) max(0, -b)/pi; yaw = (DNa02 L - DNa02 R)/(L + R + 1) per "
            f"{WINDOW_MS:g} ms window, mean over the run")
    rep.say(f"pass: sign correct in >= {100 * SIGN_MIN:g}% of cases with |b| >= {SIGN_FROM_DEG} deg, "
            f"and Spearman(|b|, mean |yaw|) >= {RHO_MIN:g}; margin = min of the two excesses")
    outcomes = []
    for lo, hi in settings:
        yaws = {}
        for b in bearings:
            rl, rr = lc10_rates(math.radians(b), lo, hi)
            yaws[b] = [yaw_signal(windows[(round(rl, 6), round(rr, 6), s)]) for s in SEEDS]
        ev = evaluate(yaws, bearings)
        outcomes.append(((lo, hi), ev))
        rep.say(f"== r_min {lo:g} Hz, r_max {hi:g} Hz: {'PASS' if ev['pass'] else 'fail'}  sign "
                f"{100 * ev['sign']:.1f}%  rho {ev['rho']:.3f}  margin {ev['margin']:+.3f}")
        rep.say("  bearing_deg  mean_yaw     sd    correct")
        for b in sorted(bearings):
            y = np.array(yaws[b])
            right_sign = (y > 0) if b > 0 else (y < 0)
            rep.say(f"  {b:11d} {y.mean():9.4f} {y.std(ddof=1):6.4f}  {int(right_sign.sum()):2d}/20")
        rep.say("  |b| mean |yaw|: " + ", ".join(
            f"{lv} {m:.4f}" for lv, m in zip(ev["levels"], ev["mean_abs"])))
    passing = [o for o in outcomes if o[1]["pass"]]
    pool = passing or outcomes
    (lo, hi), ev = max(pool, key=lambda o: o[1]["margin"])
    if passing:
        rep.say(f"chosen: r_min {lo:g} Hz, r_max {hi:g} Hz (largest pass margin {ev['margin']:+.3f})")
        mapping = {
            "network": {"type": "connectome", "path": "data/network_cache/network_A.npz",
                        "weight_scale": scale},
            "input_mode": "spike",
            "steering": {"input_left": f"{LC10},somaSide=L", "input_right": f"{LC10},somaSide=R",
                         "r_min_hz": lo, "r_max_hz": hi,
                         "rate_left": "r_min + (r_max - r_min) * max(0, bearing) / pi",
                         "rate_right": "r_min + (r_max - r_min) * max(0, -bearing) / pi",
                         "output_left": f"{DNA02},somaSide=L", "output_right": f"{DNA02},somaSide=R",
                         "window_ms": WINDOW_MS,
                         "yaw": "(spikes_left - spikes_right) / (spikes_left + spikes_right + 1)"},
            "thrust": None,
            "chosen_by": f"T3 open loop, sign {ev['sign']:.3f}, rho {ev['rho']:.3f}",
        }
        MAPPING_PATH.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8",
                                newline="\n")
        rep.say(f"wrote configs/{MAPPING_PATH.name}")
    else:
        rep.say(f"NO setting passes; closest is r_min {lo:g} Hz, r_max {hi:g} Hz "
                f"(margin {ev['margin']:+.3f}); no mapping written")
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
