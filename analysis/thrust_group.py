import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analysis.fly_pool import NETWORKS, WINDOW_MS, fly_model, run_all, side_ids
from analysis.steering_test import LC10, MAPPING_PATH, lc10_rates
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import ConnectomeNetwork

OUT_PATH = REPO_ROOT / "results" / "thrust_group_test.txt"
CANDIDATES = {
    "DLMn + DVMn": ["type^=DLMn", "type^=DVMn"],
    "DNg33": ["type=DNg33"],
    "DNp31": ["type=DNp31"],
}
BASELINES_HZ = (5.0, 10.0, 20.0, 30.0, 60.0)
SEEDS = tuple(range(20))
RISE_RHO_MIN = 0.9
MAX_HZ = 200.0
MAX_HZ_AT = 30.0
CV_MAX = 0.3
CV_AT = 20.0
STEER_BEARINGS_DEG = (90, -90)
STEER_BASELINE_HZ = 10.0
STEER_CHANGE_MAX = 0.25


def group_ids(net: ConnectomeNetwork, queries: list[str]) -> np.ndarray:
    return np.unique(np.concatenate([net.group(q) for q in queries]))


def steering_setting() -> tuple[float, float]:
    steer = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["steering"]
    return float(steer["r_min_hz"]), float(steer["r_max_hz"])


def lc10_drive(lc10: dict, baseline_hz: float, bearing_deg: float | None,
               setting: tuple[float, float]) -> list[tuple[np.ndarray, float]]:
    left = right = baseline_hz
    if bearing_deg is not None:
        sl, sr = lc10_rates(math.radians(bearing_deg), *setting)
        left, right = left + sl, right + sr
    return [(lc10["L"], left), (lc10["R"], right)]


def per_neuron_decode(windows: np.ndarray, sizes: list[int]) -> np.ndarray:
    return windows.mean(axis=0) / np.array(sizes, dtype=float)


def check(curve: dict, steer: dict) -> dict:
    levels = list(curve)
    means = [float(np.mean(curve[b])) for b in levels]
    rho = float(spearmanr(levels, means).statistic)
    rho = 0.0 if math.isnan(rho) else rho
    hz_at = float(np.mean(curve[MAX_HZ_AT])) * 1000.0 / WINDOW_MS
    cv = float(np.std(curve[CV_AT], ddof=1) / np.mean(curve[CV_AT])) if np.mean(curve[CV_AT]) > 0 \
        else math.inf
    base = float(np.mean(curve[STEER_BASELINE_HZ]))
    changes = {b: abs(float(np.mean(v)) - base) / base if base > 0 else math.inf
               for b, v in steer.items()}
    rules = {"rise": rho >= RISE_RHO_MIN, "max_hz": hz_at < MAX_HZ, "cv": cv < CV_MAX,
             "steer": max(changes.values()) < STEER_CHANGE_MAX}
    return {"means": means, "rho": rho, "hz_at": hz_at, "cv": cv, "changes": changes,
            "rules": rules, "pass": all(rules.values()), "passed": sum(rules.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    scale = fly_model()["weight_scale"]
    setting = steering_setting()
    net = ConnectomeNetwork(NETWORKS["orig"])
    lc10 = side_ids(net, LC10)
    groups = [group_ids(net, q) for q in CANDIDATES.values()]
    sizes = [g.size for g in groups]
    conditions = [(b, None) for b in BASELINES_HZ] + [(STEER_BASELINE_HZ, d) for d in STEER_BEARINGS_DEG]
    jobs = [{"kind": "windows", "net": "orig", "scale": scale, "seed": s, "groups": groups,
             "drive": lc10_drive(lc10, b, d, setting)} for b, d in conditions for s in SEEDS]
    start = time.perf_counter()
    results = run_all(jobs, args.workers)
    decoded = {}
    for k, cond in enumerate(conditions):
        runs = results[k * len(SEEDS):(k + 1) * len(SEEDS)]
        decoded[cond] = np.array([per_neuron_decode(w, sizes) for w in runs])
    rep = Report()
    rep.say(f"thrust group test: weight scale {scale:g}, spike input, 1 s, seeds 0-19; both LC10 "
            f"sides at the baseline, no steering unless stated")
    rep.say(f"decode = spikes per {WINDOW_MS:g} ms window per neuron (group pooled), mean over windows")
    rep.say(f"rules: Spearman(baseline, decode) >= {RISE_RHO_MIN:g}; per-neuron rate under {MAX_HZ:g} "
            f"Hz at {MAX_HZ_AT:g} Hz; CV across seeds under {CV_MAX:g} at {CV_AT:g} Hz; steering at "
            f"+-90 deg with baseline {STEER_BASELINE_HZ:g} Hz changes the decode by under "
            f"{100 * STEER_CHANGE_MAX:g}%")
    rep.say(f"steering drive = Task 1 mapping (r_min {setting[0]:g}, r_max {setting[1]:g} Hz) added "
            f"on top of the baseline on each side")
    outcomes = []
    for j, (name, queries) in enumerate(CANDIDATES.items()):
        curve = {b: decoded[(b, None)][:, j] for b in BASELINES_HZ}
        steer = {d: decoded[(STEER_BASELINE_HZ, d)][:, j] for d in STEER_BEARINGS_DEG}
        ev = check(curve, steer)
        outcomes.append((name, ev))
        rep.say(f"== {name} ({' + '.join(queries)}, {sizes[j]} neurons): "
                f"{'PASS' if ev['pass'] else 'fail'} ({ev['passed']}/4 rules)")
        rep.say("  baseline_Hz  decode/window  sd      Hz/neuron")
        for b in BASELINES_HZ:
            rep.say(f"  {b:11g} {np.mean(curve[b]):14.4f} {np.std(curve[b], ddof=1):7.4f} "
                    f"{np.mean(curve[b]) * 1000.0 / WINDOW_MS:9.2f}")
        for d in STEER_BEARINGS_DEG:
            rep.say(f"  steer {d:+4d} deg at {STEER_BASELINE_HZ:g} Hz: decode {np.mean(steer[d]):.4f} "
                    f"(sd {np.std(steer[d], ddof=1):.4f}), change {100 * ev['changes'][d]:.1f}%")
        r = ev["rules"]
        rep.say(f"  rise rho {ev['rho']:.3f} {'ok' if r['rise'] else 'no'}; rate at {MAX_HZ_AT:g} Hz "
                f"{ev['hz_at']:.2f} Hz {'ok' if r['max_hz'] else 'no'}; CV at {CV_AT:g} Hz "
                f"{ev['cv']:.3f} {'ok' if r['cv'] else 'no'}; steer change "
                f"{100 * max(ev['changes'].values()):.1f}% {'ok' if r['steer'] else 'no'}")
    winner = next((o for o in outcomes if o[1]["pass"]), None)
    if winner:
        rep.say(f"chosen thrust group: {winner[0]} (first candidate that passes)")
    else:
        best = max(outcomes, key=lambda o: o[1]["passed"])
        rep.say(f"NO candidate passes; best is {best[0]} with {best[1]['passed']}/4 rules "
                f"(ties go to the earlier candidate)")
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
