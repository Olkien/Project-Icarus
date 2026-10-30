import argparse
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analysis.fly_pool import NETWORKS, WINDOW_MS, fly_model, run_all, side_ids
from analysis.push_pull import push_pull_rates
from analysis.steering_test import LC10, SEEDS
from analysis.thrust_group import CANDIDATES, group_ids, per_neuron_decode
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import ConnectomeNetwork

OUT_PATH = REPO_ROOT / "results" / "push_pull_thrust.txt"
ORDER = ("DLMn + DVMn", "DNp31", "DNg33")
CENTRES_HZ = (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0)
RISE_RHO_MIN = 0.9
MAX_HZ = 200.0
MAX_HZ_AT = 30.0
CV_MAX = 0.3
CV_AT = 20.0
STEER_DEG = (90, -90)
STEER_CHANGE_MAX = 0.25
RATIO_HI, RATIO_LO = 40.0, 25.0


def setting_arg(text: str) -> tuple[float, float]:
    c, a = text.split(",")
    return float(c), float(a)


def drive(lc10: dict, bearing_deg: float, c: float, a: float) -> list[tuple[np.ndarray, float]]:
    rl, rr = push_pull_rates(math.radians(bearing_deg), c, a)
    return [(lc10["L"], rl), (lc10["R"], rr)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--settings", nargs="*", type=setting_arg, default=[],
                        help="push-pull settings that passed the steering test, as c,a")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    scale = fly_model()["weight_scale"]
    net = ConnectomeNetwork(NETWORKS["orig"])
    lc10 = side_ids(net, LC10)
    groups = [group_ids(net, CANDIDATES[name]) for name in ORDER]
    sizes = [g.size for g in groups]
    conditions = [(c, 0.0, 0) for c in CENTRES_HZ]
    conditions += [(c, a, d) for c, a in args.settings for d in STEER_DEG]
    jobs = [{"kind": "windows", "net": "orig", "scale": scale, "seed": s, "groups": groups,
             "drive": drive(lc10, d, c, a)} for c, a, d in conditions for s in SEEDS]
    start = time.perf_counter()
    results = run_all(jobs, args.workers)
    decoded = {}
    for k, cond in enumerate(conditions):
        runs = results[k * len(SEEDS):(k + 1) * len(SEEDS)]
        decoded[cond] = np.array([per_neuron_decode(w, sizes) for w in runs])
    rep = Report()
    rep.say(f"push-pull thrust test: weight scale {scale:g}, spike input, 1 s, seeds 0-19, "
            f"{len(jobs)} runs")
    rep.say(f"decode = spikes per {WINDOW_MS:g} ms window per neuron (group pooled), mean over windows")
    rep.say(f"rules: Spearman(c, decode at bearing 0) >= {RISE_RHO_MIN:g}; per-neuron rate under "
            f"{MAX_HZ:g} Hz at c = {MAX_HZ_AT:g}; CV across seeds under {CV_MAX:g} at c = {CV_AT:g}; "
            f"at the setting, bearing +-90 changes the decode by under {100 * STEER_CHANGE_MAX:g}% "
            f"against bearing 0 at the same c")
    rep.say("settings from the steering test: " + (", ".join(f"({c:g}, {a:g})" for c, a in
                                                            args.settings) or "none"))
    winner = None
    best = None
    for j, name in enumerate(ORDER):
        curve = {c: decoded[(c, 0.0, 0)][:, j] for c in CENTRES_HZ}
        means = [float(np.mean(curve[c])) for c in CENTRES_HZ]
        rho = float(spearmanr(CENTRES_HZ, means).statistic)
        rho = 0.0 if math.isnan(rho) else rho
        hz_at = float(np.mean(curve[MAX_HZ_AT])) * 1000.0 / WINDOW_MS
        m20 = float(np.mean(curve[CV_AT]))
        cv = float(np.std(curve[CV_AT], ddof=1)) / m20 if m20 > 0 else math.inf
        lo = float(np.mean(curve[RATIO_LO]))
        ratio = float(np.mean(curve[RATIO_HI])) / lo if lo > 0 else math.inf
        base_rules = {"rise": rho >= RISE_RHO_MIN, "max_hz": hz_at < MAX_HZ, "cv": cv < CV_MAX}
        rep.say(f"== {name} ({sizes[j]} neurons)")
        rep.say("  c_Hz  decode/window  sd      Hz/neuron")
        for c in CENTRES_HZ:
            rep.say(f"  {c:4g} {np.mean(curve[c]):14.4f} {np.std(curve[c], ddof=1):7.4f} "
                    f"{np.mean(curve[c]) * 1000.0 / WINDOW_MS:9.2f}")
        rep.say(f"  rise rho {rho:.3f} {'ok' if base_rules['rise'] else 'no'}; rate at c = "
                f"{MAX_HZ_AT:g} {hz_at:.2f} Hz {'ok' if base_rules['max_hz'] else 'no'}; CV at c = "
                f"{CV_AT:g} {cv:.3f} {'ok' if base_rules['cv'] else 'no'}; decode ratio c "
                f"{RATIO_HI:g} / c {RATIO_LO:g} = {ratio:.3f}")
        for c, a in args.settings:
            base = float(np.mean(curve[c]))
            changes = {d: abs(float(np.mean(decoded[(c, a, d)][:, j])) - base) / base
                       if base > 0 else math.inf for d in STEER_DEG}
            steer_ok = max(changes.values()) < STEER_CHANGE_MAX
            n_ok = sum(base_rules.values()) + steer_ok
            ok = n_ok == 4
            rep.say(f"  setting c {c:g}, a {a:g}: bearing 0 {base:.4f}; " + "; ".join(
                f"{d:+d} deg {np.mean(decoded[(c, a, d)][:, j]):.4f} ({100 * changes[d]:.1f}%)"
                for d in STEER_DEG) + f"; steer {'ok' if steer_ok else 'no'}; "
                f"{'PASS' if ok else 'fail'} ({n_ok}/4)")
            if ok and winner is None:
                winner = (name, c, a)
            if best is None or n_ok > best[0]:
                best = (n_ok, name, c, a, max(changes.values()))
    if winner:
        rep.say(f"chosen: {winner[0]} with c {winner[1]:g}, a {winner[2]:g} (first candidate and "
                f"setting that pass all four)")
    elif best:
        rep.say(f"NO candidate and setting pass; best is {best[1]} at c {best[2]:g}, a {best[3]:g} "
                f"with {best[0]}/4 rules (largest steer change {100 * best[4]:.1f}%); Task 3 stops")
    else:
        rep.say("NO steering setting passed, so no setting to test; Task 3 stops")
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
