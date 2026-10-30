import argparse
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analysis.fly_pool import NETWORKS, WINDOW_MS, fly_model, run_all, side_ids
from analysis.push_pull_thrust import (
    CV_MAX,
    MAX_HZ,
    RISE_RHO_MIN,
    STEER_CHANGE_MAX,
    STEER_DEG,
    drive,
)
from analysis.steering_test import LC10, SEEDS
from analysis.thrust_group import CANDIDATES, group_ids, per_neuron_decode
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import ConnectomeNetwork

OUT_PATH = REPO_ROOT / "results" / "band_thrust.txt"
ORDER = ("DNp31", "DLMn + DVMn")
RATIO_INFO_MIN = 1.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--band", nargs="+", type=float, required=True)
    parser.add_argument("--a", type=float, required=True)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    scale = fly_model()["weight_scale"]
    net = ConnectomeNetwork(NETWORKS["orig"])
    lc10 = side_ids(net, LC10)
    groups = [group_ids(net, CANDIDATES[name]) for name in ORDER]
    sizes = [g.size for g in groups]
    conditions = [(c, d) for c in args.band for d in (0, *STEER_DEG)]
    jobs = [{"kind": "windows", "net": "orig", "scale": scale, "seed": s, "groups": groups,
             "drive": drive(lc10, d, c, args.a)} for c, d in conditions for s in SEEDS]
    start = time.perf_counter()
    results = run_all(jobs, args.workers)
    decoded = {}
    for k, cond in enumerate(conditions):
        runs = results[k * len(SEEDS):(k + 1) * len(SEEDS)]
        decoded[cond] = np.array([per_neuron_decode(w, sizes) for w in runs])
    bottom, top = args.band[0], args.band[-1]
    rep = Report()
    rep.say(f"band thrust test: weight scale {scale:g}, spike input, 1 s, seeds 0-19, a {args.a:g} Hz, "
            f"band {', '.join(f'{c:g}' for c in args.band)} Hz, {len(jobs)} runs")
    rep.say(f"rules: Spearman(c, decode) >= {RISE_RHO_MIN:g}; under {MAX_HZ:g} Hz at c {top:g}; CV under "
            f"{CV_MAX:g} at c {bottom:g}; coupling under {100 * STEER_CHANGE_MAX:g}% at every c in the band; "
            f"ratio >= {RATIO_INFO_MIN:g} is information only")
    for j, name in enumerate(ORDER):
        means = {c: float(np.mean(decoded[(c, 0)][:, j])) for c in args.band}
        rep.say(f"== {name} ({sizes[j]} neurons)")
        rep.say("  c_Hz  decode  Hz/neuron     CV   +90 deg   -90 deg  coupling")
        worst = 0.0
        cvs = {}
        for c in args.band:
            base = means[c]
            cvs[c] = float(np.std(decoded[(c, 0)][:, j], ddof=1)) / base if base > 0 else math.inf
            steer = {d: float(np.mean(decoded[(c, d)][:, j])) for d in STEER_DEG}
            change = max(abs(v - base) / base if base > 0 else math.inf for v in steer.values())
            worst = max(worst, change)
            rep.say(f"  {c:4g} {base:7.4f} {base * 1000.0 / WINDOW_MS:10.2f} {cvs[c]:6.3f} "
                    f"{steer[90]:9.4f} {steer[-90]:9.4f} {100 * change:8.1f}%")
        rho = float(spearmanr(args.band, list(means.values())).statistic)
        hz_top = means[top] * 1000.0 / WINDOW_MS
        ratio = means[top] / means[bottom] if means[bottom] > 0 else math.inf
        rules = {"rise": rho >= RISE_RHO_MIN, "max_hz": hz_top < MAX_HZ, "cv": cvs[bottom] < CV_MAX,
                 "coupling": worst < STEER_CHANGE_MAX}
        rep.say(f"  rise rho {rho:.3f} {'ok' if rules['rise'] else 'no'}; rate at c {top:g} "
                f"{hz_top:.2f} Hz {'ok' if rules['max_hz'] else 'no'}; CV at c {bottom:g} "
                f"{cvs[bottom]:.3f} {'ok' if rules['cv'] else 'no'}; worst coupling {100 * worst:.1f}% "
                f"{'ok' if rules['coupling'] else 'no'}; {'PASS' if all(rules.values()) else 'fail'} "
                f"({sum(rules.values())}/4)")
        rep.say(f"  decode ratio c {top:g} / c {bottom:g} = {ratio:.3f}; at least {RATIO_INFO_MIN:g}: "
                f"{'yes' if ratio >= RATIO_INFO_MIN else 'no'}")
        if all(rules.values()):
            rep.say(f"chosen: {name}")
            break
    else:
        rep.say("NO group passes")
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
