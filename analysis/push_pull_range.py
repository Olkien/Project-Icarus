import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analysis.fly_pool import NETWORKS, WINDOW_MS, fly_model, run_all, side_ids
from analysis.push_pull import RATE_MAX_HZ
from analysis.push_pull_thrust import STEER_CHANGE_MAX, STEER_DEG, drive
from analysis.steering_test import (
    BEARINGS_DEG,
    DNA02,
    LC10,
    MAPPING_PATH,
    SEEDS,
    evaluate,
    yaw_signal,
)
from analysis.thrust_group import CANDIDATES, group_ids, per_neuron_decode
from analysis.yaw_zero import SMALL_DEG, SMALL_MIN, small_angle
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import ConnectomeNetwork

OUT_PATH = REPO_ROOT / "results" / "push_pull_range.txt"
C_NEAR_HZ = (20.0, 25.0, 30.0)
C_FAR_HZ = (35.0, 40.0, 45.0)
RANGE_REF_MM = 400.0
RANGES_MM = (20.0, 50.0, 100.0, 200.0, 300.0, 400.0)
RHO_MIN = 0.9


def centre_hz(range_mm: float, c_near: float, c_far: float) -> float:
    return c_near + (c_far - c_near) * min(max(range_mm / RANGE_REF_MM, 0.0), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--group", required=True, choices=list(CANDIDATES))
    parser.add_argument("--a", type=float, required=True)
    parser.add_argument("--steer-c", type=float, required=True)
    parser.add_argument("--c-near", nargs="+", type=float, default=list(C_NEAR_HZ))
    parser.add_argument("--c-far", nargs="+", type=float, default=list(C_FAR_HZ))
    parser.add_argument("--check-steering", action="store_true")
    parser.add_argument("--write-closest", action="store_true")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    scale = fly_model()["weight_scale"]
    net = ConnectomeNetwork(NETWORKS["orig"])
    lc10 = side_ids(net, LC10)
    thrust_ids = group_ids(net, CANDIDATES[args.group])
    dna02 = side_ids(net, DNA02)
    settings = [(lo, hi) for lo in args.c_near for hi in args.c_far if lo < hi]
    bearings = (0, *STEER_DEG)
    needed = {}
    for lo, hi in settings:
        for r in RANGES_MM:
            for d in bearings:
                for s in SEEDS:
                    needed[(round(centre_hz(r, lo, hi), 6), d, s)] = None
    keys = list(needed)
    jobs = [{"kind": "windows", "net": "orig", "scale": scale, "seed": s, "groups": [thrust_ids],
             "drive": drive(lc10, d, c, args.a)} for c, d, s in keys]
    pass_bearings = [s * b for b in BEARINGS_DEG for s in (1, -1)]
    steer_bearings = sorted(set(pass_bearings) | {s * b for b in SMALL_DEG for s in (1, -1)})
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    used = sorted({round(centre_hz(r, lo, hi), 6) for lo, hi in settings for r in RANGES_MM})
    steer_keys = [(c, b, s) for c in used for b in steer_bearings for s in SEEDS]
    if args.check_steering:
        jobs += [{"kind": "windows", "net": "orig", "scale": scale, "seed": s, "groups": [dna02["L"], dna02["R"]],
                  "drive": drive(lc10, b, c, args.a)} for c, b, s in steer_keys]
    start = time.perf_counter()
    results = run_all(jobs, args.workers)
    decoded = {k: float(per_neuron_decode(w, [thrust_ids.size])[0])
               for k, w in zip(keys, results)}
    steer_raw = {k: yaw_signal(w) for k, w in zip(steer_keys, results[len(keys):])}
    rep = Report()
    rep.say(f"push-pull range mapping: weight scale {scale:g}, spike input, 1 s, seeds 0-19, "
            f"thrust group {args.group} ({thrust_ids.size} neurons), a {args.a:g} Hz, "
            f"{len(jobs)} distinct runs")
    rep.say(f"centre c = c_near + (c_far - c_near) clip(range / {RANGE_REF_MM:g}, 0, 1); LC10 L = c + "
            f"a b/pi, LC10 R = c - a b/pi, clipped to 0-{RATE_MAX_HZ:g} Hz")
    rep.say(f"rule: Spearman(range, decode at bearing 0) >= {RHO_MIN:g}, and bearing +-90 changes "
            f"the decode by under {100 * STEER_CHANGE_MAX:g}% at every range; pick by largest span "
            f"= decode at {RANGES_MM[-1]:g} mm minus decode at {RANGES_MM[0]:g} mm")
    steer_ok = {}
    if args.check_steering:
        rep.say(f"steering check at every c the mapping uses, a {args.a:g} Hz, yaw_zero interpolated in c: "
                f"sign >= 95%, Spearman >= 0.8, pooled small-angle >= {100 * SMALL_MIN:g}%")
        for c in used:
            zero = float(np.interp(c, mapping["yaw_zero"]["c_hz"], mapping["yaw_zero"]["yaw"]))
            yaws = {b: [steer_raw[(c, b, s)] - zero for s in SEEDS] for b in steer_bearings}
            ev = evaluate(yaws, pass_bearings)
            small, per_level = small_angle(yaws)
            steer_ok[c] = ev["pass"] and small >= SMALL_MIN
            rep.say(f"  c {c:g} Hz: {'PASS' if steer_ok[c] else 'fail'}  zero {zero:.4f}  sign "
                    f"{100 * ev['sign']:.1f}%  rho {ev['rho']:.3f}  small {100 * small:.1f}% ("
                    + ", ".join(f"{lv} deg {100 * v:.1f}%" for lv, v in per_level.items())
                    + f")  |yaw| at 180 {ev['mean_abs'][-1]:.4f}")
    outcomes = []
    for lo, hi in settings:
        mean = {(r, d): float(np.mean([decoded[(round(centre_hz(r, lo, hi), 6), d, s)]
                                       for s in SEEDS])) for r in RANGES_MM for d in bearings}
        curve = [mean[(r, 0)] for r in RANGES_MM]
        rho = float(spearmanr(RANGES_MM, curve).statistic)
        rho = 0.0 if math.isnan(rho) else rho
        changes = {r: max(abs(mean[(r, d)] - mean[(r, 0)]) / mean[(r, 0)] if mean[(r, 0)] > 0
                          else math.inf for d in STEER_DEG) for r in RANGES_MM}
        worst = max(changes.values())
        span = curve[-1] - curve[0]
        steers = all(steer_ok[round(centre_hz(r, lo, hi), 6)] for r in RANGES_MM) if steer_ok else True
        ok = rho >= RHO_MIN and worst < STEER_CHANGE_MAX and steers
        outcomes.append(((lo, hi), ok, rho, worst, span, curve))
        rep.say(f"== c_near {lo:g} Hz, c_far {hi:g} Hz: {'PASS' if ok else 'fail'}  rho {rho:.3f}  "
                f"worst coupling {100 * worst:.1f}%  span {span:.4f}  "
                f"steering {'ok' if steers else 'no'}")
        rep.say("  range_mm  c_Hz   decode b0  Hz/neuron   +90 deg   -90 deg  coupling")
        for r in RANGES_MM:
            rep.say(f"  {r:8g} {centre_hz(r, lo, hi):6.2f} {mean[(r, 0)]:10.4f} "
                    f"{mean[(r, 0)] * 1000.0 / WINDOW_MS:10.2f} {mean[(r, 90)]:9.4f} "
                    f"{mean[(r, -90)]:9.4f} {100 * changes[r]:8.1f}%")
    passing = [o for o in outcomes if o[1]]
    pool = passing or (outcomes if args.write_closest else [])
    if pool:
        (lo, hi), _, rho, worst, span, curve = max(pool, key=lambda o: o[4])
        rep.say(f"{'chosen' if passing else 'NO setting passes; closest'}: c_near {lo:g} Hz, c_far "
                f"{hi:g} Hz (rho {rho:.3f}, worst coupling {100 * worst:.1f}%, span {span:.4f})")
        mapping["steering"].update(c_hz=args.steer_c, a_hz=args.a)
        mapping["thrust"] = {
            "group": args.group, "queries": CANDIDATES[args.group],
            "decode": f"spikes per {WINDOW_MS:g} ms window per neuron",
            "c_near_hz": lo, "c_far_hz": hi, "range_ref_mm": RANGE_REF_MM,
            "centre": "c_near + (c_far - c_near) * clip(range / range_ref_mm, 0, 1)",
            "decode_near": round(curve[0], 4), "decode_far": round(curve[-1], 4),
            "chosen_by": f"range test at bearing 0, rho {rho:.3f}, worst coupling "
                         f"{100 * worst:.1f}%, span {span:.4f}",
            "rule": "pass" if passing else "fail, closest setting, kept to fly",
        }
        mapping["chosen_by"] = (f"push-pull steering c {args.steer_c:g}, a {args.a:g}; thrust "
                                f"{args.group} from the push-pull thrust test")
        MAPPING_PATH.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8",
                                newline="\n")
        rep.say(f"wrote push-pull steering and thrust to configs/{MAPPING_PATH.name}")
    else:
        best = max(outcomes, key=lambda o: (o[2] >= RHO_MIN, -o[3], o[4]))
        rep.say(f"NO setting passes; closest is c_near {best[0][0]:g}, c_far {best[0][1]:g} Hz "
                f"(rho {best[2]:.3f}, worst coupling {100 * best[3]:.1f}%); mapping not changed")
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
