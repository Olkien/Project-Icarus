import argparse
import time
from pathlib import Path

import numpy as np

from analysis.fly_pool import NETWORKS, fly_model, run_all, side_ids
from analysis.push_pull_thrust import drive
from analysis.steering_test import BEARINGS_DEG, DNA02, LC10, SEEDS, evaluate, yaw_signal
from analysis.yaw_zero import SMALL_DEG, small_angle
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import ConnectomeNetwork

OUT_PATH = REPO_ROOT / "results" / "alt_yaw.txt"
CENTRES_HZ = (35.0, 40.0)
A_HZ = 25.0


def line(name: str, yaws: dict, pass_bearings: list[int]) -> str:
    ev = evaluate(yaws, pass_bearings)
    small, per_level = small_angle(yaws)
    return (f"  {name}: sign {100 * ev['sign']:.1f}%  rho {ev['rho']:.3f}  small {100 * small:.1f}% ("
            + ", ".join(f"{lv} deg {100 * v:.1f}%" for lv, v in per_level.items()) + ")  "
            + ("PASS" if ev["pass"] and small >= 0.8 else "fail"))


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
    keys = [(c, b, s) for c in CENTRES_HZ for b in (0, *bearings) for s in SEEDS]
    jobs = [{"kind": "windows", "net": "orig", "scale": scale, "seed": s,
             "drive": drive(lc10, b, c, A_HZ), "groups": [dna02["L"], dna02["R"]]} for c, b, s in keys]
    start = time.perf_counter()
    windows = dict(zip(keys, run_all(jobs, args.workers)))
    rep = Report()
    rep.say(f"alternative yaw decode: weight scale {scale:g}, spike input, 1 s, seeds 0-19, a {A_HZ:g} Hz, "
            f"{len(jobs)} runs; controller not changed")
    rep.say("old: mean over windows of (L - R)/(L + R + 1); new: mean over windows of (L - R) / mean "
            "total spikes per window at bearing 0 for that c; zero-point subtracted from both")
    for c in CENTRES_HZ:
        base = [windows[(c, 0, s)] for s in SEEDS]
        total = float(np.mean([w.sum(axis=1).mean() for w in base]))
        new_raw = {k: float((w[:, 0] - w[:, 1]).mean()) / total for k, w in windows.items() if k[0] == c}
        old_raw = {k: yaw_signal(w) for k, w in windows.items() if k[0] == c}
        new_zero = float(np.mean([new_raw[(c, 0, s)] for s in SEEDS]))
        old_zero = float(np.mean([old_raw[(c, 0, s)] for s in SEEDS]))
        new = {b: [new_raw[(c, b, s)] - new_zero for s in SEEDS] for b in bearings}
        old = {b: [old_raw[(c, b, s)] - old_zero for s in SEEDS] for b in bearings}
        rep.say(f"== c {c:g} Hz, a {A_HZ:g} Hz: mean total at bearing 0 {total:.4f} spikes per window; "
                f"zero old {old_zero:.4f}, new {new_zero:.4f}")
        rep.say(line("old", old, pass_bearings))
        rep.say(line("new", new, pass_bearings))
        rep.say("  bearing_deg  old_yaw     sd  new_yaw     sd")
        for b in bearings:
            rep.say(f"  {b:11d} {np.mean(old[b]):8.4f} {np.std(old[b], ddof=1):6.4f} "
                    f"{np.mean(new[b]):8.4f} {np.std(new[b], ddof=1):6.4f}")
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
