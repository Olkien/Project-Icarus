import argparse
import time
from pathlib import Path

import numpy as np

from analysis.drive_scan import GROUPS, RATES_HZ, RUNAWAY_HZ, WEIGHTS_MV, drive, load
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import CACHE_DIR
from fly.lif import mean_drive_mV, psp_peak_mV

OUT_PATH = REPO_ROOT / "results" / "stability_check.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(CACHE_DIR / "network_A.npz"))
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--rate-hz", type=float, default=max(RATES_HZ))
    parser.add_argument("--weight-mv", type=float, default=max(WEIGHTS_MV))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stepper", choices=("numba", "numpy"), default="numba")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    rep = Report()
    start = time.perf_counter()
    net = load(Path(args.cache), args.stepper)
    driven = np.unique(np.concatenate([net.group(q) for q in GROUPS.values()]))
    rep.say(f"T1 stability: {Path(args.cache).name}, {net.n} neurons, stepper {args.stepper}")
    rep.say(f"inputs {', '.join(GROUPS)} (both sides): {driven.size} neurons at "
            f"{args.rate_hz:g} Hz, {args.weight_mv:g} mV per spike to g "
            f"(peak {psp_peak_mV(args.weight_mv):.2f} mV, mean drive "
            f"{mean_drive_mV(args.rate_hz, args.weight_mv):.1f} mV)")
    rate, wall = drive(net, driven, args.rate_hz, args.weight_mv, args.seconds, args.seed)
    steps = round(args.seconds * 1000.0 / net.dt_ms)
    fired = rate > 0
    rep.say(f"{args.seconds:g} s model time, {steps} steps, {steps / wall:.0f} steps/s, "
            f"wall {wall:.1f} s")
    rep.say(f"network mean rate {rate.mean():.3f} Hz, driven group {rate[driven].mean():.2f} Hz")
    rep.say(f"neurons that fired {int(fired.sum())} of {net.n} ({100 * fired.mean():.2f}%), "
            f"max single neuron {rate.max():.1f} Hz")
    sc = net.columns["superclass"]
    rep.say("by superclass (neurons, share fired, mean Hz):")
    for name in sorted(set(sc.tolist()), key=lambda s: -(sc == s).sum()):
        mask = sc == name
        rep.say(f"  {name or '<missing>':24s} {int(mask.sum()):7d}  "
                f"{100 * fired[mask].mean():6.2f}%  {rate[mask].mean():8.3f}")
    verdict = "FAIL (runaway)" if rate.mean() > RUNAWAY_HZ else "PASS (no runaway)"
    rep.say(f"T1 {verdict}: network mean {rate.mean():.3f} Hz vs limit {RUNAWAY_HZ:g} Hz")
    rep.say(f"peak memory {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.1f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
