import argparse
import json
import time
from pathlib import Path

import numpy as np

from analysis.drive_scan import GROUPS, load
from env.params import CONFIG_DIR
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import CACHE_DIR

OUT_PATH = REPO_ROOT / "results" / "stability_scan.txt"
FLY_MODEL_PATH = CONFIG_DIR / "fly_model.json"
SCALES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
RATE_HZ = 100.0
ON_S = 1.0
OFF_S = 1.0
BIN_MS = 50.0
SETTLE_S = 0.2
OFF_LIMIT_HZ = 0.5
DRIVE_LIMIT_HZ = 20.0
CEILING_SHARE_LIMIT = 0.005
CEILING_FRACTION = 0.9
RULE = (f"stable = after input stops, every {BIN_MS:g} ms bin from {SETTLE_S:g} s on is under "
        f"{OFF_LIMIT_HZ:g} Hz network mean; every drive bin is under {DRIVE_LIMIT_HZ:g} Hz; "
        f"under {100 * CEILING_SHARE_LIMIT:g}% of neurons fire at >= {CEILING_FRACTION:g} x the "
        f"refractory ceiling during drive")


def input_sets(net) -> dict[str, np.ndarray]:
    return {f"{name} {side}": net.group(f"{query},somaSide={side}")
            for name, query in GROUPS.items() for side in ("L", "R")}


def ceiling_hz(net) -> float:
    p = net.lif_params
    return 1000.0 / (p["t_ref_ms"] + p["dt_ms"])


def stability_run(net, driven: np.ndarray, rate_hz: float, seed: int) -> dict:
    rates = np.zeros(net.n, dtype=np.float32)
    rates[driven] = rate_hz
    net.reset(seed)
    net.set_spike_rates(rates)
    bin_steps = round(BIN_MS / net.dt_ms)
    on_bins = round(ON_S * 1000.0 / BIN_MS)
    off_bins = round(OFF_S * 1000.0 / BIN_MS)
    bin_s = BIN_MS / 1000.0
    drive_counts = np.zeros(net.n, dtype=np.int64)
    drive_bins, off_rates = [], []
    for b in range(on_bins + off_bins):
        if b == on_bins:
            net.set_spike_rates(np.zeros(net.n, dtype=np.float32))
        total = 0
        for _ in range(bin_steps):
            spikes = net.step()
            total += spikes.size
            if b < on_bins:
                drive_counts[spikes] += 1
        rate = total / net.n / bin_s
        (drive_bins if b < on_bins else off_rates).append(rate)
    settle = round(SETTLE_S * 1000.0 / BIN_MS)
    neuron_rate = drive_counts / ON_S
    at_ceiling = float((neuron_rate >= CEILING_FRACTION * ceiling_hz(net)).mean())
    off_after = max(off_rates[settle:])
    below = next((i for i, r in enumerate(off_rates) if all(x < OFF_LIMIT_HZ
                                                           for x in off_rates[i:])), None)
    out = {
        "drive_mean_hz": float(np.mean(drive_bins)), "drive_max_hz": float(max(drive_bins)),
        "ceiling_share": at_ceiling, "off_max_after_hz": float(off_after),
        "settle_s": None if below is None else below * bin_s,
        "driven_hz": float(neuron_rate[driven].mean()),
    }
    out["pass"] = (off_after < OFF_LIMIT_HZ and out["drive_max_hz"] < DRIVE_LIMIT_HZ
                   and at_ceiling < CEILING_SHARE_LIMIT)
    return out


def row(scale: float, name: str, r: dict) -> str:
    settle = "never" if r["settle_s"] is None else f"{r['settle_s']:.2f}"
    return (f"  {scale:4.1f}  {name:8s} {'PASS' if r['pass'] else 'fail'}  "
            f"{r['drive_mean_hz']:9.3f} {r['drive_max_hz']:9.3f} {100 * r['ceiling_share']:8.3f}%"
            f" {r['off_max_after_hz']:10.3f} {settle:>8s} {r['driven_hz']:8.1f}")


HEADER = ("  scale input    pass   drive_Hz  drive_max  ceiling%  off_max_Hz  quiet_s "
          "driven_Hz")


def scan_scales(rep: Report, net, inputs: dict[str, np.ndarray], scales, seed: int) -> dict:
    results = {}
    rep.say(HEADER)
    for scale in scales:
        net.set_weight_scale(scale)
        results[scale] = {}
        for name, driven in inputs.items():
            r = stability_run(net, driven, RATE_HZ, seed)
            results[scale][name] = r
            rep.say(row(scale, name, r))
    return results


def largest_passing(results: dict) -> float | None:
    passing = [s for s, by_input in results.items() if all(r["pass"] for r in by_input.values())]
    return max(passing) if passing else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(CACHE_DIR / "network_A.npz"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stepper", choices=("numba", "numpy"), default="numba")
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--no-write", action="store_true", help="do not write fly_model.json")
    args = parser.parse_args()
    rep = Report()
    start = time.perf_counter()
    net = load(Path(args.cache), args.stepper)
    rep.say(f"stability scan: {Path(args.cache).name}, spike input {RATE_HZ:g} Hz, "
            f"on {ON_S:g} s, off {OFF_S:g} s, Poisson seed {args.seed}")
    rep.say(RULE + f" (ceiling {ceiling_hz(net):.0f} Hz)")
    results = scan_scales(rep, net, input_sets(net), SCALES, args.seed)
    chosen = largest_passing(results)
    if chosen is None:
        best = max(results, key=lambda s: sum(r["pass"] for r in results[s].values()))
        rep.say(f"NO scale passes for all six inputs; closest is {best:g} "
                f"({sum(r['pass'] for r in results[best].values())} of 6 pass)")
    else:
        rep.say(f"chosen weight_scale {chosen:g} (largest scale where all six inputs pass)")
        if not args.no_write:
            model = {"weight_scale": chosen, "input_mode": "spike", "stability_rule": RULE,
                     "stability_rate_hz": RATE_HZ, "cache": Path(args.cache).name}
            FLY_MODEL_PATH.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8",
                                      newline="\n")
            rep.say(f"wrote configs/{FLY_MODEL_PATH.name}")
    rep.say(f"peak memory {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
