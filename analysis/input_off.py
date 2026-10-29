import argparse
import time
from pathlib import Path

import numpy as np

from analysis.drive_scan import DN, load
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import CACHE_DIR, ConnectomeNetwork

OUT_PATH = REPO_ROOT / "results" / "input_off.txt"
CASES = {"LC4 R": "type=LC4,somaSide=R", "LPLC2 L": "type=LPLC2,somaSide=L"}
RATE_HZ = 100.0
WEIGHT_MV = 25.0
ON_S = 1.0
OFF_S = 2.0
BIN_MS = 100.0


def binned_run(net: ConnectomeNetwork, driven: np.ndarray, rate_hz: float, weight_mV: float,
               on_s: float, off_s: float, bin_ms: float, seed: int) -> list[dict]:
    net.lif.input_weight = np.float32(weight_mV)
    rates = np.zeros(net.n, dtype=np.float32)
    rates[driven] = rate_hz
    net.reset(seed)
    net.set_rates(rates)
    dn = net.group(DN)
    side = net.columns["somaSide"]
    dn_l, dn_r = dn[side[dn] == "L"], dn[side[dn] == "R"]
    bin_steps = round(bin_ms / net.dt_ms)
    on_bins = round(on_s * 1000.0 / bin_ms)
    total_bins = on_bins + round(off_s * 1000.0 / bin_ms)
    bin_s = bin_ms / 1000.0
    out = []
    for b in range(total_bins):
        if b == on_bins:
            net.set_rates(np.zeros(net.n, dtype=np.float32))
        counts = np.zeros(net.n, dtype=np.int64)
        for _ in range(bin_steps):
            counts[net.step()] += 1
        out.append({
            "t0": b * bin_s, "input": b < on_bins,
            "net_hz": counts.sum() / net.n / bin_s,
            "share": float((counts > 0).mean()),
            "spikes": int(counts.sum()),
            "driven_hz": counts[driven].sum() / max(1, driven.size) / bin_s,
            "dn_l_hz": counts[dn_l].sum() / max(1, dn_l.size) / bin_s,
            "dn_r_hz": counts[dn_r].sum() / max(1, dn_r.size) / bin_s,
        })
    return out


def died_out(bins: list[dict]) -> str:
    off = [b for b in bins if not b["input"]]
    for i, b in enumerate(off):
        if all(x["spikes"] == 0 for x in off[i:]):
            return f"activity dies out: silent from {b['t0'] - off[0]['t0']:.1f} s after input off"
    return f"activity does NOT die out: last bin still has {off[-1]['spikes']} spikes"


def report_case(rep: Report, label: str, bins: list[dict], extra: bool) -> None:
    head = "  t_s    input  net_Hz   share_firing"
    if extra:
        head += "  driven_Hz  DN_L_Hz  DN_R_Hz"
    rep.say(head)
    for b in bins:
        row = (f"  {b['t0']:4.1f}  {'on ' if b['input'] else 'off'}  {b['net_hz']:8.3f}"
               f"   {100 * b['share']:7.3f}%")
        if extra:
            row += f"  {b['driven_hz']:9.2f}  {b['dn_l_hz']:7.2f}  {b['dn_r_hz']:7.2f}"
        rep.say(row)
    rep.say(f"  {label}: {died_out(bins)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(CACHE_DIR / "network_A.npz"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stepper", choices=("numba", "numpy"), default="numba")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    rep = Report()
    start = time.perf_counter()
    net = load(Path(args.cache), args.stepper)
    rep.say(f"input off test: {RATE_HZ:g} Hz, {WEIGHT_MV:g} mV per spike to g, on {ON_S:g} s, "
            f"off {OFF_S:g} s, {BIN_MS:g} ms bins")
    for label, query in CASES.items():
        driven = net.group(query)
        rep.say(f"== {label}: {driven.size} neurons ({query})")
        bins = binned_run(net, driven, RATE_HZ, WEIGHT_MV, ON_S, OFF_S, BIN_MS, args.seed)
        report_case(rep, label, bins, extra=True)
    rep.say(f"peak memory {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
