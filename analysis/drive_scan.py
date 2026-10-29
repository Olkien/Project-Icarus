import argparse
import time
from pathlib import Path

import numpy as np

from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import CACHE_DIR, ConnectomeNetwork
from fly.lif import mean_drive_mV, psp_peak_mV

OUT_PATH = REPO_ROOT / "results" / "drive_scan.txt"
GROUPS = {"LPLC2": "type=LPLC2", "LC4": "type=LC4", "LC10": "type^=LC10"}
RATES_HZ = (20.0, 50.0, 100.0, 200.0)
WEIGHTS_MV = (10.0, 25.0, 50.0)
RUNAWAY_HZ = 50.0
TOP_TYPES = 20
DN = "superclass=descending_neuron"


def load(cache: Path, stepper: str) -> ConnectomeNetwork:
    net = ConnectomeNetwork(cache)
    if stepper == "numba":
        from fly.lif_numba import NumbaLIFStepper

        net.lif = NumbaLIFStepper(net.weights, net.lif_params)
    return net


def drive(net: ConnectomeNetwork, driven: np.ndarray, rate_hz: float, weight_mV: float,
          seconds: float, seed: int) -> tuple[np.ndarray, float]:
    net.lif.input_weight = np.float32(weight_mV)
    rates = np.zeros(net.n, dtype=np.float32)
    rates[driven] = rate_hz
    net.reset(seed)
    net.set_rates(rates)
    counts = np.zeros(net.n, dtype=np.int64)
    steps = round(seconds * 1000.0 / net.dt_ms)
    start = time.perf_counter()
    for _ in range(steps):
        counts[net.step()] += 1
    return counts / seconds, time.perf_counter() - start


def summarize(net: ConnectomeNetwork, rate: np.ndarray, driven: np.ndarray) -> dict:
    dn = net.group(DN)
    side = net.columns["somaSide"]
    fired = rate > 0
    others = np.ones(net.n, dtype=bool)
    others[driven] = False
    return {
        "driven_hz": float(rate[driven].mean()),
        "fired": int(fired.sum()),
        "fired_downstream": int((fired & others).sum()),
        "dn_L": float(rate[dn[side[dn] == "L"]].mean()),
        "dn_R": float(rate[dn[side[dn] == "R"]].mean()),
        "dn_fired": int(fired[dn].sum()),
        "net_hz": float(rate.mean()),
    }


def top_dn_types(net: ConnectomeNetwork, rate: np.ndarray) -> list[str]:
    dn = net.group(DN)
    types = net.columns["type"][dn]
    sides = net.columns["somaSide"][dn]
    rows = []
    for t in sorted(set(types.tolist())):
        mask = types == t
        mean = float(rate[dn[mask]].mean())
        if mean > 0:
            left = rate[dn[mask & (sides == "L")]]
            right = rate[dn[mask & (sides == "R")]]
            rows.append((mean, t, left.mean() if left.size else 0.0,
                         right.mean() if right.size else 0.0))
    rows.sort(reverse=True)
    return [f"      {t:16s} mean {m:7.2f} Hz   L {lv:7.2f}   R {rv:7.2f}"
            for m, t, lv, rv in rows[:TOP_TYPES]]


def scan(cache: Path, groups: dict[str, str], rates: tuple, weights: tuple, seconds: float,
         seed: int, stepper: str, out: Path) -> None:
    rep = Report()
    start = time.perf_counter()
    net = load(cache, stepper)
    rep.say(f"cache {cache.name}: {net.n} neurons; stepper {stepper}; {seconds:g} s per setting")
    rep.say("per-spike input adds the stated mV to g; peak membrane step and mean drive:")
    for w in weights:
        rep.say(f"  {w:g} mV -> peak {psp_peak_mV(w):.2f} mV; mean drive at "
                + ", ".join(f"{r:g} Hz {mean_drive_mV(r, w):.1f} mV" for r in rates))
    rep.say(f"runaway means network mean rate above {RUNAWAY_HZ:g} Hz")
    any_downstream = False
    for name, query in groups.items():
        for side in ("L", "R"):
            driven = net.group(f"{query},somaSide={side}")
            rep.say(f"== {name} {side}: {driven.size} neurons ({query},somaSide={side})")
            rep.say("   rate  w_mV  driven_Hz  fired(share)        downstream  DN_fired"
                    "   DN_L_Hz   DN_R_Hz   net_Hz  wall_s  runaway")
            for weight in weights:
                for rate_hz in rates:
                    rate, wall = drive(net, driven, rate_hz, weight, seconds, seed)
                    s = summarize(net, rate, driven)
                    any_downstream |= s["fired_downstream"] > 0
                    runaway = "YES" if s["net_hz"] > RUNAWAY_HZ else "no"
                    rep.say(f"  {rate_hz:5.0f} {weight:5.0f} {s['driven_hz']:10.2f} "
                            f"{s['fired']:7d} ({100 * s['fired'] / net.n:5.2f}%) "
                            f"{s['fired_downstream']:10d} {s['dn_fired']:9d} "
                            f"{s['dn_L']:9.3f} {s['dn_R']:9.3f} {s['net_hz']:8.3f} "
                            f"{wall:7.1f}  {runaway}")
                    types = top_dn_types(net, rate)
                    if types:
                        rep.say("    top firing DN types:")
                        for line in types:
                            rep.say(line)
    rep.say(f"anything downstream fired: {'yes' if any_downstream else 'NO'}")
    rep.say(f"peak memory {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(CACHE_DIR / "network_A.npz"))
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stepper", choices=("numba", "numpy"), default="numba")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    scan(Path(args.cache), GROUPS, RATES_HZ, WEIGHTS_MV, args.seconds, args.seed, args.stepper,
         Path(args.out))


if __name__ == "__main__":
    main()
