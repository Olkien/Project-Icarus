import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from analysis.drive_scan import RATES_HZ, WEIGHTS_MV, drive, load
from analysis.input_off import CASES, RATE_HZ, WEIGHT_MV
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import CACHE_DIR, ConnectomeNetwork
from fly.lif import psp_peak_mV

OUT_PATH = REPO_ROOT / "results" / "wiring_check.txt"
TOP = 20
DEGREE_TOL = 0.1


def describe(net: ConnectomeNetwork, i: int) -> str:
    c = net.columns
    return (f"{c['type'][i] or '-':14s} {c['somaSide'][i] or '-':2s} "
            f"{c['superclass'][i] or '-':18s} {c['consensus_nt'][i] or '-':13s}")


def matched_controls(in_syn: np.ndarray, targets: np.ndarray, excluded: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    pool = np.flatnonzero(~excluded)
    picks = []
    for t in targets:
        lo, hi = in_syn[t] * (1 - DEGREE_TOL), in_syn[t] * (1 + DEGREE_TOL)
        close = pool[(in_syn[pool] >= lo) & (in_syn[pool] <= hi)]
        close = np.setdiff1d(close, np.array(picks, dtype=np.int64))
        if close.size == 0:
            close = pool[np.argsort(np.abs(in_syn[pool] - in_syn[t]))[:5]]
        picks.append(int(rng.choice(close)))
    return np.array(picks)


def strong_edges(rep: Report, net: ConnectomeNetwork) -> None:
    scale = net.lif_params["weight_mV_per_synapse"]
    gap = net.lif_params["v_thresh_mV"] - net.lif_params["v_rest_mV"]
    per_unit = psp_peak_mV(1.0)
    alone = int(np.ceil(gap / per_unit / scale))
    counts = net.counts
    excit = net.sign[np.repeat(np.arange(net.n), np.diff(counts.indptr))] > 0
    strong = (counts.data >= alone) & excit
    targets = np.unique(counts.indices[strong])
    rep.say(f"one presynaptic spike moves the membrane {per_unit * scale:.4f} mV per synapse; "
            f"a single spike crosses the {gap:g} mV gap alone at {alone} synapses or more")
    rep.say(f"excitatory edges with >= {alone} synapses: {int(strong.sum())} of {counts.nnz}; "
            f"neurons receiving one: {targets.size} ({100 * targets.size / net.n:.2f}%)")
    in_exc = np.bincount(counts.indices, weights=counts.data * (net.sign[np.repeat(
        np.arange(net.n), np.diff(counts.indptr))] > 0), minlength=net.n)
    rep.say(f"excitatory synapses received per neuron: median {np.median(in_exc):.0f}, "
            f"90th percentile {np.percentile(in_exc, 90):.0f}, max {in_exc.max():.0f}")


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
    rng = np.random.default_rng(args.seed)
    counts = net.counts
    in_syn = np.asarray(counts.sum(axis=0)).ravel()
    rep.say("orientation: counts row = presynaptic, column = postsynaptic (unit test in "
            "tests/test_connectome.py); LIF weights are the transpose, post by pre")
    strong_edges(rep, net)

    for label, query in CASES.items():
        driven = net.group(query)
        from_driven = np.asarray(counts[driven].sum(axis=0)).ravel()
        partners = np.flatnonzero(from_driven > 0)
        partners = np.setdiff1d(partners, driven)
        excluded = np.zeros(net.n, dtype=bool)
        excluded[driven] = True
        excluded[partners] = True
        rep.say(f"== {label}: {driven.size} driven neurons, {partners.size} direct partners")
        rate, _ = drive(net, driven, RATE_HZ, WEIGHT_MV, 1.0, args.seed)
        top = np.argsort(from_driven)[::-1]
        top = [int(i) for i in top if i not in set(driven.tolist())][:TOP]
        controls = matched_controls(in_syn, np.array(top), excluded, rng)
        rep.say(f"top {TOP} postsynaptic neurons by synapses from the group, rates at "
                f"{RATE_HZ:g} Hz, {WEIGHT_MV:g} mV, 1 s (driven group {rate[driven].mean():.2f} Hz):")
        rep.say("  syn_from_group  in_syn  rate_Hz  type           side superclass         nt")
        for i in top:
            rep.say(f"  {from_driven[i]:14.0f} {in_syn[i]:7.0f} {rate[i]:8.2f}  {describe(net, i)}")
        rep.say(f"  top partners mean {rate[top].mean():.2f} Hz vs matched random non-partners "
                f"{rate[controls].mean():.2f} Hz")
        rep.say("  matched controls:")
        for i in controls.tolist():
            rep.say(f"  {'':14s} {in_syn[i]:7.0f} {rate[i]:8.2f}  {describe(net, i)}")
        others = ~excluded
        rep.say("per setting (scale 1.0, 1 s): rate_Hz w_mV driven_Hz partner_Hz "
                "non_partner_Hz net_Hz")
        for w in WEIGHTS_MV:
            for r in RATES_HZ:
                rate, _ = drive(net, driven, r, w, 1.0, args.seed)
                rep.say(f"  {r:6.0f} {w:5.0f} {rate[driven].mean():9.2f} "
                        f"{rate[partners].mean():10.3f} {rate[others].mean():14.3f} "
                        f"{rate.mean():7.3f}")

    rep.say("== What keeps firing after input off (LC4 R, 100 Hz, 25 mV, scale 1.0)")
    driven = net.group(CASES["LC4 R"])
    rates = np.zeros(net.n, dtype=np.float32)
    rates[driven] = RATE_HZ
    net.lif.input_weight = np.float32(WEIGHT_MV)
    net.reset(args.seed)
    net.set_rates(rates)
    for _ in range(10000):
        net.step()
    net.set_rates(np.zeros(net.n, dtype=np.float32))
    after = np.zeros(net.n, dtype=np.int64)
    for _ in range(10000):
        after[net.step()] += 1
    active = after > 0
    rep.say(f"neurons firing in the second after input off: {int(active.sum())}")
    sc = Counter(net.columns["superclass"][active].tolist())
    rep.say("  by superclass: " + ", ".join(f"{k or '-'} {v}" for k, v in sc.most_common(8)))
    nt = Counter(net.columns["consensus_nt"][active].tolist())
    rep.say("  by transmitter: " + ", ".join(f"{k or '-'} {v}" for k, v in nt.most_common()))
    order = np.argsort(after)[::-1][:TOP]
    rep.say("  top firing neurons:")
    for i in order.tolist():
        rep.say(f"    {after[i]:5d} Hz  {describe(net, i)}")
    rep.say(f"peak memory {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
