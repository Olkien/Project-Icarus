import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from fly.connectome import CACHE_DIR, ConnectomeNetwork

DEFAULT_CACHE = CACHE_DIR / "network_A.npz"


def build_query(args) -> str:
    terms = []
    if args.type:
        terms.append(f"type={args.type}")
    if args.like:
        terms.append(f"type^={args.like}")
    if args.cls:
        terms.append(f"class={args.cls}")
    if args.superclass:
        terms.append(f"superclass={args.superclass}")
    if args.side:
        terms.append(f"somaSide={args.side}")
    if not terms:
        raise SystemExit("give at least one of --type, --like, --class, --superclass")
    return ",".join(terms)


def stats(values: np.ndarray) -> str:
    if values.size == 0:
        return "-"
    return (f"min {values.min():.0f}  median {np.median(values):.0f}  "
            f"mean {values.mean():.1f}  max {values.max():.0f}")


def describe(net: ConnectomeNetwork, query: str, max_ids: int) -> list[str]:
    idx = net.query(query)
    lines = [f"query {query}: {idx.size} neurons"]
    if idx.size == 0:
        return lines
    sides = net.columns["somaSide"][idx]
    nts = net.columns["consensus_nt"][idx]
    side_names = sorted({s or "-" for s in sides.tolist()})
    nt_names = [n for n, _ in Counter(nts.tolist()).most_common()]
    pairs = Counter(zip([s or "-" for s in sides.tolist()], nts.tolist()))
    header = "side   " + "".join(f"{(n or '-')[:13]:>14s}" for n in nt_names) + "     total"
    lines.append(header)
    for side in side_names:
        row = [pairs[(side, n)] for n in nt_names]
        lines.append(f"{side:6s} " + "".join(f"{c:14d}" for c in row) + f"{sum(row):10d}")
    types = Counter(net.columns["type"][idx].tolist())
    if len(types) > 1:
        lines.append("types: " + ", ".join(f"{t or '-'} {c}" for t, c in types.most_common(12)))
    counts = net.counts
    out_edges = np.diff(counts.indptr)[idx]
    in_edges = np.bincount(counts.indices, minlength=net.n)[idx]
    out_syn = np.asarray(counts.sum(axis=1)).ravel()[idx]
    in_syn = np.asarray(counts.sum(axis=0)).ravel()[idx]
    lines.append(f"out degree (partners): {stats(out_edges)}")
    lines.append(f"in degree (partners):  {stats(in_edges)}")
    lines.append(f"out synapses: {stats(out_syn)}")
    lines.append(f"in synapses:  {stats(in_syn)}")
    shown = idx[:max_ids]
    lines.append(f"body ids ({shown.size} of {idx.size}):")
    for i in shown.tolist():
        lines.append(f"  {int(net.body_ids[i])}  index {i}  type {net.columns['type'][i] or '-'}  "
                     f"side {net.columns['somaSide'][i] or '-'}  "
                     f"nt {net.columns['consensus_nt'][i] or '-'}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--type")
    parser.add_argument("--like", help="type starts with this")
    parser.add_argument("--class", dest="cls")
    parser.add_argument("--superclass")
    parser.add_argument("--side", help="somaSide L, R or M")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--max-ids", type=int, default=50)
    args = parser.parse_args()
    net = ConnectomeNetwork(Path(args.cache))
    for line in describe(net, build_query(args), args.max_ids):
        print(line)


if __name__ == "__main__":
    main()
