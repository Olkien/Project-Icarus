import argparse
import sys
import time
from pathlib import Path

import numpy as np

from fly.download_data import DATA_DIR, FILES

EXPECTED_NEURONS = 166_700
EXPECTED_EDGES = 25_600_000
EDGE_RANGE = (0.8, 1.2)
NEURON_RANGE = (0.95, 1.05)
NT_SIGN = {
    "acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
    "dopamine": 0, "serotonin": 0, "octopamine": 0,
}


def read_feather(path: Path, columns: list[str] | None = None):
    try:
        from pyarrow import feather
    except ImportError:
        sys.exit("needs pyarrow: pip install pyarrow")
    return feather.read_table(path, columns=columns)


def feather_columns(path: Path) -> list[str]:
    from pyarrow import ipc

    with ipc.open_file(path) as reader:
        return reader.schema.names


def need(table, names: list[str], label: str) -> None:
    missing = [c for c in names if c not in table.column_names]
    if missing:
        sys.exit(f"{label}: missing columns {missing}, has {table.column_names}")


def report(ok: bool, text: str) -> bool:
    print(("PASS " if ok else "FAIL ") + text)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--body-col", default="bodyId")
    parser.add_argument("--nt-col", default="consensus_nt")
    parser.add_argument("--pre-col", default="body_pre")
    parser.add_argument("--post-col", default="body_post")
    parser.add_argument("--weight-col", default="weight")
    args = parser.parse_args()
    data = Path(args.data)
    ann_path, nt_path, w_path = (data / name for name in FILES)
    results = []

    t0 = time.perf_counter()
    ann = read_feather(ann_path)
    print(f"annotations: {ann.num_rows} rows, columns {ann.column_names}")
    nt = read_feather(nt_path)
    print(f"neurotransmitters: {nt.num_rows} rows, columns {nt.column_names}")
    need(nt, [args.body_col, args.nt_col], "neurotransmitters")

    print(f"weights columns {feather_columns(w_path)}")
    weights = read_feather(w_path, columns=[args.pre_col, args.post_col, args.weight_col])
    print(f"read in {time.perf_counter() - t0:.1f} s")

    pre = weights[args.pre_col].to_numpy()
    post = weights[args.post_col].to_numpy()
    syn = weights[args.weight_col].to_numpy()
    edges = len(syn)
    neurons = np.union1d(np.unique(pre), np.unique(post)).size

    lo, hi = NEURON_RANGE
    results.append(report(lo * EXPECTED_NEURONS <= neurons <= hi * EXPECTED_NEURONS,
                          f"neurons in edges: {neurons} (expected near {EXPECTED_NEURONS})"))
    lo, hi = EDGE_RANGE
    results.append(report(lo * EXPECTED_EDGES <= edges <= hi * EXPECTED_EDGES,
                          f"edges: {edges} (expected near {EXPECTED_EDGES}, sanity range only)"))
    nan = np.isnan(syn.astype(np.float64)).sum() if syn.dtype.kind == "f" else 0
    results.append(report(nan == 0, f"NaN weights: {nan}"))
    results.append(report(bool((syn > 0).all()), f"weights > 0: min {syn.min()}"))

    nt_names = nt[args.nt_col].to_pylist()
    nt_bodies = nt[args.body_col].to_numpy()
    lowered = [str(v).lower() if v is not None else "none" for v in nt_names]
    values, counts = np.unique(lowered, return_counts=True)
    print("consensus_nt counts:")
    for v, c in zip(values, counts):
        print(f"  {v:15s} {c:8d}  sign {NT_SIGN.get(v, 'unknown')}")
    unknown = [v for v in values if v not in NT_SIGN]
    results.append(report(not unknown, f"unknown consensus_nt values: {unknown}"))

    sign_of = dict(zip(nt_bodies.tolist(), (NT_SIGN.get(v, 0) for v in lowered)))
    pre_ids = np.unique(pre)
    covered = sum(1 for b in pre_ids.tolist() if b in sign_of)
    results.append(report(covered == pre_ids.size,
                          f"presynaptic neurons with a consensus_nt: {covered} / {pre_ids.size}"))
    sample = pre_ids[: min(1000, pre_ids.size)].tolist()
    signs = [sign_of.get(b, 0) for b in sample]
    print(f"sign check on {len(sample)} presynaptic neurons: "
          f"+1 {signs.count(1)}, -1 {signs.count(-1)}, 0 {signs.count(0)}")

    print(f"{sum(results)} / {len(results)} checks passed")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
