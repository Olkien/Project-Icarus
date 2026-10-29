import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from fly.bench import peak_memory_mb
from fly.download_data import DATA_DIR, FILES

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "results" / "check_load_output.txt"
EXPECTED_NEURONS = 166_700
EXPECTED_EDGES = 25_600_000
EDGE_RANGE = (0.8, 1.2)
NEURON_RANGE = (0.95, 1.05)
NT_SIGN = {
    "acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
    "dopamine": 0, "serotonin": 0, "octopamine": 0,
}
ANNOTATION_COUNTS = ("superclass", "class", "statusLabel", "somaSide", "dimorphism")
MISSING = "<missing>"
UNCLEAR = "unclear"
MERGE_ROWS = 20_000_000


class CheckFailed(Exception):
    pass


class Report:
    def __init__(self):
        self.lines: list[str] = []

    def say(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def fail(self, text: str) -> None:
        self.say("ERROR " + text)
        raise CheckFailed(text)

    def warn(self, text: str) -> None:
        self.say("WARN  " + text)

    def counts(self, title: str, values) -> None:
        tally = Counter(MISSING if is_missing(v) else str(v) for v in values)
        self.say(f"{title} ({len(tally)} values):")
        for value, count in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0])):
            self.say(f"  {value:32s} {count:9d}")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8", newline="\n")


def is_missing(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def sorted_unique(a: np.ndarray, in_place: bool = False) -> np.ndarray:
    if in_place:
        a.sort()
        s = a
    else:
        s = np.sort(a)
    if s.size == 0:
        return s
    keep = np.empty(s.size, dtype=bool)
    keep[0] = True
    np.not_equal(s[1:], s[:-1], out=keep[1:])
    return s[keep]


class IdSet:
    def __init__(self):
        self.ids = np.empty(0, dtype=np.int64)
        self.pending: list[np.ndarray] = []
        self.pending_rows = 0

    def add(self, ids: np.ndarray) -> None:
        self.pending.append(sorted_unique(ids))
        self.pending_rows += self.pending[-1].size
        if self.pending_rows >= MERGE_ROWS:
            self.merge()

    def merge(self) -> np.ndarray:
        if self.pending:
            merged = np.concatenate([self.ids, *self.pending])
            self.ids = np.empty(0, dtype=np.int64)
            self.pending = []
            self.ids = sorted_unique(merged, in_place=True)
            del merged
            self.pending = []
            self.pending_rows = 0
        return self.ids


def member(sorted_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    if sorted_ids.size == 0:
        return np.zeros(ids.size, dtype=bool)
    pos = np.searchsorted(sorted_ids, ids)
    pos[pos == sorted_ids.size] = 0
    return sorted_ids[pos] == ids


def open_feather(path: Path):
    try:
        import pyarrow as pa
        from pyarrow import ipc
    except ImportError:
        sys.exit("needs pyarrow: pip install pyarrow")
    if not path.exists():
        raise CheckFailed(f"missing file {path}")
    return ipc.open_file(pa.memory_map(str(path), "r"))


def need(rep: Report, names: list[str], have: list[str], label: str) -> None:
    missing = [c for c in names if c not in have]
    if missing:
        rep.fail(f"{label}: missing columns {missing}, has {have}")


def check_annotations(rep: Report, path: Path, id_col: str) -> np.ndarray:
    reader = open_feather(path)
    names = reader.schema.names
    need(rep, [id_col, "type", *ANNOTATION_COUNTS], names, "annotations")
    table = reader.read_all().select([id_col, "type", *ANNOTATION_COUNTS])
    rep.say(f"== Annotations: {path.name}")
    rep.say(f"rows {table.num_rows}, id column {id_col}")
    ids = table[id_col]
    if ids.null_count:
        rep.fail(f"annotations: {ids.null_count} rows with a missing {id_col}")
    ann_ids = ids.to_numpy()
    unique = sorted_unique(ann_ids)
    if unique.size != ann_ids.size:
        rep.fail(f"annotations: {ann_ids.size - unique.size} duplicate {id_col} values")
    for col in ANNOTATION_COUNTS:
        rep.counts(col, table[col].to_pylist())
    missing_type = sum(is_missing(v) for v in table["type"].to_pylist())
    rep.say(f"rows with a missing type: {missing_type}")
    return unique


def check_neurotransmitters(
    rep: Report, path: Path, id_col: str, nt_col: str, ann_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    reader = open_feather(path)
    need(rep, [id_col, nt_col], reader.schema.names, "neurotransmitters")
    table = reader.read_all().select([id_col, nt_col])
    rep.say(f"== Neurotransmitters: {path.name}")
    rep.say(f"rows {table.num_rows}, id column {id_col}, type column {nt_col}")
    if table[id_col].null_count:
        rep.fail(f"neurotransmitters: {table[id_col].null_count} rows with a missing {id_col}")
    bodies = table[id_col].to_numpy()
    names = table[nt_col].to_pylist()
    if sorted_unique(bodies).size != bodies.size:
        rep.fail(f"neurotransmitters: duplicate {id_col} values")
    in_ann = member(ann_ids, bodies)
    rep.say(f"body ids also in annotations: {int(in_ann.sum())} of {bodies.size}")
    rep.say(f"annotation ids with no neurotransmitter row: "
            f"{int((~member(sorted_unique(bodies), ann_ids)).sum())}")
    rep.counts(f"{nt_col} for those", [n for n, keep in zip(names, in_ann) if keep])
    unknown = sorted({str(n).lower() for n in names if not is_missing(n)} - set(NT_SIGN))
    if unknown:
        rep.warn(f"{nt_col} values with no sign rule (read as 0): {unknown}")
    has_nt = np.array([not is_missing(n) for n in names], dtype=bool)
    unclear = np.array([str(n).lower() == UNCLEAR for n in names], dtype=bool)
    order = np.argsort(bodies)
    return bodies[order], has_nt[order], unclear[order]


def check_weights(
    rep: Report, path: Path, pre_col: str, post_col: str, w_col: str,
    ann_ids: np.ndarray, nt_ids: np.ndarray, nt_has: np.ndarray, nt_unclear: np.ndarray,
) -> int:
    reader = open_feather(path)
    schema = reader.schema
    need(rep, [pre_col, post_col, w_col], schema.names, "weights")
    rep.say(f"== Weights: {path.name}")
    rep.say(f"columns: {schema.names}")
    rep.say("dtypes: " + ", ".join(f"{f.name} {f.type}" for f in schema))
    rep.say(f"pre id column {pre_col}, post id column {post_col}, weight column {w_col}")
    rows = 0
    synapses = 0
    both_ann = 0
    pre_no_nt = 0
    pre_unclear = 0
    pre_seen, post_seen = IdSet(), IdSet()
    w_min, w_max = None, None
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        cols = [batch.column(batch.schema.get_field_index(c)) for c in (pre_col, post_col, w_col)]
        for name, col in zip((pre_col, post_col, w_col), cols):
            if col.null_count:
                rep.fail(f"weights batch {i}: {col.null_count} nulls in {name}")
        pre, post, w = (c.to_numpy(zero_copy_only=False) for c in cols)
        if w.dtype.kind == "f" and np.isnan(w).any():
            rep.fail(f"weights batch {i}: NaN in {w_col}")
        if (w <= 0).any():
            rep.fail(f"weights batch {i}: {int((w <= 0).sum())} weights <= 0")
        rows += pre.size
        synapses += int(w.sum())
        w_min = w.min() if w_min is None else min(w_min, w.min())
        w_max = w.max() if w_max is None else max(w_max, w.max())
        both_ann += int((member(ann_ids, pre) & member(ann_ids, post)).sum())
        pos = np.searchsorted(nt_ids, pre)
        pos[pos == nt_ids.size] = 0
        found = nt_ids[pos] == pre
        pre_no_nt += int((~(found & nt_has[pos])).sum())
        pre_unclear += int((found & nt_unclear[pos]).sum())
        pre_seen.add(pre)
        post_seen.add(post)
        del pre, post, w, cols, batch
    pre_ids = pre_seen.merge()
    post_ids = post_seen.merge()
    rep.say(f"rows (edges) {rows}, batches {reader.num_record_batches}")
    rep.say(f"total synapses {synapses}, weight min {w_min}, max {w_max}")
    rep.say(f"distinct pre ids {pre_ids.size}, distinct post ids {post_ids.size}")
    lo, hi = EDGE_RANGE
    if not lo * EXPECTED_EDGES <= rows <= hi * EXPECTED_EDGES:
        rep.warn(f"edge count {rows} outside the sanity range near {EXPECTED_EDGES}")
    in_weights = pre_ids.size + post_ids.size - int(member(post_ids, pre_ids).sum())
    both = int((member(pre_ids, ann_ids) | member(post_ids, ann_ids)).sum())
    del pre_seen, post_seen, pre_ids, post_ids
    rep.say("== Overlap")
    rep.say(f"neurons in annotations {ann_ids.size}, in weights {in_weights}, in both {both}")
    lo, hi = NEURON_RANGE
    if not lo * EXPECTED_NEURONS <= in_weights <= hi * EXPECTED_NEURONS:
        rep.warn(f"neurons in weights {in_weights} outside the range near {EXPECTED_NEURONS}")
    rep.say(f"edges with pre and post both in annotations: {both_ann} of {rows} "
            f"({100 * both_ann / max(1, rows):.2f}%)")
    rep.say(f"edges whose presynaptic neuron has a missing consensus_nt: {pre_no_nt} "
            f"({100 * pre_no_nt / max(1, rows):.2f}%)")
    rep.say(f"edges whose presynaptic consensus_nt is {UNCLEAR}: {pre_unclear} "
            f"({100 * pre_unclear / max(1, rows):.2f}%)")
    return in_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--body-col", default="bodyId", help="annotations id column")
    parser.add_argument("--nt-body-col", default="body", help="neurotransmitter id column")
    parser.add_argument("--nt-col", default="consensus_nt")
    parser.add_argument("--pre-col", default="body_pre")
    parser.add_argument("--post-col", default="body_post")
    parser.add_argument("--weight-col", default="weight")
    parser.add_argument("--out", default=str(OUTPUT_PATH))
    args = parser.parse_args()
    data = Path(args.data)
    ann_path, nt_path, w_path = (data / name for name in FILES)
    rep = Report()
    start = time.perf_counter()
    failed = False
    try:
        ann_ids = check_annotations(rep, ann_path, args.body_col)
        nt_ids, nt_has, nt_unclear = check_neurotransmitters(
            rep, nt_path, args.nt_body_col, args.nt_col, ann_ids)
        check_weights(rep, w_path, args.pre_col, args.post_col, args.weight_col, ann_ids,
                      nt_ids, nt_has, nt_unclear)
    except CheckFailed:
        failed = True
    rep.say("== Run")
    rep.say(f"total time {time.perf_counter() - start:.1f} s, peak memory {peak_memory_mb():.0f} MB")
    rep.say("stopped at the first error" if failed else "all checks passed")
    rep.save(Path(args.out))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
