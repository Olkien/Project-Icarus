import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report, open_feather
from fly.download_data import DATA_DIR, FILES

REPORT_PATH = REPO_ROOT / "results" / "neuron_filter_report.txt"
ANN_COLUMNS = ("bodyId", "statusLabel", "superclass", "class", "type", "somaSide")
OPTIONAL_ANN_COLUMNS = ("rootSide", "entryNerve", "exitNerve")
NT_COLUMNS = ("body", "consensus_nt", "celltype_predicted_nt", "predicted_nt")
EXCLUDE_A = frozenset({
    "Orphan", "Orphan-artifact", "Orphan hotknife", "PRT Orphan", "RT Orphan", "Glia",
    "Unimportant", "Out of scope", "",
})
EXCLUDE_B = EXCLUDE_A | {"0.5assign", "Leaves"}
RULES = {"A": EXCLUDE_A, "B": EXCLUDE_B}
SIGNED = ("acetylcholine", "gaba", "glutamate")
UNCLEAR = "unclear"
NO_SUPERCLASS = "<missing>"


def text_column(values: list) -> np.ndarray:
    return np.array(["" if v is None else str(v) for v in values], dtype=str)


def read_columns(path: Path, columns: tuple[str, ...]):
    reader = open_feather(path)
    missing = [c for c in columns if c not in reader.schema.names]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return reader.read_all().select(list(columns))


def load_annotations(data_dir: Path) -> dict[str, np.ndarray]:
    path = Path(data_dir) / FILES[0]
    present = open_feather(path).schema.names
    optional = tuple(c for c in OPTIONAL_ANN_COLUMNS if c in present)
    table = read_columns(path, ANN_COLUMNS + optional)
    ids = table["bodyId"].to_numpy()
    order = np.argsort(ids, kind="stable")
    ann = {"bodyId": ids[order]}
    for col in ANN_COLUMNS[1:] + optional:
        ann[col] = text_column(table[col].to_pylist())[order]
    for col in OPTIONAL_ANN_COLUMNS:
        ann.setdefault(col, np.full(ids.size, "", dtype=str))
    return ann


def keep_mask(status: np.ndarray, rule: str) -> np.ndarray:
    excluded = RULES[rule]
    return np.array([s not in excluded for s in status.tolist()], dtype=bool)


def kept_ids(data_dir: Path, rule: str = "A") -> np.ndarray:
    ann = load_annotations(data_dir)
    return ann["bodyId"][keep_mask(ann["statusLabel"], rule)]


def load_nt(data_dir: Path, ids: np.ndarray) -> dict[str, np.ndarray]:
    import pyarrow as pa
    import pyarrow.compute as pc

    table = read_columns(Path(data_dir) / FILES[1], NT_COLUMNS)
    table = table.filter(pc.is_in(table["body"], value_set=pa.array(ids)))
    bodies = table["body"].to_numpy()
    pos = np.searchsorted(ids, bodies)
    out = {"has_row": np.zeros(ids.size, dtype=bool)}
    out["has_row"][pos] = True
    for col in NT_COLUMNS[1:]:
        column = np.full(ids.size, "", dtype=object)
        column[pos] = [("" if v is None else str(v).lower()) for v in table[col].to_pylist()]
        out[col] = column.astype(str)
    return out


def lookup(sorted_ids: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.searchsorted(sorted_ids, ids)
    pos[pos == sorted_ids.size] = 0
    return pos, sorted_ids[pos] == ids


def report(data_dir: Path, out: Path) -> None:
    start = time.perf_counter()
    rep = Report()
    data_dir = Path(data_dir)
    ann = load_annotations(data_dir)
    ids = ann["bodyId"]
    nt = load_nt(data_dir, ids)
    keeps = {rule: keep_mask(ann["statusLabel"], rule) for rule in RULES}
    sc_names = sorted({s or NO_SUPERCLASS for s in ann["superclass"].tolist()})
    sc_code = np.array([sc_names.index(s or NO_SUPERCLASS) for s in ann["superclass"].tolist()])
    unclear = nt["consensus_nt"] == UNCLEAR
    no_row = ~nt["has_row"]

    edges = {rule: 0 for rule in RULES}
    unclear_edges = {rule: 0 for rule in RULES}
    by_sc = {rule: np.zeros(len(sc_names), dtype=np.int64) for rule in RULES}
    no_row_out_edges = 0
    no_row_has_out = np.zeros(ids.size, dtype=bool)
    reader = open_feather(data_dir / FILES[2])
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        pre = batch.column(batch.schema.get_field_index("body_pre")).to_numpy()
        post = batch.column(batch.schema.get_field_index("body_post")).to_numpy()
        pre_pos, pre_in = lookup(ids, pre)
        post_pos, post_in = lookup(ids, post)
        pre_no_row = pre_in & no_row[pre_pos]
        no_row_out_edges += int(pre_no_row.sum())
        no_row_has_out[pre_pos[pre_no_row]] = True
        for rule, keep in keeps.items():
            both = pre_in & post_in & keep[pre_pos] & keep[post_pos]
            edges[rule] += int(both.sum())
            unclear_edges[rule] += int((both & unclear[pre_pos]).sum())
            by_sc[rule] += np.bincount(sc_code[post_pos[both]], minlength=len(sc_names))

    rep.say(f"annotated bodies {ids.size}, with no NT row {int(no_row.sum())}")
    rep.say(f"bodies with no NT row that have outgoing edges: {int(no_row_has_out.sum())}, "
            f"their outgoing edges: {no_row_out_edges}")
    for rule, keep in keeps.items():
        excluded = sorted(s or "<missing>" for s in RULES[rule])
        rep.say(f"== Rule {rule}: drop statusLabel {excluded}")
        rep.say(f"neurons kept {int(keep.sum())}")
        rep.say(f"edges with both ends kept {edges[rule]}")
        rep.say("edges by superclass of the postsynaptic neuron:")
        for name, count in sorted(zip(sc_names, by_sc[rule].tolist()), key=lambda x: -x[1]):
            rep.say(f"  {name:28s} {count:10d}")
        kept_unclear = keep & unclear
        signed_fallback = np.isin(nt["celltype_predicted_nt"], SIGNED) | np.isin(
            nt["predicted_nt"], SIGNED)
        rep.say(f"kept neurons with no NT row {int((keep & no_row).sum())}")
        rep.say(f"kept neurons with consensus_nt unclear {int(kept_unclear.sum())}")
        rep.say(f"  of those with a signed celltype_predicted_nt "
                f"{int((kept_unclear & np.isin(nt['celltype_predicted_nt'], SIGNED)).sum())}, "
                f"signed predicted_nt {int((kept_unclear & np.isin(nt['predicted_nt'], SIGNED)).sum())}, "
                f"either {int((kept_unclear & signed_fallback).sum())}")
        counts = Counter(nt["celltype_predicted_nt"][kept_unclear].tolist())
        rep.say("  celltype_predicted_nt of unclear kept neurons: " + ", ".join(
            f"{k or '<missing>'} {v}" for k, v in counts.most_common()))
        share = 100 * unclear_edges[rule] / max(1, edges[rule])
        rep.say(f"edges with an unclear presynaptic neuron: {unclear_edges[rule]} ({share:.2f}%)")
    rep.say(f"== Run: {time.perf_counter() - start:.1f} s, peak memory {peak_memory_mb():.0f} MB")
    rep.save(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--out", default=str(REPORT_PATH))
    args = parser.parse_args()
    report(Path(args.data), Path(args.out))


if __name__ == "__main__":
    main()
