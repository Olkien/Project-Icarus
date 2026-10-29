import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from fly.bench import peak_memory_mb
from fly.check_load import open_feather
from fly.download_data import DATA_DIR, FILES
from fly.lif import LIF_DEFAULTS, LIFStepper
from fly.neurons import keep_mask, load_annotations, load_nt, lookup

CACHE_DIR = DATA_DIR / "network_cache"
CACHE_COLUMNS = ("superclass", "class", "type", "somaSide", "statusLabel")
NT_CACHE_COLUMNS = ("consensus_nt", "celltype_predicted_nt")
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1}
QUERY_FIELDS = {"superclass", "class", "type", "somaSide", "statusLabel", "consensus_nt",
                "celltype_predicted_nt"}


def cache_name(rule: str, glu_excitatory: bool, unclear_fallback: bool) -> str:
    parts = [f"network_{rule}"]
    if glu_excitatory:
        parts.append("glu")
    if unclear_fallback:
        parts.append("fallback")
    return "_".join(parts) + ".npz"


def neuron_signs(consensus: np.ndarray, celltype: np.ndarray, glu_excitatory: bool,
                 unclear_fallback: bool) -> np.ndarray:
    table = {**SIGN, "glutamate": 1 if glu_excitatory else -1}
    sign = np.array([table.get(nt, 0) for nt in consensus.tolist()], dtype=np.int8)
    if unclear_fallback:
        fallback = np.array([table.get(nt, 0) for nt in celltype.tolist()], dtype=np.int8)
        unclear = consensus == "unclear"
        sign[unclear] = fallback[unclear]
    return sign


def build_cache(
    data_dir: Path = DATA_DIR, out_dir: Path = CACHE_DIR, rule: str = "A",
    glu_excitatory: bool = False, unclear_fallback: bool = False,
) -> Path:
    data_dir = Path(data_dir)
    ann = load_annotations(data_dir)
    keep = keep_mask(ann["statusLabel"], rule)
    ids = ann["bodyId"][keep]
    nt = load_nt(data_dir, ids)
    sign = neuron_signs(nt["consensus_nt"], nt["celltype_predicted_nt"], glu_excitatory,
                        unclear_fallback)

    pres, posts, weights = [], [], []
    reader = open_feather(data_dir / FILES[2])
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i)
        col = batch.schema.get_field_index
        pre = batch.column(col("body_pre")).to_numpy()
        post = batch.column(col("body_post")).to_numpy()
        w = batch.column(col("weight")).to_numpy()
        pre_pos, pre_in = lookup(ids, pre)
        post_pos, post_in = lookup(ids, post)
        both = pre_in & post_in
        pres.append(pre_pos[both].astype(np.int32))
        posts.append(post_pos[both].astype(np.int32))
        weights.append(w[both].astype(np.float32))
    n = ids.size
    counts = csr_matrix((np.concatenate(weights), (np.concatenate(pres), np.concatenate(posts))),
                        shape=(n, n), dtype=np.float32)
    del pres, posts, weights
    counts.sum_duplicates()
    counts.sort_indices()

    meta = {"rule": rule, "glu_excitatory": glu_excitatory, "unclear_fallback": unclear_fallback,
            "neurons": int(n), "edges": int(counts.nnz), "synapses": float(counts.sum())}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / cache_name(rule, glu_excitatory, unclear_fallback)
    np.savez(path, indptr=counts.indptr, indices=counts.indices, data=counts.data, sign=sign,
             body_ids=ids, meta=np.array(json.dumps(meta)),
             **{f"ann_{c}": ann[c][keep] for c in CACHE_COLUMNS},
             **{f"nt_{c}": nt[c] for c in NT_CACHE_COLUMNS})
    return path


class ConnectomeNetwork:
    def __init__(self, path: Path | str | None = None, lif_params: dict | None = None):
        self.lif_params = {**LIF_DEFAULTS, **(lif_params or {})}
        self.dt_ms = self.lif_params["dt_ms"]
        self.groups: dict[str, np.ndarray] = {}
        if path is not None:
            self.load(path)

    def load(self, path: Path | str) -> None:
        with np.load(path, allow_pickle=False) as f:
            self.meta = json.loads(str(f["meta"]))
            n = int(f["body_ids"].size)
            self.counts = csr_matrix((f["data"], f["indices"], f["indptr"]), shape=(n, n))
            self.sign = f["sign"]
            self.body_ids = f["body_ids"]
            self.columns = {c: f[f"ann_{c}"] for c in CACHE_COLUMNS}
            self.columns.update({c: f[f"nt_{c}"] for c in NT_CACHE_COLUMNS})
        self.n = n
        weights = csc_matrix(self.counts.T)
        scale = np.float32(self.lif_params["weight_mV_per_synapse"])
        per_pre = self.sign.astype(np.float32) * scale
        weights.data *= np.repeat(per_pre, np.diff(weights.indptr))
        weights.eliminate_zeros()
        self.weights = weights
        self.lif = LIFStepper(self.weights, self.lif_params)

    def add_group(self, name: str, query: str) -> np.ndarray:
        self.groups[name] = self.query(query)
        return self.groups[name]

    def query(self, query: str) -> np.ndarray:
        mask = np.ones(self.n, dtype=bool)
        for term in query.split(","):
            field, sep, value = term.partition("=")
            field = field.strip()
            prefix = field.endswith("^")
            field = field.rstrip("^")
            if not sep or field not in QUERY_FIELDS:
                raise ValueError(f"bad group query {query!r}; fields: {sorted(QUERY_FIELDS)}")
            column = self.columns[field]
            value = value.strip()
            mask &= np.char.startswith(column, value) if prefix else column == value
        return np.flatnonzero(mask)

    # Network protocol
    def reset(self, seed: int) -> None:
        self.lif.reset(seed)

    def step(self, poisson_rates: np.ndarray | None = None) -> np.ndarray:
        return self.lif.step(poisson_rates)

    def set_rates(self, poisson_rates: np.ndarray) -> None:
        self.lif.set_rates(poisson_rates)

    def group(self, name: str) -> np.ndarray:
        if name in self.groups:
            return self.groups[name]
        return self.query(name)

    def group_names(self) -> list[str]:
        return list(self.groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--out", default=str(CACHE_DIR))
    parser.add_argument("--rule", choices=("A", "B"), default="A")
    parser.add_argument("--glu-excitatory", action="store_true")
    parser.add_argument("--unclear-fallback", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    path = build_cache(Path(args.data), Path(args.out), args.rule, args.glu_excitatory,
                       args.unclear_fallback)
    built = time.perf_counter() - start
    build_peak = peak_memory_mb()
    start = time.perf_counter()
    net = ConnectomeNetwork(path)
    loaded = time.perf_counter() - start
    signs = {int(k): int(v) for k, v in zip(*np.unique(net.sign, return_counts=True))}
    print(f"cache {path}")
    print(f"neurons {net.n}, edges {net.meta['edges']}, synapses {net.meta['synapses']:.0f}")
    print(f"neuron signs {signs}, nonzero LIF weights {net.weights.nnz}")
    print(f"build {built:.1f} s (peak {build_peak:.0f} MB), load {loaded:.1f} s, "
          f"peak memory {peak_memory_mb():.0f} MB")


if __name__ == "__main__":
    main()
