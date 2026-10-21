from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.sparse import csc_matrix

from fly.lif import LIF_DEFAULTS, LIFStepper


class Network(Protocol):
    n: int
    dt_ms: float

    def load(self, path: Path | str) -> None: ...

    def reset(self, seed: int) -> None: ...

    def step(self, poisson_rates: np.ndarray | None = None) -> np.ndarray: ...

    def set_rates(self, poisson_rates: np.ndarray) -> None: ...

    def group(self, name: str) -> np.ndarray: ...

    def group_names(self) -> list[str]: ...


class WeightNetwork:
    def __init__(
        self,
        weights: csc_matrix | None = None,
        groups: dict[str, np.ndarray] | None = None,
        lif_params: dict | None = None,
    ):
        self.lif_params = {**LIF_DEFAULTS, **(lif_params or {})}
        self.dt_ms = self.lif_params["dt_ms"]
        if weights is not None:
            self.build(weights, groups or {})

    def build(self, weights: csc_matrix, groups: dict[str, np.ndarray]) -> None:
        self.weights = csc_matrix(weights, dtype=np.float32)
        self.groups = {name: np.asarray(ids, dtype=np.int64) for name, ids in groups.items()}
        self.n = self.weights.shape[0]
        self.lif = LIFStepper(self.weights, self.lif_params)

    def save(self, path: Path | str) -> None:
        w = self.weights
        np.savez(path, data=w.data, indices=w.indices, indptr=w.indptr, shape=np.array(w.shape),
                 group_names=np.array(list(self.groups)),
                 **{f"group_{i}": ids for i, ids in enumerate(self.groups.values())})

    def load(self, path: Path | str) -> None:
        with np.load(path) as f:
            weights = csc_matrix((f["data"], f["indices"], f["indptr"]), shape=tuple(f["shape"]))
            groups = {str(name): f[f"group_{i}"] for i, name in enumerate(f["group_names"])}
        self.build(weights, groups)

    def reset(self, seed: int) -> None:
        self.lif.reset(seed)

    def step(self, poisson_rates: np.ndarray | None = None) -> np.ndarray:
        return self.lif.step(poisson_rates)

    def set_rates(self, poisson_rates: np.ndarray) -> None:
        self.lif.set_rates(poisson_rates)

    def group(self, name: str) -> np.ndarray:
        return self.groups[name]

    def group_names(self) -> list[str]:
        return list(self.groups)


def build_network(spec: dict) -> Network:
    kind = spec.get("type")
    if kind == "synthetic":
        from fly.synthetic import SyntheticNetwork

        return SyntheticNetwork(**{k: v for k, v in spec.items() if k != "type"})
    raise ValueError(f"unknown network type {kind!r}")
