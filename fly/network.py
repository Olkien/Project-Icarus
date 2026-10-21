from pathlib import Path
from typing import Protocol

import numpy as np


class Network(Protocol):
    n: int
    dt_ms: float

    def load(self, path: Path | str) -> None: ...

    def reset(self, seed: int) -> None: ...

    def step(self, poisson_rates: np.ndarray | None = None) -> np.ndarray: ...

    def set_rates(self, poisson_rates: np.ndarray) -> None: ...

    def group(self, name: str) -> np.ndarray: ...

    def group_names(self) -> list[str]: ...


def build_network(spec: dict) -> Network:
    kind = spec.get("type")
    if kind == "synthetic":
        from fly.synthetic import SyntheticNetwork

        return SyntheticNetwork(**{k: v for k, v in spec.items() if k != "type"})
    raise ValueError(f"unknown network type {kind!r}")
