import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from fly.lif import LIF_DEFAULTS, LIFStepper

GROUP_SIZES = {
    "vis_L": 40, "vis_R": 40,
    "air_fwd": 20, "air_back": 20, "air_L": 20, "air_R": 20,
    "out_L": 20, "out_R": 20, "fwd": 20,
}
SENSORY = ("vis_L", "vis_R", "air_fwd", "air_back", "air_L", "air_R")


class SyntheticNetwork:
    def __init__(
        self,
        n: int = 2000,
        wiring_seed: int = 0,
        out_degree: int = 30,
        inhibitory_fraction: float = 0.2,
        lif_params: dict | None = None,
    ):
        self.config = {"n": n, "wiring_seed": wiring_seed, "out_degree": out_degree,
                       "inhibitory_fraction": inhibitory_fraction}
        self.lif_params = {**LIF_DEFAULTS, **(lif_params or {})}
        self.build()

    def load(self, path: Path | str) -> None:
        self.config.update(json.loads(Path(path).read_text(encoding="utf-8")))
        self.build()

    # wiring
    def build(self) -> None:
        cfg = self.config
        n = cfg["n"]
        rng = np.random.default_rng(cfg["wiring_seed"])
        self.groups = {}
        start = 0
        for name, size in GROUP_SIZES.items():
            self.groups[name] = np.arange(start, start + size)
            start += size
        inter = np.arange(start, n)
        self.signs = np.ones(n, dtype=np.float32)
        inhib = rng.random(inter.size) < cfg["inhibitory_fraction"]
        self.signs[inter[inhib]] = -1.0

        receivers = np.concatenate([inter, self.groups["out_L"], self.groups["out_R"],
                                    self.groups["fwd"]])
        pre, post, count = [], [], []
        for j in range(n):
            if j in self.groups["out_L"] or j in self.groups["out_R"] or j in self.groups["fwd"]:
                continue
            targets = rng.choice(receivers, size=cfg["out_degree"], replace=False)
            targets = targets[targets != j]
            pre += [j] * targets.size
            post += targets.tolist()
            count += rng.integers(1, 6, size=targets.size).tolist()

        def plant(src: str, dst: str, fan_in: int, synapses: int) -> None:
            for i in self.groups[dst]:
                for j in rng.choice(self.groups[src], size=fan_in, replace=False):
                    pre.append(int(j))
                    post.append(int(i))
                    count.append(synapses)

        plant("vis_L", "out_L", 15, 30)
        plant("vis_R", "out_R", 15, 30)
        plant("vis_L", "fwd", 8, 20)
        plant("vis_R", "fwd", 8, 20)

        pre_a = np.array(pre)
        weight = (np.array(count, dtype=np.float32) * self.lif_params["weight_mV_per_synapse"]
                  * self.signs[pre_a])
        self.weights = csc_matrix(coo_matrix((weight, (np.array(post), pre_a)), shape=(n, n)))
        self.n = n
        self.dt_ms = self.lif_params["dt_ms"]
        self.lif = LIFStepper(self.weights, self.lif_params)

    # Network protocol
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
