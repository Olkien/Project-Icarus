from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from fly.network import WeightNetwork


@dataclass
class PruneReport:
    neurons_before: int
    neurons_kept: int
    edges_before: int
    edges_kept: int
    hops: int

    def line(self) -> str:
        return (f"hops {self.hops}: kept {self.neurons_kept}/{self.neurons_before} neurons "
                f"({100 * self.neurons_kept / self.neurons_before:.1f}%), "
                f"{self.edges_kept}/{self.edges_before} edges "
                f"({100 * self.edges_kept / max(1, self.edges_before):.1f}%)")


def reachable(adjacency: csr_matrix, start: np.ndarray, hops: int) -> np.ndarray:
    seen = np.zeros(adjacency.shape[0], dtype=bool)
    seen[start] = True
    frontier = seen.copy()
    for _ in range(hops):
        nxt = (adjacency @ frontier.astype(np.float32)) > 0
        frontier = nxt & ~seen
        if not frontier.any():
            break
        seen |= frontier
    return seen


def prune(
    network, input_ids: np.ndarray, output_ids: np.ndarray, hops: int
) -> tuple[WeightNetwork, np.ndarray, PruneReport]:
    weights = csc_matrix(network.weights)
    pattern = weights.copy()
    pattern.data = np.ones_like(pattern.data)
    forward = csr_matrix(pattern)
    backward = csr_matrix(pattern.T)
    inputs = np.asarray(input_ids, dtype=np.int64)
    outputs = np.asarray(output_ids, dtype=np.int64)
    from_inputs = reachable(forward, inputs, hops)
    to_outputs = reachable(backward, outputs, hops)
    keep = from_inputs & to_outputs
    keep[inputs] = True
    keep[outputs] = True

    kept_ids = np.flatnonzero(keep)
    new_index = np.full(weights.shape[0], -1, dtype=np.int64)
    new_index[kept_ids] = np.arange(kept_ids.size)
    sub = weights[kept_ids][:, kept_ids]
    groups = {}
    for name in network.group_names():
        ids = new_index[network.group(name)]
        groups[name] = ids[ids >= 0]
    pruned = WeightNetwork(sub, groups, getattr(network, "lif_params", None))
    report = PruneReport(weights.shape[0], int(kept_ids.size), int(weights.nnz), int(sub.nnz), hops)
    return pruned, kept_ids, report
