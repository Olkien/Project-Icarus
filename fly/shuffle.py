import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from fly.network import WeightNetwork

MAX_IDLE_ROUNDS = 5


def edge_arrays(weights: csc_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = csc_matrix(weights)
    w.sum_duplicates()
    pre = np.repeat(np.arange(w.shape[1], dtype=np.int64), np.diff(w.indptr))
    post = w.indices.astype(np.int64)
    return pre, post, w.data.copy()


def contains(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sorted_keys, keys)
    pos[pos == sorted_keys.size] = 0
    return sorted_keys[pos] == keys


def swap_round(pre: np.ndarray, post: np.ndarray, n: int, rng: np.random.Generator) -> int:
    m = pre.size // 2
    perm = rng.permutation(pre.size)
    i, j = perm[:m], perm[m:2 * m]
    a, b, c, d = pre[i], post[i], pre[j], post[j]
    ok = (a != d) & (c != b) & (a != c) & (b != d)
    current = np.sort(pre * n + post)
    new_ad = a * n + d
    new_cb = c * n + b
    ok &= ~contains(current, new_ad) & ~contains(current, new_cb)
    keys, counts = np.unique(np.concatenate([new_ad[ok], new_cb[ok]]), return_counts=True)
    clash = keys[counts > 1]
    if clash.size:
        ok &= ~np.isin(new_ad, clash) & ~np.isin(new_cb, clash)
    post[i[ok]] = d[ok]
    post[j[ok]] = b[ok]
    return int(ok.sum())


def shuffle_edges(
    pre: np.ndarray, post: np.ndarray, n: int, seed: int, swaps_per_edge: float = 10
) -> tuple[np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    post = post.copy()
    target = int(swaps_per_edge * pre.size)
    done = 0
    rounds = 0
    idle = 0
    while done < target and idle < MAX_IDLE_ROUNDS:
        accepted = swap_round(pre, post, n, rng)
        done += accepted
        rounds += 1
        idle = idle + 1 if accepted == 0 else 0
    return post, done, rounds


def shuffle(network, seed: int, swaps_per_edge: float = 10) -> WeightNetwork:
    pre, post, weight = edge_arrays(network.weights)
    n = network.weights.shape[0]
    new_post, swaps, rounds = shuffle_edges(pre, post, n, seed, swaps_per_edge)
    weights = csc_matrix(coo_matrix((weight, (new_post, pre)), shape=network.weights.shape))
    groups = {name: network.group(name) for name in network.group_names()}
    out = WeightNetwork(weights, groups, getattr(network, "lif_params", None))
    out.shuffle_report = {"seed": seed, "edges": int(pre.size), "swaps": swaps, "rounds": rounds,
                          "swaps_per_edge": swaps / max(1, pre.size)}
    return out
