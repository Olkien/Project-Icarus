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


# fast batched swaps
class SwapState:
    def __init__(self, pre: np.ndarray, post: np.ndarray, n: int, seed: int):
        self.pre = pre.astype(np.int32, copy=False)
        self.post = post.astype(np.int32)
        self.pre_key = self.pre.astype(np.int64) * n
        self.pair_index = np.full(post.size, -1, dtype=np.int32)
        self.rng = np.random.default_rng(seed)

    def round(self) -> int:
        size = self.post.size
        m = size // 2
        perm = self.rng.permutation(size)
        i, j = perm[:m], perm[m:2 * m]
        a, b, c, d = self.pre[i], self.post[i], self.pre[j], self.post[j]
        ok = (a != d) & (c != b) & (a != c) & (b != d)
        ii, jj, bb, dd = i[ok], j[ok], b[ok], d[ok]
        del perm, i, j, a, b, c, d, ok
        self.post[ii] = dd
        self.post[jj] = bb
        accepted = np.ones(ii.size, dtype=bool)
        self.pair_index[ii] = np.arange(ii.size, dtype=np.int32)
        self.pair_index[jj] = np.arange(ii.size, dtype=np.int32)

        keys = self.pre_key + self.post
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        dup = np.flatnonzero(keys[1:] == keys[:-1])
        bad_edges = np.concatenate([order[dup], order[dup + 1]])
        bad = self.changed_pairs(bad_edges, accepted)
        while bad.size:
            self.post[ii[bad]] = bb[bad]
            self.post[jj[bad]] = dd[bad]
            accepted[bad] = False
            restored = np.concatenate([self.pre_key[ii[bad]] + bb[bad],
                                       self.pre_key[jj[bad]] + dd[bad]])
            bad = self.changed_pairs(self.edges_with_keys(keys, order, restored), accepted)

        self.pair_index[ii] = -1
        self.pair_index[jj] = -1
        return int(accepted.sum())

    def changed_pairs(self, edges: np.ndarray, accepted: np.ndarray) -> np.ndarray:
        pairs = self.pair_index[edges]
        pairs = np.unique(pairs[pairs >= 0])
        return pairs[accepted[pairs]]

    def edges_with_keys(self, sorted_keys: np.ndarray, order: np.ndarray,
                        wanted: np.ndarray) -> np.ndarray:
        lo = np.searchsorted(sorted_keys, wanted, side="left")
        hi = np.searchsorted(sorted_keys, wanted, side="right")
        found = [order[s:e] for s, e in zip(lo.tolist(), hi.tolist()) if e > s]
        if not found:
            return np.empty(0, dtype=np.int64)
        edges = np.unique(np.concatenate(found))
        current = self.pre_key[edges] + self.post[edges]
        return edges[np.isin(current, wanted)]


def shuffle_edges(
    pre: np.ndarray, post: np.ndarray, n: int, seed: int, swaps_per_edge: float = 10
) -> tuple[np.ndarray, int, int]:
    state = SwapState(pre, post, n, seed)
    target = int(swaps_per_edge * post.size)
    done = 0
    rounds = 0
    idle = 0
    while done < target and idle < MAX_IDLE_ROUNDS:
        accepted = state.round()
        done += accepted
        rounds += 1
        idle = idle + 1 if accepted == 0 else 0
    return state.post, done, rounds


# reference version
def contains(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sorted_keys, keys)
    pos[pos == sorted_keys.size] = 0
    return sorted_keys[pos] == keys


def swap_round_reference(
    pre: np.ndarray, post: np.ndarray, n: int, rng: np.random.Generator
) -> int:
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


def shuffle_edges_reference(
    pre: np.ndarray, post: np.ndarray, n: int, seed: int, swaps_per_edge: float = 10
) -> tuple[np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    post = post.copy()
    target = int(swaps_per_edge * pre.size)
    done = 0
    rounds = 0
    idle = 0
    while done < target and idle < MAX_IDLE_ROUNDS:
        accepted = swap_round_reference(pre, post, n, rng)
        done += accepted
        rounds += 1
        idle = idle + 1 if accepted == 0 else 0
    return post, done, rounds


def shuffle(
    network, seed: int, swaps_per_edge: float = 10, reference: bool = False
) -> WeightNetwork:
    n = network.weights.shape[0]
    if reference:
        pre, post, weight = edge_arrays(network.weights)
        new_post, swaps, rounds = shuffle_edges_reference(pre, post, n, seed, swaps_per_edge)
    else:
        w = csc_matrix(network.weights)
        w.sum_duplicates()
        pre = np.repeat(np.arange(w.shape[1], dtype=np.int32), np.diff(w.indptr))
        weight = w.data.copy()
        new_post, swaps, rounds = shuffle_edges(pre, w.indices, n, seed, swaps_per_edge)
        del w
    weights = csc_matrix(coo_matrix((weight, (new_post, pre)), shape=network.weights.shape))
    groups = {name: network.group(name) for name in network.group_names()}
    out = WeightNetwork(weights, groups, getattr(network, "lif_params", None))
    out.shuffle_report = {"seed": seed, "edges": int(pre.size), "swaps": swaps, "rounds": rounds,
                          "swaps_per_edge": swaps / max(1, pre.size)}
    return out
