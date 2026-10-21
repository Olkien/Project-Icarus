import numpy as np
import pytest
from scipy.sparse import csc_matrix

from controllers.shuffled import Shuffled
from fly.shuffle import edge_arrays, shuffle
from fly.synthetic import SyntheticNetwork


@pytest.fixture(scope="module")
def original() -> SyntheticNetwork:
    return SyntheticNetwork()


@pytest.fixture(scope="module")
def shuffled(original):
    return shuffle(original, seed=1)


def degrees(w: csc_matrix) -> tuple[np.ndarray, np.ndarray]:
    pattern = csc_matrix(w)
    out_deg = np.diff(pattern.indptr)
    in_deg = np.bincount(pattern.indices, minlength=pattern.shape[0])
    return out_deg, in_deg


def keys(w: csc_matrix) -> set[int]:
    pre, post, _ = edge_arrays(w)
    return set((pre * w.shape[0] + post).tolist())


def test_degrees_preserved(original, shuffled):
    out_a, in_a = degrees(original.weights)
    out_b, in_b = degrees(shuffled.weights)
    assert np.array_equal(out_a, out_b)
    assert np.array_equal(in_a, in_b)


def test_no_self_loops_or_duplicates(shuffled):
    pre, post, _ = edge_arrays(shuffled.weights)
    assert not (pre == post).any()
    assert np.unique(pre * shuffled.n + post).size == pre.size
    assert shuffled.weights.nnz == pre.size


def test_weight_sum_and_signs_kept(original, shuffled):
    assert float(shuffled.weights.sum()) == pytest.approx(float(original.weights.sum()), rel=1e-6)
    pre, _, w = edge_arrays(shuffled.weights)
    assert np.array_equal(np.sign(w), original.signs[pre])


def test_enough_swaps(shuffled):
    assert shuffled.shuffle_report["swaps_per_edge"] >= 10


def test_most_edges_changed(original, shuffled):
    before = keys(original.weights)
    after = keys(shuffled.weights)
    assert len(after - before) >= 0.9 * len(after)


def test_seeded(original, shuffled):
    again = shuffle(original, seed=1)
    other = shuffle(original, seed=2)
    assert (again.weights != shuffled.weights).nnz == 0
    assert (other.weights != shuffled.weights).nnz > 0


def test_groups_unchanged(original, shuffled):
    for name in original.group_names():
        assert np.array_equal(original.group(name), shuffled.group(name))


def test_original_not_modified():
    net = SyntheticNetwork()
    before = net.weights.copy()
    shuffle(net, seed=3)
    assert (net.weights != before).nnz == 0


def test_shuffled_controller_uses_shuffle_seed():
    a = Shuffled({"shuffle_seed": 4})
    b = Shuffled({"shuffle_seed": 5})
    assert (a.net.weights != b.net.weights).nnz > 0
