import numpy as np
from scipy.sparse import csc_matrix

from fly.network import WeightNetwork
from fly.prune import prune, reachable
from fly.synthetic import SENSORY, SyntheticNetwork

OUTPUTS = ("out_L", "out_R", "fwd")


def stand_in() -> SyntheticNetwork:
    return SyntheticNetwork()


def ids(net, names) -> np.ndarray:
    return np.concatenate([net.group(n) for n in names])


def test_reachable_on_chain():
    # 0 -> 1 -> 2 -> 3, weights[post, pre]
    w = csc_matrix(([1.0, 1.0, 1.0], ([1, 2, 3], [0, 1, 2])), shape=(4, 4))
    forward = csc_matrix(w).tocsr()
    assert reachable(forward, np.array([0]), 2).tolist() == [True, True, True, False]
    assert reachable(forward, np.array([0]), 3).all()


def test_keeps_all_inputs_and_outputs():
    net = stand_in()
    pruned, kept, report = prune(net, ids(net, SENSORY), ids(net, OUTPUTS), hops=1)
    for name in SENSORY + OUTPUTS:
        assert np.isin(net.group(name), kept).all()
        assert pruned.group(name).size == net.group(name).size
    assert report.neurons_kept == kept.size < net.n
    assert report.edges_kept == pruned.weights.nnz


def test_kept_neurons_satisfy_both_rules():
    net = stand_in()
    inputs, outputs = ids(net, SENSORY), ids(net, OUTPUTS)
    hops = 2
    _, kept, _ = prune(net, inputs, outputs, hops)
    pattern = csc_matrix(net.weights)
    pattern.data[:] = 1.0
    fwd = reachable(pattern.tocsr(), inputs, hops)
    bwd = reachable(csc_matrix(pattern.T).tocsr(), outputs, hops)
    io = np.zeros(net.n, dtype=bool)
    io[inputs] = True
    io[outputs] = True
    expected = np.flatnonzero((fwd & bwd) | io)
    assert np.array_equal(kept, expected)


def test_large_hops_keeps_output_behavior():
    net = stand_in()
    inputs, outputs = ids(net, SENSORY), ids(net, OUTPUTS)
    pruned, kept, _ = prune(net, inputs, outputs, hops=50)
    rates = np.zeros(net.n, dtype=np.float32)
    rates[net.group("vis_L")] = 300.0
    rates[net.group("air_fwd")] = 200.0
    net.reset(3)
    pruned.reset(3)
    net.set_rates(rates)
    pruned.set_rates(rates[kept])
    out_new = {name: set(pruned.group(name).tolist()) for name in OUTPUTS}
    out_old = {name: set(net.group(name).tolist()) for name in OUTPUTS}
    for _ in range(3000):
        a = net.step()
        b = kept[pruned.step()]
        for name in OUTPUTS:
            assert sorted(set(a.tolist()) & out_old[name]) == sorted(
                set(b.tolist()) & {int(kept[i]) for i in out_new[name]})


def test_pruned_network_saves_and_loads(tmp_path):
    net = stand_in()
    pruned, _, _ = prune(net, ids(net, SENSORY), ids(net, OUTPUTS), hops=2)
    path = tmp_path / "pruned.npz"
    pruned.save(path)
    loaded = WeightNetwork()
    loaded.load(path)
    assert (loaded.weights != pruned.weights).nnz == 0
    assert loaded.group_names() == pruned.group_names()
    assert np.array_equal(loaded.group("out_L"), pruned.group("out_L"))
