import json

import numpy as np
import pytest

from controllers.fly import DEFAULT_MAPPING, Fly, load_mapping
from fly.connectome import CACHE_DIR, ConnectomeNetwork, build_cache
from fly.download_data import FILES
from fly.lif import LIF_DEFAULTS
from fly.neurons import keep_mask, kept_ids, load_annotations, report

feather = pytest.importorskip("pyarrow.feather")
pa = pytest.importorskip("pyarrow")

NEURONS = [
    (30, "Reviewed", "ol_sensory", "visual", "R7", "L", "acetylcholine", "acetylcholine"),
    (10, "Roughly traced", "cb_intrinsic", None, "LC4", "R", "gaba", "gaba"),
    (20, "Reviewed", "descending_neuron", None, "DNa02", "L", "glutamate", "glutamate"),
    (40, "0.5assign", "cb_intrinsic", None, "X1", "R", "unclear", "gaba"),
    (50, "Leaves", "cb_intrinsic", None, "X2", "L", "histamine", "histamine"),
    (60, "Orphan", None, None, None, None, "acetylcholine", "acetylcholine"),
    (70, "Glia", None, None, None, None, "gaba", "gaba"),
    (80, None, "cb_intrinsic", None, "X3", "R", "acetylcholine", "acetylcholine"),
    (90, "Reviewed", "cb_intrinsic", None, "DAN1", "R", "dopamine", "dopamine"),
    (100, "Reviewed", "vnc_motor", None, "MN1", "L", None, None),
]
EDGES = [
    (30, 10, 5), (30, 20, 3), (10, 20, 2), (20, 90, 4), (40, 30, 1), (50, 10, 7),
    (90, 100, 6), (60, 10, 9), (10, 60, 2), (999, 10, 1), (30, 999, 1), (70, 20, 3),
]
KEPT_A = [10, 20, 30, 40, 50, 90, 100]
KEPT_B = [10, 20, 30, 90, 100]
SCALE = LIF_DEFAULTS["weight_mV_per_synapse"]


def write_fake(path, neurons, edges, root_side=None):
    path.mkdir(parents=True, exist_ok=True)
    cols = list(zip(*neurons))
    ann = {
        "bodyId": pa.array(cols[0], pa.int64()), "statusLabel": list(cols[1]),
        "superclass": list(cols[2]), "class": list(cols[3]), "type": list(cols[4]),
        "somaSide": list(cols[5]),
    }
    if root_side is not None:
        ann["rootSide"] = [root_side.get(b) for b in cols[0]]
        ann["entryNerve"] = [("AN" if b == 30 else None) for b in cols[0]]
        ann["exitNerve"] = [("ProLN" if b == 100 else None) for b in cols[0]]
    feather.write_feather(pa.table(ann), path / FILES[0])
    with_nt = [n for n in neurons if n[6] is not None]
    feather.write_feather(pa.table({
        "body": pa.array([n[0] for n in with_nt], pa.int64()),
        "consensus_nt": [n[6] for n in with_nt],
        "celltype_predicted_nt": [n[7] for n in with_nt],
        "predicted_nt": [n[7] for n in with_nt],
    }), path / FILES[1])
    pre, post, w = zip(*edges)
    feather.write_feather(pa.table({
        "body_pre": pa.array(pre, pa.int64()), "body_post": pa.array(post, pa.int64()),
        "weight": pa.array(w, pa.int64()),
    }), path / FILES[2])
    return path


@pytest.fixture
def fake_data(tmp_path):
    return write_fake(tmp_path, NEURONS, EDGES)


def index(body: int) -> int:
    return KEPT_A.index(body)


def test_filter_rules(fake_data):
    assert kept_ids(fake_data, "A").tolist() == KEPT_A
    assert kept_ids(fake_data, "B").tolist() == KEPT_B
    ann = load_annotations(fake_data)
    assert ann["bodyId"].tolist() == sorted(n[0] for n in NEURONS)
    assert keep_mask(np.array(["", "Glia", "Reviewed"]), "A").tolist() == [False, False, True]


def test_cache_reindexes_and_keeps_counts(fake_data, tmp_path):
    net = ConnectomeNetwork(build_cache(fake_data, tmp_path / "cache"))
    assert net.n == len(KEPT_A)
    assert net.body_ids.tolist() == KEPT_A
    expected = {(30, 10): 5, (30, 20): 3, (10, 20): 2, (20, 90): 4, (40, 30): 1, (50, 10): 7,
                (90, 100): 6}
    assert net.counts.nnz == len(expected)
    for (pre, post), w in expected.items():
        assert net.counts[index(pre), index(post)] == w
    assert net.counts.dtype == np.float32
    assert net.meta["edges"] == len(expected) and net.meta["neurons"] == len(KEPT_A)


def test_signs_and_lif_weights(fake_data, tmp_path):
    net = ConnectomeNetwork(build_cache(fake_data, tmp_path / "cache"))
    signs = dict(zip(net.body_ids.tolist(), net.sign.tolist()))
    assert signs == {10: -1, 20: -1, 30: 1, 40: 0, 50: -1, 90: 0, 100: 0}
    w = net.weights
    assert w[index(10), index(30)] == pytest.approx(5 * SCALE)
    assert w[index(20), index(10)] == pytest.approx(-2 * SCALE)
    assert w[index(100), index(90)] == 0.0
    assert w.dtype == np.float32


def test_sign_options(fake_data, tmp_path):
    glu = ConnectomeNetwork(build_cache(fake_data, tmp_path / "c", glu_excitatory=True))
    assert dict(zip(glu.body_ids.tolist(), glu.sign.tolist()))[20] == 1
    fb = ConnectomeNetwork(build_cache(fake_data, tmp_path / "c", unclear_fallback=True))
    assert dict(zip(fb.body_ids.tolist(), fb.sign.tolist()))[40] == -1
    assert glu.meta["glu_excitatory"] and fb.meta["unclear_fallback"]


def test_rule_b_cache(fake_data, tmp_path):
    net = ConnectomeNetwork(build_cache(fake_data, tmp_path / "cache", rule="B"))
    assert net.body_ids.tolist() == KEPT_B
    assert net.counts.nnz == 5


def test_cache_round_trip_columns_and_groups(fake_data, tmp_path):
    path = build_cache(fake_data, tmp_path / "cache")
    a, b = ConnectomeNetwork(path), ConnectomeNetwork(path)
    assert (a.counts != b.counts).nnz == 0
    assert a.columns["type"].tolist() == ["LC4", "DNa02", "R7", "X1", "X2", "DAN1", "MN1"]
    assert a.columns["consensus_nt"][index(100)] == ""
    assert a.group("type=DNa02").tolist() == [index(20)]
    assert a.group("type^=LC").tolist() == [index(10)]
    assert a.group("superclass=cb_intrinsic,somaSide=R").tolist() == [index(10), index(40),
                                                                       index(90)]
    a.add_group("vis_L", "class=visual,somaSide=L")
    assert a.group_names() == ["vis_L"] and a.group("vis_L").tolist() == [index(30)]
    with pytest.raises(ValueError):
        a.query("colour=red")


def test_network_protocol_with_stepper(fake_data, tmp_path):
    net = ConnectomeNetwork(build_cache(fake_data, tmp_path / "cache"))
    rates = np.zeros(net.n, dtype=np.float32)
    rates[net.group("class=visual")] = 800.0
    runs = []
    for _ in range(2):
        net.reset(4)
        runs.append([tuple(net.step(rates).tolist()) for _ in range(3000)])
    assert runs[0] == runs[1]
    assert any(index(30) in s for s in runs[0])


def test_fly_controller_runs_on_cache(fake_data, tmp_path):
    path = build_cache(fake_data, tmp_path / "cache")
    mapping = load_mapping(DEFAULT_MAPPING)
    groups = {"vis_L": "type=R7", "vis_R": "type=LC4", "air_fwd": "type=X1",
              "air_back": "type=X2", "air_L": "type=X1", "air_R": "type=X2",
              "out_L": "type=DNa02", "out_R": "type=DAN1", "fwd": "type=MN1"}
    mapping["network"] = {"type": "connectome", "path": str(path), "groups": groups}
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    fly = Fly({"mapping": str(mapping_path)})
    fly.reset(0, {})
    obs = {"bearing": 0.5, "range": 100.0, "vel_body": [0.0, 0.0], "yaw_rate": 0.0,
           "air_rel": [50.0, 0.0]}
    action = fly.act(obs, 0.0)
    assert set(action) == {"thrust", "yaw_torque"}
    assert set(fly.info()["groups"]) == set(groups)


def test_report_runs_on_fake_data(fake_data, tmp_path):
    out = tmp_path / "report.txt"
    report(fake_data, out)
    text = out.read_text(encoding="utf-8")
    assert "neurons kept 7" in text and "neurons kept 5" in text
    assert "edges with both ends kept 7" in text
    assert "bodies with no NT row that have outgoing edges: 0" in text


REAL_CACHE = CACHE_DIR / "network_A.npz"


@pytest.mark.skipif(not REAL_CACHE.exists(), reason="real data cache not built")
def test_real_cache_counts():
    net = ConnectomeNetwork(REAL_CACHE)
    assert net.n == pytest.approx(167_526, rel=0.01)
    assert net.counts.nnz == pytest.approx(25_617_721, rel=0.01)
    assert set(np.unique(net.sign).tolist()) <= {-1, 0, 1}
    assert net.group("type=DNa02").size == 2


def test_explorer_describes_group(fake_data, tmp_path):
    from fly.explore import describe

    net = ConnectomeNetwork(build_cache(fake_data, tmp_path / "cache"))
    lines = describe(net, "superclass=cb_intrinsic", max_ids=10)
    assert lines[0] == "query superclass=cb_intrinsic: 4 neurons"
    text = "\n".join(lines)
    assert "out degree (partners)" in text and "in synapses" in text
    for body in (10, 40, 50, 90):
        assert f"  {body}  index" in text
    assert describe(net, "type=none", 5) == ["query type=none: 0 neurons"]


def test_weight_scale_scales_synapses_not_input(fake_data, tmp_path):
    path = build_cache(fake_data, tmp_path / "cache")
    base = ConnectomeNetwork(path)
    half = ConnectomeNetwork(path, weight_scale=0.5)
    assert half.weight_scale == 0.5
    assert np.allclose(half.weights.toarray(), 0.5 * base.weights.toarray())
    assert half.lif.input_weight == base.lif.input_weight
    assert (half.counts != base.counts).nnz == 0
    half.set_weight_scale(1.0)
    assert np.allclose(half.weights.toarray(), base.weights.toarray())


def test_fly_weight_scale_param():
    from fly.synthetic import SyntheticNetwork

    assert "weight_scale" in Fly.param_space()
    unit = SyntheticNetwork()
    fly = Fly({"weight_scale": 0.25})
    assert np.allclose(fly.net.weights.toarray(), 0.25 * unit.weights.toarray())
    assert np.allclose(fly.net.lif.data, 0.25 * unit.lif.data)


def test_orientation_row_is_presynaptic(tmp_path):
    pair = [(1, "Reviewed", "cb_intrinsic", None, "A", "L", "acetylcholine", "acetylcholine"),
            (2, "Reviewed", "cb_intrinsic", None, "B", "L", "acetylcholine", "acetylcholine")]
    fired = {}
    for label, edge in (("a_to_b", (1, 2, 200)), ("b_to_a", (2, 1, 200))):
        net = ConnectomeNetwork(build_cache(write_fake(tmp_path / label, pair, [edge]),
                                            tmp_path / label / "cache"))
        assert net.counts[0, 1] == (200 if label == "a_to_b" else 0)
        assert net.counts[1, 0] == (0 if label == "a_to_b" else 200)
        rates = np.zeros(2, dtype=np.float32)
        rates[net.group("type=A")] = 1000.0
        net.reset(0)
        spikes = [net.step(rates) for _ in range(2000)]
        fired[label] = sum(int(1 in s) for s in spikes)
    assert fired["a_to_b"] > 0
    assert fired["b_to_a"] == 0


def test_root_side_and_nerves_in_cache(tmp_path):
    data = write_fake(tmp_path / "d", NEURONS, EDGES, root_side={30: "RHS", 10: "LHS"})
    net = ConnectomeNetwork(build_cache(data, tmp_path / "cache"))
    assert net.columns["rootSide"][index(30)] == "RHS"
    assert net.columns["rootSide"][index(20)] == ""
    assert net.group("rootSide=LHS").tolist() == [index(10)]
    assert net.group("entryNerve=AN").tolist() == [index(30)]
    assert net.group("exitNerve=ProLN").tolist() == [index(100)]


def test_cache_without_side_columns_still_loads(fake_data, tmp_path):
    net = ConnectomeNetwork(build_cache(fake_data, tmp_path / "cache"))
    assert (net.columns["rootSide"] == "").all()
    assert net.group("rootSide=LHS").size == 0
