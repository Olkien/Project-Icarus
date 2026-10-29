import math

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from fly.lif import LIF_DEFAULTS, LIFStepper
from fly.synthetic import SyntheticNetwork


def single(params: dict | None = None) -> LIFStepper:
    return LIFStepper(csc_matrix((1, 1), dtype=np.float32), params)


def test_constant_drive_spikes_at_analytic_time():
    lif = single()
    drive = 10.0
    lif.bias[:] = drive
    p = LIF_DEFAULTS
    gap = p["v_thresh_mV"] - p["v_rest_mV"]
    t_first_ms = -p["tau_m_ms"] * math.log(1.0 - gap / drive)
    spike_steps = [k for k in range(1, 2000) if lif.step().size]
    first_ms = spike_steps[0] * p["dt_ms"]
    assert abs(first_ms - t_first_ms) <= p["dt_ms"]
    isi_ms = (spike_steps[1] - spike_steps[0]) * p["dt_ms"]
    assert abs(isi_ms - (t_first_ms + p["t_ref_ms"])) <= p["dt_ms"]


def test_below_threshold_drive_never_spikes():
    lif = single()
    lif.bias[:] = 6.9
    assert not any(lif.step().size for _ in range(5000))


def test_state_is_float32():
    lif = single()
    lif.step()
    assert lif.v.dtype == np.float32 and lif.g.dtype == np.float32


def test_synaptic_delay_is_18_steps():
    w = csc_matrix(np.array([[0.0, 0.0], [50.0, 0.0]], dtype=np.float32))
    lif = LIFStepper(w)
    lif.bias[0] = 100.0
    first_pre = None
    first_g = None
    for k in range(200):
        spikes = lif.step()
        if first_pre is None and 0 in spikes:
            first_pre = k
        if first_pre is not None and first_g is None and lif.g[1] > 0.0:
            first_g = k
            break
    assert first_g - first_pre == LIF_DEFAULTS["delay_steps"]


def test_poisson_input_seeded():
    def trace(seed: int) -> list[float]:
        lif = single()
        lif.reset(seed)
        lif.set_rates(np.array([500.0]))
        out = []
        for _ in range(2000):
            lif.step()
            out.append(float(lif.g[0]))
        return out

    assert trace(1) == trace(1)
    assert trace(1) != trace(2)


def test_poisson_input_mean_drive():
    lif = single({"input_weight_mV": 1.0, "tau_syn_ms": 1.0e9})
    lif.reset(3)
    lif.set_rates(np.array([400.0]))
    steps = 50000
    for _ in range(steps):
        lif.step()
    expected = 400.0 * steps * LIF_DEFAULTS["dt_ms"] / 1000.0
    assert abs(float(lif.g[0]) - expected) < 0.1 * expected


def stand_in() -> SyntheticNetwork:
    net = SyntheticNetwork()
    net.reset(0)
    return net


def test_no_runaway_firing():
    net = stand_in()
    rates = np.zeros(net.n, dtype=np.float32)
    for name in ("vis_L", "vis_R"):
        rates[net.group(name)] = 200.0
    total = 0
    steps = 5000
    for _ in range(steps):
        total += net.step(rates).size
    mean_hz = total / net.n / (steps * net.dt_ms / 1000.0)
    assert 0.0 < mean_hz < 50.0


def test_silent_without_input():
    net = stand_in()
    rates = np.zeros(net.n, dtype=np.float32)
    assert sum(net.step(rates).size for _ in range(2000)) == 0


def test_seeded_determinism():
    runs = []
    for seed in (4, 4, 5):
        net = SyntheticNetwork()
        net.reset(seed)
        rates = np.zeros(net.n, dtype=np.float32)
        rates[net.group("vis_L")] = 150.0
        runs.append([tuple(net.step(rates)) for _ in range(1000)])
    assert runs[0] == runs[1]
    assert runs[0] != runs[2]


def test_wiring_fixed_by_wiring_seed():
    a = SyntheticNetwork()
    b = SyntheticNetwork()
    c = SyntheticNetwork(wiring_seed=1)
    assert (a.weights != b.weights).nnz == 0
    assert (a.weights != c.weights).nnz > 0


def test_planted_pathway_is_lateral():
    net = stand_in()
    for side, other in (("L", "R"), ("R", "L")):
        net.reset(1)
        rates = np.zeros(net.n, dtype=np.float32)
        rates[net.group("vis_" + side)] = 300.0
        same = 0
        cross = 0
        out_same = set(net.group("out_" + side).tolist())
        out_cross = set(net.group("out_" + other).tolist())
        for _ in range(5000):
            spikes = net.step(rates).tolist()
            same += sum(s in out_same for s in spikes)
            cross += sum(s in out_cross for s in spikes)
        assert same > 2 * cross + 10


def test_groups_exist():
    net = SyntheticNetwork()
    for name in ("vis_L", "vis_R", "air_fwd", "air_back", "air_L", "air_R", "out_L", "out_R",
                 "fwd"):
        ids = net.group(name)
        assert ids.size > 0 and ids.max() < net.n
    excit = net.signs > 0
    assert 0.6 < excit.mean() < 0.95


def test_numba_stepper_matches_numpy():
    pytest.importorskip("numba")
    from fly.lif_numba import NumbaLIFStepper

    a = SyntheticNetwork()
    b = SyntheticNetwork()
    b.lif = NumbaLIFStepper(b.weights, b.lif_params)
    rates = np.zeros(a.n, dtype=np.float32)
    rates[a.group("vis_L")] = 300.0
    rates[a.group("air_fwd")] = 150.0
    for net in (a, b):
        net.reset(7)
        net.set_rates(rates)
    total = 0
    for _ in range(4000):
        sa = a.step()
        sb = b.step()
        assert np.array_equal(sa, sb)
        total += sa.size
    assert total > 0
    assert np.array_equal(a.lif.v, b.lif.v)
    assert np.array_equal(a.lif.g, b.lif.g)


def test_input_units():
    from fly.lif import mean_drive_mV, psp_peak_mV

    p = LIF_DEFAULTS
    tm, ts = p["tau_m_ms"], p["tau_syn_ms"]
    t_peak = math.log(tm / ts) * tm * ts / (tm - ts)
    analytic = ts / (tm - ts) * (math.exp(-t_peak / tm) - math.exp(-t_peak / ts))
    assert psp_peak_mV(10.0) == pytest.approx(10.0 * analytic, rel=0.03)
    assert psp_peak_mV(0.275) == pytest.approx(0.275 * analytic, rel=0.03)
    assert mean_drive_mV(20.0, 10.0) == pytest.approx(1.0)
    assert mean_drive_mV(200.0, 10.0) == pytest.approx(10.0)


def test_spike_mode_rate_and_delivery():
    w = csc_matrix(np.array([[0.0, 0.0], [50.0, 0.0]], dtype=np.float32))
    lif = LIFStepper(w)
    lif.reset(2)
    lif.set_spike_rates(np.array([200.0, 0.0]))
    steps = 50_000
    count = 0
    target_g = 0.0
    for _ in range(steps):
        spikes = lif.step()
        count += int(0 in spikes)
        target_g = max(target_g, float(lif.g[1]))
    expected = 200.0 * steps * LIF_DEFAULTS["dt_ms"] / 1000.0
    assert count == pytest.approx(expected, rel=0.1)
    assert target_g > 0.0
    assert lif.input_ids.size == 0


def test_spike_mode_seeded_and_off_by_default():
    def run(seed: int, spike: bool) -> list:
        net = SyntheticNetwork()
        net.reset(seed)
        rates = np.zeros(net.n, dtype=np.float32)
        rates[net.group("vis_L")] = 100.0
        if spike:
            net.set_spike_rates(rates)
        return [tuple(net.step().tolist()) for _ in range(500)]

    assert run(3, True) == run(3, True)
    assert run(3, True) != run(4, True)
    assert all(s == () for s in run(3, False))


def test_numba_spike_mode_matches_numpy():
    pytest.importorskip("numba")
    from fly.lif_numba import NumbaLIFStepper

    a = SyntheticNetwork()
    b = SyntheticNetwork()
    b.lif = NumbaLIFStepper(b.weights, b.lif_params)
    rates = np.zeros(a.n, dtype=np.float32)
    rates[a.group("vis_R")] = 150.0
    for net in (a, b):
        net.reset(5)
        net.set_spike_rates(rates)
    for _ in range(2000):
        assert np.array_equal(a.step(), b.step())
