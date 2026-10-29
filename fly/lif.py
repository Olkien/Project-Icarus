import math

import numpy as np
from scipy.sparse import csc_matrix

LIF_DEFAULTS = {
    "dt_ms": 0.1,
    "v_rest_mV": -52.0,
    "v_reset_mV": -52.0,
    "v_thresh_mV": -45.0,
    "t_ref_ms": 2.2,
    "delay_steps": 18,
    "tau_syn_ms": 5.0,
    "tau_m_ms": 20.0,
    "weight_mV_per_synapse": 0.275,
    "input_weight_mV": 10.0,  # added to g per input spike, not to v
}


def psp_peak_mV(weight_mV: float, params: dict | None = None) -> float:
    lif = LIFStepper(csc_matrix((1, 1), dtype=np.float32), params)
    lif.g[0] = weight_mV
    peak = 0.0
    for _ in range(round(10 * lif.p["tau_m_ms"] / lif.dt_ms)):
        lif.step()
        peak = max(peak, float(lif.v[0] - lif.v_rest))
    return peak


def mean_drive_mV(rate_hz: float, weight_mV: float, params: dict | None = None) -> float:
    p = {**LIF_DEFAULTS, **(params or {})}
    return weight_mV * rate_hz * p["tau_syn_ms"] / 1000.0


class LIFStepper:
    def __init__(self, weights: csc_matrix, params: dict | None = None):
        self.p = {**LIF_DEFAULTS, **(params or {})}
        w = csc_matrix(weights, dtype=np.float32)
        w.sort_indices()
        self.n = w.shape[0]
        self.indptr = w.indptr
        self.indices = w.indices
        self.data = w.data
        p = self.p
        self.dt_ms = p["dt_ms"]
        self.v_rest = np.float32(p["v_rest_mV"])
        self.v_reset = np.float32(p["v_reset_mV"])
        self.v_thresh = np.float32(p["v_thresh_mV"])
        self.ref_steps = round(p["t_ref_ms"] / self.dt_ms)
        self.delay = int(p["delay_steps"])
        self.decay_m = np.float32(math.exp(-self.dt_ms / p["tau_m_ms"]))
        self.decay_syn = np.float32(math.exp(-self.dt_ms / p["tau_syn_ms"]))
        self.input_weight = np.float32(p["input_weight_mV"])
        self.p_scale = self.dt_ms / 1000.0
        self.bias = np.zeros(self.n, dtype=np.float32)
        self.reset(0)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.v = np.full(self.n, self.v_rest, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.ref = np.zeros(self.n, dtype=np.int32)
        self.ring = [np.empty(0, dtype=np.int64) for _ in range(self.delay)]
        self.pos = 0
        self.steps = 0
        self.input_ids = np.empty(0, dtype=np.int64)
        self.input_p = np.empty(0, dtype=np.float64)
        self.spike_ids = np.empty(0, dtype=np.int64)
        self.spike_p = np.empty(0, dtype=np.float64)

    def set_spike_rates(self, rates_hz: np.ndarray) -> None:
        self.spike_ids = np.flatnonzero(rates_hz > 0.0)
        self.spike_p = (rates_hz[self.spike_ids] * self.p_scale).astype(np.float64)

    def add_forced(self, spikes: np.ndarray) -> np.ndarray:
        forced = self.spike_ids[self.rng.random(self.spike_ids.size) < self.spike_p]
        if forced.size == 0:
            return spikes
        self.v[forced] = self.v_reset
        self.ref[forced] = self.ref_steps
        return np.union1d(spikes, forced)

    def set_rates(self, rates_hz: np.ndarray) -> None:
        self.input_ids = np.flatnonzero(rates_hz > 0.0)
        self.input_p = (rates_hz[self.input_ids] * self.p_scale).astype(np.float64)

    def deliver(self, spikes: np.ndarray) -> None:
        indptr, indices, data, g = self.indptr, self.indices, self.data, self.g
        for j in spikes:
            a, b = indptr[j], indptr[j + 1]
            if a != b:
                g[indices[a:b]] += data[a:b]

    def step(self, rates_hz: np.ndarray | None = None) -> np.ndarray:
        if rates_hz is not None:
            self.set_rates(rates_hz)
        if self.input_ids.size:
            fired = self.input_ids[self.rng.random(self.input_ids.size) < self.input_p]
            self.g[fired] += self.input_weight
        self.deliver(self.ring[self.pos])

        v_inf = self.v_rest + self.g + self.bias
        active = self.ref == 0
        self.v[active] = v_inf[active] + (self.v[active] - v_inf[active]) * self.decay_m
        self.ref[~active] -= 1
        self.g *= self.decay_syn

        spikes = np.flatnonzero(self.v >= self.v_thresh)
        self.v[spikes] = self.v_reset
        self.ref[spikes] = self.ref_steps
        if self.spike_ids.size:
            spikes = self.add_forced(spikes)
        self.ring[self.pos] = spikes
        self.pos = (self.pos + 1) % self.delay
        self.steps += 1
        return spikes
