import numpy as np
from numba import njit

from fly.lif import LIFStepper


@njit(cache=True)
def step_kernel(v, g, ref, bias, fired, w_in, delayed, indptr, indices, data, v_rest, v_reset,
                v_thresh, decay_m, decay_syn, ref_steps):
    for k in range(fired.size):
        g[fired[k]] += w_in
    for k in range(delayed.size):
        j = delayed[k]
        for p in range(indptr[j], indptr[j + 1]):
            g[indices[p]] += data[p]
    count = 0
    for i in range(v.size):
        if ref[i] == 0:
            v_inf = v_rest + g[i] + bias[i]
            v[i] = v_inf + (v[i] - v_inf) * decay_m
        else:
            ref[i] -= 1
        g[i] *= decay_syn
        if v[i] >= v_thresh:
            count += 1
    spikes = np.empty(count, dtype=np.int64)
    n = 0
    for i in range(v.size):
        if v[i] >= v_thresh:
            spikes[n] = i
            n += 1
            v[i] = v_reset
            ref[i] = ref_steps
    return spikes


class NumbaLIFStepper(LIFStepper):
    def step(self, rates_hz: np.ndarray | None = None) -> np.ndarray:
        if rates_hz is not None:
            self.set_rates(rates_hz)
        if self.input_ids.size:
            fired = self.input_ids[self.rng.random(self.input_ids.size) < self.input_p]
        else:
            fired = self.input_ids
        spikes = step_kernel(
            self.v, self.g, self.ref, self.bias, fired.astype(np.int64), self.input_weight,
            self.ring[self.pos], self.indptr, self.indices, self.data, self.v_rest, self.v_reset,
            self.v_thresh, self.decay_m, self.decay_syn, np.int32(self.ref_steps),
        )
        self.ring[self.pos] = spikes
        self.pos = (self.pos + 1) % self.delay
        self.steps += 1
        return spikes
