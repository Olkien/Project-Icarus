import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from fly.lif import LIF_DEFAULTS
from fly.network import WeightNetwork

RESULTS_CSV = Path(__file__).resolve().parent.parent / "results" / "bench_synthetic.csv"
STEPS_PER_TICK = 50
EPISODE_STEPS = 200_000
FIELDS = (
    "neurons", "edges_per_neuron", "edges", "build_s", "ticks", "steps_per_s", "ms_per_step",
    "peak_mem_mb", "mean_rate_hz", "episode_s", "stepper",
)


def peak_memory_mb() -> float:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        query = kernel.K32GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 2**20
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def random_network(
    neurons: int, edges_per_neuron: int, seed: int = 0, inhibitory_fraction: float = 0.2,
    input_fraction: float = 0.01, weight_scale: float = 1.0,
) -> WeightNetwork:
    rng = np.random.default_rng(seed)
    signs = np.where(rng.random(neurons) < inhibitory_fraction, -1.0, 1.0).astype(np.float32)
    pre = np.repeat(np.arange(neurons, dtype=np.int32), edges_per_neuron)
    post = rng.integers(0, neurons, size=pre.size, dtype=np.int32)
    synapses = rng.integers(1, 6, size=pre.size).astype(np.float32)
    weight = synapses * np.float32(LIF_DEFAULTS["weight_mV_per_synapse"] * weight_scale)
    weight *= signs[pre]
    del synapses
    weights = csc_matrix(coo_matrix((weight, (post, pre)), shape=(neurons, neurons)))
    del pre, post, weight
    n_inputs = max(1, int(neurons * input_fraction))
    groups = {"input": np.arange(n_inputs)}
    return WeightNetwork(weights, groups)


def run_bench(
    neurons: int, edges_per_neuron: int, ticks: int, input_hz: float = 300.0,
    weight_scale: float = 1.0, warmup_ticks: int = 20, seed: int = 0, stepper: str = "numpy",
    input_fraction: float = 0.05,
) -> dict:
    start = time.perf_counter()
    net = random_network(neurons, edges_per_neuron, seed, input_fraction=input_fraction,
                         weight_scale=weight_scale)
    if stepper == "numba":
        from fly.lif_numba import NumbaLIFStepper

        net.lif = NumbaLIFStepper(net.weights, net.lif_params)
    build_s = time.perf_counter() - start
    net.reset(seed)
    rates = np.zeros(neurons, dtype=np.float32)
    rates[net.group("input")] = input_hz
    net.set_rates(rates)
    for _ in range(warmup_ticks * STEPS_PER_TICK):
        net.step()
    steps = ticks * STEPS_PER_TICK
    spikes = 0
    start = time.perf_counter()
    for _ in range(steps):
        spikes += net.step().size
    elapsed = time.perf_counter() - start
    steps_per_s = steps / elapsed
    sim_s = steps * net.dt_ms / 1000.0
    return {
        "neurons": neurons, "edges_per_neuron": edges_per_neuron, "edges": int(net.weights.nnz),
        "build_s": round(build_s, 2), "ticks": ticks, "steps_per_s": round(steps_per_s, 1),
        "ms_per_step": round(1000.0 / steps_per_s, 4), "peak_mem_mb": round(peak_memory_mb(), 1),
        "mean_rate_hz": round(spikes / neurons / sim_s, 2),
        "episode_s": round(EPISODE_STEPS / steps_per_s, 1), "stepper": stepper,
    }


def append_csv(row: dict, path: Path) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", type=int, required=True)
    parser.add_argument("--edges-per-neuron", type=int, default=154)
    parser.add_argument("--ticks", type=int, default=40)
    parser.add_argument("--input-hz", type=float, default=300.0)
    parser.add_argument("--input-fraction", type=float, default=0.05)
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--stepper", choices=("numpy", "numba"), default="numpy")
    parser.add_argument("--out", default=str(RESULTS_CSV))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    row = run_bench(args.neurons, args.edges_per_neuron, args.ticks, args.input_hz,
                    args.weight_scale, stepper=args.stepper, input_fraction=args.input_fraction)
    for key in FIELDS:
        print(f"{key:18s} {row[key]}")
    if not args.no_save:
        append_csv(row, Path(args.out))


if __name__ == "__main__":
    main()
