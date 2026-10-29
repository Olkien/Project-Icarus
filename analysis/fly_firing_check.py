import argparse
import time
from pathlib import Path

import numpy as np

from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import CACHE_DIR, ConnectomeNetwork
from fly.lif import mean_drive_mV, psp_peak_mV

OUT_PATH = REPO_ROOT / "results" / "fly_firing_check.txt"
INPUT_GROUP = "superclass=ol_sensory,class=visual"
MAX_MEMORY_MB = 12_000
MIN_STEPS_PER_S = 1.0
WARMUP_STEPS = 50
CHECK_EVERY = 500


def run(cache: Path, seconds: float, rate_hz: float, seed: int, stepper: str, out: Path) -> None:
    rep = Report()
    start = time.perf_counter()
    net = ConnectomeNetwork(cache)
    if stepper == "numba":
        try:
            from fly.lif_numba import NumbaLIFStepper
        except ImportError:
            rep.say("numba not installed, using the NumPy stepper")
            stepper = "numpy"
        else:
            net.lif = NumbaLIFStepper(net.weights, net.lif_params)
    load_s = time.perf_counter() - start
    inputs = net.group(INPUT_GROUP)
    rates = np.zeros(net.n, dtype=np.float32)
    rates[inputs] = rate_hz
    net.reset(seed)
    net.set_rates(rates)
    steps = round(seconds * 1000.0 / net.dt_ms)
    rep.say(f"cache {cache.name}: {net.n} neurons, {net.weights.nnz} nonzero weights, "
            f"loaded in {load_s:.1f} s")
    rep.say(f"input {INPUT_GROUP}: {inputs.size} neurons at {rate_hz:g} Hz Poisson")
    w_in = net.lif_params["input_weight_mV"]
    rep.say(f"input units: each neuron gets its own Poisson train; each input spike adds "
            f"{w_in:g} mV to g (peak membrane step {psp_peak_mV(w_in):.2f} mV); "
            f"mean drive {mean_drive_mV(rate_hz, w_in):.2f} mV, threshold is 7 mV above rest")
    rep.say(f"stepper {stepper}, {steps} steps of {net.dt_ms} ms ({seconds:g} s model time)")

    counts = np.zeros(net.n, dtype=np.int64)
    stopped = None
    for _ in range(WARMUP_STEPS):
        counts[net.step()] += 1
    timed_start = time.perf_counter()
    done = WARMUP_STEPS
    while done < steps:
        counts[net.step()] += 1
        done += 1
        if done % CHECK_EVERY == 0:
            speed = (done - WARMUP_STEPS) / (time.perf_counter() - timed_start)
            if peak_memory_mb() > MAX_MEMORY_MB:
                stopped = f"memory over {MAX_MEMORY_MB} MB at step {done}"
                break
            if speed < MIN_STEPS_PER_S:
                stopped = f"speed {speed:.2f} steps/s under {MIN_STEPS_PER_S} at step {done}"
                break
    timed = time.perf_counter() - timed_start
    sim_s = done * net.dt_ms / 1000.0
    rate = counts / sim_s
    fired = counts > 0
    rep.say(f"steps run {done}, {(done - WARMUP_STEPS) / timed:.0f} steps/s "
            f"(numba compile excluded), wall {timed:.1f} s")
    rep.say(f"mean firing rate over all neurons {rate.mean():.3f} Hz")
    rep.say(f"neurons that fired at least once {int(fired.sum())} of {net.n} "
            f"({100 * fired.mean():.2f}%)")
    in_mask = np.zeros(net.n, dtype=bool)
    in_mask[inputs] = True
    rep.say(f"input group mean rate {rate[in_mask].mean():.2f} Hz, "
            f"rest of network {rate[~in_mask].mean():.4f} Hz, "
            f"max single neuron {rate.max():.1f} Hz")
    rep.say("by superclass (neurons, share fired, mean Hz):")
    sc = net.columns["superclass"]
    for name in sorted(set(sc.tolist()), key=lambda s: -(sc == s).sum()):
        mask = sc == name
        rep.say(f"  {name or '<missing>':24s} {int(mask.sum()):7d}  "
                f"{100 * fired[mask].mean():6.2f}%  {rate[mask].mean():8.3f}")
    rep.say(f"peak memory {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.1f} s")
    rep.say(f"STOPPED: {stopped}" if stopped else "finished")
    rep.save(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(CACHE_DIR / "network_A.npz"))
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stepper", choices=("numba", "numpy"), default="numba")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    run(Path(args.cache), args.seconds, args.rate_hz, args.seed, args.stepper, Path(args.out))


if __name__ == "__main__":
    main()
