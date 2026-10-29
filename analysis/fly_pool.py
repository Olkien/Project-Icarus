import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from analysis.stability_scan import FLY_MODEL_PATH, stability_run
from fly.connectome import CACHE_DIR, ConnectomeNetwork
from fly.network import WeightNetwork

NETWORKS = {
    "orig": CACHE_DIR / "network_A.npz",
    "glu": CACHE_DIR / "network_A_glu.npz",
}
WINDOW_MS = 20.0

_current: dict = {}


def shuffled_path(seed: int) -> Path:
    return CACHE_DIR / f"shuffled_A_s{seed}.npz"


def network_path(key: str) -> Path:
    if key.startswith("shuf"):
        return shuffled_path(int(key[4:]))
    return NETWORKS[key]


def fly_model() -> dict:
    return json.loads(FLY_MODEL_PATH.read_text(encoding="utf-8"))


def get_net(key: str, scale: float):
    from fly.lif_numba import NumbaLIFStepper

    if _current.get("key") != key:
        _current.clear()
        path = network_path(key)
        if key.startswith("shuf"):
            net = WeightNetwork()
            net.load(path)
        else:
            net = ConnectomeNetwork(path)
        net.lif = NumbaLIFStepper(net.weights, net.lif_params)
        _current.update(key=key, net=net, scale=1.0)
    net = _current["net"]
    if _current["scale"] != scale:
        net.set_weight_scale(scale)
        _current["scale"] = scale
    return net


def rate_vector(n: int, drive: list[tuple[np.ndarray, float]]) -> np.ndarray:
    rates = np.zeros(n, dtype=np.float32)
    for ids, rate in drive:
        rates[ids] = np.maximum(rates[ids], rate)
    return rates


def run_job(job: dict):
    net = get_net(job["net"], job["scale"])
    kind = job["kind"]
    if kind == "stability":
        return stability_run(net, job["ids"], job["rate"], job["seed"])
    net.reset(job["seed"])
    net.set_spike_rates(rate_vector(net.n, job["drive"]))
    steps = round(job.get("seconds", 1.0) * 1000.0 / net.dt_ms)
    if kind == "windows":
        groups = job["groups"]
        lookup = np.full(net.n, len(groups), dtype=np.int64)
        for g, ids in enumerate(groups):
            lookup[ids] = g
        win = round(WINDOW_MS / net.dt_ms)
        out = np.zeros((steps // win, len(groups)), dtype=np.int64)
        for k in range(steps):
            spikes = net.step()
            if spikes.size:
                counts = np.bincount(lookup[spikes], minlength=len(groups) + 1)
                out[k // win] += counts[:len(groups)]
        return out
    if kind == "units":
        counts = np.zeros(net.n, dtype=np.int64)
        for _ in range(steps):
            counts[net.step()] += 1
        rate = counts / job.get("seconds", 1.0)
        return np.array([rate[ids].mean() for ids in job["units"]]), float(rate.mean())
    raise ValueError(f"unknown job kind {kind!r}")


def run_all(jobs: list[dict], workers: int) -> list:
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(run_job, jobs, chunksize=1))


def side_ids(net: ConnectomeNetwork, query: str) -> dict[str, np.ndarray]:
    return {s: net.group(f"{query},somaSide={s}") for s in ("L", "R")}
