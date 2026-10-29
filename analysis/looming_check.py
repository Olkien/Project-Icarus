import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from analysis.drive_scan import drive, load
from env.params import CONFIG_DIR
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import CACHE_DIR

OUT_PATH = REPO_ROOT / "results" / "looming_check.txt"
FLY_MODEL = CONFIG_DIR / "fly_model.json"
RATE_HZ = 100.0
SEEDS = tuple(range(20))
CONDITIONS = {
    "LPLC2 both": ["type=LPLC2,somaSide=L", "type=LPLC2,somaSide=R"],
    "LPLC2 left only": ["type=LPLC2,somaSide=L"],
    "LPLC2 right only": ["type=LPLC2,somaSide=R"],
    "LC4 both": ["type=LC4,somaSide=L", "type=LC4,somaSide=R"],
    "LPLC2 + LC4 both": ["type=LPLC2", "type=LC4"],
}
TARGET = "type=DNp01"

_net = None


def init_worker(cache: str, scale: float) -> None:
    global _net
    _net = load(Path(cache), "numba")
    _net.set_weight_scale(scale)


def run_job(job: tuple) -> tuple:
    name, seed = job
    driven = np.unique(np.concatenate([_net.group(q) for q in CONDITIONS[name]]))
    rate, _ = drive(_net, driven, RATE_HZ, None, 1.0, seed, mode="spike")
    gf = _net.group(TARGET)
    side = _net.columns["somaSide"][gf]
    return job, {s: float(rate[gf[side == s]].mean()) for s in ("L", "R")}, float(rate.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(CACHE_DIR / "network_A.npz"))
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    scale = json.loads(FLY_MODEL.read_text(encoding="utf-8"))["weight_scale"]
    rep = Report()
    start = time.perf_counter()
    jobs = [(c, s) for c in CONDITIONS for s in SEEDS]
    with ProcessPoolExecutor(args.jobs, initializer=init_worker,
                             initargs=(args.cache, scale)) as pool:
        results = {job: (gf, net_hz) for job, gf, net_hz in pool.map(run_job, jobs)}
    rep.say(f"T2 looming check: DNp01 (giant fiber) rate, weight scale {scale:g}, spike input "
            f"{RATE_HZ:g} Hz, 1 s, Poisson seeds {SEEDS[0]}-{SEEDS[-1]}")
    rep.say(f"  {'condition':18s} {'DNp01 L mean':>13s} {'sd':>7s} {'DNp01 R mean':>13s} "
            f"{'sd':>7s} {'net_Hz':>8s}")
    for c in CONDITIONS:
        left = np.array([results[(c, s)][0]["L"] for s in SEEDS])
        right = np.array([results[(c, s)][0]["R"] for s in SEEDS])
        net_hz = np.array([results[(c, s)][1] for s in SEEDS])
        rep.say(f"  {c:18s} {left.mean():13.2f} {left.std(ddof=1):7.2f} {right.mean():13.2f} "
                f"{right.std(ddof=1):7.2f} {net_hz.mean():8.3f}")
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
