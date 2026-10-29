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

OUT_PATH = REPO_ROOT / "results" / "direction_table.txt"
FLY_MODEL = CONFIG_DIR / "fly_model.json"
INPUTS = {
    "LC4": "type=LC4,somaSide={side}",
    "LPLC2": "type=LPLC2,somaSide={side}",
    "LC10": "type^=LC10,somaSide={side}",
    "mechanosensory": "class=mechanosensory,rootSide={side}",
}
RATES_HZ = (50.0, 100.0, 150.0)
SEEDS = tuple(range(20))
TOP = 30
SD_FLOOR_HZ = 1.0
RAW_PATH = CACHE_DIR / "direction_table_raw.npz"

_net = None
_units = None


def output_units(net) -> list[tuple[str, np.ndarray]]:
    c = net.columns
    units = []
    dn = np.flatnonzero((c["superclass"] == "descending_neuron") & np.isin(c["somaSide"], ["L", "R"]))
    for t, s in sorted({(c["type"][i], c["somaSide"][i]) for i in dn.tolist()}):
        ids = dn[(c["type"][dn] == t) & (c["somaSide"][dn] == s)]
        units.append((f"DN {t or '-'} {s}", ids))
    mn = np.flatnonzero(c["superclass"] == "vnc_motor")
    for nerve, s in sorted({(c["exitNerve"][i], c["somaSide"][i]) for i in mn.tolist()}):
        ids = mn[(c["exitNerve"][mn] == nerve) & (c["somaSide"][mn] == s)]
        units.append((f"MN nerve {nerve or '-'} {s or '-'}", ids))
    return units


def init_worker(cache: str, scale: float, stepper: str) -> None:
    global _net, _units
    _net = load(Path(cache), stepper)
    _net.set_weight_scale(scale)
    _units = output_units(_net)


def run_job(job: tuple) -> tuple:
    name, side, rate_hz, seed = job
    driven = _net.group(INPUTS[name].format(side=side))
    rate, _ = drive(_net, driven, rate_hz, None, 1.0, seed, mode="spike")
    return job, np.array([rate[ids].mean() for _, ids in _units]), driven.size


def effect_size(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = right.mean(axis=0) - left.mean(axis=0)
    pooled = np.sqrt((left.var(axis=0, ddof=1) + right.var(axis=0, ddof=1)) / 2.0)
    return diff, diff / np.maximum(pooled, SD_FLOOR_HZ)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(CACHE_DIR / "network_A.npz"))
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--stepper", choices=("numba", "numpy"), default="numba")
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--inputs", nargs="+", choices=list(INPUTS), default=list(INPUTS))
    parser.add_argument("--rates", nargs="+", type=float, default=list(RATES_HZ))
    args = parser.parse_args()
    scale = json.loads(FLY_MODEL.read_text(encoding="utf-8"))["weight_scale"]
    rep = Report()
    start = time.perf_counter()
    jobs = [(n, s, r, seed) for n in args.inputs for r in args.rates for s in ("L", "R")
            for seed in SEEDS]
    with ProcessPoolExecutor(args.jobs, initializer=init_worker,
                             initargs=(args.cache, scale, args.stepper)) as pool:
        results = {job: (rates, size) for job, rates, size in pool.map(run_job, jobs)}
    init_worker(args.cache, scale, args.stepper)
    names = [u for u, _ in _units]
    rep.say(f"direction table: {Path(args.cache).name}, weight scale {scale:g}, spike input, "
            f"1 s per run, Poisson seeds {SEEDS[0]}-{SEEDS[-1]}")
    rep.say("units: DN type x somaSide (sides only), vnc_motor exitNerve x somaSide; rate is "
            "the unit mean; diff = right drive minus left drive; d = diff / pooled SD over seeds")
    rep.say(f"pooled SD is floored at {SD_FLOOR_HZ:g} Hz (one spike in a 1 s run), so units with "
            f"the same count in every seed do not get an infinite d")
    raw = {}
    for name in args.inputs:
        for rate_hz in args.rates:
            left = np.array([results[(name, "L", rate_hz, s)][0] for s in SEEDS])
            right = np.array([results[(name, "R", rate_hz, s)][0] for s in SEEDS])
            sizes = (results[(name, "L", rate_hz, 0)][1], results[(name, "R", rate_hz, 0)][1])
            raw[f"{name}_{rate_hz:g}_L"] = left
            raw[f"{name}_{rate_hz:g}_R"] = right
            diff, d = effect_size(left, right)
            order = sorted(range(len(names)), key=lambda i: (-abs(d[i]), -abs(diff[i])))
            rep.say(f"== {name} at {rate_hz:g} Hz (left {sizes[0]} neurons, right {sizes[1]})")
            rep.say(f"  {'unit':34s} {'left_Hz':>9s} {'right_Hz':>9s} {'R-L':>9s} {'d':>8s}")
            shown = 0
            for i in order:
                if d[i] == 0 and diff[i] == 0:
                    continue
                d_text = f"{d[i]:.2f}"
                rep.say(f"  {names[i]:34s} {left[:, i].mean():9.3f} {right[:, i].mean():9.3f} "
                        f"{diff[i]:9.3f} {d_text:>8s}")
                shown += 1
                if shown == TOP:
                    break
            if shown == 0:
                rep.say("  no unit differs between left and right drive")
    np.savez(RAW_PATH, units=np.array(names), **raw)
    rep.say(f"raw per-seed rates saved to data/network_cache/{RAW_PATH.name} (git-ignored)")
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
