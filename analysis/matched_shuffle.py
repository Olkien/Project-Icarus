import argparse
import json
import time
from pathlib import Path

import numpy as np

from analysis.fly_pool import NETWORKS, fly_model, run_all, shuffled_path
from analysis.scale_margin import SEEDS, inputs
from analysis.stability_scan import FLY_MODEL_PATH, RATE_HZ
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import ConnectomeNetwork
from fly.shuffle import shuffle

OUT_PATH = REPO_ROOT / "results" / "matched_shuffle.txt"
SHUFFLE_SEEDS = (0, 1, 2)
STEP = 0.05
MAX_SCALE = 1.0
TOLERANCE = 0.10


def prepare(seeds=SHUFFLE_SEEDS) -> None:
    missing = [s for s in seeds if not shuffled_path(s).exists()]
    if not missing:
        return
    net = ConnectomeNetwork(NETWORKS["orig"])
    for s in missing:
        t = time.perf_counter()
        sh = shuffle(net, seed=s)
        sh.save(shuffled_path(s))
        print(f"saved {shuffled_path(s).name}: {sh.shuffle_report['swaps_per_edge']:.2f} swaps "
              f"per edge, {time.perf_counter() - t:.0f} s", flush=True)


def measure(net_key: str, scale: float, ids: dict, workers: int) -> tuple[float, bool, int]:
    jobs = [{"kind": "stability", "net": net_key, "scale": scale, "ids": ids[name],
             "rate": RATE_HZ, "seed": s} for name in ids for s in SEEDS]
    runs = run_all(jobs, workers)
    mean_drive = float(np.mean([r["drive_mean_hz"] for r in runs]))
    passed = sum(r["pass"] for r in runs)
    return mean_drive, passed == len(runs), passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    prepare()
    if args.prepare_only:
        return
    rep = Report()
    start = time.perf_counter()
    ids = inputs()
    model = fly_model()
    chosen = model["weight_scale"]
    target, _, passed = measure("orig", chosen, ids, args.workers)
    rep.say(f"matched shuffle scale: target is the original at weight scale {chosen:g}: mean drive "
            f"activity {target:.3f} Hz over six inputs x 20 seeds ({passed}/120 stable)")
    rep.say(f"match = within {100 * TOLERANCE:g}% of the target and stable on all 120 runs; "
            f"scales {STEP:g} to {MAX_SCALE:g} in steps of {STEP:g}")
    matched = {}
    for s in SHUFFLE_SEEDS:
        key = f"shuf{s}"
        rep.say(f"== shuffle seed {s}")
        best = None
        scale = STEP
        while scale <= MAX_SCALE + 1e-9:
            mean_drive, stable, passed = measure(key, round(scale, 2), ids, args.workers)
            ratio = mean_drive / target
            rep.say(f"  scale {scale:4.2f}: mean drive {mean_drive:7.3f} Hz ({100 * ratio:6.1f}% of "
                    f"target), stable {passed}/120")
            if stable and abs(ratio - 1.0) <= TOLERANCE and (
                    best is None or abs(ratio - 1.0) < abs(best[1] - 1.0)):
                best = (round(scale, 2), ratio)
            if ratio > 1.0 + TOLERANCE or not stable:
                break
            scale += STEP
        if best is None:
            rep.say(f"  no stable scale up to {MAX_SCALE:g} matches the target within "
                    f"{100 * TOLERANCE:g}%")
            matched[str(s)] = None
        else:
            rep.say(f"  matched scale {best[0]:g} ({100 * best[1]:.1f}% of target)")
            matched[str(s)] = best[0]
    model["shuffled_weight_scale"] = matched
    FLY_MODEL_PATH.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8", newline="\n")
    rep.say(f"wrote shuffled_weight_scale {matched} to configs/{FLY_MODEL_PATH.name}")
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
