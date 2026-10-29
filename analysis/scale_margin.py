import argparse
import json
import time
from pathlib import Path

import numpy as np

from analysis.drive_scan import GROUPS
from analysis.fly_pool import NETWORKS, fly_model, run_all
from analysis.stability_scan import FLY_MODEL_PATH, RATE_HZ, RULE
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import ConnectomeNetwork

OUT_PATH = REPO_ROOT / "results" / "scale_margin.txt"
SEEDS = tuple(range(20))
SCALES = (0.5, 0.6, 0.7)
GLU_SCALES = (0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def inputs() -> dict[str, np.ndarray]:
    net = ConnectomeNetwork(NETWORKS["orig"])
    return {f"{name} {side}": net.group(f"{q},somaSide={side}")
            for name, q in GROUPS.items() for side in ("L", "R")}


def test_scale(net_key: str, scale: float, ids: dict, workers: int) -> dict:
    jobs = [{"kind": "stability", "net": net_key, "scale": scale, "ids": ids[name],
             "rate": RATE_HZ, "seed": s} for name in ids for s in SEEDS]
    results = run_all(jobs, workers)
    out = {}
    for i, name in enumerate(ids):
        out[name] = results[i * len(SEEDS):(i + 1) * len(SEEDS)]
    return out


def report_scale(rep: Report, label: str, scale: float, res: dict) -> bool:
    ok_all = True
    for name, runs in res.items():
        passed = sum(r["pass"] for r in runs)
        ok_all &= passed == len(runs)
        worst_drive = max(r["drive_max_hz"] for r in runs)
        worst_off = max(r["off_max_after_hz"] for r in runs)
        worst_ceiling = max(r["ceiling_share"] for r in runs)
        mean_drive = float(np.mean([r["drive_mean_hz"] for r in runs]))
        rep.say(f"  {label:5s} {scale:4.2f} {name:8s} {passed:2d}/{len(runs)}  mean drive "
                f"{mean_drive:7.3f}  worst drive bin {worst_drive:7.3f}  worst off "
                f"{worst_off:7.3f}  worst ceiling {100 * worst_ceiling:6.3f}%")
    rep.say(f"  {label} scale {scale:g}: {'PASS' if ok_all else 'fail'} (all six on all 20 seeds)")
    return ok_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    rep = Report()
    start = time.perf_counter()
    ids = inputs()
    rep.say(f"scale margin: spike input {RATE_HZ:g} Hz, 1 s on, 1 s off, seeds 0-19, six inputs")
    rep.say(RULE)
    passing = []
    for scale in SCALES:
        if report_scale(rep, "orig", scale, test_scale("orig", scale, ids, args.workers)):
            passing.append(scale)
    glu_pass = None
    for scale in GLU_SCALES:
        if report_scale(rep, "glu", scale, test_scale("glu", scale, ids, args.workers)):
            glu_pass = scale
            break
    model = fly_model()
    chosen = max(passing) if passing else None
    rep.say(f"original: largest scale passing on all 20 seeds: {chosen}")
    rep.say(f"glutamate +1: largest passing scale from 0.6 down: {glu_pass}")
    if chosen is not None:
        model["weight_scale"] = chosen
    model["weight_scale_glu"] = glu_pass
    model["stability_seeds"] = f"{SEEDS[0]}-{SEEDS[-1]}"
    FLY_MODEL_PATH.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8", newline="\n")
    rep.say(f"wrote configs/{FLY_MODEL_PATH.name}: weight_scale {model['weight_scale']}, "
            f"weight_scale_glu {glu_pass}")
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
