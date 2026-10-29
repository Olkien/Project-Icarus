import argparse
import json
import time
from pathlib import Path

from analysis.drive_scan import load
from analysis.stability_scan import (
    FLY_MODEL_PATH,
    HEADER,
    RATE_HZ,
    RULE,
    SCALES,
    input_sets,
    row,
    stability_run,
)
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import CACHE_DIR
from fly.shuffle import shuffle

OUT_PATH = REPO_ROOT / "results" / "shuffled_stability.txt"
SHUFFLE_SEEDS = (0, 1, 2)


def test_at(rep: Report, label: str, net, inputs: dict, scale: float, seed: int) -> dict:
    net.set_weight_scale(scale)
    results = {name: stability_run(net, ids, RATE_HZ, seed) for name, ids in inputs.items()}
    for name, r in results.items():
        rep.say(row(scale, f"{name}", r) + f"  {label}")
    return results


def summary(results: dict) -> str:
    passed = sum(r["pass"] for r in results.values())
    drive = max(r["drive_max_hz"] for r in results.values())
    off = max(r["off_max_after_hz"] for r in results.values())
    return f"{passed} of {len(results)} pass, worst drive bin {drive:.3f} Hz, worst off {off:.3f} Hz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(CACHE_DIR / "network_A.npz"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    chosen = json.loads(FLY_MODEL_PATH.read_text(encoding="utf-8"))["weight_scale"]
    rep = Report()
    start = time.perf_counter()
    net = load(Path(args.cache), "numba")
    inputs = input_sets(net)
    rep.say(f"shuffled stability at the chosen weight scale {chosen:g}, Poisson seed {args.seed}")
    rep.say(RULE)
    rep.say(HEADER + "  network")
    lines = [f"  original:  {summary(test_at(rep, 'original', net, inputs, chosen, args.seed))}"]
    net.set_weight_scale(1.0)
    from fly.lif_numba import NumbaLIFStepper

    for s in SHUFFLE_SEEDS:
        t = time.perf_counter()
        sh = shuffle(net, seed=s)
        sh.lif = NumbaLIFStepper(sh.weights, sh.lif_params)
        report = sh.shuffle_report
        rep.say(f"-- shuffle seed {s}: {report['swaps_per_edge']:.2f} swaps per edge, "
                f"{time.perf_counter() - t:.0f} s")
        results = test_at(rep, f"shuffle {s}", sh, inputs, chosen, args.seed)
        line = f"  shuffle {s}: {summary(results)}"
        if not all(r["pass"] for r in results.values()):
            stable = None
            for scale in sorted((x for x in SCALES if x < chosen), reverse=True):
                lower = test_at(rep, f"shuffle {s}", sh, inputs, scale, args.seed)
                if all(r["pass"] for r in lower.values()):
                    stable = scale
                    break
            line += (f"; largest stable scale {stable:g}" if stable is not None
                     else "; not stable at any tested scale")
        lines.append(line)
        del sh
    rep.say("== Summary")
    for line in lines:
        rep.say(line)
    rep.say(f"peak memory {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
