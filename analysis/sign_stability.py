import argparse
import json
import time
from pathlib import Path

from analysis.drive_scan import load
from analysis.shuffled_stability import summary, test_at
from analysis.stability_scan import FLY_MODEL_PATH, HEADER, RULE, input_sets
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import CACHE_DIR

OUT_PATH = REPO_ROOT / "results" / "sign_stability.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(CACHE_DIR / "network_A.npz"))
    parser.add_argument("--glu-cache", default=str(CACHE_DIR / "network_A_glu.npz"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    chosen = json.loads(FLY_MODEL_PATH.read_text(encoding="utf-8"))["weight_scale"]
    rep = Report()
    start = time.perf_counter()
    rep.say(f"sign sensitivity at weight scale {chosen:g}, Poisson seed {args.seed}: glutamate "
            f"-1 (original) against glutamate +1 (--glu-excitatory)")
    rep.say(RULE)
    rep.say(HEADER + "  network")
    lines = []
    for label, cache in (("original", args.cache), ("glu +1", args.glu_cache)):
        net = load(Path(cache), "numba")
        signs = {int(k): int((net.sign == k).sum()) for k in (-1, 0, 1)}
        results = test_at(rep, label, net, input_sets(net), chosen, args.seed)
        lines.append(f"  {label:9s} signs {signs}: {summary(results)}")
        del net
    rep.say("== Summary")
    for line in lines:
        rep.say(line)
    rep.say(f"peak memory {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
