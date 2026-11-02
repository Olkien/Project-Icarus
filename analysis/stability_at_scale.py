import argparse
import time
from pathlib import Path

from analysis.fly_pool import fly_model
from analysis.scale_margin import inputs, report_scale, test_scale
from analysis.stability_scan import RATE_HZ, RULE
from fly.bench import peak_memory_mb
from fly.check_load import Report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    scale = fly_model()["weight_scale"]
    rep = Report()
    start = time.perf_counter()
    rep.say(f"stability at scale {scale:g}: spike input {RATE_HZ:g} Hz, 1 s on, 1 s off, seeds 0-19, six inputs")
    rep.say(RULE)
    report_scale(rep, "orig", scale, test_scale("orig", scale, inputs(), args.workers))
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
