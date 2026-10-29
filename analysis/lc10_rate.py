import argparse
import time
from pathlib import Path

import numpy as np

from analysis.fly_pool import NETWORKS, WINDOW_MS, fly_model, run_all, side_ids
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import ConnectomeNetwork

OUT_PATH = REPO_ROOT / "results" / "lc10_rate.txt"
RATES_HZ = (10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0)
SEEDS = tuple(range(20))
MAX_HZ = 200.0
MIN_PER_WINDOW = 3.0
LC10 = "type^=LC10"
DNA02 = "type=DNa02"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    scale = fly_model()["weight_scale"]
    net = ConnectomeNetwork(NETWORKS["orig"])
    lc10 = side_ids(net, LC10)
    dna02 = side_ids(net, DNA02)
    jobs = [{"kind": "windows", "net": "orig", "scale": scale, "seed": s,
             "drive": [(lc10[side], r)], "groups": [dna02["L"], dna02["R"]]}
            for side in ("L", "R") for r in RATES_HZ for s in SEEDS]
    start = time.perf_counter()
    results = run_all(jobs, args.workers)
    rep = Report()
    rep.say(f"LC10 rate response: weight scale {scale:g}, spike input, 1 s, seeds 0-19; DNa02 L "
            f"{dna02['L'].size} and R {dna02['R'].size} neurons; {WINDOW_MS:g} ms windows")
    rep.say("  drive    rate   DNa02 L Hz (sd)   DNa02 R Hz (sd)   per window L   R   "
            "driven side ok")
    i = 0
    ok_rates = {"L": [], "R": []}
    for side in ("L", "R"):
        for r in RATES_HZ:
            runs = results[i:i + len(SEEDS)]
            i += len(SEEDS)
            hz = np.array([w.sum(axis=0) for w in runs], dtype=float)
            per_win = np.array([w.mean(axis=0) for w in runs])
            own = 0 if side == "L" else 1
            ok = hz[:, own].mean() < MAX_HZ and per_win[:, own].mean() >= MIN_PER_WINDOW
            if ok:
                ok_rates[side].append(r)
            rep.say(f"  LC10 {side} {r:6.0f}   {hz[:, 0].mean():8.2f} ({hz[:, 0].std(ddof=1):5.2f})"
                    f"   {hz[:, 1].mean():8.2f} ({hz[:, 1].std(ddof=1):5.2f})   "
                    f"{per_win[:, 0].mean():9.2f} {per_win[:, 1].mean():5.2f}   "
                    f"{'yes' if ok else 'no'}")
    both = sorted(set(ok_rates["L"]) & set(ok_rates["R"]))
    rep.say(f"rule: driven-side DNa02 under {MAX_HZ:g} Hz and at least {MIN_PER_WINDOW:g} spikes per "
            f"{WINDOW_MS:g} ms window (mean over seeds)")
    rep.say(f"  left LC10 rates that meet it: {ok_rates['L'] or 'none'}")
    rep.say(f"  right LC10 rates that meet it: {ok_rates['R'] or 'none'}")
    rep.say(f"  both sides: {both or 'none'}"
            + (f" (range {both[0]:g} to {both[-1]:g} Hz)" if both else ""))
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
