import argparse
import time
from pathlib import Path

import numpy as np

from analysis.fly_pool import NETWORKS, fly_model, run_all
from fly.bench import peak_memory_mb
from fly.check_load import REPO_ROOT, Report
from fly.connectome import ConnectomeNetwork

OUT_PATH = REPO_ROOT / "results" / "thrust_candidates.txt"
RATES_HZ = (10.0, 20.0, 40.0, 60.0)
SEEDS = tuple(range(20))
INPUTS = {"LC10 both": "type^=LC10", "mechanosensory both": "class=mechanosensory"}
TOP_DN = 25
TOP_MN = 15
SAME_RATIO = 0.5


def units(net: ConnectomeNetwork) -> list[tuple[str, str, str, np.ndarray]]:
    c = net.columns
    out = []
    for kind, superclass in (("DN", "descending_neuron"), ("MN", "vnc_motor")):
        base = np.flatnonzero((c["superclass"] == superclass) & np.isin(c["somaSide"], ["L", "R"]))
        for t, s in sorted({(c["type"][i], c["somaSide"][i]) for i in base.tolist()}):
            ids = base[(c["type"][base] == t) & (c["somaSide"][base] == s)]
            out.append((kind, t or "-", s, ids))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    scale = fly_model()["weight_scale"]
    net = ConnectomeNetwork(NETWORKS["orig"])
    unit_list = units(net)
    unit_ids = [u[3] for u in unit_list]
    jobs = [{"kind": "units", "net": "orig", "scale": scale, "seed": s, "units": unit_ids,
             "drive": [(net.group(q), r)]}
            for q in INPUTS.values() for r in RATES_HZ for s in SEEDS]
    start = time.perf_counter()
    results = run_all(jobs, args.workers)
    rep = Report()
    rep.say(f"thrust candidates: weight scale {scale:g}, spike input, 1 s, seeds 0-19; both sides "
            f"driven together; increase = rate at 60 Hz input minus rate at 10 Hz input")
    rep.say(f"same on both sides = both sides increase and the smaller is at least "
            f"{SAME_RATIO:g} x the larger")
    i = 0
    for label, query in INPUTS.items():
        per_rate = {}
        for r in RATES_HZ:
            runs = results[i:i + len(SEEDS)]
            i += len(SEEDS)
            per_rate[r] = np.array([u for u, _ in runs])
        rep.say(f"== {label} ({query}): network mean " + ", ".join(
            f"{r:g} Hz {np.mean([n for _, n in results[i - len(SEEDS) * (len(RATES_HZ) - k): i - len(SEEDS) * (len(RATES_HZ) - k - 1)]]):.3f}"
            for k, r in enumerate(RATES_HZ)))
        rise = per_rate[RATES_HZ[-1]] - per_rate[RATES_HZ[0]]
        by_type: dict[tuple[str, str], dict[str, int]] = {}
        for j, (kind, t, s, _) in enumerate(unit_list):
            by_type.setdefault((kind, t), {})[s] = j
        rows = []
        for (kind, t), sides in by_type.items():
            incs = {s: float(rise[:, j].mean()) for s, j in sides.items()}
            sds = {s: float(rise[:, j].std(ddof=1)) for s, j in sides.items()}
            mean_inc = float(np.mean(list(incs.values())))
            if len(incs) == 2:
                lo_, hi_ = sorted(incs.values())
                same = "yes" if lo_ > 0 and lo_ >= SAME_RATIO * hi_ else "no"
            else:
                same = "one side only"
            curve = " ".join(f"{np.mean([per_rate[r][:, j].mean() for j in sides.values()]):.1f}"
                             for r in RATES_HZ)
            rows.append((kind, mean_inc, t, incs, sds, same, curve))
        for kind, top in (("DN", TOP_DN), ("MN", TOP_MN)):
            rep.say(f"  {kind} types by mean increase (L and R separately; curve = mean Hz at "
                    + "/".join(f"{r:g}" for r in RATES_HZ) + " Hz input)")
            rep.say(f"    {'type':28s} {'inc L':>8s} {'sd L':>6s} {'inc R':>8s} {'sd R':>6s}  same  curve")
            for k, mean_inc, t, incs, sds, same, curve in sorted(
                    (x for x in rows if x[0] == kind), key=lambda x: -x[1])[:top]:
                rep.say(f"    {t:28s} {incs.get('L', float('nan')):8.2f} {sds.get('L', float('nan')):6.2f} "
                        f"{incs.get('R', float('nan')):8.2f} {sds.get('R', float('nan')):6.2f}  "
                        f"{same:4s}  {curve}")
    rep.say(f"peak memory (main) {peak_memory_mb():.0f} MB, total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
