import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from analysis.lockin_check import TICK_S, lockin_events, read_spikes
from analysis.run import default_params
from controllers import make_controller
from env.engine import Sim, run_episode
from env.params import load_physics
from env.runlog import run_path
from env.scenario import load_scenario
from fly.check_load import REPO_ROOT, Report

OUT_PATH = REPO_ROOT / "results" / "lockin_replay.txt"
FLIGHTS = (("fly_steer_only", "s1_still", 0), ("fly_real", "s3_gust", 4))
SCALES = (0.6, 0.55, 0.5, 0.45)
COMPARE_TICKS = 20
BEFORE_TICKS = 400
BIN_TICKS = 40
TOP_N = 30


def replay(job: tuple) -> dict:
    controller, name, seed, scale = job
    physics = load_physics()
    sim = Sim(physics)
    ctrl = make_controller(controller, default_params(controller))
    net = ctrl.net
    net.set_weight_scale(scale)
    fired: list[np.ndarray] = []
    original = net.step

    def recording_step() -> np.ndarray:
        spikes = original()
        fired.append(spikes.astype(np.int32))
        return spikes

    net.step = recording_step
    try:
        result = run_episode(sim, ctrl, seed, load_scenario(name))
    finally:
        del net.step
        net.set_weight_scale(ctrl.weight_scale)
    per_tick = ctrl.steps_per_tick
    spikes = np.array([t["info"]["spikes_total"] for t in result.ticks])
    events = lockin_events(spikes)
    out = {"job": job, "ticks": len(spikes), "spikes": spikes, "events": events,
           "hits": sum(1 for t in result.ticks for e in t["events"] if "hit" in str(e)),
           "median": float(np.median(spikes))}
    if not events or scale != SCALES[0]:
        return out
    onset = round(events[0]["start_s"] / TICK_S)
    ids = {"dna02_L": ctrl.dna02_L, "dna02_R": ctrl.dna02_R, "dnp31": net.group("type=DNp31")}
    lookup = np.full(net.n, len(ids), dtype=np.int64)
    for i, group in enumerate(ids.values()):
        lookup[group] = i
    first = max(0, onset - BEFORE_TICKS)
    trace = []
    for start in range(first, onset, BIN_TICKS):
        stop = min(start + BIN_TICKS, onset)
        ticks = result.ticks[start:stop]
        chunk = np.concatenate(fired[start * per_tick:stop * per_tick])
        counts = np.bincount(lookup[chunk], minlength=len(ids) + 1)
        trace.append({"t_s": start * TICK_S, "bearing_deg": float(np.degrees(np.mean(
            [t["obs"]["bearing"] for t in ticks]))), "range_mm": float(np.mean(
                [t["obs"]["range"] for t in ticks])), "dna02_L": int(counts[0]),
            "dna02_R": int(counts[1]), "dnp31": int(counts[2]), "spikes": int(spikes[start:stop].sum())})
    width = min(COMPARE_TICKS, onset)
    before = np.bincount(np.concatenate(fired[(onset - width) * per_tick:onset * per_tick]),
                         minlength=net.n)
    after = np.bincount(np.concatenate(fired[onset * per_tick:(onset + COMPARE_TICKS) * per_tick]),
                        minlength=net.n)
    change = after - before
    top = np.argsort(-change, kind="stable")[:TOP_N]
    column = net.columns
    out["onset"] = onset
    out["trace"] = trace
    out["changed"] = int((change > 0).sum())
    out["top"] = [{"body_id": int(net.body_ids[i]), "type": str(column["type"][i]),
                   "superclass": str(column["superclass"][i]), "side": str(column["somaSide"][i]),
                   "nt": str(column["consensus_nt"][i]), "before": int(before[i]),
                   "after": int(after[i])} for i in top]
    superclass = np.array([str(v) for v in column["superclass"]])
    added = {s: int(change[superclass == s].sum()) for s in np.unique(superclass[change > 0])}
    out["by_superclass"] = sorted(added.items(), key=lambda kv: -kv[1])[:10]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    jobs = [(c, n, s, scale) for c, n, s in FLIGHTS for scale in SCALES]
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(replay, jobs))
    rep = Report()
    rep.say("lock-in replay, absolute rule (above 1500 spikes per tick for 0.5 s), weight scale "
            "set in this script only")
    rep.say("controller scenario seed scale ticks median_spikes max_spikes hits events start_s duration_s")
    for r in results:
        c, n, s, scale = r["job"]
        first = r["events"][0] if r["events"] else None
        rep.say(f"{c} {n} {s} {scale:g} {r['ticks']} {r['median']:.0f} {r['spikes'].max()} {r['hits']} "
                f"{len(r['events'])} " + (f"{first['start_s']:.3f} {first['duration_s']:.3f}"
                                          if first else "- -"))
    for r in results:
        c, n, s, scale = r["job"]
        if scale == SCALES[0]:
            logged = read_spikes(run_path(n, c, s))
            rep.say(f"{c} {n} seed {s}: replay equals the run file tick for tick: "
                    f"{np.array_equal(logged, r['spikes'])}")
    for r in results:
        if "onset" not in r:
            continue
        c, n, s, scale = r["job"]
        rep.say(f"== {c} {n} seed {s}, scale {scale:g}, onset tick {r['onset']}")
        rep.say("  before onset, 200 ms bins: t_s bearing_deg range_mm DNa02_L DNa02_R DNp31 spikes_per_tick")
        for b in r["trace"]:
            rep.say(f"  {b['t_s']:.3f} {b['bearing_deg']:8.1f} {b['range_mm']:7.1f} {b['dna02_L']:4d} "
                    f"{b['dna02_R']:4d} {b['dnp31']:4d} {b['spikes'] / BIN_TICKS:7.1f}")
        rep.say(f"  neurons with more spikes after than before: {r['changed']}")
        rep.say("  spikes added by superclass: " + ", ".join(f"{k} {v}" for k, v in r["by_superclass"]))
        rep.say(f"  top {TOP_N}, 100 ms after minus before: body_id type superclass side nt before after")
        for t in r["top"]:
            rep.say(f"  {t['body_id']} {t['type']} {t['superclass']} {t['side']} {t['nt']} "
                    f"{t['before']} {t['after']}")
    rep.say(f"total {time.perf_counter() - start:.0f} s")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
