import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from env.runlog import run_path
from env.sensors import wrap_angle
from fly.check_load import REPO_ROOT, Report

OUT_PATH = REPO_ROOT / "results" / "failure_breakdown.txt"
RUNS = [("s4_turbulent", s) for s in range(5)] + [("s2_crosswind", 1), ("s3_gust", 4)]
CONTROLLERS = ("fly_real", "pid")
AT_LIMIT = 1.0 - 1e-9


def load(path: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r["type"] == "tick"], rows[-1]["metrics"]


def true_bearing(t: dict) -> float:
    s, g = t["state"], t["target"]
    return wrap_angle(math.atan2(g["y"] - s["y"], g["x"] - s["x"]) - s["psi"])


def true_range(t: dict) -> float:
    return math.hypot(t["target"]["x"] - t["state"]["x"], t["target"]["y"] - t["state"]["y"])


def dropout_mask(ticks: list[dict]) -> np.ndarray:
    frozen = [False] + [a["obs"]["bearing"] == b["obs"]["bearing"] and a["obs"]["range"] == b["obs"]["range"]
                        for a, b in itertools.pairwise(ticks)]
    return np.array(frozen)


def ending(m: dict) -> str:
    if m["episode_success"]:
        return "success"
    return "out_of_bounds" if m["out_of_bounds"] else "timeout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()
    rep = Report()
    rep.say("failure breakdown from run files. closest = smallest true distance to the target held at "
            "the end. yaw limit = |yaw_torque| at 1. thrust shares at 1.0 and 0.0")
    rep.say("scenario controller seed end hits closest_mm mean|b|_rad yaw_limit% thrust1% thrust0% ticks")
    drop = {c: [] for c in CONTROLLERS}
    for name, seed in RUNS:
        for controller in CONTROLLERS:
            ticks, m = load(run_path(name, controller, seed))
            last = ticks[-1]["target"]["i"]
            closest = min(true_range(t) for t in ticks if t["target"]["i"] == last)
            bearing = np.array([abs(t["obs"]["bearing"]) for t in ticks])
            yaw = np.array([abs(t["action"]["yaw_torque"]) for t in ticks])
            thrust = np.array([t["action"]["thrust"] for t in ticks])
            rep.say(f"{name} {controller} {seed} {ending(m)} {m['targets_hit']} {closest:.2f} "
                    f"{bearing.mean():.3f} {100 * (yaw >= AT_LIMIT).mean():.1f} "
                    f"{100 * (thrust >= AT_LIMIT).mean():.1f} {100 * (thrust <= 1e-9).mean():.1f} "
                    f"{len(ticks)}")
            if name == "s4_turbulent":
                drop[controller].append((seed, ticks))
    rep.say("S4 dropouts: ticks where bearing and range both repeat the logged value of the tick before")
    rep.say("controller seed dropout_ticks ticks share% spells")
    for controller in CONTROLLERS:
        for seed, ticks in drop[controller]:
            mask = dropout_mask(ticks)
            spells = int(((mask[1:]) & ~mask[:-1]).sum())
            rep.say(f"{controller} {seed} {int(mask.sum())} {len(ticks)} {100 * mask.mean():.1f} {spells}")
    rep.say("S4 behaviour, pooled over seeds, dropout ticks against the rest")
    rep.say("controller part ticks mean_true|b|_rad bearing_seen_err_rad mean|yaw| mean_thrust speed_mm_s yaw_sign_right%")
    for controller in CONTROLLERS:
        ticks = [t for _, seed_ticks in drop[controller] for t in seed_ticks]
        mask = np.concatenate([dropout_mask(seed_ticks) for _, seed_ticks in drop[controller]])
        true_b = np.array([true_bearing(t) for t in ticks])
        seen = np.array([t["obs"]["bearing"] for t in ticks])
        err = np.abs(np.array([wrap_angle(float(a - b)) for a, b in zip(seen, true_b)]))
        yaw = np.array([t["action"]["yaw_torque"] for t in ticks])
        thrust = np.array([t["action"]["thrust"] for t in ticks])
        speed = np.array([math.hypot(t["state"]["vx"], t["state"]["vy"]) for t in ticks])
        right = np.sign(yaw) == np.sign(true_b)
        for part, m in (("dropout", mask), ("rest", ~mask)):
            rep.say(f"{controller} {part} {int(m.sum())} {np.abs(true_b[m]).mean():.3f} {err[m].mean():.3f} "
                    f"{np.abs(yaw[m]).mean():.3f} {thrust[m].mean():.3f} {speed[m].mean():.1f} "
                    f"{100 * right[m].mean():.1f}")
    rep.save(Path(args.out))


if __name__ == "__main__":
    main()
