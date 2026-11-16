"""Score a trained SAC model and compare it with PID. Run from the repo root: python -m ml.evaluate_sac"""
import csv
import math

import numpy as np
from scipy.stats import binomtest, wilcoxon
from stable_baselines3 import SAC

from env.scenario import expand_noise_sweep, load_scenario
from ml.SAC import SCENARIOS, score

# Settings
MODEL = "ml/models/sac_best.zip"
SEEDS = range(1000, 1050)                     # scoring seeds, same as PID
HELD_OUT = False                              # True = score S5 + S6. Only once, at the very end!
PID_RESULTS = "results/eval_pid_v2_scoring.csv"


def wilson(k, n, z=1.96): #better than standard interval which breaks near 100 and 0 
    """95% confidence interval for k successes out of n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(x, n=10_000):
    """95% confidence interval for the mean of x, by resampling."""
    means = np.random.default_rng(0).choice(x, (n, len(x))).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def summarize(rows):
    """One line of results per scenario."""
    print(f"\n{'scenario':<18}{'success':>9}{'95% CI':>15}{'hit rate':>10}{'timeouts':>10}"
          f"{'oob':>5}{'time/target':>13}{'path eff':>10}{'cross-track':>13}{'jitter':>8}")
    for name in dict.fromkeys(r["scenario"] for r in rows):
        g = [r for r in rows if r["scenario"] == name]
        n, wins = len(g), sum(r["episode_success"] for r in g)
        oob = sum(r["out_of_bounds"] for r in g)
        lo, hi = wilson(wins, n)
        hit_rate = sum(r["targets_hit"] for r in g) / (5 * n)
        time = np.median([r["time_per_target_miss6_s"] for r in g])
        eff = np.median([r["path_efficiency"] or 0.0 for r in g])
        cross = np.median([r["rms_cross_track_mm"] for r in g])
        jitter = np.mean([r["jitter_yaw"] for r in g])
        print(f"{name:<18}{wins / n:>9.2f}{f'{lo:.2f}-{hi:.2f}':>15}{hit_rate:>10.2f}"
              f"{n - wins - oob:>10}{oob:>5}{time:>12.2f}s{eff:>10.2f}{cross:>11.1f}mm"
              f"{jitter:>8.3f}")


def compare_with_pid(rows):
    """Pair SAC and PID on the same scenario + seed and test the differences."""
    with open(PID_RESULTS) as f:
        pid = {(r["scenario"], int(r["seed"])): r for r in csv.DictReader(f)}
    print(f"\n{'scenario':<18}{'SAC - PID time':>16}{'95% CI':>18}{'SAC faster':>12}"
          f"{'wilcoxon p':>12}{'only SAC ok':>13}{'only PID ok':>13}{'mcnemar p':>11}")
    for name in dict.fromkeys(r["scenario"] for r in rows):
        pairs = [(r, pid[(name, r["seed"])]) for r in rows
                 if r["scenario"] == name and (name, r["seed"]) in pid]
        if not pairs:
            continue
        diff = np.array([s["time_per_target_miss6_s"] - float(p["time_per_target_miss6_s"])
                         for s, p in pairs])
        lo, hi = bootstrap_ci(diff)
        p_time = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        only_sac = sum(s["episode_success"] and p["episode_success"] == "False" for s, p in pairs)
        only_pid = sum(not s["episode_success"] and p["episode_success"] == "True" for s, p in pairs)
        disagree = only_sac + only_pid
        p_success = binomtest(only_sac, disagree).pvalue if disagree else 1.0
        print(f"{name:<18}{diff.mean():>15.2f}s{f'{lo:.2f} to {hi:.2f}':>18}"
              f"{np.mean(diff < 0):>12.0%}{p_time:>12.3f}{only_sac:>13}{only_pid:>13}"
              f"{p_success:>11.3f}")


def main():
    names = ["s5_noise_sweep", "s6_unseen_wind"] if HELD_OUT else SCENARIOS
    scenarios = [level for name in names for level in expand_noise_sweep(load_scenario(name))]
    rows = score(SAC.load(MODEL, device="cpu"), scenarios, SEEDS)

    out = "results/eval_sac_heldout.csv" if HELD_OUT else "results/eval_sac_scoring.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved every episode to {out}")

    summarize(rows)
    if not HELD_OUT:
        compare_with_pid(rows)


if __name__ == "__main__":
    main()
