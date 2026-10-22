import hashlib
import json
import sys
from pathlib import Path

import pytest

from controllers.pid import PID
from env.engine import Sim, run_episode
from env.params import CONFIG_DIR, load_physics
from env.scenario import SCENARIO_DIR

DATA_DIR = Path(__file__).resolve().parent / "data"
NAMES = ("s1_still", "s2_crosswind", "s3_gust", "s4_turbulent")
SEED = 5
SAMPLE_EVERY = 100


def trajectory_hash(ticks: list[dict]) -> str:
    text = json.dumps(ticks, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(scenario: dict, params: dict) -> list[dict]:
    return run_episode(Sim(load_physics()), PID(params), SEED, scenario).ticks


def write_golden() -> None:
    params = json.loads((CONFIG_DIR / "pid_default.json").read_text(encoding="utf-8"))
    DATA_DIR.mkdir(exist_ok=True)
    for name in NAMES:
        scenario = json.loads((SCENARIO_DIR / f"{name}.json").read_text(encoding="utf-8"))
        ticks = run(scenario, params)
        golden = {"scenario": scenario, "params": params, "seed": SEED, "ticks": len(ticks),
                  "sha256": trajectory_hash(ticks), "samples": ticks[::SAMPLE_EVERY]}
        path = DATA_DIR / f"golden_{name}.json"
        path.write_text(json.dumps(golden) + "\n", encoding="utf-8", newline="\n")


@pytest.mark.parametrize("name", NAMES)
def test_old_scenarios_reproduce_exactly(name):
    golden = json.loads((DATA_DIR / f"golden_{name}.json").read_text(encoding="utf-8"))
    ticks = run(golden["scenario"], golden["params"])
    assert len(ticks) == golden["ticks"]
    assert ticks[::SAMPLE_EVERY] == golden["samples"]
    assert trajectory_hash(ticks) == golden["sha256"]


if __name__ == "__main__" and "--write" in sys.argv:
    write_golden()
