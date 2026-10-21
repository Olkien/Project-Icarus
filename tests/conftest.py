from pathlib import Path

import pytest

from env.params import load_physics
from env.scenario import load_scenario

SCENARIOS = Path(__file__).resolve().parent.parent / "configs" / "scenarios"


@pytest.fixture
def physics() -> dict:
    return load_physics()


@pytest.fixture
def still() -> dict:
    return load_scenario(SCENARIOS / "s1_still.json")


def scenario(name: str) -> dict:
    return load_scenario(SCENARIOS / f"{name}.json")
