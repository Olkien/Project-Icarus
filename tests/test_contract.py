import json
import math

import pytest

from controllers import REGISTRY, make_controller
from env.engine import Sim, run_episode
from tests.conftest import scenario

SHORT_S = 3.0


def short(name: str) -> dict:
    return {**scenario(name), "time_limit_s": SHORT_S}


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_controller_contract(name, physics):
    for scen, seed in (("s1_still", 0), ("s3_gust", 1), ("s4_turbulent", 2)):
        controller = make_controller(name)
        result = run_episode(Sim(physics), controller, seed, short(scen))
        assert result.ticks
        for record in result.ticks:
            action = record["action"]
            assert set(action) == {"thrust", "yaw_torque"}
            assert 0.0 <= action["thrust"] <= 1.0
            assert -1.0 <= action["yaw_torque"] <= 1.0
            assert not any(math.isnan(v) for v in action.values())
            json.dumps(record["info"], allow_nan=False)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_controller_raw_output(name, physics):
    controller = make_controller(name)
    sim = Sim(physics)
    obs = sim.reset(5, short("s2_crosswind"))
    controller.reset(5, sim.scenario)
    for _ in range(50):
        raw = controller.act(obs, sim.t)
        assert set(raw) == {"thrust", "yaw_torque"}
        assert all(isinstance(v, float) and not math.isnan(v) for v in raw.values())
        assert 0.0 <= raw["thrust"] <= 1.0 and -1.0 <= raw["yaw_torque"] <= 1.0
        obs, events = sim.step_tick(raw)
        if "done" in events:
            break


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_controller_deterministic(name, physics):
    a = run_episode(Sim(physics), make_controller(name), 8, short("s4_turbulent"))
    b = run_episode(Sim(physics), make_controller(name), 8, short("s4_turbulent"))
    assert a.ticks == b.ticks
