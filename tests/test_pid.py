import json
from pathlib import Path

import pytest

from controllers.pid import PID
from env.engine import Sim, run_episode
from tests.conftest import scenario

DEFAULT = Path(__file__).resolve().parent.parent / "configs" / "pid_default.json"


def obs(bearing=0.0, range_mm=200.0, fwd=0.0, yaw_rate=0.0) -> dict:
    return {"bearing": bearing, "range": range_mm, "vel_body": [fwd, 0.0], "yaw_rate": yaw_rate,
            "air_rel": [fwd, 0.0]}


def make() -> PID:
    pid = PID(json.loads(DEFAULT.read_text(encoding="utf-8")))
    pid.reset(0, scenario("s1_still"))
    return pid


def test_turns_toward_target():
    assert make().act(obs(bearing=0.5), 0.0)["yaw_torque"] > 0.0
    assert make().act(obs(bearing=-0.5), 0.0)["yaw_torque"] < 0.0


def test_yaw_damping():
    assert make().act(obs(yaw_rate=5.0), 0.0)["yaw_torque"] < 0.0


def test_no_forward_drive_when_target_behind():
    pid = make()
    pid.act(obs(bearing=2.5), 0.0)
    assert pid.v_des == 0.0


def test_speed_target_falls_near_target():
    pid = make()
    pid.act(obs(range_mm=500.0), 0.0)
    far = pid.v_des
    pid.act(obs(range_mm=5.0), 0.0)
    assert pid.v_des < far
    assert pid.v_des >= pid.params["v_min"]


def test_integrators_bounded():
    pid = make()
    for _ in range(10000):
        action = pid.act(obs(bearing=3.0, fwd=0.0, range_mm=500.0), 0.0)
        assert -1.0 <= action["yaw_torque"] <= 1.0
        assert 0.0 <= action["thrust"] <= 1.0
    assert abs(pid.i_bearing) <= pid.params["i_bearing_max"] + 1e-12
    assert abs(pid.i_speed) <= pid.params["i_speed_max"] + 1e-12


def test_info_is_error_terms():
    pid = make()
    pid.act(obs(bearing=0.2), 0.0)
    assert set(pid.info()) == {"e_bearing", "i_bearing", "v_des", "e_speed", "i_speed"}


def test_param_space_covers_tuned_gains():
    space = PID.param_space()
    for key in ("Kp", "Kd", "Ki", "Kr", "v_min", "v_cruise", "Kp_v", "Ki_v"):
        lo, hi = space[key]
        assert lo < hi


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_default_pid_finishes_still(physics, seed):
    pid = PID(json.loads(DEFAULT.read_text(encoding="utf-8")))
    result = run_episode(Sim(physics), pid, seed, scenario("s1_still"))
    assert result.success
