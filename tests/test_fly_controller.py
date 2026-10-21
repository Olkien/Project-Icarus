import math

import pytest

from controllers.fly import DEFAULT_MAPPING, Fly, load_mapping
from env.engine import Sim, run_episode
from fly.encoding import WindowDecoder, encode
from tests.conftest import scenario

MAPPING = load_mapping(DEFAULT_MAPPING)
ENC = MAPPING["encoding"]


def obs(bearing=0.0, range_mm=250.0, air=(0.0, 0.0)) -> dict:
    return {"bearing": bearing, "range": range_mm, "vel_body": [0.0, 0.0], "yaw_rate": 0.0,
            "air_rel": list(air)}


def test_encode_bearing_side():
    left = encode(obs(bearing=0.6), ENC)
    right = encode(obs(bearing=-0.6), ENC)
    ahead = encode(obs(bearing=0.0), ENC)
    assert left["vis_L"] > left["vis_R"] == ahead["vis_R"]
    assert right["vis_R"] > right["vis_L"] == ahead["vis_L"]
    assert ahead["vis_L"] == ahead["vis_R"]


def test_encode_looming():
    far = encode(obs(range_mm=400.0), ENC)
    near = encode(obs(range_mm=50.0), ENC)
    assert near["vis_L"] > far["vis_L"] and near["vis_R"] > far["vis_R"]


def test_encode_air_sides():
    rates = encode(obs(air=(200.0, -100.0)), ENC)
    assert rates["air_fwd"] > rates["air_back"]
    assert rates["air_R"] > rates["air_L"]


def test_window_decoder_uses_last_four_ticks():
    dec = {**MAPPING["decoding"], "yaw_sign": 1.0, "gain_yaw": 1.0, "norm_yaw_hz": 100.0}
    decoder = WindowDecoder(dec, {"out_L": 10, "out_R": 10, "fwd": 10}, 0.005)
    action = None
    for _ in range(10):
        action = decoder.push({"out_L": 0, "out_R": 2, "fwd": 0})
    assert action["yaw_torque"] == pytest.approx(2 / (10 * 0.005) / 100.0)
    for _ in range(4):
        action = decoder.push({"out_L": 0, "out_R": 0, "fwd": 0})
    assert action["yaw_torque"] == 0.0
    assert action["thrust"] == pytest.approx(dec["thrust_base"])


def mean_yaw(bearing: float, seed: int, ticks: int = 20) -> float:
    fly = Fly()
    fly.reset(seed, {})
    yaws = [fly.act(obs(bearing=bearing), 0.0)["yaw_torque"] for _ in range(ticks)]
    return sum(yaws[4:]) / len(yaws[4:])


@pytest.mark.parametrize("seed", range(20))
def test_left_target_turns_left_right_turns_right(seed):
    assert mean_yaw(0.8, seed) > 0.2
    assert mean_yaw(-0.8, seed) < -0.2


def test_closed_loop_turns_toward_target(physics, still):
    for side in (1.0, -1.0):
        sim = Sim(physics)
        obs_now = sim.reset(0, still)
        sim.psi = 0.0
        sim.targets[0] = (0.0, side * 200.0)
        fly = Fly()
        fly.reset(0, still)
        start = sim.true_bearing()
        for _ in range(30):
            obs_now, _ = sim.step_tick(fly.act(obs_now, sim.t))
        assert abs(sim.true_bearing()) < abs(start)
        assert math.copysign(1.0, sim.psi) == side


def test_info_logs_groups_and_spikes():
    fly = Fly()
    fly.reset(0, {})
    for _ in range(5):
        fly.act(obs(bearing=0.8, range_mm=100.0), 0.0)
    info = fly.info()
    assert set(info["groups"]) == {"vis_L", "vis_R", "air_fwd", "air_back", "air_L", "air_R",
                                   "out_L", "out_R", "fwd"}
    assert info["spikes_total"] >= sum(info["groups"].values()) > 0


def test_run_has_activity(physics):
    result = run_episode(Sim(physics), Fly(), 3, {**scenario("s3_gust"), "time_limit_s": 1.0})
    assert any(r["info"]["groups"]["out_L"] + r["info"]["groups"]["out_R"] > 0
               for r in result.ticks)
    assert all(isinstance(r["info"]["spikes_total"], int) for r in result.ticks)
