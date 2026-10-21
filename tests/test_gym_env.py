import math

import numpy as np
from gymnasium.utils.env_checker import check_env

from controllers.dummy import Straight
from env.engine import Sim, run_episode
from env.gym_env import FlightEnv, tick_reward
from tests.conftest import scenario


def scripted_actions(n: int) -> list[tuple[float, float]]:
    return [(0.5 + 0.4 * math.sin(0.03 * i), math.cos(0.05 * i)) for i in range(n)]


def test_check_env_passes():
    check_env(FlightEnv("s4_turbulent"), skip_render_check=True)


def test_gym_and_engine_same_trajectory(physics):
    s3 = scenario("s3_gust")
    env = FlightEnv(s3, physics)
    env.reset(seed=21)
    sim = Sim(physics)
    sim.reset(21, s3)
    for thrust, yaw in scripted_actions(800):
        a = np.array([thrust, yaw], dtype=np.float32)
        _, _, terminated, truncated, info = env.step(a)
        sim.step_tick({"thrust": float(a[0]), "yaw_torque": float(a[1])})
        assert info["state"] == sim.snapshot()["state"]
        if terminated or truncated:
            break


def test_straight_runs_through_both_paths(physics):
    s1 = scenario("s1_still")
    result = run_episode(Sim(physics), Straight(), 3, s1)
    env = FlightEnv(s1, physics)
    env.reset(seed=3)
    steps = 0
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(np.array([0.5, 0.0], dtype=np.float32))
        steps += 1
        done = terminated or truncated
    assert steps == len(result.ticks)
    last = result.ticks[-1]
    assert info["events"] == last["events"]


def test_obs_vector_order(physics):
    env = FlightEnv("s2_crosswind", physics)
    vec, info = env.reset(seed=4)
    obs = info["obs"]
    expected = [obs["bearing"], obs["range"], *obs["vel_body"], obs["yaw_rate"], *obs["air_rel"]]
    assert np.allclose(vec, np.array(expected, dtype=np.float32))


def test_reward_terms():
    act = {"thrust": 1.0, "yaw_torque": 0.0}
    assert math.isclose(tick_reward(100.0, 90.0, 0, act, False), 0.5 - 0.01 - 0.002)
    assert math.isclose(tick_reward(100.0, 300.0, 1, act, False), 10.0 - 0.01 - 0.002)
    assert math.isclose(tick_reward(100.0, 100.0, 0, act, True), -20.0 - 0.01 - 0.002)
