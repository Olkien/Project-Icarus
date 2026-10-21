import math
from typing import ClassVar

import gymnasium as gym
import numpy as np

from env.engine import Sim
from env.params import load_physics
from env.scenario import load_scenario
from env.sensors import obs_to_vector

INF = np.inf
SEED_LIMIT = 2**31 - 1


def tick_reward(range_prev: float, range_now: float, hits: int, action: dict, oob: bool) -> float:
    reward = 10.0 * hits - 0.01
    if hits == 0:
        reward += 0.05 * (range_prev - range_now)
    reward -= 0.002 * (action["thrust"] ** 2 + action["yaw_torque"] ** 2)
    if oob:
        reward -= 20.0
    return reward


class FlightEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, scenario: dict | str = "s1_still", physics: dict | None = None):
        self.scenario = load_scenario(scenario) if isinstance(scenario, str) else scenario
        self.sim = Sim(physics or load_physics())
        self.observation_space = gym.spaces.Box(
            low=np.array([-math.pi, 0.0, -INF, -INF, -INF, -INF, -INF], dtype=np.float32),
            high=np.array([math.pi, INF, INF, INF, INF, INF, INF], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.seed_used = 0

    def vector(self, obs: dict) -> np.ndarray:
        return np.array(obs_to_vector(obs), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, SEED_LIMIT))
        self.seed_used = seed
        obs = self.sim.reset(seed, self.scenario)
        return self.vector(obs), {"seed": seed, "obs": obs}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        act = {"thrust": float(np.clip(a[0], 0.0, 1.0)),
               "yaw_torque": float(np.clip(a[1], -1.0, 1.0))}
        range_prev = self.sim.true_range()
        obs, events = self.sim.step_tick(act)
        hits = events.count("hit")
        oob = "oob" in events
        reward = tick_reward(range_prev, self.sim.true_range(), hits, act, oob)
        terminated = self.sim.success or oob
        truncated = "timeout" in events
        info = {"events": events, "obs": obs, **self.sim.snapshot()}
        return self.vector(obs), reward, terminated, truncated, info
