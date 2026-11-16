"""Train a SAC controller for the fly flight task. Run from the repo root: python -m ml.SAC"""
import csv
import math
from collections import deque

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from env.engine import Sim, run_episode
from env.gym_env import FlightEnv
from env.metrics import compute_metrics, tuning_cost
from env.params import load_physics
from env.scenario import load_scenario
from env.sensors import obs_to_vector

# Settings
TOTAL_STEPS = 1_500_000     # change if you want more or less training
EPISODE_BUDGET = 2400       # budget so cross checking works with fly and PID (PID used 4 scenarios x 4 seeds x 150 evaluations)
VALIDATE_EVERY = 100_000    # steps between validation checks
SEED = 0
SCENARIOS = ["s1_still", "s2_crosswind", "s3_gust", "s4_turbulent"]
TRAIN_SEEDS = range(0, 40)
VAL_SEEDS = range(40, 50)
HISTORY = (0, 1, 2, 4, 8, 16, 32)   # how many ticks back the network looks (32 ticks = 160 ms), not using all 32 previous as that would take too long to train

SAC_SETTINGS = dict(learning_rate=3e-4, buffer_size=1_000_000, learning_starts=10_000,
                    batch_size=256, tau=0.005, gamma=0.99, train_freq=16, gradient_steps=16,
                    ent_coef="auto", use_sde=True, sde_sample_freq=16,
                    policy_kwargs=dict(net_arch=[256, 256], log_std_init=-2.0))

"""
Learning rate of 3e-4 is a standard value
Buffer size of 1 mil so nearly the whole run fits into it so rare gusts are kept
Learning starts at 10k so the buffer can be slightly filled before training
Batch size of 256 is standard
Tau of 0.005 is standard
Gamma of 0.99, which means the controller looks about half a second ahead. Thats enough to see the effects of its actions, like turning or slowing down, and looking further ahead wasnt needed since the reward already tells it when its getting closer to the target.
Train frequency and gradient steps of 16, to extend noise since 5ms too short, and gradient steps is same to ensure 1 update per tick
End coefficient set to auto so SAC tunes entropy itself otherwise a very sensitive option
Use sde set to true and sde sample frequency to 16 again since delay 5ms nosie would be smoothed over so we need 16, and true because we are using state dependent exploration
Net arch with 2 layers of 256 which is standard, basically plenty while still fast
Initial log spread at -2 to counteract the noise smoothing
"""

def features(obs, last_action):
    """One tick of input first scaled and last action turned into one frame"""
    bearing, rng, vf, vl, yaw_rate, af, al = obs
    return np.array([math.sin(bearing), math.cos(bearing), rng / 300, #sin and cos to avoid + and - problems for target right behind
                     vf / 300, vl / 300, yaw_rate / 10, af / 300, al / 300,
                     last_action[0], last_action[1]], dtype=np.float32) #divide by typical size to scale to a number roughly between -1 and 1


class History:
    """Remembers last 33 ticks and combines 0, 1, 2, 4, 8, 16 and 32 into one vector justsification at top"""
    def reset(self, obs):
        self.frames = deque([features(obs, (0.0, 0.0))] * 33, maxlen=33)
        return self.get()

    def push(self, obs, action):
        self.frames.append(features(obs, action))
        return self.get()

    def get(self):
        return np.concatenate([self.frames[-1 - k] for k in HISTORY])


class TrainEnv(gym.Env):
    """For every episode a random seed and random scenario to prevent overfitting"""
    def __init__(self):
        self.envs = [FlightEnv(name) for name in SCENARIOS]
        self.action_space = self.envs[0].action_space
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (10 * len(HISTORY),), np.float32)
        self.history = History()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env = self.envs[self.np_random.integers(len(self.envs))]
        obs, info = self.env.reset(seed=int(self.np_random.choice(TRAIN_SEEDS)))
        return self.history.reset(obs), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.history.push(obs, action), reward, terminated, truncated, info


class SACController:
    """Lets a trained model fly through run_episode, exactly like PID and the fly."""
    def __init__(self, model):
        self.model = model
        self.history = History()

    def reset(self, seed, scenario):
        self.action = None

    def act(self, obs, t):
        obs = obs_to_vector(obs)
        x = self.history.reset(obs) if self.action is None else self.history.push(obs, self.action)
        action, _ = self.model.predict(x, deterministic=True) #deterministic true so we don't act on exploration noise
        self.action = np.clip(action, [0.0, -1.0], [1.0, 1.0])
        return {"thrust": float(self.action[0]), "yaw_torque": float(self.action[1])}

    def info(self):  # run_episode asks for extra info but don't have any 
        return {}


def score(model, scenarios, seeds):
    """Flies the model through every scenario and seed pair and then returns a row of metrics for each."""
    sim, controller, rows = Sim(load_physics()), SACController(model), []
    for scenario in scenarios:
        for seed in seeds:
            result = run_episode(sim, controller, seed, scenario)
            metrics = compute_metrics(result.ticks, scenario, sim.tick_s, result.act_s_total)
            rows.append({"scenario": scenario["name"], "controller": "sac", "seed": seed, **metrics})
    return rows


class Validation(BaseCallback):
    """Counts episodes, stops at the budget, and keeps the model with the best validation cost."""
    def __init__(self):
        super().__init__()
        self.train_episodes = self.val_episodes = 0
        self.best_cost = math.inf
        self.log = []

    def _on_step(self):#counts validation and training episodes and makes sure they stay on budget
        self.train_episodes += sum("episode" in info for info in self.locals["infos"])
        if self.num_timesteps % VALIDATE_EVERY == 0:
            self.validate()
        return self.train_episodes + self.val_episodes < EPISODE_BUDGET

    def validate(self):#uses current policy on s1 to s4 and keeps the lowest tuning cost one
        rows = score(self.model, [load_scenario(s) for s in SCENARIOS], VAL_SEEDS)
        self.val_episodes += len(rows)
        cost = np.mean([tuning_cost(r) for r in rows])
        success = np.mean([r["episode_success"] for r in rows])
        self.log.append({"step": self.num_timesteps, "cost": cost, "success": success,
                         "episodes": self.train_episodes + self.val_episodes})
        if cost < self.best_cost:
            self.best_cost = cost
            self.model.save("ml/models/sac_best")
        print(f"step {self.num_timesteps:>8}   cost {cost:.3f}   success {success:.2f}"
              f"   episodes used {self.train_episodes + self.val_episodes}")


def main():
    model = SAC("MlpPolicy", Monitor(TrainEnv()), seed=SEED, device="cpu", **SAC_SETTINGS)
    callback = Validation()
    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)
    callback.validate()
    model.save("ml/models/sac_final")
    with open("results/sac_validation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=callback.log[0].keys())
        writer.writeheader()
        writer.writerows(callback.log)
    print(f"Done. Best validation cost {callback.best_cost:.3f}, saved to ml/models/sac_best.zip")


if __name__ == "__main__":
    main()
