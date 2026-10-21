import math
from collections import deque

import numpy as np

CHANNEL_LAYOUT = (
    "bearing", "range", "vel_body", "vel_body", "yaw_rate", "air_rel", "air_rel",
)
NOISE_BLOCK = 1024


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class Sensors:
    def __init__(self, config: dict, noise_multiplier: float, rng: np.random.Generator):
        self.sigmas = [config[c]["sigma"] * noise_multiplier for c in CHANNEL_LAYOUT]
        self.delays = [config[c]["delay_ticks"] for c in CHANNEL_LAYOUT]
        self.rng = rng
        self.buffers: list[deque] = []
        self.noise: list[list[float]] = []
        self.noise_i = 0

    def reset(self, sample: list[float]) -> None:
        self.buffers = [deque([v] * (d + 1), maxlen=d + 1) for v, d in zip(sample, self.delays)]

    def next_noise(self) -> list[float]:
        if self.noise_i >= len(self.noise):
            self.noise = self.rng.standard_normal((NOISE_BLOCK, len(CHANNEL_LAYOUT))).tolist()
            self.noise_i = 0
        row = self.noise[self.noise_i]
        self.noise_i += 1
        return row

    def observe(self, sample: list[float]) -> list[float]:
        noise = self.next_noise()
        out = []
        for buf, value, sigma, n in zip(self.buffers, sample, self.sigmas, noise):
            buf.append(value)
            out.append(buf[0] + sigma * n)
        out[0] = wrap_angle(out[0])
        out[1] = max(0.0, out[1])
        return out


def to_obs_dict(values: list[float]) -> dict:
    return {
        "bearing": values[0],
        "range": values[1],
        "vel_body": [values[2], values[3]],
        "yaw_rate": values[4],
        "air_rel": [values[5], values[6]],
    }


def obs_to_vector(obs: dict) -> list[float]:
    return [
        obs["bearing"], obs["range"], obs["vel_body"][0], obs["vel_body"][1], obs["yaw_rate"],
        obs["air_rel"][0], obs["air_rel"][1],
    ]
