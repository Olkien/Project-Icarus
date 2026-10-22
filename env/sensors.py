import math
from collections import deque

import numpy as np

CHANNEL_LAYOUT = (
    "bearing", "range", "vel_body", "vel_body", "yaw_rate", "air_rel", "air_rel",
)
NOISE_BLOCK = 1024


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def scaled_delay(base_ticks: int, scale: float) -> int:
    return math.floor(base_ticks * scale + 0.5)


class Sensors:
    def __init__(
        self,
        config: dict,
        noise_multiplier: float,
        rng: np.random.Generator,
        delay_scale: float = 1.0,
        dropout: dict | None = None,
    ):
        self.sigmas = [config[c]["sigma"] * noise_multiplier for c in CHANNEL_LAYOUT]
        self.delays = [scaled_delay(config[c]["delay_ticks"], delay_scale) for c in CHANNEL_LAYOUT]
        self.rng = rng
        dropout = dropout or {}
        self.p_start = float(dropout.get("p_start", 0.0))
        self.drop_ticks = int(dropout.get("length_ticks", 0))
        self.buffers: list[deque] = []
        self.noise: list[list[float]] = []
        self.noise_i = 0

    def reset(self, sample: list[float]) -> None:
        self.buffers = [deque([v] * (d + 1), maxlen=d + 1) for v, d in zip(sample, self.delays)]
        self.frozen: tuple[float, float] | None = None
        self.drop_left = 0
        self.cooldown = False
        self.dropouts = 0
        self.eligible_ticks = 0
        self.in_dropout = False

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
        if self.p_start > 0.0:
            self.apply_dropout(out)
        return out

    def apply_dropout(self, out: list[float]) -> None:
        self.in_dropout = False
        if self.frozen is not None and self.drop_ticks > 0:
            if self.drop_left > 0:
                self.drop_left -= 1
                self.in_dropout = True
                self.cooldown = self.drop_left == 0
            elif self.cooldown:
                self.cooldown = False
            else:
                self.eligible_ticks += 1
                if self.rng.random() < self.p_start:
                    self.drop_left = self.drop_ticks - 1
                    self.dropouts += 1
                    self.in_dropout = True
                    self.cooldown = self.drop_left == 0
        if self.in_dropout:
            out[0], out[1] = self.frozen
        self.frozen = (out[0], out[1])


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
