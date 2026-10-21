import numpy as np
import pytest

from env.sensors import CHANNEL_LAYOUT, Sensors


def zero_noise(config: dict) -> dict:
    return {k: {**v, "sigma": 0.0} for k, v in config.items()}


def test_layout_is_seven_values():
    assert len(CHANNEL_LAYOUT) == 7


def test_delay_shows_step_exactly_n_ticks_later(physics):
    cfg = zero_noise(physics["sensors"])
    sensors = Sensors(cfg, 1.0, np.random.default_rng(0))
    n = len(CHANNEL_LAYOUT)
    sensors.reset([0.0] * n)
    step_tick = 10
    seen = []
    for k in range(1, 30):
        value = 1.0 if k >= step_tick else 0.0
        seen.append((k, sensors.observe([value] * n)))
    for i, channel in enumerate(CHANNEL_LAYOUT):
        delay = cfg[channel]["delay_ticks"]
        first = next(k for k, obs in seen if obs[i] == 1.0)
        assert first == step_tick + delay, channel


def test_noise_sigma(physics):
    cfg = physics["sensors"]
    base = [0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for mult in (1.0, 2.0):
        sensors = Sensors(cfg, mult, np.random.default_rng(3))
        sensors.reset(base)
        samples = np.array([sensors.observe(base) for _ in range(20000)])
        for i, channel in enumerate(CHANNEL_LAYOUT):
            expected = cfg[channel]["sigma"] * mult
            assert samples[:, i].std() == pytest.approx(expected, rel=0.1), channel


def test_bearing_wrapped_and_range_non_negative(physics):
    sensors = Sensors(physics["sensors"], 4.0, np.random.default_rng(0))
    near = [3.13, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    sensors.reset(near)
    for _ in range(2000):
        obs = sensors.observe(near)
        assert -np.pi <= obs[0] <= np.pi
        assert obs[1] >= 0.0
