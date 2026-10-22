import math

import numpy as np
import pytest

from controllers.pid import PID
from env.engine import STREAMS, Sim, make_generators, run_episode
from env.scenario import CAPS, expand_noise_sweep, validate_scenario
from env.sensors import CHANNEL_LAYOUT, Sensors, scaled_delay
from tests.conftest import scenario

ALL_ON = {
    "sensor_delay_scale": 2.0,
    "actuator_lag_scale": 2.0,
    "dropout": {"p_start": 0.03, "length_ticks": 8},
    "actuator_noise": {"thrust_sigma": 0.1, "torque_sigma": 0.15},
}
NO_DELAY = {c: {"sigma": 0.0, "delay_ticks": 0} for c in CHANNEL_LAYOUT}


def zero_noise(config: dict) -> dict:
    return {k: {**v, "sigma": 0.0} for k, v in config.items()}


def far_targets(sim: Sim) -> None:
    sim.targets = [(1.0e5, 1.0e5)] * len(sim.targets)


def test_actuator_stream_added_last():
    assert STREAMS[-1] == "actuators"
    assert len(make_generators(3)) == 6


@pytest.mark.parametrize("scale", [0.0, 0.5, 2.0, 3.0, 8.0])
def test_delay_scale_shifts_step(physics, scale):
    cfg = zero_noise(physics["sensors"])
    sensors = Sensors(cfg, 1.0, np.random.default_rng(0), delay_scale=scale)
    n = len(CHANNEL_LAYOUT)
    sensors.reset([0.0] * n)
    step_tick = 5
    seen = [(k, sensors.observe([1.0 if k >= step_tick else 0.0] * n)) for k in range(1, 120)]
    for i, channel in enumerate(CHANNEL_LAYOUT):
        delay = scaled_delay(cfg[channel]["delay_ticks"], scale)
        first = next(k for k, obs in seen if obs[i] == 1.0)
        assert first == step_tick + delay, channel
        if float(scale).is_integer():
            assert delay == int(scale) * cfg[channel]["delay_ticks"]


@pytest.mark.parametrize("scale", [1.0, 2.0, 4.0, 5.0])
def test_lag_scale_time_constant(physics, still, scale):
    sim = Sim(physics)
    sim.reset(0, {**still, "actuator_lag_scale": scale})
    far_targets(sim)
    thrust = 0.1
    t_cmd = physics["Tmax"] * thrust
    tau = physics["tau_a"] * scale
    steps = round(3 * tau / physics["dt_s"])
    sim.step_physics(steps, thrust, 0.0)
    measured = -steps * physics["dt_s"] / math.log(1.0 - sim.T / t_cmd)
    assert measured == pytest.approx(tau, rel=0.05)


def drive_sensors(p_start: float, length: int, ticks: int, seed: int = 0):
    sensors = Sensors(NO_DELAY, 1.0, np.random.default_rng(seed),
                      dropout={"p_start": p_start, "length_ticks": length})
    sensors.reset([0.0] * 7)
    out = []
    for k in range(ticks):
        out.append(sensors.observe([0.001 * k, 1000.0 - 0.01 * k, k, -k, 2 * k, 3 * k, -3 * k]))
    return sensors, np.array(out)


def freeze_runs(column: np.ndarray) -> list[int]:
    runs, length = [], 0
    for same in column[1:] == column[:-1]:
        if same:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    if length:
        runs.append(length)
    return runs


def test_dropout_freezes_exactly_length_ticks():
    sensors, out = drive_sensors(0.05, 7, 60_000)
    runs = freeze_runs(out[:, 0])
    assert runs and set(runs[:-1]) == {7}
    assert freeze_runs(out[:, 1])[:-1] == runs[:-1]
    for i in range(2, 7):
        assert freeze_runs(out[:, i]) == []
    assert sensors.dropouts == len(runs)


def test_dropout_start_rate():
    sensors, _ = drive_sensors(0.05, 5, 120_000, seed=4)
    assert sensors.dropouts / sensors.eligible_ticks == pytest.approx(0.05, rel=0.1)


def test_dropout_off_draws_nothing():
    rngs = []
    for dropout in ({"p_start": 0.0, "length_ticks": 10}, None):
        rng = np.random.default_rng(9)
        sensors = Sensors(NO_DELAY, 1.0, rng, dropout=dropout)
        sensors.reset([0.0] * 7)
        for k in range(50):
            sensors.observe([float(k)] * 7)
        rngs.append(rng.random())
    assert rngs[0] == rngs[1]


def run_commands(physics, still, action: dict, sigma: float, ticks: int = 3000) -> np.ndarray:
    sim = Sim(physics)
    sim.reset(1, {**still, "actuator_noise": {"thrust_sigma": sigma, "torque_sigma": sigma}})
    far_targets(sim)
    out = []
    for _ in range(ticks):
        sim.step_tick(action)
        out.append(sim.command)
        if sim.done:
            break
    return np.array(out)


def test_actuator_noise_sigma(physics, still):
    cmds = run_commands(physics, still, {"thrust": 0.5, "yaw_torque": 0.5}, 0.1)
    assert len(cmds) == 3000
    assert cmds[:, 0].std() == pytest.approx(0.1, rel=0.1)
    assert cmds[:, 1].std() == pytest.approx(0.1, rel=0.1)
    assert cmds[:, 0].mean() == pytest.approx(0.5, abs=0.01)


def test_actuator_noise_stays_in_range(physics, still):
    cmds = run_commands(physics, still, {"thrust": 1.0, "yaw_torque": -1.0}, 0.3, ticks=500)
    assert (cmds[:, 0] >= 0.0).all() and (cmds[:, 0] <= 1.0).all()
    assert (cmds[:, 1] >= -1.0).all() and (cmds[:, 1] <= 1.0).all()
    assert (cmds[:, 0] < 1.0).any()


def test_actuator_noise_off_is_exact(physics, still):
    cmds = run_commands(physics, still, {"thrust": 0.4, "yaw_torque": 0.2}, 0.0, ticks=50)
    assert (cmds == [0.4, 0.2]).all()


def test_all_levers_deterministic(physics):
    s4 = {**scenario("s4_turbulent"), **ALL_ON}
    a = run_episode(Sim(physics), PID(), 11, s4)
    b = run_episode(Sim(physics), PID(), 11, s4)
    c = run_episode(Sim(physics), PID(), 12, s4)
    assert a.ticks == b.ticks
    assert a.ticks != c.ticks


@pytest.mark.parametrize("lever", sorted(ALL_ON))
def test_lever_does_not_change_world(physics, lever):
    s4 = scenario("s4_turbulent")
    a = Sim(physics)
    a.reset(9, s4)
    b = Sim(physics)
    b.reset(9, {**s4, lever: ALL_ON[lever]})
    assert a.targets == b.targets
    assert a.wind_x == b.wind_x and a.wind_y == b.wind_y
    assert a.psi == b.psi


CAP_VIOLATIONS = [
    {"type": "steady", "speed_mm_s": 150.1, "direction_rad": "seed"},
    {"type": "sinusoid", "amplitude_mm_s": 151.0, "freq_hz": 0.5, "direction_rad": "seed"},
    {"type": "gust", "amplitude_mm_s": 300.5, "direction_rad": "seed", "t_start_s": 2.0,
     "duration_s": 0.4},
    {"type": "gust", "amplitude_mm_s": 100.0, "direction_rad": "seed", "t_start_s": 2.0,
     "duration_s": 0.51},
    {"type": "gust_train", "amplitude_mm_s": 301.0, "count": 3, "duration_s": 0.4,
     "spacing_s": [2.0, 5.0]},
    {"type": "gust_train", "amplitude_mm_s": 200.0, "count": 3, "duration_s": 0.6,
     "spacing_s": [2.0, 5.0]},
    {"type": "turbulence", "sigma_mm_s": 100.1, "tau_c_s": 0.25},
]
LEVER_VIOLATIONS = [
    {"sensor_delay_scale": 8.1},
    {"sensor_delay_scale": -1.0},
    {"actuator_lag_scale": 5.01},
    {"actuator_lag_scale": 0.0},
    {"dropout": {"p_start": 0.051, "length_ticks": 5}},
    {"dropout": {"p_start": 0.01, "length_ticks": 21}},
    {"dropout": {"p_start": 0.01, "length_ticks": 2.5}},
    {"actuator_noise": {"thrust_sigma": 0.31, "torque_sigma": 0.0}},
    {"actuator_noise": {"thrust_sigma": 0.0, "torque_sigma": 0.301}},
    {"sensor_delay_scale": [1.0, 9.0]},
]


@pytest.mark.parametrize("comp", CAP_VIOLATIONS)
def test_validator_rejects_wind_caps(still, comp):
    with pytest.raises(ValueError):
        validate_scenario({**still, "wind": [comp]})


@pytest.mark.parametrize("extra", LEVER_VIOLATIONS)
def test_validator_rejects_lever_caps(still, extra):
    with pytest.raises(ValueError):
        validate_scenario({**still, **extra})


def test_validator_accepts_values_at_caps(still):
    at_caps = {
        "sensor_delay_scale": CAPS["sensor_delay_scale"],
        "actuator_lag_scale": CAPS["actuator_lag_scale"],
        "dropout": {"p_start": CAPS["dropout_p_start"],
                    "length_ticks": CAPS["dropout_length_ticks"]},
        "actuator_noise": {"thrust_sigma": 0.3, "torque_sigma": 0.3},
        "wind": [
            {"type": "steady", "speed_mm_s": 150.0, "direction_rad": "seed"},
            {"type": "gust", "amplitude_mm_s": 300.0, "direction_rad": "seed", "t_start_s": 2.0,
             "duration_s": 0.5},
            {"type": "turbulence", "sigma_mm_s": 100.0, "tau_c_s": 0.25},
        ],
    }
    validate_scenario({**still, **at_caps})


def test_level_lists_expand_together(still):
    s = {**still, "name": "sweep", "levels": [0.5, 1, 2, 3],
         "noise_multiplier": [1.0, 2.0, 4.0, 6.0], "sensor_delay_scale": [1.0, 2.0, 4.0, 6.0],
         "dropout": {"p_start": [0.01, 0.02, 0.04, 0.05], "length_ticks": 10}}
    validate_scenario(s)
    levels = expand_noise_sweep(s)
    assert [lv["name"] for lv in levels] == ["sweep_x0.5", "sweep_x1", "sweep_x2", "sweep_x3"]
    assert levels[2]["noise_multiplier"] == 4.0
    assert levels[2]["sensor_delay_scale"] == 4.0
    assert levels[2]["dropout"] == {"p_start": 0.04, "length_ticks": 10}
    for lv in levels:
        assert "levels" not in lv
        validate_scenario(lv)
