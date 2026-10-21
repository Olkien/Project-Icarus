import math

import pytest

from env.engine import Sim
from tests.conftest import scenario


def far_targets(sim: Sim) -> None:
    sim.targets = [(1.0e5, 1.0e5)] * len(sim.targets)


def speed(sim: Sim) -> float:
    return math.hypot(sim.vx, sim.vy)


def test_drag_decay_small_speed(physics, still):
    sim = Sim(physics)
    sim.reset(0, still)
    far_targets(sim)
    v0 = 5.0
    sim.vx, sim.vy = v0 * math.cos(sim.psi), v0 * math.sin(sim.psi)
    t_s = 0.2
    sim.step_physics(round(t_s / physics["dt_s"]), 0.0, 0.0)
    expected = v0 * math.exp(-physics["c1"] / physics["m"] * t_s)
    assert speed(sim) == pytest.approx(expected, rel=0.02)


def test_full_thrust_terminal_speed(physics, still):
    sim = Sim(physics)
    sim.reset(0, still)
    far_targets(sim)
    sim.step_physics(1500, 1.0, 0.0)
    assert speed(sim) == pytest.approx(300.0, rel=0.01)


def test_cruise_thrust_speed(physics, still):
    sim = Sim(physics)
    sim.reset(0, still)
    far_targets(sim)
    sim.step_physics(1500, 0.62, 0.0)
    assert speed(sim) == pytest.approx(200.0, rel=0.03)


def test_full_yaw_torque_steady_rate(physics, still):
    sim = Sim(physics)
    sim.reset(0, still)
    far_targets(sim)
    sim.step_physics(1000, 0.0, 1.0)
    assert sim.omega == pytest.approx(10.0, rel=0.01)


def test_no_thrust_goes_to_wind_velocity(physics):
    s2 = scenario("s2_crosswind")
    sim = Sim(physics)
    sim.reset(3, s2)
    far_targets(sim)
    sim.step_physics(2000, 0.0, 0.0)
    wx, wy = sim.wind_now()
    assert math.hypot(wx, wy) == pytest.approx(s2["wind"][0]["speed_mm_s"])
    assert sim.vx == pytest.approx(wx, abs=0.5)
    assert sim.vy == pytest.approx(wy, abs=0.5)


def test_actuator_lag_and_slew(physics, still):
    sim = Sim(physics)
    sim.reset(0, still)
    far_targets(sim)
    sim.step_physics(1, 1.0, 0.0)
    dt = physics["dt_s"]
    lag_rate = physics["Tmax"] / physics["tau_a"]
    assert sim.T == pytest.approx(min(physics["slew_T"], lag_rate) * dt)
    sim.step_physics(300, 1.0, 0.0)
    assert sim.T == pytest.approx(physics["Tmax"], rel=1e-3)


def test_z_stays_zero_in_xy(physics):
    sim = Sim(physics)
    sim.reset(5, scenario("s4_turbulent"))
    for _ in range(400):
        _, events = sim.step_tick({"thrust": 0.7, "yaw_torque": 0.3})
        assert sim.z == 0.0 and sim.vz == 0.0
        if "done" in events:
            break


def test_other_modes_not_implemented(physics, still):
    sim = Sim(physics)
    for mode in ("xz", "xyz"):
        with pytest.raises(NotImplementedError):
            sim.reset(0, {**still, "mode": mode})


def test_target_left_gives_positive_bearing(physics, still):
    sim = Sim(physics)
    sim.reset(0, still)
    sim.psi = 0.0
    sim.targets[0] = (100.0, 100.0)
    assert sim.true_bearing() == pytest.approx(math.pi / 4)
    sim.targets[0] = (100.0, -100.0)
    assert sim.true_bearing() == pytest.approx(-math.pi / 4)


def test_positive_yaw_torque_moves_bearing_to_zero(physics, still):
    sim = Sim(physics)
    sim.reset(0, still)
    sim.psi = 0.0
    sim.targets[0] = (0.0, 200.0)
    before = sim.true_bearing()
    sim.step_physics(50, 0.0, 1.0)
    after = sim.true_bearing()
    assert 0.0 < after < before


def test_hit_advances_target(physics, still):
    sim = Sim(physics)
    sim.reset(0, still)
    sim.psi = 0.0
    sim.targets[0] = (15.0, 0.0)
    sim.vx = 100.0
    events = sim.step_physics(100, 0.0, 0.0)
    assert events.count("hit") == 1
    assert sim.target_index == 1


def test_out_of_bounds_ends(physics, still):
    sim = Sim(physics)
    sim.reset(0, still)
    far_targets(sim)
    sim.x = 799.0
    sim.vx = 300.0
    events = sim.step_physics(20, 0.0, 0.0)
    assert "oob" in events and "done" in events
    assert sim.done and not sim.success


def test_timeout_after_limit(physics, still):
    sim = Sim(physics)
    sim.reset(0, {**still, "time_limit_s": 0.05})
    far_targets(sim)
    events = []
    for _ in range(20):
        _, events = sim.step_tick({"thrust": 0.0, "yaw_torque": 0.0})
        if "done" in events:
            break
    assert sim.k == 10
    assert "timeout" in events


def test_targets_spawn_rules(physics, still):
    for seed in range(30):
        sim = Sim(physics)
        sim.reset(seed, still)
        assert len(sim.targets) == 5
        prev = (0.0, 0.0)
        for tx, ty in sim.targets:
            d = math.hypot(tx - prev[0], ty - prev[1])
            assert 150.0 <= d <= 350.0
            assert abs(tx) <= 500.0 and abs(ty) <= 500.0
            prev = (tx, ty)
