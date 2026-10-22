import pytest

from analysis.difficulty import apply_settings, ideal_sensors, random_settings, set_lever
from tests.conftest import scenario


def test_set_lever_nested_paths():
    s4 = scenario("s4_turbulent")
    out = apply_settings(s4, {"wind.1.sigma_mm_s": 90.0, "dropout.p_start": 0.02,
                              "dropout.length_ticks": 6.6, "actuator_noise.torque_sigma": 0.1,
                              "sensor_delay_scale": 3.0})
    assert out["wind"][1]["sigma_mm_s"] == 90.0
    assert out["dropout"] == {"p_start": 0.02, "length_ticks": 7}
    assert out["actuator_noise"] == {"torque_sigma": 0.1}
    assert out["sensor_delay_scale"] == 3.0
    assert s4 == scenario("s4_turbulent")


def test_set_lever_rejects_cap():
    with pytest.raises(ValueError):
        set_lever(scenario("s2_crosswind"), "wind.0.speed_mm_s", 200.0)


def test_ideal_sensors_keeps_wind_and_actuators():
    s = apply_settings(scenario("s4_turbulent"), {"sensor_delay_scale": 4.0,
                                                  "dropout.p_start": 0.03,
                                                  "dropout.length_ticks": 5,
                                                  "actuator_lag_scale": 3.0})
    ideal = ideal_sensors(s)
    assert ideal["sensor_delay_scale"] == 0.0
    assert ideal["noise_multiplier"] == 0.0
    assert "dropout" not in ideal
    assert ideal["actuator_lag_scale"] == 3.0
    assert ideal["wind"] == s["wind"]


def test_random_settings_in_range_and_seeded():
    ranges = {"wind.0.speed_mm_s": (10.0, 150.0), "dropout.length_ticks": (1.0, 20.0)}
    a = random_settings(ranges, 50, seed=3)
    assert a == random_settings(ranges, 50, seed=3)
    assert a != random_settings(ranges, 50, seed=4)
    for s in a:
        assert 10.0 <= s["wind.0.speed_mm_s"] <= 150.0
        assert isinstance(s["dropout.length_ticks"], int) and 1 <= s["dropout.length_ticks"] <= 20
