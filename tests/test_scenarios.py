import pytest

from env.scenario import SCENARIO_DIR, expand_noise_sweep, load_scenario, scenario_hash

DAY1_HASHES = {
    "s1_still": "5477135efdca61b3fc0cf0acf5be3d772e7cc737276070c51c24342748c12ca8",
    "s2_crosswind": "89c59a928e8f06c89763a74d58f01d3773186909120249679fc8332d08abf9cd",
    "s3_gust": "8bb474c0c353b4a51cdf062af924b755418b96e0adc7cd8b22cfb2015db16701",
    "s4_turbulent": "3e5d865e328883b9d2685afd0f5fec8992ce553611f4820203fb6dea45c8d821",
    "s5_noise_sweep": "8f6d8687c96a0f1cb6363da6264e82131e793a41912bc2d8e4ab7ba673d88418",
    "s6_unseen_wind": "6be04c8e6706ffca1b10ad030ea27dfd9b50b3ed79148b4f99a0cc5131a091fd",
}


@pytest.mark.parametrize("name", sorted(DAY1_HASHES))
def test_scenario_validates(name):
    scenario = load_scenario(SCENARIO_DIR / f"{name}.json")
    assert scenario["name"] == name
    assert scenario["time_limit_s"] == 20.0
    assert scenario["targets"]["hit_radius_mm"] == 10.0


def test_s1_unchanged_and_still():
    s1 = load_scenario(SCENARIO_DIR / "s1_still.json")
    assert scenario_hash(s1) == DAY1_HASHES["s1_still"]
    assert s1["wind"] == []


def test_s5_wind_is_s4_wind():
    s4 = load_scenario(SCENARIO_DIR / "s4_turbulent.json")
    s5 = load_scenario(SCENARIO_DIR / "s5_noise_sweep.json")
    assert s5["wind"] == s4["wind"]
    assert len(expand_noise_sweep(s5)) == 4


def test_s5_levels_scale_s4_sensor_levers():
    from env.scenario import CAPS

    s4 = load_scenario(SCENARIO_DIR / "s4_turbulent.json")
    levels = expand_noise_sweep(load_scenario(SCENARIO_DIR / "s5_noise_sweep.json"))
    for factor, level in zip((0.5, 1.0, 2.0, 3.0), levels):
        assert level["wind"] == s4["wind"]
        assert level["noise_multiplier"] == pytest.approx(s4["noise_multiplier"] * factor)
        assert level["sensor_delay_scale"] == pytest.approx(
            min(CAPS["sensor_delay_scale"], s4["sensor_delay_scale"] * factor))
        assert level["dropout"]["p_start"] == pytest.approx(
            min(CAPS["dropout_p_start"], s4["dropout"]["p_start"] * factor))
        assert level["dropout"]["length_ticks"] == s4["dropout"]["length_ticks"]


def test_s6_uses_s4_levers():
    s4 = load_scenario(SCENARIO_DIR / "s4_turbulent.json")
    s6 = load_scenario(SCENARIO_DIR / "s6_unseen_wind.json")
    for key in ("noise_multiplier", "sensor_delay_scale", "dropout"):
        assert s6[key] == s4[key]
    kinds = {c["type"]: c for c in s6["wind"]}
    assert kinds["gust_train"]["amplitude_mm_s"] == 250.0
    assert kinds["sinusoid"]["amplitude_mm_s"] == 100.0
