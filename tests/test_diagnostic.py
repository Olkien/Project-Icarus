import json

import numpy as np

from analysis.s4_parts import VARIANTS_PATH, load_variants
from analysis.step_response import POST_TICKS, PRE_TICKS, rise_ms
from env.scenario import load_scenario


def test_variants_cover_each_ingredient_both_ways():
    names = {v["name"] for v in load_variants(VARIANTS_PATH)}
    parts = ("wind", "turbulence", "noise", "delay", "dropout")
    assert names == {f"s4d_{way}_{p}" for way in ("alone", "without") for p in parts}


def test_without_noise_keeps_the_rest_of_s4():
    s4 = load_scenario("s4_turbulent")
    v = {x["name"]: x for x in json.loads(VARIANTS_PATH.read_text(encoding="utf-8"))}["s4d_without_noise"]
    assert v["noise_multiplier"] == 1.0
    assert {k: v[k] for k in ("wind", "sensor_delay_scale", "dropout")} == {
        k: s4[k] for k in ("wind", "sensor_delay_scale", "dropout")}


def test_rise_ms_on_a_clean_step():
    trace = np.zeros(PRE_TICKS + POST_TICKS)
    trace[PRE_TICKS + 3:] = 1.0
    assert rise_ms(trace, 0.5) == 20.0
    assert rise_ms(trace, 0.9) == 20.0
    assert rise_ms(np.zeros(PRE_TICKS + POST_TICKS), 0.5) is None


def test_decode_ema_step():
    from types import SimpleNamespace

    from controllers.fly import RealFlyBase

    ns = SimpleNamespace(alpha=0.25)
    assert RealFlyBase.ema(ns, 0.0, 1.0) == 0.25
    assert abs(RealFlyBase.ema(SimpleNamespace(alpha=1.0), 0.3, 0.9) - 0.9) < 1e-12


def test_decode_smooth_finds_variant():
    from analysis.decode_smooth import scenario_of

    assert scenario_of("s4d_alone_delay")["sensor_delay_scale"] == 7.0
    assert scenario_of("s1_still")["name"] == "s1_still"


def test_yaw_damp_settings_scale_pid_kd():
    from analysis.yaw_damp import PID_KD, SETTINGS

    assert [round(k / PID_KD, 2) for k in SETTINGS] == [0.5, 1.0, 2.0]
