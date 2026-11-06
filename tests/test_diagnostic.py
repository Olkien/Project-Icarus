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
