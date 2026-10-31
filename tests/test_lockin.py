import numpy as np
import pytest

from analysis.lockin_check import lockin_events


def test_absolute_event_inside_first_two_seconds():
    spikes = np.r_[np.full(165, 450.0), np.full(3835, 4800.0)]
    events = lockin_events(spikes)
    assert events[0]["start_s"] == pytest.approx(0.825)
    assert events[0]["duration_s"] == pytest.approx(19.175)
    assert events[0]["to_end"]
    assert lockin_events(spikes, "relative") == []


def test_absolute_needs_half_a_second_in_a_row():
    spikes = np.full(4000, 400.0)
    spikes[1000:1099] = 1501.0
    assert lockin_events(spikes) == []
    spikes[1000:1100] = 1501.0
    assert [e["duration_s"] for e in lockin_events(spikes)] == [0.5]


def test_absolute_threshold_is_exclusive():
    assert lockin_events(np.full(400, 1500.0)) == []


def test_step_found_and_runs_past_one_second():
    spikes = np.r_[np.full(1500, 400.0), np.full(2500, 4800.0)]
    events = lockin_events(spikes, "relative")
    assert len(events) == 1
    assert events[0]["start_s"] == 7.5
    assert events[0]["duration_s"] == 12.5
    assert events[0]["to_end"]


def test_short_burst_is_not_an_event():
    spikes = np.full(4000, 400.0)
    spikes[1000:1199] = 4800.0
    assert lockin_events(spikes, "relative") == []


def test_recovered_event_ends():
    spikes = np.full(4000, 400.0)
    spikes[1000:1300] = 4800.0
    events = lockin_events(spikes, "relative")
    assert len(events) == 1
    assert events[0]["duration_s"] == 1.5
    assert not events[0]["to_end"]


def test_early_event_uses_first_ticks_median():
    spikes = np.full(4000, 400.0)
    spikes[300:600] = 4800.0
    assert [e["start_s"] for e in lockin_events(spikes, "relative")] == [1.5]
