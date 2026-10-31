import numpy as np

from analysis.lockin_check import lockin_events


def test_step_found_and_runs_past_one_second():
    spikes = np.r_[np.full(1500, 400.0), np.full(2500, 4800.0)]
    events = lockin_events(spikes)
    assert len(events) == 1
    assert events[0]["start_s"] == 7.5
    assert events[0]["duration_s"] == 12.5
    assert events[0]["to_end"]


def test_short_burst_is_not_an_event():
    spikes = np.full(4000, 400.0)
    spikes[1000:1199] = 4800.0
    assert lockin_events(spikes) == []


def test_recovered_event_ends():
    spikes = np.full(4000, 400.0)
    spikes[1000:1300] = 4800.0
    events = lockin_events(spikes)
    assert len(events) == 1
    assert events[0]["duration_s"] == 1.5
    assert not events[0]["to_end"]


def test_early_event_uses_first_ticks_median():
    spikes = np.full(4000, 400.0)
    spikes[300:600] = 4800.0
    assert [e["start_s"] for e in lockin_events(spikes)] == [1.5]
