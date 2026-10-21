import math

import pytest

from env.metrics import compute_metrics, cross_track_mm

TICK_S = 0.005
STILL = {"name": "s1_still", "wind": [], "targets": {"count": 5}}
GUST = {"name": "s3_gust", "targets": {"count": 5},
        "wind": [{"type": "gust", "amplitude_mm_s": 150.0, "direction_rad": 0.0,
                  "t_start_s": 0.1, "duration_s": 0.1}]}


def tick(k, x, y, target, events=(), thrust=0.5, yaw=0.0, info=None):
    i, tx, ty = target
    return {"type": "tick", "k": k, "t": round(k * TICK_S, 9),
            "state": {"x": x, "y": y, "z": 0.0, "psi": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0,
                      "omega": 0.0},
            "wind": [0.0, 0.0, 0.0], "target": {"i": i, "x": tx, "y": ty}, "obs": {},
            "action": {"thrust": thrust, "yaw_torque": yaw}, "info": info or {},
            "events": list(events)}


def straight_run(offset_y=0.0, n=100, step=1.0):
    ticks = []
    for k in range(n):
        last = k == n - 1
        ticks.append(tick(k, k * step, offset_y, (0, n * step, 0.0),
                          events=("hit",) if last else ()))
    return ticks


def test_cross_track_point_to_segment():
    assert cross_track_mm(5.0, 3.0, 0.0, 0.0, 10.0, 0.0) == pytest.approx(3.0)
    assert cross_track_mm(-4.0, 3.0, 0.0, 0.0, 10.0, 0.0) == pytest.approx(5.0)
    assert cross_track_mm(1.0, 1.0, 1.0, 1.0, 1.0, 1.0) == 0.0


def test_straight_path_efficiency_one_and_zero_cross_track():
    ticks = straight_run()
    ticks.append(tick(100, 100.0, 0.0, (1, 300.0, 0.0)))
    m = compute_metrics(ticks, STILL, TICK_S)
    assert m["targets_hit"] == 1
    assert m["path_efficiency"] == pytest.approx(1.0)
    assert m["rms_cross_track_mm"] == pytest.approx(0.0)
    assert m["time_per_target_s"] == pytest.approx(100 * TICK_S)


def test_fixed_offset_cross_track():
    ticks = [tick(0, 0.0, 0.0, (0, 200.0, 0.0))]
    ticks += [tick(k, float(k), 4.0, (0, 200.0, 0.0)) for k in range(1, 50)]
    m = compute_metrics(ticks, STILL, TICK_S)
    expected_rms = math.sqrt(49 * 16.0 / 50)
    assert m["rms_cross_track_mm"] == pytest.approx(expected_rms)
    assert m["max_cross_track_mm"] == pytest.approx(4.0)
    assert m["targets_hit"] == 0
    assert m["time_per_target_s"] is None
    assert m["time_per_target_miss6_s"] == pytest.approx(6.0)
    assert m["misses"] == 5


def test_detour_halves_efficiency():
    ticks = [tick(0, 0.0, 0.0, (0, 10.0, 0.0)), tick(1, 0.0, 5.0, (0, 10.0, 0.0)),
             tick(2, 10.0, 5.0, (0, 10.0, 0.0), events=("hit",)),
             tick(3, 10.0, 0.0, (1, 99.0, 0.0))]
    m = compute_metrics(ticks, STILL, TICK_S)
    assert m["path_efficiency"] == pytest.approx(10.0 / 20.0)


def test_success_and_times():
    targets = [(i, 10.0 * (i + 1), 0.0) for i in range(5)]
    ticks = []
    k = 0
    for i in range(5):
        for j in range(4):
            last = j == 3
            ticks.append(tick(k, 10.0 * i + 2.5 * j, 0.0, targets[i],
                              events=("hit", "done") if last and i == 4 else
                              ("hit",) if last else ()))
            k += 1
    m = compute_metrics(ticks, STILL, TICK_S, act_s_total=0.02)
    assert m["episode_success"] is True
    assert m["targets_hit"] == 5
    assert m["time_per_target_s"] == pytest.approx(4 * TICK_S)
    assert m["time_per_target_miss6_s"] == pytest.approx(4 * TICK_S)
    assert m["ctrl_ms_per_tick"] == pytest.approx(1.0)
    assert m["out_of_bounds"] is False


def test_out_of_bounds_is_failure():
    ticks = [tick(0, 0.0, 0.0, (0, 200.0, 0.0)), tick(1, 801.0, 0.0, (0, 200.0, 0.0),
                                                     events=("oob", "done"))]
    m = compute_metrics(ticks, STILL, TICK_S)
    assert m["out_of_bounds"] is True
    assert m["episode_success"] is False


def test_effort_and_jitter():
    ticks = [tick(k, 0.0, 0.0, (0, 200.0, 0.0), thrust=0.2 * (k % 2), yaw=(-1.0) ** k * 0.5)
             for k in range(10)]
    m = compute_metrics(ticks, STILL, TICK_S)
    assert m["effort_thrust"] == pytest.approx(0.1)
    assert m["effort_yaw"] == pytest.approx(0.5)
    assert m["jitter_yaw"] == pytest.approx(1.0)


def test_spikes_per_tick_only_with_spikes():
    plain = [tick(k, 0.0, 0.0, (0, 200.0, 0.0)) for k in range(4)]
    assert compute_metrics(plain, STILL, TICK_S)["spikes_per_tick"] is None
    fly = [tick(k, 0.0, 0.0, (0, 200.0, 0.0), info={"spikes_total": 10 * k}) for k in range(4)]
    assert compute_metrics(fly, STILL, TICK_S)["spikes_per_tick"] == pytest.approx(15.0)


def test_recovery_time_after_gust():
    target = (0, 1000.0, 0.0)
    ticks = []
    for k in range(200):
        t = k * TICK_S
        y = 30.0 if 0.0 < t < 0.5 else 0.0
        ticks.append(tick(k, float(k), y, target))
    m = compute_metrics(ticks, GUST, TICK_S)
    assert m["recovery_time_s"] == pytest.approx(0.5 - 0.2)


def test_recovery_never_is_time_to_end():
    ticks = [tick(k, float(k), 30.0 if k else 0.0, (0, 1000.0, 0.0)) for k in range(200)]
    m = compute_metrics(ticks, GUST, TICK_S)
    assert m["recovery_time_s"] == pytest.approx(200 * TICK_S - 0.2)


def test_recovery_only_with_gust():
    ticks = straight_run()
    assert compute_metrics(ticks, STILL, TICK_S)["recovery_time_s"] is None
