import math

import numpy as np
import pytest

from env.wind import build_wind, gust_profile, ou_series


def test_gust_peak_and_area():
    dt = 1.0e-4
    t = np.arange(0.0, 4.0, dt)
    w = gust_profile(t, 150.0, 2.0, 0.4)
    assert w.max() == pytest.approx(150.0, rel=1e-6)
    assert w.sum() * dt == pytest.approx(150.0 * 0.4 / 2.0, rel=1e-3)
    assert w[t < 2.0].max() == 0.0
    assert w[t > 2.4 + dt].max() == 0.0


def test_turbulence_sigma_and_correlation_time():
    rng = np.random.default_rng(1)
    dt = 0.001
    sigma, tau_c = 40.0, 0.25
    w = ou_series(sigma, tau_c, dt, 1_000_000, rng)
    assert w.std() == pytest.approx(sigma, rel=0.1)
    assert abs(w.mean()) < 0.1 * sigma
    lag = round(tau_c / dt)
    x = w - w.mean()
    rho = float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))
    assert rho == pytest.approx(math.exp(-1.0), abs=0.07)


def test_steady_fixed_direction():
    comps = [{"type": "steady", "speed_mm_s": 80.0, "direction_rad": math.pi / 2}]
    wx, wy = build_wind(comps, np.random.default_rng(0), 10, 0.001)
    assert wx[0] == pytest.approx(0.0, abs=1e-9)
    assert wy[5] == pytest.approx(80.0)


def test_components_add():
    comps = [
        {"type": "steady", "speed_mm_s": 40.0, "direction_rad": 0.0},
        {"type": "gust", "amplitude_mm_s": 150.0, "direction_rad": 0.0, "t_start_s": 0.0,
         "duration_s": 0.4},
    ]
    wx, _ = build_wind(comps, np.random.default_rng(0), 400, 0.001)
    assert wx[200] == pytest.approx(190.0)


def test_seeded_wind_repeats():
    comps = [
        {"type": "steady", "speed_mm_s": 40.0, "direction_rad": "seed"},
        {"type": "turbulence", "sigma_mm_s": 40.0, "tau_c_s": 0.25},
        {"type": "gust_train", "amplitude_mm_s": 120.0, "count": 3, "duration_s": 0.4,
         "spacing_s": [2.0, 5.0]},
        {"type": "sinusoid", "amplitude_mm_s": 60.0, "freq_hz": 0.5, "direction_rad": "seed"},
    ]
    a = build_wind(comps, np.random.default_rng(7), 5000, 0.001)
    b = build_wind(comps, np.random.default_rng(7), 5000, 0.001)
    c = build_wind(comps, np.random.default_rng(8), 5000, 0.001)
    assert a == b
    assert a != c


def test_gust_train_has_count_gusts():
    comps = [{"type": "gust_train", "amplitude_mm_s": 120.0, "count": 3, "duration_s": 0.4,
              "spacing_s": [2.0, 5.0], "direction_rad": 0.0}]
    wx, _ = build_wind(comps, np.random.default_rng(2), 20000, 0.001)
    w = np.array(wx)
    active = w > 1e-9
    starts = np.flatnonzero(active[1:] & ~active[:-1])
    assert len(starts) == 3
    assert w.max() == pytest.approx(120.0, rel=1e-3)
