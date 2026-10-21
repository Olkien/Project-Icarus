import math

import numpy as np
from scipy.signal import lfilter


def gust_profile(t: np.ndarray, amplitude: float, t_start: float, duration: float) -> np.ndarray:
    phase = (t - t_start) / duration
    inside = (phase >= 0.0) & (phase <= 1.0)
    return np.where(inside, 0.5 * amplitude * (1.0 - np.cos(2.0 * math.pi * phase)), 0.0)


def ou_series(sigma: float, tau_c: float, dt: float, n: int, rng: np.random.Generator) -> np.ndarray:
    a = math.exp(-dt / tau_c)
    noise = rng.standard_normal(n) * (sigma * math.sqrt(1.0 - a * a))
    noise[0] = sigma * rng.standard_normal()
    return lfilter([1.0], [1.0, -a], noise)


def direction(comp: dict, rng: np.random.Generator) -> float:
    value = comp.get("direction_rad", "seed")
    if value == "seed":
        return float(rng.uniform(-math.pi, math.pi))
    return float(value)


def build_wind(
    components: list[dict], rng: np.random.Generator, n_steps: int, dt: float
) -> tuple[list[float], list[float]]:
    t = np.arange(n_steps + 1) * dt
    wx = np.zeros(n_steps + 1)
    wy = np.zeros(n_steps + 1)
    for comp in components:
        kind = comp["type"]
        if kind == "steady":
            ang = direction(comp, rng)
            wx += comp["speed_mm_s"] * math.cos(ang)
            wy += comp["speed_mm_s"] * math.sin(ang)
        elif kind == "gust":
            ang = direction(comp, rng)
            g = gust_profile(t, comp["amplitude_mm_s"], comp["t_start_s"], comp["duration_s"])
            wx += g * math.cos(ang)
            wy += g * math.sin(ang)
        elif kind == "turbulence":
            wx += ou_series(comp["sigma_mm_s"], comp["tau_c_s"], dt, n_steps + 1, rng)
            wy += ou_series(comp["sigma_mm_s"], comp["tau_c_s"], dt, n_steps + 1, rng)
        elif kind == "gust_train":
            lo, hi = comp["spacing_s"]
            t_start = 0.0
            for _ in range(comp["count"]):
                t_start += float(rng.uniform(lo, hi))
                ang = direction(comp, rng)
                g = gust_profile(t, comp["amplitude_mm_s"], t_start, comp["duration_s"])
                wx += g * math.cos(ang)
                wy += g * math.sin(ang)
        elif kind == "sinusoid":
            ang = direction(comp, rng)
            s = comp["amplitude_mm_s"] * np.sin(2.0 * math.pi * comp["freq_hz"] * t)
            wx += s * math.cos(ang)
            wy += s * math.sin(ang)
        else:
            raise ValueError(f"unknown wind type {kind!r}")
    return wx.tolist(), wy.tolist()
