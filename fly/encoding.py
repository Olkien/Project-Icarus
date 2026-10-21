from collections import deque

import numpy as np

INPUT_ROLES = ("vis_L", "vis_R", "air_fwd", "air_back", "air_L", "air_R")
OUTPUT_ROLES = ("out_L", "out_R", "fwd")


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def encode(obs: dict, enc: dict) -> dict[str, float]:
    side = enc["bearing_hz"] * clip(abs(obs["bearing"]) / enc["bearing_scale_rad"], 0.0, 1.0)
    loom = enc["range_hz"] * clip(1.0 - obs["range"] / enc["range_ref_mm"], 0.0, 1.0)
    vis_L = enc["base_hz"] + loom + (side if obs["bearing"] > 0.0 else 0.0)
    vis_R = enc["base_hz"] + loom + (side if obs["bearing"] < 0.0 else 0.0)
    fwd, left = obs["air_rel"]
    ref = enc["air_ref_mm_s"]

    def air(v: float) -> float:
        return enc["air_base_hz"] + enc["air_hz"] * clip(v / ref, 0.0, 1.0)

    return {"vis_L": vis_L, "vis_R": vis_R, "air_fwd": air(fwd), "air_back": air(-fwd),
            "air_L": air(left), "air_R": air(-left)}


def rates_array(n: int, group_ids: dict[str, np.ndarray], rates: dict[str, float]) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    for role, hz in rates.items():
        out[group_ids[role]] = hz
    return out


class WindowDecoder:
    def __init__(self, dec: dict, group_sizes: dict[str, int], tick_s: float):
        self.dec = dec
        self.sizes = group_sizes
        self.window = deque(maxlen=int(dec["window_ticks"]))
        self.tick_s = tick_s

    def reset(self) -> None:
        self.window.clear()

    def rates_hz(self) -> dict[str, float]:
        span_s = self.window.maxlen * self.tick_s
        totals = {role: sum(c[role] for c in self.window) for role in OUTPUT_ROLES}
        return {role: totals[role] / (self.sizes[role] * span_s) for role in OUTPUT_ROLES}

    def push(self, counts: dict[str, int]) -> dict:
        self.window.append({role: counts.get(role, 0) for role in OUTPUT_ROLES})
        d = self.dec
        r = self.rates_hz()
        yaw = d["yaw_sign"] * d["gain_yaw"] * (r["out_R"] - r["out_L"]) / d["norm_yaw_hz"]
        thrust = d["thrust_base"] + d["gain_thrust"] * r["fwd"] / d["norm_fwd_hz"]
        return {"thrust": clip(thrust, 0.0, 1.0), "yaw_torque": clip(yaw, -1.0, 1.0)}
