import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from controllers.base import Controller
from env.params import CONFIG_DIR
from fly.encoding import INPUT_ROLES, OUTPUT_ROLES, WindowDecoder, clip, encode, rates_array
from fly.network import Network, build_network

DEFAULT_MAPPING = CONFIG_DIR / "fly_mapping_synthetic.json"
TUNABLE_ENCODING = ("bearing_hz", "range_hz", "air_hz")
TUNABLE_DECODING = ("gain_yaw", "gain_thrust", "thrust_base")
SEED_LIMIT = 2**31 - 1


def load_mapping(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Fly(Controller):
    name = "fly"

    @classmethod
    def default_params(cls) -> dict:
        return {"mapping": str(DEFAULT_MAPPING), "stepper": "numpy", "weight_scale": 1.0}

    @classmethod
    def param_space(cls) -> dict[str, tuple[float, float]]:
        return {"bearing_hz": (50.0, 500.0), "range_hz": (0.0, 300.0), "air_hz": (0.0, 300.0),
                "gain_yaw": (0.2, 4.0), "gain_thrust": (0.0, 1.0), "thrust_base": (0.1, 0.7)}

    def __init__(self, params: dict | None = None, network: Network | None = None):
        super().__init__(params)
        mapping = load_mapping(self.params["mapping"])
        self.enc = {**mapping["encoding"],
                    **{k: self.params[k] for k in TUNABLE_ENCODING if k in self.params}}
        self.dec = {**mapping["decoding"],
                    **{k: self.params[k] for k in TUNABLE_DECODING if k in self.params}}
        self.steps_per_tick = int(mapping["steps_per_tick"])
        self.net = network or build_network(mapping["network"])
        scale = float(self.params.get("weight_scale", 1.0))
        if scale != 1.0:
            self.net.set_weight_scale(scale)
        if self.params.get("stepper") == "numba":
            from fly.lif_numba import NumbaLIFStepper

            self.net.lif = NumbaLIFStepper(self.net.weights, self.net.lif_params)
        roles = mapping["groups"]
        self.ids = {role: self.net.group(roles[role]) for role in INPUT_ROLES + OUTPUT_ROLES}
        self.log_roles = list(INPUT_ROLES + OUTPUT_ROLES)
        self.lookup = np.full(self.net.n, len(self.log_roles), dtype=np.int64)
        for i, role in enumerate(self.log_roles):
            self.lookup[self.ids[role]] = i
        tick_s = self.steps_per_tick * self.net.dt_ms / 1000.0
        sizes = {role: int(self.ids[role].size) for role in OUTPUT_ROLES}
        self.decoder = WindowDecoder(self.dec, sizes, tick_s)
        self.counts: dict[str, int] = {}
        self.spikes_total = 0

    def reset(self, seed: int, scenario: dict) -> None:
        super().reset(seed, scenario)
        self.net.reset(int(self.rng.integers(0, SEED_LIMIT)))
        self.decoder.reset()
        self.counts = {role: 0 for role in self.log_roles}
        self.spikes_total = 0

    def act(self, obs: dict, t: float) -> dict:
        rates = rates_array(self.net.n, self.ids, encode(obs, self.enc))
        self.net.set_rates(rates)
        fired = [self.net.step() for _ in range(self.steps_per_tick)]
        spikes = np.concatenate(fired)
        per_role = np.bincount(self.lookup[spikes], minlength=len(self.log_roles) + 1)
        self.counts = {role: int(per_role[i]) for i, role in enumerate(self.log_roles)}
        self.spikes_total = int(spikes.size)
        action = self.decoder.push(self.counts)
        return {"thrust": float(action["thrust"]), "yaw_torque": float(action["yaw_torque"])}

    def info(self) -> dict:
        return {"groups": dict(self.counts), "spikes_total": self.spikes_total}

    def calibration_rates(self, cal: dict) -> dict[str, float]:
        self.reset(cal["seed"], {})
        obs = {"bearing": cal["bearing_rad"], "range": cal["range_mm"], "vel_body": [0.0, 0.0],
               "yaw_rate": 0.0, "air_rel": list(cal["air_rel"])}
        tick_s = self.steps_per_tick * self.net.dt_ms / 1000.0
        totals = dict.fromkeys(OUTPUT_ROLES, 0)
        for _ in range(round(cal["seconds"] / tick_s)):
            self.act(obs, 0.0)
            for role in OUTPUT_ROLES:
                totals[role] += self.counts[role]
        return {role: totals[role] / self.ids[role].size / cal["seconds"] for role in OUTPUT_ROLES}


REAL_MAPPING = CONFIG_DIR / "fly_mapping_real.json"
YAW_AT_180 = 0.6668
STEER_ONLY_THRUST = 0.5
_real_cache: dict = {}


def real_network(spec: dict):
    from fly.connectome import ConnectomeNetwork
    from fly.lif_numba import NumbaLIFStepper

    key = (spec["path"], float(spec["weight_scale"]))
    if key not in _real_cache:
        _real_cache.clear()
        net = ConnectomeNetwork(Path(spec["path"]))
        net.set_weight_scale(key[1])
        net.lif = NumbaLIFStepper(net.weights, net.lif_params)
        _real_cache[key] = net
    return _real_cache[key]


def side_groups(net, query: str) -> tuple[np.ndarray, np.ndarray]:
    return net.group(f"{query},somaSide=L"), net.group(f"{query},somaSide=R")


def push_pull_rates(bearing: float, c: float, a: float, rate_max: float) -> tuple[float, float]:
    push = a * bearing / math.pi
    return clip(c + push, 0.0, rate_max), clip(c - push, 0.0, rate_max)


def rectified_rates(bearing: float, r_min: float, r_max: float) -> tuple[float, float]:
    span = r_max - r_min
    return r_min + span * max(0.0, bearing) / math.pi, r_min + span * max(0.0, -bearing) / math.pi


class RealFlyBase(Controller):
    name = "real_base"
    thrust_source = "network"

    @classmethod
    def default_params(cls) -> dict:
        return {"mapping": str(REAL_MAPPING), "gain_yaw": round(1.0 / YAW_AT_180, 3),
                "tick_ms": 5.0}

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        mapping = load_mapping(self.params["mapping"])
        spec = mapping["network"]
        path = Path(spec["path"])
        if not path.is_absolute():
            path = CONFIG_DIR.parent / path
        self.weight_scale = float(spec["weight_scale"])
        self.net = real_network({"path": str(path), "weight_scale": self.weight_scale})
        self.steer = mapping["steering"]
        self.thrust_map = mapping.get("thrust")
        self.gain_yaw = float(self.params["gain_yaw"])
        self.lc10_L, self.lc10_R = side_groups(self.net, self.steer["input_left"].split(",")[0])
        self.dna02_L, self.dna02_R = side_groups(self.net, self.steer["output_left"].split(",")[0])
        self.thrust_ids = self.thrust_group()
        self.roles = ("dna02_L", "dna02_R", "thrust")
        self.lookup = np.full(self.net.n, len(self.roles), dtype=np.int64)
        for i, ids in enumerate((self.dna02_L, self.dna02_R, self.thrust_ids)):
            self.lookup[ids] = i
        tick_ms = float(self.params["tick_ms"])
        self.steps_per_tick = round(tick_ms / self.net.dt_ms)
        self.window = deque(maxlen=round(float(self.steer["window_ms"]) / tick_ms))
        self.rates = np.zeros(self.net.n, dtype=np.float32)
        self.counts = dict.fromkeys(self.roles, 0)
        self.spikes_total = 0
        self.yaw_signal = 0.0
        self.thrust_decode = 0.0

    def thrust_group(self) -> np.ndarray:
        return np.zeros(0, dtype=np.int64)

    def centre_hz(self, range_mm: float) -> float:
        return float(self.steer["c_hz"])

    def header_info(self) -> dict:
        return {"weight_scale": self.weight_scale, "mapping": Path(self.params["mapping"]).name,
                "steering": self.steer.get("type", "rectified"), "gain_yaw": self.gain_yaw,
                "thrust_source": self.thrust_source}

    def reset(self, seed: int, scenario: dict) -> None:
        super().reset(seed, scenario)
        self.net.reset(int(self.rng.integers(0, SEED_LIMIT)))
        self.window.clear()
        self.counts = dict.fromkeys(self.roles, 0)
        self.spikes_total = 0
        self.yaw_signal = 0.0
        self.thrust_decode = 0.0

    def lc10_rates(self, bearing: float, range_mm: float) -> tuple[float, float]:
        s = self.steer
        if s.get("type") == "push_pull":
            return push_pull_rates(bearing, self.centre_hz(range_mm), float(s["a_hz"]),
                                   float(s["rate_max_hz"]))
        return rectified_rates(bearing, float(s["r_min_hz"]), float(s["r_max_hz"]))

    def thrust(self) -> float:
        raise NotImplementedError

    def act(self, obs: dict, t: float) -> dict:
        left, right = self.lc10_rates(obs["bearing"], obs["range"])
        self.rates[self.lc10_L] = left
        self.rates[self.lc10_R] = right
        self.net.set_spike_rates(self.rates)
        fired = np.concatenate([self.net.step() for _ in range(self.steps_per_tick)])
        per_role = np.bincount(self.lookup[fired], minlength=len(self.roles) + 1)
        self.counts = {role: int(per_role[i]) for i, role in enumerate(self.roles)}
        self.spikes_total = int(fired.size)
        self.window.append(self.counts)
        n_l = sum(c["dna02_L"] for c in self.window)
        n_r = sum(c["dna02_R"] for c in self.window)
        self.yaw_signal = (n_l - n_r) / (n_l + n_r + 1.0)
        if self.thrust_ids.size:
            self.thrust_decode = sum(c["thrust"] for c in self.window) / self.thrust_ids.size
        return {"thrust": float(clip(self.thrust(), 0.0, 1.0)),
                "yaw_torque": float(clip(self.gain_yaw * self.yaw_signal, -1.0, 1.0))}

    def info(self) -> dict:
        return {"groups": dict(self.counts), "spikes_total": self.spikes_total,
                "yaw_signal": round(self.yaw_signal, 4),
                "thrust_decode": round(self.thrust_decode, 4)}


class FlyReal(RealFlyBase):
    name = "fly_real"

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.gain_thrust = float(self.params.get("gain_thrust",
                                                 1.0 / self.thrust_map["decode_far"]))

    def thrust_group(self) -> np.ndarray:
        return np.unique(np.concatenate([self.net.group(q) for q in self.thrust_map["queries"]]))

    def centre_hz(self, range_mm: float) -> float:
        t = self.thrust_map
        frac = clip(range_mm / float(t["range_ref_mm"]), 0.0, 1.0)
        return float(t["c_near_hz"]) + (float(t["c_far_hz"]) - float(t["c_near_hz"])) * frac

    def header_info(self) -> dict:
        return {**super().header_info(), "thrust_group": self.thrust_map["group"],
                "gain_thrust": round(self.gain_thrust, 4)}

    def thrust(self) -> float:
        return self.gain_thrust * self.thrust_decode


class FlySteerOnly(RealFlyBase):
    name = "fly_steer_only"
    thrust_source = f"fixed {STEER_ONLY_THRUST:g}, not from the network"

    def header_info(self) -> dict:
        return {**super().header_info(), "thrust_fixed": STEER_ONLY_THRUST}

    def thrust(self) -> float:
        return STEER_ONLY_THRUST
