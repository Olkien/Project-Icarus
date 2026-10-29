import json
from pathlib import Path

import numpy as np

from controllers.base import Controller
from env.params import CONFIG_DIR
from fly.encoding import INPUT_ROLES, OUTPUT_ROLES, WindowDecoder, encode, rates_array
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
        return {"mapping": str(DEFAULT_MAPPING), "stepper": "numpy"}

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
