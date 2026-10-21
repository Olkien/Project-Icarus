import math
import time
from dataclasses import dataclass, field

import numpy as np

from env.sensors import Sensors, to_obs_dict, wrap_angle
from env.wind import build_wind

STREAMS = ("targets", "wind", "sensors", "start", "controller")


def make_generators(seed: int) -> dict[str, np.random.Generator]:
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.default_rng(child) for name, child in zip(STREAMS, children)}


def spawn_targets(cfg: dict, rng: np.random.Generator) -> list[tuple[float, float]]:
    targets = []
    px, py = 0.0, 0.0
    bound = cfg["bound_mm"]
    for _ in range(int(cfg["count"])):
        while True:
            dist = rng.uniform(cfg["spawn_min_mm"], cfg["spawn_max_mm"])
            ang = rng.uniform(-math.pi, math.pi)
            tx = px + dist * math.cos(ang)
            ty = py + dist * math.sin(ang)
            if abs(tx) <= bound and abs(ty) <= bound:
                break
        targets.append((float(tx), float(ty)))
        px, py = tx, ty
    return targets


def clip(v: float, lo: float, hi: float) -> float:
    if math.isnan(v):
        return 0.0
    return max(lo, min(v, hi))


class Sim:
    def __init__(self, params: dict):
        self.p = params
        self.dt = params["dt_s"]
        self.tick_steps = params["tick_steps"]
        self.tick_s = self.dt * self.tick_steps
        self.scenario: dict = {}

    # episode setup
    def reset(self, seed: int, scenario: dict) -> dict:
        if scenario["mode"] != "xy":
            raise NotImplementedError(f"mode {scenario['mode']!r} is not built yet")
        self.scenario = scenario
        self.seed = seed
        gens = make_generators(seed)
        self.ticks_limit = round(scenario["time_limit_s"] / self.tick_s)
        n_steps = self.ticks_limit * self.tick_steps
        self.wind_x, self.wind_y = build_wind(scenario["wind"], gens["wind"], n_steps, self.dt)
        tcfg = scenario["targets"]
        self.targets = spawn_targets(tcfg, gens["targets"])
        self.hit_radius = tcfg["hit_radius_mm"]
        self.sensors = Sensors(self.p["sensors"], scenario["noise_multiplier"], gens["sensors"])

        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.psi = float(gens["start"].uniform(-math.pi, math.pi))
        self.omega = 0.0
        self.T = 0.0
        self.tau = 0.0
        self.n = 0
        self.k = 0
        self.target_index = 0
        self.done = False
        self.success = False
        self.out_of_bounds = False
        self.sensors.reset(self.true_sample())
        self.obs = to_obs_dict(self.sensors.observe(self.true_sample()))
        return self.obs

    # true channels
    def current_target(self) -> tuple[float, float]:
        return self.targets[min(self.target_index, len(self.targets) - 1)]

    def true_bearing(self) -> float:
        tx, ty = self.current_target()
        return wrap_angle(math.atan2(ty - self.y, tx - self.x) - self.psi)

    def true_range(self) -> float:
        tx, ty = self.current_target()
        return math.hypot(tx - self.x, ty - self.y)

    def wind_now(self) -> tuple[float, float]:
        i = min(self.n, len(self.wind_x) - 1)
        return self.wind_x[i], self.wind_y[i]

    def true_sample(self) -> list[float]:
        c, s = math.cos(self.psi), math.sin(self.psi)
        wx, wy = self.wind_now()
        ax, ay = self.vx - wx, self.vy - wy
        return [
            self.true_bearing(), self.true_range(),
            c * self.vx + s * self.vy, -s * self.vx + c * self.vy,
            self.omega,
            c * ax + s * ay, -s * ax + c * ay,
        ]

    # physics inner loop
    def step_physics(self, steps: int, thrust: float, yaw_torque: float) -> list[str]:
        p = self.p
        dt, m, inertia = self.dt, p["m"], p["I"]
        c1, c2, c_r, tau_a = p["c1"], p["c2"], p["c_r"], p["tau_a"]
        slew_T, slew_tau = p["slew_T"], p["slew_tau"]
        T_cmd = p["Tmax"] * thrust
        tau_cmd = p["taumax"] * yaw_torque
        limit = p["arena_half_mm"]
        r2_hit = self.hit_radius * self.hit_radius
        last = len(self.wind_x) - 1
        x, y, vx, vy = self.x, self.y, self.vx, self.vy
        psi, omega, T, tau = self.psi, self.omega, self.T, self.tau
        tx, ty = self.current_target()
        events: list[str] = []
        for _ in range(steps):
            i = min(last, self.n)
            wx = self.wind_x[i]
            wy = self.wind_y[i]
            dT = (T_cmd - T) / tau_a
            dT = max(-slew_T, min(dT, slew_T))
            T += dT * dt
            dtau = (tau_cmd - tau) / tau_a
            dtau = max(-slew_tau, min(dtau, slew_tau))
            tau += dtau * dt
            omega += (tau - c_r * omega) / inertia * dt
            psi += omega * dt
            rx = vx - wx
            ry = vy - wy
            drag = c1 + c2 * math.sqrt(rx * rx + ry * ry)
            vx += (T * math.cos(psi) - drag * rx) / m * dt
            vy += (T * math.sin(psi) - drag * ry) / m * dt
            x += vx * dt
            y += vy * dt
            self.n += 1
            dx = tx - x
            dy = ty - y
            if dx * dx + dy * dy < r2_hit:
                events.append("hit")
                self.target_index += 1
                if self.target_index >= len(self.targets):
                    self.success = True
                    self.done = True
                    events.append("done")
                    break
                tx, ty = self.current_target()
            if x > limit or x < -limit or y > limit or y < -limit:
                self.out_of_bounds = True
                self.done = True
                events += ["oob", "done"]
                break
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.psi, self.omega, self.T, self.tau = wrap_angle(psi), omega, T, tau
        return events

    # one control tick
    def step_tick(self, action: dict) -> tuple[dict, list[str]]:
        thrust = clip(float(action.get("thrust", 0.0)), 0.0, 1.0)
        yaw = clip(float(action.get("yaw_torque", 0.0)), -1.0, 1.0)
        events = self.step_physics(self.tick_steps, thrust, yaw)
        self.k += 1
        if not self.done and self.k >= self.ticks_limit:
            self.done = True
            events += ["timeout", "done"]
        self.obs = to_obs_dict(self.sensors.observe(self.true_sample()))
        return self.obs, events

    @property
    def t(self) -> float:
        return round(self.n * self.dt, 9)

    def snapshot(self) -> dict:
        wx, wy = self.wind_now()
        tx, ty = self.current_target()
        return {
            "state": {"x": self.x, "y": self.y, "z": self.z, "psi": self.psi, "vx": self.vx,
                      "vy": self.vy, "vz": self.vz, "omega": self.omega},
            "wind": [wx, wy, 0.0],
            "target": {"i": self.target_index, "x": tx, "y": ty},
        }


@dataclass
class EpisodeResult:
    ticks: list[dict] = field(default_factory=list)
    act_s_total: float = 0.0
    success: bool = False
    out_of_bounds: bool = False


def run_episode(sim: Sim, controller, seed: int, scenario: dict, on_tick=None) -> EpisodeResult:
    obs = sim.reset(seed, scenario)
    controller.reset(seed, scenario)
    result = EpisodeResult()
    while not sim.done:
        snap = sim.snapshot()
        k, t = sim.k, sim.t
        start = time.perf_counter()
        raw = controller.act(obs, t)
        result.act_s_total += time.perf_counter() - start
        action = {
            "thrust": clip(float(raw.get("thrust", 0.0)), 0.0, 1.0),
            "yaw_torque": clip(float(raw.get("yaw_torque", 0.0)), -1.0, 1.0),
        }
        info = controller.info()
        next_obs, events = sim.step_tick(action)
        record = {"type": "tick", "k": k, "t": t, **snap, "obs": obs, "action": action,
                  "info": info, "events": events}
        result.ticks.append(record)
        if on_tick is not None:
            on_tick(record)
        obs = next_obs
    result.success = sim.success
    result.out_of_bounds = sim.out_of_bounds
    return result
