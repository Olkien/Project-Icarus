import math

from controllers.base import Controller


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


class PID(Controller):
    name = "pid"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "Kp": 3.0, "Kd": 0.3, "Ki": 0.0, "i_bearing_max": 0.5,
            "Kr": 2.0, "v_min": 40.0, "v_cruise": 200.0,
            "Kp_v": 0.005, "Ki_v": 0.01, "i_speed_max": 20.0,
            "ff_c1": 10.0, "ff_c2": 0.01, "ff_Tmax": 3900.0, "tick_s": 0.005,
        }

    @classmethod
    def param_space(cls) -> dict[str, tuple[float, float]]:
        return {
            "Kp": (0.5, 10.0), "Kd": (0.0, 1.0), "Ki": (0.0, 2.0),
            "Kr": (0.5, 10.0), "v_min": (10.0, 120.0), "v_cruise": (100.0, 300.0),
            "Kp_v": (0.0, 0.02), "Ki_v": (0.0, 0.05),
        }

    def reset(self, seed: int, scenario: dict) -> None:
        super().reset(seed, scenario)
        self.i_bearing = 0.0
        self.i_speed = 0.0
        self.e_bearing = 0.0
        self.e_speed = 0.0
        self.v_des = 0.0

    def act(self, obs: dict, t: float) -> dict:
        p = self.params
        dt = p["tick_s"]
        bearing = obs["bearing"]

        # heading loop
        yaw_pd = p["Kp"] * bearing - p["Kd"] * obs["yaw_rate"]
        yaw = yaw_pd + p["Ki"] * self.i_bearing
        if -1.0 < yaw < 1.0 or yaw * bearing < 0.0:
            self.i_bearing = clip(self.i_bearing + bearing * dt, -p["i_bearing_max"],
                                  p["i_bearing_max"])
        yaw = clip(yaw_pd + p["Ki"] * self.i_bearing, -1.0, 1.0)

        # speed loop
        v_des = clip(p["Kr"] * obs["range"], p["v_min"], p["v_cruise"])
        self.v_des = v_des * max(0.0, math.cos(bearing))
        error = self.v_des - obs["vel_body"][0]
        ff = (p["ff_c1"] * self.v_des + p["ff_c2"] * self.v_des * self.v_des) / p["ff_Tmax"]
        thrust = ff + p["Kp_v"] * error + p["Ki_v"] * self.i_speed
        if 0.0 < thrust < 1.0 or thrust * error < 0.0:
            self.i_speed = clip(self.i_speed + error * dt, -p["i_speed_max"], p["i_speed_max"])
        thrust = clip(ff + p["Kp_v"] * error + p["Ki_v"] * self.i_speed, 0.0, 1.0)

        self.e_bearing = bearing
        self.e_speed = error
        return {"thrust": float(thrust), "yaw_torque": float(yaw)}

    def info(self) -> dict:
        return {"e_bearing": self.e_bearing, "i_bearing": self.i_bearing, "v_des": self.v_des,
                "e_speed": self.e_speed, "i_speed": self.i_speed}
