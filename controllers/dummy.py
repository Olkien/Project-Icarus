from controllers.base import Controller


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


class Straight(Controller):
    name = "straight"

    def act(self, obs: dict, t: float) -> dict:
        return {"thrust": 0.5, "yaw_torque": 0.0}


class Random(Controller):
    name = "random"

    def act(self, obs: dict, t: float) -> dict:
        return {"thrust": float(self.rng.uniform(0.0, 1.0)),
                "yaw_torque": float(self.rng.uniform(-1.0, 1.0))}


class PointAndGo(Controller):
    name = "point_and_go"

    @classmethod
    def default_params(cls) -> dict:
        return {"k_yaw": 2.0, "k_damp": 0.2, "aim_rad": 0.5, "thrust_go": 0.6, "thrust_turn": 0.2}

    def reset(self, seed: int, scenario: dict) -> None:
        super().reset(seed, scenario)
        self.last = {"thrust": 0.0, "yaw_torque": 0.0}

    def act(self, obs: dict, t: float) -> dict:
        p = self.params
        yaw = clip(p["k_yaw"] * obs["bearing"] - p["k_damp"] * obs["yaw_rate"], -1.0, 1.0)
        aimed = abs(obs["bearing"]) < p["aim_rad"]
        self.last = {"thrust": p["thrust_go"] if aimed else p["thrust_turn"], "yaw_torque": yaw}
        return self.last

    def info(self) -> dict:
        return {"aimed": self.last["thrust"] == self.params["thrust_go"]}
