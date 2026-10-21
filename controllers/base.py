from env.engine import make_generators


class Controller:
    name: str = "base"

    def __init__(self, params: dict | None = None):
        self.params = {**self.default_params(), **(params or {})}
        self.rng = make_generators(0)["controller"]

    @classmethod
    def default_params(cls) -> dict:
        return {}

    @classmethod
    def param_space(cls) -> dict[str, tuple[float, float]]:
        return {}

    def reset(self, seed: int, scenario: dict) -> None:
        self.rng = make_generators(seed)["controller"]

    def act(self, obs: dict, t: float) -> dict:
        raise NotImplementedError

    def info(self) -> dict:
        return {}
