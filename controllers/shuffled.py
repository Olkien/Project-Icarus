from controllers.fly import Fly, load_mapping
from fly.network import build_network
from fly.shuffle import shuffle


class Shuffled(Fly):
    name = "shuffled"

    @classmethod
    def default_params(cls) -> dict:
        return {**Fly.default_params(), "shuffle_seed": 0}

    def __init__(self, params: dict | None = None):
        merged = {**self.default_params(), **(params or {})}
        base = build_network(load_mapping(merged["mapping"])["network"])
        super().__init__(merged, network=shuffle(base, int(merged["shuffle_seed"])))
