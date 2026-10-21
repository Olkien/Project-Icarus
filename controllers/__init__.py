import importlib

REGISTRY = {
    "straight": "controllers.dummy:Straight",
    "random": "controllers.dummy:Random",
    "point_and_go": "controllers.dummy:PointAndGo",
}


def controller_class(name: str) -> type:
    if name not in REGISTRY:
        raise KeyError(f"unknown controller {name!r}, known: {sorted(REGISTRY)}")
    module, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), cls)


def make_controller(name: str, params: dict | None = None):
    return controller_class(name)(params)
