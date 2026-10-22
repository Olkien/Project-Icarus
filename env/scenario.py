import json
from pathlib import Path

from env.params import CONFIG_DIR, canonical_hash

SCENARIO_DIR = CONFIG_DIR / "scenarios"
MODES = ("xy", "xz", "xyz")
WIND_KEYS = {
    "steady": ("speed_mm_s", "direction_rad"),
    "gust": ("amplitude_mm_s", "direction_rad", "t_start_s", "duration_s"),
    "turbulence": ("sigma_mm_s", "tau_c_s"),
    "gust_train": ("amplitude_mm_s", "count", "duration_s", "spacing_s"),
    "sinusoid": ("amplitude_mm_s", "freq_hz", "direction_rad"),
}
TARGET_KEYS = ("count", "spawn_min_mm", "spawn_max_mm", "bound_mm", "hit_radius_mm")
CAPS = {
    "steady_mm_s": 150.0,
    "gust_amplitude_mm_s": 300.0,
    "gust_duration_s": 0.5,
    "turbulence_sigma_mm_s": 100.0,
    "sensor_delay_scale": 8.0,
    "actuator_lag_scale": 5.0,
    "dropout_p_start": 0.05,
    "dropout_length_ticks": 20,
    "actuator_noise_sigma": 0.3,
}
LEVEL_KEYS = ("noise_multiplier", "sensor_delay_scale", "dropout.p_start")


def as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_range(name: str, label: str, value, lo: float, hi: float, lo_open: bool = False) -> None:
    for v in as_list(value):
        if not is_number(v) or v > hi or v < lo or (lo_open and v == lo):
            raise ValueError(f"{name}: {label} {v!r} outside {'(' if lo_open else '['}{lo}, {hi}]")


def check_wind_caps(name: str, comp: dict) -> None:
    kind = comp["type"]
    if kind in ("steady", "sinusoid"):
        key = "speed_mm_s" if kind == "steady" else "amplitude_mm_s"
        check_range(name, f"{kind} {key}", comp[key], 0.0, CAPS["steady_mm_s"])
    elif kind in ("gust", "gust_train"):
        check_range(name, f"{kind} amplitude_mm_s", comp["amplitude_mm_s"], 0.0,
                    CAPS["gust_amplitude_mm_s"])
        check_range(name, f"{kind} duration_s", comp["duration_s"], 0.0,
                    CAPS["gust_duration_s"], lo_open=True)
    elif kind == "turbulence":
        check_range(name, "turbulence sigma_mm_s", comp["sigma_mm_s"], 0.0,
                    CAPS["turbulence_sigma_mm_s"])


def check_levers(name: str, scenario: dict) -> None:
    if "sensor_delay_scale" in scenario:
        check_range(name, "sensor_delay_scale", scenario["sensor_delay_scale"], 0.0,
                    CAPS["sensor_delay_scale"])
    if "actuator_lag_scale" in scenario:
        check_range(name, "actuator_lag_scale", scenario["actuator_lag_scale"], 0.0,
                    CAPS["actuator_lag_scale"], lo_open=True)
    if "dropout" in scenario:
        drop = scenario["dropout"]
        check_range(name, "dropout p_start", drop.get("p_start", 0.0), 0.0,
                    CAPS["dropout_p_start"])
        length = drop.get("length_ticks", 0)
        if isinstance(length, float):
            raise ValueError(f"{name}: dropout length_ticks must be an int")
        check_range(name, "dropout length_ticks", length, 0, CAPS["dropout_length_ticks"])
    if "actuator_noise" in scenario:
        for key in ("thrust_sigma", "torque_sigma"):
            check_range(name, f"actuator_noise {key}", scenario["actuator_noise"].get(key, 0.0),
                        0.0, CAPS["actuator_noise_sigma"])


def validate_scenario(scenario: dict) -> None:
    name = scenario.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("scenario: name must be a non-empty string")
    if scenario.get("mode") not in MODES:
        raise ValueError(f"{name}: mode must be one of {MODES}")
    for comp in scenario.get("wind", []):
        kind = comp.get("type")
        if kind not in WIND_KEYS:
            raise ValueError(f"{name}: unknown wind type {kind!r}")
        for key in WIND_KEYS[kind]:
            if key not in comp:
                raise ValueError(f"{name}: wind {kind} needs {key}")
        direction = comp.get("direction_rad", "seed")
        if direction != "seed" and not isinstance(direction, (int, float)):
            raise ValueError(f"{name}: direction_rad must be a number or 'seed'")
        check_wind_caps(name, comp)
    check_levers(name, scenario)
    noise = scenario.get("noise_multiplier")
    values = noise if isinstance(noise, list) else [noise]
    if not values or any(not isinstance(v, (int, float)) or v < 0 for v in values):
        raise ValueError(f"{name}: noise_multiplier must be >= 0")
    targets = scenario.get("targets", {})
    for key in TARGET_KEYS:
        if not isinstance(targets.get(key), (int, float)) or targets[key] <= 0:
            raise ValueError(f"{name}: targets.{key} must be positive")
    if targets["spawn_min_mm"] > targets["spawn_max_mm"]:
        raise ValueError(f"{name}: spawn_min_mm > spawn_max_mm")
    limit = scenario.get("time_limit_s")
    if not isinstance(limit, (int, float)) or limit <= 0:
        raise ValueError(f"{name}: time_limit_s must be positive")


def load_scenario(path: Path | str, validate: bool = True) -> dict:
    path = Path(path)
    if not path.suffix:
        path = SCENARIO_DIR / f"{path.name}.json"
    scenario = json.loads(path.read_text(encoding="utf-8"))
    if validate:
        validate_scenario(scenario)
    return scenario


def scenario_hash(scenario: dict) -> str:
    return canonical_hash(scenario)


def get_path(scenario: dict, key: str):
    head, _, tail = key.partition(".")
    value = scenario.get(head)
    return value.get(tail) if tail and isinstance(value, dict) else value


def expand_noise_sweep(scenario: dict) -> list[dict]:
    lists = {k: get_path(scenario, k) for k in LEVEL_KEYS if isinstance(get_path(scenario, k), list)}
    if not lists:
        return [scenario]
    sizes = {len(v) for v in lists.values()}
    if len(sizes) != 1:
        raise ValueError(f"{scenario['name']}: level lists differ in length")
    count = sizes.pop()
    labels = scenario.get("levels", lists.get("noise_multiplier", list(range(count))))
    out = []
    for i in range(count):
        level = {**scenario, "name": f"{scenario['name']}_x{labels[i]:g}"}
        level.pop("levels", None)
        for key, values in lists.items():
            head, _, tail = key.partition(".")
            if tail:
                level[head] = {**scenario[head], tail: values[i]}
            else:
                level[head] = values[i]
        out.append(level)
    return out
