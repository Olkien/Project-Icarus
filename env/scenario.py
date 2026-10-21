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


def load_scenario(path: Path | str) -> dict:
    path = Path(path)
    if not path.suffix:
        path = SCENARIO_DIR / f"{path.name}.json"
    scenario = json.loads(path.read_text(encoding="utf-8"))
    validate_scenario(scenario)
    return scenario


def scenario_hash(scenario: dict) -> str:
    return canonical_hash(scenario)


def expand_noise_sweep(scenario: dict) -> list[dict]:
    noise = scenario["noise_multiplier"]
    if not isinstance(noise, list):
        return [scenario]
    return [{**scenario, "name": f"{scenario['name']}_x{v:g}", "noise_multiplier": v}
            for v in noise]
