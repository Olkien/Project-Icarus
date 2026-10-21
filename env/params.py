import hashlib
import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
PHYSICS_PATH = CONFIG_DIR / "physics.json"

POSITIVE_KEYS = (
    "dt_s", "arena_half_mm", "m", "Tmax", "c1", "tau_a", "slew_T", "I", "c_r", "taumax",
    "slew_tau",
)
SENSOR_CHANNELS = ("bearing", "range", "vel_body", "yaw_rate", "air_rel")


def canonical_hash(data: dict) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_physics(params: dict) -> None:
    for key in POSITIVE_KEYS:
        value = params.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"physics: {key} must be a positive number, got {value!r}")
    if params.get("c2", -1) < 0:
        raise ValueError("physics: c2 must be >= 0")
    steps = params.get("tick_steps")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("physics: tick_steps must be a positive int")
    sensors = params.get("sensors", {})
    for channel in SENSOR_CHANNELS:
        cfg = sensors.get(channel)
        if cfg is None:
            raise ValueError(f"physics: missing sensor {channel}")
        if cfg.get("sigma", -1) < 0:
            raise ValueError(f"physics: sensor {channel} sigma must be >= 0")
        delay = cfg.get("delay_ticks")
        if not isinstance(delay, int) or delay < 0:
            raise ValueError(f"physics: sensor {channel} delay_ticks must be an int >= 0")


def load_physics(path: Path | str = PHYSICS_PATH) -> dict:
    params = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_physics(params)
    return params


def physics_hash(params: dict) -> str:
    return canonical_hash(params)
