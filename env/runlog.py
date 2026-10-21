import json
import subprocess
from pathlib import Path
from typing import Self

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "results" / "runs"
SCHEMA = 1


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            check=False, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def run_path(scenario: str, controller: str, seed: int, out_dir: Path = RUNS_DIR) -> Path:
    return Path(out_dir) / f"{scenario}__{controller}__s{seed}.jsonl"


LENGTH_DP = 3
ANGLE_DP = 4
TIME_DP = 4
STATE_ANGLES = ("psi", "omega")
OBS_ANGLES = ("bearing", "yaw_rate")


def rounded(value, dp: int):
    if isinstance(value, list):
        return [rounded(v, dp) for v in value]
    return round(value, dp) if isinstance(value, float) else value


def round_tick(record: dict) -> dict:
    out = dict(record)
    out["t"] = rounded(record["t"], TIME_DP)
    out["state"] = {k: rounded(v, ANGLE_DP if k in STATE_ANGLES else LENGTH_DP)
                    for k, v in record["state"].items()}
    out["wind"] = rounded(record["wind"], LENGTH_DP)
    out["target"] = {k: rounded(v, LENGTH_DP) for k, v in record["target"].items()}
    out["obs"] = {k: rounded(v, ANGLE_DP if k in OBS_ANGLES else LENGTH_DP)
                  for k, v in record["obs"].items()}
    return out


def dumps(record: dict) -> str:
    if record.get("type") == "tick":
        record = round_tick(record)
    return json.dumps(record, separators=(",", ":"), allow_nan=False)


class RunWriter:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.file = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w", encoding="utf-8", newline="\n")
        return self

    def write(self, record: dict) -> None:
        self.file.write(dumps(record) + "\n")

    def __exit__(self, *exc) -> None:
        self.file.close()


def write_run(path: Path | str, header: dict, ticks: list[dict], summary: dict) -> Path:
    with RunWriter(path) as w:
        w.write(header)
        for record in ticks:
            w.write(record)
        w.write(summary)
    return Path(path)


def read_run(path: Path | str) -> tuple[dict, list[dict], dict | None]:
    header = None
    ticks = []
    summary = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.get("type")
            if kind == "header":
                header = record
            elif kind == "tick":
                ticks.append(record)
            elif kind == "summary":
                summary = record
    if header is None:
        raise ValueError(f"{path}: no header line")
    return header, ticks, summary
