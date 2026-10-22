import argparse
import copy
import json
import math
import re
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from analysis.run import default_params, make_header, run_one
from controllers import REGISTRY
from env.params import load_physics
from env.runlog import RUNS_DIR, dumps, git_commit, run_path
from env.scenario import SCENARIO_DIR, load_scenario, validate_scenario

VIZ_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = VIZ_DIR / "samples"
HELD_OUT = ("s5_noise_sweep", "s6_unseen_wind")
RUN_NAME = re.compile(r"^[\w.-]+\.jsonl$")
MAX_SEED = 2**31 - 1
WIND_SCALE_RANGE = (0.0, 2.0)
NOISE_RANGE = (0.0, 4.0)
WIND_FIELDS = ("speed_mm_s", "amplitude_mm_s", "sigma_mm_s")


class ClientGone(Exception):
    pass


def live_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.json") if p.stem not in HELD_OUT)


def live_scenario(name: str, wind_scale: float = 1.0, noise: float | None = None) -> dict:
    lo, hi = WIND_SCALE_RANGE
    if not (math.isfinite(wind_scale) and lo <= wind_scale <= hi):
        raise ValueError(f"wind scale must be in {WIND_SCALE_RANGE}")
    base = load_scenario(name)
    scenario = copy.deepcopy(base)
    for comp in scenario["wind"]:
        for key in WIND_FIELDS:
            if key in comp:
                comp[key] = comp[key] * wind_scale
    if noise is not None:
        lo, hi = NOISE_RANGE
        if not (math.isfinite(noise) and lo <= noise <= hi):
            raise ValueError(f"noise multiplier must be in {NOISE_RANGE}")
        scenario["noise_multiplier"] = noise
    if wind_scale != 1.0 or scenario["noise_multiplier"] != base["noise_multiplier"]:
        scenario["name"] = f"{name}_w{wind_scale:g}_n{scenario['noise_multiplier']:g}"
    validate_scenario(scenario)
    return scenario


def query_float(query: dict, key: str, default: float | None) -> float | None:
    text = query.get(key, [""])[0]
    return float(text) if text else default


def samples() -> list[str]:
    return sorted(f"samples/{p.name}" for p in SAMPLES_DIR.glob("*.jsonl"))


class Handler(SimpleHTTPRequestHandler):
    runs_dir = RUNS_DIR

    def log_message(self, format: str, *args) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/api/options":
            self.send_json({"scenarios": live_scenarios(), "controllers": sorted(REGISTRY),
                            "samples": samples()})
        elif url.path == "/api/live":
            self.live(parse_qs(url.query))
        elif url.path.startswith("/runs/"):
            self.run_file(url.path[len("/runs/"):])
        else:
            super().do_GET()

    def run_file(self, name: str) -> None:
        path = self.runs_dir / name
        if not RUN_NAME.match(name) or not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_event(self, record: dict) -> None:
        try:
            self.wfile.write(f"data: {dumps(record)}\n\n".encode())
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as err:
            raise ClientGone from err

    def live(self, query: dict) -> None:
        scenario_name = query.get("scenario", [""])[0]
        controller = query.get("controller", [""])[0]
        try:
            seed = int(query.get("seed", ["0"])[0])
        except ValueError:
            seed = -1
        if scenario_name not in live_scenarios() or controller not in REGISTRY:
            self.send_json({"error": "unknown scenario or controller"}, 400)
            return
        if not 0 <= seed <= MAX_SEED:
            self.send_json({"error": "bad seed"}, 400)
            return
        try:
            scenario = live_scenario(scenario_name, query_float(query, "wind", 1.0),
                                     query_float(query, "noise", None))
        except ValueError as err:
            self.send_json({"error": str(err)}, 400)
            return
        physics = load_physics()
        commit = git_commit()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.send_event(make_header(scenario, controller, seed, physics, commit))
            metrics = run_one(scenario, controller, seed, default_params(controller),
                              physics=physics,
                              out=run_path(scenario["name"], controller, seed, self.runs_dir),
                              commit=commit, on_tick=self.send_event)
            self.send_event({"type": "summary", "metrics": metrics})
        except ClientGone:
            return
        self.close_connection = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    handler = partial(Handler, directory=str(VIZ_DIR))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"serving {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
