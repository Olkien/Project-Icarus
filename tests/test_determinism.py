import json
import math

from env.engine import Sim, make_generators, run_episode
from env.runlog import RunWriter, read_run, round_tick
from tests.conftest import scenario


class Scripted:
    name = "scripted"

    def reset(self, seed: int, scenario: dict) -> None:
        self.n = 0

    def act(self, obs: dict, t: float) -> dict:
        self.n += 1
        return {"thrust": 0.6, "yaw_torque": 0.8 * math.sin(0.02 * self.n)}

    def info(self) -> dict:
        return {"n": self.n}


class Spinner(Scripted):
    def act(self, obs: dict, t: float) -> dict:
        return {"thrust": 0.2, "yaw_torque": -1.0}


def test_same_seed_same_trajectory(physics):
    s4 = scenario("s4_turbulent")
    a = run_episode(Sim(physics), Scripted(), 11, s4)
    b = run_episode(Sim(physics), Scripted(), 11, s4)
    assert a.ticks == b.ticks


def test_different_seed_differs(physics):
    s4 = scenario("s4_turbulent")
    a = run_episode(Sim(physics), Scripted(), 11, s4)
    b = run_episode(Sim(physics), Scripted(), 12, s4)
    assert a.ticks != b.ticks


def test_world_same_for_every_controller(physics):
    s4 = scenario("s4_turbulent")
    a = Sim(physics)
    a.reset(4, s4)
    b = Sim(physics)
    b.reset(4, s4)
    assert a.psi == b.psi
    for _ in range(100):
        a.step_tick({"thrust": 1.0, "yaw_torque": 0.0})
        b.step_tick({"thrust": 0.0, "yaw_torque": -1.0})
    assert a.targets == b.targets
    assert a.wind_x == b.wind_x and a.wind_y == b.wind_y


def test_noise_level_does_not_change_world(physics):
    s4 = scenario("s4_turbulent")
    a = Sim(physics)
    a.reset(9, s4)
    b = Sim(physics)
    b.reset(9, {**s4, "noise_multiplier": 4.0})
    assert a.targets == b.targets
    assert a.wind_x == b.wind_x
    assert a.psi == b.psi


def test_generators_independent():
    g1 = make_generators(5)
    g2 = make_generators(5)
    draws = {k: g.random() for k, g in g1.items()}
    assert len(set(draws.values())) == len(draws)
    g2["sensors"].random(1000)
    assert g2["targets"].random() == draws["targets"]


def test_run_file_round_trip_and_repeat(physics, tmp_path):
    s3 = scenario("s3_gust")
    paths = []
    for name in ("a", "b"):
        result = run_episode(Sim(physics), Spinner(), 7, s3)
        header = {"type": "header", "schema": 1, "scenario": s3["name"], "controller": "spinner",
                  "seed": 7, "commit": "test"}
        summary = {"type": "summary", "metrics": {"ticks": len(result.ticks)}}
        path = tmp_path / f"{name}.jsonl"
        with RunWriter(path) as w:
            w.write(header)
            for r in result.ticks:
                w.write(r)
            w.write(summary)
        h, ticks, s = read_run(path)
        assert h == header
        assert ticks == [round_tick(r) for r in result.ticks]
        assert s == summary
        paths.append(path)
    assert paths[0].read_bytes() == paths[1].read_bytes()
    first = json.loads(paths[0].read_text(encoding="utf-8").splitlines()[1])
    assert set(first) == {"type", "k", "t", "state", "wind", "target", "obs", "action", "info",
                          "events"}
    assert first["k"] == 0 and first["t"] == 0.0


def test_run_file_rounding(physics, tmp_path):
    result = run_episode(Sim(physics), Scripted(), 5, scenario("s4_turbulent"))
    path = tmp_path / "r.jsonl"
    with RunWriter(path) as w:
        w.write({"type": "header", "controller": "scripted"})
        for r in result.ticks:
            w.write(r)
    _, ticks, _ = read_run(path)

    def places(v: float) -> int:
        text = repr(v)
        return len(text.split(".")[1]) if "." in text and "e" not in text else 0

    for r in ticks[:200]:
        assert places(r["t"]) <= 4
        assert all(places(r["state"][k]) <= 3 for k in ("x", "y", "vx", "vy"))
        assert places(r["state"]["psi"]) <= 4
        assert all(places(v) <= 3 for v in r["wind"])
        assert places(r["obs"]["bearing"]) <= 4
        assert all(places(v) <= 3 for v in r["obs"]["vel_body"])
