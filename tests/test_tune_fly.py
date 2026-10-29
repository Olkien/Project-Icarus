import json

import pytest

from analysis.tune_fly import SPACE_PATH, load_space, plan, read_log, tune
from controllers.fly import Fly

SHORT = {"time_limit_s": 1.0}


def tiny(evals: int = 3, stepper: str = "numpy") -> dict:
    cfg = load_space(SPACE_PATH)
    cfg.update({"evals": evals, "scenarios": ["s1_still", "s4_turbulent"], "seeds": [0],
                "fixed": {"stepper": stepper}})
    return cfg


def test_space_comes_from_param_space():
    cfg = load_space(SPACE_PATH)
    space = Fly.param_space()
    assert cfg["evals"] == 40
    assert cfg["scenarios"] == ["s1_still", "s2_crosswind", "s3_gust", "s4_turbulent"]
    assert cfg["seeds"] == [0, 1]
    for name, (lo, hi) in cfg["bounds"].items():
        assert tuple(space[name]) == (lo, hi)


def test_plan_is_seeded_and_in_bounds():
    cfg = load_space(SPACE_PATH)
    a, b = plan(cfg), plan(cfg)
    assert a == b and len(a) == 40
    assert a[0] == cfg["fixed"]
    for trial in a[1:]:
        for name, (lo, hi) in cfg["bounds"].items():
            assert lo <= trial[name] <= hi
    longer = plan({**cfg, "evals": 50})
    assert longer[:40] == a


def test_end_to_end_and_resume(tmp_path):
    summary = tune(tiny(3), tmp_path, overrides=SHORT)
    rows = read_log(tmp_path / "tune_fly_log.csv")
    assert [r["eval"] for r in rows] == [0, 1, 2]
    assert all(r["episodes"] == "2" for r in rows)
    assert summary["evals_run_now"] == 3
    assert summary["best_cost"] == min(r["cost"] for r in rows)
    saved = json.loads((tmp_path / "tune_fly.json").read_text(encoding="utf-8"))
    assert saved["best_eval"] == summary["best_eval"]

    again = tune(tiny(3), tmp_path, overrides=SHORT)
    assert again["evals_run_now"] == 0
    more = tune(tiny(5), tmp_path, overrides=SHORT)
    assert more["evals_run_now"] == 2
    assert [r["eval"] for r in read_log(tmp_path / "tune_fly_log.csv")] == [0, 1, 2, 3, 4]


def test_resume_refuses_other_settings(tmp_path):
    tune(tiny(1), tmp_path, overrides=SHORT)
    changed = tiny(1)
    changed["seeds"] = [1]
    with pytest.raises(ValueError):
        tune(changed, tmp_path, overrides=SHORT)


def test_parallel_matches_serial(tmp_path):
    serial = tune(tiny(2), tmp_path / "a", overrides=SHORT)
    parallel = tune(tiny(2), tmp_path / "b", jobs=2, overrides=SHORT)
    assert serial["best_cost"] == parallel["best_cost"]


def test_numba_stepper_gives_same_costs(tmp_path):
    pytest.importorskip("numba")
    a = tune(tiny(2, "numpy"), tmp_path / "np", overrides=SHORT)
    b = tune(tiny(2, "numba"), tmp_path / "nb", overrides=SHORT)
    rows_a = read_log(tmp_path / "np" / "tune_fly_log.csv")
    rows_b = read_log(tmp_path / "nb" / "tune_fly_log.csv")
    assert [r["cost"] for r in rows_a] == [r["cost"] for r in rows_b]
    assert a["best_eval"] == b["best_eval"]
