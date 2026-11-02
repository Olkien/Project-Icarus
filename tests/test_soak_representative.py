import numpy as np
import pytest

from analysis.soak_representative import B_TICKS, clopper_pearson, family_b_obs, rate_row


def test_clopper_pearson_known_values():
    assert clopper_pearson(0, 50) == pytest.approx((0.0, 0.0711), abs=1e-4)
    assert clopper_pearson(50, 50) == pytest.approx((0.9289, 1.0), abs=1e-4)
    low, high = clopper_pearson(10, 50)
    assert low < 0.2 < high


def test_rate_row_handles_family_not_run():
    assert rate_row(0, 0) == "0 0 - -"
    assert rate_row(0, 50) == "0 50 0.0000 0.0711"


def test_family_b_is_seeded_and_in_bounds():
    a, drawn = family_b_obs(2000)
    b, _ = family_b_obs(2000)
    assert a == b and len(a) == B_TICKS
    assert -2.4 <= drawn["hold_rad"] <= -1.2
    assert -0.2 <= drawn["end_rad"] <= 0.2
    assert 160 <= drawn["hold_ticks"] <= 240
    assert 40 <= drawn["swing_ticks"] <= 100
    assert drawn != family_b_obs(2001)[1]


def test_family_b_swings_back_with_noise():
    obs, _ = family_b_obs(2000)
    bearing = np.array([o["bearing"] for o in obs])
    assert bearing[:100].mean() < -1.0
    assert abs(bearing[-500:].mean()) < 0.4
    assert bearing[-500:].std() > 0.05
