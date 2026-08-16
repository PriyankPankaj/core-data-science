"""Tests verifying statistical functions against known reference cases."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from src.decision_engine import make_decision


def test_proportions_ztest_identical_groups_not_significant():
    """Two identical groups should never be significant."""
    count = np.array([500, 500])
    nobs = np.array([1000, 1000])
    _, p_value = proportions_ztest(count, nobs)
    assert p_value > 0.05


def test_proportions_ztest_large_difference_significant():
    """A large, obvious difference should be significant."""
    count = np.array([800, 200])
    nobs = np.array([1000, 1000])
    _, p_value = proportions_ztest(count, nobs)
    assert p_value < 0.05


def test_decision_engine_ships_significant_positive_effect():
    result = {"p_value": 0.001, "absolute_lift": 0.02, "significant": True}
    decision = make_decision("test_metric", result, randomization_valid=True, power_achieved=0.9)
    assert decision["decision"] == "SHIP"


def test_decision_engine_rejects_negative_significant_effect():
    result = {"p_value": 0.001, "absolute_lift": -0.02, "significant": True}
    decision = make_decision("test_metric", result, randomization_valid=True, power_achieved=0.9)
    assert decision["decision"] == "DO NOT SHIP"


def test_decision_engine_flags_invalid_randomization():
    result = {"p_value": 0.001, "absolute_lift": 0.02, "significant": True}
    decision = make_decision("test_metric", result, randomization_valid=False)
    assert decision["decision"] == "INCONCLUSIVE"