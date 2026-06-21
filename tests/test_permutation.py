"""Tests for the permutation null and bootstrap (Section 6)."""

from __future__ import annotations

import numpy as np

from relcal import simulate
from relcal.permutation import bootstrap_ci, permutation_test


def test_false_positive_rate_near_nominal_under_no_structure():
    """Under no structure the permutation test rejects at about its nominal rate.

    We run a batch of independent no-structure datasets and check the empirical
    false-positive rate at alpha = 0.05 is not materially above nominal. The bound is
    generous to keep the test fast and non-flaky while still catching an estimator that
    rejects far too often (which a biased, null-free RCE would).
    """
    alpha = 0.05
    n_trials = 30
    rejections = 0
    for s in range(n_trials):
        ds, truth = simulate.generate_no_structure(n_items=100, n_raters=12, seed=500 + s)
        arrays = ds.arrays()
        preds = truth.marginal_predictions(arrays)
        result = permutation_test(
            preds, arrays.judgment, arrays.context, arrays.item_id,
            n_permutations=150, rng=s,
        )
        rejections += int(result.p_value < alpha)
    fpr = rejections / n_trials
    # Nominal 0.05; allow up to 0.20 to absorb Monte Carlo noise over 30 trials.
    assert fpr <= 0.20, f"false-positive rate {fpr} materially exceeds nominal {alpha}"


def test_permutation_has_power_under_structure():
    """With real relational structure the permutation test rejects the null."""
    ds, truth = simulate.generate(n_items=200, n_raters=30, dispersion=0.15, seed=1)
    arrays = ds.arrays()
    preds = truth.marginal_predictions(arrays)
    result = permutation_test(
        preds, arrays.judgment, arrays.context, arrays.item_id, n_permutations=200, rng=1
    )
    assert result.observed > 0.05
    assert result.p_value < 0.05


def test_pvalue_respects_add_one_bounds():
    ds, truth = simulate.generate(n_items=80, n_raters=10, dispersion=0.1, seed=2)
    arrays = ds.arrays()
    preds = truth.marginal_predictions(arrays)
    b = 100
    result = permutation_test(
        preds, arrays.judgment, arrays.context, arrays.item_id, n_permutations=b, rng=2
    )
    assert 1 / (1 + b) <= result.p_value <= 1.0


def test_bootstrap_ci_brackets_point_and_orders_bounds():
    ds, truth = simulate.generate(n_items=200, n_raters=30, dispersion=0.15, seed=4)
    arrays = ds.arrays()
    preds = truth.marginal_predictions(arrays)
    boot = bootstrap_ci(
        preds, arrays.judgment, arrays.context, arrays.item_id, n_boot=300, rng=4
    )
    assert boot.low <= boot.high
    assert boot.low <= boot.point <= boot.high
