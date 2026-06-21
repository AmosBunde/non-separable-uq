"""Tests for the calibration estimators (Propositions 2 and 3)."""

from __future__ import annotations

import numpy as np
import pytest

from relcal import simulate
from relcal.calibration import (
    expected_calibration_error,
    relational_calibration_error,
    relational_calibration_error_components,
)


def test_rce_non_negative_on_random_valid_inputs():
    """Proposition 2: RCE >= 0 on random valid inputs.

    Proposition 2 is a statement about the calibration audit: refining the audit by context
    can only reveal more miscalibration. The binned raw estimator is the audit quantity and
    is upward biased, so it respects non-negativity up to a tiny finite-sample tolerance.
    The debiased estimator targets the population value and is near-unbiased; at a true RCE
    of zero it can dip slightly negative by design, so non-negativity is asserted on the raw
    estimator, which is the faithful test of the proposition (see Section 4.2 of the
    formalization).
    """
    rng = np.random.default_rng(0)
    tol = 1e-2
    for trial in range(50):
        n = rng.integers(200, 1200)
        predictions = rng.uniform(0.0, 1.0, size=n)
        outcomes = (rng.uniform(0.0, 1.0, size=n) < predictions).astype(int)
        n_contexts = int(rng.integers(2, 5))
        contexts = rng.integers(0, n_contexts, size=n).astype(object)
        rce = relational_calibration_error(predictions, outcomes, contexts, debias=False)
        assert rce >= -tol, f"trial {trial}: raw RCE={rce} below tolerance {tol}"


def test_rce_matches_analytic_value_within_tolerance():
    """Proposition 3: on synthetic data, estimated RCE is close to the analytic MARD.

    The test is two-sided (|RCE - MARD| < tol), not one-sided. The estimate approaches
    MARD from below by the finite-sample pooled-ECE bias; at this sample size that gap is
    small. A one-sided RCE >= MARD assertion would encode the wrong direction.
    """
    tol = 0.02
    for d in (0.05, 0.10, 0.20):
        ds, truth = simulate.generate(n_items=400, n_raters=100, dispersion=d, seed=7)
        arrays = ds.arrays()
        preds = truth.marginal_predictions(arrays)
        rce = relational_calibration_error(preds, arrays.judgment, arrays.context)
        assert abs(rce - truth.mard) < tol, (
            f"d={d}: RCE={rce:.4f} not within {tol} of MARD={truth.mard:.4f}"
        )


def test_rce_below_or_at_mard_for_marginal_model():
    """The marginal model's RCE sits at or just below MARD, never far above it.

    This pins the measured direction (Section 4.4 of the formalization): the point estimate
    approaches MARD from below. We assert it does not exceed MARD by more than a small
    margin.
    """
    ds, truth = simulate.generate(n_items=400, n_raters=100, dispersion=0.1, seed=7)
    arrays = ds.arrays()
    preds = truth.marginal_predictions(arrays)
    comp = relational_calibration_error_components(preds, arrays.judgment, arrays.context)
    assert comp.rce <= truth.mard + 0.01
    assert comp.pooled_ece >= 0.0


def test_rce_recovers_zero_under_no_structure():
    """Under the no-structure generator the population RCE is zero."""
    ds, truth = simulate.generate_no_structure(n_items=400, n_raters=60, seed=3)
    arrays = ds.arrays()
    preds = truth.marginal_predictions(arrays)
    rce = relational_calibration_error(preds, arrays.judgment, arrays.context)
    assert abs(rce) < 0.02, f"RCE={rce} should be near zero under no structure"


def test_oracle_model_has_near_zero_rce():
    """A context-aware oracle f(x,c)=pi(x,c) is within-context calibrated, so RCE ~ 0."""
    ds, truth = simulate.generate(n_items=400, n_raters=100, dispersion=0.2, seed=5)
    arrays = ds.arrays()
    preds = truth.oracle_predictions(arrays)
    rce = relational_calibration_error(preds, arrays.judgment, arrays.context)
    assert abs(rce) < 0.02, f"oracle RCE={rce} should be near zero"


def test_ece_rejects_out_of_range_predictions():
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0.5, 1.5]), np.array([0, 1]))


def test_ece_rejects_non_binary_outcomes():
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0.5, 0.5]), np.array([0, 2]))
