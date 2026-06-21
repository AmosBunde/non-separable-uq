"""Tests for the reporting contract (Section 7): no bare RCE."""

from __future__ import annotations

import numpy as np
import pytest

from relcal import simulate
from relcal.report import RCEReport, report_rce


def _synthetic_report_inputs(dispersion=0.15, seed=11):
    ds, truth = simulate.generate(n_items=120, n_raters=20, dispersion=dispersion, seed=seed)
    preds = truth.marginal_predictions(ds.arrays())
    return ds, preds


def test_report_raises_when_permutation_null_disabled():
    """The structural no-bare-RCE rule: disabling the null must raise."""
    ds, preds = _synthetic_report_inputs()
    with pytest.raises(ValueError, match="permutation null may not be disabled"):
        report_rce(ds, preds, run_permutation_null=False)


def test_report_bundles_estimate_pvalue_and_ci():
    """A valid report carries RCE together with a p-value and a confidence interval."""
    ds, preds = _synthetic_report_inputs()
    report = report_rce(ds, preds, n_permutations=100, n_boot=100, rng_seed=0)
    assert isinstance(report, RCEReport)
    assert np.isfinite(report.rce)
    assert 0.0 < report.p_value <= 1.0
    assert report.ci_low <= report.ci_high
    assert report.language == "synthetic"


def test_report_rejects_misaligned_predictions():
    ds, preds = _synthetic_report_inputs()
    with pytest.raises(ValueError, match="does not match"):
        report_rce(ds, preds[:-1], n_permutations=10, n_boot=10)
