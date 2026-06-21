"""The only public reporting entry point for RCE.

This module implements Section 7 of docs/theory/Relational_UQ_Formalization.md: the
reporting function never returns a bare RCE value. It returns the estimate together with a
permutation-null p-value and a bootstrap confidence interval, and it raises if asked to
skip the null. This is a structural rule, enforced by a test, because a bare RCE invites
the finite-sample-artifact misreading that the null exists to rule out.

The reporter also enforces the within-language control (Section 8): the data passed to a
single report must use one language, or the call raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from relcal.calibration import DEFAULT_N_BINS, BinningScheme, RCEComponents, relational_calibration_error_components
from relcal.permutation import bootstrap_ci, permutation_test
from relcal.schema import PreferenceDataset


@dataclass(frozen=True)
class RCEReport:
    """An RCE estimate with its null p-value and bootstrap interval, never bare."""

    rce: float
    p_value: float
    ci_low: float
    ci_high: float
    confidence: float
    n_permutations: int
    n_boot: int
    language: str
    n_items: int
    n_observations: int
    components: RCEComponents

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def report_rce(
    data: PreferenceDataset,
    predictions: np.ndarray,
    *,
    run_permutation_null: bool = True,
    n_permutations: int = 1000,
    n_boot: int = 1000,
    confidence: float = 0.95,
    n_bins: int = DEFAULT_N_BINS,
    scheme: BinningScheme = "adaptive",
    debias: bool = True,
    rng_seed: int = 0,
) -> RCEReport:
    """Estimate RCE with a permutation-null p-value and a bootstrap confidence interval.

    Args:
        data: The preference dataset. Must use a single language (within-language control).
        predictions: Per-observation predicted preference probabilities, aligned row by row
            with ``data.arrays()`` (that is, with the order of ``data.records``).
        run_permutation_null: Must be True. Passing False raises, by design: the null may
            not be disabled. This is the structural no-bare-RCE rule.

    Returns:
        An RCEReport bundling the estimate, p-value, and confidence interval.

    Raises:
        ValueError: if ``run_permutation_null`` is False, if the data span more than one
            language, or if the predictions do not align with the data.
    """
    if not run_permutation_null:
        raise ValueError(
            "the permutation null may not be disabled: report_rce never returns a bare "
            "RCE value. It returns RCE only together with a permutation-null p-value and a "
            "bootstrap confidence interval (Section 7 of the formalization)."
        )

    language = data.require_single_language()
    arrays = data.arrays()
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape[0] != len(arrays):
        raise ValueError(
            f"predictions length {predictions.shape[0]} does not match the number of "
            f"observations {len(arrays)}"
        )

    components = relational_calibration_error_components(
        predictions, arrays.judgment, arrays.context, n_bins=n_bins, scheme=scheme, debias=debias
    )
    perm = permutation_test(
        predictions,
        arrays.judgment,
        arrays.context,
        arrays.item_id,
        n_permutations=n_permutations,
        n_bins=n_bins,
        scheme=scheme,
        debias=debias,
        rng=rng_seed,
    )
    boot = bootstrap_ci(
        predictions,
        arrays.judgment,
        arrays.context,
        arrays.item_id,
        n_boot=n_boot,
        confidence=confidence,
        n_bins=n_bins,
        scheme=scheme,
        debias=debias,
        rng=rng_seed + 1,
    )

    n_items = len(set(arrays.item_id.tolist()))
    return RCEReport(
        rce=components.rce,
        p_value=perm.p_value,
        ci_low=boot.low,
        ci_high=boot.high,
        confidence=confidence,
        n_permutations=n_permutations,
        n_boot=n_boot,
        language=language,
        n_items=n_items,
        n_observations=len(arrays),
        components=components,
    )


@dataclass(frozen=True)
class RelationalControlComparison:
    """The paired relational and control reports of a within-language comparison."""

    language: str
    relational: RCEReport
    control: RCEReport


def compare_relational_vs_control(
    relational_data: PreferenceDataset,
    relational_predictions: np.ndarray,
    control_data: PreferenceDataset,
    control_predictions: np.ndarray,
    *,
    n_permutations: int = 1000,
    n_boot: int = 1000,
    confidence: float = 0.95,
    n_bins: int = DEFAULT_N_BINS,
    scheme: BinningScheme = "adaptive",
    debias: bool = True,
    rng_seed: int = 0,
) -> RelationalControlComparison:
    """Report RCE on the relational set and the control set, in one language.

    This is the comparison entry point. It enforces the within-language control across the
    two sets (Section 8): the relational set and the control set must share a single
    language. Reporting them separately would otherwise let a relational set in one language
    be compared against a control set in another, which is exactly the confound the control
    rules out. The two sets are reported by separate ``report_rce`` calls so that each RCE
    is estimated within its own set, but the language match is checked first.
    """
    rel_language = relational_data.require_single_language()
    ctl_language = control_data.require_single_language()
    if rel_language != ctl_language:
        raise ValueError(
            "within-language control violated: the relational set is in language "
            f"{rel_language!r} but the control set is in language {ctl_language!r}. A "
            "relational-versus-control comparison must hold the language fixed."
        )

    relational_report = report_rce(
        relational_data, relational_predictions,
        n_permutations=n_permutations, n_boot=n_boot, confidence=confidence,
        n_bins=n_bins, scheme=scheme, debias=debias, rng_seed=rng_seed,
    )
    control_report = report_rce(
        control_data, control_predictions,
        n_permutations=n_permutations, n_boot=n_boot, confidence=confidence,
        n_bins=n_bins, scheme=scheme, debias=debias, rng_seed=rng_seed + 7,
    )
    return RelationalControlComparison(
        language=rel_language,
        relational=relational_report,
        control=control_report,
    )


def format_report(report: RCEReport, *, title: Optional[str] = None) -> str:
    """A human-readable multi-line rendering of an RCEReport."""
    lines = []
    if title:
        lines.append(title)
    pct = int(round(report.confidence * 100))
    lines.append(
        f"  RCE        = {report.rce:+.4f}  "
        f"(p = {report.p_value:.4f}, {pct}% CI [{report.ci_low:+.4f}, {report.ci_high:+.4f}])"
    )
    lines.append(
        f"  pooled ECE = {report.components.pooled_ece:.4f}   "
        f"weighted within-context ECE = {report.components.weighted_within_ece:.4f}"
    )
    lines.append(
        f"  language   = {report.language}   items = {report.n_items}   "
        f"observations = {report.n_observations}"
    )
    lines.append(
        f"  permutations = {report.n_permutations}   bootstrap resamples = {report.n_boot}"
    )
    return "\n".join(lines)
