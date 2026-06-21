"""Calibration estimators: pooled ECE, within-context ECE_c, and RCE.

This module implements Section 3 and Section 4 of
docs/theory/Relational_UQ_Formalization.md. All errors are L1 (absolute) expected
calibration errors.

Finite-sample bias of binned ECE
--------------------------------
The binned estimate of an absolute calibration gap is biased upward. Within a bin the
empirical outcome frequency ``acc_b`` carries sampling noise around the true conditional
outcome, and the absolute value is convex, so

    E | acc_b - conf_b | >= | E[acc_b] - conf_b |.

The inflation grows as bins become sparse, which is exactly the within-context regime
where each context holds only a share of the data. Left uncorrected, this inflates ECE_c
more than pooled ECE and therefore inflates RCE.

Two mitigations are provided and are on by default:

1. Adaptive (equal-mass) binning. Quantile bin edges keep every bin populated, which
   lowers per-bin variance relative to fixed-width binning on a skewed prediction
   distribution.

2. A debiased per-bin gap. For an L2 (squared) gap the bias is exactly removable, because
   E[(acc_b - conf_b) ** 2] = (true gap) ** 2 + Var(acc_b). We estimate the variance of the
   bin mean for independent Bernoulli outcomes as

       var_hat_b = acc_b * (1 - acc_b) / (n_b - 1),  for n_b > 1,

   and form the debiased absolute gap

       gap_b = sqrt( max(0, (acc_b - conf_b) ** 2 - var_hat_b) ).

   This removes the leading-order variance inflation of the absolute gap and floors the
   result at zero. Assumptions: outcomes within a bin are independent Bernoulli draws and
   each populated bin has more than one observation. With a single observation the bin
   variance is undefined and the raw gap is used. The debiased estimator targets the same
   population estimands ECE and ECE_c; it changes only the finite-sample correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

BinningScheme = Literal["adaptive", "uniform"]

DEFAULT_N_BINS = 15


def _validate_inputs(predictions: np.ndarray, outcomes: np.ndarray) -> None:
    if predictions.shape != outcomes.shape:
        raise ValueError(
            f"predictions and outcomes must align: {predictions.shape} vs {outcomes.shape}"
        )
    if predictions.ndim != 1:
        raise ValueError("predictions and outcomes must be one-dimensional")
    if predictions.size == 0:
        raise ValueError("cannot compute calibration error on empty input")
    if np.any((predictions < 0.0) | (predictions > 1.0)):
        raise ValueError("predictions must lie in [0, 1]")
    unique_outcomes = np.unique(outcomes)
    if not np.all(np.isin(unique_outcomes, (0, 1))):
        raise ValueError("outcomes must be binary (0 or 1)")


def _bin_assignments(
    predictions: np.ndarray, n_bins: int, scheme: BinningScheme
) -> np.ndarray:
    """Assign each prediction to a bin index in [0, n_bins_effective).

    For uniform binning the edges are fixed on [0, 1]. For adaptive binning the edges are
    prediction quantiles, so bins carry approximately equal counts. Degenerate cases (fewer
    unique predictions than requested bins) collapse to as many bins as there are distinct
    edges.
    """
    n_points = predictions.shape[0]
    n_bins = max(1, min(n_bins, n_points))

    if scheme == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif scheme == "adaptive":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(predictions, quantiles)
        edges = np.unique(edges)
        if edges.size < 2:
            # All predictions identical: a single bin.
            return np.zeros(n_points, dtype=int)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown binning scheme {scheme!r}")

    # Interior edges only; clip assignments into the valid bin range.
    interior = edges[1:-1]
    assignments = np.digitize(predictions, interior, right=False)
    n_effective = interior.shape[0] + 1
    np.clip(assignments, 0, n_effective - 1, out=assignments)
    return assignments


def _bin_gaps(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    assignments: np.ndarray,
    debias: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (weights, gaps) over populated bins.

    weights[b] = n_b / N and gaps[b] is the per-bin absolute calibration gap, debiased if
    requested. Empty bins are dropped.
    """
    total = predictions.shape[0]
    weights = []
    gaps = []
    for b in np.unique(assignments):
        mask = assignments == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        conf_b = float(predictions[mask].mean())
        acc_b = float(outcomes[mask].mean())
        raw_gap = abs(acc_b - conf_b)
        if debias and n_b > 1:
            var_hat = acc_b * (1.0 - acc_b) / (n_b - 1)
            debiased_sq = max(0.0, raw_gap * raw_gap - var_hat)
            gap = float(np.sqrt(debiased_sq))
        else:
            gap = raw_gap
        weights.append(n_b / total)
        gaps.append(gap)
    return np.asarray(weights, dtype=float), np.asarray(gaps, dtype=float)


def expected_calibration_error(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_BINS,
    scheme: BinningScheme = "adaptive",
    debias: bool = True,
) -> float:
    """Pooled expected calibration error (Section 3.1).

    The L1 calibration error sum_b (n_b / N) | acc_b - conf_b |, with adaptive binning and
    debiasing on by default. See the module docstring for the bias discussion.
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    _validate_inputs(predictions, outcomes)
    assignments = _bin_assignments(predictions, n_bins, scheme)
    weights, gaps = _bin_gaps(predictions, outcomes, assignments, debias)
    return float(np.sum(weights * gaps))


def within_context_calibration_errors(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    contexts: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_BINS,
    scheme: BinningScheme = "adaptive",
    debias: bool = True,
) -> tuple[dict[str, float], dict[str, float]]:
    """Within-context calibration errors and their context weights (Section 3.1, 4.1).

    Returns (ece_c, rho) where ece_c maps each context label to ECE_c, and rho maps each
    context label to its mass weight n_c / N. The weights are the rho(c) used by RCE.
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    contexts = np.asarray(contexts, dtype=object)
    _validate_inputs(predictions, outcomes)
    if contexts.shape != predictions.shape:
        raise ValueError("contexts must align with predictions and outcomes")

    total = predictions.shape[0]
    ece_c: dict[str, float] = {}
    rho: dict[str, float] = {}
    for c in _ordered_unique(contexts):
        mask = contexts == c
        n_c = int(mask.sum())
        ece_c[c] = expected_calibration_error(
            predictions[mask],
            outcomes[mask],
            n_bins=n_bins,
            scheme=scheme,
            debias=debias,
        )
        rho[c] = n_c / total
    return ece_c, rho


@dataclass(frozen=True)
class RCEComponents:
    """The pieces of an RCE computation, for transparency in reporting."""

    rce: float
    pooled_ece: float
    weighted_within_ece: float
    per_context_ece: dict[str, float]
    context_weights: dict[str, float]


def relational_calibration_error(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    contexts: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_BINS,
    scheme: BinningScheme = "adaptive",
    debias: bool = True,
) -> float:
    """Relational Calibration Error (Section 4.1).

    RCE = ( sum_c rho(c) ECE_c ) - pooled ECE. Non-negative in the population
    (Proposition 2); the debiased estimator concentrates around the non-negative value.
    """
    return relational_calibration_error_components(
        predictions,
        outcomes,
        contexts,
        n_bins=n_bins,
        scheme=scheme,
        debias=debias,
    ).rce


def relational_calibration_error_components(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    contexts: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_BINS,
    scheme: BinningScheme = "adaptive",
    debias: bool = True,
) -> RCEComponents:
    """RCE together with its pooled and within-context components."""
    pooled = expected_calibration_error(
        predictions, outcomes, n_bins=n_bins, scheme=scheme, debias=debias
    )
    ece_c, rho = within_context_calibration_errors(
        predictions, outcomes, contexts, n_bins=n_bins, scheme=scheme, debias=debias
    )
    weighted_within = float(sum(rho[c] * ece_c[c] for c in ece_c))
    return RCEComponents(
        rce=weighted_within - pooled,
        pooled_ece=pooled,
        weighted_within_ece=weighted_within,
        per_context_ece=ece_c,
        context_weights=rho,
    )


def _ordered_unique(values: np.ndarray) -> list:
    """Distinct values in first-seen order (np.unique sorts, which we do not want here)."""
    seen: dict = {}
    for v in values.tolist():
        seen.setdefault(v, None)
    return list(seen.keys())
