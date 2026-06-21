"""Permutation null and bootstrap confidence interval for RCE.

This module implements Section 6 of docs/theory/Relational_UQ_Formalization.md.

The permutation null shuffles the context labels within each item, preserving each item's
set of labels and its per-context counts while breaking the association between context and
outcome. Under the null that context carries no information beyond the item, the labels are
exchangeable within an item, so the shuffled statistics form a valid reference distribution.
The observed statistic is compared against statistics recomputed under the same binning and
sample size, which automatically calibrates away the shared finite-sample bias.

The bootstrap resamples items with replacement, the item being the independent unit, and
recomputes RCE on each resample to form a percentile confidence interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from relcal.calibration import DEFAULT_N_BINS, BinningScheme, relational_calibration_error

RngLike = Union[int, np.random.Generator, None]


def _as_rng(rng: RngLike) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _item_groups(item_ids: np.ndarray) -> list[np.ndarray]:
    """Row indices grouped by item, in first-seen item order."""
    groups: dict = {}
    for idx, item in enumerate(item_ids.tolist()):
        groups.setdefault(item, []).append(idx)
    return [np.asarray(v, dtype=int) for v in groups.values()]


def _permute_contexts_within_item(
    contexts: np.ndarray, groups: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Return a copy of contexts with labels permuted within each item group."""
    permuted = contexts.copy()
    for idx in groups:
        permuted[idx] = rng.permutation(contexts[idx])
    return permuted


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    p_value: float
    n_permutations: int
    null_mean: float
    null_quantile_95: float


def permutation_test(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    contexts: np.ndarray,
    item_ids: np.ndarray,
    *,
    n_permutations: int = 1000,
    n_bins: int = DEFAULT_N_BINS,
    scheme: BinningScheme = "adaptive",
    debias: bool = True,
    rng: RngLike = None,
) -> PermutationResult:
    """Permutation-null test for RCE (Section 6.1).

    The p-value uses the add-one correction:

        p = (1 + #{ RCE_perm >= RCE_obs }) / (1 + B).
    """
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    contexts = np.asarray(contexts, dtype=object)
    item_ids = np.asarray(item_ids, dtype=object)
    if not (predictions.shape == outcomes.shape == contexts.shape == item_ids.shape):
        raise ValueError("predictions, outcomes, contexts, item_ids must align in shape")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")

    generator = _as_rng(rng)
    groups = _item_groups(item_ids)

    def statistic(ctx: np.ndarray) -> float:
        return relational_calibration_error(
            predictions, outcomes, ctx, n_bins=n_bins, scheme=scheme, debias=debias
        )

    observed = statistic(contexts)

    null = np.empty(n_permutations, dtype=float)
    for b in range(n_permutations):
        permuted = _permute_contexts_within_item(contexts, groups, generator)
        null[b] = statistic(permuted)

    n_ge = int(np.sum(null >= observed))
    p_value = (1 + n_ge) / (1 + n_permutations)
    return PermutationResult(
        observed=float(observed),
        p_value=float(p_value),
        n_permutations=n_permutations,
        null_mean=float(null.mean()),
        null_quantile_95=float(np.quantile(null, 0.95)),
    )


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    low: float
    high: float
    confidence: float
    n_boot: int


def bootstrap_ci(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    contexts: np.ndarray,
    item_ids: np.ndarray,
    *,
    n_boot: int = 1000,
    confidence: float = 0.95,
    n_bins: int = DEFAULT_N_BINS,
    scheme: BinningScheme = "adaptive",
    debias: bool = True,
    rng: RngLike = None,
) -> BootstrapResult:
    """Percentile bootstrap confidence interval for RCE, resampling items (Section 6.2)."""
    predictions = np.asarray(predictions, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    contexts = np.asarray(contexts, dtype=object)
    item_ids = np.asarray(item_ids, dtype=object)
    if not (predictions.shape == outcomes.shape == contexts.shape == item_ids.shape):
        raise ValueError("predictions, outcomes, contexts, item_ids must align in shape")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    generator = _as_rng(rng)
    groups = _item_groups(item_ids)
    n_items = len(groups)

    point = relational_calibration_error(
        predictions, outcomes, contexts, n_bins=n_bins, scheme=scheme, debias=debias
    )

    estimates = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        chosen = generator.integers(0, n_items, size=n_items)
        rows = np.concatenate([groups[i] for i in chosen])
        estimates[b] = relational_calibration_error(
            predictions[rows],
            outcomes[rows],
            contexts[rows],
            n_bins=n_bins,
            scheme=scheme,
            debias=debias,
        )

    alpha = 1.0 - confidence
    low = float(np.quantile(estimates, alpha / 2))
    high = float(np.quantile(estimates, 1.0 - alpha / 2))
    return BootstrapResult(
        point=float(point),
        low=low,
        high=high,
        confidence=confidence,
        n_boot=n_boot,
    )
