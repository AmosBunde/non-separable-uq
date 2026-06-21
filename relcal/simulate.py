"""Synthetic preference generators with a known relational dispersion.

This module implements the ground-truth side of Section 4.3 and Section 6 of
docs/theory/Relational_UQ_Formalization.md. It produces binary rater judgments whose
underlying preference structure has an exactly controllable mean absolute relational
dispersion (MARD), so the RCE estimator can be checked against an analytic value.

Design for an exact analytic target
-----------------------------------
The context offset is constant across items: pi(x, c) = clip(bar_pi(x) + delta(c)), where
delta(c) depends only on the context. With this construction the within-context gap of the
marginal model f(x) = bar_pi(x) equals |delta(c)| for every item in context c, so

    ECE_c(bar_pi) = |delta(c)|   exactly, independent of the binning,

because there is no within-bin averaging across items to trigger Jensen shrinkage. The
offsets are built to be zero-mean across contexts and to have mean absolute value equal to
the requested dispersion d:

    mean_c rho(c) delta(c) = 0,      mean_c rho(c) |delta(c)| = d.

Hence, in the population, RCE(bar_pi) = MARD = d and pooled ECE(bar_pi) = 0. This is the
analytic target the estimator is checked against. Real preference data need not have
constant offsets; this generator is a validation harness with a known answer, not a model
of real annotation.

The realized MARD is also computed exactly from the constructed pi matrix and returned, so
any future generator variant can still be checked against its own ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from relcal.schema import Arrays, Judgment, PreferenceDataset

DEFAULT_CONTEXTS: tuple[str, ...] = ("context_a", "context_b")


def _context_offsets(n_contexts: int, dispersion: float) -> np.ndarray:
    """Zero-mean offsets with mean absolute value equal to ``dispersion``.

    A symmetric base pattern is centered (already zero mean) and rescaled so its mean
    absolute value is one, then multiplied by the dispersion. For two contexts this gives
    (-d, +d); for three it gives (-1.5 d, 0, +1.5 d); and so on.
    """
    if n_contexts < 2:
        raise ValueError("need at least two contexts for relational structure")
    base = np.linspace(-1.0, 1.0, n_contexts)
    mean_abs = float(np.mean(np.abs(base)))
    if mean_abs == 0.0:  # pragma: no cover - only if n_contexts == 1, already guarded
        raise ValueError("degenerate context pattern")
    scaled = base / mean_abs
    return dispersion * scaled


@dataclass(frozen=True)
class SyntheticTruth:
    """Exact ground truth for a synthetic dataset."""

    item_ids: list[str]
    contexts: list[str]
    pi: np.ndarray  # shape (n_items, n_contexts): true context-conditional preference
    bar_pi: np.ndarray  # shape (n_items,): context-marginal preference
    context_weights: np.ndarray  # shape (n_contexts,): rho(c)
    dispersion: float  # the requested d parameter
    mard: float  # realized mean absolute relational dispersion, computed from pi

    def _item_index(self) -> dict[str, int]:
        return {item: i for i, item in enumerate(self.item_ids)}

    def marginal_predictions(self, arrays: Arrays) -> np.ndarray:
        """The marginal model f(x) = bar_pi(x), aligned to the rows of ``arrays``.

        The prediction ignores the context, which is the context-blind model of
        Proposition 3.
        """
        index = self._item_index()
        return np.array([self.bar_pi[index[item]] for item in arrays.item_id], dtype=float)

    def oracle_predictions(self, arrays: Arrays) -> np.ndarray:
        """The context-aware oracle f(x, c) = pi(x, c), aligned to the rows of ``arrays``.

        Its RCE is zero in the population; useful as a contrast in experiments.
        """
        item_index = self._item_index()
        ctx_index = {c: j for j, c in enumerate(self.contexts)}
        return np.array(
            [self.pi[item_index[i], ctx_index[c]] for i, c in zip(arrays.item_id, arrays.context)],
            dtype=float,
        )


def generate(
    *,
    n_items: int = 200,
    contexts: Sequence[str] = DEFAULT_CONTEXTS,
    n_raters: int = 30,
    dispersion: float = 0.1,
    marginal_spread: float = 0.25,
    relational: bool = True,
    language: str = "synthetic",
    seed: int = 0,
    item_prefix: str = "item",
) -> tuple[PreferenceDataset, SyntheticTruth]:
    """Generate synthetic judgments with a known relational dispersion.

    Args:
        n_items: Number of preference items.
        contexts: Context labels. Contexts are balanced (equal weight rho(c)).
        n_raters: Independent binary judgments per (item, context).
        dispersion: The mean absolute relational dispersion d. Use 0.0 for the
            no-structure generator (true RCE is zero).
        marginal_spread: Half-width of the band the marginal preference is drawn from,
            centered at 0.5. Must leave room for the largest offset so that no clipping is
            needed (clipping would break the exact analytic target).
        relational: Value of the relational flag on every generated item.
        language: Single language for all items (satisfies the within-language control).
        seed: Seed for the random generator.
        item_prefix: Prefix for generated item identifiers.

    Returns:
        (dataset, truth). The dataset holds sampled binary judgments; truth holds the exact
        pi matrix, marginal, dispersion, and realized MARD.
    """
    if dispersion < 0.0:
        raise ValueError("dispersion must be non-negative")
    if n_items < 1 or n_raters < 1:
        raise ValueError("n_items and n_raters must be positive")

    context_list = list(contexts)
    n_contexts = len(context_list)
    offsets = _context_offsets(n_contexts, dispersion) if dispersion > 0 else np.zeros(n_contexts)

    max_offset = float(np.max(np.abs(offsets))) if dispersion > 0 else 0.0
    if marginal_spread + max_offset > 0.5:
        raise ValueError(
            "marginal_spread plus the largest context offset exceeds the unit interval; "
            f"reduce dispersion ({dispersion}) or marginal_spread ({marginal_spread}). "
            "Clipping is disallowed because it would break the exact analytic target."
        )

    rng = np.random.default_rng(seed)
    bar_pi = rng.uniform(0.5 - marginal_spread, 0.5 + marginal_spread, size=n_items)

    pi = np.clip(bar_pi[:, None] + offsets[None, :], 0.0, 1.0)
    # With the spread constraint above, the clip is a no-op; it remains as a guard.

    item_ids = [f"{item_prefix}_{i:04d}" for i in range(n_items)]
    context_weights = np.full(n_contexts, 1.0 / n_contexts)

    # Realized MARD from the constructed pi, computed exactly (not from samples).
    abs_dispersion = np.abs(pi - bar_pi[:, None])
    mard = float(np.sum(context_weights[None, :] * abs_dispersion) / n_items)

    records: list[Judgment] = []
    for i, item_id in enumerate(item_ids):
        for j, context in enumerate(context_list):
            draws = rng.binomial(1, pi[i, j], size=n_raters)
            for k in range(n_raters):
                records.append(
                    Judgment(
                        item_id=item_id,
                        language=language,
                        context=context,
                        relational=relational,
                        judgment=int(draws[k]),
                        rater_id=f"rater_{k:03d}",
                    )
                )

    truth = SyntheticTruth(
        item_ids=item_ids,
        contexts=context_list,
        pi=pi,
        bar_pi=bar_pi,
        context_weights=context_weights,
        dispersion=dispersion,
        mard=mard,
    )
    return PreferenceDataset(records), truth


def generate_no_structure(
    *,
    n_items: int = 200,
    contexts: Sequence[str] = DEFAULT_CONTEXTS,
    n_raters: int = 30,
    language: str = "synthetic",
    seed: int = 0,
) -> tuple[PreferenceDataset, SyntheticTruth]:
    """No-structure generator: the preference does not depend on context (true RCE = 0).

    A thin wrapper over ``generate`` with dispersion zero, used for the permutation null
    calibration check (Section 6.1).
    """
    return generate(
        n_items=n_items,
        contexts=contexts,
        n_raters=n_raters,
        dispersion=0.0,
        relational=False,
        language=language,
        seed=seed,
    )
