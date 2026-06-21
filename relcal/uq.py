"""Uncertainty quantification interface.

A UQ backend returns, per item, a predicted preference probability and an epistemic
uncertainty. This supports Section 5 (Proposition 1) of
docs/theory/Relational_UQ_Formalization.md: a context-blind model with context-blind
uncertainty cannot have its uncertainty track relational miscalibration, because the
uncertainty never sees the context.

Two backends are provided:

- MockEnsembleUQ: a small ensemble of context-blind predictors over synthetic ground
  truth, for tests and for the pilot on the synthetic sample. Its epistemic uncertainty is
  the ensemble disagreement, which reflects estimation noise rather than relational
  dispersion. This is exactly the structure Proposition 1 describes; nothing is rigged.

- RealEnsembleUQ: a stub for a real open instruct or DPO model with an ensemble or
  MC-dropout uncertainty. It raises NotImplementedError until wired to a model. Unrun paths
  must not fabricate numbers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from relcal.schema import Arrays


@dataclass(frozen=True)
class UQPredictions:
    """Per-item predicted preference probability and epistemic uncertainty."""

    item_ids: list[str]
    probability: np.ndarray  # shape (n_items,)
    epistemic_uncertainty: np.ndarray  # shape (n_items,)

    def __post_init__(self) -> None:
        if not (len(self.item_ids) == self.probability.shape[0] == self.epistemic_uncertainty.shape[0]):
            raise ValueError("item_ids, probability, and epistemic_uncertainty must align")

    def _index(self) -> dict[str, int]:
        return {item: i for i, item in enumerate(self.item_ids)}

    def probability_for(self, arrays: Arrays) -> np.ndarray:
        """Broadcast the per-item probability to the rows of ``arrays``."""
        index = self._index()
        return np.array([self.probability[index[i]] for i in arrays.item_id], dtype=float)

    def uncertainty_for(self, arrays: Arrays) -> np.ndarray:
        """Broadcast the per-item epistemic uncertainty to the rows of ``arrays``."""
        index = self._index()
        return np.array(
            [self.epistemic_uncertainty[index[i]] for i in arrays.item_id], dtype=float
        )


class UQBackend(ABC):
    """Abstract backend returning per-item probability and epistemic uncertainty."""

    @abstractmethod
    def predict_items(self, item_ids: Sequence[str]) -> UQPredictions:
        """Return predictions for the given item identifiers, in the same order."""
        raise NotImplementedError


class MockEnsembleUQ(UQBackend):
    """A small ensemble of context-blind predictors over synthetic ground truth.

    Each member predicts the marginal preference bar_pi(x) plus independent Gaussian noise.
    The reported probability is the ensemble mean and the epistemic uncertainty is the
    ensemble standard deviation. Because no member sees the context, the uncertainty cannot
    encode relational dispersion; it tracks only estimation noise. This is the context-blind
    model of Proposition 1.
    """

    def __init__(
        self,
        bar_pi_by_item: dict[str, float],
        *,
        n_ensemble: int = 8,
        noise: float = 0.02,
        seed: int = 0,
    ) -> None:
        if n_ensemble < 2:
            raise ValueError("an ensemble needs at least two members")
        if noise < 0.0:
            raise ValueError("noise must be non-negative")
        self._bar_pi = dict(bar_pi_by_item)
        self._n_ensemble = n_ensemble
        self._noise = noise
        self._seed = seed

    def predict_items(self, item_ids: Sequence[str]) -> UQPredictions:
        rng = np.random.default_rng(self._seed)
        items = list(item_ids)
        probs = np.empty(len(items), dtype=float)
        uncert = np.empty(len(items), dtype=float)
        for i, item in enumerate(items):
            if item not in self._bar_pi:
                raise KeyError(f"item {item!r} is unknown to the mock backend")
            center = self._bar_pi[item]
            members = np.clip(
                center + rng.normal(0.0, self._noise, size=self._n_ensemble), 0.0, 1.0
            )
            probs[i] = float(members.mean())
            uncert[i] = float(members.std(ddof=1))
        return UQPredictions(item_ids=items, probability=probs, epistemic_uncertainty=uncert)


class RealEnsembleUQ(UQBackend):
    """Stub for a real open instruct or DPO model with ensemble or MC-dropout uncertainty.

    Wiring this to a model is future work. It must load a model, score each item's
    preference probability, and obtain an epistemic uncertainty from an ensemble of
    fine-tunes or from MC-dropout passes. Until then it raises, so that no experiment can
    silently report fabricated numbers.
    """

    def __init__(self, model_name: str, *, method: str = "ensemble") -> None:
        self.model_name = model_name
        self.method = method

    def predict_items(self, item_ids: Sequence[str]) -> UQPredictions:
        raise NotImplementedError(
            "RealEnsembleUQ is not yet wired to a model. TODO: load "
            f"{self.model_name!r}, score preference probabilities, and compute epistemic "
            f"uncertainty via {self.method!r}. Use MockEnsembleUQ for tests and the "
            "synthetic sample."
        )
