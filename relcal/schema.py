"""Record types for relational preference data.

A single observation is one rater's binary preference judgment for one item under one
relational context. Items carry a language and a relational flag distinguishing the
relational set from the matched nonrelational control set. See Section 2 and Section 8 of
docs/theory/Relational_UQ_Formalization.md.

No rater personally identifying information is represented here. The rater_id field is an
opaque, non-reversible label used only to group judgments by rater; it must never carry a
name, an email address, or any other identifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class Judgment:
    """One rater's binary preference judgment for one item under one context.

    Attributes:
        item_id: Identifier of the preference item (the comparison). The item is the
            independent unit for the bootstrap.
        language: Language of the item. Relational-versus-control comparisons must hold
            this fixed (the within-language control, Section 8).
        context: Relational context label (for example "elder", "younger", "host",
            "guest", "insider", "outsider"). For control items the context is still
            recorded but is not expected to move the preference.
        relational: True if the item belongs to the relational set, False if it belongs to
            the matched nonrelational control set.
        judgment: Binary preference, 1 if the designated response is preferred, else 0.
        rater_id: Opaque, non-reversible label grouping judgments by rater. Never PII.
    """

    item_id: str
    language: str
    context: str
    relational: bool
    judgment: int
    rater_id: str = "anon"

    def __post_init__(self) -> None:
        if self.judgment not in (0, 1):
            raise ValueError(
                f"judgment must be 0 or 1, received {self.judgment!r} for item "
                f"{self.item_id!r}"
            )
        if not self.item_id:
            raise ValueError("item_id must be a non-empty string")
        if not self.language:
            raise ValueError("language must be a non-empty string")
        if not self.context:
            raise ValueError("context must be a non-empty string")


@dataclass(frozen=True)
class Arrays:
    """Column arrays extracted from a dataset, aligned row by row.

    Every array has the same length, one entry per observation. This is the plain-numpy
    interface the calibration and permutation estimators operate on, so those modules do
    not depend on the record types.
    """

    item_id: np.ndarray  # dtype object (strings)
    language: np.ndarray  # dtype object (strings)
    context: np.ndarray  # dtype object (strings)
    relational: np.ndarray  # dtype bool
    judgment: np.ndarray  # dtype int (0 or 1)
    rater_id: np.ndarray  # dtype object (strings)

    def __len__(self) -> int:
        return int(self.item_id.shape[0])


@dataclass
class PreferenceDataset:
    """A collection of judgments with extraction and validation helpers."""

    records: list[Judgment] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Judgment]:
        return iter(self.records)

    @classmethod
    def from_records(cls, records: Iterable[Judgment]) -> "PreferenceDataset":
        return cls(list(records))

    def languages(self) -> list[str]:
        """Distinct languages present, in first-seen order."""
        seen: dict[str, None] = {}
        for r in self.records:
            seen.setdefault(r.language, None)
        return list(seen.keys())

    def require_single_language(self) -> str:
        """Return the single language, or raise if more than one is present.

        The within-language control (Section 8 of the formalization) requires that any
        relational-versus-control comparison hold the language fixed. Callers that compare
        the relational and control sets must route through this check so that a mixed-
        language comparison fails loudly rather than silently confounding language with
        relational structure.
        """
        langs = self.languages()
        if len(langs) == 0:
            raise ValueError("dataset is empty; no language to control on")
        if len(langs) > 1:
            raise ValueError(
                "within-language control violated: comparison spans multiple languages "
                f"{langs!r}. Relational-versus-control comparisons must use one language."
            )
        return langs[0]

    def contexts(self) -> list[str]:
        """Distinct context labels present, in first-seen order."""
        seen: dict[str, None] = {}
        for r in self.records:
            seen.setdefault(r.context, None)
        return list(seen.keys())

    def filter(
        self,
        *,
        relational: Optional[bool] = None,
        language: Optional[str] = None,
    ) -> "PreferenceDataset":
        """Return a sub-dataset matching the given predicates."""
        records = self.records
        if relational is not None:
            records = [r for r in records if r.relational is relational]
        if language is not None:
            records = [r for r in records if r.language == language]
        return PreferenceDataset(list(records))

    def arrays(self) -> Arrays:
        """Extract aligned column arrays for the estimators."""
        if not self.records:
            raise ValueError("cannot extract arrays from an empty dataset")
        return Arrays(
            item_id=np.array([r.item_id for r in self.records], dtype=object),
            language=np.array([r.language for r in self.records], dtype=object),
            context=np.array([r.context for r in self.records], dtype=object),
            relational=np.array([r.relational for r in self.records], dtype=bool),
            judgment=np.array([r.judgment for r in self.records], dtype=int),
            rater_id=np.array([r.rater_id for r in self.records], dtype=object),
        )
