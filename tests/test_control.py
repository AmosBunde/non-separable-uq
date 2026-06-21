"""Tests for the within-language control (Section 8).

Any relational-versus-nonrelational comparison must hold the language fixed. A comparison
that mixes languages must fail, so that language is never confounded with relational
structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from relcal import simulate
from relcal.report import compare_relational_vs_control, report_rce
from relcal.schema import Judgment, PreferenceDataset


def _single_language_relational_and_control():
    """Build a matched relational set and control set in ONE language."""
    rel, rel_truth = simulate.generate(
        n_items=120, n_raters=20, dispersion=0.15, relational=True, language="lang_x",
        seed=1, item_prefix="rel",
    )
    ctl, ctl_truth = simulate.generate(
        n_items=120, n_raters=20, dispersion=0.0, relational=False, language="lang_x",
        seed=2, item_prefix="ctl",
    )
    return (rel, rel_truth), (ctl, ctl_truth)


def test_within_language_comparison_succeeds():
    """The relational and control reports run when both use the same language."""
    (rel, rel_truth), (ctl, ctl_truth) = _single_language_relational_and_control()
    comparison = compare_relational_vs_control(
        rel, rel_truth.marginal_predictions(rel.arrays()),
        ctl, ctl_truth.marginal_predictions(ctl.arrays()),
        n_permutations=80, n_boot=80,
    )
    assert comparison.language == "lang_x"
    assert comparison.relational.language == comparison.control.language == "lang_x"


def test_mixed_language_dataset_is_rejected_by_report():
    """A dataset spanning two languages must raise: the control would be violated."""
    records = [
        Judgment("a", "lang_x", "context_a", True, 1),
        Judgment("a", "lang_x", "context_b", True, 0),
        Judgment("b", "lang_y", "context_a", True, 1),
        Judgment("b", "lang_y", "context_b", True, 0),
    ]
    mixed = PreferenceDataset(records)
    preds = np.full(len(mixed), 0.5)
    with pytest.raises(ValueError, match="within-language control"):
        report_rce(mixed, preds, n_permutations=10, n_boot=10)


def test_require_single_language_guards_directly():
    mixed = PreferenceDataset(
        [
            Judgment("a", "lang_x", "context_a", True, 1),
            Judgment("b", "lang_y", "context_a", True, 0),
        ]
    )
    with pytest.raises(ValueError, match="multiple languages"):
        mixed.require_single_language()


def test_comparison_fails_when_sets_use_different_languages_separately():
    """The comparison path must reject sets that differ in language.

    This is the path e1 actually uses: the relational set and the control set are reported
    by separate calls. Each set is internally single-language, so per-call validation alone
    would let a relational set in lang_x be compared against a control set in lang_y. The
    comparison entry point checks the language match across the two sets and must raise.
    """
    rel, rel_truth = simulate.generate(
        n_items=20, n_raters=8, dispersion=0.15, relational=True, language="lang_x", seed=1
    )
    ctl, ctl_truth = simulate.generate(
        n_items=20, n_raters=8, dispersion=0.0, relational=False, language="lang_y", seed=2
    )
    with pytest.raises(ValueError, match="within-language control"):
        compare_relational_vs_control(
            rel, rel_truth.marginal_predictions(rel.arrays()),
            ctl, ctl_truth.marginal_predictions(ctl.arrays()),
            n_permutations=10, n_boot=10,
        )


def test_combined_mixed_language_dataset_also_rejected():
    """Pooling two languages into one dataset is also rejected at the report boundary."""
    rel, _ = simulate.generate(
        n_items=20, n_raters=8, dispersion=0.15, relational=True, language="lang_x", seed=1
    )
    ctl, _ = simulate.generate(
        n_items=20, n_raters=8, dispersion=0.0, relational=False, language="lang_y", seed=2
    )
    combined = PreferenceDataset(rel.records + ctl.records)
    preds = np.full(len(combined), 0.5)
    with pytest.raises(ValueError, match="within-language control"):
        report_rce(combined, preds, n_permutations=10, n_boot=10)
