"""Pilot guards: e1 refuses empty or one-sided data and never fabricates numbers.

These tests exercise the refusal paths of experiments/e1_pilot.py and one small
end-to-end run on a reduced synthetic dataset, so the pilot pipeline is covered without
human data.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from e1_pilot import overconfidence_check, run_pilot  # noqa: E402

from relcal import simulate  # noqa: E402
from relcal.dataio import write_judgments_jsonl  # noqa: E402
from relcal.schema import PreferenceDataset  # noqa: E402
from relcal.uq import MockEnsembleUQ  # noqa: E402


def test_refuses_empty_file(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        run_pilot(path)


def test_refuses_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_pilot(tmp_path / "nope.jsonl")


def test_refuses_when_control_set_absent(tmp_path: Path):
    data, _ = simulate.generate(n_items=6, n_raters=4, dispersion=0.1, relational=True, seed=0)
    path = tmp_path / "rel_only.jsonl"
    write_judgments_jsonl(data, path)
    with pytest.raises(ValueError, match="control"):
        run_pilot(path)


def test_refuses_when_relational_set_absent(tmp_path: Path):
    data, _ = simulate.generate_no_structure(n_items=6, n_raters=4, seed=0)
    path = tmp_path / "ctl_only.jsonl"
    write_judgments_jsonl(data, path)
    with pytest.raises(ValueError, match="relational set"):
        run_pilot(path)


def test_real_backend_raises_not_implemented(tmp_path: Path):
    rel, _ = simulate.generate(n_items=6, n_raters=4, dispersion=0.1, relational=True, seed=0)
    ctl, _ = simulate.generate_no_structure(n_items=6, n_raters=4, seed=1)
    path = tmp_path / "both.jsonl"
    write_judgments_jsonl(PreferenceDataset(rel.records + ctl.records), path)
    with pytest.raises(NotImplementedError):
        run_pilot(path, backend_name="real", model="some-model")


def test_pipeline_runs_end_to_end_small(tmp_path: Path):
    rel, _ = simulate.generate(
        n_items=25, n_raters=8, dispersion=0.15, relational=True, seed=2, item_prefix="rel"
    )
    ctl, _ = simulate.generate(
        n_items=25, n_raters=8, dispersion=0.0, relational=False, seed=3, item_prefix="ctl"
    )
    path = tmp_path / "both.jsonl"
    write_judgments_jsonl(PreferenceDataset(rel.records + ctl.records), path)

    result = run_pilot(path, n_permutations=100, n_boot=100, corr_n_boot=100)

    comparison = result.comparison
    assert comparison.language == "synthetic"
    # The report is never bare: p-value and interval accompany the estimate.
    assert 0.0 < comparison.relational.p_value <= 1.0
    assert comparison.relational.ci_low <= comparison.relational.ci_high
    oc = result.overconfidence
    assert oc.n_items == 25
    assert -1.0 <= oc.pearson <= 1.0
    assert oc.pearson_ci[0] <= oc.pearson_ci[1]


def test_overconfidence_check_reports_measured_correlation():
    data, truth = simulate.generate(n_items=30, n_raters=8, dispersion=0.1, seed=4)
    bar_pi = {item: float(p) for item, p in zip(truth.item_ids, truth.bar_pi)}
    backend = MockEnsembleUQ(bar_pi, seed=5)
    predictions = backend.predict_items(sorted(bar_pi.keys()))
    oc = overconfidence_check(data, predictions, n_boot=200, rng_seed=6)
    assert oc.n_items == 30
    assert -1.0 <= oc.spearman <= 1.0
    assert oc.spearman_ci[0] <= oc.spearman_ci[1]
    assert oc.pearson_ci[0] <= oc.pearson_ci[1]
