"""Loader tests: JSONL roundtrip, per-line validation, and the empty-data refusal.

The pilot depends on the loader failing loudly: a malformed or empty file must raise with
a clear message rather than produce a silent partial dataset.
"""

import json
from pathlib import Path

import pytest

from relcal.dataio import load_judgments_jsonl, write_judgments_jsonl
from relcal.schema import Judgment, PreferenceDataset


def _tiny_dataset() -> PreferenceDataset:
    records = [
        Judgment("item_a", "yo", "elder_addressee", True, 1, "rater_x"),
        Judgment("item_a", "yo", "younger_addressee", True, 0, "rater_x"),
        Judgment("item_b", "yo", "elder_addressee", False, 1, "rater_y"),
    ]
    return PreferenceDataset(records)


def test_roundtrip_preserves_records(tmp_path: Path):
    path = tmp_path / "judgments.jsonl"
    original = _tiny_dataset()
    write_judgments_jsonl(original, path)
    loaded = load_judgments_jsonl(path)
    assert loaded.records == original.records


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_judgments_jsonl(tmp_path / "does_not_exist.jsonl")


def test_empty_file_refused(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_judgments_jsonl(path)


def test_invalid_json_names_the_line(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    good = json.dumps(
        {"item_id": "a", "language": "yo", "context": "c", "relational": True, "judgment": 1}
    )
    path.write_text(good + "\nnot json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        load_judgments_jsonl(path)


def test_missing_field_names_the_line(tmp_path: Path):
    path = tmp_path / "missing.jsonl"
    path.write_text(
        json.dumps({"item_id": "a", "language": "yo", "context": "c", "relational": True})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="judgment"):
        load_judgments_jsonl(path)


def test_invalid_judgment_value_rejected(tmp_path: Path):
    path = tmp_path / "badval.jsonl"
    path.write_text(
        json.dumps(
            {"item_id": "a", "language": "yo", "context": "c", "relational": True, "judgment": 2}
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="judgment must be 0 or 1"):
        load_judgments_jsonl(path)


def test_shipped_sample_loads_with_both_sets():
    sample = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample.jsonl"
    data = load_judgments_jsonl(sample)
    assert len(data.filter(relational=True)) > 0
    assert len(data.filter(relational=False)) > 0
    # The sample must satisfy the within-language control by construction.
    assert data.require_single_language() == "synthetic"
