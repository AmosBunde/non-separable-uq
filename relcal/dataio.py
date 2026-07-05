"""Loading and writing judgment data as JSON Lines.

The on-disk format is defined in data/schema.md: one JSON object per line, one object per
judgment, with the exact fields of relcal.schema.Judgment. The loader validates every line
through the Judgment constructor, so a malformed record fails loudly with its line number
rather than entering the dataset silently.

Real annotation files must never be committed to version control (see
data/ANNOTATION_PROTOCOL.md and .gitignore). Only the tiny synthetic sample under
data/sample/ is tracked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from relcal.schema import Judgment, PreferenceDataset

REQUIRED_FIELDS = ("item_id", "language", "context", "relational", "judgment")


def load_judgments_jsonl(path: Union[str, Path]) -> PreferenceDataset:
    """Load a PreferenceDataset from a JSON Lines file.

    Each non-empty line must be a JSON object with the fields item_id, language, context,
    relational, and judgment; rater_id is optional and defaults to "anon". Every record is
    validated by the Judgment constructor.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is empty, a line is not valid JSON, a required field is
            missing, or a field fails Judgment validation. The error names the line.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"judgment file does not exist: {path}")

    records: list[Judgment] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object, found {type(obj).__name__}"
                )
            missing = [f for f in REQUIRED_FIELDS if f not in obj]
            if missing:
                raise ValueError(f"{path}:{line_number}: missing required fields {missing}")
            try:
                records.append(
                    Judgment(
                        item_id=obj["item_id"],
                        language=obj["language"],
                        context=obj["context"],
                        relational=bool(obj["relational"]),
                        judgment=obj["judgment"],
                        rater_id=obj.get("rater_id", "anon"),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc

    if not records:
        raise ValueError(
            f"judgment file is empty: {path}. The pilot refuses to run on empty data; "
            "it never emits placeholder results."
        )
    return PreferenceDataset(records)


def write_judgments_jsonl(dataset: PreferenceDataset, path: Union[str, Path]) -> None:
    """Write a PreferenceDataset to a JSON Lines file, one judgment per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for r in dataset.records:
            handle.write(
                json.dumps(
                    {
                        "item_id": r.item_id,
                        "language": r.language,
                        "context": r.context,
                        "relational": r.relational,
                        "judgment": r.judgment,
                        "rater_id": r.rater_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
