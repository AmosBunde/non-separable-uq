"""Regenerate the tiny synthetic sample under data/sample/.

The sample is synthetic, deterministic, and small. It exists so that the pilot experiment
(experiments/e1_pilot.py) has data to run against end to end without any human annotation.
Real annotations are never committed; see data/ANNOTATION_PROTOCOL.md and .gitignore.

The sample holds one language and two sets:

- a relational set with a known nonzero relational dispersion, and
- a matched nonrelational control set with zero dispersion,

both written to a single file so that the pilot exercises the relational-versus-control
split and the within-language control.

Run from the repository root:
    python3 data/make_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from relcal import simulate
from relcal.dataio import write_judgments_jsonl
from relcal.schema import PreferenceDataset

SAMPLE_PATH = _REPO_ROOT / "data" / "sample" / "sample.jsonl"

LANGUAGE = "synthetic"
CONTEXTS = ("elder_addressee", "younger_addressee")
N_ITEMS = 40
N_RATERS = 12
DISPERSION = 0.15
SEED = 20260622


def main() -> None:
    relational_data, _ = simulate.generate(
        n_items=N_ITEMS,
        contexts=CONTEXTS,
        n_raters=N_RATERS,
        dispersion=DISPERSION,
        relational=True,
        language=LANGUAGE,
        seed=SEED,
        item_prefix="rel",
    )
    control_data, _ = simulate.generate(
        n_items=N_ITEMS,
        contexts=CONTEXTS,
        n_raters=N_RATERS,
        dispersion=0.0,
        relational=False,
        language=LANGUAGE,
        seed=SEED + 1,
        item_prefix="ctl",
    )
    combined = PreferenceDataset(relational_data.records + control_data.records)
    write_judgments_jsonl(combined, SAMPLE_PATH)
    print(
        f"wrote {len(combined)} judgments "
        f"({len(relational_data)} relational, {len(control_data)} control) to {SAMPLE_PATH}"
    )


if __name__ == "__main__":
    main()
