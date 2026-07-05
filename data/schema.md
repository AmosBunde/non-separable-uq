# Pilot dataset schema

This document defines the on-disk format for judgment data. The loader is
`relcal.dataio.load_judgments_jsonl`; it validates every record and refuses empty files.
The in-memory record type is `relcal.schema.Judgment`, and the semantics of every field
are fixed by Section 2 and Section 8 of `docs/theory/Relational_UQ_Formalization.md`.

## Format

JSON Lines: one JSON object per line, one object per judgment. A judgment is one rater's
binary preference for one item under one relational context.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `item_id` | string, non-empty | yes | Identifier of the preference comparison. The item is the independent unit for the bootstrap. |
| `language` | string, non-empty | yes | Language of the item. Relational-versus-control comparisons must hold this fixed. |
| `context` | string, non-empty | yes | Relational context label, for example `elder_addressee`. Recorded for control items too. |
| `relational` | boolean | yes | `true` for the relational set, `false` for the matched nonrelational control set. |
| `judgment` | integer, 0 or 1 | yes | 1 if the designated response is preferred, else 0. |
| `rater_id` | string | no (default `"anon"`) | Opaque, non-reversible rater label. Never a name, an email address, or any other personally identifying information. |

Example line:

```json
{"item_id": "rel_0000", "language": "synthetic", "context": "elder_addressee", "relational": true, "judgment": 1, "rater_id": "rater_000"}
```

## Validation rules

The loader raises, naming the file and line number, when:

- a line is not valid JSON or is not a JSON object;
- a required field is missing;
- `judgment` is not 0 or 1;
- `item_id`, `language`, or `context` is empty;
- the file contains no records at all. The pilot refuses to run on empty data and never
  emits placeholder results.

## Files

- `data/sample/sample.jsonl` is a tiny synthetic sample, regenerated deterministically by
  `python3 data/make_sample.py`. It contains a relational set with known dispersion and a
  matched control set with zero dispersion, in a single language.
- Real annotation files must never be committed. `.gitignore` excludes `data/real/` and
  `*.real.jsonl`. See `data/ANNOTATION_PROTOCOL.md` and `ETHICS.md`.
