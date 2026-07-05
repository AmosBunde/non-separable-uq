# Ethics

This document fixes the rules for collecting human preference judgments for the pilot.
It applies together with `data/ANNOTATION_PROTOCOL.md`. The rules below are commitments,
not aspirations: collection that cannot satisfy them does not proceed.

## Rater consent

- Raters are told, before any work, what the judgments are for: measuring whether
  preference models are miscalibrated across relational social contexts.
- Participation is voluntary and raters may stop at any time without penalty; completed
  work is paid regardless.
- Consent is recorded per rater before their first judgment. No judgments are collected
  from anyone who has not consented.
- Raters are informed that their judgments will be used in aggregate statistical
  analysis and that no attempt will be made to identify them from their answers.

## Fair compensation

- Raters are paid at or above the living wage of their place of residence, computed
  against a realistic per-judgment time estimate measured in a paid calibration round,
  not against the speed of the fastest rater.
- Flagged, skipped, and quality-control items are paid the same as ordinary items.
- Payment does not depend on agreement with other raters or with any model.

## Privacy and the no-PII rule

- The data schema (`data/schema.md`) has no field for any personal identifier. The
  `rater_id` is an opaque random label assigned at collection time.
- If a mapping from label to person must exist for payment, it is kept outside this
  repository, accessible only to the person administering payment, and destroyed once
  payment is settled.
- Real annotation files are never committed to version control. `.gitignore` excludes
  `data/real/` and `*.real.jsonl`; only the tiny synthetic sample under `data/sample/`
  is tracked.

## Content and rater welfare

- Items are screened before presentation; raters are not asked to judge content that is
  degrading or harassing.
- Raters can flag any item as incoherent, offensive, or impossible to judge, and flagged
  items are excluded from analysis and paid in full.

## Context labelling

- Relational contexts describe situations (for example, addressing an elder), never the
  raters themselves. No demographic attributes of raters are collected or inferred.
- Context inventories are chosen with speakers of the language being studied, so that
  the scenarios are meaningful rather than imposed.

## Reporting

- Results are reported with the permutation null and confidence intervals, per the
  reporting contract of the formalization, so that noise is never presented as social
  structure. Claims about a language community are claims about the collected sample and
  are stated as such.
