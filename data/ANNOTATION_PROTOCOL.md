# Annotation protocol

This protocol governs the collection of real preference judgments for the pilot
(experiments/e1_pilot.py). It covers context labelling, rater instructions, and the
construction of the matched control set. Consent, compensation, and privacy rules are in
`ETHICS.md`; the two documents apply together and neither may be waived.

No real annotations exist yet. This protocol is written before collection so that the
design is fixed in advance rather than adjusted after seeing results.

## 1. Unit of annotation

One item is one preference comparison: a single prompt with two candidate responses in a
single language, one of which is designated. One judgment is one rater's binary answer to
the question "is the designated response the better one for this situation?", collected
under exactly one relational context.

## 2. Context labelling

- Each item in the relational set is presented under each context in a fixed, small
  context inventory chosen per language before collection begins. Example inventory:
  `elder_addressee`, `younger_addressee`, `host_to_guest`, `guest_to_host`,
  `insider`, `outsider`.
- The context is shown to the rater as a short scenario sentence, for example: "The
  speaker is addressing a respected elder." The scenario is the only element that varies
  between contexts; prompt and responses are identical.
- Context labels are assigned by the study design, not inferred by the raters, so the
  context variable is an experimental condition rather than an annotation.
- Every (item, context) cell should receive the same target number of raters, so that the
  context weights `rho(c)` are balanced by construction. Imbalances that survive
  collection are handled by the observation-count weighting in the estimator.

## 3. Rater instructions

Raters are instructed to:

1. read the prompt, the scenario sentence, and both responses in full before answering;
2. judge which response is better for the stated situation, not which is better in
   general;
3. answer the binary question only; there is no neutral option, because the estimand is a
   preference probability and abstentions would need a separate model;
4. flag items that are incoherent, offensive, or impossible to judge; flagged items are
   excluded from all sets, relational and control alike, before any estimation.

Raters never see model predictions, confidence scores, or the relational-versus-control
status of an item.

## 4. The matched nonrelational control set

For every language collected, a control set is constructed in the same language, from the
same prompt sources, with the same item construction procedure, presented to the same
rater pool under the same context scenarios. The single difference is that control items
are selected so that the relational context should not move the preference (for example,
factual or purely technical comparisons). The control set is what separates "the
estimator finds structure wherever it looks" from "the estimator finds structure where
structure is".

Matching checklist, applied before collection:

- [ ] same language;
- [ ] same prompt source and item construction procedure;
- [ ] same context inventory and scenario presentation;
- [ ] same target rater count per (item, context) cell;
- [ ] control items chosen for expected context-independence, documented per item.

## 5. Data recording

Judgments are recorded in the JSON Lines format of `data/schema.md`. The `rater_id` is an
opaque random label assigned at collection time; the mapping from label to person, if one
must exist at all for payment, is kept outside this repository and destroyed when payment
is settled. Real annotation files live under `data/real/`, which is excluded from version
control by `.gitignore`. Only the synthetic sample under `data/sample/` is ever tracked.
