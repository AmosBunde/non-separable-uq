# Repository inventory and gap review

This repository was empty at the start of this work (no commits, no source). There was
therefore nothing to inventory. This document records the target layout and tracks each
component against its status, so the gap between the plan and the code is visible at every
step. It is updated as components land.

The canonical specification is `docs/theory/Relational_UQ_Formalization.md`. Every component
below implements a part of it.

## Target layout

```
relcal/
  schema.py        record types: item id, language, context label, relational flag
                   (relational vs nonrelational control), per-rater binary judgments
  calibration.py   pooled ECE, within-context ECE_c, RCE; adaptive and debiased binning
  permutation.py   context-label shuffle null, permutation p-value, bootstrap CI
  simulate.py      synthetic generator with a known relational dispersion parameter, plus
                   a no-structure generator for the null
  uq.py            per-item predicted preference probability and epistemic uncertainty;
                   mock backend for tests, real backend stubbed
  report.py        the only public reporting entry point; returns RCE with null p-value and
                   CI; raises if the null is disabled
experiments/
  e0_estimator_validation.py   recovery, null calibration, marginal-model bound; synthetic
  e1_pilot.py                  pilot on annotated data; overconfidence check; refuses empty
data/
  schema.md                    pilot dataset schema
  ANNOTATION_PROTOCOL.md       context labelling, rater instructions, control matching
  sample/                      tiny synthetic sample only; never real annotations
tests/
  test_calibration.py          RCE non-negativity; analytic-value recovery
  test_permutation.py          false-positive rate near nominal under no structure
  test_report.py               report raises without the permutation null
  test_control.py              within-language control; fails if languages are mixed
docs/
  theory/Relational_UQ_Formalization.md   canonical (done)
  REVIEW.md                               this file
README.md                                 question, positioning, how to run e0 and e1
ETHICS.md                                 rater consent, compensation, no PII
CLAUDE.md                                 source of truth, guardrails, no-bare-RCE, controls
```

## Gap table

Status legend: done, in progress, todo.

| Component | Specification reference | Prompt | Status |
| --- | --- | --- | --- |
| Setup: gitignore, hooks, attribution | A2, A3 | 1 | done |
| `docs/theory/Relational_UQ_Formalization.md` | source of truth | 1 | done |
| `docs/REVIEW.md` | Task 1 | 1 | done |
| `relcal/schema.py` | Section 2, Section 8 | 1 | done |
| `relcal/calibration.py` | Section 3, Section 4 | 1 | done |
| `relcal/permutation.py` | Section 6 | 1 | done |
| `relcal/simulate.py` | Section 4.3, Section 6 | 1 | done |
| `relcal/uq.py` | Section 5 | 1 | done (mock backend; real backend stubbed, raises) |
| `relcal/report.py` | Section 7 | 1 | done |
| `experiments/e0_estimator_validation.py` | Section 4.2, 4.4, 6.1 | 1 | done |
| `tests/` (Prompt 1 subset) | Section 4.2, 6.1, 7, 8 | 1 | done |
| `data/schema.md`, loader (`relcal/dataio.py`) | Task 5 | 2 | done |
| `data/ANNOTATION_PROTOCOL.md` | Task 5 | 2 | done |
| `data/sample/sample.jsonl`, `data/make_sample.py` | Task 5 | 2 | done (synthetic only) |
| `experiments/e1_pilot.py` | Section 5, Task 6 | 2 | done (runs on the sample with the mock backend; real-model run pending `RealEnsembleUQ`) |
| `tests/` (Prompt 2 subset: loader, pilot guards) | Section 5, 7, Task 5, 6 | 2 | done |
| `README.md` | Task 7 | 2 | done |
| `ETHICS.md` | Task 7 | 2 | done |
| `CLAUDE.md` | Task 7 | 2 | done |

Remaining work is outside the repository code: wiring `RealEnsembleUQ` to an open
instruct or DPO model, and collecting real annotations under
`data/ANNOTATION_PROTOCOL.md` and `ETHICS.md`. Both are tracked here and both paths
currently raise rather than fabricate output.

## Honest positioning

The calibration estimators are group calibration and multicalibration (Hebert-Johnson et
al. 2018). This project does not introduce a new calibration algorithm. The contribution is
the choice of groups (relational contexts), the bridge from group calibration error to
relational dispersion (Proposition 3), and the uncertainty-quantification consequence
(Proposition 1). The README states this plainly.
