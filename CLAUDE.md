# CLAUDE.md

Working rules for this repository. These are constraints, not suggestions.

## Source of truth

`docs/theory/Relational_UQ_Formalization.md` is the canonical specification. Every
estimand the code computes is defined there, and the code follows the document, not the
other way around. If an implementation and the formalization disagree, the fix is decided
at the document first. `docs/REVIEW.md` tracks components against the target layout.

## Guardrails

- No bare RCE. The only public reporting entry point is `relcal.report.report_rce`, and
  it returns the estimate only together with the permutation-null p-value and the
  bootstrap confidence interval. It raises if asked to skip the null. Do not add any code
  path that reports an RCE value without the null; a test enforces this.
- The within-language control. Any relational-versus-control comparison holds the
  language fixed. Mixed-language comparisons raise. Use
  `relcal.report.compare_relational_vs_control` for comparisons; a test enforces the
  failure on mixed languages.
- No fabricated numbers. Unrun paths raise (`RealEnsembleUQ` raises until wired to a real
  model). Experiments refuse empty data. Never print placeholder results or hardcode an
  expected outcome as if it were measured.
- Honest positioning. The calibration machinery is group calibration and
  multicalibration (Hebert-Johnson et al. 2018). This project contributes the choice of
  groups, the bridge to relational dispersion (Proposition 3), and the UQ consequence
  (Proposition 1). Do not describe it as a new calibration algorithm.
- Data hygiene. Real annotations are never committed. Only the synthetic sample under
  `data/sample/` is tracked; `data/real/` and `*.real.jsonl` are gitignored.

## Style

- Prose in documents and docstrings uses no contractions and no em dashes; write
  full-form phrasing.
- Commit messages follow the conventional `type(scope): summary` form used in the log.
- No AI attribution on commits or pull requests. The `.githooks/commit-msg` hook strips
  `Co-Authored-By` and generated-with trailers; do not add them.

## Checks

Run the test suite from the repository root before committing:

```
python3 -m pytest -q
```

Experiments must run end to end from the repository root:

```
python3 experiments/e0_estimator_validation.py --quick
python3 experiments/e1_pilot.py --quick
```

Dependencies are Python 3.12 and NumPy only. Do not add SciPy or other runtime
dependencies without discussion.
