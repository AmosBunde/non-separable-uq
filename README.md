# Relational Calibration Error

Measuring the calibration error that pooling hides.

## The question

Preference and reward models are trained to predict a probability that one response is
preferred over another. In many settings the better response depends on a relational
context: who is speaking to whom, the relative social position of the parties, the register
expected of a host toward a guest, of a younger person toward an elder, of an insider toward
an outsider. A model that collapses these contexts into a single context-marginal preference
can look well calibrated when its predictions are pooled, while remaining systematically
miscalibrated inside individual contexts.

This project asks a precise version of that worry and answers it with a statistic:

> When a preference model is audited within relational contexts rather than in aggregate,
> how much additional miscalibration appears, is that amount real or a finite-sample
> artifact, and does the model's own reported uncertainty know about it?

The statistic is the Relational Calibration Error (RCE). The remainder of this document
defines it, states the honest positioning of the work, and explains how to run the code that
exists today and the pilot that is planned next.

## Honest positioning

The calibration machinery here is group calibration and multicalibration in the sense of
Hebert-Johnson, Kim, Reingold, and Rothblum (2018), "Multicalibration: Calibration for the
(Computationally-Identifiable) Masses." This project does not introduce a new calibration
algorithm, and it does not claim to. Computing calibration error within subgroups and
comparing it to the pooled calibration error is exactly the multicalibration and group
calibration idea.

The contribution of this project is three specific things layered on top of that established
machinery:

1. The choice of groups. The groups are relational contexts, and the relational set is
   measured against a matched nonrelational control set in the same language.
2. The bridge to relational dispersion. For a model that predicts the context-marginal, the
   Relational Calibration Error equals the mean absolute relational dispersion of the true
   preference (Proposition 3). This connects a calibration audit to a property of the world.
3. The uncertainty-quantification consequence. A context-blind model with context-blind
   uncertainty cannot have its reported uncertainty track relational miscalibration
   (Proposition 1). The pilot tests this prediction directly.

If you are looking for a new way to calibrate a classifier, this is not that. If you are
asking whether a preference model is quietly miscalibrated across social contexts, and
whether its uncertainty estimates hide the fact, this is a measurement tool for that
question.

## The Relational Calibration Error in words

Let the preference probability of an item depend on a relational context. Write `pi(x, c)`
for the true probability that the designated response is preferred for item `x` under context
`c`, and `bar_pi(x)` for the context-marginal preference, the average of `pi(x, c)` over
contexts.

- The pooled expected calibration error, `ECE`, is the usual calibration error computed over
  all observations at once: bin the predictions, and within each bin compare the average
  prediction to the empirical preference frequency.
- The within-context calibration error, `ECE_c`, is the same audit restricted to a single
  context `c`.
- The Relational Calibration Error is the context-weighted mean of the within-context
  calibration errors, minus the pooled calibration error:

  ```
  RCE = ( sum over contexts c of rho(c) * ECE_c ) - ECE
  ```

  where `rho(c)` is the share of the data in context `c`.

In words: refine the calibration audit by conditioning on the relational context, average the
within-context calibration errors over contexts, and subtract the calibration error you would
have reported by pooling. The Relational Calibration Error is the extra miscalibration that
conditioning on context reveals. It is zero when context adds nothing, and positive when the
model is more miscalibrated inside contexts than it looks in aggregate.

Three propositions, proved in `docs/theory/Relational_UQ_Formalization.md`, fix the meaning:

- Proposition 2: `RCE >= 0` for any model. Refining a calibration audit by context can only
  reveal more miscalibration, never less.
- Proposition 3: for a model that predicts the context-marginal `bar_pi(x)`, the Relational
  Calibration Error equals the mean absolute relational dispersion of the true preference,
  `MARD = E | pi(x, c) - bar_pi(x) |`. A measured note on the finite-sample direction of this
  identity appears below and in the formalization.
- Proposition 1: a context-blind model whose uncertainty is also context-blind cannot have
  its reported uncertainty track relational miscalibration, because the uncertainty never
  sees the context. The pilot estimates the relevant correlation and reports it with a
  confidence interval.

## Two guarantees built into the code

Two rules are structural, not advisory, and are enforced by tests.

- No bare Relational Calibration Error. The only public reporting entry point,
  `relcal.report.report_rce`, never returns a bare RCE value. It returns the estimate only
  together with a permutation-null p-value and a bootstrap confidence interval, and it raises
  if asked to skip the null. A positive RCE on finite data could be a real relational effect
  or a finite-sample artifact, and the null is what separates the two. Reporting RCE without
  it is disallowed at the level of the function signature.
- The within-language control. Any relational-versus-control comparison holds the language
  fixed. The relational set and the matched nonrelational control set share the language, and
  only the presence of relational context differs. A comparison that mixes languages raises,
  so language is never confounded with relational structure. The comparison entry point
  `relcal.report.compare_relational_vs_control` checks this across the two sets.

A third care is taken at the estimator level. Binned calibration error is biased upward in
finite samples, and more so in sparse within-context bins. The estimator uses adaptive
(equal-mass) binning and a debiased per-bin gap that subtracts the Bernoulli variance
contribution. The bias and the correction are documented in the `relcal.calibration` module
docstring.

## Repository layout

```
relcal/
  schema.py        record types: item id, language, context label, relational flag,
                   per-rater binary judgments; no rater PII
  calibration.py   pooled ECE, within-context ECE_c, RCE; adaptive and debiased binning
  permutation.py   context-label shuffle null, permutation p-value, bootstrap CI
  simulate.py      synthetic generators with a known relational dispersion, plus a
                   no-structure generator for the null
  uq.py            per-item predicted probability and epistemic uncertainty; a mock
                   ensemble backend for tests, a real backend stubbed
  report.py        the only public reporting entry point; RCE with null p-value and CI
experiments/
  e0_estimator_validation.py   estimator validation on synthetic data; runs today
docs/
  theory/Relational_UQ_Formalization.md   the canonical specification and proofs
  REVIEW.md                               target layout and component gap table
tests/                                    estimator properties, null calibration,
                                          report guard, within-language control
```

The canonical specification is `docs/theory/Relational_UQ_Formalization.md`. Everything the
code computes is defined there.

## Requirements

- Python 3.12
- NumPy

There are no other runtime dependencies. The resampling procedures are implemented in NumPy
and the standard library; SciPy is not required.

Put the repository root on the import path when running scripts. The experiment scripts and
the test suite do this automatically; for interactive use, run from the repository root or set
`PYTHONPATH` to it.

## How to run e0 today

The estimator validation experiment runs end to end on synthetic data and needs no human
data. It validates the estimator against ground truth.

```
python3 experiments/e0_estimator_validation.py
python3 experiments/e0_estimator_validation.py --quick   # smaller, faster configuration
```

It demonstrates three things and prints each with the generating configuration.

- Recovery. The estimated Relational Calibration Error tracks the known relational dispersion
  as the dispersion is varied from zero upward.
- Null calibration. Under the no-structure generator the permutation test rejects at
  approximately its nominal false-positive rate, not more.
- Marginal-model bound (Proposition 3). When the model predicts the context-marginal, the
  estimated Relational Calibration Error converges to the mean absolute relational dispersion.

A representative excerpt of the output, with all numbers coming from the run itself, looks
like the following.

```
a. Recovery: estimated RCE tracks the known dispersion
 dispersion d |    MARD |  RCE_est |  p-value
        0.000 |  0.0000 |   0.0047 |   0.146
        0.100 |  0.1000 |   0.0963 |   0.003
        0.200 |  0.2000 |   0.2023 |   0.003

b. Null calibration: permutation false-positive rate under no structure
 empirical false-positive rate at alpha=0.05: 0.050 (target approximately 0.05)

c. Marginal-model bound: RCE converges to MARD
 n_items n_raters | MARD   RCE    pooled  MARD-RCE  MARD in CI
    150     40     | 0.100  0.0895 0.0059   0.0105   False
   1500    500     | 0.100  0.0995 0.0007   0.0005   True
```

A note on the marginal bound, because it is a place where measurement corrected an
expectation. Proposition 3 states the population identity `RCE(bar_pi) = MARD`. In finite
samples the point estimate approaches `MARD` from below, not from above. The reason is that
the marginal model is perfectly calibrated when pooled, so its population pooled `ECE` is
exactly zero; in a finite sample the estimated pooled `ECE` is therefore pure positive bias,
and it is subtracted in `RCE = within - pooled`. The gap between `MARD` and the estimate
shrinks toward zero as the sample grows, and `MARD` enters the bootstrap confidence interval
at large samples. The experiment reports the estimate, the within-context error, the pooled
error, `MARD`, and the confidence interval together, so the convergence is visible rather than
asserted.

## How the pilot (e1) will run, once data exists

The pilot experiment is the next milestone and is not yet in the repository. It is specified
in the build plan and in the formalization. When it lands, it will:

- load one open instruct or DPO model's predicted preference probabilities and one
  uncertainty method's epistemic uncertainty per item, through `relcal.uq`;
- estimate the Relational Calibration Error separately on the relational set and on the
  matched nonrelational control set, each with the permutation null and the bootstrap
  confidence interval, through `relcal.report`;
- state, and then test rather than assume, the prediction that the Relational Calibration
  Error is significantly positive on the relational set and consistent with zero on the
  control set; and
- run the overconfidence check of Proposition 1, correlating per-item epistemic uncertainty
  against per-item relational miscalibration, reporting the correlation with a confidence
  interval and no hardcoded value.

The pilot will refuse to run on empty data with a clear message, will never emit placeholder
results, and will print the full configuration and the language so that the within-language
control is visible in the output. Only a tiny synthetic sample will be shipped in version
control; real annotations will never be committed. See `ETHICS.md` for rater consent, fair
compensation, context labelling, and the no-PII rule once the pilot phase begins.

## Using the library directly

A minimal end-to-end use on synthetic data, with the reporting contract enforced:

```python
from relcal import simulate
from relcal.report import report_rce, format_report

dataset, truth = simulate.generate(n_items=300, n_raters=20, dispersion=0.1, seed=0)
predictions = truth.marginal_predictions(dataset.arrays())   # the marginal model f(x) = bar_pi(x)

report = report_rce(dataset, predictions)   # raises if the permutation null is disabled
print(format_report(report, title="relational set"))
```

`report` carries the estimate, the permutation-null p-value, the bootstrap confidence
interval, the language, and the pooled and within-context components. There is no way to
obtain the estimate without the null.

## Running the tests

```
python3 -m pytest -q
```

The suite covers the non-negativity of the Relational Calibration Error, the analytic
recovery of the dispersion within a two-sided tolerance, the permutation false-positive rate
under no structure, the report no-bare-RCE guard, and the within-language control, including
the comparison path that the pilot will use.

## References

- Ursula Hebert-Johnson, Michael P. Kim, Omer Reingold, and Guy N. Rothblum (2018).
  Multicalibration: Calibration for the (Computationally-Identifiable) Masses. Proceedings of
  the 35th International Conference on Machine Learning.
- Mahdi Pakdaman Naeini, Gregory F. Cooper, and Milos Hauskrecht (2015). Obtaining Well
  Calibrated Probabilities Using Bayesian Binning. AAAI.
- Ananya Kumar, Percy Liang, and Tengyu Ma (2019). Verified Uncertainty Calibration. NeurIPS.
