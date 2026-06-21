# Relational Uncertainty Quantification: Formalization

Status: reconstructed from the project specification, pending author review.

This document is the source of truth for what the code in `relcal/` computes. The
definitions and propositions here fix the estimands; the implementation follows them
and the test suite checks the implementation against the analytic statements derived
below. Prose follows the project style: no contractions, no em dashes, full-form
phrasing.

## 1. Motivation

Preference and reward models are trained to predict a probability that one response is
preferred over another. In many settings the better response depends on a relational
context: who is speaking to whom, the relative social position of the parties, the
register expected of a host toward a guest, of a younger person toward an elder, of an
insider toward an outsider. A model that collapses these contexts into a single
context-marginal preference can appear well calibrated when its predictions are pooled,
while remaining systematically miscalibrated inside individual contexts. This document
defines a statistic, the Relational Calibration Error, that isolates exactly the
calibration error that pooling hides, and relates it to the dispersion of the true
preference across contexts.

The calibration machinery here is group calibration and multicalibration in the sense of
Hebert-Johnson, Kim, Reingold, and Rothblum (2018), "Multicalibration: Calibration for
the (Computationally-Identifiable) Masses." The contribution of this project is not a new
calibration algorithm. It is the choice of groups (relational contexts), the bridge from
group calibration error to relational dispersion (Proposition 3), and the consequence for
uncertainty quantification (Proposition 1).

## 2. Setup and notation

- Let `x` index an item. An item is a preference comparison between two candidate
  responses in a single language.
- Let `c` range over a finite set of relational contexts `C`. Each observation carries a
  context label `c`.
- Let `Y` be the binary preference judgment of a rater for an item under a context:
  `Y = 1` if the designated response is preferred, `Y = 0` otherwise.
- Let `pi(x, c) = P(Y = 1 | x, c)` be the context-conditional preference probability. This
  is the true, unknown probability that the response is preferred for item `x` under
  context `c`.
- Let `rho(c)` be the probability of context `c` (the context mixing weights, normalized so
  that `sum_c rho(c) = 1`). Unless stated otherwise the contexts are taken with these
  weights wherever a "mean over contexts" appears.
- Let `bar_pi(x) = sum_c rho(c) pi(x, c)` be the context-marginal preference probability of
  item `x`. This is the preference probability one obtains by averaging out the context.

A model under test produces a prediction `f(x, c)` in `[0, 1]`. Two model families matter:

- A context-aware model may use the context: `f(x, c)` depends on `c`.
- A context-blind, or marginal, model ignores the context: `f(x, c) = f(x)` for all `c`.
  The ideal marginal model is `f(x) = bar_pi(x)`.

## 3. Calibration error

We use the L1 (absolute) expected calibration error throughout.

### 3.1 Population definitions

For a predictor `f` define the level sets by prediction value `v`. The pooled calibration
error is

```
ECE(f) = E_v | E[Y | f = v] - v |
```

where the outer expectation is over the distribution of the prediction value `v` induced
by the joint distribution of `(x, c)`, and `E[Y | f = v]` averages the true outcome over
all `(x, c)` whose prediction equals `v`. Concretely, `E[Y | f = v]` is an average of
`pi(x, c)` over the pairs that land at level `v`.

The within-context calibration error for context `c` restricts every expectation to that
context:

```
ECE_c(f) = E_{v | c} | E[Y | f = v, c] - v |.
```

### 3.2 Binned and debiased estimators

In finite samples we estimate `ECE` and `ECE_c` by binning predictions. For a partition of
`[0, 1]` into bins `b`,

```
hat_ECE = sum_b (n_b / N) | acc_b - conf_b |,
```

where `n_b` is the count in bin `b`, `N` the total count, `conf_b` the mean prediction in
the bin, and `acc_b` the mean binary outcome in the bin.

The binned estimator of an absolute calibration gap is biased upward in finite samples,
because `acc_b` carries sampling noise and the absolute value is convex:

```
E | acc_b - conf_b | >= | E[acc_b] - conf_b |.
```

The bias is larger when bins are sparse, which is exactly the regime of within-context
estimation, where each context holds a fraction of the data. If this bias is left
uncorrected it inflates `ECE_c` more than it inflates the pooled `ECE`, and therefore
inflates the Relational Calibration Error defined below. Two mitigations are implemented in
`relcal/calibration.py` and are documented there:

1. Adaptive (equal-mass) binning, which keeps every bin populated and reduces the per-bin
   variance relative to fixed-width binning on skewed prediction distributions.
2. A debiased estimator that subtracts an estimate of the noise contribution to each bin
   gap. For a bin whose outcomes are Bernoulli with `n_b` independent draws, the per-bin
   absolute gap has an expected noise floor that is estimated from the bin variance and
   subtracted, with the result floored at zero. The exact estimator and its assumptions are
   stated in the module docstring.

The debiased estimator removes the leading finite-sample inflation. It does not change any
of the population statements in Section 4, which concern the unbiased estimands `ECE` and
`ECE_c`.

## 4. Relational Calibration Error

### 4.1 Definition

The Relational Calibration Error of a predictor `f` is the context-weighted mean of the
within-context calibration errors, minus the pooled calibration error:

```
RCE(f) = ( sum_c rho(c) ECE_c(f) ) - ECE(f).
```

In words: refine the calibration audit by conditioning on the relational context, average
the resulting within-context calibration errors over contexts, and subtract the calibration
error one would have reported by pooling. The Relational Calibration Error is the extra
miscalibration that conditioning on context reveals.

### 4.2 Proposition 2: non-negativity

**Proposition 2.** For any predictor `f`, `RCE(f) >= 0`.

**Proof.** Fix a prediction level `v`. Let `w_c(v) = P(c | f = v)` be the share of the mass
at level `v` coming from context `c`, so that `sum_c w_c(v) = 1`. The pooled conditional
outcome is the mixture `E[Y | f = v] = sum_c w_c(v) E[Y | f = v, c]`. By the triangle
inequality (convexity of the absolute value),

```
| E[Y | f = v] - v | = | sum_c w_c(v) ( E[Y | f = v, c] - v ) |
                     <= sum_c w_c(v) | E[Y | f = v, c] - v |.
```

Take the expectation over the pooled distribution of `v`, written `P(f = v)`:

```
ECE(f) = sum_v P(f = v) | E[Y | f = v] - v |
      <= sum_v P(f = v) sum_c w_c(v) | E[Y | f = v, c] - v |
       = sum_c sum_v P(f = v, c) | E[Y | f = v, c] - v |.
```

Now `P(f = v, c) = rho(c) P(f = v | c)`, so the right-hand side equals

```
sum_c rho(c) sum_v P(f = v | c) | E[Y | f = v, c] - v | = sum_c rho(c) ECE_c(f).
```

Therefore `ECE(f) <= sum_c rho(c) ECE_c(f)`, which is `RCE(f) >= 0`. The mass weighting by
`rho(c)` is what makes the inequality hold for an arbitrary design; an unweighted mean over
contexts does not in general satisfy `RCE >= 0`. This is why the definition in Section 4.1
weights by `rho(c)`. When the contexts are balanced, `rho(c) = 1 / |C|`, the weighted and
unweighted means coincide.

The estimator inherits this only up to finite-sample noise. The debiased estimator of
Section 3.2 is what keeps the empirical `RCE` from being dominated by the upward bias of the
within-context bins; with debiasing, the empirical `RCE` concentrates around the
non-negative population value.

### 4.3 Relational dispersion

Define the mean absolute relational dispersion of the true preference as

```
MARD = E_x sum_c rho(c) | pi(x, c) - bar_pi(x) | = E_{x, c} | pi(x, c) - bar_pi(x) |.
```

This measures how much the true context-conditional preference moves away from its
context-marginal, averaged over items and contexts. It is a property of the world (the
preference structure), not of any model. `MARD = 0` exactly when the preference does not
depend on context for any item, that is, when `pi(x, c) = bar_pi(x)` almost surely.

### 4.4 Proposition 3: the marginal-model bound

**Proposition 3.** Consider a context-blind predictor `f(x, c) = f(x)`. In the population,
with binning fine enough that each item occupies its own level set (predictions injective in
`x`),

```
RCE(f) >= MARD - 2 * E_x | f(x) - bar_pi(x) |.
```

In particular, for the ideal marginal model `f(x) = bar_pi(x)`,

```
RCE(bar_pi) = MARD,
```

so `RCE(bar_pi) >= MARD` holds with equality.

**Proof.** With injective predictions, each level set is a single item `x`. The pooled
conditional outcome at the level of item `x` is `E[Y | f = f(x)] = bar_pi(x)`, and the
within-context conditional outcome is `E[Y | f = f(x), c] = pi(x, c)`. Hence

```
ECE(f)                = E_x | f(x) - bar_pi(x) |,
sum_c rho(c) ECE_c(f) = E_{x, c} | f(x) - pi(x, c) |,
RCE(f)                = E_{x, c} | f(x) - pi(x, c) | - E_x | f(x) - bar_pi(x) |.
```

By the reverse triangle inequality, for every `x` and `c`,

```
| f(x) - pi(x, c) | >= | pi(x, c) - bar_pi(x) | - | f(x) - bar_pi(x) |.
```

Taking `E_{x, c}` and using that `f(x)` and `bar_pi(x)` do not depend on `c`,

```
E_{x, c} | f(x) - pi(x, c) | >= MARD - E_x | f(x) - bar_pi(x) |.
```

Subtracting `ECE(f) = E_x | f(x) - bar_pi(x) |` gives

```
RCE(f) >= MARD - 2 * E_x | f(x) - bar_pi(x) |.
```

For `f = bar_pi` the term `E_x | f(x) - bar_pi(x) |` vanishes, so `RCE(bar_pi) >= MARD`. The
direct computation above gives `ECE(bar_pi) = 0` and
`sum_c rho(c) ECE_c(bar_pi) = MARD`, hence `RCE(bar_pi) = MARD` exactly. ∎

**Remarks for the estimator.**

- The population identity `RCE(bar_pi) = MARD` is what experiment `e0` checks: when the
  model predicts the marginal, the estimated Relational Calibration Error should sit at or
  above `MARD`. It lands at `MARD` in the large-sample, debiased limit, and above `MARD`
  when residual finite-sample bias remains, never systematically below it. The phrase "at or
  above" in the experiment is the honest finite-sample reading of the equality.
- The identity requires the prediction to resolve items rather than collapse several items
  into one bin. When several items share a bin, Jensen averaging inside the bin can only
  lower `sum_c rho(c) ECE_c`, so coarse binning biases the estimate downward, while
  per-bin sampling noise biases it upward. Adaptive binning with enough raters per item is
  chosen so that these effects are small and the upward (conservative) side dominates.

## 5. Proposition 1: the uncertainty-quantification consequence

A model's reported epistemic uncertainty `u(x)` is meant to be large where the model is
most likely to be wrong. For relational items the dangerous error is relational
miscalibration: predicting near `bar_pi(x)` while the context-conditional truth `pi(x, c)`
is far away.

**Proposition 1 (overconfidence on relational structure).** A context-blind model, together
with an epistemic uncertainty `u(x)` that is itself a function of `x` only, cannot have its
uncertainty track per-item relational miscalibration through the context dimension: the
relational miscalibration `m(x, c) = | f(x) - pi(x, c) |` varies with `c` while `u(x)` does
not, so within an item the two are uncorrelated by construction. Across items, if `u(x)`
reflects context-marginal confidence (small where `bar_pi(x)` is near `0` or `1`), then
items with the most extreme marginals, which also admit the largest possible relational
dispersion, receive the smallest reported uncertainty. The reported uncertainty is then
lowest exactly where relational miscalibration is highest.

The empirical content is a testable sign: the correlation between per-item epistemic
uncertainty and per-item relational miscalibration is non-positive for a context-blind model
with context-blind uncertainty. Experiment `e1` estimates this correlation with a confidence
interval and reports it; it does not assume the sign. The proposition is the prediction; the
data adjudicate it.

## 6. The permutation null and the bootstrap

A positive estimated `RCE` on finite data could be a real relational effect or a
finite-sample artifact, since even with `pi(x, c) = bar_pi(x)` the within-context bins are
noisier than the pooled bins. Two resampling procedures separate these.

### 6.1 Permutation null

Under the null hypothesis that the context labels carry no information about the preference
beyond what the item already carries, the labels are exchangeable across observations of the
same item. The permutation null shuffles the context labels (within item, preserving each
item's set of labels and the per-context marginal counts) and recomputes `RCE` on the
shuffled data. Repeating this builds the null distribution of the estimator. The permutation
p-value is the fraction of shuffled statistics greater than or equal to the observed
statistic, with the usual add-one correction:

```
p = (1 + #{ RCE_perm >= RCE_obs }) / (1 + B).
```

Because `RCE` is non-negative and the null distribution is centered above zero by the same
finite-sample bias that affects the observed statistic, the permutation test automatically
calibrates the bias away: the observed statistic is compared against statistics computed
under the same binning and sample size. Experiment `e0` checks that, under the no-structure
generator, this test rejects at approximately its nominal level and not more.

### 6.2 Bootstrap confidence interval

A confidence interval for `RCE` is obtained by resampling items with replacement (the item
is the independent unit), recomputing `RCE` on each resample, and taking percentile bounds.
The interval is reported alongside the point estimate and the p-value.

## 7. The reporting contract

The only public reporting entry point, `relcal/report.py`, never returns a bare `RCE`
value. It returns the point estimate together with the permutation-null p-value and the
bootstrap confidence interval, and it raises if asked to skip the null. This is a structural
rule, enforced by a test, precisely because a bare `RCE` invites the finite-sample artifact
reading that Section 6 exists to rule out.

## 8. The within-language control

Relational structure must be measured against a matched control, not against a different
language. Every relational-versus-nonrelational comparison is conducted within a single
language. The relational set and the nonrelational control set share the language and are
matched in construction; only the presence of relational context differs. The code enforces
this: a comparison that mixes languages raises, and a test asserts the failure.

## 9. Summary of estimands and their checks

| Symbol | Meaning | Population statement | Checked in |
| --- | --- | --- | --- |
| `ECE(f)` | pooled calibration error | non-negative | unit test |
| `ECE_c(f)` | within-context calibration error | non-negative | unit test |
| `RCE(f)` | relational calibration error | `>= 0` (Prop. 2) | unit test, e0 |
| `MARD` | mean absolute relational dispersion | `RCE(bar_pi) = MARD` (Prop. 3) | e0 marginal bound |
| `p` | permutation p-value | nominal false-positive rate under no structure | e0 null calibration |
| `corr(u, m)` | uncertainty vs relational miscalibration | non-positive for context-blind model (Prop. 1) | e1 overconfidence |

## References

- Ursula Hebert-Johnson, Michael P. Kim, Omer Reingold, and Guy N. Rothblum (2018).
  Multicalibration: Calibration for the (Computationally-Identifiable) Masses. Proceedings
  of the 35th International Conference on Machine Learning.
- Mahdi Pakdaman Naeini, Gregory F. Cooper, and Milos Hauskrecht (2015). Obtaining Well
  Calibrated Probabilities Using Bayesian Binning. AAAI.
- Ananya Kumar, Percy Liang, and Tengyu Ma (2019). Verified Uncertainty Calibration.
  NeurIPS. (Source for the binned-calibration bias discussion and debiasing.)
