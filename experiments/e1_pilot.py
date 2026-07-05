"""e1: pilot on annotated data, with the overconfidence check of Proposition 1.

This experiment is the pilot specified in Section 5 and Section 7 of
docs/theory/Relational_UQ_Formalization.md and in the README. It:

  1. loads per-item predicted preference probabilities and per-item epistemic uncertainty
     from a UQ backend (relcal.uq);
  2. estimates the Relational Calibration Error separately on the relational set and on
     the matched nonrelational control set, each with the permutation null and the
     bootstrap confidence interval, through relcal.report (the within-language control is
     enforced across the two sets);
  3. states, and then tests rather than assumes, the prediction that RCE is significantly
     positive on the relational set and consistent with zero on the control set; and
  4. runs the overconfidence check of Proposition 1, correlating per-item epistemic
     uncertainty against per-item relational miscalibration, reporting the correlation
     with a bootstrap confidence interval and no hardcoded value.

The pilot refuses to run on empty data with a clear message and never emits placeholder
results. It prints the full configuration and the language so that the within-language
control is visible in the output.

Backends:
  --backend mock   A context-blind mock ensemble (relcal.uq.MockEnsembleUQ) whose per-item
                   center is the empirical context-marginal preference estimated from the
                   loaded data. This is the context-blind model of Proposition 1 and it
                   runs today on the synthetic sample.
  --backend real   The real-model stub (relcal.uq.RealEnsembleUQ). It raises
                   NotImplementedError until wired to a model, so this path cannot
                   fabricate numbers.

A note on the shipped synthetic sample: its generator uses a context offset that is
constant across items, so the per-item relational miscalibration is nearly constant across
items and the population correlation of Proposition 1 is near zero rather than negative.
On the sample this experiment is therefore a plumbing validation of the pilot pipeline;
the sign prediction of Proposition 1 is adjudicated on real annotated data, where the
relational dispersion varies across items. The experiment reports whatever the data say.

Run from the repository root:
    python3 experiments/e1_pilot.py
    python3 experiments/e1_pilot.py --quick
    python3 experiments/e1_pilot.py --data path/to/judgments.jsonl
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running as a script from anywhere: put the repository root on the path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from relcal.dataio import load_judgments_jsonl
from relcal.report import RelationalControlComparison, compare_relational_vs_control, format_report
from relcal.schema import Arrays, PreferenceDataset
from relcal.uq import MockEnsembleUQ, RealEnsembleUQ, UQBackend, UQPredictions

DEFAULT_DATA = _REPO_ROOT / "data" / "sample" / "sample.jsonl"


@dataclass(frozen=True)
class OverconfidenceResult:
    """The Proposition 1 correlation with its bootstrap confidence interval."""

    n_items: int
    pearson: float
    pearson_ci: tuple[float, float]
    spearman: float
    spearman_ci: tuple[float, float]
    confidence: float
    n_boot: int


@dataclass(frozen=True)
class PilotResult:
    """Everything the pilot reports: the paired RCE reports and the Prop 1 check."""

    comparison: RelationalControlComparison
    overconfidence: OverconfidenceResult


def _empirical_marginal_by_item(arrays: Arrays) -> dict[str, float]:
    """Per-item empirical context-marginal preference: the mean judgment over all rows.

    This is the context-blind estimate of bar_pi(x). It pools every context's judgments
    for the item, weighting contexts by their observation counts, which matches the
    rho(c) weighting of the formalization when raters per (item, context) are balanced.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for item, judgment in zip(arrays.item_id, arrays.judgment):
        sums[item] = sums.get(item, 0.0) + float(judgment)
        counts[item] = counts.get(item, 0) + 1
    return {item: sums[item] / counts[item] for item in sums}


def _relational_miscalibration_by_item(
    arrays: Arrays, predictions: UQPredictions
) -> tuple[list[str], np.ndarray]:
    """Per-item relational miscalibration m(x) = sum_c w_c(x) | f(x) - pi_hat(x, c) |.

    pi_hat(x, c) is the empirical preference frequency in the (item, context) cell and
    w_c(x) is the share of the item's observations in context c. This is the empirical
    counterpart of the miscalibration in Proposition 1.
    """
    index = {item: i for i, item in enumerate(predictions.item_ids)}
    cell_sums: dict[tuple[str, str], float] = {}
    cell_counts: dict[tuple[str, str], int] = {}
    item_counts: dict[str, int] = {}
    for item, context, judgment in zip(arrays.item_id, arrays.context, arrays.judgment):
        key = (item, context)
        cell_sums[key] = cell_sums.get(key, 0.0) + float(judgment)
        cell_counts[key] = cell_counts.get(key, 0) + 1
        item_counts[item] = item_counts.get(item, 0) + 1

    items = sorted(item_counts.keys())
    m = np.zeros(len(items), dtype=float)
    for i, item in enumerate(items):
        f_x = float(predictions.probability[index[item]])
        total = item_counts[item]
        for (cell_item, _context), count in cell_counts.items():
            if cell_item != item:
                continue
            pi_hat = cell_sums[(cell_item, _context)] / count
            m[i] += (count / total) * abs(f_x - pi_hat)
    return items, m


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties sharing the mean of their positions."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        mean_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = mean_rank
        i = j + 1
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_average_ranks(a), _average_ranks(b))


def overconfidence_check(
    data: PreferenceDataset,
    predictions: UQPredictions,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng_seed: int = 0,
) -> OverconfidenceResult:
    """Correlate per-item epistemic uncertainty with per-item relational miscalibration.

    Proposition 1 predicts a non-positive correlation for a context-blind model with
    context-blind uncertainty. The check estimates the correlation and a bootstrap
    percentile confidence interval over items; it does not assume the sign.
    """
    arrays = data.arrays()
    items, m = _relational_miscalibration_by_item(arrays, predictions)
    index = {item: i for i, item in enumerate(predictions.item_ids)}
    u = np.array([predictions.epistemic_uncertainty[index[item]] for item in items], dtype=float)

    pearson = _pearson(u, m)
    spearman = _spearman(u, m)

    rng = np.random.default_rng(rng_seed)
    n = len(items)
    boot_p: list[float] = []
    boot_s: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p = _pearson(u[idx], m[idx])
        s = _spearman(u[idx], m[idx])
        if np.isfinite(p):
            boot_p.append(p)
        if np.isfinite(s):
            boot_s.append(s)

    alpha = 1.0 - confidence
    def _ci(samples: list[float]) -> tuple[float, float]:
        if not samples:
            return (float("nan"), float("nan"))
        lo, hi = np.quantile(np.array(samples), [alpha / 2.0, 1.0 - alpha / 2.0])
        return (float(lo), float(hi))

    return OverconfidenceResult(
        n_items=n,
        pearson=pearson,
        pearson_ci=_ci(boot_p),
        spearman=spearman,
        spearman_ci=_ci(boot_s),
        confidence=confidence,
        n_boot=n_boot,
    )


def _build_backend(name: str, data: PreferenceDataset, *, model: Optional[str], seed: int) -> UQBackend:
    if name == "mock":
        marginals = _empirical_marginal_by_item(data.arrays())
        return MockEnsembleUQ(marginals, seed=seed)
    if name == "real":
        if not model:
            raise ValueError("--backend real requires --model NAME")
        return RealEnsembleUQ(model)
    raise ValueError(f"unknown backend {name!r}; use 'mock' or 'real'")


def run_pilot(
    data_path: Path,
    *,
    backend_name: str = "mock",
    model: Optional[str] = None,
    n_permutations: int = 1000,
    n_boot: int = 1000,
    corr_n_boot: int = 2000,
    confidence: float = 0.95,
    rng_seed: int = 0,
) -> PilotResult:
    """Load the data, report RCE on both sets, and run the overconfidence check.

    Raises:
        FileNotFoundError: if the data file does not exist.
        ValueError: if the file is empty, either set is empty, or the within-language
            control is violated. The pilot refuses to run rather than emit placeholders.
    """
    data = load_judgments_jsonl(data_path)
    relational = data.filter(relational=True)
    control = data.filter(relational=False)
    if len(relational) == 0:
        raise ValueError(
            f"the relational set in {data_path} is empty; the pilot refuses to run on "
            "empty data"
        )
    if len(control) == 0:
        raise ValueError(
            f"the nonrelational control set in {data_path} is empty; the pilot requires "
            "the matched control (Section 8 of the formalization)"
        )

    backend = _build_backend(backend_name, data, model=model, seed=rng_seed)

    rel_arrays = relational.arrays()
    ctl_arrays = control.arrays()
    rel_items = sorted(set(rel_arrays.item_id.tolist()))
    ctl_items = sorted(set(ctl_arrays.item_id.tolist()))
    rel_pred = backend.predict_items(rel_items)
    ctl_pred = backend.predict_items(ctl_items)

    comparison = compare_relational_vs_control(
        relational,
        rel_pred.probability_for(rel_arrays),
        control,
        ctl_pred.probability_for(ctl_arrays),
        n_permutations=n_permutations,
        n_boot=n_boot,
        confidence=confidence,
        rng_seed=rng_seed,
    )
    overconfidence = overconfidence_check(
        relational,
        rel_pred,
        n_boot=corr_n_boot,
        confidence=confidence,
        rng_seed=rng_seed + 13,
    )
    return PilotResult(comparison=comparison, overconfidence=overconfidence)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="JSONL judgment file")
    parser.add_argument("--backend", choices=("mock", "real"), default="mock")
    parser.add_argument("--model", default=None, help="model name for --backend real")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--corr-n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--quick", action="store_true", help="smaller, faster configuration")
    args = parser.parse_args(argv)

    n_permutations = 200 if args.quick else args.n_permutations
    n_boot = 200 if args.quick else args.n_boot
    corr_n_boot = 500 if args.quick else args.corr_n_boot

    print("e1: pilot with relational-versus-control comparison and the Prop 1 check")
    print(f"  data          = {args.data}")
    print(f"  backend       = {args.backend}" + (f" (model = {args.model})" if args.model else ""))
    print(f"  permutations  = {n_permutations}   bootstrap = {n_boot}   corr bootstrap = {corr_n_boot}")
    print(f"  seed          = {args.seed}   alpha = {args.alpha}")
    print()

    result = run_pilot(
        args.data,
        backend_name=args.backend,
        model=args.model,
        n_permutations=n_permutations,
        n_boot=n_boot,
        corr_n_boot=corr_n_boot,
        rng_seed=args.seed,
    )

    comparison = result.comparison
    print(f"within-language control: both sets are in language {comparison.language!r}")
    print()
    print(format_report(comparison.relational, title="relational set"))
    print()
    print(format_report(comparison.control, title="nonrelational control set"))
    print()

    rel_sig = comparison.relational.is_significant(args.alpha)
    ctl_sig = comparison.control.is_significant(args.alpha)
    print("prediction under test: RCE significantly positive on the relational set and")
    print("consistent with zero on the control set.")
    print(f"  relational set: p = {comparison.relational.p_value:.4f} -> "
          f"{'significant' if rel_sig else 'not significant'} at alpha = {args.alpha}")
    print(f"  control set:    p = {comparison.control.p_value:.4f} -> "
          f"{'significant' if ctl_sig else 'not significant'} at alpha = {args.alpha}")
    verdict = "consistent with the prediction" if (rel_sig and not ctl_sig) else "NOT consistent with the prediction"
    print(f"  outcome: {verdict}")
    print()

    oc = result.overconfidence
    pct = int(round(oc.confidence * 100))
    print("overconfidence check (Proposition 1), relational set:")
    print("  correlation of per-item epistemic uncertainty with per-item relational")
    print("  miscalibration; the proposition predicts a non-positive value for a")
    print("  context-blind model. The data adjudicate; nothing is assumed.")
    print(f"  items = {oc.n_items}")
    print(f"  Pearson  = {oc.pearson:+.4f}  {pct}% CI [{oc.pearson_ci[0]:+.4f}, {oc.pearson_ci[1]:+.4f}]")
    print(f"  Spearman = {oc.spearman:+.4f}  {pct}% CI [{oc.spearman_ci[0]:+.4f}, {oc.spearman_ci[1]:+.4f}]")
    if args.data == DEFAULT_DATA:
        print("  note: on the shipped constant-offset synthetic sample the per-item")
        print("  miscalibration is nearly constant across items, so the correlation is")
        print("  expected to be near zero here; the sign prediction is adjudicated on")
        print("  real annotated data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
