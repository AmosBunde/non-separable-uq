"""relcal: relational calibration error estimation.

The canonical specification is docs/theory/Relational_UQ_Formalization.md. The public
reporting entry point is relcal.report.report_rce, which never returns a bare RCE value;
it returns the estimate together with a permutation-null p-value and a bootstrap
confidence interval. See the module for the contract.
"""

from relcal.schema import Judgment, PreferenceDataset, Arrays

__all__ = ["Judgment", "PreferenceDataset", "Arrays"]

__version__ = "0.1.0"
