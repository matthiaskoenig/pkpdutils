"""Non-compartmental analysis of timecourses.

`nca` analyses a `Timecourses` batch, `nca_single` one `Timecourse`;
`NCAOptions` selects the methods, `NCAResult` holds the parameters, see
`docs/nca.md`.
"""

from pkpdutils.nca.nca import nca, nca_single, partial_auc
from pkpdutils.nca.options import (
    Acceptance,
    AUCMethod,
    BLQAction,
    BLQHandling,
    BLQRules,
    BootstrapDistribution,
    BootstrapSpread,
    C0Method,
    Kind,
    NCAFlag,
    NCAOptions,
    TerminalMethod,
    TerminalPhase,
    UncertaintyMethod,
    decode_flags,
)
from pkpdutils.nca.report import M13A_STATISTICS, acceptability_table, methods_line
from pkpdutils.nca.result import NCAResult
from pkpdutils.nca.steady_state import accumulation_ratio, superposition

__all__ = [
    "M13A_STATISTICS",
    "AUCMethod",
    "Acceptance",
    "BLQAction",
    "BLQHandling",
    "BLQRules",
    "BootstrapDistribution",
    "BootstrapSpread",
    "C0Method",
    "Kind",
    "NCAFlag",
    "NCAOptions",
    "NCAResult",
    "TerminalMethod",
    "TerminalPhase",
    "UncertaintyMethod",
    "acceptability_table",
    "accumulation_ratio",
    "decode_flags",
    "methods_line",
    "nca",
    "nca_single",
    "partial_auc",
    "superposition",
]
