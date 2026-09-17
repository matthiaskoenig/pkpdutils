"""Non-compartmental analysis of timecourses.

`nca` analyses a `Timecourses` batch, `nca_single` one `Timecourse`;
`NCAOptions` selects the methods, `NCAResult` holds the parameters, see
`docs/nca.md`. `nca_urine` analyses the urinary excretion of a subject
(`Excretion`, see `docs/urine.md`) and `nca_sparse` a sparse or destructive
sampling design (`docs/sparse.md`).
"""

from pkpdutils.nca.analytes import metabolite_ratio
from pkpdutils.nca.bioavailability import bioavailability
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
from pkpdutils.nca.sparse import nca_sparse, sparse_mean
from pkpdutils.nca.steady_state import accumulation_ratio, superposition
from pkpdutils.nca.tss import TSSResult, time_to_steady_state
from pkpdutils.nca.urine import Excretion, nca_urine

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
    "Excretion",
    "Kind",
    "NCAFlag",
    "NCAOptions",
    "NCAResult",
    "TSSResult",
    "TerminalMethod",
    "TerminalPhase",
    "UncertaintyMethod",
    "acceptability_table",
    "accumulation_ratio",
    "bioavailability",
    "decode_flags",
    "metabolite_ratio",
    "methods_line",
    "nca",
    "nca_single",
    "nca_sparse",
    "nca_urine",
    "partial_auc",
    "sparse_mean",
    "superposition",
    "time_to_steady_state",
]
