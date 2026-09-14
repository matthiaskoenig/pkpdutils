"""Non-compartmental analysis of timecourses.

`nca` analyses a `Timecourses` batch, `nca_single` one `Timecourse`;
`NCAOptions` selects the methods, `NCAResult` holds the parameters, see
`docs/nca.md`.
"""

from pkpdutils.nca.nca import nca, nca_single
from pkpdutils.nca.options import (
    AUCMethod,
    BLQHandling,
    C0Method,
    Kind,
    NCAFlag,
    NCAOptions,
    TerminalMethod,
    TerminalPhase,
    decode_flags,
)
from pkpdutils.nca.result import NCAResult

__all__ = [
    "AUCMethod",
    "BLQHandling",
    "C0Method",
    "Kind",
    "NCAFlag",
    "NCAOptions",
    "NCAResult",
    "TerminalMethod",
    "TerminalPhase",
    "decode_flags",
    "nca",
    "nca_single",
]
