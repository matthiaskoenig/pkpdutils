"""pkpdutils: pharmacokinetic and pharmacodynamic analysis of timecourses and parameters.

The data model is `Timecourse` (one curve) and `Timecourses` (a batch as an
xarray dataset), see `pkpdutils.timecourse`; units are pint quantities of the
shared registry `ureg`, see `pkpdutils.units`.
"""

from pkpdutils.nca import (
    NCAOptions,
    NCAResult,
    TerminalPhase,
    nca,
    nca_single,
    partial_auc,
)
from pkpdutils.timecourse import Dose, DosingRegimen, Route, Timecourse, Timecourses
from pkpdutils.units import Q_, Quantity, ureg

__version__ = "1.0.0.dev0"

__all__ = [
    "Q_",
    "Dose",
    "DosingRegimen",
    "NCAOptions",
    "NCAResult",
    "Quantity",
    "Route",
    "TerminalPhase",
    "Timecourse",
    "Timecourses",
    "__version__",
    "nca",
    "nca_single",
    "partial_auc",
    "ureg",
]
