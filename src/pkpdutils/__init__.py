"""pkpdutils: pharmacokinetic and pharmacodynamic analysis of timecourses and parameters.

The data model is `Timecourse` (one curve) and `Timecourses` (a batch as an
xarray dataset), see `pkpdutils.timecourse`; units are pint quantities of the
shared registry `ureg`, see `pkpdutils.units`. The analyses are
`pkpdutils.nca` (non-compartmental analysis) and `pkpdutils.fit` (curve
fitting), both returning a `pkpdutils.result.ParameterResult`.
"""

from pkpdutils.fit import (
    FitOptions,
    FitResult,
    compare_models,
    fit,
    fit_table,
    fit_timecourse,
    fit_timecourses,
)
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
    "FitOptions",
    "FitResult",
    "NCAOptions",
    "NCAResult",
    "Quantity",
    "Route",
    "TerminalPhase",
    "Timecourse",
    "Timecourses",
    "__version__",
    "compare_models",
    "fit",
    "fit_table",
    "fit_timecourse",
    "fit_timecourses",
    "nca",
    "nca_single",
    "partial_auc",
    "ureg",
]
