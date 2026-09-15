"""pkpdutils: pharmacokinetic and pharmacodynamic analysis of timecourses and parameters.

The data model is `Timecourse` (one curve) and `Timecourses` (a batch as an
xarray dataset), see `pkpdutils.timecourse`; units are pint quantities of the
shared registry `ureg`, see `pkpdutils.units`. The analyses are
`pkpdutils.nca` (non-compartmental analysis) and `pkpdutils.fit` (curve
fitting), both returning a `pkpdutils.result.ParameterResult`. `pkpdutils.stats`
holds the statistics on parameters (tests, ratios, bioequivalence,
drug-drug interactions, meta-analysis).
"""

from pkpdutils.fit import (
    FitOptions,
    FitResult,
    compare_models,
    fit,
    fit_table,
    fit_timecourse,
    fit_timecourses,
    proportionality_test,
)
from pkpdutils.nca import (
    NCAOptions,
    NCAResult,
    TerminalPhase,
    nca,
    nca_single,
    partial_auc,
)
from pkpdutils.stats import (
    ParameterSample,
    bioequivalence,
    compare,
    ddi_classification,
    meta_analysis,
    ratio,
)
from pkpdutils.timecourse import (
    Dose,
    Dosing,
    DosingRegimen,
    Route,
    Timecourse,
    Timecourses,
)
from pkpdutils.units import Q_, Quantity, ureg

__version__ = "1.0.1.dev0"

__all__ = [
    "Q_",
    "Dose",
    "Dosing",
    "DosingRegimen",
    "FitOptions",
    "FitResult",
    "NCAOptions",
    "NCAResult",
    "ParameterSample",
    "Quantity",
    "Route",
    "TerminalPhase",
    "Timecourse",
    "Timecourses",
    "__version__",
    "bioequivalence",
    "compare",
    "compare_models",
    "ddi_classification",
    "fit",
    "fit_table",
    "fit_timecourse",
    "fit_timecourses",
    "meta_analysis",
    "nca",
    "nca_single",
    "partial_auc",
    "proportionality_test",
    "ratio",
    "ureg",
]
