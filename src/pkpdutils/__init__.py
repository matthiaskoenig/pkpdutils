"""pkpdutils: pharmacokinetic and pharmacodynamic analysis of timecourses and parameters.

The data model is `Timecourse` (one curve) and `Timecourses` (a batch as an
xarray dataset), see `pkpdutils.timecourse`; units are pint quantities of the
shared registry `ureg`, see `pkpdutils.units`. The analyses are
`pkpdutils.nca` (non-compartmental analysis) and `pkpdutils.fit` (curve
fitting), both returning a `pkpdutils.result.ParameterResult`. `pkpdutils.stats`
holds the statistics on parameters (tests, ratios, bioequivalence,
drug-drug interactions, meta-analysis). `pkpdutils.io` reads and writes the
exchange formats of the field (event records, PKNCA tables, CDISC ADNCA) and
`pkpdutils.plot` draws the figures; both are reachable after
`import pkpdutils`, `plot` on first use (it imports matplotlib, which a
script that draws nothing does not pay for).

The namespace carries what an analysis needs: the data model, the front ends
of the analyses, the options with the enumerations which configure them, the
model library of the fit and `summary_table`, the parameter table of a
publication, so that a script imports from `pkpdutils` and `pkpdutils.plot`
only.
"""

import importlib
from typing import TYPE_CHECKING, Any

# the re-export makes `pkpdutils.io` reachable after `import pkpdutils`;
# `pkpdutils.plot` is imported by `__getattr__` on first use
from pkpdutils import io as io
from pkpdutils.fit import (
    FitFlag,
    FitOptions,
    FitResult,
    ParameterScale,
    Weighting,
    compare_models,
    fit,
    fit_table,
    fit_timecourse,
    fit_timecourses,
    proportionality_test,
)
from pkpdutils.fit.models import (
    Allometric,
    Bateman,
    BiExp,
    Emax,
    Imax,
    Linear,
    LogLinear,
    MonoExp,
    Power,
    SigmoidEmax,
    SigmoidImax,
    TriExp,
)
from pkpdutils.fit.proportionality import ProportionalityResult
from pkpdutils.nca import (
    Acceptance,
    AUCMethod,
    BLQAction,
    BLQHandling,
    BLQRules,
    BootstrapDistribution,
    BootstrapSpread,
    C0Method,
    Excretion,
    Kind,
    NCAFlag,
    NCAOptions,
    NCAResult,
    TerminalMethod,
    TerminalPhase,
    TSSResult,
    UncertaintyMethod,
    bioavailability,
    nca,
    nca_single,
    nca_sparse,
    nca_urine,
    partial_auc,
    sparse_mean,
    time_to_steady_state,
)
from pkpdutils.result import summary_table
from pkpdutils.stats import (
    ParameterSample,
    bioequivalence,
    carryover_table,
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

if TYPE_CHECKING:
    # the lazy `plot` of `__getattr__` as a name a type checker resolves
    from pkpdutils import plot as plot

__version__ = "1.2.0.dev0"

__all__ = [
    "Q_",
    "AUCMethod",
    "Acceptance",
    "Allometric",
    "BLQAction",
    "BLQHandling",
    "BLQRules",
    "Bateman",
    "BiExp",
    "BootstrapDistribution",
    "BootstrapSpread",
    "C0Method",
    "Dose",
    "Dosing",
    "DosingRegimen",
    "Emax",
    "Excretion",
    "FitFlag",
    "FitOptions",
    "FitResult",
    "Imax",
    "Kind",
    "Linear",
    "LogLinear",
    "MonoExp",
    "NCAFlag",
    "NCAOptions",
    "NCAResult",
    "ParameterSample",
    "ParameterScale",
    "Power",
    "ProportionalityResult",
    "Quantity",
    "Route",
    "SigmoidEmax",
    "SigmoidImax",
    "TSSResult",
    "TerminalMethod",
    "TerminalPhase",
    "Timecourse",
    "Timecourses",
    "TriExp",
    "UncertaintyMethod",
    "Weighting",
    "__version__",
    "bioavailability",
    "bioequivalence",
    "carryover_table",
    "compare",
    "compare_models",
    "ddi_classification",
    "fit",
    "fit_table",
    "fit_timecourse",
    "fit_timecourses",
    "io",
    "meta_analysis",
    "nca",
    "nca_single",
    "nca_sparse",
    "nca_urine",
    "partial_auc",
    "plot",
    "proportionality_test",
    "ratio",
    "sparse_mean",
    "summary_table",
    "time_to_steady_state",
    "ureg",
]


def __getattr__(name: str) -> Any:
    """Import `pkpdutils.plot` on first use (PEP 562).

    The figures need matplotlib, which is slow to import and useless to a
    script that draws nothing, so the module is imported when the attribute
    is first read and cached in the namespace afterwards.

    Args:
        name: name of the attribute.

    Returns:
        The `pkpdutils.plot` module.

    Raises:
        AttributeError: for any other name.
    """
    if name == "plot":
        module = importlib.import_module("pkpdutils.plot")
        globals()["plot"] = module
        return module
    raise AttributeError(f"module 'pkpdutils' has no attribute '{name}'")
