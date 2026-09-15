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
`import pkpdutils`.

The namespace carries what an analysis needs: the data model, the front ends
of the analyses, the options with the enumerations which configure them and
the model library of the fit, so that a script imports from `pkpdutils` and
`pkpdutils.plot` only.
"""

# the re-exports make `pkpdutils.io` and `pkpdutils.plot` reachable after
# `import pkpdutils`
from pkpdutils import io as io
from pkpdutils import plot as plot
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
    AUCMethod,
    BLQHandling,
    BootstrapDistribution,
    BootstrapSpread,
    C0Method,
    Kind,
    NCAFlag,
    NCAOptions,
    NCAResult,
    TerminalMethod,
    TerminalPhase,
    UncertaintyMethod,
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

# `summary_table` is exported by Task C2
__all__ = [
    "Q_",
    "AUCMethod",
    "Allometric",
    "BLQHandling",
    "Bateman",
    "BiExp",
    "BootstrapDistribution",
    "BootstrapSpread",
    "C0Method",
    "Dose",
    "Dosing",
    "DosingRegimen",
    "Emax",
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
    "TerminalMethod",
    "TerminalPhase",
    "Timecourse",
    "Timecourses",
    "TriExp",
    "UncertaintyMethod",
    "Weighting",
    "__version__",
    "bioequivalence",
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
    "partial_auc",
    "plot",
    "proportionality_test",
    "ratio",
    "ureg",
]
