"""Curve fitting of timecourses and parameters.

`fit`, `fit_timecourses` and `fit_table` fit a `Model` from
`pkpdutils.fit.models` to data with `scipy.optimize.least_squares`, see
`docs/fitting.md`; `FitOptions` selects the scale, the weighting and the
multi-start and bootstrap settings, `FitResult` holds the parameters.
`compare_models` fits several models to the same data and ranks them by the
corrected Akaike information criterion. `proportionality_test` applies the
confidence interval criterion of dose proportionality to a `Power` fit.
"""

from pkpdutils.fit.compare import ModelComparison, compare_models
from pkpdutils.fit.engine import RowFit, build_result, fit, fit_row, fit_rows
from pkpdutils.fit.frontends import fit_table, fit_timecourses
from pkpdutils.fit.model import Model, ModelParameter, parameter_unit_expression
from pkpdutils.fit.options import (
    FitFlag,
    FitOptions,
    ParameterScale,
    Weighting,
    decode_fit_flags,
)
from pkpdutils.fit.proportionality import proportionality_test
from pkpdutils.fit.result import FitResult

__all__ = [
    "FitFlag",
    "FitOptions",
    "FitResult",
    "Model",
    "ModelComparison",
    "ModelParameter",
    "ParameterScale",
    "RowFit",
    "Weighting",
    "build_result",
    "compare_models",
    "decode_fit_flags",
    "fit",
    "fit_row",
    "fit_rows",
    "fit_table",
    "fit_timecourses",
    "parameter_unit_expression",
    "proportionality_test",
]
