"""Curve fitting of timecourses and parameters.

`fit`, `fit_timecourses` and `fit_table` fit a `Model` from
`pkpdutils.fit.models` to data with `scipy.optimize.least_squares`, see
`docs/fitting.md`; `FitOptions` selects the scale, the weighting and the
multi-start and bootstrap settings, `FitResult` holds the parameters.
"""

from pkpdutils.fit.engine import RowFit, build_result, fit, fit_row, fit_rows
from pkpdutils.fit.model import Model, ModelParameter, parameter_unit_expression
from pkpdutils.fit.options import (
    FitFlag,
    FitOptions,
    ParameterScale,
    Weighting,
    decode_fit_flags,
)
from pkpdutils.fit.result import FitResult

__all__ = [
    "FitFlag",
    "FitOptions",
    "FitResult",
    "Model",
    "ModelParameter",
    "ParameterScale",
    "RowFit",
    "Weighting",
    "build_result",
    "decode_fit_flags",
    "fit",
    "fit_row",
    "fit_rows",
    "parameter_unit_expression",
]
