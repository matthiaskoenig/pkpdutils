"""Curve fitting of timecourses and parameters.

`fit`, `fit_timecourses` and `fit_table` fit a `Model` from
`pkpdutils.fit.models` to data with `scipy.optimize.least_squares`, see
`docs/fitting.md`; `FitOptions` selects the scale, the weighting and the
multi-start and bootstrap settings, `FitResult` holds the parameters.
"""

from pkpdutils.fit.model import Model, ModelParameter, parameter_unit_expression
from pkpdutils.fit.options import (
    FitFlag,
    FitOptions,
    ParameterScale,
    Weighting,
    decode_fit_flags,
)

__all__ = [
    "FitFlag",
    "FitOptions",
    "Model",
    "ModelParameter",
    "ParameterScale",
    "Weighting",
    "decode_fit_flags",
    "parameter_unit_expression",
]
