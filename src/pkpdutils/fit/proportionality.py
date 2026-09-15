"""Dose proportionality by the power model and the confidence interval criterion.

With `AUC = a D^b` the exposure is dose proportional when `b = 1`. Smith et
al. (2000) accept proportionality over a dose range `r = D_high / D_low` when
the confidence interval of `b` lies within
`[1 + ln(theta_L) / ln(r), 1 + ln(theta_H) / ln(r)]` with the acceptance limits
`(theta_L, theta_H) = (0.8, 1.25)` of the dose-normalized exposure ratio.
"""

import math

import numpy as np
import xarray as xr

from pkpdutils.fit.result import FitResult


def proportionality_test(
    result: FitResult,
    *,
    dose_range: tuple[float, float],
    criterion: tuple[float, float] = (0.8, 1.25),
) -> xr.Dataset:
    """Apply the confidence interval criterion to the exponent of a power model fit.

    The bounds of the exponent are `bound_low = 1 + ln(theta_L) / ln(r)` and
    `bound_high = 1 + ln(theta_H) / ln(r)`, with `r = D_high / D_low` of
    `dose_range` and the acceptance limits `(theta_L, theta_H)` of `criterion`
    (Smith et al. 2000). `proportional` is set when the confidence interval of
    `b` lies inside `[bound_low, bound_high]`; `inconclusive` when it overlaps
    the bounds without lying inside.

    Args:
        result: fit of `Power` (or `Allometric` with a free exponent), with `b`, `b_ci_low`, `b_ci_high`
        dose_range: lowest and highest dose of the range the criterion refers to
        criterion: acceptance limits of the dose-normalized exposure ratio

    Returns:
        Dataset over the sample dimensions with `b`, `b_ci_low`, `b_ci_high`,
        `bound_low`, `bound_high`, `proportional` (the interval is inside the
        bounds) and `inconclusive` (the interval overlaps the bounds but is
        not inside), and `attrs["dose_range"]`/`attrs["criterion"]`.

    Raises:
        ValueError: if the result has no exponent `b` with a confidence
            interval, or `dose_range` is not `0 < low < high`.
    """
    if "b" not in result or "b_ci_low" not in result or "b_ci_high" not in result:
        raise ValueError("The result has no exponent 'b' with a confidence interval")
    low, high = dose_range
    if not (high > low > 0):
        raise ValueError(f"'dose_range' must be 0 < low < high, got {dose_range}")
    ratio = high / low
    bound_low = 1.0 + math.log(criterion[0]) / math.log(ratio)
    bound_high = 1.0 + math.log(criterion[1]) / math.log(ratio)
    b, lo, hi = result["b"], result["b_ci_low"], result["b_ci_high"]
    inside = (lo >= bound_low) & (hi <= bound_high)
    overlap = (hi >= bound_low) & (lo <= bound_high)
    finite = np.isfinite(lo) & np.isfinite(hi)
    ds = xr.Dataset(
        {
            "b": b,
            "b_ci_low": lo,
            "b_ci_high": hi,
            "bound_low": xr.full_like(b, bound_low),
            "bound_high": xr.full_like(b, bound_high),
            "proportional": inside & finite,
            "inconclusive": overlap & ~inside & finite,
        }
    )
    for name in ds.data_vars:
        ds[name].attrs["units"] = "dimensionless"
    ds.attrs.update({"dose_range": list(dose_range), "criterion": list(criterion)})
    return ds
