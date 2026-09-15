"""Dose proportionality by the power model and the confidence interval criterion.

With `AUC = a D^b` the exposure is dose proportional when `b = 1`. Smith et
al. (2000) accept proportionality over a dose range `r = D_high / D_low` when
the confidence interval of `b` lies within
`[1 + ln(theta_L) / ln(r), 1 + ln(theta_H) / ln(r)]` with the acceptance limits
`(theta_L, theta_H) = (0.8, 1.25)` of the dose-normalized exposure ratio.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from pkpdutils.fit.result import FitResult


def _values(da: xr.DataArray) -> Any:
    """A data array as a plain python value, a scalar for a 0-D array.

    Args:
        da: the array.

    Returns:
        The value of a 0-D array, else the nested list of the values.
    """
    return da.item() if da.ndim == 0 else da.to_numpy().tolist()


@dataclass(frozen=True)
class ProportionalityResult:
    """Verdict of the confidence interval criterion of dose proportionality.

    The three variables of the fit (`slope`, `ci_low`, `ci_high`) and the two
    verdicts (`proportional`, `inconclusive`) are `xarray.DataArray` objects
    over the sample dimensions of the fit, 0-D for a fit of one dose
    escalation; `bounds`, `dose_range` and `criterion` describe the criterion
    itself and are the same for every sample. Use `sel` to pick one sample.

    Attributes:
        slope: the exponent `b` of the power model
        ci_low: lower bound of the confidence interval of `b`
        ci_high: upper bound of the confidence interval of `b`
        bounds: the acceptance bounds of `b`, `(bound_low, bound_high)`
        proportional: whether the interval of `b` lies inside `bounds`
        inconclusive: whether the interval overlaps `bounds` without lying inside
        dose_range: lowest and highest dose the criterion refers to
        criterion: acceptance limits of the dose-normalized exposure ratio
    """

    slope: xr.DataArray
    ci_low: xr.DataArray
    ci_high: xr.DataArray
    bounds: tuple[float, float]
    proportional: xr.DataArray
    inconclusive: xr.DataArray
    dose_range: tuple[float, float]
    criterion: tuple[float, float]

    def sel(self, **indexers: Any) -> "ProportionalityResult":
        """The verdict of one sample, selected by coordinate label.

        Args:
            **indexers: coordinate label per sample dimension.

        Returns:
            The result of the selected sample, with 0-D variables.
        """
        if not indexers:
            return self
        return ProportionalityResult(
            slope=self.slope.sel(indexers),
            ci_low=self.ci_low.sel(indexers),
            ci_high=self.ci_high.sel(indexers),
            bounds=self.bounds,
            proportional=self.proportional.sel(indexers),
            inconclusive=self.inconclusive.sel(indexers),
            dose_range=self.dose_range,
            criterion=self.criterion,
        )

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dictionary of plain python values.

        Returns:
            Field name to value; the variables of a 0-D result are floats and
            booleans, those of a batch result nested lists.
        """
        return {
            "slope": _values(self.slope),
            "ci_low": _values(self.ci_low),
            "ci_high": _values(self.ci_high),
            "bounds": list(self.bounds),
            "proportional": _values(self.proportional),
            "inconclusive": _values(self.inconclusive),
            "dose_range": list(self.dose_range),
            "criterion": list(self.criterion),
        }


def proportionality_test(
    result: FitResult,
    *,
    dose_range: tuple[float, float],
    criterion: tuple[float, float] = (0.8, 1.25),
) -> ProportionalityResult:
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
        The verdict over the sample dimensions of the fit; the variables carry
        no unit, the exponent and the verdicts are dimensionless by
        construction.

    Raises:
        ValueError: if the result has no exponent `b` with a confidence
            interval, or `dose_range` or `criterion` is not `0 < low < high`.
    """
    if "b" not in result or "b_ci_low" not in result or "b_ci_high" not in result:
        raise ValueError("The result has no exponent 'b' with a confidence interval")
    low, high = dose_range
    if not (high > low > 0):
        raise ValueError(f"'dose_range' must be 0 < low < high, got {dose_range}")
    if not (criterion[1] > criterion[0] > 0):
        raise ValueError(f"'criterion' must be 0 < low < high, got {criterion}")
    ratio = high / low
    bound_low = 1.0 + math.log(criterion[0]) / math.log(ratio)
    bound_high = 1.0 + math.log(criterion[1]) / math.log(ratio)
    b, lo, hi = result["b"], result["b_ci_low"], result["b_ci_high"]
    inside = (lo >= bound_low) & (hi <= bound_high)
    overlap = (hi >= bound_low) & (lo <= bound_high)
    finite = np.isfinite(lo) & np.isfinite(hi)
    return ProportionalityResult(
        slope=b,
        ci_low=lo,
        ci_high=hi,
        bounds=(bound_low, bound_high),
        proportional=(inside & finite).rename("proportional"),
        inconclusive=(overlap & ~inside & finite).rename("inconclusive"),
        dose_range=(float(low), float(high)),
        criterion=(float(criterion[0]), float(criterion[1])),
    )
