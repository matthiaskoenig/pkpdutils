r"""Time to steady state from the trough concentrations of the dosing intervals.

Repeated dosing approaches a plateau: the trough of every dosing interval rises
until the amount eliminated over an interval equals the amount given, and the
time at which that plateau is practically reached is what a study design and a
dose escalation decision need. ICH M13A asks for the evidence outright
("applicants should document appropriate dosage administration and sampling to
demonstrate the attainment of steady-state").

A multiple dose analysis already reports the trough of every dosing interval
(`interval_ctrough`, the value at the end of the interval, and `interval_cmin`,
its smallest value), so the estimate is a curve through those troughs. The
trough of an interval is the value at its **end**, so `interval_end` is the
time it was taken at and the time the troughs are read against;
`interval_start` is what the stepwise estimate reports, the start of the
interval from which the troughs no longer rise. `time_to_steady_state` offers
the two estimators of PKNCA (`pk.tss.monoexponential`,
`pk.tss.stepwise.linear`):

- **monoexponential**: the troughs approach the plateau as
  \(C_\mathrm{trough}(t) = C_\mathrm{ss}\left(1 - e^{-k t}\right)\), which is
  the accumulation of a one compartment drug, and the time to reach a fraction
  \(f\) of \(C_\mathrm{ss}\) is \(t_\mathrm{ss} = -\ln(1 - f) / k\). It is a
  smooth estimate which uses every interval and reports the plateau itself.
- **stepwise**: no model. The troughs from interval \(i\) on are regressed
  linearly against time and the slope is tested against 0; the first interval
  from which the trend is no longer significant at `alpha` is where the
  plateau starts, and its start is the estimate. It is the conservative
  estimate of a study report, since it asks only that the troughs stop rising.

Both are estimates of a design quantity and not of a parameter of the drug: a
study which stops before the plateau reports a time to steady state which is
its own last interval, and a study with two intervals reports nothing.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import least_squares
from scipy.stats import t as student_t

from pkpdutils.nca.intervals import INTERVAL_DIM
from pkpdutils.nca.result import NCAResult
from pkpdutils.nca.terminal import window_statistics

logger = logging.getLogger(__name__)

#: the trough variables of the intervals, in the order they are looked for
TROUGH_VARIABLES: tuple[str, ...] = ("interval_ctrough", "interval_cmin")

#: methods of `time_to_steady_state`
TSSMethod = Literal["monoexponential", "stepwise"]

#: starting values of `k` of the monoexponential fit, as multiples of
#: `1 / t_last`; the best of the three starts is kept
_STARTS: tuple[float, ...] = (0.5, 2.0, 8.0)


@dataclass(frozen=True)
class TSSResult:
    """Time to steady state of every sample of a multiple dose analysis.

    Attributes:
        tss: the time to steady state per sample, in the time unit of the
            analysis, over the sample dimensions of the result; `NaN` for a
            sample whose troughs do not carry the estimate
        method: the estimator, `"monoexponential"` or `"stepwise"`
        fraction: the fraction of the plateau the monoexponential estimate
            reports the time to; it does not apply to the stepwise estimate
        c_ss: the estimated plateau per sample of the monoexponential
            estimate, `None` for the stepwise one
    """

    tss: xr.DataArray
    method: TSSMethod
    fraction: float
    c_ss: xr.DataArray | None

    def to_dataframe(self) -> pd.DataFrame:
        """The estimate as one row per sample.

        Returns:
            The sample coordinates, `tss` and, for the monoexponential
            estimate, `c_ss`; the single row of a result of one curve
            (`nca_single`, no sample dimensions) with its scalar coordinates.
        """
        if self.tss.ndim == 0:
            # a scalar cannot be turned into a frame by xarray; the one row is
            # the scalar coordinates of the result plus the estimate
            row: dict[str, Any] = {
                str(name): coord.item() for name, coord in self.tss.coords.items()
            }
            row["tss"] = float(self.tss)
            if self.c_ss is not None:
                row["c_ss"] = float(self.c_ss)
            return pd.DataFrame([row])
        frame = self.tss.rename("tss").to_dataframe().reset_index()
        if self.c_ss is not None:
            frame["c_ss"] = self.c_ss.to_numpy().reshape(-1)
        return frame


def _monoexponential_row(
    time: np.ndarray, value: np.ndarray, fraction: float
) -> tuple[float, float]:
    r"""The plateau and the time to a fraction of it, for one sample.

    The model is \(C(t) = C_\mathrm{ss}\left(1 - e^{-k t}\right)\). For a given
    `k` the plateau is linear in the data,
    \(C_\mathrm{ss} = \sum_i y_i g_i / \sum_i g_i^2\) with
    \(g_i = 1 - e^{-k t_i}\), so only `k` is searched (over \(\log_{10} k\),
    which keeps it positive) and the plateau is profiled out at every step.

    Args:
        time: the times of the troughs, increasing and non-negative
        value: the troughs
        fraction: the fraction of the plateau

    Returns:
        The time to `fraction` of the plateau and the plateau; both `NaN` when
        the fit does not converge to a rising curve.
    """
    if time.size < 3 or not np.all(value > 0.0) or time[-1] <= 0.0:
        return float("nan"), float("nan")

    def plateau(k: float) -> float:
        """The least squares plateau for a rate constant."""
        g = 1.0 - np.exp(-k * time)
        denominator = float(np.sum(g * g))
        return float(np.sum(value * g) / denominator) if denominator > 0.0 else 0.0

    def residuals(x: np.ndarray) -> np.ndarray:
        """The residuals of the profiled model at `k = 10 ** x`."""
        k = float(10.0 ** x[0])
        return value - plateau(k) * (1.0 - np.exp(-k * time))

    best: Any = None
    for factor in _STARTS:
        start = np.array([np.log10(factor / time[-1])])
        try:
            fit = least_squares(residuals, start, method="lm")
        except ValueError:  # pragma: no cover - guarded by the checks above
            continue
        if best is None or fit.cost < best.cost:
            best = fit
    if best is None or not best.success:
        return float("nan"), float("nan")
    k = float(10.0 ** best.x[0])
    if not np.isfinite(k) or k <= 0.0:
        return float("nan"), float("nan")
    return float(-np.log1p(-fraction) / k), plateau(k)


def _stepwise(
    time: np.ndarray,
    value: np.ndarray,
    start: np.ndarray,
    valid: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """The first interval from which the troughs no longer trend, per sample.

    The troughs from position `i` on are regressed against time and the slope
    is tested against 0 with the two-sided t test of the regression (`n - 2`
    degrees of freedom); the first `i` whose p value is above `alpha` is the
    answer, the time to steady state being the start of that interval (PKNCA
    `pk.tss.stepwise.linear`). Every window of every sample is regressed at
    once by `pkpdutils.nca.terminal.window_statistics`, which is the same
    suffix-sum computation the terminal phase selection runs on.

    Args:
        time: the times the troughs were taken at `(N, K)`, increasing per row
            with the valid entries packed to the front
        value: the troughs `(N, K)`
        start: the start of the interval of every trough `(N, K)`
        valid: which entries are troughs `(N, K)`
        alpha: significance level of the trend test

    Returns:
        The start of the first interval without a significant trend per sample
        `(N,)`, `NaN` when every window of at least three troughs still trends.
    """
    stats = window_statistics(time, value, valid)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistic = np.abs(stats["slope"]) / stats["se_slope"]
        p_value = 2.0 * student_t.sf(t_statistic, stats["n"] - 2.0)
        # an exact straight line has no residual scatter: a slope of 0 is no
        # trend, any other slope is one, whatever the t statistic does there
        exact = stats["se_slope"] == 0.0
        no_trend = np.where(exact, stats["slope"] == 0.0, p_value > alpha)
    no_trend = no_trend & np.isfinite(stats["n"])
    first = no_trend.argmax(axis=1)
    return np.where(
        no_trend.any(axis=1),
        np.take_along_axis(start, first[:, None], axis=1)[:, 0],
        np.nan,
    )


def _troughs(
    result: NCAResult,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """The trough variable and the bounds of every interval of a result.

    The trough of an interval is the value at its **end**
    (`pkpdutils.nca.intervals`), so that is the time it was taken at and the
    time the curve through the troughs is read against; the start of the
    interval is what the stepwise estimate reports.

    Args:
        result: the result of a multiple dose analysis

    Returns:
        The troughs, the interval starts and the interval ends, all three over
        the sample dimensions and `interval`.

    Raises:
        ValueError: if the result carries no per-interval trough.
    """
    bounds = ("interval_start", "interval_end")
    for name in TROUGH_VARIABLES:
        if name in result.ds.data_vars:
            if any(bound not in result.ds.data_vars for bound in bounds):
                break
            return result.ds[name], result.ds[bounds[0]], result.ds[bounds[1]]
    raise ValueError(
        "the result carries no per-interval troughs; analyse a multiple dose "
        f"batch with NCAOptions(intervals=True), which reports {TROUGH_VARIABLES[0]}, "
        "interval_start and interval_end"
    )


def time_to_steady_state(
    result: NCAResult,
    *,
    method: TSSMethod = "monoexponential",
    fraction: float = 0.9,
    alpha: float = 0.05,
) -> TSSResult:
    r"""Time to steady state from the troughs of the dosing intervals.

    The troughs of every sample (`interval_ctrough`, `interval_cmin` when the
    analysis reports no trough) are read against the end of their interval
    (`interval_end`), the time the trough was taken at, and the plateau is
    estimated with one of the two methods of the module, which are those of
    PKNCA (`pk.tss`):

    - `"monoexponential"` fits
      \(C_\mathrm{trough}(t) = C_\mathrm{ss}\left(1 - e^{-k t}\right)\) by
      least squares and reports \(t_\mathrm{ss} = -\ln(1 - f) / k\), the time
      to the fraction \(f\) of the plateau, together with the plateau; \(t\)
      is measured from the start of the first dosing interval and the estimate
      is reported in the times of the analysis;
    - `"stepwise"` regresses the troughs from every interval on linearly and
      reports `interval_start` of the first interval from which the slope is
      no longer significant at `alpha`.

    Args:
        result: the result of a multiple dose analysis with per-interval
            parameters (`NCAOptions(intervals=True)`, the default)

    Keyword Args:
        method: the estimator
        fraction: the fraction of the plateau of the monoexponential estimate,
            0.9 by default (ninety percent of steady state)
        alpha: significance level of the trend test of the stepwise estimate

    Returns:
        The estimate per sample.

    Raises:
        ValueError: if the result carries no per-interval troughs, if
            `fraction` is not in `(0, 1)` or if `alpha` is not in `(0, 1)`.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"'fraction' is {fraction}, it must be in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"'alpha' is {alpha}, it must be in (0, 1)")
    troughs, starts, ends = _troughs(result)
    sample_dims: Sequence[str] = result.sample_dims

    def rows(da: xr.DataArray) -> np.ndarray:
        """One row per sample, one column per dosing interval."""
        array = da.transpose(*sample_dims, INTERVAL_DIM).to_numpy()
        return array.reshape(-1, array.shape[-1])

    flat_values = rows(troughs)
    flat_starts = rows(starts)
    flat_ends = rows(ends)
    shape = troughs.transpose(*sample_dims, INTERVAL_DIM).shape[:-1]
    # the intervals of a row which carry a trough, packed to the front of the
    # row in their (increasing) interval order, as `pack_valid` packs a curve
    finite = (
        np.isfinite(flat_starts) & np.isfinite(flat_ends) & np.isfinite(flat_values)
    )
    order = np.argsort(~finite, axis=1, kind="stable")
    start = np.take_along_axis(flat_starts, order, axis=1)
    end = np.take_along_axis(flat_ends, order, axis=1)
    value = np.take_along_axis(flat_values, order, axis=1)
    valid = np.take_along_axis(finite, order, axis=1)
    n_valid = finite.sum(axis=1)

    plateau = np.full(flat_values.shape[0], np.nan)
    if method == "stepwise":
        tss = _stepwise(end, value, start, valid, alpha)
    else:
        tss = np.full(flat_values.shape[0], np.nan)
        for row in range(flat_values.shape[0]):
            k = int(n_valid[row])
            if k == 0:
                continue
            # the model runs from the start of the dosing, and the trough of an
            # interval is the value at its end
            estimate, plateau[row] = _monoexponential_row(
                end[row, :k] - start[row, 0], value[row, :k], fraction
            )
            tss[row] = estimate + start[row, 0]
    n_missing = int(np.isnan(tss).sum())
    if n_missing:
        logger.info(
            "time to steady state: %d of %d samples carry no estimate",
            n_missing,
            tss.size,
        )
    coords = {
        str(name): coord
        for name, coord in result.ds.coords.items()
        if {str(d) for d in coord.dims} <= set(sample_dims)
    }
    unit = result.units(str(troughs.name))
    estimate = xr.DataArray(
        tss.reshape(shape),
        dims=tuple(sample_dims),
        coords=coords,
        attrs={"units": result.units("interval_start")},
        name="tss",
    )
    c_ss = (
        xr.DataArray(
            plateau.reshape(shape),
            dims=tuple(sample_dims),
            coords=coords,
            attrs={"units": unit},
            name="c_ss",
        )
        if method == "monoexponential"
        else None
    )
    return TSSResult(tss=estimate, method=method, fraction=fraction, c_ss=c_ss)
