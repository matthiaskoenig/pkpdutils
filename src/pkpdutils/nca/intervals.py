r"""Parameters of the single dosing intervals of a multiple dose timecourse.

A dosing protocol with the dose times $t_1 < \dots < t_K$ splits a timecourse
into the dosing intervals $[t_k, t_{k+1}]$ and the last interval
$[t_K, t_K + \tau_K]$, whose length comes from the protocol or from
`NCAOptions.tau`. `compute_intervals` computes the exposure of every interval
of every row of a batch, the analysis of a multiple dose curve of
Gabrielsson & Weiner (2016, ch. 2.8) and Rowland & Tozer (2011, ch. 11):

- `AUC(0-tau)` of the interval, with the values at its bounds interpolated so
  that samples outside it add no area,
- `Cmax`, `Tmax` (relative to the start of the interval), `Cmin`,
- `Ctrough`, the value at the end of the interval, and `Cpre`, the value at its
  start,
- `Cavg = AUC(0-tau) / tau`, `fluctuation = (Cmax - Cmin) / Cavg` and
  `swing = (Cmax - Cmin) / Cmin`,

and, for effect timecourses, `AUEC(0-tau)`, `Emax`, `TEmax`, `Emin`, `Eavg`
and the time above `NCAOptions.effect_threshold`.

The variables are named with the prefix `interval_` and live over the extra
dimension `interval` of a result, so that they do not clash with the single
dose and the steady state parameters of the same curve.

**The bounds of an interval.** The value at the start and the value at the end
are interpolated from the curve (`pkpdutils.nca.auc.interpolate_at`), which
returns an observed value when a sample was taken at the bound. Two cases need
more than an interpolation:

- after an intravenous bolus the concentration jumps at the dose. An interval
  which starts before the first sample of the curve therefore gets the
  log-linearly back-extrapolated `C0` of its first two samples, the estimate
  `compute_parameters` uses for the single dose areas, and an interval whose
  end carries the next bolus ends *before* that dose: a sample recorded at the
  end which lies above the preceding sample is a post-dose sample of the next
  dose and belongs to the next interval, so the trough of this one is
  extrapolated log-linearly from its own last segment. A sample at the end
  which continues the decline is the observed trough and is used as it is.
- an interval whose end is not covered by the data at all is incomplete: its
  parameters are `NaN` and only the number of samples inside it is reported
  (`pkpdutils.nca.options.NCAFlag.INCOMPLETE_INTERVAL` for the last interval).
"""

import numpy as np

from pkpdutils.nca.auc import (
    auc_aumc,
    insert_point,
    interpolate_at,
    pack_valid,
    time_above_threshold,
)
from pkpdutils.nca.options import AUCMethod, C0Method, Kind, NCAOptions
from pkpdutils.timecourse import Route

#: name of the dimension the per-interval variables live over
INTERVAL_DIM = "interval"

#: prefix of the per-interval variables
INTERVAL_PREFIX = "interval_"

#: unit expression per interval variable, see `pkpdutils.nca.result.parameter_unit`
INTERVAL_UNITS: dict[str, str] = {
    "interval_start": "{time}",
    "interval_end": "{time}",
    "interval_dose": "{dose}",
    "interval_n_points": "dimensionless",
    "interval_auc": "({unit}) * ({time})",
    "interval_cmax": "{unit}",
    "interval_tmax": "{time}",
    "interval_cmin": "{unit}",
    "interval_ctrough": "{unit}",
    "interval_c_pre": "{unit}",
    "interval_cavg": "{unit}",
    "interval_fluctuation": "dimensionless",
    "interval_swing": "dimensionless",
    "interval_auec": "({unit}) * ({time})",
    "interval_emax": "{unit}",
    "interval_temax": "{time}",
    "interval_emin": "{unit}",
    "interval_eavg": "{unit}",
    "interval_time_above": "{time}",
}

#: interval variables of a concentration timecourse, after the shared ones
CONCENTRATION_VARIABLES: tuple[str, ...] = (
    "interval_auc",
    "interval_cmax",
    "interval_tmax",
    "interval_cmin",
    "interval_ctrough",
    "interval_c_pre",
    "interval_cavg",
    "interval_fluctuation",
    "interval_swing",
)

#: interval variables of an effect timecourse, after the shared ones
EFFECT_VARIABLES: tuple[str, ...] = (
    "interval_auec",
    "interval_emax",
    "interval_temax",
    "interval_emin",
    "interval_eavg",
    "interval_time_above",
)


def interval_variables(options: NCAOptions, *, has_dose: bool) -> tuple[str, ...]:
    """Names of the interval variables of an analysis, in the order of the result.

    Args:
        options: the options, `kind` selects the concentration or the effect
            variables

    Keyword Args:
        has_dose: whether the batch carries dose amounts (`interval_dose`)

    Returns:
        The variable names.
    """
    shared = ("interval_start", "interval_end")
    if has_dose:
        shared = (*shared, "interval_dose")
    shared = (*shared, "interval_n_points")
    kind = (
        CONCENTRATION_VARIABLES
        if options.kind is Kind.CONCENTRATION
        else EFFECT_VARIABLES
    )
    return (*shared, *kind)


def _take(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Element `idx[i]` of row `i`.

    Args:
        a: array `(N, n)`
        idx: one column index per row `(N,)`

    Returns:
        The selected elements `(N,)`.
    """
    return np.take_along_axis(a, idx[:, None], axis=1)[:, 0]


def _use_log(
    c1: np.ndarray, c2: np.ndarray, method: AUCMethod
) -> np.ndarray | np.bool_:
    """Whether a segment is treated logarithmically by a trapezoid rule.

    Args:
        c1: value at the start of the segment `(N,)`
        c2: value at the end of the segment `(N,)`
        method: the trapezoid rule

    Returns:
        A boolean array `(N,)`.
    """
    with np.errstate(invalid="ignore"):
        positive = (c1 > 0) & (c2 > 0) & (c1 != c2)
        if method is AUCMethod.LINEAR:
            return np.zeros_like(positive)
        if method is AUCMethod.LINEAR_LOG:
            return positive & (c2 < c1)
        return positive


def _back_extrapolate(
    tp: np.ndarray,
    cp: np.ndarray,
    n_valid: np.ndarray,
    t_start: np.ndarray,
    options: NCAOptions,
) -> np.ndarray:
    """Concentration at the start of an interval of an intravenous bolus.

    The value is the log-linear back-extrapolation of the first two samples
    after `t_start`, `C0 = exp(ln C1 - (ln C2 - ln C1) / (t2 - t1) (t1 - t_start))`,
    the estimate `pkpdutils.nca.nca.compute_parameters` uses at the dose of a
    single dose curve (Gabrielsson & Weiner 2016, ch. 2.8); with
    `C0Method.FIRST_VALUE`, or when the two samples do not decline, the first
    sample is used.

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        n_valid: valid points per row `(N,)`
        t_start: start of the interval per row `(N,)`
        options: the options, `c0_method` is used

    Returns:
        The estimate per row `(N,)`, `NaN` for a row without a sample after
        `t_start`.
    """
    n = tp.shape[1]
    valid = np.arange(n)[None, :] < n_valid[:, None]
    with np.errstate(invalid="ignore"):
        at_or_before = (tp <= t_start[:, None]) & valid
    first = at_or_before.sum(axis=1)
    n_after = n_valid - first
    i1 = np.clip(first, 0, n - 1)
    i2 = np.clip(first + 1, 0, n - 1)
    t1, c1 = _take(tp, i1), _take(cp, i1)
    t2, c2 = _take(tp, i2), _take(cp, i2)
    with np.errstate(divide="ignore", invalid="ignore"):
        back = np.exp(
            np.log(c1) - (np.log(c2) - np.log(c1)) / (t2 - t1) * (t1 - t_start)
        )
        usable = (n_after >= 2) & (c1 > 0) & (c2 > 0) & (c2 < c1) & (t2 > t1)
    first_value = np.where(n_after >= 1, c1, np.nan)
    if options.c0_method is C0Method.FIRST_VALUE:
        return first_value
    return np.where(usable, back, first_value)


def _extrapolate_end(
    t_prev: np.ndarray,
    c_prev: np.ndarray,
    t_last: np.ndarray,
    c_last: np.ndarray,
    t_end: np.ndarray,
    options: NCAOptions,
) -> np.ndarray:
    """Value at the end of an interval extrapolated from its last segment.

    The segment `(t_prev, c_prev)` to `(t_last, c_last)` is continued to
    `t_end`, logarithmically when the trapezoid rule treats it
    logarithmically (exact for a mono-exponential decline) and linearly
    otherwise.

    Args:
        t_prev: time of the second to last sample of the interval `(N,)`
        c_prev: value of the second to last sample `(N,)`
        t_last: time of the last sample of the interval `(N,)`
        c_last: value of the last sample `(N,)`
        t_end: end of the interval `(N,)`
        options: the options, `auc_method` and `kind` are used

    Returns:
        The extrapolated value `(N,)`, `NaN` where the segment is missing.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (t_end - t_last) / (t_last - t_prev)
        linear = c_last + frac * (c_last - c_prev)
        logarithmic = np.exp(np.log(c_last) + frac * (np.log(c_last) - np.log(c_prev)))
        value = np.where(
            _use_log(c_prev, c_last, options.auc_method), logarithmic, linear
        )
        value = np.where(t_last > t_prev, value, np.nan)
        if options.kind is Kind.CONCENTRATION:
            value = np.maximum(value, 0.0)
    return value


def _interval_column(
    tp: np.ndarray,
    cp: np.ndarray,
    n_valid: np.ndarray,
    *,
    t_start: np.ndarray,
    t_end: np.ndarray,
    ends_at_dose: np.ndarray,
    route: Route | None,
    options: NCAOptions,
) -> dict[str, np.ndarray]:
    """Parameters of one dosing interval of every row.

    Args:
        tp: packed times of the curves `(N, n)`
        cp: packed values of the curves `(N, n)`
        n_valid: valid points per row `(N,)`

    Keyword Args:
        t_start: start of the interval per row `(N,)`, `NaN` where the row has
            no such interval
        t_end: end of the interval per row `(N,)`
        ends_at_dose: whether the next dose is given at the end of the
            interval `(N,)` (every interval but the last of a protocol)
        route: route of the batch
        options: the options

    Returns:
        One `(N,)` array per interval variable, without the prefixed shared
        variables `interval_start`, `interval_end` and `interval_dose`.
    """
    n_rows, n = tp.shape
    nan = np.full(n_rows, np.nan)
    valid = np.arange(n)[None, :] < n_valid[:, None]
    with np.errstate(invalid="ignore"):
        inside = valid & (tp >= t_start[:, None]) & (tp < t_end[:, None])
        at_end = valid & (tp == t_end[:, None])
        before = (tp < t_start[:, None]) & valid
    n_inside = inside.sum(axis=1)
    # the samples of the interval are a block of the packed row
    first = np.clip(before.sum(axis=1), 0, n - 1)
    last_idx = np.clip(first + n_inside - 1, 0, n - 1)
    prev_idx = np.clip(last_idx - 1, 0, n - 1)
    c_last_inside = np.where(n_inside >= 1, _take(cp, last_idx), np.nan)

    # the value at the start: interpolated, or back-extrapolated for an
    # intravenous bolus whose interval starts before the first sample
    c_start = interpolate_at(tp, cp, n_valid, t_start, options.auc_method)
    if route is Route.IV_BOLUS:
        c_start = np.where(
            np.isnan(c_start),
            _back_extrapolate(tp, cp, n_valid, t_start, options),
            c_start,
        )

    # the value at the end: the observed or interpolated value, unless the
    # concentration jumps up at the next bolus, where the sample belongs to the
    # next interval and the trough is extrapolated from the last segment
    c_end = interpolate_at(tp, cp, n_valid, t_end, options.auc_method)
    with np.errstate(invalid="ignore"):
        jumped = (route is Route.IV_BOLUS) & ends_at_dose & (c_end > c_last_inside)
    if np.any(jumped):
        extrapolated = _extrapolate_end(
            _take(tp, prev_idx),
            _take(cp, prev_idx),
            _take(tp, last_idx),
            c_last_inside,
            t_end,
            options,
        )
        c_end = np.where(jumped, np.where(n_inside >= 2, extrapolated, np.nan), c_end)

    # the interval as its own curve: the samples inside plus both bounds
    tq, cq, nq = pack_valid(np.where(inside, tp, np.nan), np.where(inside, cp, np.nan))
    tq, cq, nq = insert_point(
        tq, cq, nq, np.where(np.isfinite(c_start), t_start, np.nan), c_start
    )
    tq, cq, nq = insert_point(
        tq, cq, nq, np.where(np.isfinite(c_end), t_end, np.nan), c_end
    )
    exists = np.isfinite(t_start) & np.isfinite(t_end)
    complete = exists & np.isfinite(c_end) & (nq >= 2)
    in_row = np.arange(tq.shape[1])[None, :] < nq[:, None]
    with np.errstate(invalid="ignore"):
        imax = np.where(in_row, cq, -np.inf).argmax(axis=1)
        value_max = np.where(complete, _take(cq, imax), nan)
        time_max = np.where(complete, _take(tq, imax) - t_start, nan)
        value_min = np.where(complete, np.where(in_row, cq, np.inf).min(axis=1), nan)
    method = (
        options.auc_method if options.kind is Kind.CONCENTRATION else AUCMethod.LINEAR
    )
    area, _ = auc_aumc(tq, cq, nq, method)
    area = np.where(complete, area, nan)
    # a sample at the end of the interval is used unless it was a post-dose
    # sample of the next dose
    n_points = np.where(
        exists, n_inside + np.where(at_end.any(axis=1) & ~jumped, 1, 0), nan
    ).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        average = area / (t_end - t_start)
        out: dict[str, np.ndarray] = {"interval_n_points": n_points}
        if options.kind is Kind.CONCENTRATION:
            out["interval_auc"] = area
            out["interval_cmax"] = value_max
            out["interval_tmax"] = time_max
            out["interval_cmin"] = value_min
            out["interval_ctrough"] = np.where(complete, c_end, nan)
            out["interval_c_pre"] = np.where(exists, c_start, nan)
            out["interval_cavg"] = average
            out["interval_fluctuation"] = (value_max - value_min) / average
            out["interval_swing"] = (value_max - value_min) / value_min
        else:
            out["interval_auec"] = area
            out["interval_emax"] = value_max
            out["interval_temax"] = time_max
            out["interval_emin"] = value_min
            out["interval_eavg"] = average
            out["interval_time_above"] = (
                np.where(
                    complete,
                    time_above_threshold(tq, cq, nq, options.effect_threshold),
                    nan,
                )
                if options.effect_threshold is not None
                else nan.copy()
            )
    return out


def compute_intervals(
    t: np.ndarray,
    c: np.ndarray,
    *,
    dose_amount: np.ndarray | None,
    dose_time: np.ndarray,
    tau: np.ndarray,
    route: Route | None,
    options: NCAOptions,
) -> dict[str, np.ndarray]:
    """Parameters of every dosing interval of every row of a batch.

    Interval `k` of a row runs from the dose time `t_k` to the next dose time
    `t_{k+1}`, the last one from `t_K` to `t_K + tau`. The rows are vectorized
    and the (few) intervals are looped over; a row with fewer doses than the
    widest protocol of the batch has `NaN` in its trailing columns.

    Args:
        t: times `(N, n)` in the frame of the curves, `NaN` for missing points
        c: values `(N, n)`, `NaN` for missing values

    Keyword Args:
        dose_amount: dose amounts `(N, K)`, `NaN` padded, `None` without
            amounts (`interval_dose` is then not reported)
        dose_time: dose times `(N, K)`, `NaN` padded, sorted per row
        tau: length of the last interval per row `(N,)`, `NaN` when it is
            unknown (the row then has no last interval)
        route: route of the batch
        options: the options

    Returns:
        One `(N, K)` array per interval variable (`interval_variables`).
    """
    t = np.asarray(t, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    times = np.asarray(dose_time, dtype=np.float64)
    amounts = None if dose_amount is None else np.asarray(dose_amount, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    n_rows, n_dose = times.shape

    given = np.isfinite(times)
    counts = given.sum(axis=1)
    is_last = np.arange(n_dose)[None, :] == (counts - 1)[:, None]
    following = np.concatenate([times[:, 1:], np.full((n_rows, 1), np.nan)], axis=1)
    with np.errstate(invalid="ignore"):
        ends = np.where(is_last, times + tau[:, None], following)
        usable = given & np.isfinite(ends) & (ends > times)
    starts = np.where(usable, times, np.nan)
    ends = np.where(usable, ends, np.nan)

    tp, cp, n_valid = pack_valid(t, c)
    names = interval_variables(options, has_dose=amounts is not None)
    out = {name: np.full((n_rows, n_dose), np.nan) for name in names}
    out["interval_start"] = starts
    out["interval_end"] = ends
    if amounts is not None:
        out["interval_dose"] = np.where(usable, amounts, np.nan)
    for k in range(n_dose):
        column = _interval_column(
            tp,
            cp,
            n_valid,
            t_start=starts[:, k],
            t_end=ends[:, k],
            ends_at_dose=~is_last[:, k],
            route=route,
            options=options,
        )
        for name, values in column.items():
            out[name][:, k] = values
    return out
