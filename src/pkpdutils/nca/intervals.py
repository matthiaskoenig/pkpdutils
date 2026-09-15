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
- `Ctrough`, the value at the end of the interval, and `Cstart`, the value at
  its start (after an intravenous bolus the post-dose value; the pre-dose value
  of interval `k` is the `Ctrough` of interval `k-1`),
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
  `compute_parameters` uses for the single dose areas.
- an interval whose end carries the next bolus ends *before* that dose, so a
  sample recorded exactly at its end may be a post-dose sample of the next
  dose. It is taken as such only when it lies above the last sample inside the
  interval, which no decline can do; the trough is then the log-linear
  regression of the last (up to three) positive samples inside the interval
  evaluated at the end of the interval, and the row carries
  `pkpdutils.nca.options.NCAFlag.EXTRAPOLATED_TROUGH`. Every other sample at
  the end, and every interpolated end value, is the observed trough and is
  used as it is; the substitution never applies to another route, since only a
  bolus makes the concentration jump.
- an interval whose end is not covered by the data, or which holds no sample at
  all, is incomplete: its parameters are `NaN` and only the number of samples
  is reported (`pkpdutils.nca.options.NCAFlag.INCOMPLETE_INTERVAL` for the last
  interval).

`interval_n_points` counts the samples the interval uses; a sample at a
boundary is used by both neighbouring intervals, so the counts of the
intervals of a curve do not partition its samples.
"""

import numpy as np

from pkpdutils.nca.auc import (
    auc_aumc,
    interpolate_at,
    pack_valid,
    take_rows,
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
    "interval_c_start": "{unit}",
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
    "interval_c_start",
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
    t1, c1 = take_rows(tp, i1), take_rows(cp, i1)
    t2, c2 = take_rows(tp, i2), take_rows(cp, i2)
    with np.errstate(divide="ignore", invalid="ignore"):
        back = np.exp(
            np.log(c1) - (np.log(c2) - np.log(c1)) / (t2 - t1) * (t1 - t_start)
        )
        usable = (n_after >= 2) & (c1 > 0) & (c2 > 0) & (c2 < c1) & (t2 > t1)
    first_value = np.where(n_after >= 1, c1, np.nan)
    if options.c0_method is C0Method.FIRST_VALUE:
        return first_value
    return np.where(usable, back, first_value)


#: samples of the log-linear regression of an extrapolated trough
TROUGH_POINTS = 3


def _interval_block(
    tp: np.ndarray, cp: np.ndarray, first: np.ndarray, n_inside: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The samples of one interval gathered out of the packed rows.

    `pack_valid` keeps the time order, so the samples of an interval are the
    contiguous block `first ... first + n_inside - 1` of the packed row. The
    block is gathered by index, which is `max(n_inside)` columns wide instead
    of the full width of the row and needs no sort, where masking and packing
    the full row again would cost an `argsort` over every sample of the curve
    for every interval.

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        first: index of the first sample of the interval per row `(N,)`
        n_inside: number of samples of the interval per row `(N,)`

    Returns:
        The times and the values of the block `(N, w)` with `w = max(n_inside)`
        (`NaN` outside the block) and the mask of its entries.
    """
    n = tp.shape[1]
    width = int(n_inside.max()) if n_inside.size else 0
    columns = np.arange(width)
    index = np.minimum(first[:, None] + columns[None, :], max(n - 1, 0))
    in_block = columns[None, :] < n_inside[:, None]
    t_block = np.where(in_block, np.take_along_axis(tp, index, axis=1), np.nan)
    c_block = np.where(in_block, np.take_along_axis(cp, index, axis=1), np.nan)
    return t_block, c_block, in_block


def _with_bounds(
    t_block: np.ndarray,
    c_block: np.ndarray,
    n_inside: np.ndarray,
    *,
    t_start: np.ndarray,
    c_start: np.ndarray,
    t_end: np.ndarray,
    c_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The curve of one interval: the block of its samples with both bounds inserted.

    The bounds need no sort either: every sample of the interval lies at or
    after its start and strictly before its end, so the start goes in front of
    the block and the end behind it. A bound whose time or value is not finite
    is not inserted, as in `pkpdutils.nca.auc.insert_point`.

    Args:
        t_block: times of the samples of the interval `(N, w)`
        c_block: values of the samples of the interval `(N, w)`
        n_inside: number of samples of the interval per row `(N,)`

    Keyword Args:
        t_start: start of the interval per row `(N,)`
        c_start: value at the start per row `(N,)`
        t_end: end of the interval per row `(N,)`
        c_end: value at the end per row `(N,)`

    Returns:
        The packed times and values `(N, w + 2)` and the number of points per
        row.
    """
    n_rows, width = t_block.shape
    has_start = np.isfinite(t_start) & np.isfinite(c_start)
    has_end = np.isfinite(t_end) & np.isfinite(c_end)
    if width:
        # a sample recorded exactly at the start of the interval keeps its
        # place in front of the inserted bound (the order a stable insertion
        # gives); both carry the same time, so only their values change places
        with np.errstate(invalid="ignore"):
            tie = has_start & (n_inside >= 1) & (t_block[:, 0] == t_start)
        if tie.any():
            c_block = c_block.copy()
            observed = c_block[:, 0].copy()
            c_block[:, 0] = np.where(tie, c_start, observed)
            c_start = np.where(tie, observed, c_start)

    offset = has_start.astype(np.int64)
    index = offset[:, None] + np.arange(width)[None, :]
    tq = np.full((n_rows, width + 2), np.nan)
    cq = np.full((n_rows, width + 2), np.nan)
    np.put_along_axis(tq, index, t_block, axis=1)
    np.put_along_axis(cq, index, c_block, axis=1)
    tq[:, 0] = np.where(has_start, t_start, tq[:, 0])
    cq[:, 0] = np.where(has_start, c_start, cq[:, 0])
    end = (offset + n_inside)[:, None]
    np.put_along_axis(tq, end, np.where(has_end, t_end, np.nan)[:, None], axis=1)
    np.put_along_axis(cq, end, np.where(has_end, c_end, np.nan)[:, None], axis=1)
    return tq, cq, n_inside + offset + has_end.astype(np.int64)


def _extrapolate_end(
    tp: np.ndarray,
    cp: np.ndarray,
    inside: np.ndarray,
    t_end: np.ndarray,
    *,
    max_points: int = TROUGH_POINTS,
) -> np.ndarray:
    r"""Value at the end of an interval from the log-linear regression of its last samples.

    The least squares line through $(t_i, \ln C_i)$ of the last `max_points`
    positive samples of the interval is evaluated at `t_end`, the estimate of a
    trough which was not observed (Gabrielsson & Weiner 2016, ch. 2.8). The
    regression is exact for a mono-exponential decline and damps the noise of a
    single sample, which a two point extrapolation would carry over in full.

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        inside: mask of the samples of the interval `(N, n)`
        t_end: end of the interval `(N,)`

    Keyword Args:
        max_points: number of samples of the regression at most

    Returns:
        The extrapolated value `(N,)`, `NaN` for a row with fewer than two
        positive samples in the interval.
    """
    with np.errstate(invalid="ignore"):
        positive = inside & (cp > 0)
    # the last `max_points` positive samples of every row
    counts = positive.sum(axis=1)
    from_end = counts[:, None] - np.cumsum(positive, axis=1)
    selected = positive & (from_end < max_points)
    n = selected.sum(axis=1).astype(np.float64)
    t = np.where(selected, tp, 0.0)
    y = np.where(selected, np.log(np.where(selected, cp, 1.0)), 0.0)
    sx, sy = t.sum(axis=1), y.sum(axis=1)
    sxx, sxy = (t * t).sum(axis=1), (t * y).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = n * sxx - sx * sx
        slope = (n * sxy - sx * sy) / denominator
        intercept = (sy - slope * sx) / n
        value = np.exp(intercept + slope * t_end)
    return np.where((n >= 2) & (denominator > 0), value, np.nan)


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
) -> tuple[dict[str, np.ndarray], np.ndarray]:
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
        One `(N,)` array per interval variable, without the shared variables
        `interval_start`, `interval_end` and `interval_dose`, and the mask of
        the rows whose trough was extrapolated (`NCAFlag.EXTRAPOLATED_TROUGH`).
    """
    n_rows, n = tp.shape
    nan = np.full(n_rows, np.nan)
    valid = np.arange(n)[None, :] < n_valid[:, None]
    tolerance = 1e-9 * np.maximum(1.0, np.abs(t_end))
    with np.errstate(invalid="ignore"):
        inside = valid & (tp >= t_start[:, None]) & (tp < t_end[:, None])
        at_end = valid & (np.abs(tp - t_end[:, None]) <= tolerance[:, None])
        before = (tp < t_start[:, None]) & valid
    n_inside = inside.sum(axis=1)
    # the samples of the interval are a block of the packed row
    first = np.clip(before.sum(axis=1), 0, n - 1)
    t_block, c_block, in_block = _interval_block(tp, cp, first, n_inside)
    last_idx = np.clip(first + n_inside - 1, 0, n - 1)
    c_last_inside = np.where(n_inside >= 1, take_rows(cp, last_idx), np.nan)
    has_sample_at_end = at_end.any(axis=1)
    with np.errstate(invalid="ignore"):
        c_sample_at_end = np.where(
            has_sample_at_end, np.where(at_end, cp, -np.inf).max(axis=1), np.nan
        )

    # the value at the start: interpolated, or back-extrapolated for an
    # intravenous bolus whose interval starts before the first sample
    c_start = interpolate_at(tp, cp, n_valid, t_start, options.auc_method)
    if route is Route.IV_BOLUS:
        c_start = np.where(
            np.isnan(c_start),
            _back_extrapolate(tp, cp, n_valid, t_start, options),
            c_start,
        )

    # the value at the end: the observed or interpolated value, unless a sample
    # recorded at the next bolus lies above the last sample of the interval and
    # is therefore a post-dose sample of the next dose; the trough is then the
    # log-linear regression of the last samples of the interval
    c_end = interpolate_at(tp, cp, n_valid, t_end, options.auc_method)
    with np.errstate(invalid="ignore"):
        post_dose = (
            (route is Route.IV_BOLUS)
            & ends_at_dose
            & has_sample_at_end
            & (c_sample_at_end > c_last_inside)
        )
    extrapolated = np.zeros(n_rows, dtype=bool)
    if np.any(post_dose):
        trough = _extrapolate_end(t_block, c_block, in_block, t_end)
        c_end = np.where(post_dose, trough, c_end)
        extrapolated = post_dose & np.isfinite(trough)

    # the interval as its own curve: the samples inside plus both bounds
    tq, cq, nq = _with_bounds(
        t_block,
        c_block,
        n_inside,
        t_start=t_start,
        c_start=c_start,
        t_end=t_end,
        c_end=c_end,
    )
    exists = np.isfinite(t_start) & np.isfinite(t_end)
    # an interval without an observation of its own is not analysed: a sample at
    # its end belongs to the next interval, except in the last interval
    covering = n_inside + np.where(has_sample_at_end & ~ends_at_dose, 1, 0)
    complete = exists & np.isfinite(c_end) & (nq >= 2) & (covering >= 1)
    in_row = np.arange(tq.shape[1])[None, :] < nq[:, None]
    with np.errstate(invalid="ignore"):
        imax = np.where(in_row, cq, -np.inf).argmax(axis=1)
        value_max = np.where(complete, take_rows(cq, imax), nan)
        time_max = np.where(complete, take_rows(tq, imax) - t_start, nan)
        value_min = np.where(complete, np.where(in_row, cq, np.inf).min(axis=1), nan)
    method = (
        options.auc_method if options.kind is Kind.CONCENTRATION else AUCMethod.LINEAR
    )
    area, _ = auc_aumc(tq, cq, nq, method)
    area = np.where(complete, area, nan)
    # the samples the interval uses: those inside plus the one at its end,
    # unless that one was a post-dose sample of the next dose. A sample at a
    # boundary is used by both neighbouring intervals
    n_points = np.where(
        exists, n_inside + np.where(has_sample_at_end & ~post_dose, 1, 0), nan
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
            out["interval_c_start"] = np.where(exists, c_start, nan)
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
    return out, extrapolated


def compute_intervals(
    t: np.ndarray,
    c: np.ndarray,
    *,
    dose_amount: np.ndarray | None,
    dose_time: np.ndarray,
    tau: np.ndarray,
    route: Route | None,
    options: NCAOptions,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
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
        One `(N, K)` array per interval variable (`interval_variables`) and the
        mask `(N,)` of the rows in which the trough of at least one interval
        was extrapolated (`NCAFlag.EXTRAPOLATED_TROUGH`).
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
    extrapolated = np.zeros(n_rows, dtype=bool)
    for k in range(n_dose):
        column, extrapolated_column = _interval_column(
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
        extrapolated |= extrapolated_column
    return out, extrapolated
