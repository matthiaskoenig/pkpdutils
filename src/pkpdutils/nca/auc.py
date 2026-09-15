"""Vectorized trapezoid areas of timecourses.

Every function works on `(N, n)` arrays, one row per curve, without a loop
over the rows. Missing points (`NaN` in the time or the value) are moved to the
end of every row by `pack_valid`, so that the segments between consecutive
valid points are the columns of the arrays `segment_areas` returns.

The trapezoid rules are the ones of Gabrielsson & Weiner (2016, ch. 2.8) and
of the Phoenix WinNonlin NCA: on a segment from `(t1, c1)` to `(t2, c2)` with
`dt = t2 - t1` the linear rule gives the area `dt (c1 + c2) / 2` and the first
moment `dt (t1 c1 + t2 c2) / 2`; the logarithmic rule, exact for a
mono-exponential decline, gives the area `dt (c1 - c2) / L` and the moment
`dt (t1 c1 - t2 c2) / L + dt² (c1 - c2) / L²` with `L = ln(c1 / c2)`.
"""

import numpy as np

from pkpdutils.nca.options import AUCMethod


def take_rows(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Element `idx[i]` of row `i`.

    Args:
        a: array `(N, n)`
        idx: one column index per row `(N,)`

    Returns:
        The selected elements `(N,)`.
    """
    return np.take_along_axis(a, idx[:, None], axis=1)[:, 0]


def pack_valid(
    t: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move the valid points of every row to the front, keeping their order.

    Args:
        t: times of shape `(N, n)`
        c: values of shape `(N, n)`

    Returns:
        The packed times, the packed values (both padded with `NaN`) and the
        number of valid points per row.
    """
    t = np.asarray(t, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(c)
    order = np.argsort(~valid, axis=1, kind="stable")
    tp = np.where(valid, t, np.nan)
    cp = np.where(valid, c, np.nan)
    tp = np.take_along_axis(tp, order, axis=1)
    cp = np.take_along_axis(cp, order, axis=1)
    return tp, cp, valid.sum(axis=1)


def segment_areas(
    tp: np.ndarray, cp: np.ndarray, n_valid: np.ndarray, method: AUCMethod
) -> tuple[np.ndarray, np.ndarray]:
    """Areas and first moments of the segments between consecutive packed points.

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        n_valid: valid points per row `(N,)`
        method: trapezoid rule

    Returns:
        The areas and the first moments, both of shape `(N, n - 1)`; a segment
        beyond the valid points of its row is 0.
    """
    t1, t2 = tp[:, :-1], tp[:, 1:]
    c1, c2 = cp[:, :-1], cp[:, 1:]
    dt = t2 - t1
    in_curve = np.arange(tp.shape[1] - 1)[None, :] < (n_valid - 1)[:, None]

    lin_area = 0.5 * (c1 + c2) * dt
    lin_moment = 0.5 * (t1 * c1 + t2 * c2) * dt
    positive = (c1 > 0) & (c2 > 0) & (c1 != c2)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log(c1 / c2)
        log_area = (c1 - c2) / log_ratio * dt
        log_moment = dt * (t1 * c1 - t2 * c2) / log_ratio + dt * dt * (c1 - c2) / (
            log_ratio * log_ratio
        )

    if method is AUCMethod.LINEAR:
        use_log = np.zeros_like(positive)
    elif method is AUCMethod.LINEAR_LOG:
        use_log = positive & (c2 < c1)
    else:
        use_log = positive
    area = np.where(use_log, log_area, lin_area)
    moment = np.where(use_log, log_moment, lin_moment)
    area = np.where(in_curve, area, 0.0)
    moment = np.where(in_curve, moment, 0.0)
    return area, moment


def auc_aumc(
    tp: np.ndarray,
    cp: np.ndarray,
    n_valid: np.ndarray,
    method: AUCMethod,
    t_start: np.ndarray | None = None,
    t_end: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Area and first moment of every row, optionally only over a time window.

    A segment counts as a whole or not at all, so the window covers the intended
    interval exactly when a point of the packed arrays lies on each of its
    bounds; `interpolate_at` and `insert_point` add such a point.

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        n_valid: valid points per row
        method: trapezoid rule
        t_start: per row, only segments starting at or after this time count
        t_end: per row, only segments ending at or before this time count

    Returns:
        `auc` and `aumc` of shape `(N,)`.
    """
    area, moment = segment_areas(tp, cp, n_valid, method)
    keep = np.ones(area.shape, dtype=bool)
    if t_start is not None:
        keep &= tp[:, :-1] >= t_start[:, None]
    if t_end is not None:
        keep &= tp[:, 1:] <= t_end[:, None]
    area = np.where(keep, area, 0.0)
    moment = np.where(keep, moment, 0.0)
    return area.sum(axis=1), moment.sum(axis=1)


def time_above_threshold(
    tp: np.ndarray, cp: np.ndarray, n_valid: np.ndarray, threshold: float
) -> np.ndarray:
    """Total time the linearly interpolated curve is above a threshold, per row.

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        n_valid: valid points per row
        threshold: the threshold

    Returns:
        The total time above the threshold `(N,)`.
    """
    t1, t2 = tp[:, :-1], tp[:, 1:]
    c1, c2 = cp[:, :-1], cp[:, 1:]
    in_curve = np.arange(tp.shape[1] - 1)[None, :] < (n_valid - 1)[:, None]
    dt = t2 - t1
    with np.errstate(invalid="ignore"):
        a1 = c1 > threshold
        a2 = c2 > threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (threshold - c1) / (c2 - c1)  # position of the crossing in the segment
    both = np.where(a1 & a2, dt, 0.0)
    rising = np.where(~a1 & a2, dt * (1.0 - frac), 0.0)
    falling = np.where(a1 & ~a2, dt * frac, 0.0)
    total = np.where(in_curve, both + rising + falling, 0.0)
    return np.nansum(total, axis=1)


def interpolate_at(
    tp: np.ndarray,
    cp: np.ndarray,
    n_valid: np.ndarray,
    t_query: np.ndarray,
    method: AUCMethod,
) -> np.ndarray:
    """Value of every row at a query time by interpolation between the bracketing points.

    Linear interpolation, or logarithmic interpolation on a segment the
    trapezoid `method` treats logarithmically. `NaN` outside the observed
    times of a row.

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        n_valid: valid points per row
        t_query: one query time per row `(N,)`
        method: trapezoid rule

    Returns:
        The interpolated values `(N,)`.
    """
    n = tp.shape[1]
    idx = np.arange(n)[None, :]
    valid = idx < n_valid[:, None]
    tq = t_query[:, None]
    # number of valid points at or before the query time
    count = ((tp <= tq) & valid).sum(axis=1)
    out = np.full(tp.shape[0], np.nan)
    with np.errstate(invalid="ignore"):
        # exactly on the last point
        last_idx = np.clip(n_valid - 1, 0, n - 1)
        t_last = take_rows(tp, last_idx)
        c_last = take_rows(cp, last_idx)
        on_last = (n_valid > 0) & (t_query == t_last)
        out = np.where(on_last, c_last, out)
        inside = (count >= 1) & (count < n_valid)
        i2 = np.clip(count, 1, n - 1)
        i1 = i2 - 1
        t1 = take_rows(tp, i1)
        t2 = take_rows(tp, i2)
        c1 = take_rows(cp, i1)
        c2 = take_rows(cp, i2)
        frac = (t_query - t1) / (t2 - t1)
        lin = c1 + frac * (c2 - c1)
        positive = (c1 > 0) & (c2 > 0) & (c1 != c2)
        if method is AUCMethod.LINEAR:
            use_log = np.zeros_like(positive)
        elif method is AUCMethod.LINEAR_LOG:
            use_log = positive & (c2 < c1)
        else:
            use_log = positive
        with np.errstate(divide="ignore"):
            log = np.exp(np.log(c1) + frac * (np.log(c2) - np.log(c1)))
        value = np.where(use_log, log, lin)
        return np.where(inside & ~on_last, value, out)


def insert_point(
    tp: np.ndarray,
    cp: np.ndarray,
    n_valid: np.ndarray,
    t_new: np.ndarray,
    c_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert one point per row and repack in time order.

    A row whose new time or value is `NaN` is left unchanged (its arrays are
    still one column wider).

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        n_valid: valid points per row
        t_new: time of the new point per row
        c_new: value of the new point per row

    Returns:
        The packed times and values `(N, n + 1)` and the new counts.
    """
    t_all = np.concatenate([tp, t_new[:, None]], axis=1)
    c_all = np.concatenate([cp, c_new[:, None]], axis=1)
    order = np.argsort(np.where(np.isnan(t_all), np.inf, t_all), axis=1, kind="stable")
    t_sorted = np.take_along_axis(t_all, order, axis=1)
    c_sorted = np.take_along_axis(c_all, order, axis=1)
    return pack_valid(t_sorted, c_sorted)
