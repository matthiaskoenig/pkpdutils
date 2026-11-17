"""Vectorized terminal phase regression.

The elimination rate constant `lambda_z` is the negative slope of the linear
regression of `ln c` on `t` over the points of the terminal phase
(Gabrielsson & Weiner 2016, ch. 2.8; Phoenix WinNonlin NCA). Which points form
the terminal phase is decided by `TerminalPhase.method`; `BEST_FIT` evaluates
every window of consecutive points that ends at the last measurable point and
takes the largest adjusted R², preferring more points within a tolerance,
which is the rule of Phoenix.

All windows of all rows are evaluated at once: with suffix sums of `x`, `y`,
`x²`, `xy` and `y²` over the packed points the statistics of the window
starting at index `s` are closed-form expressions of the sums from `s` to the
end, so `window_statistics` returns `(N, n)` arrays without a loop over rows
or windows.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from pkpdutils.nca.auc import take_rows
from pkpdutils.nca.options import NCAFlag, TerminalMethod, TerminalPhase

#: dimension the candidate windows of the terminal regression live on
CANDIDATE_DIM: str = "candidate"

#: prefix of the variables which describe the candidate windows of a result
CANDIDATE_PREFIX: str = "candidate_"


@dataclass(frozen=True)
class TerminalFit:
    """Result of the terminal regression per row, `NaN` (and `start = -1`) without a fit.

    Attributes:
        slope: slope of `ln c` against `t` (`-lambda_z`)
        intercept: intercept of the regression, `ln c` at `t = 0`
        r2: coefficient of determination
        r2_adj: adjusted coefficient of determination
        se_slope: standard error of the slope
        n_points: number of points of the regression
        t_first: time of the first point of the regression
        t_last: time of the last point of the regression
        start: packed index of the first point of the window
        flags: `NCAFlag` bits `POSITIVE_SLOPE`, `TOO_FEW_POINTS` and
            `TERMINAL_LOW_CONCENTRATION`
        candidates: every candidate window of every row with the columns `row`
            (the index of the row), `start_time`, `n`, `r2_adj` and `slope`,
            `None` unless `TerminalPhase.keep_candidates` asked for it
            (`candidate_table`)
    """

    slope: np.ndarray
    intercept: np.ndarray
    r2: np.ndarray
    r2_adj: np.ndarray
    se_slope: np.ndarray
    n_points: np.ndarray
    t_first: np.ndarray
    t_last: np.ndarray
    start: np.ndarray
    flags: np.ndarray
    candidates: pd.DataFrame | None = None


RESIDUAL_FLOOR: float = 16.0 * float(np.finfo(np.float64).eps)
"""Multiple of `n * eps * syy` below which a residual sum of squares is zero.

The residual of a window is `syy_c - slope * sxy_c`, a difference of two
quantities which are equal for a perfect fit, so what is left there is the
rounding of the sums themselves and its sign is arbitrary: clamping it at zero
reports either 0 or the square root of that noise, and which one depends on the
platform and on whether the python core or the rust kernel computed it.

The floor is taken against the **uncentred** `syy` and not against `syy_c`,
because every window of a row shares one origin, the row's first sample, so
`window_statistics` can evaluate them all from one set of suffix sums. For a
short window late in the curve that makes `syy` and `sy * sy / n` both large,
of the order of the whole drop of the curve, while their difference `syy_c` is
small - the cancellation, and therefore the noise, is set by the uncentred sums
and not by the centred one. A floor written against `syy_c` leaves the three
and four point windows of a noise free curve reporting a standard error of up
to 4e-08, which is what measurement showed.

The multiple is 16 rather than tighter or looser from measurement over 120 000
windows of 20 000 curves. The ratio `|ss_res| / (n * eps * syy)` reaches 1.20
for a noise free curve, at three points and falling with the window, while a
curve with 1 % noise stays above 1.9e+05 from four points up. The two regimes
therefore overlap only at three points, where a window has one degree of
freedom and three noisy samples can fall nearly collinear by chance: at this
floor that zeroes 2 windows in 120 000 of a noisy batch, whose residual really
is at the rounding floor, and leaves no exact fit reporting a standard error.
A multiple of 4 would do the same with a margin of only 3x over the worst
exact fit measured, so 16 buys robustness to another time grid for the same
cost.
"""


def _suffix_sum(a: np.ndarray) -> np.ndarray:
    """Sum of every row from each index to the end."""
    return np.cumsum(a[:, ::-1], axis=1)[:, ::-1]


def window_statistics(
    x: np.ndarray, y: np.ndarray, valid: np.ndarray
) -> dict[str, np.ndarray]:
    """Regression statistics of every window from an index to the end of the row.

    Args:
        x: regressor `(N, n)`
        y: response `(N, n)`
        valid: which points enter the regression `(N, n)`

    Returns:
        Arrays `(N, n)` keyed `n`, `slope`, `intercept`, `r2`, `r2_adj`,
        `se_slope`; column `s` describes the window `s..end`. Windows with
        fewer than 3 points are `NaN`.
    """
    # Translate before forming raw moments to avoid cancellation for large
    # time offsets and almost constant logarithmic concentrations.
    if x.shape[1]:
        first = np.argmax(valid, axis=1)
        x_origin = np.take_along_axis(x, first[:, None], axis=1)
        y_origin = np.take_along_axis(y, first[:, None], axis=1)
    else:
        x_origin = y_origin = np.zeros((x.shape[0], 1))
    x0 = np.where(valid, x - x_origin, 0.0)
    y0 = np.where(valid, y - y_origin, 0.0)
    n = _suffix_sum(valid.astype(np.float64))
    sx = _suffix_sum(x0)
    sy = _suffix_sum(y0)
    sxx = _suffix_sum(x0 * x0)
    sxy = _suffix_sum(x0 * y0)
    syy = _suffix_sum(y0 * y0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sxx_c = sxx - sx * sx / n
        sxy_c = sxy - sx * sy / n
        syy_c = syy - sy * sy / n
        slope = sxy_c / sxx_c
        intercept = (sy - slope * sx) / n + y_origin - slope * x_origin
        # the residual sum of squares subtracts two nearly equal sums, so an
        # exact fit leaves rounding noise whose sign is arbitrary: clamping it
        # at zero would report either 0 or `sqrt` of that noise, which differs
        # between platforms and between this core and the rust kernel. Below
        # the rounding of its own operands the residual is not a small number
        # but no number, and the fit is exact.
        ss_res = syy_c - slope * sxy_c
        ss_res = np.where(ss_res < RESIDUAL_FLOOR * n * syy, 0.0, ss_res)
        r2 = np.clip(1.0 - ss_res / syy_c, 0.0, 1.0)
        r2_adj = 1.0 - (1.0 - r2) * (n - 1.0) / (n - 2.0)
        se_slope = np.sqrt(ss_res / (n - 2.0) / sxx_c)
    enough = (n >= 3) & (sxx_c > 0) & (syy_c > 0)
    nan = np.nan
    return {
        "n": np.where(enough, n, nan),
        "slope": np.where(enough, slope, nan),
        "intercept": np.where(enough, intercept, nan),
        "r2": np.where(enough, r2, nan),
        "r2_adj": np.where(enough, r2_adj, nan),
        "se_slope": np.where(enough, se_slope, nan),
    }


def terminal_fit(
    tp: np.ndarray,
    cp: np.ndarray,
    n_valid: np.ndarray,
    tmax_idx: np.ndarray,
    phase: TerminalPhase,
    manual_mask: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
    windows: np.ndarray | None = None,
) -> TerminalFit:
    """Terminal log-linear regression of every row.

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        n_valid: valid points per row
        tmax_idx: packed index of the maximum per row
        phase: the selection rule and its parameters
        manual_mask: packed points of the regression for `TerminalMethod.MANUAL`
        exclude: packed points which may not enter the regression `(N, n)`,
            the values below the limit of quantification a BLQ rule kept or
            imputed (`pkpdutils.nca.options.BLQRules`)
        windows: the terminal window of single rows `(N, 2)`, `NaN` for a row
            without one (`TerminalPhase.windows`). A row with a window
            regresses the points whose time lies in `[t_first, t_last]`, every
            other row follows `phase.method`.

    Returns:
        The fit per row, with the table of every candidate window
        (`candidate_table`) when `phase.keep_candidates` is set.

    Raises:
        ValueError: `phase.method` is `TerminalMethod.MANUAL` and `manual_mask` is `None`.
    """
    n_rows, n = tp.shape
    idx = np.arange(n)[None, :]
    in_row = idx < n_valid[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(in_row & (cp > 0), np.log(cp), np.nan)
    regressable = in_row & np.isfinite(y)
    if exclude is not None:
        regressable = regressable & ~exclude
    flags = np.zeros(n_rows, dtype=np.int64)

    given = (
        np.zeros(n_rows, dtype=bool)
        if windows is None
        else np.isfinite(windows).all(axis=1)
    )
    if not given.any():
        fit = _fit_by_method(tp, y, regressable, n_valid, tmax_idx, phase, manual_mask)
    else:
        assert windows is not None
        with np.errstate(invalid="ignore"):
            inside = regressable & (tp >= windows[:, :1]) & (tp <= windows[:, 1:2])
        windowed = _fit_selected(tp, y, inside, phase, flags)
        fit = (
            # every row carries a window, the rule of the batch decides nothing
            windowed
            if given.all()
            else _merge_fits(
                _fit_by_method(
                    tp, y, regressable, n_valid, tmax_idx, phase, manual_mask
                ),
                windowed,
                given,
            )
        )
    if phase.relative_warning is not None:
        # Only actual regression points count, including sparse manual masks
        # and per-row windows. The diagnostic never changes point selection.
        selected = (
            regressable & (tp >= fit.t_first[:, None]) & (tp <= fit.t_last[:, None])
        )
        if phase.method is TerminalMethod.MANUAL and manual_mask is not None:
            selected &= given[:, None] | manual_mask
        peak = np.max(np.where(in_row & np.isfinite(cp), cp, 0.0), axis=1)
        low = np.any(selected & (cp < phase.relative_warning * peak[:, None]), axis=1)
        fit.flags[low] |= int(NCAFlag.TERMINAL_LOW_CONCENTRATION)
    if not phase.keep_candidates:
        return fit
    return replace(fit, candidates=candidate_table(tp, y, regressable, tmax_idx, phase))


def candidate_table(
    tp: np.ndarray,
    y: np.ndarray,
    regressable: np.ndarray,
    tmax_idx: np.ndarray,
    phase: TerminalPhase,
) -> pd.DataFrame:
    """Every window the selection of the terminal phase may choose from.

    A candidate is a window which starts at a point of the regression, holds at
    least `phase.min_points` regressable points and, with
    `phase.exclude_cmax`, starts after the maximum: the windows `BEST_FIT`
    ranks by the adjusted R², and the same set for the other rules, which pick
    one of them by a different criterion. A window whose first point cannot be
    regressed is left out, since its statistics are those of the window
    starting at the next regressable point (`_collect`). The slope is reported
    as it is, so a window of a still rising curve is in the table with a
    positive slope, which `BEST_FIT` never chooses.

    Args:
        tp: packed times `(N, n)`
        y: the logarithms of the values `(N, n)`, `NaN` where there is none
        regressable: the points which may enter a regression `(N, n)`
        tmax_idx: packed index of the maximum per row
        phase: the selection rule and its parameters

    Returns:
        One row per candidate window with the columns `row` (the index of the
        row of the batch), `start_time` (the time of the first point of the
        window), `n` (points of the window), `r2_adj` and `slope`; the rows
        are ordered by row and by the start time within a row.
    """
    stats = window_statistics(tp, y, regressable)
    first_allowed = (
        tmax_idx + 1 if phase.exclude_cmax else np.zeros_like(tmax_idx)
    ).astype(np.int64)
    index = np.arange(tp.shape[1])[None, :]
    with np.errstate(invalid="ignore"):
        candidate = (
            regressable
            & (index >= first_allowed[:, None])
            & (stats["n"] >= phase.min_points)
            & np.isfinite(stats["r2_adj"])
        )
    rows, columns = np.nonzero(candidate)
    return pd.DataFrame(
        {
            "row": rows.astype(np.int64),
            "start_time": tp[rows, columns],
            "n": stats["n"][rows, columns].astype(np.int64),
            "r2_adj": stats["r2_adj"][rows, columns],
            "slope": stats["slope"][rows, columns],
        }
    )


def _merge_fits(base: TerminalFit, other: TerminalFit, use: np.ndarray) -> TerminalFit:
    """Take the fit of `other` in the rows of `use` and the fit of `base` elsewhere.

    Args:
        base: the fit of the batch rule (`TerminalPhase.method`)
        other: the fit of the rows with a window of their own
        use: the rows which take the fit of `other` `(N,)`

    Returns:
        The combined fit.
    """
    return TerminalFit(
        slope=np.where(use, other.slope, base.slope),
        intercept=np.where(use, other.intercept, base.intercept),
        r2=np.where(use, other.r2, base.r2),
        r2_adj=np.where(use, other.r2_adj, base.r2_adj),
        se_slope=np.where(use, other.se_slope, base.se_slope),
        n_points=np.where(use, other.n_points, base.n_points),
        t_first=np.where(use, other.t_first, base.t_first),
        t_last=np.where(use, other.t_last, base.t_last),
        start=np.where(use, other.start, base.start).astype(np.int64),
        flags=np.where(use, other.flags, base.flags).astype(np.int64),
    )


def _fit_selected(
    tp: np.ndarray,
    y: np.ndarray,
    selected: np.ndarray,
    phase: TerminalPhase,
    flags: np.ndarray,
) -> TerminalFit:
    """Regress exactly the selected points of every row.

    A single window per row: the selected points, with the statistics of the
    suffix sums and everything before the first selected point excluded.

    Args:
        tp: packed times `(N, n)`
        y: the logarithms of the values `(N, n)`
        selected: the points of the regression `(N, n)`
        phase: the selection rule, for `min_points` and `min_adj_r2`
        flags: the flags of every row so far `(N,)`

    Returns:
        The fit per row.
    """
    stats = window_statistics(tp, np.where(selected, y, 0.0), selected)
    start = np.where(selected.any(axis=1), selected.argmax(axis=1), 0)
    return _collect(tp, stats, start, selected.any(axis=1), phase, flags, selected)


def _fit_by_method(
    tp: np.ndarray,
    y: np.ndarray,
    regressable: np.ndarray,
    n_valid: np.ndarray,
    tmax_idx: np.ndarray,
    phase: TerminalPhase,
    manual_mask: np.ndarray | None,
) -> TerminalFit:
    """The regression of every row under the rule of `phase.method`.

    Args:
        tp: packed times `(N, n)`
        y: the logarithms of the values `(N, n)`, `NaN` where there is none
        regressable: the points which may enter a regression `(N, n)`
        n_valid: valid points per row
        tmax_idx: packed index of the maximum per row
        phase: the selection rule and its parameters
        manual_mask: packed points of the regression for `TerminalMethod.MANUAL`

    Returns:
        The fit per row.

    Raises:
        ValueError: `phase.method` is `TerminalMethod.MANUAL` and `manual_mask` is `None`.
    """
    n_rows, n = tp.shape
    idx = np.arange(n)[None, :]
    flags = np.zeros(n_rows, dtype=np.int64)

    if phase.method is TerminalMethod.MANUAL:
        if manual_mask is None:
            raise ValueError("TerminalMethod.MANUAL needs 'manual_mask'")
        return _fit_selected(tp, y, regressable & manual_mask, phase, flags)

    # with `exclude_cmax` a window may only start after the point of the maximum,
    # otherwise it may start anywhere in the row, also before the maximum
    first_allowed = tmax_idx + 1 if phase.exclude_cmax else np.zeros_like(tmax_idx)
    stats = window_statistics(tp, y, regressable)
    if phase.method is TerminalMethod.BEST_FIT:
        count_ok = (idx >= first_allowed[:, None]) & (stats["n"] >= phase.min_points)
        # a window is a fit candidate once it has enough points and a declining
        # slope; `min_adj_r2` only gates the acceptance of the selected window in
        # `_collect`, not the selection itself
        candidate = count_ok & (stats["slope"] < 0)
        r2_adj = np.where(candidate, stats["r2_adj"], -np.inf)
        best = r2_adj.max(axis=1)
        within = candidate & (r2_adj >= best[:, None] - phase.tie_tolerance)
        start = np.where(within.any(axis=1), within.argmax(axis=1), 0)
        has_candidate = candidate.any(axis=1)
        # windows with enough points exist but none with a negative slope
        flags = np.where(
            ~has_candidate & count_ok.any(axis=1), NCAFlag.POSITIVE_SLOPE, 0
        ).astype(np.int64)
        return _collect(tp, stats, start, has_candidate, phase, flags, regressable)

    if phase.method is TerminalMethod.LAST_N:
        assert phase.n_points is not None
        # the window whose regressable count equals n_points: the largest s with n >= n_points
        enough_n = stats["n"] >= phase.n_points
        start = np.where(
            enough_n.any(axis=1), n - 1 - enough_n[:, ::-1].argmax(axis=1), 0
        )
        # with `exclude_cmax` the window may not reach into the absorption phase:
        # it then holds the points after the maximum, fewer than `n_points`
        start = np.clip(np.maximum(start, first_allowed), 0, n - 1)
        has_fit = enough_n.any(axis=1)
    else:  # ALL_AFTER_TMAX
        start = tmax_idx + 1
        has_fit = start < n_valid
        start = np.clip(start, 0, n - 1)
    n_at_start = take_rows(stats["n"], start)
    slope_at_start = take_rows(stats["slope"], start)
    with np.errstate(invalid="ignore"):
        enough = has_fit & (n_at_start >= phase.min_points)
        positive = enough & ~(slope_at_start < 0)
    flags = np.where(positive, NCAFlag.POSITIVE_SLOPE, 0).astype(np.int64)
    return _collect(tp, stats, start, enough & ~positive, phase, flags, regressable)


def _collect(
    tp: np.ndarray,
    stats: dict[str, np.ndarray],
    start: np.ndarray,
    has_fit: np.ndarray,
    phase: TerminalPhase,
    flags: np.ndarray,
    regressable: np.ndarray,
) -> TerminalFit:
    """Pick the statistics of the chosen window per row and set the flags.

    The statistics of a window starting at a point that cannot be regressed
    (`NaN`, zero or negative) are those of the window starting at the next
    regressable point, so the start is snapped forward to that point before the
    times are read: `t_first` always names a point of the regression, and
    `t_last` the last regressable point at or after it, the point every window
    ends at.
    """
    # snap the start of every row forward to the first regressable point
    n_columns = tp.shape[1]
    at_or_after = regressable & (np.arange(n_columns)[None, :] >= start[:, None])
    start = np.where(at_or_after.any(axis=1), at_or_after.argmax(axis=1), start)
    end = np.where(
        at_or_after.any(axis=1),
        n_columns - 1 - at_or_after[:, ::-1].argmax(axis=1),
        start,
    )

    def take(a: np.ndarray) -> np.ndarray:
        """Value of `a` at the chosen `start` index of every row."""
        return take_rows(a, start)

    n_points = take(stats["n"])
    with np.errstate(invalid="ignore"):
        has_fit = has_fit & (n_points >= phase.min_points)
        if phase.min_adj_r2 is not None:
            has_fit &= take(stats["r2_adj"]) >= phase.min_adj_r2
    nan = np.nan

    def pick(a: np.ndarray) -> np.ndarray:
        """Value of `a` at `start`, `NaN` for rows without a fit."""
        return np.where(has_fit, take(a), nan)

    too_few = ~has_fit & (flags == 0)
    flags = flags | np.where(too_few, NCAFlag.TOO_FEW_POINTS, 0).astype(np.int64)
    return TerminalFit(
        slope=pick(stats["slope"]),
        intercept=pick(stats["intercept"]),
        r2=pick(stats["r2"]),
        r2_adj=pick(stats["r2_adj"]),
        se_slope=pick(stats["se_slope"]),
        n_points=pick(stats["n"]),
        t_first=pick(tp),
        t_last=np.where(has_fit, take_rows(tp, end), nan),
        start=np.where(has_fit, start, -1),
        flags=flags,
    )
