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

from dataclasses import dataclass

import numpy as np

from pkpdutils.nca.auc import take_rows
from pkpdutils.nca.options import NCAFlag, TerminalMethod, TerminalPhase


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
        start: packed index of the first point of the window
        flags: `NCAFlag` bits `POSITIVE_SLOPE` and `TOO_FEW_POINTS`
    """

    slope: np.ndarray
    intercept: np.ndarray
    r2: np.ndarray
    r2_adj: np.ndarray
    se_slope: np.ndarray
    n_points: np.ndarray
    t_first: np.ndarray
    start: np.ndarray
    flags: np.ndarray


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
    x0 = np.where(valid, x, 0.0)
    y0 = np.where(valid, y, 0.0)
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
        intercept = (sy - slope * sx) / n
        ss_res = syy_c - slope * sxy_c
        r2 = 1.0 - ss_res / syy_c
        r2_adj = 1.0 - (1.0 - r2) * (n - 1.0) / (n - 2.0)
        se_slope = np.sqrt(np.maximum(ss_res, 0.0) / (n - 2.0) / sxx_c)
    enough = n >= 3
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
) -> TerminalFit:
    """Terminal log-linear regression of every row.

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
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
    in_row = idx < n_valid[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(in_row & (cp > 0), np.log(cp), np.nan)
    regressable = in_row & np.isfinite(y)
    flags = np.zeros(n_rows, dtype=np.int64)

    if phase.method is TerminalMethod.MANUAL:
        if manual_mask is None:
            raise ValueError("TerminalMethod.MANUAL needs 'manual_mask'")
        selected = regressable & manual_mask
        # a single window per row: the selected points, statistics via the
        # suffix sums with everything before the first selected point excluded
        stats = window_statistics(tp, np.where(selected, y, 0.0), selected)
        start = np.where(selected.any(axis=1), selected.argmax(axis=1), 0)
        return _collect(tp, stats, start, selected.any(axis=1), phase, flags, selected)

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
    times are read: `t_first` always names a point of the regression.
    """
    # snap the start of every row forward to the first regressable point
    at_or_after = regressable & (np.arange(tp.shape[1])[None, :] >= start[:, None])
    start = np.where(at_or_after.any(axis=1), at_or_after.argmax(axis=1), start)

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
        start=np.where(has_fit, start, -1),
        flags=flags,
    )
