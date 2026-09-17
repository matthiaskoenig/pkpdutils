"""The fitting engine.

`fit` fits a `Model` to one or many `(x, y)` rows with
`scipy.optimize.least_squares` (trust region reflective, bounds) in a scaled
parameter space, derives the standard errors from the Jacobian, transforms the
t-based intervals back to the linear scale, and computes the goodness-of-fit
statistics (Seber & Wild 1989, ch. 2; Gabrielsson & Weiner 2016, ch. 6):

- weighted residuals `r = (y - f) / sqrt(var)` with the variance model of `Weighting`
- `cov(q) = s² (JᵀJ)⁻¹`, `s² = Σ r² / (n - k)`, in the scaled space `q`
- `se(p) = se(q) |dp/dq|`, interval `q ± t_{n-k} se(q)` transformed back
- derived parameters by the delta method with a central difference gradient
- `R²`, `RMSE` on the unweighted residuals; `AIC`, `AICc`, `BIC` on the weighted
  ones, with `K = k + 1` estimated parameters (the residual variance is one of
  them, Burnham & Anderson 2002, sec. 2.2, 6.9.6): `AIC = n ln(Σr²/n) + 2K`,
  `AICc = AIC + 2K(K+1)/(n-K-1)`, `BIC = n ln(Σr²/n) + K ln n`; the reported
  `n_parameters` stays `k`, the free model parameters

When `options.bootstrap > 0`, `fit_row` also runs a residual bootstrap (Efron
& Tibshirani 1993, ch. 9): the weighted residuals of the fit are centered and
inflated by `sqrt(n / (n - k))` and resampled with replacement `B` times,
each replicate is refitted from the fitted `p`, and the standard errors, the
confidence intervals and the correlation matrix are the empirical statistics
of the replicate parameters, an alternative to the Jacobian-based ones above
that does not rely on the local linear approximation; fewer than two
converged replicates fall back to the Jacobian-based statistics and set
`FitFlag.BOOTSTRAP_FALLBACK`. `fit_rows` distributes the rows over the shared
process pool (`pkpdutils.parallel`) for a batch of more than
`FIT_WORKER_THRESHOLD` rows or an explicit `options.n_workers > 1`, with one
child seed per row drawn up front so serial and pooled runs agree;
`pkpdutils.fit.compare` ranks several models on the same data by the corrected
Akaike information criterion (Burnham & Anderson 2002).

A row whose numbers leave the range of double precision (unweighted values
beyond about `1e154`, whose squares overflow) is guarded rather than computed:
the initial guess, the start box, the trust region search, the covariance, the
statistics and the bootstrap run under `numpy.errstate`, so no floating-point
warning escapes and aborts a batch under a strict warning filter, and a start,
a covariance or a statistic which is not finite is `NaN` and flagged
`FitFlag.OVERFLOW`. The other rows of a batch are not affected.
"""

import functools
import logging
import math
import warnings
from collections.abc import Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from scipy.optimize import least_squares
from scipy.stats import qmc
from scipy.stats import t as student_t

from pkpdutils.fit.model import Model, parameter_unit_expression
from pkpdutils.fit.options import FitFlag, FitOptions, ParameterScale, Weighting
from pkpdutils.fit.result import FitResult
from pkpdutils.parallel import evict, executor, resolve_workers
from pkpdutils.result import base_name, check_coordinate_collision, nan_percentile

logger = logging.getLogger(__name__)

LN10 = math.log(10.0)

#: rows from which `fit_rows` uses the process pool with `n_workers=None`.
#: The pool is shared and started once per process, but that start is an
#: import of `pkpdutils` and its dependencies in every worker, about a second,
#: which a batch has to be big enough to earn back on its own: the measured
#: break-even against the serial run is around 1 500 rows of the cheapest model
#: (a mono-exponential row of about 1 ms), so the automatic default stays
#: serial below 2 000 rows and a script whose rows are more expensive - several
#: starts, a residual bootstrap, a sum of exponentials - asks for the pool with
#: an explicit `FitOptions.n_workers`.
FIT_WORKER_THRESHOLD = 2_000


@dataclass
class RowFit:
    """The fit of one row; arrays are in model parameter order (fixed parameters included).

    Attributes:
        p: the fitted parameters on the linear scale
        q: the fitted parameters on the search scale
        se_p: standard error per parameter (`NaN` for a fixed one); the
            residual bootstrap replicate standard deviation when
            `options.bootstrap > 0` produced at least 2 converged
            replicates, else the Jacobian-based one (`FitFlag.
            BOOTSTRAP_FALLBACK` is set in that case)
        ci_low: lower end of the confidence interval per parameter (a
            bootstrap percentile under the same condition as `se_p`)
        ci_high: upper end of the confidence interval per parameter (a
            bootstrap percentile under the same condition as `se_p`)
        cov_q: covariance of the parameters on the search scale; always the
            Jacobian-based covariance, never replaced by the bootstrap (the
            bootstrap does not produce a covariance in the search scale, only
            replicate statistics of the linear-scale parameters)
        correlation: correlation matrix of the parameters; from the
            bootstrap replicates under the same condition as `se_p`, else
            from `cov_q`
        derived: the derived parameters of the model
        derived_se: standard error per derived parameter (delta method, or
            the bootstrap replicate standard deviation, under the same
            condition as `se_p`)
        derived_ci_low: lower end of the interval per derived parameter
        derived_ci_high: upper end of the interval per derived parameter
        cost: the value of the scipy cost function `0.5 Σ ρ(r²)`
        r2: coefficient of determination of the unweighted residuals
        rmse: root mean squared error of the unweighted residuals
        aic: Akaike information criterion, `K = k + 1` estimated parameters
        aicc: Akaike information criterion with the small sample correction,
            `NaN` when `n - K - 1 <= 0`
        bic: Bayesian information criterion, `K = k + 1` estimated parameters
        n_points: number of points used in the fit
        n_starts_converged: number of start points which converged
        y_pred: the prediction per point of the row (`NaN` for unused points)
        residuals: the weighted residual per point (`NaN` for unused points)
        flags: the `FitFlag` combination of the row
        nfev: number of function evaluations over all starts
        n_bootstrap: number of successful residual bootstrap replicates, 0 without bootstrap
    """

    p: np.ndarray
    q: np.ndarray
    se_p: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    cov_q: np.ndarray
    correlation: np.ndarray
    derived: dict[str, float]
    derived_se: dict[str, float]
    derived_ci_low: dict[str, float]
    derived_ci_high: dict[str, float]
    cost: float
    r2: float
    rmse: float
    aic: float
    aicc: float
    bic: float
    n_points: int
    n_starts_converged: int
    y_pred: np.ndarray
    residuals: np.ndarray
    flags: int
    nfev: int
    n_bootstrap: int = 0


def residual_sd(
    y: np.ndarray, sd: np.ndarray | None, weighting: Weighting
) -> np.ndarray:
    """Standard deviation `sqrt(var)` of every point under the weighting (`y <= 0` uses the smallest positive `|y|`).

    The weighted residual is `(y - f) / sqrt(var)` with `var = 1` (`NONE`),
    `|y|` (`INV_Y`), `y²` (`INV_Y2`) or `sd²` (`INV_SD`). The square root is
    taken analytically, `|y|` rather than `sqrt(y²)`: the square of a value
    beyond `1e154` overflows and the square of one below `1e-154` underflows
    to zero, which would give the point no weight at all or divide by zero.

    Args:
        y: the dependent variable of the row.
        sd: standard deviation per point, needed for `Weighting.INV_SD`.
        weighting: the variance model.

    Returns:
        The standard deviation per point.

    Raises:
        ValueError: for `Weighting.INV_SD` without `sd`.
    """
    if weighting is Weighting.NONE:
        return np.ones_like(y)
    if weighting is Weighting.INV_SD:
        if sd is None:
            raise ValueError("Weighting.INV_SD needs 'sd'")
        return np.abs(np.asarray(sd, dtype=np.float64))
    positive = np.abs(y[np.isfinite(y) & (y != 0)])
    floor = positive.min() if positive.size else 1.0
    base = np.where(np.isfinite(y) & (np.abs(y) > 0), np.abs(y), floor)
    return np.sqrt(base) if weighting is Weighting.INV_Y else base


def to_scale(p: np.ndarray, scales: Sequence[ParameterScale]) -> np.ndarray:
    """Linear parameters to the scaled space.

    A positive parameter which underflowed to exactly zero (or which a fixed
    value or a bound put at zero or below) has no logarithm; the logarithm is
    evaluated under `numpy.errstate(divide="ignore", invalid="ignore")` and
    becomes `-inf` or `NaN` silently rather than raising a `RuntimeWarning`
    under a strict warning filter, the counterpart of the overflow guard of
    `from_scale`. The callers treat a non-finite search-scale parameter as an
    uncertainty that cannot be computed.

    Args:
        p: the parameters on the linear scale.
        scales: the scale per parameter.

    Returns:
        The parameters on the search scale.
    """
    q = np.array(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, scale in enumerate(scales):
            if scale is ParameterScale.LOG10:
                q[i] = np.log10(p[i])
            elif scale is ParameterScale.LOG:
                q[i] = np.log(p[i])
    return q


def from_scale(q: np.ndarray, scales: Sequence[ParameterScale]) -> np.ndarray:
    """Scaled parameters back to the linear space.

    A wildly out-of-range search point (a bad start, or a step the optimizer
    proposes before it is rejected) can make `10 ** q` or `exp(q)` overflow;
    that overflow is evaluated under `numpy.errstate(over="ignore")` and
    becomes `inf` silently rather than raising a `RuntimeWarning`, which
    would otherwise escape as an exception under a strict warning filter and
    is not one of the exceptions the caller expects from a failed start. The
    resulting non-finite parameter makes the residuals non-finite too, which
    `scipy.optimize.least_squares` already turns into a `ValueError` the
    caller catches, so the start fails cleanly instead of the process
    crashing.

    Args:
        q: the parameters on the search scale.
        scales: the scale per parameter.

    Returns:
        The parameters on the linear scale.
    """
    p = np.array(q, dtype=np.float64)
    with np.errstate(over="ignore"):
        for i, scale in enumerate(scales):
            if scale is ParameterScale.LOG10:
                p[i] = 10.0 ** q[i]
            elif scale is ParameterScale.LOG:
                p[i] = np.exp(q[i])
    return p


def scale_derivative(p: np.ndarray, scales: Sequence[ParameterScale]) -> np.ndarray:
    """`dp/dq` per parameter.

    Args:
        p: the parameters on the linear scale.
        scales: the scale per parameter.

    Returns:
        The derivative of the linear parameter with respect to the scaled one.
    """
    d = np.ones_like(p, dtype=np.float64)
    # a parameter within a factor of `ln 10` of the largest double overflows
    # to `inf`, the derivative it has
    with np.errstate(over="ignore"):
        for i, scale in enumerate(scales):
            if scale is ParameterScale.LOG10:
                d[i] = p[i] * LN10
            elif scale is ParameterScale.LOG:
                d[i] = p[i]
    return d


def bounds_in_scale(
    lower: np.ndarray, upper: np.ndarray, scales: Sequence[ParameterScale]
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds in the scaled space (a non-positive lower bound of a log parameter becomes -inf).

    Args:
        lower: lower bounds on the linear scale.
        upper: upper bounds on the linear scale.
        scales: the scale per parameter.

    Returns:
        The `(lower, upper)` bounds on the search scale.
    """
    lq, uq = np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, scale in enumerate(scales):
            if scale is ParameterScale.LINEAR:
                continue
            log = np.log10 if scale is ParameterScale.LOG10 else np.log
            lq[i] = log(lower[i]) if lower[i] > 0 else -np.inf
            uq[i] = log(upper[i]) if np.isfinite(upper[i]) else np.inf
    return lq, uq


def covariance(
    jac: np.ndarray, cost: float, n: int, k: int
) -> tuple[np.ndarray, FitFlag]:
    """`s² (JᵀJ)⁻¹` with `s² = 2 cost / (n - k)`.

    This is the least-squares covariance and it is exact only for
    `loss="linear"`; for a robust loss (`soft_l1`, `huber`, `cauchy`,
    `arctan`) scipy returns the Jacobian and the cost of the transformed
    problem, so the covariance, the standard errors and the intervals derived
    from it are approximations.

    A numerically singular `JᵀJ` is reported as singular, and so is an
    inverse with a negative variance on the diagonal (the covariance is then
    not positive semidefinite, the standard errors are meaningless); the
    values are returned unchanged and the caller clips the variances at zero
    so that the standard errors stay finite. A Jacobian, a cost, `JᵀJ` or a
    covariance which is not finite (a row beyond the range of double
    precision) is an overflow; the products run under `numpy.errstate`.

    Args:
        jac: the Jacobian of the residuals in the scaled space, `(n, k)`.
        cost: the scipy cost `0.5 Σ r²`.
        n: number of points.
        k: number of free parameters.

    Returns:
        The covariance of the free parameters and its condition:
        `FitFlag.NONE`, `FitFlag.SINGULAR` (`NaN`, or the values with a
        negative variance) or `FitFlag.OVERFLOW` (`NaN`).
    """
    unusable = np.full((k, k), np.nan)
    if n <= k:
        return unusable, FitFlag.SINGULAR
    if not (math.isfinite(cost) and np.all(np.isfinite(jac))):
        return unusable, FitFlag.OVERFLOW
    with np.errstate(over="ignore", invalid="ignore"):
        jtj = jac.T @ jac
    if not np.all(np.isfinite(jtj)):
        return unusable, FitFlag.OVERFLOW
    try:
        inv = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        return unusable, FitFlag.SINGULAR
    if not np.all(np.isfinite(inv)):
        return unusable, FitFlag.SINGULAR
    with np.errstate(over="ignore", invalid="ignore"):
        cov = inv * (2.0 * cost / (n - k))
    if not np.all(np.isfinite(cov)):
        return unusable, FitFlag.OVERFLOW
    if np.any(np.diag(cov) < 0.0):
        return cov, FitFlag.SINGULAR
    return cov, FitFlag.NONE


def _problem(
    model: Model, options: FitOptions
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ParameterScale], np.ndarray]:
    """Free-parameter mask, bounds, scales and fixed values of a model under the options.

    Args:
        model: the model.
        options: the options.

    Returns:
        `(free, lower, upper, scales, fixed_values)` in the order of the parameters.

    Raises:
        ValueError: if an option names a parameter the model does not have.
    """
    names = model.parameter_names
    unknown = set(options.fixed) | set(options.bounds) | set(options.initial)
    unknown -= set(names)
    if unknown:
        raise ValueError(f"{model.name} has no parameters {sorted(unknown)}")
    free = np.array([name not in options.fixed for name in names])
    lower, upper = model.bounds()
    for name, (lo, hi) in options.bounds.items():
        i = names.index(name)
        lower[i], upper[i] = lo, hi
    scales = [options.scale_of(parameter) for parameter in model.parameters]
    fixed_values = np.array([options.fixed.get(name, np.nan) for name in names])
    return free, lower, upper, scales, fixed_values


def _start_vector(
    model: Model,
    x: np.ndarray,
    y: np.ndarray,
    options: FitOptions,
    lower: np.ndarray,
    upper: np.ndarray,
    scales: Sequence[ParameterScale],
) -> np.ndarray:
    """Initial guess inside the bounds, overridden by `options.initial`.

    Args:
        model: the model.
        x: the independent variable of the finite points.
        y: the dependent variable of the finite points.
        options: the options.
        lower: lower bounds on the linear scale.
        upper: upper bounds on the linear scale.
        scales: the scale per parameter.

    Returns:
        A start vector strictly inside the bounds.
    """
    # the guess is a heuristic on the raw data (regressions, crossings, the
    # exponential of an intercept) which can leave the range of double
    # precision; an entry which is not finite is replaced below, and a
    # regression whose design matrix under- or overflows into `NaN` (the
    # `LinAlgError` of `numpy.polyfit` on subnormal `x`) falls back to the
    # default start of every parameter
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            p0 = np.array(model.initial_guess(x, y), dtype=np.float64)
    except (np.linalg.LinAlgError, ValueError) as err:
        logger.debug("initial guess of %s failed: %s", model.name, err)
        p0 = np.full(model.n_parameters, np.nan)
    for name, value in options.initial.items():
        p0[model.parameter_names.index(name)] = value
    for i, scale in enumerate(scales):
        if scale is not ParameterScale.LINEAR and (
            not np.isfinite(p0[i]) or p0[i] <= 0
        ):
            p0[i] = 1e-12
    p0 = np.where(np.isfinite(p0), p0, 0.0)
    inside = np.clip(p0, lower, upper)
    # strictly inside for the optimizer
    span = np.where(np.isfinite(upper - lower), (upper - lower) * 1e-6, 0.0)
    return np.clip(inside, lower + span, upper - span)


def _nan_rowfit(k: int, width: int, n_points: int, flags: int, model: Model) -> RowFit:
    """A row without a fit.

    `n_points` is the number of points the fit would have used, so that a row
    which was never fitted reports how much data it had: zero without data,
    the finite points otherwise. The point arrays keep the width of the row,
    like every fitted row, so the result stacks into one `point` dimension.

    Args:
        k: number of parameters of the model.
        width: number of points of the row (the width of the point arrays).
        n_points: number of usable points of the row.
        flags: the flags explaining why there is no fit.
        model: the model (for the names of the derived parameters).

    Returns:
        A `RowFit` of `NaN` values.
    """
    nan_k = np.full(k, np.nan)
    nan_d = dict.fromkeys(model.derived_units, math.nan)
    return RowFit(
        p=nan_k,
        q=nan_k.copy(),
        se_p=nan_k.copy(),
        ci_low=nan_k.copy(),
        ci_high=nan_k.copy(),
        cov_q=np.full((k, k), np.nan),
        correlation=np.full((k, k), np.nan),
        derived=dict(nan_d),
        derived_se=dict(nan_d),
        derived_ci_low=dict(nan_d),
        derived_ci_high=dict(nan_d),
        cost=math.nan,
        r2=math.nan,
        rmse=math.nan,
        aic=math.nan,
        aicc=math.nan,
        bic=math.nan,
        n_points=n_points,
        n_starts_converged=0,
        y_pred=np.full(width, np.nan),
        residuals=np.full(width, np.nan),
        flags=flags,
        nfev=0,
    )


def _starts(
    q0: np.ndarray,
    lq: np.ndarray,
    uq: np.ndarray,
    scales: Sequence[ParameterScale],
    options: FitOptions,
    rng: np.random.Generator,
) -> np.ndarray:
    """Start points in the scaled space: the guess plus Latin hypercube samples in the start box.

    Args:
        q0: the initial guess on the search scale.
        lq: lower bounds on the search scale.
        uq: upper bounds on the search scale.
        scales: the scale per free parameter.
        options: the options (`n_starts`, `start_spread`).
        rng: random generator of the sampling.

    Returns:
        The start points, `(n_starts, k)`, the guess first.
    """
    if options.n_starts == 1:
        return q0[None, :]
    unit = qmc.LatinHypercube(d=q0.size, seed=rng).random(options.n_starts - 1)
    # the box of a linear parameter near the largest double overflows; its
    # starts are then not finite, and `fit_row` skips them
    with np.errstate(over="ignore", invalid="ignore"):
        half = np.array(
            [
                math.log10(options.start_spread)
                if s is ParameterScale.LOG10
                else math.log(options.start_spread)
                if s is ParameterScale.LOG
                else options.start_spread * max(abs(v), 1.0)
                for s, v in zip(scales, q0, strict=True)
            ]
        )
        lo = np.maximum(q0 - half, lq)
        hi = np.minimum(q0 + half, uq)
        return np.vstack([q0[None, :], lo + unit * (hi - lo)])


def replicate_statistics(
    values: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard deviation and percentile interval of bootstrap replicates, column by column.

    The finite replicates of a column are counted first and a column with
    fewer than two of them is reported as `NaN`, the guard
    `pkpdutils.result.ParameterResult.summarize` uses: `numpy.nanstd` with
    `ddof=1` on such a column has no degrees of freedom left (and would warn,
    and abort the batch under a strict warning filter) and the percentiles of
    an all-`NaN` column have nothing to interpolate, so neither returns a
    meaningful number.

    Args:
        values: the replicates, `(B, m)`, `NaN` where a replicate has no value.
        alpha: `1 - ci_level`, the total tail probability of the interval.

    Returns:
        `(sd, ci_low, ci_high)`, one value per column.
    """
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(values, axis=0, ddof=1)
        low, high = nan_percentile(
            values, (100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)), axis=0
        )
    usable = count > 1
    return (
        np.where(usable, sd, np.nan),
        np.where(usable, low, np.nan),
        np.where(usable, high, np.nan),
    )


def _statistics(
    y: np.ndarray, y_pred: np.ndarray, r: np.ndarray, k: int
) -> tuple[float, float, float, float, float, bool]:
    """`r2`, `rmse`, `aic`, `aicc` and `bic` of a fitted row, and whether one of them overflowed.

    `R² = 1 - Σe²/Σ(y - ȳ)²` and `RMSE = sqrt(Σe²/n)` on the unweighted
    residuals `e = y - f`; `AIC = n ln(Σr²/n) + 2K`, `AICc = AIC +
    2K(K+1)/(n-K-1)` and `BIC = n ln(Σr²/n) + K ln n` on the weighted ones,
    with `K = k + 1` estimated parameters, the residual variance being one of
    them (Burnham & Anderson 2002, sec. 2.2, 6.9.6); the reported
    `n_parameters` stays `k`. The sums run under `numpy.errstate`, and a sum
    or a statistic which is not finite is `NaN` and reported as an overflow;
    `r2` of constant data is `NaN`, `aic` and `bic` of a perfect fit are
    `-inf` and `aicc` is `NaN` when `n - K - 1 <= 0`, none of them an
    overflow.

    Args:
        y: the values of the row which were fitted.
        y_pred: the prediction at them.
        r: the weighted residuals.
        k: number of free parameters.

    Returns:
        `(r2, rmse, aic, aicc, bic, overflow)`.
    """
    n = y.size
    big_k = k + 1
    with np.errstate(over="ignore", invalid="ignore"):
        e = y - y_pred
        rss_w = float(np.sum(r**2))
        rss = float(np.sum(e**2))
        tss = float(np.sum((y - y.mean()) ** 2))
    overflow = False
    if math.isfinite(rss_w):
        ln_term = n * math.log(rss_w / n) if rss_w > 0 else -math.inf
        aic = ln_term + 2 * big_k
        aicc = (
            aic + 2 * big_k * (big_k + 1) / (n - big_k - 1)
            if n - big_k - 1 > 0
            else math.nan
        )
        bic = ln_term + big_k * math.log(n)
    else:
        overflow = True
        aic = aicc = bic = math.nan
    r2 = 1.0 - rss / tss if tss > 0 else math.nan
    rmse = math.sqrt(rss / n)
    if not (math.isfinite(rss) and math.isfinite(tss)) or math.isinf(r2):
        # `1 - rss / inf` would be a perfect fit
        overflow = True
        r2 = rmse = math.nan
    return r2, rmse, aic, aicc, bic, overflow


def _embed(cov_free: np.ndarray, free: np.ndarray, k_all: int) -> np.ndarray:
    """The covariance of the free parameters in the full parameter grid (NaN for fixed ones).

    Args:
        cov_free: covariance of the free parameters.
        free: mask of the free parameters.
        k_all: number of parameters of the model.

    Returns:
        The `(k_all, k_all)` covariance with `NaN` for the fixed parameters.
    """
    cov = np.full((k_all, k_all), np.nan)
    cov[np.ix_(free, free)] = cov_free
    return cov


def fit_row(
    model: Model,
    x: np.ndarray,
    y: np.ndarray,
    sd: np.ndarray | None,
    options: FitOptions,
    rng: np.random.Generator,
) -> RowFit:
    """Fit one row from one or several start points and compute its statistics.

    A model which defines `parameter_order` (the sums of exponentials) has its
    parameters permuted after the fit, together with every positional array
    (the free mask, the bounds, the scales, the fixed values, the start
    vector and the columns of the Jacobian), so the standard errors, the
    intervals, the correlation matrix and the `AT_BOUND` flag describe the
    parameter they are labelled with. The permutation is skipped when
    `options.fixed` or `options.bounds` names a parameter of the model: the
    user pinned that label, so the phases keep the labelling of the options
    even when they are then not ordered by decreasing rate.

    Args:
        model: the model
        x: independent variable `(n,)`
        y: dependent variable `(n,)`, `NaN` for missing points
        sd: standard deviation per point for `Weighting.INV_SD`, else `None`
        options: the options
        rng: random generator of the start points

    Returns:
        The fit of the row.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if sd is not None:
        sd = np.asarray(sd, dtype=np.float64)
        if options.weighting is Weighting.INV_SD:
            ok &= np.isfinite(sd) & (sd > 0)
    names = model.parameter_names
    k_all = len(names)
    free, lower, upper, scales, fixed_values = _problem(model, options)
    k = int(free.sum())
    n = int(ok.sum())
    if n < 2:
        # fewer than two points is no curve at all, there is nothing to fit
        return _nan_rowfit(k_all, x.size, 0, int(FitFlag.NO_DATA), model)
    if n < k + 1:
        return _nan_rowfit(k_all, x.size, n, int(FitFlag.TOO_FEW_POINTS), model)
    xs, ys = x[ok], y[ok]
    sqrt_var = residual_sd(ys, None if sd is None else sd[ok], options.weighting)
    p0 = _start_vector(model, xs, ys, options, lower, upper, scales)
    p0 = np.where(free, p0, fixed_values)
    lq_all, uq_all = bounds_in_scale(lower, upper, scales)
    q0_all = to_scale(p0, scales)

    def full_p(q_free: np.ndarray) -> np.ndarray:
        """The full parameter vector on the linear scale from the free parameters."""
        q = q0_all.copy()
        q[free] = q_free
        p_full = from_scale(q, scales)
        # the fixed parameters keep their exact value, no scale round trip
        p_full[~free] = p0[~free]
        return p_full

    def residuals(q_free: np.ndarray) -> np.ndarray:
        """The weighted residuals of the free parameters."""
        return (ys - model.predict(xs, full_p(q_free))) / sqrt_var

    starts = _starts(
        q0_all[free],
        lq_all[free],
        uq_all[free],
        [s for s, f in zip(scales, free, strict=True) if f],
        options,
        rng,
    )
    best: tuple[Any, bool] | None = None
    n_converged = 0
    nfev = 0
    overflowed = False
    for q_start in starts:
        if not np.all(np.isfinite(q_start)):
            # a start box beyond the range of double precision (`_starts`)
            continue
        # finite residuals whose sum of squares overflows would have the search
        # compare infinite costs; residuals which are not finite are refused
        # by scipy itself
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            r_start = residuals(q_start)
            cost_start = 0.5 * float(r_start @ r_start)
        if np.all(np.isfinite(r_start)) and not math.isfinite(cost_start):
            overflowed = True
            logger.debug("start skipped: its sum of squares overflows")
            continue
        try:
            # a step of the search can make the model overflow to `inf` or
            # `nan` (`from_scale` already silences its own overflow), and on
            # data spanning many orders of magnitude the trust region itself
            # overflows and divides by zero (the Levenberg-Marquardt parameter
            # of `scipy.optimize._lsq.common.solve_lsq_trust_region`); scipy
            # rejects a step with a non-finite residual or cost, so ignoring
            # the floating-point warnings here lets that path run instead of
            # the warning escaping as an exception under a strict filter
            # (`pytest -W error`). The cost of the solution is checked below.
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                solution: Any = least_squares(
                    residuals,
                    q_start,
                    bounds=(lq_all[free], uq_all[free]),
                    method="trf",
                    loss=options.loss,
                    max_nfev=options.max_nfev,
                    ftol=options.ftol,
                    xtol=options.xtol,
                    gtol=options.gtol,
                )
        except (ValueError, np.linalg.LinAlgError) as err:
            logger.debug("start failed: %s", err)
            continue
        nfev += int(solution.nfev)
        if not (
            math.isfinite(float(solution.cost)) and np.all(np.isfinite(solution.x))
        ):
            # the search compared costs which are not finite, so its end point
            # means nothing
            overflowed = True
            logger.debug("start overflowed: the cost is %s", solution.cost)
            continue
        converged = bool(solution.status > 0)
        n_converged += int(converged)
        if (
            best is None
            or (converged and not best[1])
            or (converged == best[1] and solution.cost < best[0].cost)
        ):
            best = (solution, converged)
    if best is None:
        failed = FitFlag.NOT_CONVERGED | (FitFlag.OVERFLOW if overflowed else 0)
        return _nan_rowfit(k_all, x.size, n, int(failed), model)
    solution, converged = best
    flags = 0 if converged else int(FitFlag.NOT_CONVERGED)

    jac = np.asarray(solution.jac, dtype=np.float64)
    p = full_p(np.array(solution.x, dtype=np.float64))
    # reorder the phases of a sum of exponentials, with every positional array
    order_fn = getattr(model, "parameter_order", None)
    pinned = (set(options.fixed) | set(options.bounds)) & set(names)
    if order_fn is not None and not pinned:
        order = np.asarray(order_fn(p))
        if not np.array_equal(order, np.arange(k_all)):
            free_order = [
                int(np.flatnonzero(np.flatnonzero(free) == i)[0])
                for i in order
                if free[i]
            ]
            jac = jac[:, free_order]
            p = p[order]
            p0 = p0[order]
            free = free[order]
            lower, upper = lower[order], upper[order]
            scales = [scales[i] for i in order]
            fixed_values = fixed_values[order]
            q0_all = q0_all[order]
            # the scaled bounds are derived from the bounds and the scales, so
            # they are recomputed rather than permuted; the bootstrap below
            # refits under `lq_all[free]`/`uq_all[free]` and would otherwise
            # bound every phase by the bounds of another one
            lq_all, uq_all = bounds_in_scale(lower, upper, scales)
    cov_free, condition = covariance(jac, float(solution.cost), n, k)
    flags |= int(condition)
    unusable = condition != FitFlag.NONE
    q_all = to_scale(p, scales)
    dpdq = scale_derivative(p, scales)
    se_q = np.sqrt(np.clip(np.diag(cov_free), 0.0, None))
    tq = float(student_t.ppf(1.0 - (1.0 - options.ci_level) / 2.0, max(n - k, 1)))
    se_p = np.full(k_all, np.nan)
    # a parameter near the largest double has an infinite `dp/dq`, and so an
    # infinite standard error (`NaN` for a standard error of zero)
    with np.errstate(over="ignore", invalid="ignore"):
        se_p[free] = se_q * np.abs(dpdq[free])
    lo_q, hi_q = q_all.copy(), q_all.copy()
    lo_q[free] = q_all[free] - tq * se_q
    hi_q[free] = q_all[free] + tq * se_q
    ci_low = np.where(free, from_scale(lo_q, scales), np.nan)
    ci_high = np.where(free, from_scale(hi_q, scales), np.nan)
    corr = np.full((k_all, k_all), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_free = cov_free / np.outer(se_q, se_q)
    corr[np.ix_(free, free)] = corr_free
    # parameters resting on a bound, measured relative to their start value so
    # that a lower bound of 0 (which is -inf on the log scale) is detected too
    tol = options.at_bound_tolerance
    for i in np.flatnonzero(free):
        atol = tol * max(abs(float(p0[i])), 1e-300)
        for bound in (lower[i], upper[i]):
            if np.isfinite(bound) and np.isclose(p[i], bound, rtol=tol, atol=atol):
                flags |= int(FitFlag.AT_BOUND)
    # derived parameters with the delta method. A parameter which ended at
    # zero or beyond the range of `float64` has a non-finite search-scale
    # value, so the central difference around it is `inf - inf` or a step of
    # `inf`: the gradient is then meaningless and the uncertainty of every
    # derived parameter is reported as `NaN` instead. The block runs under
    # `numpy.errstate` so that a derived parameter of such a degenerate row
    # (a half-life of a rate constant of zero, an overflow of the model at a
    # perturbed parameter) stays a non-finite number instead of escaping as a
    # `RuntimeWarning` under a strict filter and aborting the whole batch.
    derived_se: dict[str, float] = {}
    derived_lo: dict[str, float] = {}
    derived_hi: dict[str, float] = {}
    usable_gradient = bool(np.all(np.isfinite(q_all)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        derived = model.derived(p)
        for name, value in derived.items():
            if not usable_gradient:
                derived_se[name] = math.nan
                derived_lo[name] = math.nan
                derived_hi[name] = math.nan
                continue
            grad = np.zeros(k)
            for j, i in enumerate(np.flatnonzero(free)):
                h = 1e-6 * max(abs(q_all[i]), 1.0)
                q_plus, q_minus = q_all.copy(), q_all.copy()
                q_plus[i] += h
                q_minus[i] -= h
                grad[j] = (
                    model.derived(from_scale(q_plus, scales))[name]
                    - model.derived(from_scale(q_minus, scales))[name]
                ) / (2 * h)
            var_d = math.nan if unusable else float(grad @ cov_free @ grad)
            se_d = math.sqrt(var_d) if var_d >= 0 else math.nan
            derived_se[name] = se_d
            derived_lo[name] = value - tq * se_d
            derived_hi[name] = value + tq * se_d
    if derived.get("flip_flop", 0.0) >= 1.0:
        flags |= int(FitFlag.FLIP_FLOP)
    # statistics
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        y_pred = model.predict(xs, p)
        r = (ys - y_pred) / sqrt_var
    r2, rmse, aic, aicc, bic, overflow = _statistics(ys, y_pred, r, k)
    if overflow:
        flags |= int(FitFlag.OVERFLOW)
    full_pred = np.full(x.size, np.nan)
    full_res = np.full(x.size, np.nan)
    full_pred[ok] = y_pred
    full_res[ok] = r
    # residual bootstrap (Efron & Tibshirani 1993, ch. 9): the weighted
    # residuals of the fit are centered and inflated by `sqrt(n / (n - k))`
    # so their variance matches the residual variance of the fit rather than
    # the smaller variance of the raw residuals (ch. 9, eq. 9.10), then
    # resampled with replacement, and the model is refitted from the fitted
    # `p`, single start, the same bounds and scale; the standard errors, the
    # intervals and the correlation matrix are then the empirical statistics
    # of the replicate parameters, an alternative to the Jacobian-based ones
    # above that does not rely on the local linear approximation of the
    # covariance. A replicate whose refit fails or does not converge is
    # skipped; the phases of a sum of exponentials are reordered exactly as
    # the reported fit was (never when a parameter is pinned by
    # `options.fixed` or `options.bounds`), so every replicate is compared
    # under the same parameter labelling as `p`. Fewer than two converged
    # replicates cannot estimate an uncertainty, so the Jacobian-based
    # statistics are kept and `FitFlag.BOOTSTRAP_FALLBACK` is set.
    n_boot = 0
    if options.bootstrap > 0:
        # center and inflate the weighted residuals so their empirical
        # variance matches the residual variance of the fit, `s^2 = rss_w /
        # (n - k)`, rather than the (smaller) variance of the raw residuals,
        # `rss_w / n` (Efron & Tibshirani 1993, ch. 9, eq. 9.10): `r_adj =
        # (r - mean(r)) * sqrt(n / (n - k))`. `n - k >= 1` here, the row
        # already returned `TOO_FEW_POINTS` otherwise.
        # the replicate data of a row beyond the range of double precision
        # are not finite, and such a replicate is skipped like a failed refit
        with np.errstate(over="ignore", invalid="ignore"):
            r_adj = (r - r.mean()) * math.sqrt(n / (n - k))
        q_free = q_all[free]
        rows_p: list[np.ndarray] = []
        rows_q: list[np.ndarray] = []
        rows_d: list[dict[str, float]] = []
        for _ in range(options.bootstrap):
            r_star = rng.choice(r_adj, size=r_adj.size, replace=True)
            with np.errstate(over="ignore", invalid="ignore"):
                y_star = y_pred + r_star * sqrt_var
            if not np.all(np.isfinite(y_star)):
                continue

            def residuals_star(
                qf: np.ndarray, y_star: np.ndarray = y_star
            ) -> np.ndarray:
                """The weighted residuals of one bootstrap replicate."""
                return (y_star - model.predict(xs, full_p(qf))) / sqrt_var

            try:
                # the floating-point warnings of the search, as for the fit
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    sol_star: Any = least_squares(
                        residuals_star,
                        q_free,
                        bounds=(lq_all[free], uq_all[free]),
                        method="trf",
                        loss=options.loss,
                        max_nfev=options.max_nfev,
                        ftol=options.ftol,
                        xtol=options.xtol,
                        gtol=options.gtol,
                    )
            except (ValueError, np.linalg.LinAlgError) as err:
                logger.debug("bootstrap replicate failed: %s", err)
                continue
            if sol_star.status <= 0 or not math.isfinite(float(sol_star.cost)):
                continue
            p_star = full_p(np.array(sol_star.x, dtype=np.float64))
            if order_fn is not None and not pinned:
                order_star = np.asarray(order_fn(p_star))
                p_star = p_star[order_star]
            rows_p.append(p_star)
            rows_q.append(to_scale(p_star, scales))
            # a replicate can be degenerate in the same way as the fit above
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                rows_d.append(model.derived(p_star))
        n_boot = len(rows_p)
        if n_boot < 2:
            # too few replicates converged to estimate the uncertainty from;
            # the Jacobian-based se_p/ci_low/ci_high/correlation computed
            # above are kept and the caller is told so through the flag.
            flags |= int(FitFlag.BOOTSTRAP_FALLBACK)
        else:
            arr_p = np.stack(rows_p)
            arr_q = np.stack(rows_q)
            alpha = 1.0 - options.ci_level
            # nan-tolerant: a replicate can be finite in its own parameters
            # but give a non-finite derived quantity (e.g. thalf for k ~ 0)
            sd_b, low_b, high_b = replicate_statistics(arr_p, alpha)
            se_p = np.where(free, sd_b, np.nan)
            ci_low = np.where(free, low_b, np.nan)
            ci_high = np.where(free, high_b, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr_b = np.corrcoef(arr_q[:, free], rowvar=False)
            corr = np.full((k_all, k_all), np.nan)
            corr[np.ix_(free, free)] = np.atleast_2d(corr_b)
            derived_names = list(derived)
            if derived_names:
                arr_d = np.array(
                    [[d[name] for name in derived_names] for d in rows_d],
                    dtype=np.float64,
                )
                sd_d, low_d, high_d = replicate_statistics(arr_d, alpha)
                for j, name in enumerate(derived_names):
                    derived_se[name] = float(sd_d[j])
                    derived_lo[name] = float(low_d[j])
                    derived_hi[name] = float(high_d[j])
    return RowFit(
        p=p,
        q=q_all,
        se_p=se_p,
        ci_low=ci_low,
        ci_high=ci_high,
        cov_q=_embed(cov_free, free, k_all),
        correlation=corr,
        derived=derived,
        derived_se=derived_se,
        derived_ci_low=derived_lo,
        derived_ci_high=derived_hi,
        cost=float(solution.cost),
        r2=r2,
        rmse=rmse,
        aic=aic,
        aicc=aicc,
        bic=bic,
        n_points=n,
        n_starts_converged=n_converged,
        y_pred=full_pred,
        residuals=full_res,
        flags=flags,
        nfev=nfev,
        n_bootstrap=n_boot,
    )


def _fit_row_job(
    model: Model,
    options: FitOptions,
    row: tuple[np.ndarray, np.ndarray, np.ndarray | None, int],
) -> RowFit:
    """Worker entry point for `fit_rows`: fit one row from its data and seed.

    A module-level function so that it can be pickled for a
    `ProcessPoolExecutor`. The model and the options come first, so that
    `fit_rows` can bind them with `functools.partial`: they are then pickled
    once per task of the pool instead of once per row, and a row sends only
    its own arrays and its seed.

    Args:
        model: the model.
        options: the options.
        row: `(x, y, sd, seed)` of one row.

    Returns:
        The fit of the row.
    """
    x, y, sd, seed = row
    return fit_row(model, x, y, sd, options, np.random.default_rng(int(seed)))


def fit_rows(
    model: Model,
    x: np.ndarray,
    y: np.ndarray,
    sd: np.ndarray | None,
    options: FitOptions,
) -> list[RowFit]:
    """Fit every row of `(N, n)` arrays, serially or in the shared process pool.

    Row seeds are drawn from `options.seed` before the rows are distributed,
    one child seed per row, so the result does not depend on
    `options.n_workers` or the order the rows finish in.

    `options.n_workers` decides how many workers run the rows
    (`pkpdutils.parallel.resolve_workers`): `None` is automatic and stays in
    the calling process below `FIT_WORKER_THRESHOLD` rows, `1` is serial and
    any other number is taken as given. A row is a python-heavy
    `scipy.optimize.least_squares` search, so a parallel run maps the rows
    over the shared process pool (`pkpdutils.parallel.executor`) in batches of
    about a quarter of the rows of a worker, which keeps the number of tasks
    (and with them the pickling of the model and the options) small. A worker
    that dies takes the pool with it (`BrokenProcessPool`): the pool is then
    evicted and the batch is fitted once more in a fresh one, with a warning.
    A pooled call must run under an `if __name__ == "__main__":` guard, since
    the pool starts its workers with `forkserver` or `spawn` on every python
    version (`pkpdutils.parallel.PROCESS_START_METHOD`), which re-import the
    module without re-running it.

    Args:
        model: the model
        x: independent variable `(N, n)`
        y: dependent variable `(N, n)`
        sd: standard deviations `(N, n)` or `None`
        options: the options

    Returns:
        One `RowFit` per row, in row order.
    """
    n_rows = int(y.shape[0])
    rng = np.random.default_rng(options.seed)
    seeds = rng.integers(0, 2**32 - 1, size=n_rows)
    rows = [
        (x[i], y[i], None if sd is None else sd[i], int(seeds[i]))
        for i in range(n_rows)
    ]
    n_workers = resolve_workers(
        options.n_workers, n_rows, threshold=FIT_WORKER_THRESHOLD
    )
    job = functools.partial(_fit_row_job, model, options)
    if n_workers > 1 and n_rows > 1:
        chunksize = max(1, n_rows // (4 * n_workers))
        logger.debug(
            "fit: %d rows over %d processes, chunksize %d",
            n_rows,
            n_workers,
            chunksize,
        )
        try:
            pool = executor("process", n_workers)
            return list(pool.map(job, rows, chunksize=chunksize))
        except BrokenProcessPool:
            # a worker died (killed by the operating system, or by a crash in
            # a native extension); the pool is shared, so it is dropped before
            # the batch is fitted once more in a fresh one
            logger.warning(
                "fit: a worker of the shared process pool died, retrying the "
                "%d rows in a new pool",
                n_rows,
            )
            evict("process", n_workers)
            pool = executor("process", n_workers)
            return list(pool.map(job, rows, chunksize=chunksize))
    return [job(row) for row in rows]


def _as_rows(
    x: Any, y: Any, sd: Any | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]:
    """`(N, n)` arrays from 1-D or 2-D inputs.

    Args:
        x: independent variable, `(n,)` or `(N, n)`
        y: dependent variable, `(n,)` or `(N, n)`
        sd: standard deviations like `y`, or `None`

    Returns:
        `(x, y, sd, single)`; `single` says whether the input was one row.

    Raises:
        ValueError: if the shapes of `x` or `sd` do not match `y`.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    single = y_arr.ndim == 1
    if single:
        y_arr = y_arr[None, :]
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = np.broadcast_to(x_arr, y_arr.shape).copy()
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"'x' has shape {x_arr.shape}, 'y' has shape {y_arr.shape}")
    sd_arr = None
    if sd is not None:
        sd_arr = np.asarray(sd, dtype=np.float64)
        if sd_arr.ndim == 1:
            sd_arr = (
                sd_arr[None, :]
                if single
                else np.broadcast_to(sd_arr, y_arr.shape).copy()
            )
        if sd_arr.shape != y_arr.shape:
            raise ValueError(
                f"'sd' has shape {sd_arr.shape}, 'y' has shape {y_arr.shape}"
            )
    return x_arr, y_arr, sd_arr, single


def _named(
    result: FitResult, x_name: str | None = None, y_name: str | None = None
) -> FitResult:
    """The result with the names of its variables in `attrs`.

    `attrs["x_name"]` and `attrs["y_name"]` name what was fitted against
    what, so that a figure of the result labels its axes with them
    (`plot_fit`, `plot_dose_proportionality`) instead of `x` and `y`. A name
    which is `None` is not written, and a result without the attributes falls
    back to `x` and `y` in the figures.

    Args:
        result: the result of the engine.
        x_name: name of the independent variable, `None` to leave it out.
        y_name: name of the dependent variable, `None` to leave it out.

    Returns:
        The result carrying the names which were given, the result itself
        when neither was.
    """
    names = {
        key: value
        for key, value in (("x_name", x_name), ("y_name", y_name))
        if value is not None
    }
    if not names:
        return result
    return FitResult(result.ds.assign_attrs(**names), result.model)


def fit(
    model: Model,
    x: Any,
    y: Any,
    *,
    sd: Any | None = None,
    options: FitOptions | None = None,
    x_unit: str = "dimensionless",
    y_unit: str = "dimensionless",
    x_name: str | None = None,
    y_name: str | None = None,
    dims: Sequence[str] | None = None,
    coords: dict[str, Any] | None = None,
) -> FitResult:
    """Fit a model to one or many rows of data.

    Args:
        model: the model
        x: independent variable, `(n,)` shared by all rows or `(N, n)`
        y: dependent variable, `(n,)` for one sample or `(N, n)`; `NaN` for missing points
        sd: standard deviation per point (needed for `Weighting.INV_SD`), like `y`
        options: the options, defaults for `None`
        x_unit: unit of `x`
        y_unit: unit of `y`
        x_name: name of the independent variable, stored as `attrs["x_name"]`
            and used as the axis label by the figures of the fit
            (`concentration`, `dose`, `weight`); the figures fall back to `x`
            without it, as the arrays carry no name of their own
        y_name: name of the dependent variable, stored as `attrs["y_name"]`,
            the label of the value axis (`effect`, `auc_inf_obs`)
        dims: sample dimension names for a 2-D `y`, `("sample",)` by default
        coords: coordinate labels of the sample dimensions

    Returns:
        The result over the sample dimensions (none for a 1-D `y`).

    Raises:
        ValueError: for 2-D data with more than one sample dimension, or if
            the sample dimension or a coordinate collides with a variable or
            a dimension of the result (see `build_result`).
    """
    options = options or FitOptions()
    x_arr, y_arr, sd_arr, single = _as_rows(x, y, sd)
    sample_dims: tuple[str, ...] = () if single else tuple(dims or ("sample",))
    if not single and len(sample_dims) != 1:
        raise ValueError(
            "2-D data has exactly one sample dimension; use fit_timecourses or fit_table for more"
        )
    rows = fit_rows(model, x_arr, y_arr, sd_arr, options)
    result = build_result(
        model,
        rows,
        x=x_arr,
        y=y_arr,
        sd=sd_arr,
        x_unit=x_unit,
        y_unit=y_unit,
        dims=sample_dims,
        coords=coords or {},
        options=options,
    )
    return _named(result, x_name, y_name)


def _cv(se: float, value: float) -> float:
    """The coefficient of variation as a fraction, `NaN` for a zero or missing value.

    The package reports every coefficient of variation as a fraction
    (`NCAResult` `x_geocv`, `stats.Summary.cv`); a table which shows percent
    multiplies by 100 where it formats.

    Args:
        se: the standard error.
        value: the estimate.

    Returns:
        `se / |value|`.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(se) / np.abs(np.float64(value)))


def _check_no_reserved_suffix(model: Model) -> None:
    """Reject a model whose parameter or derived name ends in a suffix of the result variables.

    The result writes the uncertainty of a parameter `p` as `p_se`,
    `p_ci_low`, ... and the statistics of a summary as `p_median`, `p_min`,
    `p_max`, ...; `ParameterResult` reads that structure back with
    `pkpdutils.result.base_name`, so a parameter named `k_n`, `auc_se`,
    `e_max` or `c_min` would be classified as a derived variable of a
    parameter `k`, `auc`, `e` or `c` that does not exist, and `summarize`
    would then drop it. Write such a parameter as `emax` or `cmin`, the
    spelling of the rest of the package.

    Args:
        model: the model, for its parameter and derived names.

    Raises:
        ValueError: if a parameter or derived name ends in a reserved suffix.
    """
    for name in (*model.parameter_names, *model.derived_units):
        stem = base_name(name)
        if stem is not None:
            raise ValueError(
                f"'{name}' of {model.name} ends in the reserved suffix "
                f"'{name[len(stem) :]}', which the result uses for the "
                f"derived variables of a parameter; rename the parameter"
            )


def build_result(
    model: Model,
    rows: list[RowFit],
    *,
    x: np.ndarray,
    y: np.ndarray,
    sd: np.ndarray | None,
    x_unit: str,
    y_unit: str,
    dims: tuple[str, ...],
    coords: dict[str, Any],
    options: FitOptions,
    shape: tuple[int, ...] | None = None,
) -> FitResult:
    """Assemble the result dataset of the fitted rows.

    The parameters, their uncertainties and the derived parameters are
    reported in the raw units of `x` and `y` (`parameter_unit_expression`), so
    they are on the same scale as `y_data` and `y_pred`. `residuals` holds the
    weighted residuals `(y - f) / sqrt(var)`, which are dimensionless only
    under `Weighting.INV_SD` and otherwise carry the unit of `y` divided by
    the square root of the variance model; `rmse` is the root mean square of
    the unweighted residuals `y - f` and carries the unit of `y`. Both are
    reported as `dimensionless`, like the other goodness-of-fit statistics.
    Derived parameters listed in `FitResult.discrete_parameters` (indicators
    such as `flip_flop`) are written without `_se`, `_ci_low`, `_ci_high` and
    `_cv`.

    Args:
        model: the model
        rows: one `RowFit` per row
        x: `(N, n)` independent variable
        y: `(N, n)` dependent variable
        sd: `(N, n)` standard deviations or `None`, reported as `sd_data`
            (`NaN` for every point when `None`)
        x_unit: unit of `x`
        y_unit: unit of `y`
        dims: sample dimension names (`()` for a single row)
        coords: coordinates of the sample dimensions
        options: the options (stored in `attrs`)
        shape: sample shape for several sample dimensions (`N = prod(shape)`), `(N,)` by default

    Returns:
        The `FitResult`.

    Raises:
        ValueError: if a parameter or derived name of the model ends in a
            reserved suffix (`_check_no_reserved_suffix`), or if a name in
            `dims` or in `coords` collides with a data variable of the result
            or with one of the dimensions `point`, `parameter` and
            `parameter_` it adds (`check_coordinate_collision`, which reads
            the variables and the dimensions from the assembled layout).
    """
    n_rows, n_points = y.shape
    sample_shape: tuple[int, ...] = (
        () if not dims else (shape if shape is not None else (n_rows,))
    )
    names = model.parameter_names
    k = len(names)
    _check_no_reserved_suffix(model)

    def units_of(expr: str) -> str:
        """The unit of a unit expression in the units of the data."""
        return parameter_unit_expression(expr, x_unit=x_unit, y_unit=y_unit)

    data_vars: dict[str, Any] = {}

    def scalar(name: str, values: list[float] | np.ndarray, unit: str) -> None:
        """Add a scalar variable over the sample dimensions."""
        data_vars[name] = (
            dims,
            np.asarray(values, dtype=np.float64).reshape(sample_shape),
            {"units": unit},
        )

    for j, parameter in enumerate(model.parameters):
        unit = units_of(parameter.unit_expr)
        scalar(parameter.name, [r.p[j] for r in rows], unit)
        scalar(f"{parameter.name}_se", [r.se_p[j] for r in rows], unit)
        scalar(f"{parameter.name}_ci_low", [r.ci_low[j] for r in rows], unit)
        scalar(f"{parameter.name}_ci_high", [r.ci_high[j] for r in rows], unit)
        scalar(
            f"{parameter.name}_cv",
            [_cv(r.se_p[j], r.p[j]) for r in rows],
            "dimensionless",
        )
    for name, expr in model.derived_units.items():
        unit = units_of(expr)
        scalar(name, [r.derived.get(name, math.nan) for r in rows], unit)
        if name in FitResult.discrete_parameters:
            # an indicator such as `flip_flop` carries no uncertainty
            continue
        scalar(f"{name}_se", [r.derived_se.get(name, math.nan) for r in rows], unit)
        scalar(
            f"{name}_ci_low",
            [r.derived_ci_low.get(name, math.nan) for r in rows],
            unit,
        )
        scalar(
            f"{name}_ci_high",
            [r.derived_ci_high.get(name, math.nan) for r in rows],
            unit,
        )
        scalar(
            f"{name}_cv",
            [
                _cv(r.derived_se.get(name, math.nan), r.derived.get(name, math.nan))
                for r in rows
            ],
            "dimensionless",
        )
    for stat in ("cost", "r2", "rmse", "aic", "aicc", "bic"):
        scalar(stat, [getattr(r, stat) for r in rows], "dimensionless")
    scalar("n_points", [r.n_points for r in rows], "dimensionless")
    n_free = int(np.sum([name not in options.fixed for name in names]))
    scalar("n_parameters", [n_free for _ in rows], "dimensionless")
    scalar("n_starts_converged", [r.n_starts_converged for r in rows], "dimensionless")
    scalar("n_bootstrap", [r.n_bootstrap for r in rows], "dimensionless")
    point_dims = (*dims, "point")
    point_shape = (*sample_shape, n_points)
    data_vars["x_data"] = (point_dims, x.reshape(point_shape), {"units": x_unit})
    data_vars["y_data"] = (point_dims, y.reshape(point_shape), {"units": y_unit})
    sd_values = np.full(point_shape, np.nan) if sd is None else sd.reshape(point_shape)
    data_vars["sd_data"] = (point_dims, sd_values, {"units": y_unit})
    data_vars["y_pred"] = (
        point_dims,
        np.stack([r.y_pred for r in rows]).reshape(point_shape),
        {"units": y_unit},
    )
    data_vars["residuals"] = (
        point_dims,
        np.stack([r.residuals for r in rows]).reshape(point_shape),
        {"units": "dimensionless"},
    )
    data_vars["correlation"] = (
        (*dims, "parameter", "parameter_"),
        np.stack([r.correlation for r in rows]).reshape(*sample_shape, k, k),
        {"units": "dimensionless"},
    )
    data_vars["flags"] = (
        dims,
        np.array([r.flags for r in rows], dtype=np.int64).reshape(sample_shape),
        {"units": "dimensionless"},
    )
    # the names of the batch against the variables and the dimensions of the
    # layout above, before xarray sees them
    check_coordinate_collision(
        coords, {name: spec[0] for name, spec in data_vars.items()}, dims
    )
    all_coords: dict[str, Any] = {
        **coords,
        "parameter": list(names),
        "parameter_": list(names),
    }
    attrs = {
        "model": model.name,
        "parameter_names": list(names),
        "x_unit": x_unit,
        "y_unit": y_unit,
        "weighting": str(options.weighting),
        "parameter_scale": str(options.parameter_scale),
        "bootstrap": options.bootstrap,
    }
    return FitResult(
        xr.Dataset(data_vars=data_vars, coords=all_coords, attrs=attrs), model
    )
