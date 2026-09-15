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
`FitFlag.BOOTSTRAP_FALLBACK`. `fit_rows` distributes the rows over a
`ProcessPoolExecutor` when `options.n_workers > 1`, with one child seed per
row drawn up front so serial and pooled runs agree; `pkpdutils.fit.compare`
ranks several models on the same data by the corrected Akaike information
criterion (Burnham & Anderson 2002).
"""

import logging
import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
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

logger = logging.getLogger(__name__)

LN10 = math.log(10.0)


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


def variance_of(
    y: np.ndarray, sd: np.ndarray | None, weighting: Weighting
) -> np.ndarray:
    """Variance of every point under the weighting (`y <= 0` uses the smallest positive `|y|`).

    Args:
        y: the dependent variable of the row.
        sd: standard deviation per point, needed for `Weighting.INV_SD`.
        weighting: the variance model.

    Returns:
        The variance per point.

    Raises:
        ValueError: for `Weighting.INV_SD` without `sd`.
    """
    if weighting is Weighting.NONE:
        return np.ones_like(y)
    if weighting is Weighting.INV_SD:
        if sd is None:
            raise ValueError("Weighting.INV_SD needs 'sd'")
        return np.asarray(sd, dtype=np.float64) ** 2
    positive = np.abs(y[np.isfinite(y) & (y != 0)])
    floor = positive.min() if positive.size else 1.0
    base = np.where(np.isfinite(y) & (np.abs(y) > 0), np.abs(y), floor)
    return base if weighting is Weighting.INV_Y else base**2


def to_scale(p: np.ndarray, scales: Sequence[ParameterScale]) -> np.ndarray:
    """Linear parameters to the scaled space.

    Args:
        p: the parameters on the linear scale.
        scales: the scale per parameter.

    Returns:
        The parameters on the search scale.
    """
    q = np.array(p, dtype=np.float64)
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


def covariance(jac: np.ndarray, cost: float, n: int, k: int) -> tuple[np.ndarray, bool]:
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
    so that the standard errors stay finite.

    Args:
        jac: the Jacobian of the residuals in the scaled space, `(n, k)`.
        cost: the scipy cost `0.5 Σ r²`.
        n: number of points.
        k: number of free parameters.

    Returns:
        The covariance of the free parameters and whether it is unusable
        (singular `JᵀJ` or a negative variance).
    """
    if n <= k:
        return np.full((k, k), np.nan), True
    jtj = jac.T @ jac
    try:
        inv = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        return np.full((k, k), np.nan), True
    if not np.all(np.isfinite(inv)):
        return np.full((k, k), np.nan), True
    cov = inv * (2.0 * cost / (n - k))
    return cov, bool(np.any(np.diag(cov) < 0.0))


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
    p0 = np.array(model.initial_guess(x, y), dtype=np.float64)
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


def _nan_rowfit(k: int, n: int, flags: int, model: Model) -> RowFit:
    """A row without a fit.

    Args:
        k: number of parameters of the model.
        n: number of points of the row.
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
        n_points=n,
        n_starts_converged=0,
        y_pred=np.full(n, np.nan),
        residuals=np.full(n, np.nan),
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
    unit = qmc.LatinHypercube(d=q0.size, seed=rng).random(options.n_starts - 1)
    return np.vstack([q0[None, :], lo + unit * (hi - lo)])


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
        return _nan_rowfit(k_all, x.size, int(FitFlag.NO_DATA), model)
    if n < k + 1:
        return _nan_rowfit(k_all, x.size, int(FitFlag.TOO_FEW_POINTS), model)
    xs, ys = x[ok], y[ok]
    var = variance_of(ys, None if sd is None else sd[ok], options.weighting)
    sqrt_var = np.sqrt(var)
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
    for q_start in starts:
        try:
            # a wild start can make the model overflow to `inf` or `nan`
            # (`from_scale` already silences its own overflow); scipy turns
            # a non-finite residual into a `ValueError` it raises itself, so
            # ignoring the floating-point warning here just lets that failure
            # path run instead of the warning escaping as an exception under
            # a strict filter (`pytest -W error`).
            with np.errstate(over="ignore", invalid="ignore"):
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
        converged = bool(solution.status > 0)
        n_converged += int(converged)
        if (
            best is None
            or (converged and not best[1])
            or (converged == best[1] and solution.cost < best[0].cost)
        ):
            best = (solution, converged)
    if best is None:
        return _nan_rowfit(k_all, x.size, int(FitFlag.NOT_CONVERGED), model)
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
    cov_free, singular = covariance(jac, float(solution.cost), n, k)
    if singular:
        flags |= int(FitFlag.SINGULAR)
    q_all = to_scale(p, scales)
    dpdq = scale_derivative(p, scales)
    se_q = np.sqrt(np.clip(np.diag(cov_free), 0.0, None))
    tq = float(student_t.ppf(1.0 - (1.0 - options.ci_level) / 2.0, max(n - k, 1)))
    se_p = np.full(k_all, np.nan)
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
    # derived parameters with the delta method
    derived = model.derived(p)
    derived_se: dict[str, float] = {}
    derived_lo: dict[str, float] = {}
    derived_hi: dict[str, float] = {}
    for name, value in derived.items():
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
        var_d = math.nan if singular else float(grad @ cov_free @ grad)
        se_d = math.sqrt(var_d) if var_d >= 0 else math.nan
        derived_se[name] = se_d
        derived_lo[name] = value - tq * se_d
        derived_hi[name] = value + tq * se_d
    if derived.get("flip_flop", 0.0) >= 1.0:
        flags |= int(FitFlag.FLIP_FLOP)
    # statistics
    y_pred = model.predict(xs, p)
    e = ys - y_pred
    r = e / sqrt_var
    rss_w = float(np.sum(r**2))
    tss = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - float(np.sum(e**2)) / tss if tss > 0 else math.nan
    rmse = math.sqrt(float(np.sum(e**2)) / n)
    ln_term = n * math.log(rss_w / n) if rss_w > 0 else -math.inf
    # AIC/AICc/BIC count the residual variance as an estimated parameter,
    # `K = k + 1` (Burnham & Anderson 2002, sec. 2.2, 6.9.6); `n_parameters`
    # of the result stays `k`, the free model parameters.
    big_k = k + 1
    aic = ln_term + 2 * big_k
    aicc = (
        aic + 2 * big_k * (big_k + 1) / (n - big_k - 1)
        if n - big_k - 1 > 0
        else math.nan
    )
    bic = ln_term + big_k * math.log(n)
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
        r_adj = (r - r.mean()) * math.sqrt(n / (n - k))
        q_free = q_all[free]
        rows_p: list[np.ndarray] = []
        rows_q: list[np.ndarray] = []
        rows_d: list[dict[str, float]] = []
        for _ in range(options.bootstrap):
            r_star = rng.choice(r_adj, size=r_adj.size, replace=True)
            y_star = y_pred + r_star * sqrt_var

            def residuals_star(
                qf: np.ndarray, y_star: np.ndarray = y_star
            ) -> np.ndarray:
                """The weighted residuals of one bootstrap replicate."""
                return (y_star - model.predict(xs, full_p(qf))) / sqrt_var

            try:
                with np.errstate(over="ignore", invalid="ignore"):
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
            if sol_star.status <= 0:
                continue
            p_star = full_p(np.array(sol_star.x, dtype=np.float64))
            if order_fn is not None and not pinned:
                order_star = np.asarray(order_fn(p_star))
                p_star = p_star[order_star]
            rows_p.append(p_star)
            rows_q.append(to_scale(p_star, scales))
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
            se_p = np.where(free, np.nanstd(arr_p, axis=0, ddof=1), np.nan)
            ci_low = np.where(
                free, np.nanpercentile(arr_p, 100 * alpha / 2, axis=0), np.nan
            )
            ci_high = np.where(
                free, np.nanpercentile(arr_p, 100 * (1 - alpha / 2), axis=0), np.nan
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                corr_b = np.corrcoef(arr_q[:, free], rowvar=False)
            corr = np.full((k_all, k_all), np.nan)
            corr[np.ix_(free, free)] = np.atleast_2d(corr_b)
            for name in derived:
                values = np.array([d[name] for d in rows_d], dtype=np.float64)
                derived_se[name] = float(np.nanstd(values, ddof=1))
                derived_lo[name] = float(np.nanpercentile(values, 100 * alpha / 2))
                derived_hi[name] = float(
                    np.nanpercentile(values, 100 * (1 - alpha / 2))
                )
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
    args: tuple[Model, np.ndarray, np.ndarray, np.ndarray | None, FitOptions, int],
) -> RowFit:
    """Worker entry point for `fit_rows`: fit one row from its arguments and seed.

    A module-level function so it can be pickled for a `ProcessPoolExecutor`.

    Args:
        args: `(model, x, y, sd, options, seed)` of one row.

    Returns:
        The fit of the row.
    """
    model, x, y, sd, options, seed = args
    return fit_row(model, x, y, sd, options, np.random.default_rng(int(seed)))


def fit_rows(
    model: Model,
    x: np.ndarray,
    y: np.ndarray,
    sd: np.ndarray | None,
    options: FitOptions,
) -> list[RowFit]:
    """Fit every row of `(N, n)` arrays, serially or in a process pool.

    Row seeds are drawn from `options.seed` before the rows are distributed,
    one child seed per row, so the result does not depend on `options.n_workers`
    or the order the rows finish in.

    Args:
        model: the model
        x: independent variable `(N, n)`
        y: dependent variable `(N, n)`
        sd: standard deviations `(N, n)` or `None`
        options: the options

    Returns:
        One `RowFit` per row, in row order.
    """
    rng = np.random.default_rng(options.seed)
    seeds = rng.integers(0, 2**32 - 1, size=y.shape[0])
    jobs = [
        (model, x[i], y[i], None if sd is None else sd[i], options, int(seeds[i]))
        for i in range(y.shape[0])
    ]
    if options.n_workers is not None and options.n_workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=options.n_workers) as pool:
            return list(pool.map(_fit_row_job, jobs))
    return [_fit_row_job(job) for job in jobs]


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


def fit(
    model: Model,
    x: Any,
    y: Any,
    *,
    sd: Any | None = None,
    options: FitOptions | None = None,
    x_unit: str = "dimensionless",
    y_unit: str = "dimensionless",
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
        dims: sample dimension names for a 2-D `y`, `("sample",)` by default
        coords: coordinate labels of the sample dimensions

    Returns:
        The result over the sample dimensions (none for a 1-D `y`).

    Raises:
        ValueError: for 2-D data with more than one sample dimension, or if
            the sample dimension name collides with a variable the result
            writes or a reserved dimension (see `build_result`).
    """
    options = options or FitOptions()
    x_arr, y_arr, sd_arr, single = _as_rows(x, y, sd)
    sample_dims: tuple[str, ...] = () if single else tuple(dims or ("sample",))
    if not single and len(sample_dims) != 1:
        raise ValueError(
            "2-D data has exactly one sample dimension; use fit_timecourses or fit_table for more"
        )
    rows = fit_rows(model, x_arr, y_arr, sd_arr, options)
    return build_result(
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


def _cv(se: float, value: float) -> float:
    """The coefficient of variation in percent, `NaN` for a zero or missing value.

    Args:
        se: the standard error.
        value: the estimate.

    Returns:
        `100 se / |value|`.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(100.0 * np.float64(se) / np.abs(np.float64(value)))


#: dimension names `build_result` reserves for the parameter and point axes,
#: regardless of the model
_RESERVED_DIMS: frozenset[str] = frozenset({"point", "parameter", "parameter_"})


def _check_no_dimension_collision(model: Model, dims: tuple[str, ...]) -> None:
    """Reject a sample dimension whose name collides with a variable `build_result` writes.

    A sample dimension cannot share its name with a data variable of the
    result (e.g. a scan dimension "k" fitted with a model that also has a
    rate constant "k") or with one of the reserved dimensions `parameter`,
    `parameter_` and `point`: xarray cannot hold two different variables
    under the same name, so the name must be one or the other.

    Args:
        model: the model, for its parameter and derived names.
        dims: the sample dimension names.

    Raises:
        ValueError: if a name in `dims` collides with a written variable or a reserved dimension.
    """
    written = {
        "cost",
        "r2",
        "rmse",
        "aic",
        "aicc",
        "bic",
        "n_points",
        "n_parameters",
        "n_starts_converged",
        "n_bootstrap",
        "x_data",
        "y_data",
        "y_pred",
        "residuals",
        "correlation",
        "flags",
    }
    for parameter in model.parameters:
        written.update(
            {
                parameter.name,
                f"{parameter.name}_se",
                f"{parameter.name}_ci_low",
                f"{parameter.name}_ci_high",
                f"{parameter.name}_cv",
            }
        )
    for derived_name in model.derived_units:
        written.add(derived_name)
        if derived_name not in FitResult.discrete_parameters:
            written.update(
                {
                    f"{derived_name}_se",
                    f"{derived_name}_ci_low",
                    f"{derived_name}_ci_high",
                    f"{derived_name}_cv",
                }
            )
    for d in dims:
        if d in written or d in _RESERVED_DIMS:
            raise ValueError(
                f"the sample dimension '{d}' collides with the result variable "
                f"'{d}'; rename the dimension"
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
        sd: `(N, n)` standard deviations or `None` (reported by a later task)
        x_unit: unit of `x`
        y_unit: unit of `y`
        dims: sample dimension names (`()` for a single row)
        coords: coordinates of the sample dimensions
        options: the options (stored in `attrs`)
        shape: sample shape for several sample dimensions (`N = prod(shape)`), `(N,)` by default

    Returns:
        The `FitResult`.

    Raises:
        ValueError: if a name in `dims` collides with a variable the result
            writes or a reserved dimension (`_check_no_dimension_collision`).
    """
    n_rows, n_points = y.shape
    sample_shape: tuple[int, ...] = (
        () if not dims else (shape if shape is not None else (n_rows,))
    )
    names = model.parameter_names
    k = len(names)
    _check_no_dimension_collision(model, dims)

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
