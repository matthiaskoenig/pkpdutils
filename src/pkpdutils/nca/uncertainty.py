"""Uncertainty of the NCA parameters of group timecourses.

A group timecourse is the mean curve of several subjects with the standard
deviation (`sd`) or the standard error (`se`) per time point and the number of
subjects `n`. Two methods propagate this uncertainty to the parameters:

- the parametric **bootstrap** (Efron & Tibshirani 1993, ch. 6) draws every
  time point of every curve `n_boot` times from a normal (or log-normal)
  distribution with the observed mean and spread, analyses the replicates with
  the same vectorized code as the original curves and reduces them to the
  standard error (or standard deviation), the percentile confidence interval
  and, for log-normal parameters, the geometric mean and geometric CV;
- the **delta method** perturbs every time point once, forms the numerical
  Jacobian of every parameter with respect to the values and propagates the
  standard errors through it: `var(x) = sum_i (dx/dC_i)^2 se_i^2`.

The variables of a parameter `x` are `x_sd`, `x_se`, `x_ci_low`, `x_ci_high`
and, for log-normal parameters, `x_geomean`, `x_geocv`; `n` is the number of
subjects per sample. `x_sd` and `x_geocv` are always on the between-subject
scale (the spread of the parameter over subjects), `x_se` is always the
uncertainty of the parameter of the mean curve, and `x_ci_low`/`x_ci_high` are
always an interval of that estimate. Discrete parameters (`tmax`, `tlast`,
counts) and the diagnostics of the terminal regression carry no uncertainty.

With `BootstrapSpread.SD` the replicates are individual curves, so their
percentiles bound individuals and not the estimate; they are exported
separately as `x_pi_low`/`x_pi_high` and the confidence interval is the normal
approximation `x +- z se` (on the log scale for log-normal parameters).

Under `BootstrapSpread.SE` the interval is the percentile interval of the
replicates, so it is not guaranteed to contain the point estimate of the mean
curve: for skewed replicates (a parameter which is a strongly non-linear
function of the values, such as `lambda_z` or `mrt`) the interval is asymmetric
around the estimate and can exclude it.
"""

import logging
import warnings

import numpy as np
from scipy.stats import norm

from pkpdutils.nca.options import (
    BootstrapDistribution,
    BootstrapSpread,
    Kind,
    NCAFlag,
    NCAOptions,
)
from pkpdutils.parallel import resolve_workers
from pkpdutils.result import base_name, nan_percentile
from pkpdutils.timecourse import Timecourses

logger = logging.getLogger(__name__)

#: parameters reported with geometric statistics (positive, log-normal)
LOGNORMAL_PARAMETERS: frozenset[str] = frozenset(
    {
        "auc_last",
        "auc_all",
        "auc_inf_obs",
        "auc_inf_pred",
        "aumc_last",
        "aumc_all",
        "aumc_inf",
        "auc_tau",
        "cmax",
        "cmax_ss",
        "cmin",
        "cmin_ss",
        "ctrough",
        "cavg",
        "clast",
        "c0",
        "cl",
        "cl_f",
        "cl_ss",
        "cl_ss_f",
        "vz",
        "vz_f",
        "vss",
        "thalf",
        "lambda_z",
        "mrt",
        "clast_pred",
        "auc_inf_dn",
        "cmax_dn",
        "auc_last_dn",
        "auc_all_dn",
        "auc_tau_dn",
        "cavg_dn",
        "cmax_ss_dn",
        "c0_dn",
        "accumulation_ratio_obs",
        "auec_tau",
        "eavg",
    }
)

#: parameters without uncertainty (observed points, counts and regression diagnostics)
DISCRETE_PARAMETERS: frozenset[str] = frozenset(
    {
        "tmax",
        "tmin",
        "tlast",
        "tlag",
        "tmax_half",
        "temax",
        # the rule which produced `c0`, an integer code and not a measurement
        "c0_method",
        "lambda_z_n_points",
        "lambda_z_t_first",
        "lambda_z_t_last",
        "lambda_z_span",
        "lambda_z_intercept",
        "lambda_z_r2",
        "lambda_z_r2_adj",
        "lambda_z_stderr",
        "flags",
        "n",
        "n_doses",
        "tau",
        "interval_n_points",
        # the observed times and the bounds of a dosing interval: an interval
        # starts and ends where the protocol says, so a standard error or a
        # coefficient of variation of them is not a quantity either
        "interval_tmax",
        "interval_temax",
        "interval_start",
        "interval_end",
        "interval_dose",
        # the Satterthwaite degrees of freedom of the area of a sparse design
        # (`pkpdutils.nca.sparse`), a property of that design rather than a
        # measurement with a spread of its own
        "auc_last_df",
    }
)

#: parameters which do not depend on the terminal phase (observed points and
#: the areas to the last point); every other continuous parameter is affected
#: by a change of the terminal window, see `delta`
TERMINAL_INDEPENDENT_PARAMETERS: frozenset[str] = frozenset(
    {
        "cmax",
        "cmin",
        "clast",
        "c0",
        "cmax_half",
        "auc_last",
        "auc_all",
        "aumc_last",
        "aumc_all",
        "auc_tau",
        "cmin_ss",
        "cmax_ss",
        "ctrough",
        "cavg",
        "e0",
        "emax_obs",
        "auec_last",
        "auec_baseline",
        "emax_baseline",
        "time_above",
    }
)


def flatten_rows(a: np.ndarray | None, n_rows: int) -> np.ndarray | None:
    """A per row dose array of shape `(*sample_shape, n_dose)` as `(N, n_dose)`.

    Args:
        a: the array, or `None`
        n_rows: number of rows `N` of the batch

    Returns:
        The flattened array `(N, n_dose)`, or `None`.
    """
    if a is None:
        return None
    return np.asarray(a, dtype=np.float64).reshape(n_rows, -1)


def repeat_block(
    a: np.ndarray | None, start: int, stop: int, repeats: int
) -> np.ndarray | None:
    """Repeat every row of a block of a flattened dose array, keeping the rows grouped.

    Args:
        a: the flattened array `(N, n_dose)` (`flatten_rows`), or `None`
        start: first row of the block
        stop: row after the last one of the block
        repeats: copies per row

    Returns:
        The repeated block `((stop - start) * repeats, n_dose)`, or `None`.
    """
    if a is None:
        return None
    return np.repeat(a[start:stop], repeats, axis=0)


def repeat_rows(a: np.ndarray | None, n_rows: int, repeats: int) -> np.ndarray | None:
    """Repeat every row of a per row dose array, keeping the rows grouped.

    The replicates of the bootstrap and the perturbed curves of the delta
    method are `repeats` copies of every row of the batch, in blocks; the dose
    arrays follow them row by row.

    Args:
        a: the array of shape `(*sample_shape, n_dose)`, or `None`
        n_rows: number of rows `N` of the batch
        repeats: copies per row

    Returns:
        The repeated array `(N * repeats, n_dose)`, or `None`.
    """
    return repeat_block(flatten_rows(a, n_rows), 0, n_rows, repeats)


def _repeat_windows(windows: np.ndarray | None, repeats: int) -> np.ndarray | None:
    """Repeat every terminal window of a batch once per replicate of its row.

    Args:
        windows: the windows `(N, 2)`, or `None`
        repeats: copies per row

    Returns:
        The repeated windows `(N * repeats, 2)`, or `None`.
    """
    return None if windows is None else np.repeat(windows, repeats, axis=0)


def resolve_spread(
    timecourses: Timecourses,
    options: NCAOptions,
    *,
    spread: BootstrapSpread | None = None,
) -> np.ndarray:
    """The spread every time point is resampled with, `(n_samples, n_time)`.

    Args:
        timecourses: the batch
        options: `bootstrap_spread` selects `se` or `sd` when `spread` is not
            given; the missing one is derived from the other with `n`

    Keyword Args:
        spread: the spread to return, overriding `options.bootstrap_spread`;
            the delta method always propagates `se`, whatever the options say

    Returns:
        The spread per point (`NaN` where the batch has none).

    Raises:
        ValueError: if the requested spread is neither present nor derivable.
    """
    kind = spread if spread is not None else options.bootstrap_spread
    n_rows, n_time = timecourses.n_samples, timecourses.n_time
    se = None if timecourses.se is None else timecourses.se.reshape(n_rows, n_time)
    sd = None if timecourses.sd is None else timecourses.sd.reshape(n_rows, n_time)
    # one count per row, or one per row and time point: both convert the
    # spread of a point, the second one with the count of that point
    n = (
        None
        if timecourses.n is None
        else np.asarray(timecourses.n, dtype=np.float64).reshape(n_rows, -1)
    )
    if kind is BootstrapSpread.SE:
        if se is not None:
            return se
        if sd is not None and n is not None:
            return sd / np.sqrt(n)
        raise ValueError(
            "The batch has no 'se' and it cannot be derived ('sd' and 'n' needed)"
        )
    if sd is not None:
        return sd
    if se is not None and n is not None:
        return se * np.sqrt(n)
    raise ValueError(
        "The batch has no 'sd' and it cannot be derived ('se' and 'n' needed)"
    )


def resample_values(
    c: np.ndarray,
    spread: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    distribution: BootstrapDistribution,
    *,
    clip_at_zero: bool = True,
) -> np.ndarray:
    """Draw bootstrap replicates of every time point of every row.

    A normal draw is `C_i + s_i z`, clipped at 0 for concentrations; a
    log-normal draw has the same mean and spread,
    `sigma^2 = ln(1 + s_i^2 / C_i^2)` and `mu = ln C_i - sigma^2 / 2`
    (Efron & Tibshirani 1993, ch. 6). A log-normal point whose mean is not
    positive has no such distribution and is copied unchanged.

    Args:
        c: values `(N, n)`
        spread: spread per point `(N, n)`; a point without a finite positive
            spread is copied
        n_boot: number of replicates `B`
        rng: random generator
        distribution: normal or log-normal with the same mean and spread
        clip_at_zero: whether normal draws below 0 are set to 0; `True` for
            concentrations, which cannot be negative, and `False` for effects,
            whose values are legitimately negative. Clipping biases the mean of
            a point whose spread is large against its value upwards, see
            `BootstrapDistribution.LOGNORMAL` for an alternative.

    Returns:
        The replicates `(N, B, n)`.
    """
    n_rows, n_time = c.shape
    z = rng.standard_normal((n_rows, n_boot, n_time))
    mean = c[:, None, :]
    s = spread[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        usable = np.isfinite(s) & (s > 0) & np.isfinite(mean)
        if distribution is BootstrapDistribution.NORMAL:
            normal = mean + s * z
            draws = np.where(
                usable, np.maximum(normal, 0.0) if clip_at_zero else normal, mean
            )
        else:
            positive = usable & (mean > 0)
            sigma2 = np.log1p((s / mean) ** 2)
            mu = np.log(mean) - sigma2 / 2.0
            draws = np.where(positive, np.exp(mu + np.sqrt(sigma2) * z), mean)
    return draws


def reduce_replicates(
    replicates: dict[str, np.ndarray],
    point: dict[str, np.ndarray],
    *,
    spread_kind: BootstrapSpread,
    n_subjects: np.ndarray | None,
    ci_level: float,
    usable_rows: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Reduce the bootstrap replicates of every continuous parameter to its uncertainty variables.

    The spread of the replicates is the standard error of the parameter when
    the points were drawn with `se` and its standard deviation over subjects
    when they were drawn with `sd`; the other one follows from
    `se = sd / sqrt(n)`.

    `x_ci_low`/`x_ci_high` are always an interval of the estimate: the
    percentile interval of the replicates under `se` draws, which is asymmetric
    around the estimate of the mean curve for skewed replicates and is not
    guaranteed to contain it, and the normal approximation `x +- z se` (on the
    log scale, `x exp(+- z se / x)`, for log-normal parameters) under `sd`
    draws, whose replicates are individual curves. Their percentiles are
    exported separately as `x_pi_low`/`x_pi_high`, the interval of the
    individuals.

    `x_geocv` is, like `x_sd`, always the geometric CV over subjects,
    `geocv = sqrt(exp(var(ln x)) - 1)` (Efron & Tibshirani 1993, ch. 13): from
    `sd` draws the log variance of the replicates is used as it is, from `se`
    draws it is the log variance of the mean and is scaled by `n` first
    (`NaN` without `n`). `x_geomean` comes from the logarithms of the positive
    replicates.

    Args:
        replicates: parameter name to replicates `(N, B)`
        point: parameter name to the estimate of the original curve `(N,)`
        spread_kind: whether the replicates were drawn with `se` (their spread
            is the standard error of the parameter) or `sd` (their spread is
            the standard deviation over subjects)
        n_subjects: subjects per row, `None` or `NaN` when unknown
        ci_level: level of the confidence interval
        usable_rows: rows which carried a usable spread, `None` for all of
            them; a row without one has no uncertainty and is `NaN` everywhere

    Returns:
        `x_sd`, `x_se`, `x_ci_low`, `x_ci_high` per parameter, `x_pi_low`,
        `x_pi_high` under `sd` draws and `x_geomean`, `x_geocv` for log-normal
        parameters.
    """
    alpha = 1.0 - ci_level
    z = float(norm.ppf(1.0 - alpha / 2.0))
    out: dict[str, np.ndarray] = {}
    n_values = None if n_subjects is None else np.asarray(n_subjects, dtype=np.float64)
    sqrt_n = None if n_values is None else np.sqrt(n_values)
    for name, reps in replicates.items():
        skip = (
            name in DISCRETE_PARAMETERS
            or base_name(name) is not None
            or name not in point
        )
        if skip:
            continue
        # the nan-aware reductions warn on rows without a single finite
        # replicate ("All-NaN slice", "Degrees of freedom <= 0"); such rows are
        # masked by `valid` below, so the warnings carry no information
        with (
            np.errstate(invalid="ignore", divide="ignore", over="ignore"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", RuntimeWarning)
            finite = np.isfinite(reps)
            count = finite.sum(axis=1)
            filled = np.where(finite, reps, np.nan)
            std = np.nanstd(filled, axis=1, ddof=1)
            low, high = nan_percentile(
                filled, (100 * alpha / 2, 100 * (1 - alpha / 2)), axis=1
            )
            if spread_kind is BootstrapSpread.SE:
                se = std
                sd = std * sqrt_n if sqrt_n is not None else np.full_like(std, np.nan)
            else:
                sd = std
                se = std / sqrt_n if sqrt_n is not None else np.full_like(std, np.nan)
            estimate = point[name]
            valid = np.isfinite(estimate) & (count > 1)
            if usable_rows is not None:
                valid = valid & usable_rows
            if spread_kind is BootstrapSpread.SD:
                # the replicates are individual curves: their percentiles bound
                # the individuals, the interval of the estimate is the normal
                # approximation around it
                out[f"{name}_pi_low"] = np.where(valid, low, np.nan)
                out[f"{name}_pi_high"] = np.where(valid, high, np.nan)
                if name in LOGNORMAL_PARAMETERS:
                    rel = se / estimate
                    low = estimate * np.exp(-z * rel)
                    high = estimate * np.exp(z * rel)
                else:
                    low = estimate - z * se
                    high = estimate + z * se
            out[f"{name}_sd"] = np.where(valid, sd, np.nan)
            out[f"{name}_se"] = np.where(valid, se, np.nan)
            out[f"{name}_ci_low"] = np.where(valid, low, np.nan)
            out[f"{name}_ci_high"] = np.where(valid, high, np.nan)
            if name in LOGNORMAL_PARAMETERS:
                positive = finite & (reps > 0)
                logs = np.where(positive, np.log(np.where(positive, reps, 1.0)), np.nan)
                n_positive = positive.sum(axis=1)
                log_var = np.nanvar(logs, axis=1, ddof=1)
                if spread_kind is BootstrapSpread.SE:
                    # the log variance of the mean curve, scaled to the between
                    # subject variance; without `n` the scale is unknown
                    log_var = (
                        log_var * n_values
                        if n_values is not None
                        else np.full_like(log_var, np.nan)
                    )
                geometric = valid & (n_positive > 1)
                out[f"{name}_geomean"] = np.where(
                    geometric, np.exp(np.nanmean(logs, axis=1)), np.nan
                )
                out[f"{name}_geocv"] = np.where(
                    geometric, np.sqrt(np.expm1(log_var)), np.nan
                )
    return out


def bootstrap(
    timecourses: Timecourses,
    options: NCAOptions,
    point: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Bootstrap the parameters of a batch.

    The curves are processed in blocks: the replicates of a block are drawn,
    analysed and reduced to their parameters before the next block is drawn, so
    the `(N, B, n)` array of every replicate of the batch never exists at once.
    The draws do not depend on the blocking, the random generator produces the
    same numbers in the same order.

    Args:
        timecourses: the batch (group curves with `sd` or `se`)
        options: `n_boot`, `seed`, `bootstrap_spread`, `bootstrap_distribution`, `ci_level`
        point: the parameters of the original curves (`run_rows` output)

    Returns:
        The uncertainty variables per parameter (see `reduce_replicates`).

    Raises:
        ValueError: if the batch has no spread to resample with, or if
            log-normal draws are requested for effect timecourses.
    """
    # the analysis of the replicates runs through the same core as the original
    # curves, whose module imports this one
    from pkpdutils.nca.nca import chunk_bounds, merge_rows, run_rows, sample_windows

    if (
        options.kind is Kind.EFFECT
        and options.bootstrap_distribution is BootstrapDistribution.LOGNORMAL
    ):
        raise ValueError(
            "log-normal draws need positive values, effect timecourses use NORMAL"
        )
    n_rows, n_time = timecourses.n_samples, timecourses.n_time
    t = timecourses.times.reshape(n_rows, n_time)
    c = timecourses.values.reshape(n_rows, n_time)
    spread = resolve_spread(timecourses, options)
    with np.errstate(invalid="ignore"):
        any_usable = (np.isfinite(spread) & (spread > 0)).any(axis=1)
    rng = np.random.default_rng(options.seed)
    b = options.n_boot
    logger.info("bootstrap: %d curves x %d replicates", n_rows, b)

    # the replicates are materialized block of curves by block of curves and
    # only their parameters are kept, so the `(N, B, n)` array of every
    # replicate of the batch never exists at once; a block carries as many
    # replicate rows as the analysis would process at once anyway, so the core
    # sees the same chunks (`chunk_rows` per worker) as an unblocked run and
    # the draws, which are generated row block after row block from the same
    # generator, are the same numbers. The worker count of the replicate rows
    # decides the size of a block, so that an automatic run
    # (`options.n_workers is None`) fills every thread of `run_rows` too.
    workers = resolve_workers(options.n_workers, n_rows * b)
    block_rows = max(1, (options.chunk_rows * workers) // b)
    n_blocks = max(1, -(-n_rows // block_rows))
    dose_amount = flatten_rows(timecourses.dose_amount, n_rows)
    dose_time = flatten_rows(timecourses.dose_time, n_rows)
    dose_duration = flatten_rows(timecourses.dose_duration, n_rows)
    batch_lloq = timecourses.lloq
    lloq = None if batch_lloq is None else batch_lloq.reshape(n_rows)
    windows = sample_windows(timecourses, options.terminal)
    parts: list[dict[str, np.ndarray]] = []
    counts: list[int] = []
    for start, stop in chunk_bounds(n_rows, n_blocks):
        rows = stop - start
        draws = resample_values(
            c[start:stop],
            spread[start:stop],
            b,
            rng,
            options.bootstrap_distribution,
            clip_at_zero=options.kind is Kind.CONCENTRATION,
        )
        parts.append(
            run_rows(
                np.repeat(t[start:stop], b, axis=0),
                draws.reshape(rows * b, n_time),
                dose_amount=repeat_block(dose_amount, start, stop, b),
                dose_time=repeat_block(dose_time, start, stop, b),
                dose_duration=repeat_block(dose_duration, start, stop, b),
                route=timecourses.route,
                options=options,
                lloq=None if lloq is None else np.repeat(lloq[start:stop], b),
                windows=_repeat_windows(
                    None if windows is None else windows[start:stop], b
                ),
            )
        )
        counts.append(rows * b)
    values = merge_rows(parts, counts)
    # the per-interval parameters carry an extra dimension and are no
    # parameters of a sample: they are left to the point estimate
    replicates = {
        name: array.reshape(n_rows, b)
        for name, array in values.items()
        if array.ndim == 1
    }
    n_subjects = (
        None
        if timecourses.n_subjects is None
        else np.asarray(timecourses.n_subjects, dtype=np.float64).reshape(n_rows)
    )
    return reduce_replicates(
        replicates,
        point,
        spread_kind=options.bootstrap_spread,
        n_subjects=n_subjects,
        ci_level=options.ci_level,
        usable_rows=any_usable,
    )


def delta(
    timecourses: Timecourses,
    options: NCAOptions,
    point: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Delta method: propagate the standard errors of the points through the numerical Jacobian.

    `var(x) = sum_i (dx/dC_i)^2 se_i^2` with `dx/dC_i` from a forward difference
    of step `options.delta_step * se_i` (Efron & Tibshirani 1993, ch. 5). The
    method always propagates `se`, the uncertainty of the mean curve, so
    `options.bootstrap_spread` does not apply. The interval is the normal
    interval `x +- z se`, on the log scale for log-normal parameters; discrete
    parameters and the diagnostics of the terminal regression are skipped.

    `x_sd = x_se sqrt(n)` is the spread of the parameter over subjects and
    `x_geocv` is its geometric CV, as in the bootstrap: with `mu = x` and
    `sd = x_sd`, the log-normal moment relation gives
    `sigma_log² = ln(1 + (sd / mu)²)` and

    `geocv = sqrt(exp(sigma_log²) - 1) = sd / mu`,

    so the geometric CV equals the arithmetic CV over subjects (Efron &
    Tibshirani 1993, ch. 13). Without `n` the between-subject scale is unknown
    and `x_sd` and `x_geocv` are `NaN`.

    A perturbation which selects a different terminal window
    (`lambda_z_n_points` or `lambda_z_t_first` changes) makes the difference
    quotient a jump between two regressions instead of a derivative, which
    inflates the standard error of every terminal parameter. Such points are
    skipped for every parameter outside `TERMINAL_INDEPENDENT_PARAMETERS` and
    the row carries `NCAFlag.DELTA_WINDOW_CHANGE`, which says that the
    uncertainty of its terminal parameters is incomplete; use the bootstrap,
    which follows the window, for those rows.

    Args:
        timecourses: the batch (group curves with `se`, or `sd` and `n`)
        options: `delta_step`, `ci_level`
        point: the parameters of the original curves (`run_rows` output)

    Returns:
        The uncertainty variables per continuous parameter and, under the key
        `"flags"`, the flags of the skipped points to be combined with the
        flags of the analysis.

    Raises:
        ValueError: if the batch has no `se` and it cannot be derived.
    """
    # the analysis of the perturbed curves runs through the same core as the
    # original curves, whose module imports this one
    from pkpdutils.nca.nca import run_rows, sample_windows

    n_rows, n_time = timecourses.n_samples, timecourses.n_time
    t = timecourses.times.reshape(n_rows, n_time)
    c = timecourses.values.reshape(n_rows, n_time)
    se = resolve_spread(timecourses, options, spread=BootstrapSpread.SE)
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(se) & (se > 0) & np.isfinite(c)
    h = np.where(usable, options.delta_step * se, 0.0)

    # every row is repeated `n_time` times, the j-th copy perturbed at point j
    rows = np.arange(n_rows * n_time)
    cols = np.tile(np.arange(n_time), n_rows)
    c_pert = np.repeat(c, n_time, axis=0)
    c_pert[rows, cols] += np.repeat(h, n_time, axis=0)[rows, cols]

    logger.info("delta method: %d curves x %d perturbations", n_rows, n_time)
    perturbed = run_rows(
        np.repeat(t, n_time, axis=0),
        c_pert,
        dose_amount=repeat_rows(timecourses.dose_amount, n_rows, n_time),
        dose_time=repeat_rows(timecourses.dose_time, n_rows, n_time),
        dose_duration=repeat_rows(timecourses.dose_duration, n_rows, n_time),
        route=timecourses.route,
        options=options,
        lloq=(
            None
            if timecourses.lloq is None
            else np.repeat(timecourses.lloq.reshape(n_rows), n_time)
        ),
        windows=_repeat_windows(sample_windows(timecourses, options.terminal), n_time),
    )
    alpha = 1.0 - options.ci_level
    z = float(norm.ppf(1.0 - alpha / 2.0))
    n_subjects = (
        None
        if timecourses.n_subjects is None
        else np.asarray(timecourses.n_subjects, dtype=np.float64).reshape(n_rows)
    )
    any_usable = usable.any(axis=1)
    step = np.where(usable, h, 1.0)
    weight = np.where(usable, se, 0.0)
    window_changed = _window_changed(point, perturbed, n_rows, n_time) & usable
    flags = np.where(
        window_changed.any(axis=1), int(NCAFlag.DELTA_WINDOW_CHANGE), 0
    ).astype(np.int64)
    out: dict[str, np.ndarray] = {}
    for name, base in point.items():
        skip = (
            name in DISCRETE_PARAMETERS
            or base_name(name) is not None
            or name not in perturbed
            # the per-interval parameters carry an extra dimension
            or base.ndim > 1
        )
        if skip:
            continue
        pert = perturbed[name].reshape(n_rows, n_time)
        # a point at which the terminal window flipped carries no derivative of
        # the parameters which depend on that window
        keep = (
            usable
            if name in TERMINAL_INDEPENDENT_PARAMETERS
            else usable & ~window_changed
        )
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            derivative = np.where(keep, (pert - base[:, None]) / step, 0.0)
            var = np.sum((derivative * weight) ** 2, axis=1)
            valid = np.isfinite(base) & np.isfinite(var) & any_usable
            x_se = np.where(valid, np.sqrt(np.where(valid, var, 0.0)), np.nan)
            x_sd = (
                x_se * np.sqrt(n_subjects)
                if n_subjects is not None
                else np.full_like(x_se, np.nan)
            )
            if name in LOGNORMAL_PARAMETERS:
                rel = x_se / base
                low = base * np.exp(-z * rel)
                high = base * np.exp(z * rel)
                # the geometric CV is the spread over subjects, as in the
                # bootstrap: with mu = x and sd = x_sd the log-normal moment
                # relation gives sigma_log² = ln(1 + (sd/mu)²), so
                # geocv = sqrt(exp(sigma_log²) - 1) = |sd/mu|, the arithmetic CV
                out[f"{name}_geomean"] = np.where(valid, base, np.nan)
                out[f"{name}_geocv"] = np.where(valid, np.abs(x_sd / base), np.nan)
            else:
                low = base - z * x_se
                high = base + z * x_se
        out[f"{name}_sd"] = x_sd
        out[f"{name}_se"] = x_se
        out[f"{name}_ci_low"] = np.where(valid, low, np.nan)
        out[f"{name}_ci_high"] = np.where(valid, high, np.nan)
    out["flags"] = flags
    return out


def _window_changed(
    point: dict[str, np.ndarray],
    perturbed: dict[str, np.ndarray],
    n_rows: int,
    n_time: int,
) -> np.ndarray:
    """Points at which a perturbation selected a different terminal window.

    Args:
        point: the parameters of the original curves
        perturbed: the parameters of the perturbed curves, `(N * n,)` per name
        n_rows: number of curves
        n_time: number of time points per curve

    Returns:
        A boolean `(N, n)` mask, all `False` without a terminal regression
        (effect timecourses).
    """
    changed = np.zeros((n_rows, n_time), dtype=bool)
    for name in ("lambda_z_n_points", "lambda_z_t_first"):
        if name not in point or name not in perturbed:
            continue
        base = np.asarray(point[name], dtype=np.float64).reshape(n_rows, 1)
        pert = np.asarray(perturbed[name], dtype=np.float64).reshape(n_rows, n_time)
        # two missing windows are equal, a missing and a present one are not
        both_nan = np.isnan(base) & np.isnan(pert)
        with np.errstate(invalid="ignore"):
            changed |= ~(both_nan | (base == pert))
    return changed
