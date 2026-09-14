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
subjects per sample. Discrete parameters (`tmax`, `tlast`, counts) and the
diagnostics of the terminal regression carry no uncertainty.

The interval of the bootstrap is the percentile interval of the replicates, so
it is not guaranteed to contain the point estimate of the mean curve: for
skewed replicates (a parameter which is a strongly non-linear function of the
values, such as `lambda_z` or `mrt`) the interval is asymmetric around the
estimate and can exclude it.
"""

import logging
import warnings

import numpy as np

from pkpdutils.nca.options import BootstrapDistribution, BootstrapSpread, NCAOptions
from pkpdutils.timecourse import Timecourses

logger = logging.getLogger(__name__)

#: parameters reported with geometric statistics (positive, log-normal)
LOGNORMAL_PARAMETERS: frozenset[str] = frozenset(
    {
        "auc_last",
        "auc_inf_obs",
        "auc_inf_pred",
        "aumc_last",
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
        "vz",
        "vz_f",
        "vss",
        "thalf",
        "lambda_z",
        "mrt",
        "auc_inf_dn",
        "cmax_dn",
    }
)

#: parameters without uncertainty (observed points, counts and regression diagnostics)
DISCRETE_PARAMETERS: frozenset[str] = frozenset(
    {
        "tmax",
        "tmin",
        "tlast",
        "tmax_half",
        "temax",
        "lambda_z_n_points",
        "lambda_z_t_first",
        "lambda_z_intercept",
        "lambda_z_r2",
        "lambda_z_r2_adj",
        "lambda_z_stderr",
        "flags",
        "n",
    }
)

#: suffixes of the uncertainty variables of a parameter
UNCERTAINTY_SUFFIXES: tuple[str, ...] = (
    "_sd",
    "_se",
    "_ci_low",
    "_ci_high",
    "_geomean",
    "_geocv",
)

#: suffixes of the summary variables of a parameter (`NCAResult.summarize`)
SUMMARY_SUFFIXES: tuple[str, ...] = ("_median", "_q25", "_q75", "_n")


def base_name(name: str) -> str | None:
    """The parameter a derived variable belongs to, `None` for a parameter itself.

    Args:
        name: name of a result variable, e.g. `"auc_last_se"`.

    Returns:
        The name of the parameter the variable is derived from, `None` for a
        parameter.
    """
    for suffix in (*UNCERTAINTY_SUFFIXES, *SUMMARY_SUFFIXES):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return None


def resolve_spread(timecourses: Timecourses, options: NCAOptions) -> np.ndarray:
    """The spread every time point is resampled with, `(n_samples, n_time)`.

    Args:
        timecourses: the batch
        options: `bootstrap_spread` selects `se` or `sd`; the missing one is
            derived from the other with `n`

    Returns:
        The spread per point (`NaN` where the batch has none).

    Raises:
        ValueError: if the requested spread is neither present nor derivable.
    """
    n_rows, n_time = timecourses.n_samples, timecourses.n_time
    se = None if timecourses.se is None else timecourses.se.reshape(n_rows, n_time)
    sd = None if timecourses.sd is None else timecourses.sd.reshape(n_rows, n_time)
    n = (
        None
        if timecourses.n is None
        else np.asarray(timecourses.n, dtype=np.float64).reshape(n_rows)[:, None]
    )
    if options.bootstrap_spread is BootstrapSpread.SE:
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
) -> np.ndarray:
    """Draw bootstrap replicates of every time point of every row.

    A normal draw is `C_i + s_i z`, clipped at 0; a log-normal draw has the
    same mean and spread, `sigma^2 = ln(1 + s_i^2 / C_i^2)` and
    `mu = ln C_i - sigma^2 / 2` (Efron & Tibshirani 1993, ch. 6).

    Args:
        c: values `(N, n)`
        spread: spread per point `(N, n)`; a point without a finite positive
            spread is copied
        n_boot: number of replicates `B`
        rng: random generator
        distribution: normal (draws below 0 set to 0) or log-normal with the
            same mean and spread

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
            draws = np.where(usable, np.maximum(mean + s * z, 0.0), mean)
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
) -> dict[str, np.ndarray]:
    """Reduce the bootstrap replicates of every continuous parameter to its uncertainty variables.

    The spread of the replicates is the standard error of the parameter when
    the points were drawn with `se` and its standard deviation over subjects
    when they were drawn with `sd`; the other one follows from
    `se = sd / sqrt(n)`. The interval is the percentile interval at `ci_level`,
    which is asymmetric around the estimate of the mean curve for skewed
    replicates and is not guaranteed to contain it; the geometric statistics
    come from the logarithms of the positive replicates,
    `geocv = sqrt(exp(var(ln x)) - 1)` (Efron & Tibshirani 1993, ch. 13).

    Args:
        replicates: parameter name to replicates `(N, B)`
        point: parameter name to the estimate of the original curve `(N,)`
        spread_kind: whether the replicates were drawn with `se` (their spread
            is the standard error of the parameter) or `sd` (their spread is
            the standard deviation over subjects)
        n_subjects: subjects per row, `None` or `NaN` when unknown
        ci_level: level of the percentile interval

    Returns:
        `x_sd`, `x_se`, `x_ci_low`, `x_ci_high` per parameter and `x_geomean`,
        `x_geocv` for log-normal parameters.
    """
    alpha = 1.0 - ci_level
    out: dict[str, np.ndarray] = {}
    sqrt_n = (
        None
        if n_subjects is None
        else np.sqrt(np.asarray(n_subjects, dtype=np.float64))
    )
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
            low, high = np.nanpercentile(
                filled, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=1
            )
            if spread_kind is BootstrapSpread.SE:
                se = std
                sd = std * sqrt_n if sqrt_n is not None else np.full_like(std, np.nan)
            else:
                sd = std
                se = std / sqrt_n if sqrt_n is not None else np.full_like(std, np.nan)
            valid = np.isfinite(point[name]) & (count > 1)
            out[f"{name}_sd"] = np.where(valid, sd, np.nan)
            out[f"{name}_se"] = np.where(valid, se, np.nan)
            out[f"{name}_ci_low"] = np.where(valid, low, np.nan)
            out[f"{name}_ci_high"] = np.where(valid, high, np.nan)
            if name in LOGNORMAL_PARAMETERS:
                positive = finite & (reps > 0)
                logs = np.where(positive, np.log(np.where(positive, reps, 1.0)), np.nan)
                n_positive = positive.sum(axis=1)
                log_var = np.nanvar(logs, axis=1, ddof=1)
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

    Args:
        timecourses: the batch (group curves with `sd` or `se`)
        options: `n_boot`, `seed`, `bootstrap_spread`, `bootstrap_distribution`, `ci_level`
        point: the parameters of the original curves (`run_rows` output)

    Returns:
        The uncertainty variables per parameter (see `reduce_replicates`).

    Raises:
        ValueError: if the batch has no spread to resample with.
    """
    # the analysis of the replicates runs through the same core as the original
    # curves, whose module imports this one
    from pkpdutils.nca.nca import run_rows

    n_rows, n_time = timecourses.n_samples, timecourses.n_time
    t = timecourses.times.reshape(n_rows, n_time)
    c = timecourses.values.reshape(n_rows, n_time)
    spread = resolve_spread(timecourses, options)
    rng = np.random.default_rng(options.seed)
    draws = resample_values(
        c, spread, options.n_boot, rng, options.bootstrap_distribution
    )
    b = options.n_boot

    def repeat(a: np.ndarray | None) -> np.ndarray | None:
        """Repeat a per row array `B` times (the rows stay grouped).

        Args:
            a: the array, or `None`.

        Returns:
            The repeated array `(N * B,)`, or `None`.
        """
        return (
            None
            if a is None
            else np.repeat(np.asarray(a, dtype=np.float64).reshape(n_rows), b)
        )

    logger.info("bootstrap: %d curves x %d replicates", n_rows, b)
    values = run_rows(
        np.repeat(t, b, axis=0),
        draws.reshape(n_rows * b, n_time),
        dose_amount=repeat(timecourses.dose_amount),
        dose_time=repeat(timecourses.dose_time),
        dose_duration=repeat(timecourses.dose_duration),
        route=timecourses.route,
        options=options,
    )
    replicates = {name: array.reshape(n_rows, b) for name, array in values.items()}
    n_subjects = (
        None
        if timecourses.n is None
        else np.asarray(timecourses.n, dtype=np.float64).reshape(n_rows)
    )
    return reduce_replicates(
        replicates,
        point,
        spread_kind=options.bootstrap_spread,
        n_subjects=n_subjects,
        ci_level=options.ci_level,
    )
