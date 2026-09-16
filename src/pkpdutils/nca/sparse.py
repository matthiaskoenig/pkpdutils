r"""Non-compartmental analysis of sparse and destructive sampling designs.

A preclinical study rarely samples one animal repeatedly: the animal is
sacrificed for the sample (a destructive design, one sample per animal) or
contributes a few samples out of the schedule (a batch design). There is no
curve per animal then, only a mean curve over the animals of every time point,
and the question is how uncertain the area under that mean curve is.

Bailer (1988) answers it: the area is a fixed linear combination of the means,

$$\widehat{\mathrm{AUC}} = \sum_j w_j \bar y_j,$$

with the trapezoid weights \(w_j\), so its variance follows from the variances
of the means. With one sample per animal the means are independent and

$$\widehat{\mathrm{Var}}\left[\widehat{\mathrm{AUC}}\right] = \sum_j w_j^2
\frac{s_j^2}{n_j};$$

Nedelman, Gibiansky and Lau (1995) give the Satterthwaite degrees of freedom of
that sum so the area gets a \(t\) interval. Nedelman and Jia (1998) extend the
estimator to a batch design, where an animal contributes to several means, and
Holder (2001), commenting on that extension, gives the variance which carries
the covariance between the time points an animal is shared by.

`pkpdutils` computes all three from one identity. Writing the estimator per
animal rather than per time point,

$$\widehat{\mathrm{AUC}} = \sum_i A_i, \qquad A_i = \sum_{j \in T_i}
\frac{w_j}{n_j}\, y_{ij},$$

where \(T_i\) are the times animal \(i\) was sampled at, the animals are
independent whatever the design, so the variance is the sum over the animals
and is estimated batch by batch (a batch is a group of animals with the same
sampling times):

$$\widehat{\mathrm{Var}}\left[\widehat{\mathrm{AUC}}\right] = \sum_b m_b\,
s^2_{A,b}, \qquad \nu = \frac{\left(\sum_b c_b\right)^2}{\sum_b
\frac{c_b^2}{m_b - 1}}, \quad c_b = m_b\, s^2_{A,b}.$$

With one sample per animal a batch is one time point, \(A_i = (w_j/n_j)
y_{ij}\) and \(c_b = w_j^2 s_j^2 / n_j\): the formula of Bailer and the degrees
of freedom of Nedelman, Gibiansky and Lau, exactly. With several samples per
animal the sample variance of the \(A_i\) carries the covariances of Holder
without ever forming them.

Nominal times are used, never the actual sampling times: a mean over animals
only exists at a nominal time (the caution of Phoenix WinNonlin for its sparse
models).
"""

import logging
import warnings
from typing import Any, Literal

import numpy as np
import xarray as xr

from pkpdutils.nca.nca import PARAMETER_UNITS, unit_expression
from pkpdutils.nca.options import AUCMethod, NCAFlag, NCAOptions
from pkpdutils.nca.result import DOSE_COORDINATE, NCAResult, parameter_unit
from pkpdutils.timecourse import TIME_DIM, Dose, Timecourses

logger = logging.getLogger(__name__)

#: the sampling designs `nca_sparse` reads
Design = Literal["serial", "batch"]

#: the variables of a sparse result, in the order they are reported
SPARSE_VARIABLES: tuple[str, ...] = (
    "auc_last",
    "auc_last_se",
    "auc_last_df",
    "auc_all",
    "cmax",
    "cmax_se",
    "tmax",
)


def trapezoid_weights(times: np.ndarray) -> np.ndarray:
    r"""Weights of the linear trapezoid rule over a grid of times.

    The area under the polygon through \((t_j, y_j)\) is \(\sum_j w_j y_j\)
    with

    $$w_1 = \frac{t_2 - t_1}{2}, \qquad w_j = \frac{t_{j+1} - t_{j-1}}{2},
    \qquad w_J = \frac{t_J - t_{J-1}}{2},$$

    which is the form Bailer (1988) needs: the area is linear in the values, so
    its variance follows from theirs. The logarithmic trapezoid rules of
    `pkpdutils.nca.options.AUCMethod` are not linear in the values and have no
    such weights.

    Args:
        times: the sampling times, strictly increasing, at least two.

    Returns:
        One weight per time, of the shape of `times`.
    """
    times = np.asarray(times, dtype=np.float64)
    weights = np.zeros(times.shape, dtype=np.float64)
    if times.size < 2:
        return weights
    weights[0] = 0.5 * (times[1] - times[0])
    weights[-1] = 0.5 * (times[-1] - times[-2])
    if times.size > 2:
        weights[1:-1] = 0.5 * (times[2:] - times[:-2])
    return weights


def point_statistics(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count, mean and standard deviation of every time point of a sparse design.

    Args:
        values: the values `(n_animals, n_time)`, `NaN` where an animal has no
            sample at a time.

    Returns:
        The number of animals sampled at every time, the mean over them and
        their standard deviation (`ddof=1`, `NaN` for a single animal).
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    n = finite.sum(axis=0)
    clean = np.where(finite, values, np.nan)
    with (
        np.errstate(invalid="ignore"),
        warnings.catch_warnings(),
    ):
        # a nominal time without a sample (or with one) has no mean (no
        # standard deviation), which numpy reports as an empty-slice warning
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(clean, axis=0)
        sd = np.nanstd(clean, axis=0, ddof=1)
    return n, np.where(n > 0, mean, np.nan), np.where(n > 1, sd, np.nan)


def area_window(observed: np.ndarray, mean: np.ndarray) -> np.ndarray:
    r"""The nominal times `auc_last` of a sparse design covers.

    The observed time points up to the last one whose mean is positive, the
    \(t_\mathrm{last}\) rule of a concentration curve read on the mean curve.
    `nca_sparse` weights these points and `pkpdutils.plot.plot_sparse` shades
    them, so both read the window from here.

    Args:
        observed: whether a nominal time carries a sample at all.
        mean: the mean of every nominal time, `NaN` where there is none.

    Returns:
        The boolean mask of the covered time points; all `False` when no mean
        is positive.
    """
    window = np.asarray(observed, dtype=bool).copy()
    with np.errstate(invalid="ignore"):
        measurable = window & (np.asarray(mean, dtype=np.float64) > 0.0)
    if not measurable.any():
        return np.zeros(window.shape, dtype=bool)
    window[int(np.flatnonzero(measurable)[-1]) + 1 :] = False
    return window


def bailer_variance(
    weights: np.ndarray, values: np.ndarray
) -> tuple[float, float, int]:
    r"""Variance of the sparse area estimate and its Satterthwaite degrees of freedom.

    The estimate is written per animal, \(A_i = \sum_{j \in T_i} (w_j/n_j)
    y_{ij}\) with \(n_j\) the number of animals sampled at time \(j\), so that
    the animals are independent whatever the design and

    $$\widehat{\mathrm{Var}}\left[\sum_i A_i\right] = \sum_b m_b\, s^2_{A,b},
    \qquad \nu = \frac{\left(\sum_b c_b\right)^2}{\sum_b \frac{c_b^2}{m_b -
    1}}, \quad c_b = m_b\, s^2_{A,b},$$

    where a batch \(b\) is the group of the \(m_b\) animals with the same
    sampling times and \(s^2_{A,b}\) the sample variance of their \(A_i\). With
    one sample per animal a batch is one time point and the two formulas are
    \(\sum_j w_j^2 s_j^2/n_j\) of Bailer (1988) and the degrees of freedom of
    Nedelman, Gibiansky and Lau (1995); with several samples per animal the
    sample variance carries the covariance terms of the batch design of
    Nedelman and Jia (1998) as Holder (2001) writes them.

    Args:
        weights: the weight of every time point, 0 for a time point which is
            not part of the area.
        values: the values `(n_animals, n_time)`, `NaN` where an animal has no
            sample at a time.

    Returns:
        The variance, the degrees of freedom and the number of batches; the
        variance and the degrees of freedom are `NaN` when a batch holds a
        single animal, whose contribution cannot be estimated.
    """
    weights = np.asarray(weights, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    used = np.isfinite(values) & (weights != 0.0)[None, :]
    n_per_time = used.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_value = np.where(n_per_time > 0, weights / n_per_time, 0.0)
    contributions = np.where(used, np.nan_to_num(values) * per_value[None, :], 0.0)
    animals = contributions.sum(axis=1)
    patterns, batch_of = np.unique(used, axis=0, return_inverse=True)
    terms: list[float] = []
    sizes: list[int] = []
    for index, pattern in enumerate(patterns):
        if not pattern.any():
            # an animal without a usable sample is no batch
            continue
        members = animals[batch_of.reshape(-1) == index]
        sizes.append(members.size)
        terms.append(
            float(members.size * members.var(ddof=1)) if members.size > 1 else np.nan
        )
    if not terms:
        return np.nan, np.nan, 0
    single = [size for size in sizes if size < 2]
    if single:
        logger.warning(
            "%d of %d batches hold a single animal: the variance of the area "
            "cannot be estimated",
            len(single),
            len(sizes),
        )
        return np.nan, np.nan, len(sizes)
    variance = float(np.sum(terms))
    denominator = float(
        np.sum(
            [term * term / (size - 1) for term, size in zip(terms, sizes, strict=True)]
        )
    )
    df = variance * variance / denominator if denominator > 0.0 else np.nan
    return variance, df, len(sizes)


def _check_design(values: np.ndarray, design: Design) -> None:
    """Check the values against the design.

    Args:
        values: the values `(n_animals, n_time)`.
        design: `"serial"` (one sample per animal) or `"batch"`.

    Raises:
        ValueError: if `design` is unknown, or if a serial design holds an
            animal with more than one sample.
    """
    if design not in ("serial", "batch"):
        raise ValueError(f"unknown design '{design}', use 'serial' or 'batch'")
    if design == "serial":
        per_animal = np.isfinite(values).sum(axis=1)
        if np.any(per_animal > 1):
            raise ValueError(
                f"{int((per_animal > 1).sum())} of {values.shape[0]} animals carry "
                "more than one sample, which is not a serial design; use "
                "design='batch'"
            )


def _as_values(times: Any, values: Any) -> tuple[np.ndarray, np.ndarray]:
    """The times and the values as arrays, checked against each other.

    Args:
        times: the nominal sampling times, one-dimensional.
        values: the values `(n_animals, n_time)`.

    Returns:
        The times and the values.

    Raises:
        ValueError: if the shapes do not fit or the times are not increasing.
    """
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"'values' has {y.ndim} axes, expected (n_animals, n_time)")
    if y.shape[1] != t.size:
        raise ValueError(f"'values' has {y.shape[1]} time points, 'times' has {t.size}")
    if t.size > 1 and not np.all(np.diff(t) > 0):
        raise ValueError("'times' must be strictly increasing")
    return t, y


def sparse_mean(
    times: Any,
    values: Any,
    *,
    time_unit: str,
    unit: str,
    dose: Dose | None = None,
    substance: str = "substance",
    dim: str = "group",
    label: Any = "mean",
) -> Timecourses:
    r"""The mean curve of a sparse design, with its spread per time point.

    The curve every sparse analysis is read from: the mean over the animals
    sampled at a nominal time, their standard deviation (`ddof=1`), the
    standard error \\(s_j/\\sqrt{n_j}\\) and the count \\(n_j\\), each per time
    point, so that a time point sampled in fewer animals carries its own count.

    Args:
        times: the nominal sampling times, one-dimensional
        values: the values `(n_animals, n_time)`, `NaN` where an animal has no
            sample at a time

    Keyword Args:
        time_unit: unit of the times
        unit: unit of the values
        dose: the dose of the animals, `None` without one
        substance: name of the substance
        dim: name of the sample dimension of the batch
        label: label of the single sample of the batch

    Returns:
        A batch of one sample carrying `value`, `sd`, `se` and `n` per time
        point.

    Raises:
        ValueError: if the shapes do not fit or the times are not increasing.
    """
    t, y = _as_values(times, values)
    n, mean, sd = point_statistics(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.where(n > 0, sd / np.sqrt(n), np.nan)
    return Timecourses.from_arrays(
        t,
        mean[None, :],
        time_unit=time_unit,
        unit=unit,
        dims=(dim,),
        coords={dim: [label]},
        sd=sd[None, :],
        se=se[None, :],
        n=n.astype(np.float64)[None, :],
        dose=dose,
        substance=substance,
    )


def nca_sparse(
    times: Any,
    values: Any,
    *,
    design: Design = "serial",
    options: NCAOptions | None = None,
    time_unit: str,
    unit: str,
    dose: Dose | None = None,
) -> NCAResult:
    r"""Non-compartmental analysis of a sparse or destructive sampling design.

    The area under the mean curve with the standard error of Bailer (1988), the
    degrees of freedom of Nedelman, Gibiansky and Lau (1995) and, for the batch
    design of Nedelman and Jia (1998), the covariance of Holder (2001), all
    three from the per-animal identity of `bailer_variance`:

    $$\widehat{\mathrm{AUC}}_{0\text{-}t_\mathrm{last}} = \sum_j w_j \bar y_j,
    \qquad \mathrm{se} =
    \sqrt{\widehat{\mathrm{Var}}\left[\widehat{\mathrm{AUC}}\right]}, \qquad
    \mathrm{CI} = \widehat{\mathrm{AUC}} \pm t_{1-\alpha/2,\nu}\,\mathrm{se}.$$

    \(t_\mathrm{last}\) is the last nominal time whose mean is positive, as it
    is for a concentration curve, and `auc_all` is the area over every nominal
    time with a sample. Both run over the nominal times as they are given, from
    the first of them rather than from the dose: nothing is inserted at time 0,
    because an inserted point has no variance and the estimator has to stay a
    combination of the measured means. A design whose area starts at the dose
    carries a nominal time 0 of its own (the value 0 after an extravascular
    dose). The peak of the mean curve is `cmax` at `tmax` with the
    standard error \(s_j/\sqrt{n_j}\) of the mean at that time, the
    `SE_Cmax` of Phoenix WinNonlin. The number of animals behind every time
    point is the point variable `n_animals`.

    The estimator is a fixed linear combination of the means, so the weights
    are always those of the linear trapezoid rule (`trapezoid_weights`): a
    logarithmic rule is not linear in the values and has no variance formula of
    this kind. `options.auc_method` is therefore not used, and neither are the
    rules which read a single curve (`lloq`, `blq`, the terminal phase, the
    uncertainty of a group curve).

    Args:
        times: the nominal sampling times, one-dimensional. Nominal, not
            actual: a mean over animals only exists at a nominal time.
        values: the values `(n_animals, n_time)`, `NaN` where an animal has no
            sample at a time

    Keyword Args:
        design: `"serial"` when every animal carries a single sample
            (destructive sampling, the variance of Bailer) and `"batch"` when
            an animal carries several (the covariance of Holder); a serial
            design is checked, both are estimated by the same formula
        options: the options, defaults for `None`
        time_unit: unit of the times
        unit: unit of the values
        dose: the dose of the animals, whose amount travels into the result as
            the coordinate `dose_amount`; `None` without one

    Returns:
        The parameters of the mean curve without sample dimensions:
        `auc_last`, `auc_last_se`, `auc_last_df`, `auc_all`, `cmax`,
        `cmax_se`, `tmax` and the point variable `n_animals` over `time`.

    Raises:
        ValueError: if the shapes do not fit, if the times are not increasing,
            if the design is unknown or if a serial design holds an animal with
            several samples.
    """
    options = options or NCAOptions()
    if options.auc_method is not AUCMethod.LINEAR:
        message = (
            f"the sparse area is a linear combination of the means, so the "
            f"linear trapezoid rule is used rather than {options.auc_method}"
        )
        if "auc_method" in options.model_fields_set:
            # the caller asked for a rule which has no variance formula of this
            # kind; the default LINEAR_LOG is only noted in the log
            warnings.warn(message, UserWarning, stacklevel=2)
        else:
            logger.debug("%s", message)
    t, y = _as_values(times, values)
    _check_design(y, design)
    n, mean, sd = point_statistics(y)
    observed = n > 0
    flags = int(NCAFlag.NONE)
    if int(observed.sum()) < 2:
        flags |= int(NCAFlag.NO_DATA)

    all_weights = np.zeros(t.shape)
    if int(observed.sum()) > 1:
        all_weights[observed] = trapezoid_weights(t[observed])
    last_weights = np.zeros(t.shape)
    through_last = area_window(observed, mean)
    if int(through_last.sum()) > 1:
        last_weights[through_last] = trapezoid_weights(t[through_last])

    auc_last = float(np.nansum(last_weights * np.nan_to_num(mean)))
    auc_all = float(np.nansum(all_weights * np.nan_to_num(mean)))
    variance, df, n_batches = bailer_variance(last_weights, y)
    logger.debug(
        "sparse %s design: %d animals, %d nominal times, %d batches",
        design,
        y.shape[0],
        t.size,
        n_batches,
    )

    peak = int(np.nanargmax(np.where(observed, mean, -np.inf))) if observed.any() else 0
    cmax = float(mean[peak]) if observed.any() else np.nan
    tmax = float(t[peak]) if observed.any() else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        cmax_se = (
            float(sd[peak] / np.sqrt(n[peak]))
            if observed.any() and n[peak] > 1
            else np.nan
        )

    magnitudes: dict[str, float] = {
        "auc_last": auc_last if last_weights.any() else np.nan,
        "auc_last_se": float(np.sqrt(variance)),
        "auc_last_df": float(df),
        "auc_all": auc_all if all_weights.any() else np.nan,
        "cmax": cmax,
        "cmax_se": cmax_se,
        "tmax": tmax,
    }
    dose_unit = None if dose is None else dose.unit
    data_vars: dict[str, Any] = {}
    for name in SPARSE_VARIABLES:
        parameter_unit_string, factor = parameter_unit(
            unit_expression(name), unit=unit, time_unit=time_unit, dose_unit=dose_unit
        )
        data_vars[name] = xr.DataArray(
            magnitudes[name] * factor, attrs={"units": parameter_unit_string}
        )
    data_vars["n_animals"] = xr.DataArray(
        n.astype(np.int64), dims=TIME_DIM, attrs={"units": PARAMETER_UNITS["n_animals"]}
    )
    data_vars["flags"] = xr.DataArray(np.int64(flags), attrs={"units": "dimensionless"})

    coords: dict[str, Any] = {
        TIME_DIM: xr.DataArray(t, dims=TIME_DIM, attrs={"units": time_unit})
    }
    if dose is not None:
        coords[DOSE_COORDINATE] = xr.DataArray(
            float(dose.amount), attrs={"units": dose.unit}
        )
    return NCAResult(xr.Dataset(data_vars=data_vars, coords=coords))
