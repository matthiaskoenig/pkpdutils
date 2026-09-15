"""Parameter samples: individual values or summary statistics, on the linear or the log scale.

A `ParameterSample` is the input of every function of `pkpdutils.stats`: the
values of one parameter over the individuals of a group, or the summary
statistics of a group as published (`mean`, `sd`, `n`, optionally `geomean`
and `geocv`). Pharmacokinetic parameters are log-normal, so the statistics
work on the log scale by default (`Scale.LOG`), where the summary statistics
are translated with the moment relations of the log-normal distribution
(Rowland & Tozer 2011, ch. 8; `lognormal_from_moments`).
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import t as student_t

logger = logging.getLogger(__name__)


class Scale(StrEnum):
    """Scale of an analysis."""

    #: differences of arithmetic means
    LINEAR = "linear"
    #: ratios of geometric means: the analysis runs on the logarithms
    LOG = "log"


def lognormal_from_moments(mean: float, sd: float) -> tuple[float, float]:
    r"""Log-scale moments of a log-normal distribution with the given mean and standard deviation.

    \(\sigma^2 = \ln(1 + \mathrm{sd}^2 / \mathrm{mean}^2)\), \(\mu = \ln \mathrm{mean} - \sigma^2 / 2\).

    Args:
        mean: arithmetic mean, positive.
        sd: standard deviation, non-negative.

    Returns:
        `mu` and `sigma`, the mean and the standard deviation of the logarithm.

    Raises:
        ValueError: if `mean` is not positive or `sd` is negative.
    """
    if not mean > 0:
        raise ValueError(
            f"The mean of a log-normal sample must be positive, got {mean}"
        )
    if sd < 0:
        raise ValueError(f"The standard deviation must not be negative, got {sd}")
    sigma2 = float(np.log1p((sd / mean) ** 2))
    return float(np.log(mean) - sigma2 / 2.0), float(np.sqrt(sigma2))


def lognormal_from_geometric(geomean: float, geocv: float) -> tuple[float, float]:
    r"""Log-scale moments from a geometric mean and a geometric coefficient of variation.

    \(\mu = \ln \mathrm{geomean}\), \(\sigma = \sqrt{\ln(1 + \mathrm{geocv}^2)}\).

    Args:
        geomean: geometric mean, positive.
        geocv: geometric coefficient of variation \(\sqrt{e^{\sigma^2} - 1}\), non-negative.

    Returns:
        `mu` and `sigma`.

    Raises:
        ValueError: if `geomean` is not positive or `geocv` is negative.
    """
    if not geomean > 0:
        raise ValueError(f"The geometric mean must be positive, got {geomean}")
    if geocv < 0:
        raise ValueError(f"The geometric CV must not be negative, got {geocv}")
    return float(np.log(geomean)), float(np.sqrt(np.log1p(geocv**2)))


def moments_from_lognormal(mu: float, sigma: float) -> tuple[float, float]:
    r"""Arithmetic mean and standard deviation of a log-normal distribution.

    \(\mathrm{mean} = e^{\mu + \sigma^2/2}\), \(\mathrm{sd} = \mathrm{mean}\sqrt{e^{\sigma^2} - 1}\).

    Args:
        mu: mean of the logarithm.
        sigma: standard deviation of the logarithm.

    Returns:
        The arithmetic mean and standard deviation.
    """
    mean = float(np.exp(mu + sigma**2 / 2.0))
    return mean, float(mean * np.sqrt(np.expm1(sigma**2)))


def _log_positive(values: np.ndarray, name: str) -> np.ndarray:
    """The logarithms of an array which must be positive.

    Args:
        values: the values.
        name: name of the parameter for the error message.

    Returns:
        The logarithms.

    Raises:
        ValueError: if a value is not positive.
    """
    if np.any(values <= 0):
        raise ValueError(
            f"'{name}' has non-positive values, the log scale needs positive values"
        )
    return np.log(values)


def _array(value: Any, name: str) -> np.ndarray:
    """A 1-D array of a field.

    Args:
        value: the field value.
        name: the field name for the error message.

    Returns:
        The array.

    Raises:
        ValueError: if `value` is not 1-D.
    """
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"'{name}' must be 1-D, got shape {arr.shape}")
    return arr


@dataclass(frozen=True)
class Summary:
    r"""Summary statistics of a parameter sample.

    Attributes:
        n: number of values (finite values, or `n` of summary data)
        mean: arithmetic mean
        sd: standard deviation (`ddof=1`)
        se: standard error of the mean, `sd / sqrt(n)`
        cv: coefficient of variation, `sd / mean`
        geomean: geometric mean \(e^{\mu}\)
        geocv: geometric coefficient of variation \(\sqrt{e^{\sigma^2} - 1}\)
        median: median (`NaN` for summary data)
        q25: first quartile (`NaN` for summary data)
        q75: third quartile (`NaN` for summary data)
        min: minimum (`NaN` for summary data)
        max: maximum (`NaN` for summary data)
        ci_low: lower bound of the t interval of the mean (`LINEAR`) or of the geometric mean (`LOG`)
        ci_high: upper bound of the interval
        ci_level: level of the interval
        scale: scale of the interval
        name: name of the parameter
        unit: unit of the parameter
    """

    n: int
    mean: float
    sd: float
    se: float
    cv: float
    geomean: float
    geocv: float
    median: float
    q25: float
    q75: float
    min: float
    max: float
    ci_low: float
    ci_high: float
    ci_level: float
    scale: Scale
    name: str
    unit: str

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dictionary.

        Returns:
            Field name to value.
        """
        return {
            "n": self.n,
            "mean": self.mean,
            "sd": self.sd,
            "se": self.se,
            "cv": self.cv,
            "geomean": self.geomean,
            "geocv": self.geocv,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "min": self.min,
            "max": self.max,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "scale": str(self.scale),
            "name": self.name,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class ParameterSample:
    """The values of one parameter over a group of individuals, or the summary statistics of the group.

    Individual data: `values` (1-D, `NaN` is skipped) with optional `labels`
    (the identity of the individuals, used to pair two samples) and `coords`
    (further attributes per individual, such as `period` and `sequence` of a
    crossover study). Summary data: `n` with `mean` and `sd` and/or `geomean`
    and `geocv`, as reported in a publication.

    Attributes:
        values: individual values, `None` for summary data
        labels: label per value, `None` without labels
        coords: name to array with one entry per value
        mean: arithmetic mean of summary data
        sd: standard deviation of summary data
        n: number of individuals of summary data
        geomean: geometric mean of summary data
        geocv: geometric coefficient of variation of summary data
        name: name of the parameter
        unit: unit of the parameter
    """

    values: np.ndarray | None = None
    labels: np.ndarray | None = None
    coords: dict[str, np.ndarray] = field(default_factory=dict)
    mean: float | None = None
    sd: float | None = None
    n: float | None = None
    geomean: float | None = None
    geocv: float | None = None
    name: str = "value"
    unit: str = "dimensionless"

    def __post_init__(self) -> None:
        """Convert the arrays and validate the combination of fields.

        Raises:
            ValueError: for neither values nor summary data, summary data
                without `n` or without a pair of moments, an array which is
                not 1-D, or labels or coordinates of another length.
        """
        if self.values is not None:
            values = _array(self.values, "values").astype(np.float64)
            object.__setattr__(self, "values", values)
            if self.labels is not None:
                labels = _array(self.labels, "labels")
                if labels.size != values.size:
                    raise ValueError(
                        f"'labels' has length {labels.size}, 'values' has length {values.size}"
                    )
                object.__setattr__(self, "labels", labels)
            coords = {}
            for key, coord in self.coords.items():
                arr = _array(coord, key)
                if arr.size != values.size:
                    raise ValueError(
                        f"coordinate '{key}' has length {arr.size}, 'values' has length {values.size}"
                    )
                coords[key] = arr
            object.__setattr__(self, "coords", coords)
            return
        if self.n is None:
            raise ValueError(
                "Summary data needs 'n'; give 'values' or 'mean', 'sd' and 'n'"
            )
        has_moments = self.mean is not None and self.sd is not None
        has_geometric = self.geomean is not None and self.geocv is not None
        if not (has_moments or has_geometric):
            raise ValueError(
                "Summary data needs 'mean' and 'sd' or 'geomean' and 'geocv'; give 'values' or these"
            )
        if self.n < 1:
            raise ValueError(f"'n' must be at least 1, got {self.n}")

    @property
    def is_individual(self) -> bool:
        """Whether the sample holds individual values."""
        return self.values is not None

    @property
    def _finite(self) -> np.ndarray:
        """Mask of the finite values (all `False` for summary data)."""
        if self.values is None:
            return np.zeros(0, dtype=bool)
        return np.isfinite(self.values)

    @property
    def finite_values(self) -> np.ndarray:
        """The finite individual values, empty for summary data."""
        if self.values is None:
            return np.zeros(0, dtype=np.float64)
        return self.values[self._finite]

    @property
    def finite_labels(self) -> np.ndarray | None:
        """The labels of the finite values, `None` without labels."""
        if self.values is None or self.labels is None:
            return None
        return self.labels[self._finite]

    @property
    def size(self) -> int:
        """Number of finite values, or `n` of summary data."""
        if self.values is None:
            assert self.n is not None
            return int(self.n)
        return int(self._finite.sum())

    @property
    def log_values(self) -> np.ndarray:
        """The logarithms of the finite values.

        Raises:
            ValueError: if a finite value is not positive.
        """
        return _log_positive(self.finite_values, self.name)

    def log_moments(self) -> tuple[float, float]:
        """Mean and standard deviation of the logarithm.

        Individual data: the moments of `log_values` (`ddof=1`, `NaN` with a
        single value). Summary data: `lognormal_from_geometric` when
        `geomean` and `geocv` are given, else `lognormal_from_moments`.

        Returns:
            `mu` and `sigma`.
        """
        if self.values is not None:
            logs = self.log_values
            sigma = float(np.std(logs, ddof=1)) if logs.size > 1 else float("nan")
            return float(np.mean(logs)) if logs.size else float("nan"), sigma
        if self.geomean is not None and self.geocv is not None:
            return lognormal_from_geometric(self.geomean, self.geocv)
        assert self.mean is not None and self.sd is not None
        return lognormal_from_moments(self.mean, self.sd)

    def linear_moments(self) -> tuple[float, float]:
        """Arithmetic mean and standard deviation.

        Summary data given only as geometric statistics are translated with
        `moments_from_lognormal`.

        Returns:
            `mean` and `sd`.
        """
        if self.values is not None:
            values = self.finite_values
            sd = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
            return float(np.mean(values)) if values.size else float("nan"), sd
        if self.mean is not None and self.sd is not None:
            return float(self.mean), float(self.sd)
        return moments_from_lognormal(*self.log_moments())

    def moments(self, scale: Scale) -> tuple[float, float, int]:
        """Center, spread and size on a scale.

        Args:
            scale: `LINEAR` for the arithmetic moments, `LOG` for the log moments.

        Returns:
            The center, the standard deviation and the number of values.
        """
        center, spread = (
            self.log_moments() if scale is Scale.LOG else self.linear_moments()
        )
        return center, spread, self.size

    def select(self, mask: np.ndarray) -> "ParameterSample":
        """The individual data at a boolean mask, with its labels and coordinates.

        Args:
            mask: boolean array with one entry per value.

        Returns:
            The selected sample.

        Raises:
            ValueError: for summary data.
        """
        if self.values is None:
            raise ValueError("'select' needs individual data")
        mask = np.asarray(mask, dtype=bool)
        return ParameterSample(
            values=self.values[mask],
            labels=None if self.labels is None else self.labels[mask],
            coords={k: v[mask] for k, v in self.coords.items()},
            name=self.name,
            unit=self.unit,
        )

    def summary(self, scale: Scale = Scale.LOG, ci_level: float = 0.95) -> Summary:
        """The summary statistics, see `summarize`.

        Args:
            scale: scale of the confidence interval.
            ci_level: level of the confidence interval.

        Returns:
            The summary.
        """
        return summarize(self, scale=scale, ci_level=ci_level)


def paired_indices(
    a: ParameterSample, b: ParameterSample
) -> tuple[np.ndarray, np.ndarray]:
    """The indices of the matched pairs of two samples of individual data.

    When both samples carry labels the pairs are matched by label: a label
    which only one sample carries is dropped, so a missing individual does
    not break the pairing. Without labels on both samples (also when only
    one of them is labelled) the pairs are matched by position and the
    samples must have the same length. A pair is dropped when either of its
    two values is not finite; dropped pairs are logged at debug level.

    Args:
        a: the first sample.
        b: the second sample.

    Returns:
        The indices into `a.values` and into `b.values` of the surviving
        pairs, in the order of `a`.

    Raises:
        ValueError: for summary data, for labels which are duplicated within
            a sample or whose sets are disjoint, for unequal sizes without
            labels, or when no pair of finite values remains.
    """
    if not (a.is_individual and b.is_individual):
        raise ValueError(
            f"Pairing '{a.name}' and '{b.name}' needs individual data of both samples"
        )
    assert a.values is not None and b.values is not None
    if a.labels is not None and b.labels is not None:
        labels_a, labels_b = a.labels.tolist(), b.labels.tolist()
        for sample, labels in ((a, labels_a), (b, labels_b)):
            if len(set(labels)) != len(labels):
                raise ValueError(
                    f"'{sample.name}' has duplicate labels, pairing needs one value per individual"
                )
        position = {label: i for i, label in enumerate(labels_b)}
        shared = [
            (i, position[label])
            for i, label in enumerate(labels_a)
            if label in position
        ]
        if not shared:
            raise ValueError(
                f"The labels of '{a.name}' and '{b.name}' do not match, pairing needs the same individuals"
            )
        index_a = np.array([i for i, _ in shared], dtype=np.intp)
        index_b = np.array([j for _, j in shared], dtype=np.intp)
    else:
        if a.values.size != b.values.size:
            raise ValueError(
                f"paired samples need equal sizes, got {a.values.size} and {b.values.size}"
            )
        index_a = np.arange(a.values.size, dtype=np.intp)
        index_b = np.arange(b.values.size, dtype=np.intp)
    keep = np.isfinite(a.values[index_a]) & np.isfinite(b.values[index_b])
    dropped = int(keep.size - keep.sum())
    if dropped:
        logger.debug(
            "%d of %d pairs of '%s' and '%s' dropped: missing values",
            dropped,
            keep.size,
            a.name,
            b.name,
        )
    index_a, index_b = index_a[keep], index_b[keep]
    if index_a.size == 0:
        raise ValueError(f"'{a.name}' and '{b.name}' have no pair of finite values")
    return index_a, index_b


def paired_values(
    a: ParameterSample, b: ParameterSample
) -> tuple[np.ndarray, np.ndarray]:
    """The raw values of two samples as matched pairs, see `paired_indices`.

    Args:
        a: the first sample.
        b: the second sample.

    Returns:
        The values of `a` and of `b` of the surviving pairs, in matching order.

    Raises:
        ValueError: as `paired_indices`.
    """
    index_a, index_b = paired_indices(a, b)
    assert a.values is not None and b.values is not None
    return a.values[index_a], b.values[index_b]


def summarize(
    values: ParameterSample | ArrayLike,
    *,
    scale: Scale = Scale.LOG,
    ci_level: float = 0.95,
    name: str = "value",
    unit: str = "dimensionless",
) -> Summary:
    r"""Summary statistics of a parameter sample.

    The arithmetic statistics, the geometric mean and the geometric CV, the
    quantiles of individual data, and a t interval: of the mean on the
    `LINEAR` scale, \(\bar x \pm t_{1-\alpha/2, n-1}\,\mathrm{sd}/\sqrt{n}\), and of the
    geometric mean on the `LOG` scale, \(\exp(\mu \pm t_{1-\alpha/2, n-1}\,\sigma/\sqrt{n})\).
    The interval and the spread are `NaN` with a single value. On `LOG` a
    non-positive value raises `ValueError`; on `LINEAR` it is tolerated and
    `geomean`/`geocv` come back as `NaN` instead.

    Args:
        values: a sample, or individual values (`NaN` skipped).
        scale: scale of the interval.
        ci_level: level of the interval.
        name: name of the parameter (ignored for a sample, which carries its own).
        unit: unit of the parameter (ignored for a sample).

    Returns:
        The summary.
    """
    sample = (
        values
        if isinstance(values, ParameterSample)
        else ParameterSample(
            values=np.asarray(values, dtype=np.float64), name=name, unit=unit
        )
    )
    n = sample.size
    mean, sd = sample.linear_moments()
    nan = float("nan")
    if scale is Scale.LINEAR:
        try:
            mu, sigma = sample.log_moments()
        except ValueError:
            mu, sigma = nan, nan
    else:
        mu, sigma = sample.log_moments()
    se = sd / np.sqrt(n) if n > 1 else nan
    if sample.is_individual:
        v = sample.finite_values
        median, q25, q75 = (
            (float(np.median(v)), *(float(q) for q in np.percentile(v, [25, 75])))
            if v.size
            else (nan, nan, nan)
        )
        low_high = (float(v.min()), float(v.max())) if v.size else (nan, nan)
    else:
        median, q25, q75 = nan, nan, nan
        low_high = (nan, nan)
    if n > 1:
        tq = float(student_t.ppf(1.0 - (1.0 - ci_level) / 2.0, n - 1))
        if scale is Scale.LOG:
            ci = (
                float(np.exp(mu - tq * sigma / np.sqrt(n))),
                float(np.exp(mu + tq * sigma / np.sqrt(n))),
            )
        else:
            ci = (mean - tq * se, mean + tq * se)
    else:
        ci = (nan, nan)
    return Summary(
        n=n,
        mean=mean,
        sd=sd,
        se=float(se),
        cv=float(sd / mean) if mean else nan,
        geomean=float(np.exp(mu)),
        geocv=float(np.sqrt(np.expm1(sigma**2))),
        median=median,
        q25=q25,
        q75=q75,
        min=low_high[0],
        max=low_high[1],
        ci_low=ci[0],
        ci_high=ci[1],
        ci_level=ci_level,
        scale=scale,
        name=sample.name,
        unit=sample.unit,
    )
