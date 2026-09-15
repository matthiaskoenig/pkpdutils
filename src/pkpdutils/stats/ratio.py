r"""Geometric mean ratio of a parameter between a test and a reference sample."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from pkpdutils.stats.sample import (
    ParameterSample,
    Scale,
    _log_positive,
    paired_values,
)
from pkpdutils.stats.tests import _welch_df


@dataclass(frozen=True)
class RatioResult:
    r"""Geometric mean ratio with its t interval on the log scale.

    Attributes:
        gmr: geometric mean ratio test / reference
        ci_low: lower bound of the interval of the ratio
        ci_high: upper bound of the interval
        ci_level: level of the interval
        log_ratio: \(\ln \mathrm{GMR}\)
        se_log: standard error of `log_ratio`
        df: degrees of freedom of the t interval
        paired: whether the samples were paired
        n_test: number of values of the test sample
        n_reference: number of values of the reference sample
        name: name of the parameter
        unit: unit of the parameter
    """

    gmr: float
    ci_low: float
    ci_high: float
    ci_level: float
    log_ratio: float
    se_log: float
    df: float
    paired: bool
    n_test: int
    n_reference: int
    name: str
    unit: str

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dictionary.

        Returns:
            Field name to value.
        """
        return {
            "gmr": self.gmr,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "log_ratio": self.log_ratio,
            "se_log": self.se_log,
            "df": self.df,
            "paired": self.paired,
            "n_test": self.n_test,
            "n_reference": self.n_reference,
            "name": self.name,
            "unit": self.unit,
        }


def _pair(
    test: ParameterSample, reference: ParameterSample
) -> tuple[np.ndarray, np.ndarray]:
    """The log values of two paired samples, matched by `paired_values`.

    Args:
        test: the test sample.
        reference: the reference sample.

    Returns:
        The logarithms of the test and the reference values in matching order.

    Raises:
        ValueError: for summary data, unequal sizes, labels which do not
            match, no pair of finite values, or a non-positive value.
    """
    x, y = paired_values(test, reference)
    return _log_positive(x, test.name), _log_positive(y, reference.name)


def _labels_match(test: ParameterSample, reference: ParameterSample) -> bool:
    """Whether both samples are individual, labelled and share an individual.

    Which values are finite does not enter, so a missing value does not turn
    a paired design into an unpaired one.

    Args:
        test: the test sample.
        reference: the reference sample.

    Returns:
        `True` if the samples can be paired by label.
    """
    if not (test.is_individual and reference.is_individual):
        return False
    lx, ly = test.labels, reference.labels
    if lx is None or ly is None:
        return False
    return bool(set(lx.tolist()) & set(ly.tolist()))


def ratio(
    test: ParameterSample,
    reference: ParameterSample,
    *,
    ci_level: float = 0.90,
    paired: bool | None = None,
) -> RatioResult:
    r"""Geometric mean ratio of a parameter, test over reference, with a t interval.

    Paired (crossover design): \(d_i = \ln t_i - \ln r_i\), \(\ln \mathrm{GMR} = \bar d\),
    \(\mathrm{se} = s_d / \sqrt{n}\), \(n - 1\) degrees of freedom. Unpaired (parallel
    groups): the Welch t interval of \(\bar{\ln t} - \bar{\ln r}\). The interval of
    the ratio is the exponentiated interval (FDA 2001; Schuirmann 1987).
    Summary data uses the log moments of `ParameterSample.log_moments`.
    Paired samples are matched with `paired_values`, by label when both
    samples carry labels and by position otherwise; a pair with a missing
    value is dropped. A sample of one value or two samples without variance
    give `NaN` for `se_log`, `df` and the interval, the `gmr` stays finite.

    Args:
        test: the test sample.
        reference: the reference sample.
        ci_level: level of the interval, 0.90 by default as in bioequivalence.
        paired: pair the samples (by label when both have labels, else by
            position); `None` pairs when both samples share a label.

    Returns:
        The ratio.

    Raises:
        ValueError: for a paired ratio on summary data, unequal sizes,
            labels which do not match, or no pair of finite values.
    """
    is_paired = _labels_match(test, reference) if paired is None else paired
    alpha = 1.0 - ci_level
    if is_paired:
        x, y = _pair(test, reference)
        d = x - y
        n = d.size
        center = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        df = float(n - 1) if n > 1 else float("nan")
        n_test = n_reference = n
    else:
        mu_t, s_t, n_test = test.moments(Scale.LOG)
        mu_r, s_r, n_reference = reference.moments(Scale.LOG)
        center = mu_t - mu_r
        se = float(np.sqrt(s_t**2 / n_test + s_r**2 / n_reference))
        df = _welch_df(s_t**2, n_test, s_r**2, n_reference)
    tq = float(student_t.ppf(1.0 - alpha / 2.0, df)) if df > 0 else float("nan")
    return RatioResult(
        gmr=float(np.exp(center)),
        ci_low=float(np.exp(center - tq * se)),
        ci_high=float(np.exp(center + tq * se)),
        ci_level=ci_level,
        log_ratio=center,
        se_log=se,
        df=df,
        paired=bool(is_paired),
        n_test=int(n_test),
        n_reference=int(n_reference),
        name=test.name,
        unit=test.unit,
    )
