"""Geometric mean ratio of a parameter between a test and a reference sample."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from pkpdutils.stats.sample import (
    ParameterSample,
    Scale,
    exp_t_interval,
    labels_match,
    log_positive,
    paired_values,
    welch_df,
    welch_se,
)

logger = logging.getLogger(__name__)


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
    give `NaN` for `se_log`, `df` and the interval, the `gmr` stays finite;
    an unpaired sample without a finite value gives `NaN` throughout, a
    paired one raises, as no pair remains.

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
    is_paired = labels_match(test, reference) if paired is None else paired
    if is_paired:
        raw_test, raw_reference = paired_values(test, reference)
        x = log_positive(raw_test, test.name)
        y = log_positive(raw_reference, reference.name)
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
        if n_test < 1 or n_reference < 1:
            logger.debug(
                "'%s' or '%s' has no finite value, the ratio is NaN",
                test.name,
                reference.name,
            )
        se = welch_se(s_t**2, n_test, s_r**2, n_reference)
        df = welch_df(s_t**2, n_test, s_r**2, n_reference)
    ci = exp_t_interval(center, se, df, ci_level)
    return RatioResult(
        gmr=float(np.exp(center)),
        ci_low=ci[0],
        ci_high=ci[1],
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
