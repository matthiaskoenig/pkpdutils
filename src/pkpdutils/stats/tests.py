"""Significance tests on parameter samples and the adjustment of p values.

`compare` runs the t tests (Student, Welch, paired), the rank tests
(Mann-Whitney U, Wilcoxon signed rank) and a permutation test of scipy on
two samples, on the log scale by default, and reports the effect with its
t interval and the standardized effect sizes (Cohen's d, Hedges' g; Hedges
1981). Summary data (`mean`, `sd`, `n`) is compared with the Welch t test
from the moments (`scipy.stats.ttest_ind_from_stats`), on the log scale with
the log-normal moments of `ParameterSample.log_moments`.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.stats import t as student_t

from pkpdutils.stats.sample import (
    ParameterSample,
    Scale,
    coerce,
    cohen_d,
    log_positive,
    paired_values,
    welch_df,
    welch_se,
)

logger = logging.getLogger(__name__)


class TestMethod(StrEnum):
    """Test of `compare`."""

    # not a pytest test class, despite the name matching pytest's collection pattern
    __test__ = False

    #: paired t test for `paired=True`, else Welch t test
    AUTO = "auto"
    #: Student t test, equal variances
    STUDENT_T = "student_t"
    #: Welch t test, unequal variances
    WELCH_T = "welch_t"
    #: paired t test
    PAIRED_T = "paired_t"
    #: Mann-Whitney U rank test, unpaired
    MANN_WHITNEY = "mann_whitney"
    #: Wilcoxon signed rank test, paired
    WILCOXON = "wilcoxon"
    #: permutation test of the difference of means
    PERMUTATION = "permutation"


class Alternative(StrEnum):
    """Alternative hypothesis, as in scipy."""

    TWO_SIDED = "two-sided"
    #: the center of `a` is less than the center of `b`
    LESS = "less"
    #: the center of `a` is greater than the center of `b`
    GREATER = "greater"


class AdjustMethod(StrEnum):
    """Adjustment of p values for multiple comparisons."""

    #: Holm step-down (Holm 1979), controls the family-wise error rate
    HOLM = "holm"
    #: Bonferroni, `m p`
    BONFERRONI = "bonferroni"
    #: Benjamini-Hochberg step-up (Benjamini & Hochberg 1995), controls the false discovery rate
    BH = "bh"


#: the t tests: an effect with a t interval and degrees of freedom
_T_TESTS = frozenset({TestMethod.STUDENT_T, TestMethod.WELCH_T, TestMethod.PAIRED_T})
#: the paired tests
_PAIRED_TESTS = frozenset({TestMethod.PAIRED_T, TestMethod.WILCOXON})


@dataclass(frozen=True)
class TestResult:
    """Result of `compare`.

    Attributes:
        test: the test which was run (`AUTO` resolved)
        statistic: the test statistic
        p_value: the p value under `alternative`
        effect: difference of the means `a - b` (`LINEAR`) or ratio of the geometric means `a / b` (`LOG`); for the rank tests the difference or the ratio of the medians
        ci_low: lower bound of the interval of the effect (t tests; one-sided under a one-sided alternative), `NaN` otherwise
        ci_high: upper bound of the interval
        ci_level: level of the interval
        scale: scale of the analysis
        alternative: the alternative hypothesis
        paired: whether the samples were paired
        df: degrees of freedom of a t test, `NaN` otherwise
        cohen_d: standardized difference of the means on the analysis scale, pooled standard deviation
        hedges_g: `cohen_d` times the small sample correction `J`
        n_a: number of values of `a`
        n_b: number of values of `b`
        name: name of the parameter (of `a`)
        unit: unit of the parameter
    """

    # not a pytest test class, despite the name matching pytest's collection pattern
    __test__ = False

    test: TestMethod
    statistic: float
    p_value: float
    effect: float
    ci_low: float
    ci_high: float
    ci_level: float
    scale: Scale
    alternative: Alternative
    paired: bool
    df: float
    cohen_d: float
    hedges_g: float
    n_a: int
    n_b: int
    name: str
    unit: str

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dictionary with the enumerations as strings.

        Returns:
            Field name to value.
        """
        return {
            "test": str(self.test),
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect": self.effect,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "scale": str(self.scale),
            "alternative": str(self.alternative),
            "paired": self.paired,
            "df": self.df,
            "cohen_d": self.cohen_d,
            "hedges_g": self.hedges_g,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "name": self.name,
            "unit": self.unit,
        }


def _values(sample: ParameterSample, scale: Scale) -> np.ndarray:
    """The finite values of an individual sample on the analysis scale.

    Args:
        sample: the sample.
        scale: the scale.

    Returns:
        The values (logarithms on the log scale).

    Raises:
        ValueError: for summary data.
    """
    if not sample.is_individual:
        raise ValueError(
            f"'{sample.name}' has no individual values, this test needs individual data"
        )
    return sample.log_values if scale is Scale.LOG else sample.finite_values


def _back(value: float, scale: Scale) -> float:
    """An effect on the analysis scale as reported: exponentiated on the log scale.

    Args:
        value: the effect on the analysis scale.
        scale: the scale.

    Returns:
        The reported effect.
    """
    return float(np.exp(value)) if scale is Scale.LOG else float(value)


def _interval(
    center: float, se: float, df: float, ci_level: float, alternative: Alternative
) -> tuple[float, float]:
    r"""The t interval of an effect on the analysis scale.

    Two-sided: \(\hat\theta \pm t_{1-\alpha/2, df}\,\mathrm{se}\); under a one-sided
    alternative the interval is one-sided at `ci_level`, the other bound is
    infinite.

    Args:
        center: the estimate.
        se: its standard error.
        df: degrees of freedom.
        ci_level: level of the interval.
        alternative: the alternative hypothesis.

    Returns:
        The lower and the upper bound.
    """
    alpha = 1.0 - ci_level
    if alternative is Alternative.TWO_SIDED:
        tq = float(student_t.ppf(1.0 - alpha / 2.0, df))
        return center - tq * se, center + tq * se
    tq = float(student_t.ppf(1.0 - alpha, df))
    if alternative is Alternative.LESS:
        return -np.inf, center + tq * se
    return center - tq * se, np.inf


def _p_from_t(statistic: float, df: float, alternative: Alternative) -> float:
    """The p value of a t statistic under an alternative.

    Args:
        statistic: the t statistic.
        df: degrees of freedom.
        alternative: the alternative hypothesis.

    Returns:
        The p value.
    """
    if alternative is Alternative.TWO_SIDED:
        return float(2.0 * student_t.sf(abs(statistic), df))
    if alternative is Alternative.LESS:
        return float(student_t.cdf(statistic, df))
    return float(student_t.sf(statistic, df))


def _median_effect(a: ParameterSample, b: ParameterSample, scale: Scale) -> float:
    """Difference or ratio of the medians of two samples, on the original scale.

    The medians are taken of the original values, not of their logarithms:
    for an even sample size the median averages the two middle order
    statistics, and that average does not commute with the logarithm, so
    the ratio of the medians differs from the exponentiated difference of
    the medians of the logarithms.

    Args:
        a: the first sample.
        b: the second sample.
        scale: `LINEAR` for the difference, `LOG` for the ratio.

    Returns:
        The effect.
    """
    median_a = float(np.median(a.finite_values))
    median_b = float(np.median(b.finite_values))
    return median_a / median_b if scale is Scale.LOG else median_a - median_b


def _t_statistic(center: float, se: float, df: float) -> float:
    """A t statistic, `NaN` for a degenerate sample.

    Args:
        center: the estimate.
        se: its standard error.
        df: the degrees of freedom of the statistic.

    Returns:
        `center / se`, `NaN` if `se` is zero or `se` or `df` is not finite.
    """
    if se == 0.0 or not np.isfinite(se) or not np.isfinite(df):
        return float("nan")
    return float(center / se)


def _mean_difference(u: np.ndarray, v: np.ndarray, axis: int) -> np.ndarray:
    """Difference of the means of two arrays along an axis."""
    return np.mean(u, axis=axis) - np.mean(v, axis=axis)


def _nan_result(
    method: TestMethod,
    scale: Scale,
    alternative: Alternative,
    paired: bool,
    ci_level: float,
    n_a: int,
    n_b: int,
    a: ParameterSample,
) -> TestResult:
    """The result of a comparison of a sample without a finite value.

    Args:
        method: the test which was asked for.
        scale: scale of the analysis.
        alternative: the alternative hypothesis.
        paired: whether the samples were paired.
        ci_level: level of the interval.
        n_a: number of values of `a`.
        n_b: number of values of `b`.
        a: the first sample, for the name and the unit.

    Returns:
        The result, with every statistic `NaN`.
    """
    nan = float("nan")
    return TestResult(
        test=method,
        statistic=nan,
        p_value=nan,
        effect=nan,
        ci_low=nan,
        ci_high=nan,
        ci_level=ci_level,
        scale=scale,
        alternative=alternative,
        paired=paired,
        df=nan,
        cohen_d=nan,
        hedges_g=nan,
        n_a=n_a,
        n_b=n_b,
        name=a.name,
        unit=a.unit,
    )


def compare(
    a: ParameterSample,
    b: ParameterSample,
    *,
    test: TestMethod | str = TestMethod.AUTO,
    scale: Scale | str = Scale.LOG,
    paired: bool = False,
    alternative: Alternative | str = Alternative.TWO_SIDED,
    ci_level: float = 0.95,
    n_perm: int = 9999,
    seed: int | None = None,
) -> TestResult:
    """Compare two samples of a parameter.

    `AUTO` runs the paired t test for `paired=True` and the Welch t test
    otherwise; summary data is compared with the Welch t test from its
    moments. On the log scale the tests run on the logarithms and the effect
    is the ratio of the geometric means with the exponentiated t interval.
    The paired tests need individual data of both samples and match it with
    `paired_values`, by label when both samples carry labels and by position
    otherwise; a pair with a missing value is dropped. The permutation test
    permutes the group labels (or the signs of the paired differences) of
    the difference of the means, with `n_perm` resamples (Efron & Tibshirani
    1993, ch. 15). A sample of one value or two samples without variance
    give `NaN` for `statistic`, `p_value`, `df`, the interval and the effect
    sizes; a sample without a finite value gives a `NaN` effect as well,
    unpaired, and raises on the paired path, where no pair remains.
    `test`, `scale` and `alternative` are taken as the enumeration member or
    as its string.

    Args:
        a: the first sample.
        b: the second sample.
        test: the test.
        scale: scale of the analysis.
        paired: whether the values of `a` and `b` belong to the same
            individuals (matched by label, else by position).
        alternative: the alternative hypothesis.
        ci_level: level of the interval of the effect.
        n_perm: number of resamples of the permutation test.
        seed: seed of the permutation test.

    Returns:
        The result.

    Raises:
        ValueError: for an unknown `test`, `scale` or `alternative`, a
            paired test on unpaired or unequal samples, a non-t test on
            summary data, or non-positive values on the log scale.
    """
    method = coerce(test, TestMethod)
    scale = coerce(scale, Scale)
    alternative = coerce(alternative, Alternative)
    if method is TestMethod.AUTO:
        method = TestMethod.PAIRED_T if paired else TestMethod.WELCH_T
    if method in _PAIRED_TESTS and not paired:
        raise ValueError(f"{method} needs paired=True")
    if paired and method not in _PAIRED_TESTS and method is not TestMethod.PERMUTATION:
        raise ValueError(
            f"{method} is not a paired test, use PAIRED_T, WILCOXON or PERMUTATION"
        )
    if not (a.is_individual and b.is_individual):
        if method is not TestMethod.WELCH_T:
            raise ValueError(
                f"{method} needs individual data; summary data allows WELCH_T only"
            )
        return _welch_from_moments(a, b, scale, alternative, ci_level)
    if paired:
        raw_a, raw_b = paired_values(a, b)
        x = log_positive(raw_a, a.name) if scale is Scale.LOG else raw_a
        y = log_positive(raw_b, b.name) if scale is Scale.LOG else raw_b
        n_a = n_b = int(x.size)
    else:
        x, y = _values(a, scale), _values(b, scale)
        n_a, n_b = int(x.size), int(y.size)
    nan = float("nan")
    if n_a < 1 or n_b < 1:
        logger.debug(
            "'%s' or '%s' has no finite value, the comparison is NaN", a.name, b.name
        )
        return _nan_result(method, scale, alternative, paired, ci_level, n_a, n_b, a)
    mean_a, mean_b = float(x.mean()), float(y.mean())
    sd_a = float(x.std(ddof=1)) if n_a > 1 else nan
    sd_b = float(y.std(ddof=1)) if n_b > 1 else nan
    d, g = cohen_d(mean_a, sd_a, n_a, mean_b, sd_b, n_b)
    center = mean_a - mean_b
    if method in _T_TESTS:
        if method is TestMethod.PAIRED_T:
            diff = x - y
            se = float(diff.std(ddof=1) / np.sqrt(n_a)) if n_a > 1 else nan
            df = float(n_a - 1) if n_a > 1 else nan
        elif method is TestMethod.WELCH_T:
            se = welch_se(sd_a**2, n_a, sd_b**2, n_b)
            df = welch_df(sd_a**2, n_a, sd_b**2, n_b)
        elif n_a + n_b > 2:
            df = float(n_a + n_b - 2)
            pooled = ((n_a - 1) * sd_a**2 + (n_b - 1) * sd_b**2) / df
            se = float(np.sqrt(pooled * (1.0 / n_a + 1.0 / n_b)))
        else:
            se = df = nan
        if se == 0.0:
            df = nan
        statistic = _t_statistic(center, se, df)
        p_value = _p_from_t(statistic, df, alternative)
        low, high = _interval(center, se, df, ci_level, alternative)
        ci = (_back(low, scale), _back(high, scale))
        effect = _back(center, scale)
    elif method is TestMethod.MANN_WHITNEY:
        res = stats.mannwhitneyu(x, y, alternative=str(alternative))
        statistic, p_value, df = float(res.statistic), float(res.pvalue), nan
        effect = _median_effect(a, b, scale)
        ci = (nan, nan)
    elif method is TestMethod.WILCOXON:
        # two identical samples have no spread of the differences to divide by
        with np.errstate(invalid="ignore", divide="ignore"):
            res = stats.wilcoxon(x, y, alternative=str(alternative))
        statistic, p_value, df = float(res.statistic), float(res.pvalue), nan
        effect = _median_effect(a, b, scale)
        ci = (nan, nan)
    else:
        res = stats.permutation_test(
            (x, y),
            _mean_difference,
            permutation_type="samples" if paired else "independent",
            n_resamples=n_perm,
            alternative=str(alternative),
            rng=np.random.default_rng(seed),
        )
        statistic, p_value, df = float(res.statistic), float(res.pvalue), nan
        effect = _back(center, scale)
        ci = (nan, nan)
    return TestResult(
        test=method,
        statistic=float(statistic),
        p_value=p_value,
        effect=effect,
        ci_low=ci[0],
        ci_high=ci[1],
        ci_level=ci_level,
        scale=scale,
        alternative=alternative,
        paired=paired,
        df=df,
        cohen_d=d,
        hedges_g=g,
        n_a=n_a,
        n_b=n_b,
        name=a.name,
        unit=a.unit,
    )


def _welch_from_moments(
    a: ParameterSample,
    b: ParameterSample,
    scale: Scale,
    alternative: Alternative,
    ci_level: float,
) -> TestResult:
    """Welch t test from the moments of two samples (summary data).

    Args:
        a: the first sample.
        b: the second sample.
        scale: scale of the analysis (log-normal moments on the log scale).
        alternative: the alternative hypothesis.
        ci_level: level of the interval.

    Returns:
        The result.
    """
    mean_a, sd_a, n_a = a.moments(scale)
    mean_b, sd_b, n_b = b.moments(scale)
    if n_a < 1 or n_b < 1:
        logger.debug(
            "'%s' or '%s' has no finite value, the comparison is NaN", a.name, b.name
        )
        return _nan_result(
            TestMethod.WELCH_T, scale, alternative, False, ci_level, n_a, n_b, a
        )
    se = welch_se(sd_a**2, n_a, sd_b**2, n_b)
    df = welch_df(sd_a**2, n_a, sd_b**2, n_b)
    center = mean_a - mean_b
    statistic = _t_statistic(center, se, df)
    low, high = _interval(center, se, df, ci_level, alternative)
    d, g = cohen_d(mean_a, sd_a, n_a, mean_b, sd_b, n_b)
    return TestResult(
        test=TestMethod.WELCH_T,
        statistic=float(statistic),
        p_value=_p_from_t(statistic, df, alternative),
        effect=_back(center, scale),
        ci_low=_back(low, scale),
        ci_high=_back(high, scale),
        ci_level=ci_level,
        scale=scale,
        alternative=alternative,
        paired=False,
        df=df,
        cohen_d=d,
        hedges_g=g,
        n_a=n_a,
        n_b=n_b,
        name=a.name,
        unit=a.unit,
    )


def multiple_comparison(
    p_values: ArrayLike, *, method: AdjustMethod | str = AdjustMethod.HOLM
) -> np.ndarray:
    r"""Adjust p values for multiple comparisons.

    Bonferroni: \(\min(1, m p_i)\). Holm (step-down): sort ascending,
    \(\tilde p_{(i)} = \max_{j \le i} \min(1, (m - j + 1) p_{(j)})\).
    Benjamini-Hochberg (step-up): \(\tilde p_{(i)} = \min_{j \ge i} \min(1, m p_{(j)} / j)\).

    Args:
        p_values: the p values.

    Keyword Args:
        method: the adjustment, as the member or as its string.

    Returns:
        The adjusted p values in the order of the input.

    Raises:
        ValueError: if `method` is not an `AdjustMethod`.
    """
    resolved = coerce(method, AdjustMethod)
    p = np.asarray(p_values, dtype=np.float64).ravel()
    m = p.size
    if m == 0:
        return p
    if resolved is AdjustMethod.BONFERRONI:
        return np.minimum(1.0, m * p)
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    adjusted = np.empty(m)
    if resolved is AdjustMethod.HOLM:
        stepped = np.minimum(1.0, (m - ranks + 1) * p[order])
        adjusted[order] = np.maximum.accumulate(stepped)
    elif resolved is AdjustMethod.BH:
        stepped = np.minimum(1.0, m * p[order] / ranks)
        adjusted[order] = np.minimum.accumulate(stepped[::-1])[::-1]
    else:
        # unreachable today, `coerce` rejects anything but the three members;
        # a new member without a branch here must not fall through to another
        raise ValueError(f"'{resolved}' is not an implemented AdjustMethod")
    return adjusted
