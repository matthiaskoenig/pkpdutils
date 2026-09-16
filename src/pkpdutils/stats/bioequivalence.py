r"""Average bioequivalence: the two one-sided tests on the geometric mean ratio.

Test and reference are bioequivalent when the 90 % confidence interval of
the geometric mean ratio of the exposure lies within 80-125 % (FDA 2026, EMA 2010, ICH M13A 2024),
which is the two one-sided tests procedure of Schuirmann (1987) at
\(\alpha = 0.05\). The interval comes from the design of the study: a 2x2
crossover (each subject receives both formulations in two periods, in one
of two sequences) is analysed with the period differences of Chow & Liu
(2009, ch. 3), which is the analysis of variance with sequence, period and
subject-within-sequence effects on the log scale; a paired design uses the
within-subject differences; parallel groups use the Welch interval.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from pkpdutils.result import ParameterResult
from pkpdutils.stats.ratio import RatioResult, ratio
from pkpdutils.stats.sample import (
    ParameterSample,
    coerce,
    exp_t_interval,
    labels_match,
    log_positive,
    paired_indices,
)

logger = logging.getLogger(__name__)


class Design(StrEnum):
    """Design of a bioequivalence study."""

    #: two groups of different subjects
    PARALLEL = "parallel"
    #: every subject receives both formulations, without period information
    PAIRED = "paired"
    #: 2x2 crossover with the coordinates `period` and `sequence`
    CROSSOVER = "crossover"


@dataclass(frozen=True)
class BEParameter:
    r"""Bioequivalence of one parameter.

    Attributes:
        name: name of the parameter
        unit: unit of the parameter
        gmr: geometric mean ratio test / reference
        ci_low: lower bound of the interval of the ratio
        ci_high: upper bound of the interval
        ci_level: level of the interval
        limits: acceptance limits of the ratio
        bioequivalent: whether the interval lies within the limits
        p_lower: p value of the test against the lower limit
        p_upper: p value of the test against the upper limit
        p_value: the larger of the two, the p value of the TOST procedure
        log_ratio: \(\ln \mathrm{GMR}\)
        se_log: standard error of `log_ratio`
        df: degrees of freedom
        design: the design of the analysis
        cv_intra: within-subject coefficient of variation \(\sqrt{e^{\sigma_e^2} - 1}\), `NaN` for a parallel design
        p_period: p value of the period effect (crossover), `NaN` otherwise
        p_sequence: p value of the sequence (carryover) effect (crossover), `NaN` otherwise
        n_test: number of test values
        n_reference: number of reference values
    """

    name: str
    unit: str
    gmr: float
    ci_low: float
    ci_high: float
    ci_level: float
    limits: tuple[float, float]
    bioequivalent: bool
    p_lower: float
    p_upper: float
    p_value: float
    log_ratio: float
    se_log: float
    df: float
    design: Design
    cv_intra: float
    p_period: float
    p_sequence: float
    n_test: int
    n_reference: int

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dictionary with the enumerations as strings.

        Returns:
            Field name to value.
        """
        return {
            "parameter": self.name,
            "unit": self.unit,
            "gmr": self.gmr,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "limit_low": self.limits[0],
            "limit_high": self.limits[1],
            "bioequivalent": self.bioequivalent,
            "p_lower": self.p_lower,
            "p_upper": self.p_upper,
            "p_value": self.p_value,
            "log_ratio": self.log_ratio,
            "se_log": self.se_log,
            "df": self.df,
            "design": str(self.design),
            "cv_intra": self.cv_intra,
            "p_period": self.p_period,
            "p_sequence": self.p_sequence,
            "n_test": self.n_test,
            "n_reference": self.n_reference,
        }


@dataclass(frozen=True)
class BEResult:
    """Bioequivalence of several parameters.

    Attributes:
        parameters: parameter name to its result
        bioequivalent: whether every parameter is bioequivalent
        limits: the acceptance limits
        ci_level: the level of the intervals
    """

    parameters: dict[str, BEParameter]
    bioequivalent: bool
    limits: tuple[float, float]
    ci_level: float

    def __getitem__(self, name: str) -> BEParameter:
        """The result of one parameter.

        Args:
            name: name of the parameter.

        Returns:
            The result.
        """
        return self.parameters[name]

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """The result of every parameter as a dictionary, keyed by its name.

        Returns:
            Parameter name to the fields of its `BEParameter`.
        """
        return {name: p.to_dict() for name, p in self.parameters.items()}

    def to_dataframe(self) -> pd.DataFrame:
        """One row per parameter.

        Returns:
            The dataframe.
        """
        return pd.DataFrame([p.to_dict() for p in self.parameters.values()])


def _detect_design(test: ParameterSample, reference: ParameterSample) -> Design:
    """The design the samples allow.

    Args:
        test: the test sample.
        reference: the reference sample.

    Returns:
        `CROSSOVER` with `period` and `sequence` coordinates and matching
        labels, `PAIRED` with matching labels, else `PARALLEL`.
    """
    if not labels_match(test, reference):
        return Design.PARALLEL
    keys = {"period", "sequence"}
    if keys <= set(test.coords) and keys <= set(reference.coords):
        return Design.CROSSOVER
    return Design.PAIRED


def _crossover(
    test: ParameterSample, reference: ParameterSample
) -> tuple[float, float, float, float, float, float, int]:
    r"""The period-difference analysis of a 2x2 crossover on the log scale.

    For subject \(i\) with the log values \(y_{i1}, y_{i2}\) of the two periods,
    \(d_i = (y_{i2} - y_{i1}) / 2\) and the total \(u_i = y_{i1} + y_{i2}\). With
    the sequences A (test in period 2) and B (test in period 1):
    treatment effect \(\hat F = \bar d_A - \bar d_B\), period effect
    \(\hat P = \bar d_A + \bar d_B\), both with the variance
    \(\sigma_d^2 (1/n_A + 1/n_B)\) of the pooled within-sequence variance
    \(\sigma_d^2\) with \(n_A + n_B - 2\) degrees of freedom; the sequence
    (carryover) effect \(\hat C = \bar u_A - \bar u_B\) with the pooled
    variance of the totals. The residual variance of the ANOVA is
    \(\sigma_e^2 = 2 \sigma_d^2\) (Chow & Liu 2009, ch. 3).

    The subjects are the pairs of `paired_indices`, so a subject with a
    missing value in either period is left out of the analysis.

    Args:
        test: the test sample with `period` and `sequence` coordinates.
        reference: the reference sample with `period` and `sequence` coordinates.

    Returns:
        `log_ratio`, `se_log`, `df`, `sigma_e2`, `p_period`, `p_sequence`
        and the number of subjects of the analysis.

    Raises:
        ValueError: if a subject has no two different periods, the periods
            are not 1 and 2, the `sequence` coordinate does not agree
            between the test and the reference sample, a sequence mixes the
            order, there are not exactly two sequences with at least two
            subjects each, or both sequences have the test in the same period.
    """
    index_t, index_r = paired_indices(test, reference)
    assert test.values is not None and reference.values is not None
    x = log_positive(test.values[index_t], test.name)
    y = log_positive(reference.values[index_r], reference.name)
    period_t = np.asarray(test.coords["period"])[index_t]
    sequence = np.asarray(test.coords["sequence"])[index_t]
    period_r = np.asarray(reference.coords["period"])[index_r]
    sequence_r = np.asarray(reference.coords["sequence"])[index_r]
    if set(np.unique(period_t).tolist()) | set(np.unique(period_r).tolist()) != {
        1,
        2,
    } or np.any(period_t == period_r):
        raise ValueError(
            "A 2x2 crossover needs the periods 1 and 2, test and reference in different periods of every subject"
        )
    if not np.array_equal(sequence, sequence_r):
        raise ValueError(
            "A 2x2 crossover needs the same 'sequence' coordinate on the test and the reference sample"
        )
    sequences = np.unique(sequence)
    if sequences.size != 2:
        raise ValueError(
            f"A 2x2 crossover needs exactly two sequences, got {sequences.tolist()}"
        )
    y1 = np.where(period_t == 1, x, y)
    y2 = np.where(period_t == 2, x, y)
    d = (y2 - y1) / 2.0
    u = y1 + y2
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    test_second: list[bool] = []
    for seq in sequences:
        mask = sequence == seq
        in_second = np.unique(period_t[mask])
        if in_second.size != 1:
            raise ValueError(f"Sequence '{seq}' mixes the order of test and reference")
        if mask.sum() < 2:
            raise ValueError(f"Sequence '{seq}' needs at least two subjects")
        groups.append((d[mask], u[mask]))
        test_second.append(bool(in_second[0] == 2))
    if test_second[0] == test_second[1]:
        raise ValueError(
            "A 2x2 crossover needs the test in period 2 in one sequence and in period 1 in the other"
        )
    (d_a, u_a), (d_b, u_b) = groups if test_second[0] else groups[::-1]
    n_a, n_b = d_a.size, d_b.size
    df = float(n_a + n_b - 2)
    factor = 1.0 / n_a + 1.0 / n_b
    sigma_d2 = ((n_a - 1) * d_a.var(ddof=1) + (n_b - 1) * d_b.var(ddof=1)) / df
    sigma_u2 = ((n_a - 1) * u_a.var(ddof=1) + (n_b - 1) * u_b.var(ddof=1)) / df
    se = float(np.sqrt(sigma_d2 * factor))
    effect = float(d_a.mean() - d_b.mean())
    period = float(d_a.mean() + d_b.mean())
    carryover = float(u_a.mean() - u_b.mean())
    se_u = float(np.sqrt(sigma_u2 * factor))
    p_period = float(2.0 * student_t.sf(abs(period / se), df))
    p_sequence = float(2.0 * student_t.sf(abs(carryover / se_u), df))
    return effect, se, df, float(2.0 * sigma_d2), p_period, p_sequence, int(x.size)


def tost(
    test: ParameterSample,
    reference: ParameterSample,
    *,
    limits: tuple[float, float] = (0.8, 1.25),
    ci_level: float = 0.90,
    design: Design | str | None = None,
) -> BEParameter:
    r"""Two one-sided tests of the geometric mean ratio against the acceptance limits.

    \(t_L = (\ln \mathrm{GMR} - \ln \theta_L) / \mathrm{se}\),
    \(t_U = (\ln \theta_U - \ln \mathrm{GMR}) / \mathrm{se}\), each tested one-sided
    with the degrees of freedom of the design at \(\alpha = (1 - \mathrm{ci\_level}) / 2\);
    rejecting both is the same as the interval at `ci_level` lying within
    the limits (Schuirmann 1987). Without a standard error (a single
    subject, or two samples without a within-subject difference) the two
    tests are undefined: the p values and the interval are `NaN` and the
    parameter is not bioequivalent, as in `compare`.

    Args:
        test: the test sample.
        reference: the reference sample.
        limits: acceptance limits of the ratio.
        ci_level: level of the interval, 0.90 for the usual \(\alpha = 0.05\).
        design: the design, as the member or as its string, detected from
            the samples by default.

    Returns:
        The result of the parameter.

    Raises:
        ValueError: for reversed limits, an unknown design or a design the
            samples do not support.
    """
    if not 0 < limits[0] < limits[1]:
        raise ValueError(
            f"'limits' must be (low, high) with 0 < low < high, got {limits}"
        )
    resolved = (
        coerce(design, Design)
        if design is not None
        else _detect_design(test, reference)
    )
    nan = float("nan")
    p_period = p_sequence = cv_intra = nan
    if resolved is Design.CROSSOVER:
        if not (
            {"period", "sequence"} <= set(test.coords)
            and {"period", "sequence"} <= set(reference.coords)
        ):
            raise ValueError(
                "A crossover analysis needs the coordinates 'period' and 'sequence' on both samples"
            )
        log_ratio, se, df, sigma_e2, p_period, p_sequence, n_subjects = _crossover(
            test, reference
        )
        cv_intra = float(np.sqrt(np.expm1(sigma_e2)))
        n_test = n_reference = n_subjects
    else:
        r: RatioResult = ratio(
            test, reference, ci_level=ci_level, paired=resolved is Design.PAIRED
        )
        log_ratio, se, df = r.log_ratio, r.se_log, r.df
        n_test, n_reference = r.n_test, r.n_reference
        if resolved is Design.PAIRED:
            # the variance of a within-subject difference is twice the residual variance
            cv_intra = float(np.sqrt(np.expm1(se**2 * n_test / 2.0)))
    if se > 0 and np.isfinite(se) and df > 0:
        ci = exp_t_interval(log_ratio, se, df, ci_level)
        p_lower = float(student_t.sf((log_ratio - np.log(limits[0])) / se, df))
        p_upper = float(student_t.sf((np.log(limits[1]) - log_ratio) / se, df))
    else:
        logger.debug(
            "'%s' has no standard error of the log ratio, the two one-sided tests are NaN",
            test.name,
        )
        ci = (nan, nan)
        p_lower = p_upper = nan
    return BEParameter(
        name=test.name,
        unit=test.unit,
        gmr=float(np.exp(log_ratio)),
        ci_low=ci[0],
        ci_high=ci[1],
        ci_level=ci_level,
        limits=(float(limits[0]), float(limits[1])),
        bioequivalent=bool(limits[0] <= ci[0] and ci[1] <= limits[1]),
        p_lower=p_lower,
        p_upper=p_upper,
        p_value=float(max(p_lower, p_upper)),
        log_ratio=log_ratio,
        se_log=se,
        df=df,
        design=resolved,
        cv_intra=cv_intra,
        p_period=p_period,
        p_sequence=p_sequence,
        n_test=int(n_test),
        n_reference=int(n_reference),
    )


def bioequivalence(
    test: ParameterResult,
    reference: ParameterResult,
    parameters: Sequence[str] = ("auc_inf_obs", "cmax"),
    *,
    dim: str = "individual",
    limits: tuple[float, float] = (0.8, 1.25),
    ci_level: float = 0.90,
    design: Design | str | None = None,
    **indexers: Any,
) -> BEResult:
    """Average bioequivalence of the parameters of two results.

    Every parameter is taken with `ParameterResult.sample(name, dim, **indexers)`
    from both results and tested with `tost`; the study is bioequivalent
    when every parameter is.

    Args:
        test: the result of the test formulation.
        reference: the result of the reference formulation.
        parameters: the parameters to test.
        dim: the sample dimension of the individuals.
        limits: acceptance limits of the ratio.
        ci_level: level of the intervals.
        design: the design, as the member or as its string, detected from
            the samples by default.
        **indexers: coordinate label per remaining sample dimension.

    Returns:
        The result.

    Raises:
        ValueError: for an unknown design, or as `tost`.
    """
    if design is not None:
        design = coerce(design, Design)
    results = {
        name: tost(
            test.sample(name, dim, **indexers),
            reference.sample(name, dim, **indexers),
            limits=limits,
            ci_level=ci_level,
            design=design,
        )
        for name in parameters
    }
    return BEResult(
        parameters=results,
        bioequivalent=all(p.bioequivalent for p in results.values()),
        limits=(float(limits[0]), float(limits[1])),
        ci_level=ci_level,
    )
