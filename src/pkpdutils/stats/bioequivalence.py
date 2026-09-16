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

A replicate design, in which at least one formulation is given twice
(`Design.REPLICATE`, the sequences TRTR/RTRT, TRT/RTR and TRRT/RTTR), is
analysed with the fixed effects analysis of variance of the log values with
sequence, subject within sequence, period and formulation, which is Method A
of the EMA; it separates the within-subject variability of the reference
(`cv_intra_r`) from the one of the test (`cv_intra_t`) and is what the
reference-scaled acceptance criteria need. `bioequivalence(..., scaling=...)`
applies them: the average bioequivalence with expanding limits of the EMA
(ABEL), the reference-scaled average bioequivalence of the FDA (RSABE) and
the two narrow therapeutic index rules. The mixed model (Method B of the EMA,
the FDA model) is out of scope; it needs a restricted maximum likelihood fit
which the dependencies of the package do not carry.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import f as f_distribution
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
from pkpdutils.timecourse import Timecourses

logger = logging.getLogger(__name__)


class Design(StrEnum):
    """Design of a bioequivalence study."""

    #: two groups of different subjects
    PARALLEL = "parallel"
    #: every subject receives both formulations, without period information
    PAIRED = "paired"
    #: 2x2 crossover with the coordinates `period` and `sequence`
    CROSSOVER = "crossover"
    #: replicate crossover, at least one formulation given twice per subject
    REPLICATE = "replicate"


#: the replicate sequences the analysis knows, one set per design: the four
#: period full replicate (2x2x4), the three period replicate (2x2x3) and the
#: four period replicate with the reference in the middle
REPLICATE_DESIGNS: tuple[frozenset[str], ...] = (
    frozenset({"TRTR", "RTRT"}),
    frozenset({"TRT", "RTR"}),
    frozenset({"TRRT", "RTTR"}),
)

#: the letter of the test formulation in a sequence
TEST_LETTER: str = "T"

#: the letter of the reference formulation in a sequence
REFERENCE_LETTER: str = "R"

#: the regulatory constant of the expanding limits of the EMA (ABEL),
#: \(\theta_L = e^{-k s_{wR}}\) with \(k = 0.760\) (EMA 2010, 4.1.10)
EMA_SCALING_K: float = 0.760

#: the within-subject CV of the reference above which the EMA widens the
#: limits of \(C_\mathrm{max}\), 30 % (EMA 2010, 4.1.10)
EMA_SCALING_CV_MIN: float = 0.30

#: the within-subject CV of the reference at which the widening stops, 50 %
EMA_SCALING_CV_CAP: float = 0.50

#: the widest limits the EMA allows, reached at `EMA_SCALING_CV_CAP`
EMA_ABEL_LIMITS: tuple[float, float] = (0.6984, 1.4319)

#: the tightened limits of a narrow therapeutic index drug, 90.00-111.11 %
#: (EMA 2010, 4.1.9)
EMA_NTI_LIMITS: tuple[float, float] = (0.9000, 1.1111)

#: the point estimate of a reference-scaled analysis has to lie within these
#: limits as well (EMA 2010, 4.1.10; FDA progesterone guidance 2011)
POINT_ESTIMATE_LIMITS: tuple[float, float] = (0.8000, 1.2500)

#: the within-subject standard deviation of the reference above which the FDA
#: scales, the switching condition of RSABE (FDA progesterone guidance 2011)
FDA_SWR_CUTOFF: float = 0.294

#: the regulatory standard deviation of the FDA scaled criterion
FDA_SIGMA_W0: float = 0.25

#: the bioequivalence limit the FDA scaled criterion is built from
FDA_DELTA: float = 1.25

#: the regulatory standard deviation of the FDA narrow therapeutic index
#: criterion (FDA warfarin guidance)
FDA_NTI_SIGMA_W0: float = 0.10

#: the bioequivalence limit of the FDA narrow therapeutic index criterion,
#: \(1 / 0.9\), whose implied limits at \(\sigma_{wR} = 0.10\) are 90.00-111.11 %
FDA_NTI_DELTA: float = 1.11111

#: the largest upper 90 % bound of \(s_{wT} / s_{wR}\) the FDA accepts for a
#: narrow therapeutic index drug (FDA warfarin guidance)
FDA_NTI_SD_RATIO_MAX: float = 2.500

#: the scaling rules of `tost` and `bioequivalence`
Scaling = Literal["none", "ema", "fda", "fda_nti", "ema_nti"]


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
        carryover: the subjects whose pre-dose concentration exceeds the
            carryover threshold (`carryover_table`), either flagged here or,
            with `carryover="exclude"`, already dropped from the analysis
        cv_intra_r: within-subject CV of the reference formulation alone, from
            its replicates; `NaN` unless the design is `REPLICATE`
        cv_intra_t: within-subject CV of the test formulation alone; `NaN`
            unless the test is replicated too
        scaled: whether the acceptance limits or the criterion were derived
            from the variability of the reference (`scaling`)
        limits_scaled: the derived limits, `None` for an unscaled analysis and
            for the criterion of the FDA, which has no limits; `limits` always
            carries the limits the verdict was taken against
        criterion: the upper confidence bound of the FDA scaled criterion,
            \(\le 0\) for a bioequivalent formulation; `None` for every other
            analysis
        sd_ratio_upper: the upper 90 % bound of \(s_{wT}/s_{wR}\) of the
            narrow therapeutic index criterion of the FDA, `NaN` otherwise
        anova: the analysis of variance table of a replicate design (source,
            `df`, `sum_sq`, `mean_sq`, `f`, `p_value`), `None` otherwise
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
    carryover: tuple[str, ...] = ()
    cv_intra_r: float = float("nan")
    cv_intra_t: float = float("nan")
    scaled: bool = False
    limits_scaled: tuple[float, float] | None = None
    criterion: float | None = None
    sd_ratio_upper: float = float("nan")
    anova: pd.DataFrame | None = None

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
            "carryover": ", ".join(self.carryover),
            "cv_intra_r": self.cv_intra_r,
            "cv_intra_t": self.cv_intra_t,
            "scaled": self.scaled,
            "limit_scaled_low": (
                None if self.limits_scaled is None else self.limits_scaled[0]
            ),
            "limit_scaled_high": (
                None if self.limits_scaled is None else self.limits_scaled[1]
            ),
            "criterion": self.criterion,
            "sd_ratio_upper": self.sd_ratio_upper,
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
        `REPLICATE` when the `sequence` coordinate names one of
        `REPLICATE_DESIGNS`, `CROSSOVER` with `period` and `sequence`
        coordinates and matching labels, `PAIRED` with matching labels, else
        `PARALLEL`.
    """
    if _is_replicate(test, reference):
        return Design.REPLICATE
    if not labels_match(test, reference):
        return Design.PARALLEL
    keys = {"period", "sequence"}
    if keys <= set(test.coords) and keys <= set(reference.coords):
        return Design.CROSSOVER
    return Design.PAIRED


def _is_replicate(test: ParameterSample, reference: ParameterSample) -> bool:
    """Whether the sequences of two samples name a replicate design.

    A replicate study repeats a formulation, so its samples cannot be matched
    by label: what identifies the subject of an administration is the
    `subject` coordinate (or the label, when a sample carries one label per
    subject). The design is recognized by its sequences alone.

    Args:
        test: the test sample.
        reference: the reference sample.

    Returns:
        `True` when both samples carry `period` and `sequence` and the
        sequences of both belong to one of `REPLICATE_DESIGNS`.
    """
    keys = {"period", "sequence"}
    if not (test.is_individual and reference.is_individual):
        return False
    if not (keys <= set(test.coords) and keys <= set(reference.coords)):
        return False
    sequences = set(np.asarray(test.coords["sequence"]).astype(str).tolist())
    sequences |= set(np.asarray(reference.coords["sequence"]).astype(str).tolist())
    return len(sequences) >= 2 and any(
        sequences <= design for design in REPLICATE_DESIGNS
    )


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


@dataclass(frozen=True)
class _ReplicateFit:
    r"""The fixed effects analysis of variance of a replicate design.

    Attributes:
        log_ratio: the formulation effect \(\ln \mathrm{GMR}\)
        se: its standard error from the residual variance
        df: degrees of freedom of the residual
        sigma_e2: the residual variance, pooled over both formulations
        p_period: p value of the period effect against the residual
        p_sequence: p value of the sequence effect against subject(sequence)
        n_test: number of test observations
        n_reference: number of reference observations
        n_subjects: number of subjects of the analysis
        s2_wr: within-subject variance of the reference alone
        df_wr: its degrees of freedom
        s2_wt: within-subject variance of the test alone
        df_wt: its degrees of freedom
        difference: the subject-level estimate of \(\ln \mathrm{GMR}\) of the
            scaled criterion, the least squares mean of the within-subject
            differences over the sequences
        se_difference: its standard error
        df_difference: its degrees of freedom, the number of subjects minus
            the number of sequences
        anova: the analysis of variance table
    """

    log_ratio: float
    se: float
    df: float
    sigma_e2: float
    p_period: float
    p_sequence: float
    n_test: int
    n_reference: int
    n_subjects: int
    s2_wr: float
    df_wr: float
    s2_wt: float
    df_wt: float
    difference: float
    se_difference: float
    df_difference: float
    anova: pd.DataFrame


def _dummies(values: np.ndarray) -> np.ndarray:
    """Reference-cell dummy columns of a categorical array, the first level dropped.

    Args:
        values: the level of every observation.

    Returns:
        One column per level but the first, with no column for a single level.
    """
    levels = np.unique(values)
    if levels.size < 2:
        return np.zeros((values.size, 0))
    return np.column_stack([(values == level).astype(float) for level in levels[1:]])


def _least_squares(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Ordinary least squares of a design matrix which may be rank deficient.

    Args:
        x: the design matrix.
        y: the observations.

    Returns:
        The minimum norm solution, the residual sum of squares and the rank
        of the design matrix.
    """
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    return beta, float(residual @ residual), int(rank)


def _within_variance(
    values: np.ndarray, subjects: np.ndarray, periods: np.ndarray
) -> tuple[float, float]:
    r"""The within-subject variance of one formulation from its replicates.

    The residual of the ordinary least squares of the log values of that
    formulation alone on the subject and the period, which for the two
    administrations of a full replicate design is the pooled within-sequence
    variance of their difference,
    \(s_w^2 = \frac{1}{2}\,\mathrm{var}(R_{i2} - R_{i1})\) with
    \(n - s\) degrees of freedom, the estimator the FDA guidance defines. A
    subject with a single administration contributes no degree of freedom.

    Args:
        values: the log values of the formulation.
        subjects: the subject of every value.
        periods: the period of every value.

    Returns:
        The variance and its degrees of freedom, `(NaN, 0)` when the
        formulation is not replicated.
    """
    if values.size == 0:
        return float("nan"), 0.0
    x = np.column_stack([np.ones(values.size), _dummies(subjects), _dummies(periods)])
    _, rss, rank = _least_squares(x, values)
    df = float(values.size - rank)
    if df < 1:
        return float("nan"), 0.0
    return rss / df, df


def _subject_differences(
    values: np.ndarray,
    subjects: np.ndarray,
    sequences: np.ndarray,
    is_test: np.ndarray,
) -> tuple[float, float, float]:
    r"""The within-subject difference of the formulations, averaged over the sequences.

    For every subject \(i\), \(d_i = \overline{\ln T_i} - \overline{\ln R_i}\);
    the estimate is the least squares mean \(\hat d = \frac{1}{s}\sum_k \bar d_k\)
    over the \(s\) sequences, its variance
    \(\frac{s_d^2}{s^2}\sum_k \frac{1}{n_k}\) with the pooled within-sequence
    variance \(s_d^2\) and \(n - s\) degrees of freedom. This is the
    subject-level analysis the FDA scaled criterion is evaluated on.

    Args:
        values: the log values.
        subjects: the subject of every value.
        sequences: the sequence of every value.
        is_test: whether a value belongs to the test formulation.

    Returns:
        The estimate, its standard error and its degrees of freedom.
    """
    differences: list[float] = []
    groups: list[str] = []
    for subject in np.unique(subjects):
        rows = subjects == subject
        test_rows, reference_rows = rows & is_test, rows & ~is_test
        if not test_rows.any() or not reference_rows.any():
            continue
        differences.append(
            float(values[test_rows].mean() - values[reference_rows].mean())
        )
        groups.append(str(sequences[rows][0]))
    d = np.asarray(differences, dtype=np.float64)
    group = np.asarray(groups)
    labels = np.unique(group)
    sizes = np.array([int((group == label).sum()) for label in labels])
    df = float(d.size - labels.size)
    if df < 1 or np.any(sizes < 2):
        return float(d.mean()) if d.size else float("nan"), float("nan"), df
    pooled = (
        sum(
            (sizes[k] - 1) * float(d[group == label].var(ddof=1))
            for k, label in enumerate(labels)
        )
        / df
    )
    estimate = float(np.mean([d[group == label].mean() for label in labels]))
    variance = pooled / labels.size**2 * float(np.sum(1.0 / sizes))
    return estimate, float(np.sqrt(variance)), df


def _replicate_observations(
    test: ParameterSample, reference: ParameterSample
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The observations of both formulations of a replicate design, as one table.

    Args:
        test: the test sample, one row per test administration, with the
            coordinates `period` and `sequence` and either a `subject`
            coordinate or one label per subject.
        reference: the reference sample, the same way.

    Returns:
        The log values, the subject, the period, the sequence and whether the
        observation is a test administration, one entry per observation.

    Raises:
        ValueError: if a sample names no subject or not both coordinates, if
            the sequences are not one of `REPLICATE_DESIGNS`, or if a period
            of a subject does not carry the formulation its sequence names.
    """
    columns: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = []
    for sample, letter in ((test, TEST_LETTER), (reference, REFERENCE_LETTER)):
        if sample.values is None:
            raise ValueError(
                f"'{sample.name}' needs individual values for a replicate design"
            )
        missing = {"period", "sequence"} - set(sample.coords)
        if missing:
            raise ValueError(
                f"'{sample.name}' is missing the coordinates {sorted(missing)}, "
                "a replicate design needs 'period' and 'sequence'"
            )
        # a replicate study has several administrations per subject, so the
        # sample dimension is usually the administration and the `subject`
        # coordinate names the individual; a sample whose labels are the
        # subjects works as well
        if "subject" in sample.coords:
            subject = np.asarray(sample.coords["subject"])
        elif sample.labels is not None:
            subject = np.asarray(sample.labels)
        else:
            raise ValueError(
                f"'{sample.name}' names no subject, give a 'subject' coordinate "
                "or labels which name the subject of every administration"
            )
        keep = np.isfinite(sample.values)
        columns.append(
            (
                sample.values[keep],
                subject[keep],
                np.asarray(sample.coords["period"])[keep],
                np.asarray(sample.coords["sequence"])[keep].astype(str),
                letter == TEST_LETTER,
            )
        )
    values = np.concatenate([c[0] for c in columns])
    subjects = np.concatenate([c[1] for c in columns]).astype(str)
    periods = np.concatenate([c[2] for c in columns]).astype(int)
    sequences = np.concatenate([c[3] for c in columns])
    is_test = np.concatenate([np.full(c[0].size, c[4], dtype=bool) for c in columns])
    observed = set(np.unique(sequences).tolist())
    if not any(observed <= design for design in REPLICATE_DESIGNS) or len(observed) < 2:
        known = " or ".join("/".join(sorted(design)) for design in REPLICATE_DESIGNS)
        raise ValueError(
            f"the sequences {sorted(observed)} are no replicate design, use {known}"
        )
    for sequence, period, test_row in zip(sequences, periods, is_test, strict=True):
        if not 1 <= period <= len(sequence):
            raise ValueError(
                f"period {period} is outside the sequence '{sequence}' of "
                f"{len(sequence)} periods"
            )
        letter = TEST_LETTER if test_row else REFERENCE_LETTER
        if sequence[period - 1] != letter:
            raise ValueError(
                f"sequence '{sequence}' has '{sequence[period - 1]}' in period "
                f"{period}, the sample says '{letter}'"
            )
    return log_positive(values, test.name), subjects, periods, sequences, is_test


def _replicate(test: ParameterSample, reference: ParameterSample) -> _ReplicateFit:
    r"""The fixed effects analysis of variance of a replicate design (EMA Method A).

    The model of the log values is

    $$\ln y_{ijk} = \mu + \gamma_k + s_{i(k)} + \pi_j + \tau_{f} + e_{ijk}$$

    with the sequence \(\gamma\), the subject within sequence \(s\), the
    period \(\pi\) and the formulation \(\tau\), fitted by ordinary least
    squares; the formulation effect is \(\ln \mathrm{GMR}\) and its standard
    error comes from the residual variance, which is what the EMA calls
    Method A and asks to be the default analysis of a replicate study. The
    table of the sequential (type I) sums of squares is returned with it, the
    sequence tested against the subject-within-sequence mean square and
    everything else against the residual.

    The within-subject variance of each formulation is estimated from its own
    administrations alone (`_within_variance`), which is what the
    reference-scaled criteria of the EMA and the FDA scale with.

    Args:
        test: the test sample, one value per test administration.
        reference: the reference sample, one value per reference administration.

    Returns:
        The fit.

    Raises:
        ValueError: as `_replicate_observations`, or if the design is not
            connected enough to estimate the formulation effect.
    """
    y, subjects, periods, sequences, is_test = _replicate_observations(test, reference)
    formulation = is_test.astype(float).reshape(-1, 1)
    blocks = [
        ("sequence", _dummies(sequences)),
        ("subject(sequence)", _dummies(subjects)),
        ("period", _dummies(periods)),
        ("formulation", formulation),
    ]
    design = np.ones((y.size, 1))
    _, rss, rank = _least_squares(design, y)
    records: list[dict[str, Any]] = []
    for name, block in blocks:
        design = np.column_stack([design, block])
        _, rss_next, rank_next = _least_squares(design, y)
        records.append(
            {
                "source": name,
                "df": rank_next - rank,
                "sum_sq": max(rss - rss_next, 0.0),
            }
        )
        rss, rank = rss_next, rank_next
    df = float(y.size - rank)
    if df < 1:
        raise ValueError(
            f"the replicate design of '{test.name}' leaves no degree of freedom "
            "for the residual, it needs more subjects"
        )
    sigma_e2 = rss / df
    records.append({"source": "residual", "df": int(df), "sum_sq": rss})
    anova = pd.DataFrame.from_records(records)
    anova["mean_sq"] = anova["sum_sq"] / anova["df"].where(anova["df"] > 0)
    subject_ms = float(
        anova.loc[anova["source"] == "subject(sequence)", "mean_sq"].iloc[0]
    )
    denominators = np.where(
        anova["source"] == "sequence",
        subject_ms,
        sigma_e2,
    )
    denominator_df = np.where(
        anova["source"] == "sequence",
        float(anova.loc[anova["source"] == "subject(sequence)", "df"].iloc[0]),
        df,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        f_values = anova["mean_sq"].to_numpy() / denominators
    anova["f"] = np.where(anova["source"] == "residual", np.nan, f_values)
    anova["p_value"] = np.where(
        np.isfinite(anova["f"].to_numpy()) & (anova["df"].to_numpy() > 0),
        f_distribution.sf(
            anova["f"].to_numpy(), anova["df"].to_numpy(), denominator_df
        ),
        np.nan,
    )
    # the full model in a full rank parameterization, for the effect and its
    # standard error; the subject dummies span the sequence effect
    x = np.column_stack(
        [
            np.ones(y.size),
            _dummies(subjects),
            _dummies(periods),
            formulation,
        ]
    )
    beta, _, full_rank = _least_squares(x, y)
    if full_rank != x.shape[1]:
        raise ValueError(
            f"the replicate design of '{test.name}' is not connected, the "
            "formulation effect is not estimable from it"
        )
    covariance = np.linalg.inv(x.T @ x)
    se = float(np.sqrt(sigma_e2 * covariance[-1, -1]))
    s2_wr, df_wr = _within_variance(y[~is_test], subjects[~is_test], periods[~is_test])
    s2_wt, df_wt = _within_variance(y[is_test], subjects[is_test], periods[is_test])
    difference, se_difference, df_difference = _subject_differences(
        y, subjects, sequences, is_test
    )
    row = anova.set_index("source")
    return _ReplicateFit(
        log_ratio=float(beta[-1]),
        se=se,
        df=df,
        sigma_e2=sigma_e2,
        p_period=float(row.loc["period", "p_value"]),
        p_sequence=float(row.loc["sequence", "p_value"]),
        n_test=int(is_test.sum()),
        n_reference=int((~is_test).sum()),
        n_subjects=int(np.unique(subjects).size),
        s2_wr=s2_wr,
        df_wr=df_wr,
        s2_wt=s2_wt,
        df_wt=df_wt,
        difference=difference,
        se_difference=se_difference,
        df_difference=df_difference,
        anova=anova,
    )


def _with_limits(
    parameter: BEParameter,
    limits: tuple[float, float],
    *,
    point_estimate: bool,
) -> BEParameter:
    """The parameter judged against other limits, the two tests recomputed.

    Args:
        parameter: the unscaled result.
        limits: the limits to judge against.

    Keyword Args:
        point_estimate: require the point estimate to lie within
            `POINT_ESTIMATE_LIMITS` as well, which every reference-scaled
            criterion does.

    Returns:
        The result with `limits`, `limits_scaled`, the two p values and the
        verdict of the new limits.
    """
    se, df = parameter.se_log, parameter.df
    if se > 0 and np.isfinite(se) and df > 0:
        p_lower = float(
            student_t.sf((parameter.log_ratio - np.log(limits[0])) / se, df)
        )
        p_upper = float(
            student_t.sf((np.log(limits[1]) - parameter.log_ratio) / se, df)
        )
    else:
        p_lower = p_upper = float("nan")
    within = bool(limits[0] <= parameter.ci_low and parameter.ci_high <= limits[1])
    if point_estimate:
        within = within and bool(
            POINT_ESTIMATE_LIMITS[0] <= parameter.gmr <= POINT_ESTIMATE_LIMITS[1]
        )
    return replace(
        parameter,
        limits=(float(limits[0]), float(limits[1])),
        limits_scaled=(float(limits[0]), float(limits[1])),
        scaled=True,
        bioequivalent=within,
        p_lower=p_lower,
        p_upper=p_upper,
        p_value=float(max(p_lower, p_upper)),
    )


def abel_limits(cv_intra_r: float) -> tuple[float, float]:
    r"""The expanding limits of the EMA for a within-subject CV of the reference.

    $$\theta_{U} = e^{k\,s_{wR}}, \qquad \theta_{L} = 1/\theta_{U},
    \qquad s_{wR} = \sqrt{\ln(1 + \mathrm{CV}_{wR}^2)},$$

    with \(k = 0.760\); the CV is capped at 50 % before it is used, so that
    the limits never leave 69.84-143.19 % (EMA 2010, 4.1.10). Below a CV of
    30 % the EMA does not widen at all and the limits stay 80.00-125.00 %.

    Args:
        cv_intra_r: the within-subject coefficient of variation of the
            reference formulation, as a fraction.

    Returns:
        The lower and the upper acceptance limit.

    Raises:
        ValueError: if `cv_intra_r` is not a finite, non-negative number.
    """
    if not np.isfinite(cv_intra_r) or cv_intra_r < 0:
        raise ValueError(
            f"'cv_intra_r' must be a non-negative number, got {cv_intra_r}"
        )
    if cv_intra_r < EMA_SCALING_CV_MIN:
        return POINT_ESTIMATE_LIMITS
    s_wr = float(np.sqrt(np.log1p(min(cv_intra_r, EMA_SCALING_CV_CAP) ** 2)))
    upper = float(np.exp(EMA_SCALING_K * s_wr))
    return (
        max(1.0 / upper, EMA_ABEL_LIMITS[0]),
        min(upper, EMA_ABEL_LIMITS[1]),
    )


def rsabe_criterion(
    difference: float,
    se_difference: float,
    df_difference: float,
    s2_wr: float,
    df_wr: float,
    *,
    sigma_w0: float = FDA_SIGMA_W0,
    delta: float = FDA_DELTA,
    alpha: float = 0.05,
) -> float:
    r"""The upper confidence bound of the scaled criterion of the FDA.

    The linearized criterion is

    $$(\mu_T - \mu_R)^2 - \theta\,\sigma_{wR}^2 \le 0, \qquad
    \theta = \left(\frac{\ln \Delta}{\sigma_{w0}}\right)^2,$$

    and its upper \(1-\alpha\) confidence bound is Howe's approximation, the
    point estimate plus the root of the squared distances of the one-sided
    bounds of its two parts (FDA progesterone guidance 2011):

    $$U = \hat E + \sqrt{\left(\left(|\hat d| + t_{1-\alpha,\nu_d}\,
    \mathrm{se}_d\right)^2 - \hat d^2\right)^2 +
    \left(\theta s_{wR}^2 - \theta s_{wR}^2
    \frac{\nu_R}{\chi^2_{1-\alpha,\nu_R}}\right)^2},
    \qquad \hat E = \hat d^2 - \theta s_{wR}^2.$$

    The formulations pass the criterion when \(U \le 0\).

    Args:
        difference: the estimate \(\hat d\) of \(\ln \mathrm{GMR}\).
        se_difference: its standard error.
        df_difference: its degrees of freedom.
        s2_wr: the within-subject variance of the reference.
        df_wr: its degrees of freedom.

    Keyword Args:
        sigma_w0: the regulatory standard deviation of the criterion, 0.25
            for a highly variable drug and 0.10 for a narrow therapeutic
            index drug.
        delta: the bioequivalence limit the criterion is built from.
        alpha: the level of the one-sided bound.

    Returns:
        The upper bound `U`, `NaN` when the study carries no estimate of the
        difference or of the reference variance.
    """
    if not (np.isfinite(s2_wr) and df_wr >= 1 and np.isfinite(se_difference)):
        return float("nan")
    theta = (float(np.log(delta)) / sigma_w0) ** 2
    scaled = theta * s2_wr
    estimate = difference**2 - scaled
    bound_difference = (
        abs(difference)
        + float(student_t.ppf(1.0 - alpha, df_difference)) * se_difference
    ) ** 2
    bound_scaled = scaled * df_wr / float(chi2.ppf(1.0 - alpha, df_wr))
    return float(
        estimate
        + np.sqrt(
            (bound_difference - difference**2) ** 2 + (scaled - bound_scaled) ** 2
        )
    )


def _sd_ratio_upper(s2_wt: float, df_wt: float, s2_wr: float, df_wr: float) -> float:
    r"""The upper 90 % bound of \(s_{wT}/s_{wR}\).

    $$\left(\frac{s_{wT}}{s_{wR}}\right)_{\mathrm{upper}} =
    \sqrt{\frac{s_{wT}^2}{s_{wR}^2}\,F_{0.95}(\nu_R, \nu_T)},$$

    the upper bound of the equal-tailed 90 % interval of the ratio of two
    variances, which the FDA compares with 2.500 for a narrow therapeutic
    index drug.

    Args:
        s2_wt: within-subject variance of the test.
        df_wt: its degrees of freedom.
        s2_wr: within-subject variance of the reference.
        df_wr: its degrees of freedom.

    Returns:
        The bound, `NaN` when either formulation is not replicated.
    """
    if not (np.isfinite(s2_wt) and np.isfinite(s2_wr) and df_wt >= 1 and df_wr >= 1):
        return float("nan")
    if not s2_wr > 0:
        return float("inf")
    return float(np.sqrt(s2_wt / s2_wr * f_distribution.ppf(0.95, df_wr, df_wt)))


def _apply_scaling(
    parameter: BEParameter, scaling: Scaling | str, fit: _ReplicateFit | None
) -> BEParameter:
    """Apply a reference-scaled or tightened acceptance rule to a result.

    Args:
        parameter: the unscaled result.
        scaling: the rule.
        fit: the replicate fit, `None` for a design without replicates.

    Returns:
        The result under the rule.

    Raises:
        ValueError: for an unknown rule, or for a rule which needs the
            variability of the reference from a design which has none.
    """
    if scaling == "none":
        return parameter
    if scaling not in ("ema", "fda", "fda_nti", "ema_nti"):
        raise ValueError(
            f"'{scaling}' is not a scaling, use 'none', 'ema', 'fda', 'fda_nti' "
            "or 'ema_nti'"
        )
    if scaling == "ema_nti":
        return _with_limits(parameter, EMA_NTI_LIMITS, point_estimate=False)
    if fit is None or not np.isfinite(fit.s2_wr):
        raise ValueError(
            f"scaling='{scaling}' needs the within-subject variability of the "
            "reference formulation, which only a replicate design estimates"
        )
    if scaling == "ema":
        # the EMA widens the limits of Cmax alone, never those of an area
        if parameter.name != "cmax":
            logger.debug(
                "the EMA widens the limits of 'cmax' only, '%s' stays unscaled",
                parameter.name,
            )
            return parameter
        if parameter.cv_intra_r < EMA_SCALING_CV_MIN:
            return parameter
        return _with_limits(
            parameter, abel_limits(parameter.cv_intra_r), point_estimate=True
        )
    narrow = scaling == "fda_nti"
    s_wr = float(np.sqrt(fit.s2_wr))
    if not narrow and s_wr < FDA_SWR_CUTOFF:
        # not a highly variable drug: the unscaled analysis decides
        return parameter
    criterion = rsabe_criterion(
        fit.difference,
        fit.se_difference,
        fit.df_difference,
        fit.s2_wr,
        fit.df_wr,
        sigma_w0=FDA_NTI_SIGMA_W0 if narrow else FDA_SIGMA_W0,
        delta=FDA_NTI_DELTA if narrow else FDA_DELTA,
    )
    within = bool(POINT_ESTIMATE_LIMITS[0] <= parameter.gmr <= POINT_ESTIMATE_LIMITS[1])
    passed = bool(criterion <= 0.0) and within
    sd_ratio = float("nan")
    if narrow:
        sd_ratio = _sd_ratio_upper(fit.s2_wt, fit.df_wt, fit.s2_wr, fit.df_wr)
        unscaled = bool(
            POINT_ESTIMATE_LIMITS[0] <= parameter.ci_low
            and parameter.ci_high <= POINT_ESTIMATE_LIMITS[1]
        )
        passed = passed and unscaled and bool(sd_ratio <= FDA_NTI_SD_RATIO_MAX)
    return replace(
        parameter,
        scaled=True,
        criterion=criterion,
        sd_ratio_upper=sd_ratio,
        bioequivalent=passed,
    )


def tost(
    test: ParameterSample,
    reference: ParameterSample,
    *,
    limits: tuple[float, float] = (0.8, 1.25),
    ci_level: float = 0.90,
    design: Design | str | None = None,
    scaling: Scaling | str = "none",
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

    A `REPLICATE` design is analysed with the fixed effects analysis of
    variance of `_replicate` (EMA Method A) and reports `cv_intra_r`,
    `cv_intra_t` and the `anova` table with the ratio; `n_test` and
    `n_reference` count the administrations there, not the subjects, since a
    subject carries several of each.

    `scaling` replaces the acceptance rule by one of the reference-scaled
    rules of the guidances:

    | value | rule |
    | --- | --- |
    | `"none"` | the 90 % interval within `limits`, the default |
    | `"ema"` | average bioequivalence with expanding limits (ABEL): for `cmax` alone, and only above a `cv_intra_r` of 30 %, the limits widen to \(e^{\pm 0.760 s_{wR}}\) (capped at 69.84-143.19 %) and the point estimate must lie within 80.00-125.00 % |
    | `"fda"` | reference-scaled average bioequivalence (RSABE): above \(s_{wR} = 0.294\) the upper 95 % bound of \((\mu_T-\mu_R)^2 - \theta s_{wR}^2\) must not be positive and the point estimate must lie within 80.00-125.00 %; below it the unscaled analysis decides |
    | `"fda_nti"` | the same criterion with \(\sigma_{w0} = 0.10\) and always scaled, plus the unscaled 90 % interval within 80.00-125.00 % and the upper 90 % bound of \(s_{wT}/s_{wR}\) at most 2.500 |
    | `"ema_nti"` | the limits tightened to 90.00-111.11 % |

    Every rule but `"ema_nti"` needs the within-subject variability of the
    reference and therefore a replicate design. `limits` carries the limits
    the verdict was taken against, `limits_scaled` the derived ones and
    `criterion` the bound of the FDA rule, which has no limits at all.

    Args:
        test: the test sample.
        reference: the reference sample.
        limits: acceptance limits of the ratio.
        ci_level: level of the interval, 0.90 for the usual \(\alpha = 0.05\).
        design: the design, as the member or as its string, detected from
            the samples by default.
        scaling: the acceptance rule, see the table above.

    Returns:
        The result of the parameter.

    Raises:
        ValueError: for reversed limits, an unknown design or scaling, a
            design the samples do not support, or a scaling the design
            cannot carry.
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
    cv_intra_r = cv_intra_t = nan
    fit: _ReplicateFit | None = None
    anova: pd.DataFrame | None = None
    if resolved is Design.REPLICATE:
        fit = _replicate(test, reference)
        log_ratio, se, df = fit.log_ratio, fit.se, fit.df
        p_period, p_sequence, anova = fit.p_period, fit.p_sequence, fit.anova
        cv_intra = float(np.sqrt(np.expm1(fit.sigma_e2)))
        cv_intra_r = (
            float(np.sqrt(np.expm1(fit.s2_wr))) if np.isfinite(fit.s2_wr) else nan
        )
        cv_intra_t = (
            float(np.sqrt(np.expm1(fit.s2_wt))) if np.isfinite(fit.s2_wt) else nan
        )
        n_test, n_reference = fit.n_test, fit.n_reference
    elif resolved is Design.CROSSOVER:
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
    parameter = BEParameter(
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
        cv_intra_r=cv_intra_r,
        cv_intra_t=cv_intra_t,
        anova=anova,
    )
    return _apply_scaling(parameter, scaling, fit)


def carryover_table(
    batch: Timecourses,
    result: ParameterResult,
    *,
    threshold: float = 0.05,
) -> pd.DataFrame:
    r"""The pre-dose concentration of every subject against its own maximum.

    ICH M13A (2024, 2.2.3.3), the FDA ANDA bioequivalence guidance and the EMA
    bioequivalence guideline all draw the same line: a period whose pre-dose
    concentration is more than 5 % of that subject's \(C_\mathrm{max}\) of the
    same period carries drug from the previous period, and the subject "should
    be dropped from the study evaluation of that period"; M13A adds that a
    statistical test for carryover "is not considered relevant", so this
    comparison replaces it.

    The pre-dose value of a sample is the last value **strictly before** its
    dose time. A sample recorded at the dose time counts as a pre-dose sample
    for an extravascular route only, where it is drawn before the dose is
    swallowed; after an intravenous bolus or during an infusion the value at
    the dose time is the post-dose value of this period and says nothing about
    the previous one (reading it would flag every subject with a fraction of
    1). A sample whose schedule carries no value before the dose has no
    pre-dose value: `predose` is `NaN` and the sample is not flagged.

    Args:
        batch: the timecourses of the period, one sample per subject
        result: the analysis of that batch, for \(C_\mathrm{max}\)

    Keyword Args:
        threshold: the share of \(C_\mathrm{max}\) above which the sample is
            flagged, 0.05 of the three guidances

    Returns:
        One row per sample with the sample dimensions, `predose`, `cmax`,
        `fraction` and `flagged`.

    Raises:
        ValueError: if the result carries no `cmax`, or if the batch and the
            result do not have the same samples.
    """
    if "cmax" not in result.ds.data_vars:
        raise ValueError("the result carries no 'cmax'")
    dims = tuple(str(d) for d in batch.sample_dims)
    n_rows = batch.n_samples
    if result.ds["cmax"].size != n_rows:
        raise ValueError(
            f"the batch has {n_rows} samples and the result "
            f"{result.ds['cmax'].size}; they must be the same analysis"
        )
    t = batch.times.reshape(n_rows, batch.n_time)
    c = batch.values.reshape(n_rows, batch.n_time)
    dose_time = batch.first_dose_time
    time = (
        np.zeros(n_rows)
        if dose_time is None
        else np.asarray(dose_time, dtype=np.float64).reshape(n_rows)
    )
    route = batch.route
    # a sample at the dose time is a pre-dose sample of an extravascular period
    # only; after a bolus or during an infusion it carries the post-dose value
    at_dose = route is None or not route.is_iv
    with np.errstate(invalid="ignore"):
        measured = np.isfinite(c) & np.isfinite(t)
        candidate = measured & (
            (t <= time[:, None]) if at_dose else (t < time[:, None])
        )
    has_predose = candidate.any(axis=1)
    order = np.where(candidate, t, -np.inf).argmax(axis=1)
    predose = np.where(
        has_predose, np.take_along_axis(c, order[:, None], 1)[:, 0], np.nan
    )
    cmax = np.asarray(result.ds["cmax"].to_numpy(), dtype=np.float64).reshape(n_rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = predose / cmax
        flagged = np.isfinite(fraction) & (fraction > threshold)
    columns: dict[str, Any] = {}
    for name in dims:
        values = (
            batch.ds[name].to_numpy()
            if name in batch.ds.coords
            else np.arange(batch.ds.sizes[name])
        )
        grid = np.broadcast_to(
            values.reshape([-1 if other == name else 1 for other in dims]),
            batch.sample_shape,
        )
        columns[name] = np.asarray(grid).reshape(n_rows)
    columns["predose"] = predose
    columns["cmax"] = cmax
    columns["fraction"] = fraction
    columns["flagged"] = flagged
    return pd.DataFrame(columns)


def carryover_labels(
    batch: Timecourses,
    result: ParameterResult,
    *,
    dim: str,
    threshold: float = 0.05,
) -> list[str]:
    """The labels of the subjects `carryover_table` flags.

    Args:
        batch: the timecourses of the period
        result: the analysis of that batch

    Keyword Args:
        dim: the sample dimension whose labels name the subjects
        threshold: the share of Cmax above which a sample is flagged

    Returns:
        The labels, as strings, in the order of the samples.

    Raises:
        ValueError: if `dim` is no sample dimension of the batch, or as
            `carryover_table`.
    """
    table = carryover_table(batch, result, threshold=threshold)
    if dim not in table.columns:
        dims = [str(d) for d in batch.sample_dims]
        raise ValueError(f"'{dim}' is not a sample dimension {dims} of the batch")
    return [str(label) for label in table.loc[table["flagged"], dim]]


def bioequivalence(
    test: ParameterResult,
    reference: ParameterResult,
    parameters: Sequence[str] = ("auc_inf_obs", "cmax"),
    *,
    dim: str = "individual",
    limits: tuple[float, float] = (0.8, 1.25),
    ci_level: float = 0.90,
    design: Design | str | None = None,
    scaling: Scaling | str = "none",
    include_excluded: bool = False,
    carryover: Literal["ignore", "flag", "exclude"] = "ignore",
    carryover_threshold: float = 0.05,
    test_batch: Timecourses | None = None,
    reference_batch: Timecourses | None = None,
    **indexers: Any,
) -> BEResult:
    r"""Average bioequivalence of the parameters of two results.

    Every parameter is taken with `ParameterResult.sample(name, dim, **indexers)`
    from both results and tested with `tost`; the study is bioequivalent
    when every parameter is. A subject which a result marks `excluded`
    (`pkpdutils.nca.NCAResult.exclude`) is left out unless `include_excluded`
    asks for it.

    With the timecourses of the two periods (`test_batch`, `reference_batch`)
    the pre-dose concentrations are read as well (`carryover_table`):
    `carryover="flag"` names the subjects above the threshold in
    `BEParameter.carryover` and `carryover="exclude"` drops them from the
    analysis of every parameter, which is what ICH M13A (2024) and the FDA ANDA
    guidance ask for.

    Args:
        test: the result of the test formulation.
        reference: the result of the reference formulation.
        parameters: the parameters to test.
        dim: the sample dimension of the individuals.
        limits: acceptance limits of the ratio.
        ci_level: level of the intervals.
        design: the design, as the member or as its string, detected from
            the samples by default.
        scaling: the acceptance rule, `"none"` for the fixed limits and one
            of `"ema"`, `"fda"`, `"fda_nti"`, `"ema_nti"` for the
            reference-scaled and the narrow therapeutic index rules of the
            guidances, see `tost`. `"ema_nti"` tightens the limits of every
            parameter it is asked for, which is why the EMA rule for
            \(C_\mathrm{max}\) ("when it is of particular importance") is
            expressed by naming `cmax` in `parameters` or leaving it out.
        include_excluded: analyse the excluded subjects as well.
        carryover: what to do with a subject whose pre-dose concentration
            exceeds `carryover_threshold` of its own Cmax: `"ignore"` nothing,
            `"flag"` name it in `BEParameter.carryover`, `"exclude"` drop it
            from the analysis and name it there as well.
        carryover_threshold: the share of Cmax which counts as carryover.
        test_batch: the timecourses the test result was computed from, needed
            for the carryover check.
        reference_batch: the timecourses of the reference result.
        **indexers: coordinate label per remaining sample dimension.

    Returns:
        The result.

    Raises:
        ValueError: for an unknown design, for a carryover check without the
            batches, or as `tost`.
    """
    if design is not None:
        design = coerce(design, Design)
    if carryover not in ("ignore", "flag", "exclude"):
        raise ValueError(
            f"'carryover' must be 'ignore', 'flag' or 'exclude', got '{carryover}'"
        )
    flagged: tuple[str, ...] = ()
    if carryover != "ignore":
        if test_batch is None and reference_batch is None:
            raise ValueError(
                f"carryover='{carryover}' needs the timecourses of the periods, "
                "give 'test_batch' and 'reference_batch'"
            )
        labels: list[str] = []
        for batch, analysis in (
            (test_batch, test),
            (reference_batch, reference),
        ):
            if batch is not None:
                labels.extend(
                    carryover_labels(
                        batch, analysis, dim=dim, threshold=carryover_threshold
                    )
                )
        flagged = tuple(dict.fromkeys(labels))
        if flagged:
            logger.info(
                "carryover: %d subjects above %.1f %% of their Cmax (%s)",
                len(flagged),
                carryover_threshold * 100.0,
                carryover,
            )
    results: dict[str, BEParameter] = {}
    for name in parameters:
        sample_test = test.sample(
            name, dim, include_excluded=include_excluded, **indexers
        )
        sample_reference = reference.sample(
            name, dim, include_excluded=include_excluded, **indexers
        )
        if flagged and carryover == "exclude":
            sample_test = _without_labels(sample_test, flagged)
            sample_reference = _without_labels(sample_reference, flagged)
        parameter = tost(
            sample_test,
            sample_reference,
            limits=limits,
            ci_level=ci_level,
            design=design,
            scaling=scaling,
        )
        results[name] = replace(parameter, carryover=flagged) if flagged else parameter
    return BEResult(
        parameters=results,
        bioequivalent=all(p.bioequivalent for p in results.values()),
        limits=(float(limits[0]), float(limits[1])),
        ci_level=ci_level,
    )


def _without_labels(sample: ParameterSample, labels: Sequence[str]) -> ParameterSample:
    """The sample without the individuals of the given labels.

    Args:
        sample: the sample of individual values.
        labels: the labels to drop, as strings.

    Returns:
        The sample without them, unchanged when it carries no labels.
    """
    if sample.labels is None:
        return sample
    drop = set(labels)
    keep = np.array(
        [str(label) not in drop for label in sample.labels.tolist()], dtype=bool
    )
    if keep.all():
        return sample
    return sample.select(keep)
