r"""Replicate designs and the reference-scaled acceptance criteria.

The data is built from the model itself: the log value of an administration is
\(\mu + s_i + \pi_p + \tau\,[\text{test}] + e\), and the residual \(e\) is a
perturbation of \(\pm d_i/2\) on the two administrations of one formulation,
chosen so that it sums to zero within every sequence. It is then orthogonal to
the subject, period and formulation columns of the model, so the analysis of
variance recovers the formulation effect exactly and the within-subject
variance of a formulation is the closed form \(\sum_i d_i^2 / (2(n - s))\).
"""

import numpy as np
import pytest
from scipy.stats import chi2
from scipy.stats import f as f_distribution
from scipy.stats import t as student_t

from pkpdutils.stats import ParameterSample, abel_limits, rsabe_criterion, tost
from pkpdutils.stats.bioequivalence import (
    EMA_ABEL_LIMITS,
    EMA_NTI_LIMITS,
    Design,
)

#: the number of subjects of the built studies, half in every sequence
N = 12

#: the true formulation effect of the built studies
TAU = np.log(0.95)

#: the subject effects, balanced between the two sequences
SUBJECT_EFFECT = np.tile(np.linspace(-0.3, 0.3, N // 2), 2)

#: the period effects of a four period study
PERIOD_EFFECT = np.array([0.0, 0.05, -0.02, 0.03])


def perturbation(cv: float, n: int, sequences: int = 2) -> np.ndarray:
    r"""Within-subject perturbations whose closed form variance is `cv`.

    The perturbations alternate in sign inside every sequence, so they sum to
    zero there, and their size follows from
    \(s_w^2 = \sum_i d_i^2 / (2(n - s)) = \ln(1 + \mathrm{CV}^2)\).
    """
    variance = np.log1p(cv**2)
    size = np.sqrt(2.0 * (n - sequences) * variance / n)
    half = np.array([size if i % 2 == 0 else -size for i in range(n // sequences)])
    return np.tile(half, sequences)


def build(
    *,
    tau: float = TAU,
    cv_r: float = 0.40,
    cv_t: float = 0.25,
    sequences: tuple[str, str] = ("TRTR", "RTRT"),
    n: int = N,
) -> dict[str, ParameterSample]:
    """A replicate study built from the model, one sample per formulation."""
    order = [sequences[0]] * (n // 2) + [sequences[1]] * (n // 2)
    subjects = [f"s{i:02d}" for i in range(n)]
    shift_r, shift_t = perturbation(cv_r, n), perturbation(cv_t, n)
    rows: dict[str, list[tuple[float, str, int, str]]] = {"T": [], "R": []}
    for i, (sequence, subject) in enumerate(zip(order, subjects, strict=True)):
        seen = {"T": 0, "R": 0}
        for period, letter in enumerate(sequence, start=1):
            value = (
                np.log(100.0)
                + SUBJECT_EFFECT[i]
                + PERIOD_EFFECT[period - 1]
                + (tau if letter == "T" else 0.0)
            )
            shift = (shift_t if letter == "T" else shift_r)[i]
            value += shift / 2.0 if seen[letter] == 0 else -shift / 2.0
            seen[letter] += 1
            rows[letter].append((float(np.exp(value)), subject, period, sequence))
    samples = {}
    for letter in "TR":
        values, subject, period, sequence = zip(*rows[letter], strict=True)
        samples[letter] = ParameterSample(
            values=np.array(values),
            labels=np.array(subject),
            coords={
                "subject": np.array(subject),
                "period": np.array(period),
                "sequence": np.array(sequence),
            },
            name="cmax",
            unit="milligram / liter",
        )
    return samples


def test_the_replicate_design_is_detected_and_recovers_the_effect() -> None:
    """A TRTR/RTRT study: the design, the effect, the degrees of freedom."""
    samples = build()
    result = tost(samples["T"], samples["R"])
    assert result.design is Design.REPLICATE
    assert result.gmr == pytest.approx(0.95, abs=1e-12)
    # 4n observations minus (1 + (n - 1) subject + 3 period + 1 formulation)
    assert result.df == 3 * N - 4
    assert (result.n_test, result.n_reference) == (2 * N, 2 * N)


def test_the_within_subject_variability_of_each_formulation_is_recovered() -> None:
    """`cv_intra_r` and `cv_intra_t` come back as they were built."""
    result = tost(test=build()["T"], reference=build()["R"])
    assert result.cv_intra_r == pytest.approx(0.40, abs=1e-12)
    assert result.cv_intra_t == pytest.approx(0.25, abs=1e-12)
    assert result.cv_intra > 0


def test_the_three_period_design_is_a_replicate_design_as_well() -> None:
    """TRT/RTR replicates the test in one sequence and the reference in the other."""
    samples = build(sequences=("TRT", "RTR"), cv_r=0.35, cv_t=0.35)
    result = tost(samples["T"], samples["R"])
    assert result.design is Design.REPLICATE
    assert result.gmr == pytest.approx(0.95, abs=1e-12)
    # 3n observations minus (1 + (n - 1) subject + 2 period + 1 formulation)
    assert result.df == 2 * N - 3
    assert np.isfinite(result.cv_intra_r) and np.isfinite(result.cv_intra_t)


def test_the_anova_table_adds_up_and_names_its_sources() -> None:
    """The sequential sums of squares and the degrees of freedom of the model."""
    samples = build()
    anova = tost(samples["T"], samples["R"]).anova
    assert anova is not None
    assert list(anova["source"]) == [
        "sequence",
        "subject(sequence)",
        "period",
        "formulation",
        "residual",
    ]
    assert list(anova["df"]) == [1, N - 2, 3, 1, 3 * N - 4]
    total = float(anova["sum_sq"].sum())
    y = np.log(np.concatenate([samples["T"].values, samples["R"].values]))
    assert total == pytest.approx(float(((y - y.mean()) ** 2).sum()))


def test_a_sequence_which_contradicts_the_period_is_rejected() -> None:
    """A test administration in a period the sequence calls reference is an error."""
    samples = build()
    broken = ParameterSample(
        values=samples["T"].values,
        labels=samples["T"].labels,
        coords={**samples["T"].coords, "period": samples["T"].coords["period"] + 1},
        name="cmax",
        unit="milligram / liter",
    )
    with pytest.raises(ValueError, match="the sample says"):
        tost(broken, samples["R"])


def test_sequences_which_are_no_replicate_design_are_rejected() -> None:
    """Only the sequences of `REPLICATE_DESIGNS` are analysed this way."""
    samples = build()
    renamed = {
        letter: ParameterSample(
            values=samples[letter].values,
            labels=samples[letter].labels,
            coords={
                **samples[letter].coords,
                "sequence": np.where(
                    samples[letter].coords["sequence"] == "TRTR", "TTRR", "RRTT"
                ),
            },
            name="cmax",
            unit="milligram / liter",
        )
        for letter in "TR"
    }
    with pytest.raises(ValueError, match="no replicate design"):
        tost(renamed["T"], renamed["R"], design="replicate")


def test_a_replicate_sample_without_the_coordinates_is_rejected() -> None:
    """The analysis needs the period and the sequence of every administration."""
    samples = build()
    bare = ParameterSample(
        values=samples["T"].values, labels=samples["T"].labels, name="cmax"
    )
    with pytest.raises(ValueError, match="missing the coordinates"):
        tost(bare, samples["R"], design="replicate")


def test_ema_scaling_widens_the_limits_of_cmax_at_a_cv_of_40_percent() -> None:
    """The closed form of the expanding limits, `exp(+- 0.760 s_wR)`."""
    samples = build(cv_r=0.40)
    result = tost(samples["T"], samples["R"], scaling="ema")
    s_wr = np.sqrt(np.log1p(0.40**2))
    expected = (float(np.exp(-0.760 * s_wr)), float(np.exp(0.760 * s_wr)))
    assert expected == pytest.approx((0.7461770, 1.3401646), abs=1e-6)
    assert result.scaled is True
    assert result.limits == pytest.approx(expected)
    assert result.limits_scaled == pytest.approx(expected)


def test_the_widening_stops_at_a_cv_of_50_percent() -> None:
    """Above 50 % the limits stay at the cap of 69.84-143.19 %."""
    for cv in (0.50, 0.70, 1.20):
        assert abel_limits(cv) == pytest.approx(EMA_ABEL_LIMITS, abs=5e-5)
    assert abel_limits(0.299) == (0.8, 1.25)
    with pytest.raises(ValueError, match="non-negative"):
        abel_limits(float("nan"))


def test_ema_scaling_leaves_an_area_and_a_low_variability_alone() -> None:
    """The EMA widens `cmax` alone, and only above a `cv_intra_r` of 30 %."""
    samples = build(cv_r=0.40)
    area = ParameterSample(
        values=samples["T"].values,
        labels=samples["T"].labels,
        coords=samples["T"].coords,
        name="auc_inf_obs",
        unit="hour * milligram / liter",
    )
    result = tost(area, samples["R"], scaling="ema")
    assert result.scaled is False
    assert result.limits == (0.8, 1.25)
    low = build(cv_r=0.20)
    assert tost(low["T"], low["R"], scaling="ema").scaled is False


def test_ema_scaling_still_asks_for_the_point_estimate_within_80_to_125() -> None:
    """A widened study whose ratio leaves 80-125 % is not bioequivalent."""
    samples = build(tau=np.log(0.78), cv_r=0.45)
    result = tost(samples["T"], samples["R"], scaling="ema")
    assert result.gmr == pytest.approx(0.78, abs=1e-12)
    assert result.limits[0] < 0.78
    assert result.bioequivalent is False


def test_the_fda_criterion_matches_the_hand_computed_howe_bound() -> None:
    """One case of RSABE computed by hand from the pieces of the analysis."""
    samples = build(cv_r=0.45, cv_t=0.30)
    result = tost(samples["T"], samples["R"], scaling="fda")
    assert result.scaled is True
    assert result.criterion is not None
    # the pieces: the subject differences of the two sequences, the within
    # subject variance of the reference, and Howe's bound of the criterion
    s2_wr = float(np.log1p(0.45**2))
    theta = (np.log(1.25) / 0.25) ** 2
    difference, df_d = TAU, float(N - 2)
    # every subject of a sequence carries the same difference, so the pooled
    # within-sequence variance is zero and the standard error with it
    scaled = theta * s2_wr
    expected = (difference**2 - scaled) + np.sqrt(
        ((abs(difference) + student_t.ppf(0.95, df_d) * 0.0) ** 2 - difference**2) ** 2
        + (scaled - scaled * (N - 2) / chi2.ppf(0.95, N - 2)) ** 2
    )
    assert result.criterion == pytest.approx(float(expected), rel=1e-9)
    assert result.bioequivalent is bool(expected <= 0 and 0.8 <= result.gmr <= 1.25)


def test_the_howe_bound_is_the_formula_of_the_progesterone_guidance() -> None:
    """`rsabe_criterion` on numbers given by hand, both terms of the root."""
    difference, se, df_d = 0.05, 0.03, 22.0
    s2_wr, df_wr = 0.12, 23.0
    theta = (np.log(1.25) / 0.25) ** 2
    scaled = theta * s2_wr
    expected = (difference**2 - scaled) + np.sqrt(
        ((difference + student_t.ppf(0.95, df_d) * se) ** 2 - difference**2) ** 2
        + (scaled - scaled * df_wr / chi2.ppf(0.95, df_wr)) ** 2
    )
    assert rsabe_criterion(difference, se, df_d, s2_wr, df_wr) == pytest.approx(
        float(expected), rel=1e-12
    )
    # a reference which is not replicated carries no variance to scale with
    assert np.isnan(rsabe_criterion(difference, se, df_d, float("nan"), 0.0))


def test_the_fda_rule_does_not_scale_a_drug_which_is_not_highly_variable() -> None:
    """Below `s_wR = 0.294` the unscaled analysis decides."""
    samples = build(cv_r=0.20, cv_t=0.20)
    result = tost(samples["T"], samples["R"], scaling="fda")
    assert result.scaled is False
    assert result.criterion is None
    assert result.limits == (0.8, 1.25)


def test_the_narrow_therapeutic_index_rule_of_the_fda_adds_two_conditions() -> None:
    """The scaled criterion, the unscaled interval and the variability comparison."""
    samples = build(cv_r=0.12, cv_t=0.12)
    result = tost(samples["T"], samples["R"], scaling="fda_nti")
    assert result.scaled is True
    assert result.criterion is not None
    # equal within-subject variances, so the bound of the ratio is the square
    # root of the F quantile alone, with `n - s` degrees of freedom on both sides
    assert result.sd_ratio_upper == pytest.approx(
        float(np.sqrt(f_distribution.ppf(0.95, N - 2, N - 2)))
    )
    assert 1.0 < result.sd_ratio_upper < 2.5
    variable = build(cv_r=0.10, cv_t=0.45)
    loud = tost(variable["T"], variable["R"], scaling="fda_nti")
    assert loud.sd_ratio_upper > 2.5
    assert loud.bioequivalent is False


def test_the_narrow_therapeutic_index_limits_of_the_ema_are_tightened() -> None:
    """`ema_nti` judges against 90.00-111.11 % and needs no replicate design."""
    samples = build(cv_r=0.10, cv_t=0.10)
    result = tost(samples["T"], samples["R"], scaling="ema_nti")
    assert result.limits == EMA_NTI_LIMITS
    assert result.limits_scaled == EMA_NTI_LIMITS
    assert result.scaled is True
    assert result.bioequivalent is (
        EMA_NTI_LIMITS[0] <= result.ci_low and result.ci_high <= EMA_NTI_LIMITS[1]
    )


def test_a_scaling_which_needs_replicates_refuses_a_2x2_study() -> None:
    """Without a replicate design there is no variability of the reference."""
    values = np.array([1.0, 1.1, 0.9, 1.2, 1.05, 0.95])
    labels = np.array([f"s{i}" for i in range(6)])
    test = ParameterSample(values=values, labels=labels, name="cmax")
    reference = ParameterSample(values=values * 1.05, labels=labels, name="cmax")
    with pytest.raises(ValueError, match="only a replicate design"):
        tost(test, reference, scaling="ema")
    with pytest.raises(ValueError, match="is not a scaling"):
        tost(test, reference, scaling="who")
