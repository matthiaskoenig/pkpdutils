"""Acceptance criteria of the terminal phase and the exclusion they drive."""

import numpy as np
import pytest

from pkpdutils import (
    Acceptance,
    AUCMethod,
    Dose,
    NCAFlag,
    NCAOptions,
    Route,
    Timecourse,
    Timecourses,
    nca,
    nca_single,
)

#: an intravenous bolus of a mono-exponential curve, the times of a full profile
FULL_TIME = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])

#: the same curve sampled over one half-life only
SHORT_TIME = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0])

DOSE = Dose(amount=100.0, unit="mg", route=Route.IV_BOLUS)


def exponential(time: np.ndarray, *, k: float = 0.5, c0: float = 10.0) -> Timecourse:
    return Timecourse(
        time=time,
        value=c0 * np.exp(-k * time),
        time_unit="hr",
        unit="mg/l",
        dose=DOSE,
        substance="drug",
    )


def noisy() -> Timecourse:
    """A curve whose terminal points scatter, so that the regression is poor."""
    factors = np.array([1.0, 1.0, 1.0, 1.7, 0.5, 1.6, 0.55, 1.5, 0.6])
    return Timecourse(
        time=FULL_TIME,
        value=10.0 * np.exp(-0.5 * FULL_TIME) * factors,
        time_unit="hr",
        unit="mg/l",
        dose=DOSE,
        substance="drug",
    )


def test_the_default_analysis_accepts_every_sample() -> None:
    result = nca_single(exponential(FULL_TIME))
    assert bool(result["accepted"]) is True
    assert bool(result["excluded"]) is False
    assert result.flags() == []
    assert "accepted" not in result.parameters
    assert "excluded" not in result.parameters


def test_the_pkanalix_preset_carries_the_four_documented_thresholds() -> None:
    preset = Acceptance.pkanalix()
    assert (
        preset.r2_adj_min,
        preset.extrapolation_max,
        preset.span_min,
        preset.n_points_min,
    ) == (0.98, 0.20, 3.0, 3)
    assert preset.exclude is False
    assert Acceptance().any_threshold is False
    assert preset.any_threshold is True


def test_a_poor_regression_fails_the_r2_threshold() -> None:
    plain = nca_single(noisy())
    r2_adj = float(plain["lambda_z_r2_adj"])
    assert r2_adj < 0.98
    options = NCAOptions(acceptance=Acceptance(r2_adj_min=0.98))
    result = nca_single(noisy(), options=options)
    assert bool(result["accepted"]) is False
    assert "NOT_ACCEPTED" in result.flags()
    assert int(result["flags"]) & NCAFlag.NOT_ACCEPTED
    # the clean curve of the same shape meets it
    assert bool(nca_single(exponential(FULL_TIME), options=options)["accepted"]) is True


def test_a_short_profile_fails_the_extrapolation_and_the_span_threshold() -> None:
    curve = exponential(SHORT_TIME)
    plain = nca_single(curve, options=NCAOptions(auc_method=AUCMethod.LOG))
    # closed form of a mono-exponential: the predicted extrapolation is
    # exp(-k tlast) of the whole area, and the window covers t/thalf half-lives
    extrapolated = float(
        (plain["auc_inf_pred"] - plain["auc_last"]) / plain["auc_inf_pred"]
    )
    assert extrapolated == pytest.approx(np.exp(-0.5 * 2.0), rel=1e-3)
    assert extrapolated > 0.20
    span = float(plain["lambda_z_span"])
    assert span == pytest.approx((2.0 - 0.25) * 0.5 / np.log(2.0), rel=1e-6)
    assert span < 3.0

    for acceptance in (
        Acceptance(extrapolation_max=0.20),
        Acceptance(span_min=3.0),
    ):
        options = NCAOptions(auc_method=AUCMethod.LOG, acceptance=acceptance)
        assert bool(nca_single(curve, options=options)["accepted"]) is False
        assert (
            bool(nca_single(exponential(FULL_TIME), options=options)["accepted"])
            is True
        )


def test_the_point_count_threshold_reads_the_regression() -> None:
    curve = exponential(FULL_TIME)
    n_points = int(nca_single(curve)["lambda_z_n_points"])
    below = NCAOptions(acceptance=Acceptance(n_points_min=n_points + 1))
    at = NCAOptions(acceptance=Acceptance(n_points_min=n_points))
    assert bool(nca_single(curve, options=below)["accepted"]) is False
    assert bool(nca_single(curve, options=at)["accepted"]) is True


def test_a_sample_without_a_terminal_phase_fails_every_threshold() -> None:
    flat = Timecourse(
        time=np.array([0.0, 1.0, 2.0, 3.0]),
        value=np.array([1.0, 1.0, 1.0, 1.0]),
        time_unit="hr",
        unit="mg/l",
        dose=DOSE,
        substance="drug",
    )
    result = nca_single(flat, options=NCAOptions(acceptance=Acceptance(r2_adj_min=0.5)))
    assert np.isnan(float(result["lambda_z"]))
    assert bool(result["accepted"]) is False


def test_exclude_marks_the_samples_which_are_not_accepted() -> None:
    batch = Timecourses.from_timecourses(
        [exponential(FULL_TIME), noisy()], dim="individual", labels=["clean", "noisy"]
    )
    options = NCAOptions(acceptance=Acceptance(r2_adj_min=0.98, exclude=True))
    result = nca(batch, options=options)
    assert result["accepted"].to_numpy().tolist() == [True, False]
    assert result["excluded"].to_numpy().tolist() == [False, True]
    assert result["excluded_reason"].to_numpy().tolist() == ["", "acceptance criteria"]
    # without `exclude` the sample is flagged but stays in every statistic
    flagged = nca(batch, options=NCAOptions(acceptance=Acceptance(r2_adj_min=0.98)))
    assert flagged["excluded"].to_numpy().tolist() == [False, False]
    assert "excluded_reason" not in flagged.ds.data_vars
    assert float(flagged.summarize("individual")["cmax_n"]) == 2.0
    assert float(result.summarize("individual")["cmax_n"]) == 1.0
