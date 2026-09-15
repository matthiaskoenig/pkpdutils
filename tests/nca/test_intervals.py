import numpy as np
import pytest

from pkpdutils import (
    Dose,
    Dosing,
    NCAOptions,
    Route,
    Timecourse,
    Timecourses,
    nca,
    nca_single,
)
from pkpdutils.nca import AUCMethod, superposition
from pkpdutils.nca.options import Kind, NCAFlag

K, C0, TAU, N_DOSES = 0.2, 10.0, 12.0, 5


def multiple_dose_curve(times: np.ndarray) -> np.ndarray:
    """Superposition of a mono-exponential bolus curve given every TAU hours."""
    c = np.zeros_like(times)
    for k in range(N_DOSES):
        shifted = times - k * TAU
        c += np.where(shifted >= 0, C0 * np.exp(-K * shifted), 0.0)
    return c


def interval_auc_closed_form(k: int) -> float:
    """Area of interval k (0-based) of the superposed curve, the sum over the doses given so far."""
    return sum(
        C0 / K * (np.exp(-K * (k - j) * TAU) - np.exp(-K * (k + 1 - j) * TAU))
        for j in range(k + 1)
    )


TIMES = np.sort(
    np.concatenate([np.arange(0, N_DOSES * TAU + 0.01, 0.5), [N_DOSES * TAU + 24]])
)
PROTOCOL = Dosing.regimen(
    Dose(amount=100, unit="mg", route=Route.IV_BOLUS), interval=TAU, n_doses=N_DOSES
)
TC = Timecourse(
    time=TIMES,
    value=multiple_dose_curve(TIMES),
    time_unit="hr",
    unit="mg/l",
    dosing=PROTOCOL,
)


def test_interval_parameters_match_the_closed_form() -> None:
    result = nca_single(TC, NCAOptions(auc_method=AUCMethod.LOG))
    assert result.has_intervals
    auc = result["interval_auc"].to_numpy()
    assert auc.shape == (N_DOSES,)
    for k in range(N_DOSES):
        assert auc[k] == pytest.approx(interval_auc_closed_form(k), rel=1e-3)
    assert result["interval_start"].to_numpy().tolist() == [0, 12, 24, 36, 48]
    assert result["interval_end"].to_numpy().tolist() == [12, 24, 36, 48, 60]
    assert result["interval_dose"].to_numpy().tolist() == [100.0] * 5
    cmax = result["interval_cmax"].to_numpy()
    assert cmax[0] == pytest.approx(C0) and cmax[4] == pytest.approx(
        multiple_dose_curve(np.array([48.0]))[0]
    )
    assert result["interval_tmax"].to_numpy().tolist() == [0.0] * 5
    ctrough = result["interval_ctrough"].to_numpy()
    assert ctrough[0] == pytest.approx(C0 * np.exp(-K * TAU))
    assert np.all(np.diff(ctrough) > 0)  # accumulation
    assert result["interval_cavg"].to_numpy()[2] == pytest.approx(auc[2] / TAU)
    # the value at the start of an interval is the post-dose value after a
    # bolus; the pre-dose value of interval k is the trough of interval k - 1
    c_start = result["interval_c_start"].to_numpy()
    assert c_start[0] == pytest.approx(C0, rel=1e-6)
    assert np.all(c_start[1:] > ctrough[:-1])
    df = result.intervals()
    assert list(df["interval"]) == [1, 2, 3, 4, 5] and "interval_auc" in df.columns


def test_steady_state_is_the_last_interval_and_point_parameters_follow_the_last_dose() -> (
    None
):
    result = nca_single(TC, NCAOptions(auc_method=AUCMethod.LOG))
    q = result.to_quantities()
    assert float(q["auc_tau"].magnitude) == pytest.approx(
        interval_auc_closed_form(N_DOSES - 1), rel=1e-3
    )
    assert float(q["n_doses"].magnitude) == N_DOSES and float(q["tau"].magnitude) == TAU
    assert float(q["ctrough"].magnitude) == pytest.approx(
        result["interval_ctrough"].to_numpy()[-1]
    )
    assert float(q["accumulation_ratio_obs"].magnitude) == pytest.approx(
        interval_auc_closed_form(4) / interval_auc_closed_form(0), rel=1e-3
    )
    assert float(q["accumulation_ratio"].magnitude) == pytest.approx(
        1 / (1 - np.exp(-K * TAU)), rel=1e-2
    )
    # the point parameters are computed from the last dose on
    assert float(q["tmax"].magnitude) == 0.0
    assert float(q["cmax"].magnitude) == pytest.approx(
        multiple_dose_curve(np.array([48.0]))[0]
    )
    assert float(q["lambda_z"].magnitude) == pytest.approx(K, rel=1e-3)
    assert float(q["auc_last"].magnitude) == pytest.approx(
        result["interval_auc"].to_numpy()[-1]
        + (
            multiple_dose_curve(np.array([60.0]))[0]
            - multiple_dose_curve(np.array([84.0]))[0]
        )
        / K,
        rel=1e-2,
    )
    assert "INCOMPLETE_INTERVAL" not in result.flags()
    # the samples at the dose times are post-dose values of a bolus, so the
    # troughs of the intervals are extrapolated
    assert NCAFlag.EXTRAPOLATED_TROUGH.name in result.flags()


def test_multiple_dose_analysis_reports_no_single_dose_quantities() -> None:
    result = nca_single(TC, NCAOptions(auc_method=AUCMethod.LOG))
    q = result.to_quantities()
    # the slice after the last dose carries the exposure of the earlier doses,
    # so `CL = D / AUC(0-inf)` of that slice would be biased low (1.82 l/h here)
    for name in ("cl", "vz", "vss", "auc_inf_dn", "cmax_dn"):
        assert np.isnan(float(q[name].magnitude)), name
    # the clearance of a multiple dose analysis is the dose over the exposure
    # of the dosing interval: 100 mg / 50 mg h/l = 2 l/h
    assert float(q["cl_ss"].magnitude) == pytest.approx(2.0, abs=1e-3)
    assert str(q["cl_ss"].units) == "liter / hour"
    assert "cl_ss_f" not in result
    # the extrapolated areas stay: they describe the decline after the last dose
    assert np.isfinite(float(q["auc_inf_obs"].magnitude))
    assert np.isfinite(float(q["mrt"].magnitude))


def test_an_extravascular_multiple_dose_analysis_reports_cl_ss_f() -> None:
    t = np.arange(0, 36.5, 0.5)
    c = np.zeros_like(t)
    for k in range(3):
        shifted = t - k * 12.0
        c += np.where(
            shifted > 0, 10 * (np.exp(-0.2 * shifted) - np.exp(-shifted)), 0.0
        )
    tc = Timecourse(
        time=t,
        value=c,
        time_unit="hr",
        unit="mg/l",
        dosing=Dosing.regimen(
            Dose(amount=100, unit="mg", route=Route.ORAL), interval=12, n_doses=3
        ),
    )
    result = nca_single(tc, NCAOptions(auc_method=AUCMethod.LOG))
    assert "cl_ss" not in result and "cl_ss_f" in result
    q = result.to_quantities()
    assert float(q["cl_ss_f"].magnitude) == pytest.approx(
        100.0 / float(result["interval_auc"].to_numpy()[-1]), rel=1e-9
    )
    assert str(q["cl_ss_f"].units) == "liter / hour"
    assert np.isnan(float(q["cl_f"].magnitude)) and np.isnan(float(q["vz_f"].magnitude))


def test_bolus_trough_is_regressed_only_when_the_sample_is_post_dose() -> None:
    # the sample at 11.5 is 2 % low: the regression over the last three samples
    # of the interval damps the noise to 0.98 ** (4/3) = 0.9734 of the true
    # trough, where a two point extrapolation would carry 0.98 ** 2 = 0.9604
    noisy = multiple_dose_curve(TIMES)
    noisy[np.isclose(TIMES, 11.5)] *= 0.98
    tc = Timecourse(
        time=TIMES, value=noisy, time_unit="hr", unit="mg/l", dosing=PROTOCOL
    )
    result = nca_single(tc, NCAOptions(auc_method=AUCMethod.LOG))
    trough = float(result["interval_ctrough"].to_numpy()[0])
    expected = C0 * np.exp(-K * TAU)
    assert trough == pytest.approx(expected * 0.98 ** (4 / 3), rel=1e-6)
    assert trough == pytest.approx(expected, rel=3e-2)
    assert abs(trough - expected) < abs(expected * 0.98**2 - expected)
    assert NCAFlag.EXTRAPOLATED_TROUGH.name in result.flags()


def test_an_observed_bolus_trough_at_the_dose_time_is_used_as_it_is() -> None:
    # the sample at every dose time is the pre-dose value, the trough of the
    # interval which ends there: it is not a post-dose sample and is used
    pre_dose = multiple_dose_curve(TIMES)
    for k in range(1, N_DOSES):
        pre_dose[np.isclose(TIMES, k * TAU)] -= C0
    tc = Timecourse(
        time=TIMES, value=pre_dose, time_unit="hr", unit="mg/l", dosing=PROTOCOL
    )
    result = nca_single(tc, NCAOptions(auc_method=AUCMethod.LOG))
    observed = pre_dose[np.isclose(TIMES, TAU)][0]
    assert result["interval_ctrough"].to_numpy()[0] == pytest.approx(observed)
    assert NCAFlag.EXTRAPOLATED_TROUGH.name not in result.flags()


def test_an_oral_curve_never_extrapolates_the_trough() -> None:
    t = np.arange(0, 36.5, 0.5)
    c = np.zeros_like(t)
    for k in range(3):
        shifted = t - k * 12.0
        c += np.where(
            shifted > 0, 10 * (np.exp(-0.2 * shifted) - np.exp(-shifted)), 0.0
        )
    tc = Timecourse(
        time=t,
        value=c,
        time_unit="hr",
        unit="mg/l",
        dosing=Dosing.regimen(
            Dose(amount=100, unit="mg", route=Route.ORAL), interval=12, n_doses=3
        ),
    )
    result = nca_single(tc, NCAOptions(auc_method=AUCMethod.LOG))
    assert NCAFlag.EXTRAPOLATED_TROUGH.name not in result.flags()
    assert result["interval_ctrough"].to_numpy()[0] == pytest.approx(
        c[np.isclose(t, 12.0)][0]
    )


def test_an_interval_without_a_sample_of_its_own_is_not_analysed() -> None:
    protocol = Dosing(
        amounts=np.array([100.0, 100.0, 100.0]),
        times=np.array([1.0, 2.0, 12.0]),
        unit="mg",
        route=Route.IV_BOLUS,
    )
    tc = Timecourse(
        time=np.array([0.0, 6.0, 12.0, 18.0]),
        value=np.array([0.0, 5.0, 3.0, 1.0]),
        time_unit="hr",
        unit="mg/l",
        dosing=protocol,
    )
    result = nca_single(tc)
    assert np.isnan(result["interval_auc"].to_numpy()[0])
    assert np.isnan(result["interval_cmax"].to_numpy()[0])
    assert result["interval_n_points"].to_numpy()[0] == 0


def test_single_dose_analysis_is_unchanged() -> None:
    t = np.array([0.5, 1, 2, 4, 8, 12, 24])
    tc = Timecourse(
        time=t,
        value=C0 * np.exp(-K * t),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
    )
    result = nca_single(tc)
    assert not result.has_intervals and "auc_tau" not in result
    assert result.intervals().empty


def test_tau_option_makes_a_single_dose_curve_a_steady_state_interval() -> None:
    t = np.arange(0, 12.5, 0.5)
    tc = Timecourse(
        time=t,
        value=multiple_dose_curve(t + 48.0),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
    )
    result = nca_single(tc, NCAOptions(tau=TAU, auc_method=AUCMethod.LOG))
    assert result.has_intervals and result["interval_auc"].to_numpy().shape == (1,)
    assert float(result["auc_tau"].to_numpy()) == pytest.approx(
        interval_auc_closed_form(4), rel=1e-3
    )
    assert np.isnan(float(result["accumulation_ratio_obs"].to_numpy()))


def test_incomplete_last_interval_is_flagged() -> None:
    t = TIMES[TIMES <= 54]
    tc = Timecourse(
        time=t,
        value=multiple_dose_curve(t),
        time_unit="hr",
        unit="mg/l",
        dosing=PROTOCOL,
    )
    result = nca_single(tc)
    assert NCAFlag.INCOMPLETE_INTERVAL.name in result.flags()
    assert np.isnan(result["interval_auc"].to_numpy()[-1]) and np.isnan(
        float(result["auc_tau"].to_numpy())
    )
    assert result["interval_n_points"].to_numpy()[-1] > 0
    assert np.isfinite(result["interval_auc"].to_numpy()[:-1]).all()


def test_batch_with_different_numbers_of_doses_and_workers() -> None:
    three = Dosing.regimen(
        Dose(amount=100, unit="mg", route=Route.IV_BOLUS), interval=TAU, n_doses=3
    )
    t3 = TIMES[TIMES <= 36]
    c3 = np.zeros_like(t3)
    for k in range(3):
        c3 += np.where(t3 - k * TAU >= 0, C0 * np.exp(-K * (t3 - k * TAU)), 0.0)
    a = TC
    b = Timecourse(time=t3, value=c3, time_unit="hr", unit="mg/l", dosing=three)
    batch = Timecourses.from_timecourses([a, b], labels=["five", "three"])
    serial = nca(batch, NCAOptions(auc_method=AUCMethod.LOG))
    assert serial.ds.sizes["interval"] == 5
    auc = serial["interval_auc"]
    assert (
        np.isfinite(auc.sel(individual="three").to_numpy()[:3]).all()
        and np.isnan(auc.sel(individual="three").to_numpy()[3:]).all()
    )
    assert serial["n_doses"].to_numpy().tolist() == [5, 3]
    parallel = nca(
        batch, NCAOptions(auc_method=AUCMethod.LOG, n_workers=2, chunk_rows=1)
    )
    for variable in serial.ds.data_vars:
        name = str(variable)
        np.testing.assert_allclose(
            serial[name].to_numpy(), parallel[name].to_numpy(), equal_nan=True
        )


def test_effect_intervals() -> None:
    t = np.arange(0, 36.5, 0.5)
    effect = 5.0 + 3.0 * np.sin(
        np.pi * (t % 12) / 12
    )  # rises and falls in every 12 h interval
    tc = Timecourse(
        time=t,
        value=effect,
        time_unit="hr",
        unit="mmHg",
        dosing=Dosing.regimen(Dose(amount=1, unit="mg"), interval=12, n_doses=3),
    )
    result = nca_single(tc, NCAOptions(kind=Kind.EFFECT, effect_threshold=6.0))
    assert result["interval_emax"].to_numpy() == pytest.approx([8.0, 8.0, 8.0])
    assert result["interval_temax"].to_numpy() == pytest.approx([6.0, 6.0, 6.0])
    assert result["interval_emin"].to_numpy() == pytest.approx([5.0, 5.0, 5.0])
    auec = result["interval_auec"].to_numpy()
    assert auec == pytest.approx([5 * 12 + 3 * 24 / np.pi] * 3, rel=2e-3)
    # 5 + 3 sin(pi tau / 12) > 6 for tau in (12 arcsin(1/3) / pi, 12 - 12 arcsin(1/3) / pi),
    # so the time above the threshold is 12 - 24 arcsin(1/3) / pi = 9.4038 h
    time_above = 12 - 24 * np.arcsin(1 / 3) / np.pi
    assert result["interval_time_above"].to_numpy() == pytest.approx(
        [time_above] * 3, rel=2e-3
    )
    q = result.to_quantities()
    assert float(q["auec_tau"].magnitude) == pytest.approx(auec[-1])
    assert float(q["emax_ss"].magnitude) == pytest.approx(8.0) and float(
        q["eavg"].magnitude
    ) == pytest.approx(auec[-1] / 12)


def test_superposition_on_a_protocol_with_different_amounts() -> None:
    t = np.array([0.5, 1, 2, 4, 8, 12, 24, 36, 48])
    single = Timecourse(
        time=t,
        value=C0 * np.exp(-K * t),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
    )
    protocol = Dosing(
        amounts=[100, 200, 100], times=[0, 12, 24], unit="mg", route=Route.IV_BOLUS
    )
    predicted = superposition(single, protocol, NCAOptions(auc_method=AUCMethod.LOG))
    assert predicted.dosing == protocol
    expected = (
        C0 * np.exp(-K * 24.5) + 2 * C0 * np.exp(-K * 12.5) + C0 * np.exp(-K * 0.5)
    )
    at = predicted.value[np.isclose(predicted.time, 24.5)]
    assert at.size == 1 and at[0] == pytest.approx(expected, rel=1e-3)
