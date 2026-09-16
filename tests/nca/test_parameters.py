"""The parameters added by the feature round: tlag, auc_all, clast_pred, C0."""

import numpy as np
import pytest

from pkpdutils import (
    AUCMethod,
    C0Method,
    Dose,
    Dosing,
    NCAOptions,
    Route,
    TerminalMethod,
    TerminalPhase,
    Timecourse,
    Timecourses,
    nca,
    nca_single,
)
from pkpdutils.nca.options import C0_BACK_EXTRAPOLATION, C0_FIRST_VALUE, C0_NONE

LINEAR = NCAOptions(
    auc_method=AUCMethod.LINEAR,
    terminal=TerminalPhase(method=TerminalMethod.ALL_AFTER_TMAX),
)


def oral_curve(time: np.ndarray, value: np.ndarray) -> Timecourse:
    return Timecourse(
        time=time,
        value=value,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        substance="drug",
    )


def bolus_curve(t: np.ndarray, c0: float = 10.0, k: float = 0.5) -> Timecourse:
    return Timecourse(
        time=t,
        value=c0 * np.exp(-k * t),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
        substance="drug",
    )


def test_tlag_of_an_absorption_delay() -> None:
    # two samples at 0 before the first measurable value at 1 h
    t = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    c = np.array([0.0, 0.0, 1.0, 2.0, 1.5, 0.75, 0.375])
    q = nca_single(oral_curve(t, c), options=LINEAR).to_quantities()
    assert q["tlag"].magnitude == 0.5
    assert str(q["tlag"].units) == "hour"


def test_tlag_without_delay_and_intravenously() -> None:
    # the first sample after the dose is already measurable: the absorption
    # started at the dose and the lag is 0, as Phoenix WinNonlin reports it
    t = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    c = np.array([0.5, 2.0, 1.5, 1.0, 0.5, 0.25])
    assert float(nca_single(oral_curve(t, c), options=LINEAR)["tlag"]) == 0.0
    # an intravenous curve has no absorption and reports no lag time
    assert "tlag" not in nca_single(bolus_curve(t[1:]), options=LINEAR)


def test_tlag_without_a_measurable_value_is_nan() -> None:
    t = np.array([0.0, 0.5, 1.0, 2.0])
    c = np.array([0.0, 0.0, 0.0, 0.0])
    assert np.isnan(nca_single(oral_curve(t, c), options=LINEAR)["tlag"])


def test_tlag_ignores_pre_dose_samples() -> None:
    # the sample before the dose is not a lag of the absorption: the first
    # sample at the dose is measurable, so the lag is 0 and not -0.5
    t = np.array([-0.5, 0.0, 1.0, 2.0, 4.0, 8.0])
    c = np.array([0.0, 1.0, 2.0, 1.5, 0.75, 0.375])
    assert float(nca_single(oral_curve(t, c), options=LINEAR)["tlag"]) == 0.0


def test_auc_all_adds_the_trailing_triangle() -> None:
    t = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0])
    c = np.array([0.0, 2.0, 1.5, 1.0, 0.5, 0.0])
    q = nca_single(oral_curve(t, c), options=LINEAR).to_quantities()
    # `auc_last` ends at the last positive value, `auc_all` at the last
    # observation: the triangle from (8, 0.5) down to (12, 0)
    assert q["tlast"].magnitude == 8.0
    assert q["auc_all"].magnitude == pytest.approx(
        q["auc_last"].magnitude + 0.5 * 0.5 * 4.0
    )
    assert q["aumc_all"].magnitude == pytest.approx(
        q["aumc_last"].magnitude + 0.5 * (8.0 * 0.5 + 12.0 * 0.0) * 4.0
    )
    assert str(q["auc_all"].units) == "hour * milligram / liter"


def test_auc_all_is_auc_last_with_a_positive_tail() -> None:
    t = np.array([0.0, 1.0, 2.0, 4.0, 8.0])
    c = np.array([0.0, 2.0, 1.5, 1.0, 0.5])
    q = nca_single(oral_curve(t, c), options=LINEAR).to_quantities()
    assert q["auc_all"].magnitude == pytest.approx(q["auc_last"].magnitude)
    assert q["aumc_all"].magnitude == pytest.approx(q["aumc_last"].magnitude)


def test_clast_pred_is_the_regression_at_tlast() -> None:
    t = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12])
    result = nca_single(bolus_curve(t), options=LINEAR)
    q = result.to_quantities()
    predicted = np.exp(
        q["lambda_z_intercept"].magnitude
        - q["lambda_z"].magnitude * q["tlast"].magnitude
    )
    assert q["clast_pred"].magnitude == pytest.approx(predicted)
    # `auc_inf_pred` extrapolates with it
    assert q["auc_inf_pred"].magnitude == pytest.approx(
        q["auc_last"].magnitude + q["clast_pred"].magnitude / q["lambda_z"].magnitude
    )
    assert str(q["clast_pred"].units) == "milligram / liter"


def test_back_extrapolation_fraction() -> None:
    t = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12])
    q = nca_single(bolus_curve(t), options=LINEAR).to_quantities()
    c0 = q["c0"].magnitude
    c1 = 10.0 * np.exp(-0.5 * 0.25)
    # the linear trapezoid of the inserted segment from (0, c0) to (0.25, c1)
    segment = 0.5 * (c0 + c1) * 0.25
    assert q["auc_back_extrap_fraction"].magnitude == pytest.approx(
        segment / q["auc_inf_obs"].magnitude
    )
    moment = 0.5 * (0.0 * c0 + 0.25 * c1) * 0.25
    assert q["aumc_back_extrap_fraction"].magnitude == pytest.approx(
        moment / q["aumc_inf"].magnitude
    )
    assert str(q["auc_back_extrap_fraction"].units) == "dimensionless"


def test_back_extrapolation_fraction_is_zero_at_the_dose() -> None:
    t = np.array([0.0, 0.5, 1, 2, 4, 6, 8, 12])
    q = nca_single(bolus_curve(t), options=LINEAR).to_quantities()
    assert q["auc_back_extrap_fraction"].magnitude == 0.0
    assert q["aumc_back_extrap_fraction"].magnitude == 0.0


def test_back_extrapolation_fraction_only_for_a_bolus() -> None:
    t = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    c = np.array([2.0, 1.5, 1.0, 0.5, 0.25])
    assert "auc_back_extrap_fraction" not in nca_single(
        oral_curve(t, c), options=LINEAR
    )


def test_c0_method_records_the_rule() -> None:
    t = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12])
    declining = nca_single(bolus_curve(t), options=LINEAR)
    assert int(declining["c0_method"]) == C0_BACK_EXTRAPOLATION
    assert float(declining["c0"]) == pytest.approx(10.0)

    # the first two values do not decline: the first value is used
    rising = Timecourse(
        time=t,
        value=np.array([1.0, 2.0, 1.5, 1.0, 0.6, 0.4, 0.25, 0.1]),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
    )
    result = nca_single(rising, options=LINEAR)
    assert int(result["c0_method"]) == C0_FIRST_VALUE
    assert float(result["c0"]) == pytest.approx(1.0)

    first = nca_single(
        bolus_curve(t),
        options=LINEAR.model_copy(update={"c0_method": C0Method.FIRST_VALUE}),
    )
    assert int(first["c0_method"]) == C0_FIRST_VALUE


def test_c0_method_none() -> None:
    t = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12])
    options = LINEAR.model_copy(update={"c0_method": C0Method.NONE})
    none = nca_single(bolus_curve(t), options=options).to_quantities()
    back = nca_single(bolus_curve(t), options=LINEAR).to_quantities()
    assert np.isnan(none["c0"].magnitude)
    assert int(none["c0_method"].magnitude) == C0_NONE
    # nothing is inserted at the dose: the area starts at the first sample
    c1 = 10.0 * np.exp(-0.5 * 0.25)
    assert none["auc_last"].magnitude == pytest.approx(
        back["auc_last"].magnitude - 0.5 * (10.0 + c1) * 0.25
    )
    assert none["auc_back_extrap_fraction"].magnitude == 0.0


def test_new_parameters_of_a_batch_carry_units() -> None:
    t = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    values = np.array(
        [
            [0.0, 0.0, 1.0, 2.0, 1.5, 0.75, 0.0],
            [0.0, 0.5, 1.5, 2.5, 1.8, 0.9, 0.0],
        ]
    )
    batch = Timecourses.from_arrays(
        t,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["s1", "s2"]},
        dose={"amount": np.array([100.0, 100.0]), "unit": "mg"},
        route=Route.ORAL,
    )
    result = nca(batch, options=LINEAR)
    assert result["tlag"].attrs["units"] == "hour"
    assert result["auc_all"].attrs["units"] == "hour * milligram / liter"
    assert result["tlag"].to_numpy().tolist() == [0.5, 0.0]
    assert "tlag" in result.to_dataframe().columns


def infusion_curve(t: np.ndarray, duration: float = 0.25) -> Timecourse:
    return Timecourse(
        time=t,
        value=10.0 * np.exp(-0.5 * t),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_INFUSION, duration=duration),
        substance="drug",
    )


def test_an_infusion_starts_at_zero_at_the_dose() -> None:
    # the curve starts at 0.25 h: the area gains the triangle from the dose to
    # the first sample, as Phoenix WinNonlin inserts it for an infusion
    t = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    late = nca_single(infusion_curve(t), options=LINEAR).to_quantities()
    early = nca_single(
        infusion_curve(np.concatenate([[0.0], t])), options=LINEAR
    ).to_quantities()
    c1 = 10.0 * np.exp(-0.5 * 0.25)
    # the triangle from (0, 0) to (0.25, c1) and its first moment
    triangle = 0.5 * 0.25 * c1
    moment = 0.25 * (0.0 * 0.0 + 0.25 * c1) / 2.0
    without = 0.5 * 0.25 * (10.0 + c1)
    assert late["auc_last"].magnitude == pytest.approx(
        early["auc_last"].magnitude - without + triangle
    )
    assert late["aumc_last"].magnitude == pytest.approx(
        early["aumc_last"].magnitude - 0.25 * (0.25 * c1) / 2.0 + moment
    )
    assert late["auc_all"].magnitude == pytest.approx(late["auc_last"].magnitude)


def test_an_infusion_sampled_at_the_dose_is_unchanged() -> None:
    t = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    q = nca_single(infusion_curve(t), options=LINEAR).to_quantities()
    # nothing is inserted where a sample was taken at the dose
    times = np.concatenate([[0.0], t[1:]])
    values = 10.0 * np.exp(-0.5 * times)
    area = float(np.sum(np.diff(times) * (values[:-1] + values[1:]) / 2.0))
    assert q["auc_last"].magnitude == pytest.approx(area)


def test_a_steady_state_infusion_interval_does_not_start_at_zero() -> None:
    # the interval of a multiple dose infusion starts at its trough, not at 0
    doses = Dosing.regimen(
        Dose(amount=100, unit="mg", route=Route.IV_INFUSION, duration=0.25),
        interval=12.0,
        n_doses=2,
    )
    t = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 12.25, 13.0, 16.0, 20.0, 24.0])
    curve = Timecourse(
        time=t,
        value=10.0 * np.exp(-0.2 * (t % 12.0)) + 1.0,
        time_unit="hr",
        unit="mg/l",
        dosing=doses,
        substance="drug",
    )
    q = nca_single(curve, options=LINEAR).to_quantities()
    # the area after the last dose starts at the first sample of that interval
    # and not at a zero inserted at 12 h
    assert q["auc_last"].magnitude == pytest.approx(
        float(
            np.sum(
                np.diff(t[t >= 12.0])
                * (curve.value[t >= 12.0][:-1] + curve.value[t >= 12.0][1:])
                / 2.0
            )
        )
    )
