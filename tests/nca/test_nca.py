import numpy as np
import pytest

from pkpdutils import Dose, Route, Timecourse, Timecourses
from pkpdutils.nca import (
    AUCMethod,
    C0Method,
    Kind,
    NCAOptions,
    NCAResult,
    TerminalMethod,
    TerminalPhase,
    nca,
    nca_single,
)
from pkpdutils.nca.nca import run_rows

K, C0 = 0.5, 10.0
T_DENSE = np.linspace(0, 20, 401)
# frozen model, a module level singleton keeps it out of the argument defaults (B008)
IV_DOSE = Dose(amount=100, unit="mg", route=Route.IV_BOLUS)


def iv_timecourse(dose: Dose | None = IV_DOSE) -> Timecourse:
    t = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12])
    return Timecourse(
        time=t,
        value=C0 * np.exp(-K * t),
        time_unit="hr",
        unit="mg/l",
        dose=dose,
        substance="x",
    )


def oral_timecourse(ka: float = 2.0) -> Timecourse:
    t = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24])
    c = C0 * ka / (ka - K) * (np.exp(-K * t) - np.exp(-ka * t))
    return Timecourse(
        time=t,
        value=c,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        substance="x",
    )


def test_single_iv_bolus_analytic() -> None:
    result = nca_single(iv_timecourse(), NCAOptions(auc_method=AUCMethod.LOG))
    assert isinstance(result, NCAResult)
    assert result.sample_dims == ()
    q = result.to_quantities()
    assert q["lambda_z"].magnitude == pytest.approx(K)
    assert str(q["lambda_z"].units) == "1 / hour"
    assert q["thalf"].magnitude == pytest.approx(np.log(2) / K)
    assert q["c0"].magnitude == pytest.approx(C0)
    # with the log rule and the back extrapolated C0, AUC(0-inf) is exact: c0/k
    assert q["auc_inf_obs"].magnitude == pytest.approx(C0 / K, rel=1e-6)
    assert q["auc_inf_pred"].magnitude == pytest.approx(C0 / K, rel=1e-6)
    assert q["aumc_inf"].magnitude == pytest.approx(C0 / K**2, rel=1e-6)
    assert q["mrt"].magnitude == pytest.approx(1 / K, rel=1e-6)
    assert q["cl"].magnitude == pytest.approx(100 / (C0 / K))
    assert str(q["cl"].units) == "liter / hour"
    assert q["vz"].magnitude == pytest.approx(100 / C0)
    assert str(q["vz"].units) == "liter"
    assert q["vss"].magnitude == pytest.approx(100 / C0, rel=1e-6)
    assert q["auc_inf_dn"].magnitude == pytest.approx(C0 / K / 100)
    assert "cl_f" not in result and "vz_f" not in result
    assert result.flags() == []


def test_c0_first_value() -> None:
    result = nca_single(iv_timecourse(), NCAOptions(c0_method=C0Method.FIRST_VALUE))
    assert result.to_quantities()["c0"].magnitude == pytest.approx(
        C0 * np.exp(-K * 0.25)
    )


def test_single_oral_names_and_flags() -> None:
    tc = oral_timecourse()
    result = nca_single(tc)
    q = result.to_quantities()
    assert (
        "cl_f" in result
        and "vz_f" in result
        and "vss" not in result
        and "c0" not in result
    )
    assert q["lambda_z"].magnitude == pytest.approx(K, rel=0.02)
    assert q["tmax"].magnitude == pytest.approx(tc.time[np.argmax(tc.value)])
    assert q["cmax"].magnitude == pytest.approx(tc.value.max())
    assert q["tlast"].magnitude == 24.0
    assert q["clast"].magnitude == pytest.approx(tc.value[-1])
    assert 0 < q["tmax_half"].magnitude < q["tmax"].magnitude
    assert q["auc_extrap_fraction"].magnitude < 0.2
    assert result.flags() == []
    assert q["lambda_z_n_points"].magnitude >= 3


def test_extrapolation_flag_and_positive_slope() -> None:
    t = np.array([1, 2, 3, 4, 5.0])
    rising = Timecourse(
        time=t, value=[1, 1.2, 1.5, 1.9, 2.5], time_unit="hr", unit="mg/l"
    )
    result = nca_single(rising, NCAOptions(terminal=TerminalPhase(exclude_cmax=False)))
    assert "POSITIVE_SLOPE" in result.flags()
    assert np.isnan(result.to_quantities()["lambda_z"].magnitude)
    assert np.isnan(result.to_quantities()["auc_inf_obs"].magnitude)
    truncated = Timecourse(
        time=[0.5, 1, 2, 3, 4], value=[2, 5, 4.5, 4, 3.6], time_unit="hr", unit="mg/l"
    )
    flags = nca_single(truncated).flags()
    assert "EXTRAPOLATION_HIGH" in flags


def test_too_few_points_and_no_max() -> None:
    short = Timecourse(time=[1, 2, 3], value=[1, 3, 2], time_unit="hr", unit="mg/l")
    result = nca_single(short)
    assert "TOO_FEW_POINTS" in result.flags()
    assert np.isnan(result.to_quantities()["thalf"].magnitude)
    assert not np.isnan(result.to_quantities()["auc_last"].magnitude)
    still_rising = Timecourse(
        time=[1, 2, 3, 4], value=[1, 2, 3, 4], time_unit="hr", unit="mg/l"
    )
    assert "NO_MAX" in nca_single(still_rising).flags()


def test_no_absorption_flag_only_for_oral() -> None:
    assert (
        "NO_ABSORPTION"
        in nca_single(
            iv_timecourse(dose=Dose(amount=100, unit="mg", route=Route.ORAL))
        ).flags()
    )
    assert "NO_ABSORPTION" not in nca_single(iv_timecourse()).flags()


def test_no_data_sample() -> None:
    tc = Timecourse(
        time=[0, 1, 2], value=[np.nan, np.nan, 1.0], time_unit="hr", unit="mg/l"
    )
    result = nca_single(tc)
    assert result.flags() == ["NO_DATA"]
    assert np.isnan(result.to_quantities()["cmax"].magnitude)


def test_lloq_handling() -> None:
    t = np.array([0.5, 1, 2, 4, 8, 12, 24])
    c = np.array([0.05, 2.0, 4.0, 3.0, 1.5, 0.5, 0.05])
    tc = Timecourse(time=t, value=c, time_unit="hr", unit="mg/l")
    nan = nca_single(tc, NCAOptions(lloq=0.1, auc_method=AUCMethod.LINEAR))
    assert "BLQ_TRUNCATED" in nan.flags()
    assert nan.to_quantities()["tlast"].magnitude == 12.0
    zero = nca_single(
        tc, NCAOptions(lloq=0.1, auc_method=AUCMethod.LINEAR, blq="zero_before_tmax")
    )
    # the first point becomes 0 and adds the triangle 0.5*(0+2)*0.5 to the area
    assert zero.to_quantities()["auc_last"].magnitude == pytest.approx(
        nan.to_quantities()["auc_last"].magnitude + 0.5 * 2.0 * 0.5
    )


def test_infusion_mrt_correction() -> None:
    # the grid starts at 0 so that both curves have the same areas: an IV bolus
    # curve starting after 0 gets the back extrapolated C0 inserted at t = 0,
    # which moves its MRT by itself (2.0 instead of the raw 1 + 1/K = 3.0)
    t = np.array([0, 1, 2, 4, 6, 8, 12.0])
    c = C0 * np.exp(-K * t)
    bolus = Timecourse(
        time=t,
        value=c,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
    )
    infusion = Timecourse(
        time=t,
        value=c,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_INFUSION, duration=1.0),
    )
    mrt_bolus = nca_single(bolus).to_quantities()["mrt"].magnitude
    mrt_inf = nca_single(infusion).to_quantities()["mrt"].magnitude
    assert mrt_inf == pytest.approx(mrt_bolus - 0.5)


def test_dose_per_bodyweight_units() -> None:
    tc = iv_timecourse(dose=Dose(amount=2, unit="mg/kg", route=Route.IV_BOLUS))
    q = nca_single(tc).to_quantities()
    assert str(q["cl"].units) == "liter / hour / kilogram"
    assert str(q["vz"].units) == "liter / kilogram"


def test_dose_time_shifts_time_axis() -> None:
    tc = iv_timecourse()
    shifted = tc.model_copy(
        update={
            "time": tc.time + 10.0,
            "dose": tc.dose.model_copy(update={"time": 10.0}) if tc.dose else None,
        }
    )
    a = nca_single(tc).to_quantities()
    b = nca_single(shifted).to_quantities()
    assert b["tmax"].magnitude == pytest.approx(a["tmax"].magnitude)
    assert b["auc_last"].magnitude == pytest.approx(a["auc_last"].magnitude)


def test_batch_equals_loop_over_singles() -> None:
    curves = [oral_timecourse(ka) for ka in (1.0, 2.0, 4.0)] + [
        Timecourse(
            time=[0.5, 1, 2, 4, 8],
            value=[1, 2, 1.5, np.nan, 0.5],
            time_unit="hr",
            unit="mg/l",
            dose=Dose(amount=100, unit="mg", route=Route.ORAL),
            substance="x",
        )
    ]
    batch = Timecourses.from_timecourses(curves, dim="individual")
    result = nca(batch)
    assert result.sample_dims == ("individual",)
    for i, tc in enumerate(curves):
        single = nca_single(tc)
        for name in result.parameters:
            a = float(result[name].values[i])
            b = single.to_quantities()[name].magnitude
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b), name
        assert (
            result.flags(individual=result.ds["individual"].values[i]) == single.flags()
        )


def test_batch_two_sample_dims_and_workers() -> None:
    time = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12])
    ks = np.array([0.3, 0.5, 0.8])
    doses = np.array([50.0, 100.0])
    values = (
        doses[:, None, None] / 10 * np.exp(-ks[None, :, None] * time[None, None, :])
    )
    batch = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("dose", "k"),
        coords={"dose": doses, "k": ks},
        dose={"amount": np.broadcast_to(doses[:, None], (2, 3)), "unit": "mg"},
        route=Route.IV_BOLUS,
    )
    serial = nca(batch, NCAOptions(auc_method=AUCMethod.LOG))
    assert serial.sample_dims == ("dose", "k")
    np.testing.assert_allclose(
        serial["lambda_z"].values, np.broadcast_to(ks, (2, 3)), rtol=1e-6
    )
    np.testing.assert_allclose(
        serial["cl"].values, np.broadcast_to(ks * 10, (2, 3)), rtol=1e-6
    )
    parallel = nca(batch, NCAOptions(auc_method=AUCMethod.LOG, n_workers=2))
    for name in serial.parameters:
        np.testing.assert_allclose(
            parallel[name].values, serial[name].values, equal_nan=True
        )
    np.testing.assert_array_equal(parallel["flags"].values, serial["flags"].values)


def test_cmax_half_only_for_oral() -> None:
    no_dose = Timecourse(
        time=[0.5, 1, 2, 4, 8], value=[1, 2, 1.5, 1.0, 0.5], time_unit="hr", unit="mg/l"
    )
    result = nca_single(no_dose)
    assert "cmax_half" not in result and "tmax_half" not in result
    oral = nca_single(oral_timecourse())
    assert "cmax_half" in oral and "tmax_half" in oral
    iv = nca_single(iv_timecourse())
    assert "cmax_half" not in iv and "tmax_half" not in iv


def test_chunking_matches_one_chunk() -> None:
    curves = [oral_timecourse(ka) for ka in (0.8, 1.0, 2.0, 4.0, 6.0)]
    batch = Timecourses.from_timecourses(curves, dim="individual")
    one = nca(batch, NCAOptions(chunk_rows=5000))
    chunked = nca(batch, NCAOptions(chunk_rows=2))
    parallel = nca(batch, NCAOptions(n_workers=2, chunk_rows=2))
    for other in (chunked, parallel):
        for name in one.parameters:
            np.testing.assert_allclose(
                other[name].values, one[name].values, equal_nan=True
            )
        np.testing.assert_array_equal(other["flags"].values, one["flags"].values)


def test_effect_kind() -> None:
    t = np.array([0, 1, 2, 4, 6, 8.0])
    e = np.array([10, 14, 20, 16, 12, 10.0])
    tc = Timecourse(time=t, value=e, time_unit="hr", unit="mmHg", substance="effect")
    result = nca_single(tc, NCAOptions(kind=Kind.EFFECT, effect_threshold=15.0))
    q = result.to_quantities()
    assert q["e0"].magnitude == 10.0
    assert q["emax_obs"].magnitude == 20.0 and q["temax"].magnitude == 2.0
    assert q["auec_last"].magnitude == pytest.approx(np.trapezoid(e, t))
    assert q["auec_baseline"].magnitude == pytest.approx(np.trapezoid(e - 10, t))
    assert q["emax_baseline"].magnitude == 10.0
    # above 15 from t=1+1/6 (linear between (1,14),(2,20): 14 + 6 dt = 15) to
    # t=4.5 (between (4,16),(6,12): 16 - 2 dt = 15)
    assert q["time_above"].magnitude == pytest.approx(4.5 - (1 + 1 / 6))
    assert "lambda_z" not in result and "cl" not in result
    assert str(q["auec_last"].units) == "hour * millimeter_Hg"


def test_terminal_manual_points() -> None:
    tc = oral_timecourse()
    options = NCAOptions(
        terminal=TerminalPhase(method=TerminalMethod.MANUAL, points=(8, 9, 10, 11))
    )
    q = nca_single(tc, options).to_quantities()
    slope, _ = np.polyfit(tc.time[8:], np.log(tc.value[8:]), 1)
    assert q["lambda_z"].magnitude == pytest.approx(-slope)
    assert q["lambda_z_n_points"].magnitude == 4


def test_run_rows_matches_nca_flags_and_shapes() -> None:
    tc = oral_timecourse()
    batch = Timecourses.from_timecourses([tc, tc])
    values = run_rows(
        batch.times,
        batch.values,
        dose_amount=batch.dose_amount,
        dose_time=batch.dose_time,
        dose_duration=batch.dose_duration,
        route=batch.route,
        options=NCAOptions(),
    )
    result = nca(batch)
    assert set(values) == {*result.parameters, "flags", "n"} - {"n"}
    for name in result.parameters:
        assert values[name].shape == (2,)
    np.testing.assert_array_equal(values["flags"], result["flags"].values)


def test_nca_keeps_sample_coordinates() -> None:
    time = np.array([0.5, 1, 2, 4, 8, 12, 24])
    values = np.stack([10 * np.exp(-0.2 * time), 12 * np.exp(-0.25 * time)])
    batch = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b"], "period": ("individual", [1, 2])},
        dose={"amount": np.array([100.0, 100.0]), "unit": "mg"},
        route=Route.IV_BOLUS,
    )
    result = nca(batch)
    assert result.ds["period"].to_numpy().tolist() == [1, 2]
    sample = result.sample("auc_inf_obs", dim="individual")
    assert sample.coords["period"].tolist() == [1, 2]
    assert sample.labels is not None and sample.labels.tolist() == ["a", "b"]


def test_nca_rejects_coordinate_named_like_a_variable() -> None:
    time = np.array([0.5, 1, 2, 4, 8, 12, 24])
    values = np.stack([10 * np.exp(-0.2 * time), 12 * np.exp(-0.25 * time)])
    batch = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b"], "n": ("individual", [1, 2])},
        dose={"amount": np.array([100.0, 100.0]), "unit": "mg"},
        route=Route.IV_BOLUS,
    )
    with pytest.raises(ValueError, match="collides"):
        nca(batch)
