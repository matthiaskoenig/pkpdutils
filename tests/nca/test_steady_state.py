import warnings

import numpy as np
import pytest

from pkpdutils import Dose, Dosing, DosingRegimen, Route, Timecourse, Timecourses
from pkpdutils.nca import AUCMethod, NCAOptions, nca, nca_single
from pkpdutils.nca.steady_state import accumulation_ratio, superposition

K, C0, TAU = 0.2, 10.0, 12.0
DOSE = Dose(amount=100, unit="mg", route=Route.IV_BOLUS)
#: amplitude of the analytic steady state curve, its value at t = 0
AMPLITUDE = C0 / (1 - np.exp(-K * TAU))


def log_trapezoid(t: np.ndarray, c: np.ndarray) -> float:
    """Area under a positive curve by the logarithmic trapezoid rule."""
    dt = np.diff(t)
    c1, c2 = c[:-1], c[1:]
    return float(np.sum(dt * (c1 - c2) / np.log(c1 / c2)))


def single_dose() -> Timecourse:
    t = np.array([0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48])
    return Timecourse(
        time=t,
        value=C0 * np.exp(-K * t),
        time_unit="hr",
        unit="mg/l",
        dose=DOSE,
        substance="x",
    )


def steady_state_curve() -> Timecourse:
    """Analytic steady state of a bolus every TAU hours: c0 e^{-kt} / (1 - e^{-k tau}) on [0, tau]."""
    t = np.array([0.5, 1, 2, 4, 6, 8, 10, 12])
    c = C0 * np.exp(-K * t) / (1 - np.exp(-K * TAU))
    return Timecourse(
        time=t, value=c, time_unit="hr", unit="mg/l", dose=DOSE, substance="x"
    )


def test_steady_state_parameters_analytic() -> None:
    regimen = DosingRegimen(dose=DOSE, interval=TAU)
    options = NCAOptions(regimen=regimen, auc_method=AUCMethod.LOG)
    q = nca_single(steady_state_curve(), options).to_quantities()
    accumulation = 1 / (1 - np.exp(-K * TAU))
    # auc over one interval at steady state equals the single dose auc(0-inf) = c0/k
    assert q["auc_tau"].magnitude == pytest.approx(C0 / K, rel=1e-6)
    assert q["ctrough"].magnitude == pytest.approx(
        C0 * np.exp(-K * TAU) * accumulation, rel=1e-6
    )
    assert q["cmin_ss"].magnitude == pytest.approx(q["ctrough"].magnitude)
    assert q["cavg"].magnitude == pytest.approx(C0 / K / TAU, rel=1e-6)
    # the inserted C0 at the dose is the maximum of the interval, not the first sample
    assert q["cmax_ss"].magnitude == pytest.approx(AMPLITUDE, rel=1e-6)
    assert q["cmax"].magnitude == pytest.approx(AMPLITUDE * np.exp(-K * 0.5), rel=1e-6)
    cmax = q["cmax_ss"].magnitude
    cmin = q["cmin_ss"].magnitude
    assert q["fluctuation"].magnitude == pytest.approx(
        (cmax - cmin) / (C0 / K / TAU), rel=1e-6
    )
    assert q["swing"].magnitude == pytest.approx((cmax - cmin) / cmin, rel=1e-6)
    assert q["accumulation_ratio"].magnitude == pytest.approx(accumulation, rel=1e-3)
    assert q["cl_ss"].magnitude == pytest.approx(100 / (C0 / K), rel=1e-6)
    assert str(q["cl_ss"].units) == "liter / hour"
    assert str(q["auc_tau"].units) == "hour * milligram / liter"
    # the single dose parameters are still there
    assert q["lambda_z"].magnitude == pytest.approx(K, rel=1e-6)


def test_steady_state_ctrough_interpolated_when_tau_between_points() -> None:
    tc = steady_state_curve()
    regimen = DosingRegimen(dose=DOSE, interval=11.0)
    q = nca_single(
        tc, NCAOptions(regimen=regimen, auc_method=AUCMethod.LOG)
    ).to_quantities()
    expected = C0 * np.exp(-K * 11.0) / (1 - np.exp(-K * TAU))
    assert q["ctrough"].magnitude == pytest.approx(expected, rel=1e-6)
    assert (
        q["auc_tau"].magnitude
        < nca_single(
            tc,
            NCAOptions(
                regimen=DosingRegimen(dose=DOSE, interval=TAU), auc_method=AUCMethod.LOG
            ),
        )
        .to_quantities()["auc_tau"]
        .magnitude
    )


def test_auc_tau_excludes_the_area_before_the_dose() -> None:
    # the dose is given at t = 2, the sample at t = 0 is a pre-dose sample from
    # the previous interval; only the area over [2, 2 + tau] may count
    dose_time = 2.0
    dose = DOSE.model_copy(update={"time": dose_time})
    t_rel = np.array([-dose_time, 0.5, 1, 2, 4, 6, 8, 10, 12])
    c = AMPLITUDE * np.exp(-K * np.abs(t_rel))
    # the pre-dose sample is the steady state value late in the previous interval
    c[0] = AMPLITUDE * np.exp(-K * (TAU - dose_time))
    tc = Timecourse(
        time=t_rel + dose_time,
        value=c,
        time_unit="hr",
        unit="mg/l",
        dose=dose,
        substance="x",
    )
    options = NCAOptions(
        regimen=DosingRegimen(dose=dose, interval=TAU), auc_method=AUCMethod.LOG
    )
    q = nca_single(tc, options).to_quantities()
    # the segment straddling the dose contributes only its part after the dose:
    # the value at the dose is the logarithmic interpolation of the segment
    frac = (0.0 - t_rel[0]) / (t_rel[1] - t_rel[0])
    c_dose = float(np.exp(np.log(c[0]) + frac * (np.log(c[1]) - np.log(c[0]))))
    expected = log_trapezoid(
        np.concatenate([[0.0], t_rel[1:]]), np.concatenate([[c_dose], c[1:]])
    )
    assert q["auc_tau"].magnitude == pytest.approx(expected, rel=1e-9)
    # the pre-dose segment would add a sizable area
    assert expected < log_trapezoid(t_rel, c)
    assert q["cavg"].magnitude == pytest.approx(expected / TAU, rel=1e-9)
    assert q["cl_ss"].magnitude == pytest.approx(100.0 / expected, rel=1e-9)


def test_fluctuation_uses_the_maximum_inside_the_interval() -> None:
    tc = steady_state_curve()
    beyond = tc.model_copy(
        update={
            "time": np.append(tc.time, 13.0),
            "value": np.append(tc.value, 100.0),
        }
    )
    options = NCAOptions(
        regimen=DosingRegimen(dose=DOSE, interval=TAU), auc_method=AUCMethod.LOG
    )
    a = nca_single(tc, options).to_quantities()
    b = nca_single(beyond, options).to_quantities()
    assert b["cmax"].magnitude == pytest.approx(100.0)
    assert b["cmax_ss"].magnitude == pytest.approx(a["cmax_ss"].magnitude)
    assert b["fluctuation"].magnitude == pytest.approx(a["fluctuation"].magnitude)
    assert b["swing"].magnitude == pytest.approx(a["swing"].magnitude)


def test_steady_state_batch_workers_and_chunks_match_serial() -> None:
    options = NCAOptions(
        regimen=DosingRegimen(dose=DOSE, interval=TAU), auc_method=AUCMethod.LOG
    )
    tc = steady_state_curve()
    curves = [
        tc.model_copy(update={"value": tc.value * scale})
        for scale in (0.5, 1.0, 1.5, 2.0, 2.5)
    ]
    batch = Timecourses.from_timecourses(curves, dim="individual")
    serial = nca(batch, options)
    parallel = nca(batch, options.model_copy(update={"n_workers": 2, "chunk_rows": 2}))
    for name in serial.parameters:
        np.testing.assert_allclose(
            parallel[name].values, serial[name].values, equal_nan=True
        )
    np.testing.assert_array_equal(parallel["flags"].values, serial["flags"].values)


def test_steady_state_tau_beyond_last_point_is_nan() -> None:
    regimen = DosingRegimen(dose=DOSE, interval=48.0)
    q = nca_single(steady_state_curve(), NCAOptions(regimen=regimen)).to_quantities()
    assert np.isnan(q["auc_tau"].magnitude) and np.isnan(q["ctrough"].magnitude)


def test_accumulation_ratio_observed() -> None:
    regimen = DosingRegimen(dose=DOSE, interval=TAU)
    options = NCAOptions(regimen=regimen, auc_method=AUCMethod.LOG)
    ss = nca(Timecourses.from_timecourses([steady_state_curve()]), options)
    sd = nca(Timecourses.from_timecourses([single_dose()]), options)
    ratio = accumulation_ratio(ss, sd)
    assert float(ratio.values[0]) == pytest.approx(1 / (1 - np.exp(-K * TAU)), rel=1e-3)
    assert ratio.attrs["units"] == "dimensionless"


def test_superposition_reaches_analytic_steady_state() -> None:
    regimen = DosingRegimen(dose=DOSE, interval=TAU, n_doses=20)
    predicted = superposition(
        single_dose(), regimen, NCAOptions(auc_method=AUCMethod.LOG)
    )
    assert predicted.dose is not None and predicted.dose.amount == 100
    assert predicted.unit == "mg/l" and predicted.time_unit == "hr"
    t_last_dose = 19 * TAU
    mask = predicted.time >= t_last_dose + 0.5
    t = predicted.time[mask] - t_last_dose
    expected = C0 * np.exp(-K * t) / (1 - np.exp(-K * TAU))
    np.testing.assert_allclose(predicted.value[mask], expected, rtol=1e-3)


def test_superposition_with_a_sample_at_the_dose_time() -> None:
    tc = single_dose()
    at_zero = tc.model_copy(
        update={
            "time": np.concatenate([[0.0], tc.time]),
            "value": np.concatenate([[C0], tc.value]),
        }
    )
    regimen = DosingRegimen(dose=DOSE, interval=TAU, n_doses=3)
    with warnings.catch_warnings():
        # the linear rise before the first sample must not divide by time 0
        warnings.simplefilter("error")
        predicted = superposition(
            at_zero, regimen, NCAOptions(auc_method=AUCMethod.LOG)
        )
    assert np.isfinite(predicted.value).all()
    assert predicted.time[0] == pytest.approx(0.0)
    assert predicted.value[0] == pytest.approx(C0)


def test_superposition_needs_n_doses_and_terminal_phase() -> None:
    with pytest.raises(ValueError, match="n_doses"):
        superposition(single_dose(), DosingRegimen(dose=DOSE, interval=TAU))
    rising = Timecourse(
        time=[1, 2, 3, 4], value=[1, 2, 3, 4], time_unit="hr", unit="mg/l", dose=DOSE
    )
    with pytest.raises(ValueError, match="lambda_z"):
        superposition(rising, DosingRegimen(dose=DOSE, interval=TAU, n_doses=3))


def test_protocol_analysis_uses_the_last_dose_at_steady_state() -> None:
    # two doses, the last one at t = 2: the steady state parameters belong to
    # the interval of the last dose, while the single dose analysis is relative
    # to the first dose of the protocol
    dose_time = 2.0
    dose = DOSE.model_copy(update={"time": dose_time})
    t_rel = np.array([-dose_time, 0.5, 1, 2, 4, 6, 8, 10, 12])
    c = AMPLITUDE * np.exp(-K * np.abs(t_rel))
    c[0] = AMPLITUDE * np.exp(-K * (TAU - dose_time))
    protocol = Dosing(
        amounts=[100, 100],
        times=[dose_time - TAU, dose_time],
        unit="mg",
        route=Route.IV_BOLUS,
    )

    def curve(dosing: Dosing) -> Timecourse:
        return Timecourse(
            time=t_rel + dose_time,
            value=c,
            time_unit="hr",
            unit="mg/l",
            dosing=dosing,
            substance="x",
        )

    options = NCAOptions(
        regimen=DosingRegimen(dose=dose, interval=TAU), auc_method=AUCMethod.LOG
    )
    q = nca_single(curve(protocol), options).to_quantities()
    expected = nca_single(curve(Dosing.single(dose)), options).to_quantities()
    for name in ("auc_tau", "ctrough", "cmin_ss", "cmax_ss", "cavg", "cl_ss"):
        assert q[name].magnitude == pytest.approx(expected[name].magnitude, rel=1e-9)
    # without a regimen the times are relative to the first dose of the protocol
    plain = nca_single(
        curve(protocol), NCAOptions(auc_method=AUCMethod.LOG)
    ).to_quantities()
    assert plain["tmax"].magnitude == pytest.approx(0.5 + dose_time - (dose_time - TAU))
