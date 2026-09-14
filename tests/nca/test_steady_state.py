import numpy as np
import pytest

from pkpdutils import Dose, DosingRegimen, Route, Timecourse, Timecourses
from pkpdutils.nca import AUCMethod, NCAOptions, nca, nca_single
from pkpdutils.nca.steady_state import accumulation_ratio, superposition

K, C0, TAU = 0.2, 10.0, 12.0
DOSE = Dose(amount=100, unit="mg", route=Route.IV_BOLUS)


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
    cmax = q["cmax"].magnitude
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


def test_superposition_needs_n_doses_and_terminal_phase() -> None:
    with pytest.raises(ValueError, match="n_doses"):
        superposition(single_dose(), DosingRegimen(dose=DOSE, interval=TAU))
    rising = Timecourse(
        time=[1, 2, 3, 4], value=[1, 2, 3, 4], time_unit="hr", unit="mg/l", dose=DOSE
    )
    with pytest.raises(ValueError, match="lambda_z"):
        superposition(rising, DosingRegimen(dose=DOSE, interval=TAU, n_doses=3))
