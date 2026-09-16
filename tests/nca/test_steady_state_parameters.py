"""The steady state parameters added by the feature round.

The trough variants of the fluctuation, the peak-trough ratio, the effective
half-life, the accumulation ratios of the peak and the trough and the
completion of a last dosing interval which falls short of its end.
"""

import numpy as np
import pytest

from pkpdutils import (
    AUCMethod,
    Dose,
    Dosing,
    NCAFlag,
    NCAOptions,
    Route,
    Timecourse,
    UncertaintyMethod,
    nca_single,
)
from pkpdutils.nca import superposition
from pkpdutils.nca.uncertainty import terminal_independent

K, C0, TAU = 0.2, 10.0, 12.0
DOSE = Dose(amount=100, unit="mg", route=Route.IV_BOLUS)
#: the value of the analytic steady state curve at the dose
AMPLITUDE = C0 / (1 - np.exp(-K * TAU))
LOG = NCAOptions(auc_method=AUCMethod.LOG)


def single_dose() -> Timecourse:
    """A mono-exponential bolus curve, `C0 exp(-k t)`."""
    t = np.array([0.0, 0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48])
    return Timecourse(
        time=t,
        value=C0 * np.exp(-K * t),
        time_unit="hr",
        unit="mg/l",
        dose=DOSE,
        substance="x",
    )


def superposed(n_doses: int = 8) -> Timecourse:
    """The curve of `n_doses` boluses every `TAU` hours, predicted by superposition."""
    protocol = Dosing.regimen(DOSE, interval=TAU, n_doses=n_doses)
    return superposition(
        single_dose(),
        protocol,
        options=LOG,
        grid=np.arange(0.0, n_doses * TAU + 1e-9, 0.25),
    )


def steady_state_curve(t_last: float = TAU) -> Timecourse:
    """One analytic steady state interval sampled up to `t_last`."""
    t = np.array([0.5, 1, 2, 4, 6, 8, 10, t_last])
    return Timecourse(
        time=t,
        value=AMPLITUDE * np.exp(-K * t),
        time_unit="hr",
        unit="mg/l",
        dose=DOSE,
        substance="x",
    )


def test_the_trough_variants_and_the_peak_trough_ratio() -> None:
    q = nca_single(superposed(), options=LOG).to_quantities()
    # at steady state the interval is C0 exp(-k t) / (1 - exp(-k tau)), so
    # Cmax,ss / Ctrough = exp(k tau) and Cmax,ss - Ctrough = C0
    assert q["ptr"].magnitude == pytest.approx(np.exp(K * TAU), rel=1e-3)
    assert q["swing_tau"].magnitude == pytest.approx(np.exp(K * TAU) - 1.0, rel=1e-3)
    # (Cmax,ss - Ctrough) / Cavg = C0 / (C0 / (k tau)) = k tau
    assert q["fluctuation_tau"].magnitude == pytest.approx(K * TAU, rel=1e-3)
    # every interval of this curve has its minimum at its end
    assert q["swing_tau"].magnitude == pytest.approx(q["swing"].magnitude, rel=1e-6)
    assert str(q["ptr"].units) == "dimensionless"


def test_the_effective_half_life_of_a_monoexponential_drug_is_its_half_life() -> None:
    # thalf_eff = ln(2) MRT (PKNCA pk.calc.thalf.eff) and MRT = 1 / k for a
    # mono-exponential decline, so the effective half-life is ln(2) / k
    for result in (
        nca_single(single_dose(), options=LOG),
        nca_single(superposed(), options=LOG),
    ):
        q = result.to_quantities()
        assert q["thalf_eff"].magnitude == pytest.approx(np.log(2.0) / K, rel=1e-6)
        assert q["thalf_eff"].magnitude == pytest.approx(q["thalf"].magnitude, rel=1e-6)
        assert str(q["thalf_eff"].units) == "hour"


def test_the_effective_half_life_of_an_infusion_follows_the_corrected_mrt() -> None:
    duration = 0.5
    dose = Dose(amount=100, unit="mg", route=Route.IV_INFUSION, duration=duration)
    t = np.array([0.5, 1, 2, 4, 8, 12, 24])
    curve = Timecourse(
        time=t,
        value=C0 * np.exp(-K * t),
        time_unit="hr",
        unit="mg/l",
        dose=dose,
        substance="x",
    )
    q = nca_single(curve, options=LOG).to_quantities()
    assert q["thalf_eff"].magnitude == pytest.approx(
        np.log(2.0) * q["mrt"].magnitude, rel=1e-12
    )


def test_the_accumulation_ratios_of_the_peak_and_the_trough() -> None:
    q = nca_single(superposed(), options=LOG).to_quantities()
    accumulation = 1.0 / (1.0 - np.exp(-K * TAU))
    for name in (
        "accumulation_ratio_cmax_obs",
        "accumulation_ratio_cmin_obs",
        "accumulation_ratio_ctrough_obs",
    ):
        assert q[name].magnitude == pytest.approx(accumulation, rel=1e-3), name
        assert str(q[name].units) == "dimensionless"


def test_the_accumulation_ratios_of_a_single_interval_are_nan() -> None:
    q = nca_single(steady_state_curve(), options=LOG.model_copy(update={"tau": TAU}))
    for name in (
        "accumulation_ratio_obs",
        "accumulation_ratio_cmax_obs",
        "accumulation_ratio_cmin_obs",
        "accumulation_ratio_ctrough_obs",
    ):
        assert np.isnan(float(q[name])), name


def test_a_covered_interval_extrapolates_nothing() -> None:
    q = nca_single(steady_state_curve(), options=LOG.model_copy(update={"tau": TAU}))
    assert float(q["auc_tau_extrap_fraction"]) == 0.0


def test_a_short_last_interval_is_completed_within_the_tolerance() -> None:
    # the last sample at 0.95 tau, 5 % short of the end of the interval
    t_last = 0.95 * TAU
    options = LOG.model_copy(update={"tau": TAU})
    q = nca_single(steady_state_curve(t_last), options=options).to_quantities()
    # the log trapezoid is exact for the exponential and the terminal
    # regression recovers k, so the completed interval is the analytic one
    assert q["auc_tau"].magnitude == pytest.approx(C0 / K, rel=1e-9)
    tail = AMPLITUDE / K * (np.exp(-K * t_last) - np.exp(-K * TAU))
    assert q["auc_tau_extrap_fraction"].magnitude == pytest.approx(
        tail / (C0 / K), rel=1e-6
    )
    # the trough is the regression at the end of the interval
    assert q["ctrough"].magnitude == pytest.approx(
        AMPLITUDE * np.exp(-K * TAU), rel=1e-6
    )
    assert q["cmin_ss"].magnitude == pytest.approx(q["ctrough"].magnitude, rel=1e-9)
    assert q["cmax_ss"].magnitude == pytest.approx(AMPLITUDE, rel=1e-6)
    assert q["cavg"].magnitude == pytest.approx(C0 / K / TAU, rel=1e-9)


def test_a_last_interval_beyond_the_tolerance_stays_incomplete() -> None:
    # the last sample at 0.85 tau, beyond the 10 % tolerance
    options = LOG.model_copy(update={"tau": TAU})
    result = nca_single(steady_state_curve(0.85 * TAU), options=options)
    for name in ("auc_tau", "ctrough", "cavg", "auc_tau_extrap_fraction"):
        assert np.isnan(float(result[name])), name
    assert "INCOMPLETE_INTERVAL" in result.flags()


def test_the_tolerance_switches_the_completion_off() -> None:
    options = LOG.model_copy(update={"tau": TAU, "tau_tolerance": 0.0})
    result = nca_single(steady_state_curve(0.95 * TAU), options=options)
    assert np.isnan(float(result["auc_tau"]))
    assert "INCOMPLETE_INTERVAL" in result.flags()


def test_a_completed_interval_carries_no_incomplete_flag() -> None:
    options = LOG.model_copy(update={"tau": TAU})
    result = nca_single(steady_state_curve(0.95 * TAU), options=options)
    assert NCAFlag.INCOMPLETE_INTERVAL.name not in result.flags()
    # the per-interval variables of the completed interval are written as well
    frame = result.intervals()
    assert frame["interval_auc"].to_numpy()[-1] == pytest.approx(C0 / K, rel=1e-9)


def test_terminal_independent_drops_the_completable_parameters() -> None:
    # a completed interval carries a tail of the terminal regression, so the
    # parameters of that interval depend on the window as soon as the
    # completion is switched on
    covered = terminal_independent(NCAOptions(tau_tolerance=0.0))
    completable = terminal_independent(NCAOptions())
    for name in ("auc_tau", "cmin_ss", "cmax_ss", "ctrough", "cavg"):
        assert name in covered, name
        assert name not in completable, name
    # the parameters which never read the interval are unaffected
    for name in ("auc_last", "cmax", "clast"):
        assert name in covered and name in completable, name


def group_curve(t_last: float = TAU, n_doses: int = 3) -> Timecourse:
    """A group curve of `n_doses` boluses with a standard error per point."""
    protocol = Dosing.regimen(DOSE, interval=TAU, n_doses=n_doses)
    samples = np.array([0.0, 0.5, 1, 2, 4, 6, 8, 9, 10, 11, t_last])
    time = np.unique(np.concatenate([d + samples for d in protocol.times]))
    value = np.zeros_like(time)
    for d in protocol.times:
        value += np.where(
            time >= d, C0 * np.exp(-K * np.clip(time - d, 0.0, None)), 0.0
        )
    return Timecourse(
        time=time,
        value=value,
        se=0.1 * value,
        n=8,
        time_unit="hr",
        unit="mg/l",
        dosing=protocol,
        substance="x",
    )


def test_the_delta_method_skips_a_window_change_for_the_interval_parameters() -> None:
    # a large step makes the terminal window flip at some points, which is what
    # `NCAFlag.DELTA_WINDOW_CHANGE` reports
    options = LOG.model_copy(
        update={"uncertainty": UncertaintyMethod.DELTA, "delta_step": 0.9}
    )
    curve = group_curve()
    completable = nca_single(curve, options=options)
    covered = nca_single(
        curve, options=options.model_copy(update={"tau_tolerance": 0.0})
    )
    assert "DELTA_WINDOW_CHANGE" in completable.flags()
    # the same exposure, but its uncertainty no longer differentiates across
    # the two regressions of a point at which the window flipped
    assert float(completable["auc_tau"]) == pytest.approx(float(covered["auc_tau"]))
    assert float(completable["auc_tau_se"]) != pytest.approx(
        float(covered["auc_tau_se"])
    )
    assert float(completable["cavg_se"]) != pytest.approx(float(covered["cavg_se"]))
    # a parameter which never reads the dosing interval is untouched
    assert float(completable["auc_last_se"]) == pytest.approx(
        float(covered["auc_last_se"])
    )


def test_a_group_curve_whose_last_interval_is_completed_carries_its_uncertainty() -> (
    None
):
    options = LOG.model_copy(update={"uncertainty": UncertaintyMethod.DELTA})
    result = nca_single(group_curve(t_last=0.95 * TAU), options=options)
    assert np.isfinite(float(result["auc_tau"]))
    assert float(result["auc_tau_extrap_fraction"]) > 0.0
    assert np.isfinite(float(result["auc_tau_se"]))
    assert "INCOMPLETE_INTERVAL" not in result.flags()
