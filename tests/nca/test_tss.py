"""Time to steady state from the troughs of the dosing intervals."""

import numpy as np
import pytest

from pkpdutils import (
    AUCMethod,
    Dose,
    Dosing,
    NCAOptions,
    Route,
    Timecourse,
    Timecourses,
    nca,
    nca_single,
)
from pkpdutils.nca import superposition, time_to_steady_state

C0, TAU = 10.0, 12.0
DOSE = Dose(amount=100, unit="mg", route=Route.IV_BOLUS)
LOG = NCAOptions(auc_method=AUCMethod.LOG)


def single_dose(k: float) -> Timecourse:
    t = np.array([0.0, 0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48, 72, 96])
    return Timecourse(
        time=t,
        value=C0 * np.exp(-k * t),
        time_unit="hr",
        unit="mg/l",
        dose=DOSE,
        substance="x",
    )


def superposed(k: float, n_doses: int) -> Timecourse:
    protocol = Dosing.regimen(DOSE, interval=TAU, n_doses=n_doses)
    return superposition(
        single_dose(k),
        protocol,
        options=LOG,
        grid=np.arange(0.0, n_doses * TAU + 1e-9, 0.25),
    )


def test_the_monoexponential_estimate_recovers_the_rate_constant() -> None:
    k = 0.2
    result = nca_single(superposed(k, 10), options=LOG)
    estimate = time_to_steady_state(result)
    # the trough of interval n is c_ss (1 - exp(-k n tau)) exactly, so the fit
    # recovers k and the time to 90 % of the plateau is ln(10) / k
    assert float(estimate.tss) == pytest.approx(np.log(10.0) / k, rel=1e-6)
    assert estimate.method == "monoexponential"
    assert estimate.fraction == 0.9
    assert estimate.tss.attrs["units"] == "hour"
    # the plateau is the trough of the last interval
    assert estimate.c_ss is not None
    assert float(estimate.c_ss) == pytest.approx(
        C0 * np.exp(-k * TAU) / (1 - np.exp(-k * TAU)), rel=1e-6
    )
    assert estimate.c_ss.attrs["units"] == "milligram / liter"


def test_another_fraction_of_the_plateau() -> None:
    k = 0.05
    result = nca_single(superposed(k, 16), options=LOG)
    estimate = time_to_steady_state(result, fraction=0.95)
    assert float(estimate.tss) == pytest.approx(-np.log(0.05) / k, rel=1e-6)


def test_the_stepwise_estimate_finds_the_plateau_of_a_noisy_study() -> None:
    k = 0.05
    predicted = superposed(k, 16)
    rng = np.random.default_rng(7)
    noisy = Timecourse(
        time=predicted.time,
        value=predicted.value * rng.normal(1.0, 0.05, predicted.value.size),
        time_unit="hr",
        unit="mg/l",
        dosing=predicted.dosing,
        substance="x",
    )
    result = nca_single(noisy, options=LOG)
    estimate = time_to_steady_state(result, method="stepwise")
    # the troughs rise over the first three intervals and scatter around the
    # plateau afterwards: the trend is no longer significant from 24 h on
    assert float(estimate.tss) == 24.0
    assert estimate.c_ss is None
    assert estimate.method == "stepwise"
    starts = result.intervals()["interval_start"].to_numpy()
    assert float(estimate.tss) in set(starts.tolist())


def test_the_estimate_of_a_batch_and_its_frame() -> None:
    batch = Timecourses.from_timecourses(
        [
            superposed(k, 10).model_copy(update={"label": label})
            for k, label in ((0.2, "s1"), (0.1, "s2"))
        ],
        dim="individual",
    )
    estimate = time_to_steady_state(nca(batch, options=LOG))
    assert estimate.tss.dims == ("individual",)
    assert estimate.tss.to_numpy() == pytest.approx(
        [np.log(10.0) / 0.2, np.log(10.0) / 0.1], rel=1e-5
    )
    frame = estimate.to_dataframe()
    # the sample coordinates travel with the estimate, the dose among them
    assert list(frame.columns) == ["individual", "dose_amount", "tss", "c_ss"]
    assert frame["individual"].tolist() == ["s1", "s2"]


def test_a_single_dose_result_has_no_troughs() -> None:
    result = nca_single(single_dose(0.2), options=LOG)
    with pytest.raises(ValueError, match="no per-interval troughs"):
        time_to_steady_state(result)


def test_the_bounds_of_the_fraction_and_the_level() -> None:
    result = nca_single(superposed(0.2, 10), options=LOG)
    with pytest.raises(ValueError, match=r"'fraction' is 1\.0"):
        time_to_steady_state(result, fraction=1.0)
    with pytest.raises(ValueError, match=r"'alpha' is 0\.0"):
        time_to_steady_state(result, method="stepwise", alpha=0.0)


def test_too_few_intervals_give_no_estimate() -> None:
    result = nca_single(superposed(0.2, 2), options=LOG)
    assert np.isnan(float(time_to_steady_state(result).tss))
    assert np.isnan(float(time_to_steady_state(result, method="stepwise").tss))


def test_the_frame_of_a_single_curve_result() -> None:
    result = nca_single(superposed(0.2, 10), options=LOG)
    frame = time_to_steady_state(result).to_dataframe()
    # a result without sample dimensions is one row; the scalar coordinates of
    # the result travel with it
    assert len(frame) == 1
    assert list(frame.columns) == ["dose_amount", "tss", "c_ss"]
    assert frame["tss"].to_numpy()[0] == pytest.approx(np.log(10.0) / 0.2, rel=1e-6)
    stepwise = time_to_steady_state(result, method="stepwise").to_dataframe()
    assert list(stepwise.columns) == ["dose_amount", "tss"]
