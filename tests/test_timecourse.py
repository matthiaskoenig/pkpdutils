import numpy as np
import pandas as pd
import pytest

from pkpdutils.timecourse import Dose, DosingRegimen, Route, Timecourse


def test_route() -> None:
    assert Route.IV_BOLUS.is_iv
    assert Route.IV_INFUSION.is_iv
    assert not Route.ORAL.is_iv
    assert Route("oral") is Route.ORAL


def test_dose_defaults() -> None:
    dose = Dose(amount=100, unit="mg")
    assert dose.route is Route.ORAL
    assert dose.time == 0.0
    assert dose.duration is None
    assert dose.quantity.magnitude == 100
    assert str(dose.quantity.units) == "milligram"
    assert not dose.per_bodyweight


def test_dose_per_bodyweight() -> None:
    assert Dose(amount=2, unit="mg/kg").per_bodyweight


def test_dose_invalid_unit() -> None:
    with pytest.raises(ValueError, match="dose"):
        Dose(amount=1, unit="mg/l")


def test_dose_negative_amount() -> None:
    with pytest.raises(ValueError):
        Dose(amount=-1, unit="mg")


def test_dose_infusion_requires_duration() -> None:
    with pytest.raises(ValueError, match="duration"):
        Dose(amount=1, unit="mg", route=Route.IV_INFUSION)
    dose = Dose(amount=1, unit="mg", route=Route.IV_INFUSION, duration=0.5)
    assert dose.duration == 0.5


def test_dose_duration_only_for_infusion() -> None:
    with pytest.raises(ValueError, match="duration"):
        Dose(amount=1, unit="mg", route=Route.ORAL, duration=0.5)


def test_dose_is_frozen() -> None:
    dose = Dose(amount=1, unit="mg")
    with pytest.raises(ValueError):
        dose.amount = 2  # ty: ignore[invalid-assignment]


def test_dosing_regimen() -> None:
    regimen = DosingRegimen(dose=Dose(amount=100, unit="mg"), interval=12, n_doses=3)
    np.testing.assert_allclose(regimen.dose_times(), [0.0, 12.0, 24.0])


def test_dosing_regimen_validation() -> None:
    with pytest.raises(ValueError):
        DosingRegimen(dose=Dose(amount=100, unit="mg"), interval=0)
    with pytest.raises(ValueError):
        DosingRegimen(dose=Dose(amount=100, unit="mg"), interval=12, n_doses=0)
    with pytest.raises(ValueError, match="n_doses"):
        DosingRegimen(dose=Dose(amount=100, unit="mg"), interval=12).dose_times()


def test_timecourse_basic() -> None:
    tc = Timecourse(
        time=[0, 1, 2, 4],
        value=[0.0, 2.0, 1.5, 0.5],
        time_unit="hr",
        unit="mg/l",
        substance="caffeine",
    )
    assert tc.size == 4
    assert tc.time.dtype == np.float64
    assert tc.value.dtype == np.float64
    assert tc.dose is None
    assert tc.sd is None and tc.se is None and tc.n is None
    assert str(tc.time_q.units) == "hour"
    assert str(tc.value_q.units) == "milligram / liter"
    assert tc.sd_q is None


def test_timecourse_sorts_unsorted_time() -> None:
    tc = Timecourse(time=[2, 0, 1], value=[1.5, 0.0, 2.0], time_unit="hr", unit="mg/l")
    np.testing.assert_allclose(tc.time, [0, 1, 2])
    np.testing.assert_allclose(tc.value, [0.0, 2.0, 1.5])


def test_timecourse_sorts_uncertainties_with_time() -> None:
    tc = Timecourse(
        time=[2, 0, 1],
        value=[1.5, 0.0, 2.0],
        sd=[0.3, 0.0, 0.2],
        n=5,
        time_unit="hr",
        unit="mg/l",
    )
    np.testing.assert_allclose(tc.sd, [0.0, 0.2, 0.3])  # ty: ignore[no-matching-overload]


def test_timecourse_duplicate_time() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Timecourse(time=[0, 1, 1], value=[0, 1, 2], time_unit="hr", unit="mg/l")


def test_timecourse_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        Timecourse(time=[0, 1, 2], value=[0, 1], time_unit="hr", unit="mg/l")
    with pytest.raises(ValueError, match="length"):
        Timecourse(
            time=[0, 1, 2], value=[0, 1, 2], sd=[0, 1], time_unit="hr", unit="mg/l"
        )


def test_timecourse_too_short() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        Timecourse(time=[0], value=[1], time_unit="hr", unit="mg/l")


def test_timecourse_nan_time() -> None:
    with pytest.raises(ValueError, match="NaN"):
        Timecourse(time=[0, np.nan, 2], value=[0, 1, 2], time_unit="hr", unit="mg/l")


def test_timecourse_nan_value_allowed() -> None:
    tc = Timecourse(time=[0, 1, 2], value=[0, np.nan, 2], time_unit="hr", unit="mg/l")
    assert np.isnan(tc.value[1])


def test_timecourse_invalid_unit() -> None:
    with pytest.raises(ValueError, match="not a unit"):
        Timecourse(time=[0, 1], value=[0, 1], time_unit="hr", unit="mg/foo")


def test_timecourse_derives_se_from_sd() -> None:
    tc = Timecourse(
        time=[0, 1], value=[1, 2], sd=[2.0, 4.0], n=4, time_unit="hr", unit="mg/l"
    )
    np.testing.assert_allclose(tc.se, [1.0, 2.0])  # ty: ignore[no-matching-overload]


def test_timecourse_derives_sd_from_se() -> None:
    tc = Timecourse(
        time=[0, 1], value=[1, 2], se=[1.0, 2.0], n=[4, 9], time_unit="hr", unit="mg/l"
    )
    np.testing.assert_allclose(tc.sd, [2.0, 6.0])  # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(tc.n, [4, 9])  # ty: ignore[no-matching-overload]


def test_timecourse_sd_without_n_keeps_se_none() -> None:
    tc = Timecourse(
        time=[0, 1], value=[1, 2], sd=[2.0, 4.0], time_unit="hr", unit="mg/l"
    )
    assert tc.se is None
    assert tc.n is None


def test_timecourse_n_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        Timecourse(time=[0, 1], value=[1, 2], n=[4, 9, 1], time_unit="hr", unit="mg/l")


def test_timecourse_relative_to_dose() -> None:
    tc = Timecourse(
        time=[10, 11, 12],
        value=[0, 2, 1],
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=1, unit="mg", time=10),
    )
    rel = tc.relative_to_dose()
    np.testing.assert_allclose(rel.time, [0, 1, 2])
    assert rel.dose is not None and rel.dose.time == 0.0
    np.testing.assert_allclose(tc.time, [10, 11, 12])


def test_timecourse_relative_to_dose_without_dose() -> None:
    tc = Timecourse(time=[10, 11], value=[0, 2], time_unit="hr", unit="mg/l")
    assert tc.relative_to_dose() is tc


def test_timecourse_dataframe_roundtrip() -> None:
    tc = Timecourse(
        time=[0, 1, 2],
        value=[0, 2, 1],
        sd=[0, 0.2, 0.1],
        n=5,
        time_unit="hr",
        unit="mg/l",
        substance="caffeine",
        label="a",
    )
    df = tc.to_dataframe()
    assert list(df.columns) == ["time", "value", "sd", "se", "n"]
    tc2 = Timecourse.from_dataframe(
        df, time_unit="hr", unit="mg/l", sd="sd", n="n", substance="caffeine"
    )
    np.testing.assert_allclose(tc2.value, tc.value)
    np.testing.assert_allclose(tc2.sd, tc.sd)  # ty: ignore[no-matching-overload]
    np.testing.assert_allclose(tc2.n, 5)  # ty: ignore[no-matching-overload]


def test_timecourse_from_dataframe_columns() -> None:
    df = pd.DataFrame({"t": [0, 1, 2], "c": [0, 2, 1]})
    tc = Timecourse.from_dataframe(
        df, time="t", value="c", time_unit="min", unit="ng/ml"
    )
    np.testing.assert_allclose(tc.time, [0, 1, 2])
    assert str(tc.time_q.units) == "minute"
