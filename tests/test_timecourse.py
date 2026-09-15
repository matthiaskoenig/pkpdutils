import numpy as np
import pandas as pd
import pytest

from pkpdutils.timecourse import Dose, Dosing, DosingRegimen, Route, Timecourse
from pkpdutils.units import ureg


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


def test_dose_infusion_rejects_a_nan_duration() -> None:
    # `NaN <= 0` is False: an unknown duration must not pass as an infusion
    with pytest.raises(ValueError, match="duration"):
        Dose(amount=1, unit="mg", route=Route.IV_INFUSION, duration=float("nan"))


def test_dosing_infusion_rejects_a_nan_duration() -> None:
    with pytest.raises(ValueError, match="duration"):
        Dosing(
            amounts=[1, 1],
            times=[0, 12],
            durations=[0.5, np.nan],
            unit="mg",
            route=Route.IV_INFUSION,
        )


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
    assert tc.sd is not None
    np.testing.assert_allclose(tc.sd, [0.0, 0.2, 0.3])


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
    assert tc.se is not None
    np.testing.assert_allclose(tc.se, [1.0, 2.0])


def test_timecourse_derives_sd_from_se() -> None:
    tc = Timecourse(
        time=[0, 1], value=[1, 2], se=[1.0, 2.0], n=[4, 9], time_unit="hr", unit="mg/l"
    )
    assert tc.sd is not None
    np.testing.assert_allclose(tc.sd, [2.0, 6.0])
    assert tc.n is not None
    np.testing.assert_allclose(tc.n, [4, 9])


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


def _curve(value: list[float]) -> Timecourse:
    return Timecourse(
        time=[0, 1, 2],
        value=value,
        sd=[0.0, 0.2, 0.1],
        n=5,
        time_unit="hr",
        unit="mg/l",
        substance="caffeine",
        label="a",
        dose=Dose(amount=100, unit="mg"),
    )


def test_timecourse_equal_and_hash_equal() -> None:
    tc1 = _curve([0.0, np.nan, 1.0])
    tc2 = _curve([0.0, np.nan, 1.0])
    assert tc1 == tc2
    assert hash(tc1) == hash(tc2)
    assert len({tc1, tc2}) == 1


def test_timecourse_unequal_values() -> None:
    assert _curve([0.0, np.nan, 1.0]) != _curve([0.0, np.nan, 2.0])


def test_timecourse_equality_with_other_type() -> None:
    tc = _curve([0.0, 1.0, 2.0])
    assert tc != "not a timecourse"
    assert (tc == 42) is False


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
    assert tc2.sd is not None and tc.sd is not None
    np.testing.assert_allclose(tc2.sd, tc.sd)
    assert tc2.n is not None
    np.testing.assert_allclose(tc2.n, 5)


def test_timecourse_from_dataframe_columns() -> None:
    df = pd.DataFrame({"t": [0, 1, 2], "c": [0, 2, 1]})
    tc = Timecourse.from_dataframe(
        df, time="t", value="c", time_unit="min", unit="ng/ml"
    )
    np.testing.assert_allclose(tc.time, [0, 1, 2])
    assert str(tc.time_q.units) == "minute"


def test_dosing_validation_and_properties() -> None:
    d = Dosing(amounts=[100, 100, 50], times=[0, 12, 24], unit="mg", route=Route.ORAL)
    assert d.n_doses == 3 and len(d) == 3
    assert d.times.tolist() == [0.0, 12.0, 24.0]
    assert d.intervals.tolist() == [12.0, 12.0]
    assert d.tau == 12.0 and d.is_regular
    assert d.first.amount == 100.0 and d.last.amount == 50.0 and d.last.time == 24.0
    assert d.total_amount == 250.0 and d.quantity.units == ureg.mg
    assert not d.per_bodyweight
    assert d.durations is None
    irregular = Dosing(amounts=[1, 1, 1], times=[0, 8, 24], unit="mg")
    assert irregular.tau is None and not irregular.is_regular
    single = Dosing.single(Dose(amount=5, unit="mg/kg", route=Route.IV_BOLUS, time=2.0))
    assert (
        single.n_doses == 1
        and single.tau is None
        and single.per_bodyweight
        and single.route is Route.IV_BOLUS
    )
    assert single.shifted(2.0).times.tolist() == [0.0]


def test_dosing_sorts_and_rejects_duplicates_and_mixed_routes() -> None:
    d = Dosing(amounts=[1, 2], times=[12, 0], unit="mg")
    assert d.times.tolist() == [0.0, 12.0] and d.amounts.tolist() == [2.0, 1.0]
    with pytest.raises(ValueError, match="Duplicate"):
        Dosing(amounts=[1, 1], times=[0, 0], unit="mg")
    with pytest.raises(ValueError, match="length"):
        Dosing(amounts=[1, 1], times=[0], unit="mg")
    with pytest.raises(ValueError, match="at least one"):
        Dosing(amounts=[], times=[], unit="mg")
    with pytest.raises(ValueError, match="duration"):
        Dosing(amounts=[1], times=[0], unit="mg", route=Route.IV_INFUSION)
    with pytest.raises(ValueError, match="duration"):
        Dosing(amounts=[1], times=[0], durations=[0.5], unit="mg", route=Route.ORAL)
    inf = Dosing(
        amounts=[1, 1],
        times=[0, 12],
        durations=[0.5, 0.5],
        unit="mg",
        route=Route.IV_INFUSION,
    )
    assert inf.doses[1].duration == 0.5
    with pytest.raises(ValueError, match="route"):
        Dosing.from_doses(
            [
                Dose(amount=1, unit="mg"),
                Dose(amount=1, unit="mg", route=Route.IV_BOLUS, time=1),
            ]
        )
    with pytest.raises(ValueError, match="unit"):
        Dosing.from_doses(
            [Dose(amount=1, unit="mg"), Dose(amount=1, unit="mmol", time=1)]
        )


def test_dosing_regimen_constructors() -> None:
    dose = Dose(amount=100, unit="mg", route=Route.ORAL, time=1.0)
    d = Dosing.regimen(dose, interval=12, n_doses=4)
    assert (
        d.times.tolist() == [1.0, 13.0, 25.0, 37.0]
        and d.amounts.tolist() == [100.0] * 4
    )
    assert DosingRegimen(dose=dose, interval=12, n_doses=4).dosing() == d
    with pytest.raises(ValueError, match="n_doses"):
        DosingRegimen(dose=dose, interval=12).dosing()
    assert Dosing.from_doses(d.doses) == d


def test_timecourse_dose_keyword_and_property() -> None:
    dose = Dose(amount=100, unit="mg", route=Route.ORAL, time=0.5)
    tc = Timecourse(
        time=[1, 2, 4], value=[1, 2, 1], time_unit="hr", unit="mg/l", dose=dose
    )
    assert tc.dosing is not None and tc.dosing.n_doses == 1
    assert tc.dose == dose
    protocol = Dosing.regimen(dose, interval=12, n_doses=3)
    tc2 = Timecourse(
        time=[1, 2, 4, 13, 25, 30],
        value=[1, 2, 1, 3, 3, 2],
        time_unit="hr",
        unit="mg/l",
        dosing=protocol,
    )
    assert tc2.dose == dose and tc2.dosing == protocol
    with pytest.raises(ValueError, match="not both"):
        Timecourse(
            time=[1], value=[1], time_unit="hr", unit="mg/l", dose=dose, dosing=protocol
        )
    assert (
        Timecourse(time=[1, 2], value=[1, 2], time_unit="hr", unit="mg/l").dose is None
    )


def test_relative_to_dose_first_and_last() -> None:
    protocol = Dosing(amounts=[1, 1], times=[2, 14], unit="mg")
    tc = Timecourse(
        time=[3, 8, 15, 20],
        value=[1, 2, 3, 4],
        time_unit="hr",
        unit="mg/l",
        dosing=protocol,
    )
    first = tc.relative_to_dose()
    assert (
        first.time.tolist() == [1, 6, 13, 18]
        and first.dosing is not None
        and first.dosing.times.tolist() == [0.0, 12.0]
    )
    last = tc.relative_to_dose(which="last")
    assert (
        last.time.tolist() == [-11, -6, 1, 6]
        and last.dosing is not None
        and last.dosing.times.tolist() == [-12.0, 0.0]
    )
    assert (
        Timecourse(
            time=[1, 2], value=[1, 2], time_unit="hr", unit="mg/l"
        ).relative_to_dose()
        is not None
    )
