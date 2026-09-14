import numpy as np
import pytest

from pkpdutils.timecourse import Dose, DosingRegimen, Route


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
