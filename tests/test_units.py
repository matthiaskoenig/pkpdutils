import pytest

from pkpdutils.units import (
    Q_,
    check_dose_unit,
    is_per_bodyweight,
    normalize_clearance,
    normalize_volume,
    parse_unit,
    unit_str,
    ureg,
)


def test_registry_custom_units() -> None:
    assert Q_(1, "percent").to("dimensionless").magnitude == pytest.approx(0.01)
    assert Q_(1, "none").dimensionless
    assert Q_(1, "IU").check("[activity_amount]")


def test_parse_unit() -> None:
    assert parse_unit("ng/ml") == ureg.Unit("nanogram / milliliter")
    with pytest.raises(ValueError, match="not_a_unit"):
        parse_unit("not_a_unit")


@pytest.mark.parametrize("unit", ["", "   "])
def test_parse_unit_rejects_the_empty_unit(unit: str) -> None:
    # B4: '' parses as `dimensionless` and then breaks the derived units of the
    # analyses; the spelling of a dimensionless quantity is 'dimensionless'
    with pytest.raises(ValueError, match="dimensionless"):
        parse_unit(unit)


@pytest.mark.parametrize(
    "unit", ["mg", "mmol", "mg/kg", "µmol/kg", "g", "IU", "IU/kg", "kIU"]
)
def test_check_dose_unit_valid(unit: str) -> None:
    check_dose_unit(unit)


@pytest.mark.parametrize("unit", ["mg/l", "hr", "l", "mg/kg/hr"])
def test_check_dose_unit_invalid(unit: str) -> None:
    with pytest.raises(ValueError, match="dose"):
        check_dose_unit(unit)


def test_is_per_bodyweight() -> None:
    assert is_per_bodyweight("mg/kg")
    assert is_per_bodyweight("IU/kg")
    assert not is_per_bodyweight("mg")
    assert not is_per_bodyweight("IU")


def test_normalize_volume() -> None:
    assert normalize_volume(Q_(2000, "ml")).magnitude == pytest.approx(2.0)
    assert str(normalize_volume(Q_(2000, "ml")).units) == "liter"
    q = normalize_volume(Q_(0.5, "m**3/kg"))
    assert q.magnitude == pytest.approx(500.0)
    assert str(q.units) == "liter / kilogram"
    unchanged = normalize_volume(Q_(1, "hr"))
    assert str(unchanged.units) == "hour"


def test_normalize_clearance() -> None:
    q = normalize_clearance(Q_(100, "ml/min"))
    assert q.magnitude == pytest.approx(6.0)
    assert str(q.units) == "liter / hour"
    q = normalize_clearance(Q_(1, "ml/min/kg"))
    assert q.magnitude == pytest.approx(0.06)
    assert str(q.units) == "liter / hour / kilogram"


def test_unit_str() -> None:
    assert unit_str("ng/ml") == "nanogram / milliliter"
    assert unit_str(parse_unit("hr")) == "hour"
