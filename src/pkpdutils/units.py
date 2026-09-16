"""Units of the package.

One [pint](https://pint.readthedocs.io) registry per process, `ureg`, is shared
by every timecourse, result and quantity of the package; quantities of different
registries cannot be combined, which is why nothing creates a registry of its
own. Numerics run on plain arrays in the units of the input, pint is used at the
boundaries: parsing unit strings, deriving the units of results and converting
volumes and clearances to their conventional units.

The helpers which take a unit string and answer a question about it
(`parse_unit`, `check_dose_unit`, `is_per_bodyweight`) are cached: pint parsing
is not cheap, the same handful of unit strings is parsed for every timecourse,
dose and parameter, and units are immutable, so the answer of a string never
changes within a process.

```python
from pkpdutils.units import Q_, ureg

dose = Q_(100, "mg")
time = Q_([0, 1, 2], "hr")
```
"""

from functools import lru_cache

import pint
from pint.facets.plain import PlainQuantity, PlainUnit

#: the unit registry of the package
ureg: pint.UnitRegistry = pint.UnitRegistry()
ureg.define("none = count")
ureg.define("IU = [activity_amount]")

#: shortcut for creating quantities
Q_ = ureg.Quantity

#: type of a quantity of the registry, for annotations
Quantity = PlainQuantity

#: type of a unit of the registry, for annotations
Unit = PlainUnit

#: dimensionalities a dose may have: an amount as mass, as substance or as
#: activity (`IU`), or such an amount per body weight
DOSE_DIMENSIONS: tuple[str, ...] = (
    "[mass]",
    "[substance]",
    "[activity_amount]",
    "[mass] / [mass]",
    "[substance] / [mass]",
    "[activity_amount] / [mass]",
)

#: entries the caches of the unit helpers keep; the number of distinct unit
#: strings of an analysis is small (the units of the values, the times and the
#: doses and the unit expressions of the parameters derived from them)
CACHE_SIZE: int = 1024


def short_unit(unit: str) -> str:
    """A unit in the short symbols of pint, `mg/l` for `milligram / liter`.

    The analyses derive their units with pint and store its canonical long
    form (`milligram / liter`, `hour * milligram / liter`) in the `units`
    attributes of a result, which is too long for a table header or an axis
    label. A string in that long form is written in the short symbols of the
    registry (the `~P` format of pint); a string the user spelled themselves
    (`hr`, `ng/ml`, anything which is not the canonical form of the unit it
    names) is kept as it is, so that a table or a figure carries the unit as
    the data carries it. A dimensionless or empty unit gives the empty string,
    and a string which is not a unit of the registry is passed through
    unchanged.

    Args:
        unit: the unit string of a variable.

    Returns:
        The short unit, empty for a dimensionless or empty unit.
    """
    if not unit.strip() or unit == "dimensionless":
        return ""
    try:
        parsed = parse_unit(unit)
    except ValueError:
        return unit
    return f"{parsed:~P}" if unit == str(parsed) else unit


@lru_cache(maxsize=CACHE_SIZE)
def parse_unit(unit: str) -> Unit:
    """Parse a unit string with the registry of the package.

    The empty string is rejected: pint parses it as `dimensionless`, but an
    empty unit string then composes into the unit expressions of the derived
    parameters as `"()"`, so a dimensionless quantity (a pharmacodynamic score,
    a ratio) is spelled `"dimensionless"`.

    The result is cached per unit string (`CACHE_SIZE`): parsing is the most
    frequent pint call of the package (every timecourse, every dose, every
    parameter of a result) and a `pint.Unit` is immutable, so every caller of
    the same string can share one object.

    Args:
        unit: unit string, e.g. `"ng/ml"` or `"hr"`

    Returns:
        The unit.

    Raises:
        ValueError: if the string is empty or is not a unit of the registry.
    """
    if not unit.strip():
        raise ValueError(
            "'' is not a unit: use 'dimensionless' for a dimensionless quantity"
        )
    try:
        return ureg.Unit(unit)
    except (pint.UndefinedUnitError, pint.DefinitionSyntaxError, AttributeError) as err:
        raise ValueError(f"'{unit}' is not a unit: {err}") from err


def unit_str(unit: Unit | str) -> str:
    """Canonical string of a unit, e.g. `"nanogram / milliliter"` for `"ng/ml"`.

    Args:
        unit: a unit or a unit string.

    Returns:
        The canonical string of the unit.
    """
    return str(parse_unit(unit) if isinstance(unit, str) else unit)


@lru_cache(maxsize=CACHE_SIZE)
def check_dose_unit(unit: str) -> None:
    """Check that a unit is a dose unit.

    A dose is an amount of substance, as mass (`mg`), as substance (`mmol`) or
    as activity (`IU`, for insulin, heparin, vaccines and enzyme replacement),
    or such an amount per body weight (`mg/kg`, `µmol/kg`, `IU/kg`).

    The check is cached per unit string (`CACHE_SIZE`); a unit which fails it
    raises on every call, as `functools.lru_cache` does not cache exceptions.

    Args:
        unit: unit string of the dose

    Raises:
        ValueError: if the unit has another dimensionality.
    """
    u = parse_unit(unit)
    reduced = (1 * u).to_base_units().to_reduced_units()
    if not any(reduced.check(dimension) for dimension in DOSE_DIMENSIONS):
        raise ValueError(
            f"A dose must be in {DOSE_DIMENSIONS}, "
            f"not '{reduced.dimensionality}' ('{unit}')"
        )


@lru_cache(maxsize=CACHE_SIZE)
def is_per_bodyweight(unit: str) -> bool:
    """Check whether a dose unit is an amount per body weight.

    The answer is cached per unit string (`CACHE_SIZE`).

    Args:
        unit: unit string of the dose, e.g. `"mg/kg"`.

    Returns:
        `True` if the unit is an amount per body weight, e.g. `"mg/kg"`,
        `"µmol/kg"` or `"IU/kg"`.
    """
    reduced = (1 * parse_unit(unit)).to_base_units().to_reduced_units()
    return any(
        reduced.check(dimension)
        for dimension in (
            "[mass] / [mass]",
            "[substance] / [mass]",
            "[activity_amount] / [mass]",
        )
    )


def normalize_volume(q: Quantity) -> Quantity:
    """Convert a volume to `liter` and a volume per body weight to `liter/kilogram`.

    Anything else is returned unchanged.

    Args:
        q: quantity to normalize.

    Returns:
        The quantity converted to `liter` or `liter / kilogram`, or `q` unchanged.
    """
    reduced = q.to_base_units().to_reduced_units()
    if reduced.check("[length] ** 3"):
        return q.to("liter")
    if reduced.check("[length] ** 3 / [mass]"):
        return q.to("liter / kilogram")
    return q


def normalize_clearance(q: Quantity) -> Quantity:
    """Convert a clearance to `liter/hour` and one per body weight to `liter/hour/kilogram`.

    Anything else is returned unchanged.

    Args:
        q: quantity to normalize.

    Returns:
        The quantity converted to `liter / hour` or `liter / hour / kilogram`,
        or `q` unchanged.
    """
    reduced = q.to_base_units().to_reduced_units()
    if reduced.check("[length] ** 3 / [time]"):
        return q.to("liter / hour")
    if reduced.check("[length] ** 3 / [time] / [mass]"):
        return q.to("liter / hour / kilogram")
    return q
