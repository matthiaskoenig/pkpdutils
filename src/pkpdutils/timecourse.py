"""Timecourses, doses and dosing protocols.

The data model of the package:

- `Timecourse` is one curve, i.e. values over time with units, an optional
  uncertainty (`sd`/`se` and `n` for group data), a `Dosing` protocol and
  metadata.
- `Dosing` is the dosing protocol of a timecourse: the vector of doses given
  and the times they were given, one route and one unit for all of them; a
  single administration stays a `Dose`, `Dosing.single` wraps one into a
  protocol of one dose. `Timecourse` keeps accepting a single `dose=Dose(...)`
  keyword, converted into a protocol of one dose; `Timecourse.dose` reads back
  the first dose of the protocol.
- `Timecourses` is a batch of curves as an `xarray.Dataset` with a `time`
  dimension and any number of sample dimensions (individuals, groups, studies,
  the dimensions of a simulation scan). Every analysis of the package works on
  a `Timecourses` object and returns an `xarray.Dataset` over the same sample
  dimensions.
- `DosingRegimen` describes repeated dosing for steady state analyses;
  `DosingRegimen.dosing()` builds the corresponding `Dosing` protocol.

```python
from pkpdutils.timecourse import Dose, Route, Timecourse

tc = Timecourse(
    time=[0.5, 1, 2, 4, 8, 12],
    value=[1.2, 2.5, 2.1, 1.3, 0.5, 0.2],
    time_unit="hr",
    unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    substance="caffeine",
)
```
"""

import logging
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pkpdutils.units import Q_, Quantity, check_dose_unit, is_per_bodyweight, parse_unit

logger = logging.getLogger(__name__)


class Route(StrEnum):
    """Route of administration.

    `ORAL` stands for every extravascular route (oral, subcutaneous,
    intramuscular, ...): the substance has an absorption phase and the
    parameters which need the fraction absorbed are reported relative to it
    (`cl_f`, `vz_f`).

    A string is coerced to a member wherever a route is taken, ignoring the
    case, surrounding blanks and the separator (`"ORAL"`, `"iv bolus"` and
    `"iv-bolus"` are members).
    """

    IV_BOLUS = "iv_bolus"
    IV_INFUSION = "iv_infusion"
    ORAL = "oral"

    @classmethod
    def _missing_(cls, value: object) -> "Route | None":
        """Coerce a string which is not a member verbatim.

        Args:
            value: the value which is not a member, e.g. `"IV Bolus"`.

        Returns:
            The member it spells, or `None` when it spells none (the
            enumeration then raises its `ValueError`).
        """
        if not isinstance(value, str):
            return None
        name = value.strip().lower().replace("-", "_").replace(" ", "_")
        return next((member for member in cls if member.value == name), None)

    @property
    def is_iv(self) -> bool:
        """Whether the route is intravenous (bolus or infusion)."""
        return self in (Route.IV_BOLUS, Route.IV_INFUSION)


class Dose(BaseModel):
    """A dose of the substance of a timecourse.

    Attributes:
        amount: amount of the dose (non-negative)
        unit: unit of the amount, an amount (`mg`, `mmol`) or an amount per body
            weight (`mg/kg`, `µmol/kg`), see `pkpdutils.units.check_dose_unit`
        route: route of administration
        time: time of the dose in the time unit of the timecourse
        duration: duration of the infusion in the time unit of the timecourse;
            required for `Route.IV_INFUSION` (finite and positive), not allowed
            otherwise
    """

    model_config = ConfigDict(frozen=True)

    amount: float = Field(ge=0)
    unit: str
    route: Route = Route.ORAL
    time: float = 0.0
    duration: float | None = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """Check the dose unit and the duration against the route."""
        check_dose_unit(self.unit)
        if self.route is Route.IV_INFUSION:
            if (
                self.duration is None
                or not np.isfinite(self.duration)
                or self.duration <= 0
            ):
                raise ValueError("An infusion needs a finite, positive 'duration'")
        elif self.duration is not None:
            raise ValueError("'duration' is only allowed for Route.IV_INFUSION")
        return self

    @property
    def quantity(self) -> Quantity:
        """The dose as a quantity."""
        return Q_(self.amount, self.unit)

    @property
    def per_bodyweight(self) -> bool:
        """Whether the dose is an amount per body weight."""
        return is_per_bodyweight(self.unit)


class DosingRegimen(BaseModel):
    """Repeated administration of the same dose at a fixed interval.

    Attributes:
        dose: the dose given at every administration; its `time` is the time of
            the first dose
        interval: dosing interval `tau` in the time unit of the timecourse
        n_doses: number of doses, `None` for an unspecified number (steady
            state analyses only need `tau`)
    """

    model_config = ConfigDict(frozen=True)

    dose: Dose
    interval: float = Field(gt=0)
    n_doses: int | None = Field(default=None, ge=1)

    def dose_times(self) -> np.ndarray:
        """Times of the administrations, `dose.time + k * interval`.

        Raises:
            ValueError: if `n_doses` is `None`.
        """
        if self.n_doses is None:
            raise ValueError("'n_doses' is required for the dose times")
        return self.dose.time + self.interval * np.arange(self.n_doses, dtype=float)

    def dosing(self) -> "Dosing":
        """The protocol of the regimen, `Dosing.regimen` of `dose`, `interval` and `n_doses`.

        Returns:
            The protocol.

        Raises:
            ValueError: if `n_doses` is `None`.
        """
        if self.n_doses is None:
            raise ValueError("'n_doses' is required for the dosing protocol")
        return Dosing.regimen(self.dose, self.interval, self.n_doses)


def _as_float_array(name: str, values: Any) -> np.ndarray:
    """Convert to a 1-D float64 array.

    Args:
        name: name of the field, used in the error message
        values: the values to convert

    Returns:
        The values as a one-dimensional `float64` array.

    Raises:
        ValueError: if `values` is not one dimensional.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"'{name}' must be one dimensional, not {array.ndim}-D")
    return array


def _values_equal(a: float | np.ndarray | None, b: float | np.ndarray | None) -> bool:
    """Compare two optional numbers or arrays, `NaN` equals `NaN`.

    Args:
        a: the first value, an array, a number or `None`
        b: the second value, an array, a number or `None`

    Returns:
        Whether both are `None` or hold the same values.
    """
    if a is None or b is None:
        return a is None and b is None
    return bool(np.array_equal(a, b, equal_nan=True))


class Dosing(BaseModel):
    """The dosing protocol of a timecourse: the doses given and the times they were given.

    A protocol has one route and one unit for every dose; `Dose` stays the
    single administration and `Dosing.single` wraps one into a protocol of one
    dose. The doses are stored sorted by time.

    Attributes:
        amounts: amount of every dose (non-negative), 1-D
        times: time of every dose in the time unit of the timecourse, 1-D,
            strictly increasing after validation
        durations: duration of every infusion in the time unit of the
            timecourse, `None` when no dose is an infusion; required with
            every value finite and positive for `Route.IV_INFUSION`, not
            allowed otherwise
        unit: unit of the amounts, see `pkpdutils.units.check_dose_unit`
        route: route of administration, shared by every dose of the protocol
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    amounts: np.ndarray
    times: np.ndarray
    durations: np.ndarray | None = None
    unit: str
    route: Route = Route.ORAL

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        """Convert `amounts`, `times` and `durations` to `float64` arrays.

        Runs before pydantic's field validation, the same way
        `Timecourse._normalize` does, so that list or array input is accepted
        for a field annotated `np.ndarray`. `durations` is also normalized to
        `None` here already when it is not given or every value is `NaN`
        (allowed for every route, not only when it does not apply); the
        `mode="after"` validator below still checks it against `route`.

        Args:
            data: the raw input to the model.

        Returns:
            `data` unchanged if it is not a `dict`, otherwise with `amounts`,
            `times` and `durations` converted.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "amounts" in data:
            data["amounts"] = _as_float_array("amounts", data["amounts"])
        if "times" in data:
            data["times"] = _as_float_array("times", data["times"])
        durations = data.get("durations")
        if durations is not None:
            durations = _as_float_array("durations", durations)
            if np.isnan(durations).all():
                durations = None
        data["durations"] = durations
        return data

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """Check the lengths, sort by time and check the doses against the route.

        Returns:
            The validated protocol.

        Raises:
            ValueError: if `amounts`, `times` or `durations` mismatch in
                length, if there is no dose, if an amount or a time is not
                finite, if an amount is negative, if the dose times contain
                duplicates, or if `durations` does not fit `route`.
        """
        amounts = self.amounts
        times = self.times
        if amounts.size != times.size:
            raise ValueError(
                f"'amounts' has length {amounts.size}, 'times' has length {times.size}"
            )
        if amounts.size == 0:
            raise ValueError("A dosing protocol needs at least one dose")
        # `NaN` passes every comparison below and would reach the dose
        # variables of a batch as a half padded dose
        for name, values in (("amounts", amounts), ("times", times)):
            if not np.isfinite(values).all():
                raise ValueError(f"'{name}' must be finite, got {values}")
        if (amounts < 0).any():
            raise ValueError("'amounts' must be non-negative")
        check_dose_unit(self.unit)

        durations = self.durations
        if durations is not None and durations.size != times.size:
            raise ValueError(
                f"'durations' has length {durations.size}, "
                f"'times' has length {times.size}"
            )

        order = np.argsort(times, kind="stable")
        if not np.array_equal(order, np.arange(times.size)):
            logger.warning("The dose times of the protocol were not sorted")
            times = times[order]
            amounts = amounts[order]
            if durations is not None:
                durations = durations[order]
        if np.unique(times).size != times.size:
            raise ValueError("Duplicate dose times")

        if self.route is Route.IV_INFUSION:
            # `NaN` is not caught by the comparison and would reach the
            # analyses as "no duration" on an infusion
            if (
                durations is None
                or not (np.isfinite(durations) & (durations > 0)).all()
            ):
                raise ValueError(
                    "An infusion needs a finite, positive 'duration' for every dose"
                )
        elif durations is not None:
            raise ValueError("'durations' is only allowed for Route.IV_INFUSION")

        object.__setattr__(self, "amounts", amounts)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "durations", durations)
        return self

    @classmethod
    def single(cls, dose: Dose) -> "Dosing":
        """Wrap one dose into a protocol of one dose.

        Args:
            dose: the dose.

        Returns:
            The protocol.
        """
        return cls(
            amounts=[dose.amount],
            times=[dose.time],
            durations=None if dose.duration is None else [dose.duration],
            unit=dose.unit,
            route=dose.route,
        )

    @classmethod
    def from_doses(cls, doses: Sequence[Dose]) -> "Dosing":
        """Build a protocol from individual doses.

        Args:
            doses: the doses, at least one, all with the same `unit` and `route`.

        Returns:
            The protocol.

        Raises:
            ValueError: if `doses` is empty, or the doses have different
                `unit` or `route`.
        """
        if not doses:
            raise ValueError("'doses' needs at least one dose")
        routes = {dose.route for dose in doses}
        if len(routes) != 1:
            raise ValueError(
                f"All doses need the same route, found {sorted(r.value for r in routes)}"
            )
        units = {dose.unit for dose in doses}
        if len(units) != 1:
            raise ValueError(f"All doses need the same unit, found {sorted(units)}")
        durations = [dose.duration for dose in doses]
        return cls(
            amounts=[dose.amount for dose in doses],
            times=[dose.time for dose in doses],
            durations=None if all(d is None for d in durations) else durations,
            unit=units.pop(),
            route=routes.pop(),
        )

    @classmethod
    def regimen(cls, dose: Dose, interval: float, n_doses: int) -> "Dosing":
        """Build a regular protocol, `dose` repeated every `interval`.

        Args:
            dose: the dose given at every administration; its `time` is the
                time of the first dose
            interval: dosing interval, must be positive
            n_doses: number of doses, must be at least 1

        Returns:
            The protocol with times `dose.time + k * interval`.

        Raises:
            ValueError: if `interval` is not positive or `n_doses` is less
                than 1.
        """
        if interval <= 0:
            raise ValueError("'interval' must be positive")
        if n_doses < 1:
            raise ValueError("'n_doses' must be at least 1")
        times = dose.time + interval * np.arange(n_doses, dtype=float)
        amounts = np.full(n_doses, dose.amount)
        durations = None if dose.duration is None else np.full(n_doses, dose.duration)
        return cls(
            amounts=amounts,
            times=times,
            durations=durations,
            unit=dose.unit,
            route=dose.route,
        )

    def __eq__(self, other: object) -> bool:
        """Compare two protocols field by field, `NaN` equals `NaN` in `durations`.

        Args:
            other: the object to compare with.

        Returns:
            Whether `other` is a protocol with the same fields;
            `NotImplemented` for any other type, so that python falls back to
            the identity comparison.
        """
        if not isinstance(other, Dosing):
            return NotImplemented
        return (
            bool(np.array_equal(self.amounts, other.amounts))
            and bool(np.array_equal(self.times, other.times))
            and _values_equal(self.durations, other.durations)
            and self.unit == other.unit
            and self.route == other.route
        )

    def __len__(self) -> int:
        """Number of doses, `n_doses`."""
        return self.n_doses

    @property
    def n_doses(self) -> int:
        """Number of doses."""
        return int(self.times.size)

    @property
    def doses(self) -> list[Dose]:
        """The doses of the protocol as individual `Dose` objects."""
        return [
            Dose(
                amount=float(self.amounts[i]),
                unit=self.unit,
                route=self.route,
                time=float(self.times[i]),
                duration=None if self.durations is None else float(self.durations[i]),
            )
            for i in range(self.n_doses)
        ]

    @property
    def first(self) -> Dose:
        """The first dose of the protocol."""
        return self.doses[0]

    @property
    def last(self) -> Dose:
        """The last dose of the protocol."""
        return self.doses[-1]

    @property
    def intervals(self) -> np.ndarray:
        """Time between consecutive doses, `np.diff(times)`."""
        return np.diff(self.times)

    @property
    def tau(self) -> float | None:
        """The common dosing interval, `None` without at least two doses or an irregular protocol."""
        if self.n_doses < 2:
            return None
        intervals = self.intervals
        if not np.allclose(intervals, intervals[0], rtol=1e-9, atol=0):
            return None
        return float(intervals[0])

    @property
    def is_regular(self) -> bool:
        """Whether the protocol has a common dosing interval, `tau is not None`."""
        return self.tau is not None

    @property
    def total_amount(self) -> float:
        """Sum of the dose amounts."""
        return float(self.amounts.sum())

    @property
    def quantity(self) -> Quantity:
        """The total dose amount as a quantity."""
        return Q_(self.total_amount, self.unit)

    @property
    def per_bodyweight(self) -> bool:
        """Whether the doses are an amount per body weight."""
        return is_per_bodyweight(self.unit)

    def shifted(self, offset: float) -> "Dosing":
        """Copy with every dose time shifted by `-offset`.

        Args:
            offset: the offset to subtract from every dose time.

        Returns:
            The shifted protocol.
        """
        return self.model_copy(update={"times": self.times - offset})


class Timecourse(BaseModel):
    """One curve of values over time with units, uncertainty, dose and metadata.

    Concentration timecourses of a substance and effect timecourses of a
    pharmacodynamic response use the same class; `value` is the generic name.
    A group timecourse (mean of several subjects) carries the standard
    deviation `sd` or the standard error `se` and the number of subjects `n`;
    an individual timecourse carries none of them.

    Validation converts the arrays to `float64`, sorts them by time, derives
    `se` from `sd` and `n` (or `sd` from `se` and `n`) and checks the units.

    Attributes:
        time: sampling times, strictly increasing after validation
        value: values at the sampling times, `NaN` for missing values
        time_unit: unit of `time`, e.g. `"hr"`
        unit: unit of `value`, e.g. `"ng/ml"`
        sd: standard deviation per time point (group data)
        se: standard error per time point (group data)
        n: number of subjects, one number or one per time point
        dosing: the dosing protocol, `None` without dose information; the
            constructor also accepts a single `dose: Dose` keyword, wrapped
            into a protocol of one dose
        substance: name of the substance or of the effect
        label: label of the curve, e.g. the group or the individual
        tissue: tissue or matrix the values were measured in, e.g. `"plasma"`
        lloq: lower limit of quantification of the assay behind the values, in
            their unit; the analysis reads it when `NCAOptions.lloq` names no
            limit of its own (`pkpdutils.nca`), so that a study with two assays
            or two analytes carries a limit per curve
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    time: np.ndarray
    value: np.ndarray
    time_unit: str
    unit: str
    sd: np.ndarray | None = None
    se: np.ndarray | None = None
    n: float | np.ndarray | None = None
    dosing: Dosing | None = None
    substance: str = "substance"
    label: str | None = None
    tissue: str | None = None
    lloq: float | None = Field(default=None, gt=0.0)

    def __init__(
        self,
        *,
        dose: Dose | Dosing | None = None,
        dosing: Dosing | None = None,
        **data: Any,
    ) -> None:
        """Construct a timecourse, `dose` and `dosing` handled by `_dose_to_dosing`.

        `dose` is not a field of the model (kept only for backwards
        compatibility with `Dose(...)`); declaring it here, rather than
        relying on the `model_validator(mode="before")` alone, keeps it a
        recognized keyword argument for static type checkers.

        Args:
            dose: a single dose, converted into a protocol of one dose; not
                allowed together with `dosing`
            dosing: the dosing protocol
            **data: the remaining fields of `Timecourse`.
        """
        super().__init__(dose=dose, dosing=dosing, **data)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        """Convert arrays, validate lengths, derive `sd`/`se` and sort by time.

        Args:
            data: the raw input to the model.

        Returns:
            `data` unchanged if it is not a `dict`, otherwise the normalized
            fields.

        Raises:
            ValueError: if `time` and `value` (or `sd`/`se`/`n`) mismatch in
                length, if `time` has fewer than 2 points, contains `NaN` or
                duplicate values.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)

        time = _as_float_array("time", data.get("time"))
        value = _as_float_array("value", data.get("value"))
        if time.size < 2:
            raise ValueError("A timecourse needs at least 2 time points")
        if value.size != time.size:
            raise ValueError(
                f"'value' has length {value.size}, 'time' has length {time.size}"
            )
        if np.isnan(time).any():
            raise ValueError("'time' contains NaN")
        if np.unique(time).size != time.size:
            raise ValueError("'time' contains duplicate values")

        arrays: dict[str, np.ndarray | None] = {}
        for key in ("sd", "se"):
            raw = data.get(key)
            if raw is None:
                arrays[key] = None
                continue
            array = _as_float_array(key, raw)
            if array.size != time.size:
                raise ValueError(
                    f"'{key}' has length {array.size}, 'time' has length {time.size}"
                )
            arrays[key] = array

        n_raw = data.get("n")
        n: float | np.ndarray | None
        if n_raw is None:
            n = None
        elif np.ndim(n_raw) == 0:
            n = float(n_raw)
        else:
            n = _as_float_array("n", n_raw)
            if n.size != time.size:
                raise ValueError(
                    f"'n' has length {n.size}, 'time' has length {time.size}"
                )

        # derive the missing one of sd and se
        if n is not None:
            sqrt_n = np.sqrt(n)
            if arrays["se"] is None and arrays["sd"] is not None:
                arrays["se"] = arrays["sd"] / sqrt_n
            elif arrays["sd"] is None and arrays["se"] is not None:
                arrays["sd"] = arrays["se"] * sqrt_n

        # sort by time
        order = np.argsort(time)
        if not np.array_equal(order, np.arange(time.size)):
            logger.warning("Unsorted time points are sorted by time")
            time = time[order]
            value = value[order]
            for key, array in arrays.items():
                if array is not None:
                    arrays[key] = array[order]
            if isinstance(n, np.ndarray):
                n = n[order]

        data.update({"time": time, "value": value, "n": n, **arrays})
        return data

    @model_validator(mode="before")
    @classmethod
    def _dose_to_dosing(cls, data: Any) -> Any:
        """Move a `dose` keyword into `dosing`, converting a `Dose` to a `Dosing`.

        Declared after `_normalize` so that it runs first (pydantic runs
        several `mode="before"` validators in reverse declaration order): the
        "not both" check below must fire before `_normalize` rejects the
        arrays for an unrelated reason.

        Args:
            data: the raw input to the model.

        Returns:
            `data` unchanged if it is not a `dict` or has no `dose` key,
            otherwise `data` with `dose` removed and `dosing` set.

        Raises:
            ValueError: if both `dose` and `dosing` are given.
        """
        if not isinstance(data, dict) or "dose" not in data:
            return data
        data = dict(data)
        dose = data.pop("dose")
        if dose is None:
            return data
        if data.get("dosing") is not None:
            raise ValueError("Give 'dose' or 'dosing', not both")
        data["dosing"] = dose if isinstance(dose, Dosing) else Dosing.single(dose)
        return data

    @model_validator(mode="after")
    def _check_units(self) -> Self:
        """Check that `time_unit` and `unit` are valid units.

        Returns:
            The validated timecourse.

        Raises:
            ValueError: if `time_unit` or `unit` is not a valid unit.
        """
        parse_unit(self.time_unit)
        parse_unit(self.unit)
        return self

    def __eq__(self, other: object) -> bool:
        """Compare two timecourses field by field, `NaN` equals `NaN`.

        The generated comparison of pydantic compares the numpy fields with
        `==`, which raises for arrays; the array fields are compared with
        `numpy.array_equal` instead.

        Args:
            other: the object to compare with.

        Returns:
            Whether `other` is a timecourse with the same fields;
            `NotImplemented` for any other type, so that python falls back to
            the identity comparison.
        """
        if not isinstance(other, Timecourse):
            return NotImplemented
        arrays = ("time", "value", "sd", "se", "n")
        for name in arrays:
            if not _values_equal(getattr(self, name), getattr(other, name)):
                return False
        return all(
            getattr(self, name) == getattr(other, name)
            for name in type(self).model_fields
            if name not in arrays
        )

    def __hash__(self) -> int:
        """Hash of the immutable fields, equal for equal timecourses.

        The arrays are not hashable, only `size` enters the hash; equal
        timecourses hash equal, unequal ones may collide.
        """
        return hash(
            (
                self.time_unit,
                self.unit,
                self.substance,
                self.label,
                self.tissue,
                self.dose,
                self.size,
            )
        )

    @property
    def size(self) -> int:
        """Number of time points."""
        return int(self.time.size)

    @property
    def dose(self) -> Dose | None:
        """First dose of the protocol, `None` without `dosing`.

        Read-only: `model_copy(update={"dose": ...})` is a silent no-op (a
        property is not a field), use
        `model_copy(update={"dosing": Dosing.single(dose)})` instead.
        """
        return None if self.dosing is None else self.dosing.first

    @property
    def time_q(self) -> Quantity:
        """The times as a quantity."""
        return Q_(self.time, self.time_unit)

    @property
    def value_q(self) -> Quantity:
        """The values as a quantity."""
        return Q_(self.value, self.unit)

    @property
    def sd_q(self) -> Quantity | None:
        """The standard deviations as a quantity, `None` without `sd`."""
        return None if self.sd is None else Q_(self.sd, self.unit)

    @property
    def se_q(self) -> Quantity | None:
        """The standard errors as a quantity, `None` without `se`."""
        return None if self.se is None else Q_(self.se, self.unit)

    def relative_to_dose(
        self, which: Literal["first", "last"] = "first"
    ) -> "Timecourse":
        """Copy with the time shifted so that a dose of the protocol is given at time 0.

        Returns the timecourse itself when it has no protocol or the chosen
        dose is already at time 0.

        Args:
            which: `"first"` shifts by the time of the first dose, `"last"`
                by the time of the last dose.

        Returns:
            The shifted timecourse, or `self` when there is nothing to shift.
        """
        if self.dosing is None:
            return self
        shift = self.dosing.first.time if which == "first" else self.dosing.last.time
        if shift == 0.0:
            return self
        return self.model_copy(
            update={
                "time": self.time - shift,
                "dosing": self.dosing.shifted(shift),
            }
        )

    def to_batch(self, dim: str = "individual", label: Any = None) -> "Timecourses":
        """The curve as a batch of one sample, the counterpart of `Timecourses.sel`.

        Args:
            dim: name of the sample dimension of the batch
            label: coordinate label of the single sample, the `label` of the
                curve (or 0 when it has none) by default

        Returns:
            The batch with one sample.
        """
        return Timecourses.from_timecourses(
            [self], dim=dim, labels=None if label is None else [label]
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the curve to a data frame.

        Returns:
            A data frame with the columns `time`, `value` and, when present,
            `sd`, `se`, `n`.
        """
        columns: dict[str, Any] = {"time": self.time, "value": self.value}
        if self.sd is not None:
            columns["sd"] = self.sd
        if self.se is not None:
            columns["se"] = self.se
        if self.n is not None:
            columns["n"] = np.broadcast_to(self.n, self.time.shape)
        return pd.DataFrame(columns)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        time_unit: str,
        unit: str,
        time: str = "time",
        value: str = "value",
        sd: str | None = None,
        se: str | None = None,
        n: str | None = None,
        **fields: Any,
    ) -> "Timecourse":
        """Create a timecourse from the columns of a data frame.

        Args:
            df: the data frame, one row per time point
            time_unit: unit of the time column
            unit: unit of the value column
            time: name of the time column
            value: name of the value column
            sd: name of the standard deviation column, `None` for none
            se: name of the standard error column, `None` for none
            n: name of the column with the number of subjects, `None` for none
            **fields: the remaining fields of `Timecourse` (`dose`, `substance`, `label`, `tissue`)

        Returns:
            The timecourse.
        """
        data: dict[str, Any] = {
            "time": df[time].to_numpy(),
            "value": df[value].to_numpy(),
            "time_unit": time_unit,
            "unit": unit,
            **fields,
        }
        if sd is not None:
            data["sd"] = df[sd].to_numpy()
        if se is not None:
            data["se"] = df[se].to_numpy()
        if n is not None:
            data["n"] = df[n].to_numpy()
        return cls(**data)


#: name of the time dimension of a `Timecourses` dataset
TIME_DIM = "time"

#: name of the per sample time variable of a `Timecourses` dataset with ragged grids
TIMES_VAR = "times"

#: name of the per sample limit of quantification of a `Timecourses` dataset,
#: a coordinate along the sample dimensions
LLOQ_VAR = "lloq"

#: name of the per sample substance of a `Timecourses` dataset, a coordinate
#: along a sample dimension; a batch whose samples share one substance carries
#: it in `attrs["substance"]` instead (`Timecourses.substance`)
SUBSTANCE_VAR = "substance"

#: name of the per sample route of a `Timecourses` dataset, a coordinate along
#: a sample dimension; a batch whose samples share one route carries it in
#: `attrs["route"]` instead (`Timecourses.route`)
ROUTE_VAR = "route"

#: name of the dose dimension of the dose variables of a `Timecourses` dataset;
#: not `"dose"`, which stays free as a sample dimension (a dose group of a dose
#: proportionality study, the dose axis of a simulation scan)
DOSE_DIM = "dose_index"

#: name of the nominal (scheduled) time of every sample and time point of a
#: `Timecourses` dataset, optional; the actual times stay the times of the
#: batch and the analyses read them, the nominal grid is what a mean curve over
#: the subjects of a study is taken on (ICH M13A 2.2.2.1,
#: `pkpdutils.plot.plot_study_curves`)
NOMINAL_TIMES_VAR = "nominal_time"


def _sample_codes(
    df: pd.DataFrame, sample: Sequence[str]
) -> tuple[np.ndarray, list[Any]]:
    """Number the rows of a long frame by sample, in the order of their first appearance.

    Args:
        df: the long frame.
        sample: the columns which identify a sample.

    Returns:
        The sample of every row as an index into the labels, and the labels
        (a tuple per sample for several sample columns).
    """
    key = (
        df[sample[0]]
        if len(sample) == 1
        else pd.MultiIndex.from_frame(df[list(sample)])
    )
    codes, uniques = pd.factorize(key, use_na_sentinel=False)
    return codes, list(uniques)


def _rows_by_sample(
    codes: np.ndarray, times: np.ndarray, n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Order the rows of a long frame by sample and time, and place them in a padded row.

    The stable sort by time followed by the stable sort by sample sorts every
    sample by time and leaves rows with equal times in the order of the frame,
    which is what sorting every sample on its own does.

    Args:
        codes: the sample of every row.
        times: the time of every row.
        n_samples: number of samples.

    Returns:
        The order of the rows, the row and the column of every ordered row in
        the padded `(n_samples, n_time)` arrays, and the number of rows per
        sample.
    """
    by_time = np.argsort(times, kind="stable")
    order = by_time[np.argsort(codes[by_time], kind="stable")]
    counts = np.bincount(codes, minlength=n_samples)
    starts = np.cumsum(counts) - counts
    column = np.arange(order.size) - np.repeat(starts, counts)
    return order, codes[order], column, counts


def _numeric_column(
    df: pd.DataFrame, name: str, *, codes: np.ndarray, labels: Sequence[Any]
) -> pd.Series:
    """One column of a long frame as a float series, naming a value which is not a number.

    A value which is missing becomes `NaN`, as it does for a single curve; a
    value which is not missing and not a number is an error, so that no cell of
    the frame is silently dropped on the way into the batch.

    Args:
        df: the long frame.
        name: name of the column.

    Keyword Args:
        codes: the sample of every row.
        labels: the samples, in the order of the batch.

    Returns:
        The column as `float64`, indexed like `df`.

    Raises:
        ValueError: for the first value which is not a number, named with the
            sample and the column, and for a column which cannot be read as
            numbers at all.
    """
    column = df[name]
    try:
        coerced = pd.to_numeric(column, errors="coerce")
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"The column '{name}' of dtype '{column.dtype}' does not hold numbers"
        ) from err
    bad = coerced.isna().to_numpy() & column.notna().to_numpy()
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"sample {labels[codes[i]]}: the column '{name}' has the "
            f"non-numeric value {column.iloc[i]!r}"
        )
    return coerced.astype(np.float64)


def _constant_per_sample(
    values: np.ndarray, *, name: str, codes: np.ndarray, labels: Sequence[Any]
) -> np.ndarray:
    """One value per sample out of a column which has to be constant within a sample.

    Args:
        values: the column of the long frame as floats, one entry per row.

    Keyword Args:
        name: name of the column, for the message.
        codes: the sample of every row.
        labels: the samples, in the order of the batch.

    Returns:
        The value of every sample, `NaN` for a sample whose rows are all
        missing.

    Raises:
        ValueError: if the column holds two different values within a sample,
            naming the sample.
    """
    # one pass over the rows, grouped by sample, rather than one scan of the
    # column per sample: a study with a few thousand subjects goes through here
    grouped = pd.Series(np.asarray(values, dtype=np.float64)).groupby(codes, sort=True)
    varying = grouped.nunique(dropna=True) > 1
    if bool(varying.any()):
        index = int(varying.index[int(varying.to_numpy().argmax())])
        rows = values[codes == index]
        given = np.unique(rows[np.isfinite(rows)])
        raise ValueError(
            f"sample {labels[index]}: the column '{name}' is not constant, "
            f"found {given.tolist()}"
        )
    # `first` skips the missing values, a sample without any value gives `NaN`
    return (
        grouped.first()
        .reindex(range(len(labels)))
        .to_numpy(dtype=np.float64, na_value=np.nan)
    )


def _frame_doses(
    df: pd.DataFrame,
    *,
    codes: np.ndarray,
    labels: Sequence[Any],
    dose_amount: str,
    dose_unit: str | None,
    dose_time: str | None,
    route: Route | None,
) -> dict[str, Any]:
    """The padded dose arrays of the samples of a long data frame.

    Without a `dose_time` column a sample carries one dose at time 0 and the
    amount must be constant over its rows. With a `dose_time` column every
    distinct `(dose_time, dose_amount)` pair of a sample is one dose of its
    protocol; rows whose dose columns are `NaN` are observations only. The
    checks are the ones `Dosing` makes on a single protocol, applied to every
    sample at once.

    Args:
        df: the long frame.

    Keyword Args:
        codes: the sample of every row.
        labels: the samples, in the order of the batch.
        dose_amount: name of the dose column.
        dose_unit: unit of the doses.
        dose_time: name of the dose time column, 0 by default.
        route: route of the doses.

    Returns:
        The `dose` mapping of `Timecourses.from_arrays`.

    Raises:
        ValueError: if `dose_unit` or `route` is missing, if a dose column
            holds a value which is not a number, if a sample has no dose or the
            dose is not constant per sample (without `dose_time`), if a dose
            time of a sample carries several amounts, or if an amount is not
            finite or negative.
    """
    if dose_unit is None or route is None:
        raise ValueError("'dose_unit' and 'route' are required with 'dose_amount'")
    n_samples = len(labels)
    # a value which is not a number is an error in both branches: it would
    # otherwise become a missing dose and drop the record from the protocol
    amount_column = _numeric_column(df, dose_amount, codes=codes, labels=labels)
    if dose_time is None:
        column = amount_column
        grouped = column.groupby(codes, sort=True)
        distinct = grouped.nunique(dropna=True).reindex(range(n_samples)).to_numpy()
        if (distinct != 1).any():
            i = int(np.argmax(distinct != 1))
            found = column[codes == i].dropna().unique()
            raise ValueError(
                f"sample {labels[i]}: the dose must be constant per sample, "
                f"found {found}"
            )
        amounts = (
            grouped.first()
            .reindex(range(n_samples))
            .to_numpy(dtype=np.float64)
            .reshape(n_samples, 1)
        )
        _check_dose_amounts(amounts, labels)
        return {"amount": amounts, "unit": dose_unit, "time": np.zeros((n_samples, 1))}

    pairs = pd.DataFrame(
        {
            "sample": codes,
            "time": _numeric_column(
                df, dose_time, codes=codes, labels=labels
            ).to_numpy(),
            "amount": amount_column.to_numpy(),
        }
    ).dropna()
    pairs = pairs.drop_duplicates().sort_values(
        ["sample", "time", "amount"], kind="stable"
    )
    counts = np.bincount(pairs["sample"].to_numpy(), minlength=n_samples)
    if (counts == 0).any():
        i = int(np.argmax(counts == 0))
        raise ValueError(
            f"sample {labels[i]}: the sample has no dose in '{dose_amount}'"
        )
    n_dose = int(counts.max())
    column = np.arange(len(pairs)) - np.repeat(np.cumsum(counts) - counts, counts)
    row = pairs["sample"].to_numpy()
    times = np.full((n_samples, n_dose), np.nan)
    amounts = np.full((n_samples, n_dose), np.nan)
    times[row, column] = pairs["time"].to_numpy()
    amounts[row, column] = pairs["amount"].to_numpy()
    duplicate = np.diff(times, axis=1) == 0
    if duplicate.any():
        i = int(np.argmax(duplicate.any(axis=1)))
        valid = slice(0, counts[i])
        raise ValueError(
            f"sample {labels[i]}: the dose must be constant per dose time, found "
            f"{amounts[i, valid]} at {times[i, valid]}"
        )
    _check_dose_amounts(amounts, labels)
    return {"amount": amounts, "unit": dose_unit, "time": times}


def _check_sample_times(
    times: np.ndarray, row: np.ndarray, counts: np.ndarray, labels: Sequence[Any]
) -> None:
    """Check the sampling times of every sample of a long data frame.

    The checks `Timecourse` makes on a single curve, applied to every sample at
    once: at least two time points, no `NaN` and no duplicates. They are the
    invariant of a batch built from the padded arrays, which no longer goes
    through one `Timecourse` per sample.

    Args:
        times: the times of every row, ordered by sample and by time.
        row: the sample of every ordered row.
        counts: number of rows per sample.
        labels: the samples, in the order of the batch.

    Raises:
        ValueError: for the first sample with fewer than two time points, a
            `NaN` time or duplicate times, named with the sample.
    """
    n_samples = len(labels)
    missing = np.isnan(times)
    duplicate = np.zeros(times.size, dtype=bool)
    if times.size > 1:
        duplicate[1:] = (np.diff(times) == 0) & (row[1:] == row[:-1])
    short = counts < 2
    with_nan = np.bincount(row[missing], minlength=n_samples) > 0
    with_duplicate = np.bincount(row[duplicate], minlength=n_samples) > 0
    bad = short | with_nan | with_duplicate
    if not bad.any():
        return
    i = int(np.argmax(bad))
    if short[i]:
        raise ValueError(
            f"sample {labels[i]}: a timecourse needs at least 2 time points"
        )
    if with_nan[i]:
        raise ValueError(f"sample {labels[i]}: 'time' contains NaN")
    raise ValueError(f"sample {labels[i]}: 'time' contains duplicate values")


def _counts_per_sample(subjects: np.ndarray) -> np.ndarray:
    """The largest count of every sample, `NaN` for a sample without one.

    Args:
        subjects: the counts `(n_samples, n_time)`, `NaN` where a sample has
            no count at a time point (the padding of a shorter grid).

    Returns:
        One count per sample.
    """
    finite = np.isfinite(subjects)
    high = np.where(finite, subjects, -np.inf).max(axis=1)
    return np.where(finite.any(axis=1), high, np.nan)


def _counts_vary_over_time(subjects: np.ndarray) -> bool:
    """Whether the count of a sample changes from one time point to another.

    Args:
        subjects: the counts `(n_samples, n_time)`, `NaN` where a sample has
            no count at a time point.

    Returns:
        `True` when a sample carries two different counts, so that the batch
        has to keep them per time point rather than one number per sample.
    """
    finite = np.isfinite(subjects)
    low = np.where(finite, subjects, np.inf).min(axis=1)
    return bool(np.any(finite.any(axis=1) & (low != _counts_per_sample(subjects))))


def _count_layout(
    rest: tuple[str, ...], count: np.ndarray
) -> tuple[tuple[str, ...], np.ndarray, dict[str, str]]:
    """The `n` variable of a group curve in the layout of the batch constructors.

    A count which is the same at every time point of a group is stored per
    sample, as `from_timecourses` stores it, so a group curve of a batch on a
    shared grid round trips through the constructors unchanged; a count which
    varies along the grid (a ragged batch) is stored over `time`.

    Args:
        rest: the remaining sample dimensions of the group curve.
        count: the counts, shape `(*rest, n_time)`.

    Returns:
        The `(dims, data, attrs)` triple of the variable.
    """
    attrs = {"units": "dimensionless"}
    first = count[..., :1]
    if count.shape[-1] and np.all(count == first):
        return (rest, first[..., 0], attrs)
    return ((*rest, TIME_DIM), count, attrs)


def _batch_counts(subjects: np.ndarray) -> np.ndarray:
    """The `n` of a batch: one count per sample, or the counts per time point.

    A batch keeps one number per sample, the usual group data, and the whole
    `(n_samples, n_time)` block when a sample counts its time points
    separately, as the group curve of `Timecourses.mean` on a ragged batch
    does; `Timecourses.n_subjects` reads the number of subjects back either
    way.

    Args:
        subjects: the counts `(n_samples, n_time)`, `NaN` where a sample has
            no count at a time point.

    Returns:
        The counts in the layout the batch keeps.
    """
    return (
        subjects if _counts_vary_over_time(subjects) else _counts_per_sample(subjects)
    )


def _check_dose_amounts(amounts: np.ndarray, labels: Sequence[Any]) -> None:
    """Check the dose amounts of the samples of a batch, as `Dosing` does per protocol.

    Args:
        amounts: the padded dose amounts `(n_samples, n_dose)`.
        labels: the samples, in the order of the batch.

    Raises:
        ValueError: if an amount which is not padding is not finite or is
            negative, named with the sample.
    """
    given = ~np.isnan(amounts)
    with np.errstate(invalid="ignore"):
        bad_finite = given & ~np.isfinite(amounts)
        negative = given & (amounts < 0)
    for mask, message in (
        (bad_finite, "must be finite"),
        (negative, "must be non-negative"),
    ):
        if mask.any():
            i = int(np.argwhere(mask.any(axis=1))[0][0])
            raise ValueError(
                f"sample {labels[i]}: 'amounts' {message}, got {amounts[i]}"
            )


def pad_rows(arrays: Sequence[np.ndarray], n_columns: int) -> np.ndarray:
    """Stack 1-D arrays of different lengths into `(len(arrays), n_columns)`, padded with `NaN`.

    The padding layout of a batch: the values of a sample are the leading
    columns of its row, the trailing columns are `NaN`.

    Args:
        arrays: one array per sample, none longer than `n_columns`.
        n_columns: number of columns, at least the length of the longest array.

    Returns:
        The padded array.
    """
    out = np.full((len(arrays), n_columns), np.nan)
    for i, array in enumerate(arrays):
        out[i, : array.size] = array
    return out


def pad_protocols(
    protocols: Sequence[Dosing | None], n_dose: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack dosing protocols into `(len(protocols), n_dose)` arrays padded with `NaN`.

    Args:
        protocols: one protocol per sample, `None` for a sample without doses.
        n_dose: number of columns, at least the longest protocol.

    Returns:
        The amounts, the times and the durations; the doses of a sample are at
        the front of its row, the remaining columns are `NaN`.
    """
    amounts = np.full((len(protocols), n_dose), np.nan)
    times = np.full((len(protocols), n_dose), np.nan)
    durations = np.full((len(protocols), n_dose), np.nan)
    for i, protocol in enumerate(protocols):
        if protocol is None:
            continue
        k = protocol.n_doses
        amounts[i, :k] = protocol.amounts
        times[i, :k] = protocol.times
        if protocol.durations is not None:
            durations[i, :k] = protocol.durations
    return amounts, times, durations


def dose_mapping(
    protocols: Sequence[Dosing | None],
    *,
    allow_mixed_routes: bool = False,
) -> tuple[dict[str, Any] | None, Route | None]:
    """The `dose` mapping and the route of a batch built from per sample protocols.

    The protocols of a batch share one dose unit; they are padded to the
    longest one (`pad_protocols`), a sample without a protocol gets a row of
    `NaN`. The mapping always carries a `duration` entry, `NaN` where the
    route is not an infusion: `dose_duration` is a variable of every batch with
    doses, so that the readers of the dose variables need no case distinction.

    The protocols share one route unless `allow_mixed_routes` says otherwise;
    the route of the first protocol is returned then and the caller carries the
    route of every sample as the coordinate `route` along a sample dimension
    (`Timecourses.routes`).

    Args:
        protocols: one protocol per sample, `None` for a sample without doses.

    Keyword Args:
        allow_mixed_routes: whether the protocols may have been given by
            different routes.

    Returns:
        The mapping for `Timecourses.from_arrays` and the route, both `None`
        when no sample carries a protocol.

    Raises:
        ValueError: if the protocols do not share one dose unit, or one route
            without `allow_mixed_routes`.
    """
    given = [protocol for protocol in protocols if protocol is not None]
    if not given:
        return None, None
    routes = {protocol.route for protocol in given}
    if len(routes) != 1 and not allow_mixed_routes:
        raise ValueError(
            "A batch has one route, found "
            f"{sorted(r.value for r in routes)}; build separate batches, one "
            "per route"
        )
    units = {protocol.unit for protocol in given}
    if len(units) != 1:
        raise ValueError(f"All doses need the same unit, found {sorted(units)}")
    amounts, times, durations = pad_protocols(
        protocols, max(protocol.n_doses for protocol in given)
    )
    mapping = {
        "amount": amounts,
        "unit": units.pop(),
        "time": times,
        "duration": durations,
    }
    return mapping, given[0].route


def _batch_dose_arrays(
    dose: "Dose | Dosing | Mapping[str, Any]",
    *,
    route: Route | str | None,
    sample_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, Route]:
    """The `(*sample_shape, n_dose)` dose arrays of a batch, its dose unit and its route.

    A `Dose` or a `Dosing` gives the same protocol to every sample. A mapping
    carries the arrays: `amount` (and `time`, `duration`) either of shape
    `sample_shape`, one dose per sample, or of shape `(*sample_shape, n_dose)`,
    one protocol per sample padded with `NaN`.

    The returned arrays hold the invariant of the dose variables of a batch:
    the doses of a sample are the leading columns of its row, sorted by time,
    and the trailing columns are `NaN` padding. The rows of a mapping are
    sorted here (`_sort_protocol_rows`), so a caller may give them in any
    order.

    Args:
        dose: the dose, the protocol or the mapping.
        route: route of the doses, a `Route` or a string it coerces; required
            for a mapping, checked against the route of a `Dose` or `Dosing`.
        sample_shape: shape of the sample dimensions of the batch.

    Returns:
        The amounts, the times, the durations, the dose unit and the route.

    Raises:
        ValueError: if `route` is missing or contradicts the route of the
            protocol, if the dose times are missing for a mapping of
            protocols, if the shapes do not fit `sample_shape`, if a dose of a
            mapping has an amount without a time or a time without an amount,
            or if a row has duplicate dose times.
    """
    route = None if route is None else Route(route)
    if isinstance(dose, Dose):
        dose = Dosing.single(dose)
    if isinstance(dose, Dosing):
        if route is not None and route is not dose.route:
            raise ValueError(
                f"'route' is '{route.value}' but the dose is given '{dose.route.value}'"
            )
        shape = (*sample_shape, dose.n_doses)
        durations = (
            np.full(dose.n_doses, np.nan) if dose.durations is None else dose.durations
        )
        return (
            np.broadcast_to(dose.amounts, shape).copy(),
            np.broadcast_to(dose.times, shape).copy(),
            np.broadcast_to(durations, shape).copy(),
            dose.unit,
            dose.route,
        )

    if route is None:
        raise ValueError("'route' is required when 'dose' is given as arrays")
    dose_unit = str(dose["unit"])
    check_dose_unit(dose_unit)
    amount = np.asarray(dose["amount"], dtype=np.float64)
    per_sample = amount.shape == sample_shape
    if not per_sample and (
        amount.ndim != len(sample_shape) + 1 or amount.shape[:-1] != sample_shape
    ):
        # a scalar or anything else broadcastable: one dose per sample
        amount = np.broadcast_to(amount, sample_shape).copy()
        per_sample = True
    if per_sample:
        shape = (*sample_shape, 1)
        time = np.asarray(dose.get("time", 0.0), dtype=np.float64)
        duration = np.asarray(dose.get("duration", np.nan), dtype=np.float64)
        return (
            amount.reshape(shape),
            np.broadcast_to(time, sample_shape).copy().reshape(shape),
            np.broadcast_to(duration, sample_shape).copy().reshape(shape),
            dose_unit,
            route,
        )

    shape = amount.shape
    if dose.get("time") is None:
        raise ValueError(
            "'time' is required when 'amount' carries the dose dimension "
            f"(shape {shape} for the sample shape {sample_shape})"
        )
    time = np.broadcast_to(np.asarray(dose["time"], dtype=np.float64), shape).copy()
    duration = np.broadcast_to(
        np.asarray(dose.get("duration", np.nan), dtype=np.float64), shape
    ).copy()
    amount, time, duration = _sort_protocol_rows(amount, time, duration)
    return amount, time, duration, dose_unit, route


def _sample_position(row: int, sample_shape: tuple[int, ...]) -> Any:
    """The sample index of a flattened protocol row, for an error message.

    Args:
        row: index of the row in the flattened `(n_rows, n_dose)` arrays.
        sample_shape: shape of the sample dimensions of the batch.

    Returns:
        The index along the sample dimensions, `row` itself for a batch
        without sample dimensions or with a single one.
    """
    if len(sample_shape) < 2:
        return row
    return tuple(int(i) for i in np.unravel_index(row, sample_shape))


def _sort_protocol_rows(
    amounts: np.ndarray, times: np.ndarray, durations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort every protocol row by dose time and move the `NaN` padding to the back.

    Establishes the invariant every reader of the dose variables relies on
    (`Timecourses.n_doses`, `first_dose_time`, `pkpdutils.nca.nca.reference_dose`):
    the doses of a sample are the leading columns of its row, sorted by time,
    and the trailing columns are `NaN`.

    Args:
        amounts: the dose amounts of shape `(*sample_shape, n_dose)`.
        times: the dose times with the shape of `amounts`.
        durations: the infusion durations with the shape of `amounts`.

    Returns:
        The three arrays with every row sorted by time.

    Raises:
        ValueError: if an entry has an amount without a time or a time without
            an amount (a half padded dose), or if a row has duplicate dose
            times.
    """
    shape = amounts.shape
    sample_shape = shape[:-1]
    n_dose = shape[-1]
    # copies, so that the dataset never aliases an array of the caller
    flat = [
        np.array(a, dtype=np.float64).reshape(-1, n_dose)
        for a in (amounts, times, durations)
    ]
    finite_amount = np.isfinite(flat[0])
    finite_time = np.isfinite(flat[1])
    half = finite_amount != finite_time
    if half.any():
        row, column = (int(i) for i in np.argwhere(half)[0])
        raise ValueError(
            "A dose needs an amount and a time, or both NaN for the padding; "
            f"sample {_sample_position(row, sample_shape)} has "
            f"amount {flat[0][row, column]} at time {flat[1][row, column]}"
        )

    order = np.argsort(np.where(finite_time, flat[1], np.inf), axis=1, kind="stable")
    unsorted = (order != np.arange(n_dose)).any(axis=1)
    if unsorted.any():
        logger.warning(
            "The dose times of %d of %d dosing protocols were not sorted",
            int(unsorted.sum()),
            order.shape[0],
        )
        flat = [np.take_along_axis(a, order, axis=1) for a in flat]

    sorted_times = flat[1]
    duplicate = np.diff(sorted_times, axis=1) == 0
    if duplicate.any():
        row = int(np.argwhere(duplicate)[0][0])
        raise ValueError(
            f"Duplicate dose times for sample {_sample_position(row, sample_shape)}: "
            f"{sorted_times[row]}"
        )
    return flat[0].reshape(shape), flat[1].reshape(shape), flat[2].reshape(shape)


def _label_mask(labels: np.ndarray, value: Any) -> np.ndarray:
    """Which entries of a coordinate a label, a list of labels or a slice selects.

    Args:
        labels: the values of the coordinate.
        value: a label, a list or array of labels, or a `slice` of labels whose
            bounds are both included, as in `xarray.Dataset.sel`.

    Returns:
        The boolean mask of the selected entries.

    Raises:
        ValueError: if the bounds of a slice cannot be compared with the labels,
            e.g. integer bounds on string labels.
    """
    if isinstance(value, slice):
        mask = np.ones(labels.shape, dtype=bool)
        try:
            if value.start is not None:
                mask &= labels >= value.start
            if value.stop is not None:
                mask &= labels <= value.stop
        except TypeError as error:
            raise ValueError(
                f"the slice {value!r} cannot be compared with labels of type "
                f"{labels.dtype}; a slice of a labelled dimension is by label"
            ) from error
        return mask
    if isinstance(value, list | tuple | np.ndarray):
        return np.isin(labels, np.asarray(value))
    return labels == value


def _missing_labels(labels: np.ndarray, value: Any) -> list[Any]:
    """The labels of a list selection which no entry of a coordinate carries.

    A list names the samples the caller expects, so a label which is not there
    is a mistake (a typo, a subject of another batch) and not an empty
    selection: `select(individual=["a", "zzz"])` must say so instead of
    silently returning the batch of `a`. A `slice` is a range and a single
    label is reported by the caller when nothing matches it, so neither is
    checked here.

    Args:
        labels: the values of the coordinate.
        value: the selection, as given to `select`.

    Returns:
        The requested labels which no entry carries, in the order they were
        given; empty for a slice, a single label or a list which matches.
    """
    if not isinstance(value, list | tuple | np.ndarray):
        return []
    return [
        label for label in np.asarray(value).tolist() if not (labels == label).any()
    ]


def _dose_variables(
    amounts: np.ndarray,
    times: np.ndarray,
    durations: np.ndarray,
    *,
    dims: tuple[str, ...],
    dose_unit: str,
    time_unit: str,
) -> dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, str]]]:
    """The three dose variables of a `Timecourses` dataset over `(*dims, DOSE_DIM)`.

    Args:
        amounts: the dose amounts of shape `(*sample_shape, n_dose)`.
        times: the dose times with the shape of `amounts`.
        durations: the infusion durations with the shape of `amounts`.
        dims: names of the sample dimensions.
        dose_unit: unit of the amounts.
        time_unit: unit of the times and the durations.

    Returns:
        `dose_amount`, `dose_time` and `dose_duration` as xarray tuples.

    Raises:
        ValueError: if a sample dimension is named like the dose dimension.
    """
    if DOSE_DIM in dims:
        raise ValueError(
            f"A sample dimension must not be named '{DOSE_DIM}' in a batch with "
            f"doses: '{DOSE_DIM}' is the dimension of the doses of "
            "'dose_amount', 'dose_time' and 'dose_duration'"
        )
    all_dims = (*dims, DOSE_DIM)
    return {
        "dose_amount": (all_dims, amounts, {"units": dose_unit}),
        "dose_time": (all_dims, times, {"units": time_unit}),
        "dose_duration": (all_dims, durations, {"units": time_unit}),
    }


@dataclass(frozen=True)
class _SampleArrays:
    """The arrays and the metadata a batch builds its single timecourses from.

    Every array is aligned to `(*sample_dims, time)` or, for the dose
    variables, to `(*sample_dims, dose_index)`, so that the row of a sample is
    the entry of its position along the sample dimensions. The metadata of the
    batch is read once here rather than once per sample: every one of its
    properties goes through the dataset, which rebuilds a `DataArray`.

    Attributes:
        sample_dims: the dimensions other than `time`
        time_unit: unit of the times
        unit: unit of the values
        substance: name of the substance or effect of the batch
        substances: the substance of every sample, `None` when the batch names
            one substance for all of them
        tissue: tissue the values were measured in, `None` when not known
        route: route of the doses, `None` without doses or when the samples
            carry their own route in `routes`
        routes: the route of every sample, `None` when the batch names one
            route for all of them
        dose_unit: unit of the doses, `None` without doses
        times: the sampling times, the shared grid itself when `shared_grid`
        shared_grid: whether every sample has the same sampling times
        values: the values
        sd: the standard deviations, `None` without
        se: the standard errors, `None` without
        n: the counts, one per sample or one per sample and time point,
            `None` without
        dose_amount: the dose amounts, `None` without doses
        dose_time: the dose times, `None` without doses
        dose_duration: the infusion durations, `None` without doses
        lloq: the limit of quantification per sample, `None` without
        labels: the coordinate values of every sample dimension which has one
        complete: whether a curve of the batch needs nothing derived, i.e.
            whether `Timecourse` would leave `sd`, `se` and `n` as they are
    """

    sample_dims: tuple[str, ...]
    time_unit: str
    unit: str
    substance: str
    substances: np.ndarray | None
    tissue: str | None
    route: Route | None
    routes: np.ndarray | None
    dose_unit: str | None
    times: np.ndarray
    shared_grid: bool
    values: np.ndarray
    sd: np.ndarray | None
    se: np.ndarray | None
    n: np.ndarray | None
    dose_amount: np.ndarray | None
    dose_time: np.ndarray | None
    dose_duration: np.ndarray | None
    lloq: np.ndarray | None
    labels: dict[str, np.ndarray]
    complete: bool


class Timecourses:
    """A batch of timecourses as an `xarray.Dataset`.

    The dataset has the dimension `time` and any number of sample dimensions,
    e.g. `individual`, `group`, `study`, or the dimensions of a simulation
    scan. Its variables are

    - `value` over `(*sample_dims, time)`, the values; `NaN` marks missing points,
    - `sd`, `se` over the same dimensions and `n` over the sample dimensions
      (or over `(*sample_dims, time)` when a count varies over the curve), for
      group data (optional),
    - `dose_amount`, `dose_time`, `dose_duration` over the sample dimensions and
      the dose dimension `dose_index` (optional, the three of them together;
      `dose_duration` is a variable of every batch with doses and is `NaN` where
      the route is not an infusion, so that every reader of the dose variables
      works without a case distinction). Every sample carries its protocol
      in its row, the doses at the front and the trailing columns `NaN`, so
      that samples with different numbers of doses share one layout; a single
      dose batch has one column,
    - the coordinate `time` with the shared sampling grid, or, when the samples
      have different sampling times, the variable `times` over
      `(*sample_dims, time)` padded with `NaN` and an integer coordinate `time`.

    Every variable carries its unit in `attrs["units"]`; the dataset carries
    `substance`, `time_unit` and `unit` in its `attrs`, `tissue` when the
    curves name one and `route` only when doses are present. The properties
    `times` and `values` return the `(*sample_shape, n_time)` arrays every
    analysis of the package works on; iteration and `sel`/`isel` give single
    `Timecourse` objects.

    A batch whose samples share one substance and one route carries both in
    `attrs`; a batch of several analytes or of several routes carries them as
    the coordinates `substance` and `route` along a sample dimension, which
    `substances` and `routes` read back and the analyses follow per sample
    (`substance` and `route` raise for such a batch). `n` is one
    number per sample, or one per sample and time point when a count varies
    over the curve, as it does for the group curve of a ragged batch
    (`Timecourses.mean`); `n_subjects` is the number of subjects of a sample
    in either layout.

    Several sample dimensions span their cartesian product, which can have
    combinations without data (no curve was measured for them). Such a sample
    is all `NaN`; iteration and `sel`/`isel` return a `Timecourse` with `NaN`
    values and `dosing=None` for it.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap a dataset, see the class documentation for its layout.

        Every variable with the `time` dimension (`value`, `sd`, `se`, `times`)
        is transposed to `(*sample_dims, time)` and every dose variable to
        `(*sample_dims, dose_index)`, so that the arrays and the data frame of
        the batch are built from one layout.

        Raises:
            ValueError: if the dataset does not have the layout.
        """
        if "value" not in ds:
            raise ValueError("The dataset needs a 'value' variable")
        if TIME_DIM not in ds["value"].dims:
            raise ValueError(f"'value' needs the dimension '{TIME_DIM}'")
        layout = (*(d for d in ds["value"].dims if d != TIME_DIM), TIME_DIM)
        if TIMES_VAR in ds and set(ds[TIMES_VAR].dims) != set(ds["value"].dims):
            raise ValueError(
                f"'{TIMES_VAR}' needs the dimensions {layout} of 'value', "
                f"not {tuple(str(d) for d in ds[TIMES_VAR].dims)}"
            )
        if "dose_amount" in ds and DOSE_DIM in layout:
            raise ValueError(
                f"A sample dimension must not be named '{DOSE_DIM}' in a batch "
                f"with doses: '{DOSE_DIM}' is the dimension of the doses of "
                "'dose_amount', 'dose_time' and 'dose_duration'"
            )
        expected = layout if DOSE_DIM in layout else (*layout, DOSE_DIM)
        if any(
            tuple(ds[name].dims) != tuple(d for d in expected if d in ds[name].dims)
            for name in ds.data_vars
        ):
            ds = ds.transpose(*layout, ...)
        if "units" not in ds["value"].attrs:
            raise ValueError("'value' needs attrs['units']")
        time_var = ds[TIMES_VAR] if TIMES_VAR in ds else ds[TIME_DIM]
        if "units" not in time_var.attrs:
            raise ValueError("the time coordinate needs attrs['units']")
        self.ds: xr.Dataset = ds
        #: the units `_check_units_once` has parsed, so that building single
        #: timecourses parses the three unit strings of the batch once
        self._checked_units: tuple[str, str, str | None] | None = None

    # --- layout -------------------------------------------------------------

    @property
    def sample_dims(self) -> tuple[str, ...]:
        """The dimensions other than `time`."""
        return tuple(str(d) for d in self.ds["value"].dims if d != TIME_DIM)

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """The shape of the sample dimensions."""
        return tuple(int(self.ds.sizes[d]) for d in self.sample_dims)

    @property
    def n_samples(self) -> int:
        """Number of timecourses."""
        return int(np.prod(self.sample_shape, dtype=int)) if self.sample_shape else 1

    @property
    def n_time(self) -> int:
        """Number of time points (the length of the padded grid for ragged data)."""
        return int(self.ds.sizes[TIME_DIM])

    @property
    def time_unit(self) -> str:
        """Unit of the times."""
        time_var = self.ds[TIMES_VAR] if TIMES_VAR in self.ds else self.ds[TIME_DIM]
        return str(time_var.attrs["units"])

    @property
    def unit(self) -> str:
        """Unit of the values."""
        return str(self.ds["value"].attrs["units"])

    def _per_sample(self, name: str) -> np.ndarray | None:
        """The values of a metadata coordinate along the sample dimensions, `None` without.

        A coordinate which is given along some of the sample dimensions only
        (the `substance` of an `(analyte, individual)` batch lives on
        `analyte`) is broadcast to the full sample shape, so that the caller
        indexes it with the position of a sample.

        Args:
            name: name of the coordinate, `substance` or `route`.

        Returns:
            The values of shape `sample_shape` as an object array, or `None`
            when the dataset carries no such coordinate.

        Raises:
            ValueError: if the coordinate carries a dimension which is not a
                sample dimension, e.g. one value per time point.
        """
        if name not in self.ds.variables:
            return None
        da = self.ds[name]
        sample_dims = self.sample_dims
        extra = [str(d) for d in da.dims if str(d) not in sample_dims]
        if extra:
            raise ValueError(
                f"'{name}' must carry one value per sample; it has the "
                f"dimensions {extra}"
            )
        if any(d not in da.dims for d in sample_dims):
            da = da.broadcast_like(self.ds["value"].isel({TIME_DIM: 0}, drop=True))
        return np.asarray(da.transpose(*sample_dims).to_numpy(), dtype=object).reshape(
            self.sample_shape
        )

    @property
    def substances(self) -> np.ndarray | None:
        """The substance of every sample of shape `sample_shape`, `None` when they agree.

        The coordinate `substance` along a sample dimension, which a batch of
        several analytes carries (a parent and its metabolite over
        `(analyte, individual)`), as an array of strings. `None` for a batch
        whose samples name one substance, which `substance` returns.
        """
        per_sample = self._per_sample(SUBSTANCE_VAR)
        if per_sample is None:
            return None
        names = np.asarray([str(value) for value in per_sample.ravel()], dtype=object)
        if len(set(names.tolist())) <= 1:
            return None
        return names.reshape(self.sample_shape)

    @property
    def substance(self) -> str:
        """Name of the substance or effect, the one of the whole batch.

        Returns:
            The substance of the batch: the coordinate `substance` when the
            batch carries one, `attrs["substance"]` otherwise.

        Raises:
            ValueError: if the samples name different substances
                (`substances` reads them then).
        """
        per_sample = self._per_sample(SUBSTANCE_VAR)
        if per_sample is not None:
            names = sorted({str(value) for value in per_sample.ravel()})
            if len(names) != 1:
                raise ValueError(
                    f"The batch carries the substances {names}: read them with "
                    "'Timecourses.substances' or select one analyte with "
                    "'select'"
                )
            return names[0]
        return str(self.ds.attrs.get("substance", "substance"))

    @property
    def tissue(self) -> str | None:
        """Tissue or matrix the values were measured in, `None` when it is not known."""
        tissue = self.ds.attrs.get("tissue")
        return None if tissue is None else str(tissue)

    @property
    def routes(self) -> np.ndarray | None:
        """The route of every sample of shape `sample_shape`, `None` when they agree.

        The coordinate `route` along a sample dimension, which a batch of
        several routes carries (the intravenous reference and the oral test of
        an absolute bioavailability study), as an array of `Route` members.
        `None` for a batch whose samples were given one route, which `route`
        returns. The analysis reads the route of every row from here
        (`pkpdutils.nca.nca`).
        """
        per_sample = self._per_sample(ROUTE_VAR)
        if per_sample is None:
            return None
        given = np.asarray([Route(value) for value in per_sample.ravel()], dtype=object)
        if len({str(route) for route in given.tolist()}) <= 1:
            return None
        return given.reshape(self.sample_shape)

    @property
    def route(self) -> Route | None:
        """Route of the doses, the one of the whole batch, `None` without dose information.

        Returns:
            The route of the batch: the coordinate `route` when the batch
            carries one, `attrs["route"]` otherwise, `None` without doses.

        Raises:
            ValueError: if the samples were given different routes (`routes`
                reads them then).
        """
        per_sample = self._per_sample(ROUTE_VAR)
        if per_sample is not None:
            given = sorted({Route(value).value for value in per_sample.ravel()})
            if len(given) != 1:
                raise ValueError(
                    f"The batch carries the routes {given}: read them with "
                    "'Timecourses.routes' or select one arm with 'select'"
                )
            return Route(given[0])
        route = self.ds.attrs.get("route")
        return None if route is None else Route(route)

    @property
    def lloq(self) -> np.ndarray | None:
        """Limit of quantification per sample of shape `sample_shape`, `None` without.

        The variable or coordinate `lloq` over the sample dimensions: the
        `lloq` of the curves a batch was built from, or the column the readers
        of `pkpdutils.io` carry over (ADNCA `ALLOQ`). `NaN` for a sample whose
        assay names no limit; the analysis reads it when `NCAOptions.lloq`
        names no limit of its own.

        Raises:
            ValueError: if `lloq` carries a dimension which is not a sample
                dimension, e.g. one limit per time point.
        """
        if LLOQ_VAR not in self.ds.variables:
            return None
        da = self.ds[LLOQ_VAR]
        sample_dims = self.sample_dims
        extra = [str(d) for d in da.dims if str(d) not in sample_dims]
        if extra:
            raise ValueError(
                f"'{LLOQ_VAR}' must carry one value per sample, not per time "
                f"point; it has the dimensions {extra}"
            )
        missing = [d for d in sample_dims if d not in da.dims]
        if missing:
            da = da.broadcast_like(self.ds["value"].isel({TIME_DIM: 0}, drop=True))
        return np.asarray(
            da.transpose(*sample_dims).to_numpy(), dtype=np.float64
        ).reshape(self.sample_shape)

    @property
    def has_uncertainty(self) -> bool:
        """Whether `sd` or `se` is present."""
        return "sd" in self.ds or "se" in self.ds

    @property
    def has_dose(self) -> bool:
        """Whether doses are present."""
        return "dose_amount" in self.ds

    # --- arrays -------------------------------------------------------------

    @property
    def times(self) -> np.ndarray:
        """Times as an array of shape `(*sample_shape, n_time)`."""
        if TIMES_VAR in self.ds:
            return self.ds[TIMES_VAR].transpose(*self.sample_dims, TIME_DIM).to_numpy()
        grid = self.ds[TIME_DIM].to_numpy().astype(np.float64)
        return np.broadcast_to(grid, (*self.sample_shape, grid.size)).copy()

    @property
    def values(self) -> np.ndarray:
        """Values as an array of shape `(*sample_shape, n_time)`."""
        return self.ds["value"].transpose(*self.sample_dims, TIME_DIM).to_numpy()

    @property
    def nominal_times(self) -> np.ndarray | None:
        """The nominal (scheduled) time of every point, `None` without.

        The optional variable `nominal_time` of shape `(*sample_shape,
        n_time)`: the time the protocol of the study asked a sample to be taken
        at, where `times` are the times it was taken at. The analyses read the
        actual times; the nominal times are the grid a mean curve over the
        subjects of a study is taken on (`pkpdutils.plot.plot_study_curves`),
        since the actual times of two subjects never coincide. A batch whose
        variable carries fewer dimensions (one nominal grid for every sample)
        gets it broadcast to the shape of the values.
        """
        if NOMINAL_TIMES_VAR not in self.ds:
            return None
        da = self.ds[NOMINAL_TIMES_VAR]
        if set(self.sample_dims) - {str(d) for d in da.dims}:
            da = da.broadcast_like(self.ds["value"])
        return np.asarray(
            da.transpose(*self.sample_dims, TIME_DIM).to_numpy(), dtype=np.float64
        )

    def _optional(self, name: str) -> np.ndarray | None:
        """Return a variable as an array transposed to `(*sample_dims[, time | dose])`, or `None` if absent.

        Args:
            name: name of the data variable.

        Returns:
            The array, or `None` when `name` is not a variable of the dataset.
        """
        if name not in self.ds:
            return None
        da = self.ds[name]
        sample_dims = self.sample_dims
        extra = [
            d for d in (TIME_DIM, DOSE_DIM) if d in da.dims and d not in sample_dims
        ]
        return da.transpose(*sample_dims, *extra).to_numpy()

    @property
    def sd(self) -> np.ndarray | None:
        """Standard deviations, `None` without."""
        return self._optional("sd")

    @property
    def se(self) -> np.ndarray | None:
        """Standard errors, `None` without."""
        return self._optional("se")

    @property
    def n(self) -> np.ndarray | None:
        """Counts behind the values, `None` without.

        One number per sample, or one per sample and time point when the batch
        carries a count per point, as the group curve of `mean` does; a
        `Timecourse` keeps `n` the same way. `n_subjects` reduces the second
        form to one number per sample.
        """
        return self._optional("n")

    @property
    def n_subjects(self) -> np.ndarray | None:
        """Number of subjects per sample, `None` without.

        The stored `n` when it is one number per sample, and the largest count
        over the time points when it is one per time point: the number of
        subjects of a group is the number behind its best covered point.
        """
        n = self._optional("n")
        if n is None or TIME_DIM not in self.ds["n"].dims:
            return n
        with warnings.catch_warnings():
            # a sample without a single count gives `NaN`, not an error
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.asarray(np.nanmax(n, axis=-1), dtype=np.float64)

    @property
    def dose_amount(self) -> np.ndarray | None:
        """Dose amounts of shape `(*sample_shape, n_dose)`, `None` without doses."""
        return self._optional("dose_amount")

    @property
    def dose_time(self) -> np.ndarray | None:
        """Dose times of shape `(*sample_shape, n_dose)`, `None` without doses."""
        return self._optional("dose_time")

    @property
    def dose_duration(self) -> np.ndarray | None:
        """Infusion durations of shape `(*sample_shape, n_dose)` (`NaN` without infusion), `None` without doses."""
        return self._optional("dose_duration")

    @property
    def dose_unit(self) -> str | None:
        """Unit of the doses, `None` without doses."""
        if not self.has_dose:
            return None
        return str(self.ds["dose_amount"].attrs["units"])

    @property
    def n_dose(self) -> int:
        """Size of the dose dimension, the longest protocol of the batch; 0 without doses."""
        return int(self.ds.sizes[DOSE_DIM]) if self.has_dose else 0

    def _dose_mask(self) -> np.ndarray | None:
        """The `(*sample_shape, n_dose)` mask of the doses which are not padding.

        Returns:
            The mask, or `None` without doses.
        """
        amounts = self.dose_amount
        times = self.dose_time
        if amounts is None or times is None:
            return None
        return np.isfinite(amounts) & np.isfinite(times)

    @property
    def n_doses(self) -> np.ndarray | None:
        """Number of doses per sample of shape `sample_shape`, `None` without doses."""
        mask = self._dose_mask()
        return None if mask is None else mask.sum(axis=-1)

    def _dose_entry(self, name: str, *, last: bool) -> np.ndarray | None:
        """The first or the last dose of every sample of a dose variable.

        Args:
            name: name of the dose variable.
            last: whether to take the last dose instead of the first.

        Returns:
            The array of shape `sample_shape` (`NaN` for a sample without a
            dose), or `None` without doses.
        """
        array = self._optional(name)
        counts = self.n_doses
        if array is None or counts is None:
            return None
        index = np.maximum(counts - 1, 0) if last else np.zeros_like(counts)
        picked = np.take_along_axis(array, index[..., None], axis=-1)[..., 0]
        return np.where(counts > 0, picked, np.nan)

    @property
    def first_dose_amount(self) -> np.ndarray | None:
        """Amount of the first dose per sample, `None` without doses."""
        return self._dose_entry("dose_amount", last=False)

    @property
    def last_dose_amount(self) -> np.ndarray | None:
        """Amount of the last dose per sample, `None` without doses."""
        return self._dose_entry("dose_amount", last=True)

    @property
    def first_dose_time(self) -> np.ndarray | None:
        """Time of the first dose per sample, `None` without doses."""
        return self._dose_entry("dose_time", last=False)

    @property
    def last_dose_time(self) -> np.ndarray | None:
        """Time of the last dose per sample, `None` without doses."""
        return self._dose_entry("dose_time", last=True)

    # --- construction -------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        time: Any,
        values: Any,
        *,
        time_unit: str,
        unit: str,
        dims: Sequence[str] = ("individual",),
        coords: Mapping[str, Any] | None = None,
        sd: Any | None = None,
        se: Any | None = None,
        n: Any | None = None,
        nominal_time: Any | None = None,
        dose: Dose | Dosing | Mapping[str, Any] | None = None,
        route: Route | str | None = None,
        substance: str = "substance",
        tissue: str | None = None,
    ) -> "Timecourses":
        """Create a batch from arrays.

        Args:
            time: the sampling grid shared by all samples (1-D), or the times per
                sample with the shape of `values`
            values: values of shape `(*sample_shape, n_time)`
            time_unit: unit of the times
            unit: unit of the values
            dims: names of the sample dimensions, one per leading axis of `values`
            coords: coordinate values per sample dimension (labels of the samples)
            sd: standard deviations with the shape of `values`
            se: standard errors with the shape of `values`
            n: the counts, one number, an array of shape `sample_shape` (one
                count per sample) or an array of the shape of `values` (one
                count per sample and time point)
            nominal_time: the nominal (scheduled) time of every point, the
                sampling grid of the protocol (1-D) or one nominal time per
                sample and point (the shape of `values`); the actual `time`
                stays what the analyses read
            dose: one `Dose` or one `Dosing` protocol for all samples, or a
                mapping with `amount`, `unit` and optionally `time` and
                `duration`; the arrays of the mapping have the shape
                `sample_shape` (one dose per sample) or
                `(*sample_shape, n_dose)` (one protocol per sample, padded with
                `NaN`, `time` required); the route is then given by `route`
            route: route of the doses when `dose` is a mapping, a `Route` or
                a string it coerces (`"oral"`, `"IV_BOLUS"`)
            substance: name of the substance or effect
            tissue: tissue or matrix the values were measured in, e.g.
                `"plasma"`; `None` when it is not known

        The dose variables of the batch hold the invariant the analyses rely
        on: the doses of a sample are the leading columns of its row, sorted by
        time, and the trailing columns are `NaN` padding. The constructor
        enforces it, the rows of a mapping may be given in any order; a dose
        with an amount but no time (or the other way round) and duplicate dose
        times within a sample are errors.

        Returns:
            The batch.

        Raises:
            ValueError: if the shapes do not fit, if `route` is missing for a
                mapping or contradicts the route of a `Dose` or `Dosing`, if a
                mapping breaks the invariant above, or if a sample dimension is
                named `dose_index`.
        """
        values_arr = np.asarray(values, dtype=np.float64)
        dims = tuple(dims)
        if values_arr.ndim != len(dims) + 1:
            raise ValueError(
                f"'values' has shape {values_arr.shape}, expected {len(dims) + 1} axes for dims {dims} + time"
            )
        sample_shape = values_arr.shape[:-1]
        n_time = values_arr.shape[-1]
        all_dims = (*dims, TIME_DIM)
        parse_unit(time_unit)
        parse_unit(unit)

        time_arr = np.asarray(time, dtype=np.float64)
        data_vars: dict[str, Any] = {"value": (all_dims, values_arr, {"units": unit})}
        coordinates: dict[str, Any] = dict(coords or {})
        if time_arr.ndim == 1:
            if time_arr.size != n_time:
                raise ValueError(
                    f"'time' has length {time_arr.size}, 'values' has shape {values_arr.shape}"
                )
            coordinates[TIME_DIM] = (TIME_DIM, time_arr, {"units": time_unit})
        else:
            if time_arr.shape != values_arr.shape:
                raise ValueError(
                    f"'time' has shape {time_arr.shape}, 'values' has shape {values_arr.shape}"
                )
            data_vars[TIMES_VAR] = (all_dims, time_arr, {"units": time_unit})
            coordinates[TIME_DIM] = (TIME_DIM, np.arange(n_time), {"units": time_unit})

        for name, raw in (("sd", sd), ("se", se)):
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=np.float64)
            if arr.shape != values_arr.shape:
                raise ValueError(
                    f"'{name}' has shape {arr.shape}, 'values' has shape {values_arr.shape}"
                )
            data_vars[name] = (all_dims, arr, {"units": unit})
        if nominal_time is not None:
            nominal = np.asarray(nominal_time, dtype=np.float64)
            if nominal.ndim == 1:
                if nominal.size != n_time:
                    raise ValueError(
                        f"'nominal_time' has length {nominal.size}, 'values' has "
                        f"shape {values_arr.shape}"
                    )
                nominal = np.broadcast_to(nominal, values_arr.shape)
            elif nominal.shape != values_arr.shape:
                raise ValueError(
                    f"'nominal_time' has shape {nominal.shape}, 'values' has "
                    f"shape {values_arr.shape}"
                )
            data_vars[NOMINAL_TIMES_VAR] = (
                all_dims,
                np.array(nominal, dtype=np.float64),
                {"units": time_unit},
            )
        if n is not None:
            # `n` is one count per sample, or one per sample and time point
            # when it has the shape of the values (the group curve of `mean`)
            n_raw = np.asarray(n, dtype=np.float64)
            per_time = n_raw.shape == values_arr.shape
            n_arr = (
                n_raw.copy()
                if per_time
                else np.broadcast_to(n_raw, sample_shape).copy()
            )
            root_n = np.sqrt(n_arr) if per_time else np.sqrt(n_arr)[..., None]
            data_vars["n"] = (
                all_dims if per_time else dims,
                n_arr,
                {"units": "dimensionless"},
            )
            if "sd" in data_vars and "se" not in data_vars:
                data_vars["se"] = (
                    all_dims,
                    data_vars["sd"][1] / root_n,
                    {"units": unit},
                )
            elif "se" in data_vars and "sd" not in data_vars:
                data_vars["sd"] = (
                    all_dims,
                    data_vars["se"][1] * root_n,
                    {"units": unit},
                )

        attrs: dict[str, Any] = {
            "substance": substance,
            "time_unit": time_unit,
            "unit": unit,
        }
        if tissue is not None:
            attrs["tissue"] = tissue
        if dose is not None:
            amounts, dose_times, durations, dose_unit, dose_route = _batch_dose_arrays(
                dose, route=route, sample_shape=sample_shape
            )
            attrs["route"] = dose_route.value
            data_vars.update(
                _dose_variables(
                    amounts,
                    dose_times,
                    durations,
                    dims=dims,
                    dose_unit=dose_unit,
                    time_unit=time_unit,
                )
            )
            coordinates[DOSE_DIM] = (DOSE_DIM, np.arange(amounts.shape[-1]))

        ds = xr.Dataset(data_vars=data_vars, coords=coordinates, attrs=attrs)
        return cls(ds)

    @classmethod
    def from_timecourses(
        cls,
        timecourses: Sequence[Timecourse],
        dim: str = "individual",
        labels: Sequence[Any] | None = None,
    ) -> "Timecourses":
        """Create a batch from single timecourses along one sample dimension.

        The timecourses must share `time_unit`, `unit` and `tissue`. If all
        sampling grids are equal the grid becomes the `time` coordinate,
        otherwise the times are stored per sample and shorter curves are padded
        with `NaN`.

        Curves of different substances (a parent and its metabolite) and curves
        given by different routes (the intravenous reference and the oral test
        of a bioavailability study) go into one batch: the differing values
        become the coordinates `substance` and `route` along `dim`, which
        `Timecourses.substances` and `Timecourses.routes` read back and the
        analysis follows per sample. A batch whose curves agree carries the one
        value in `attrs` as before and grows no coordinate.

        Either all or no curves carry a dosing protocol, and all protocols need
        the same dose unit. The protocols are padded to the longest one. The batch keeps
        one `n` per sample, and the counts per time point when the `n` of a
        curve varies over its time points (`n_subjects` reads the number of
        subjects back either way). `sd`, `se` and `n` are kept only when every
        curve carries them; a field which some curves are missing is dropped
        for the whole batch and logs a warning.

        Args:
            timecourses: the curves
            dim: name of the sample dimension
            labels: coordinate labels of the samples, the `label` of every
                timecourse (or its index when missing) by default

        Returns:
            The batch.

        Raises:
            ValueError: for an empty sequence, differing units or tissues,
                doses on some but not all curves, or doses with different
                units.
        """
        if not timecourses:
            raise ValueError("At least one timecourse is required")
        first = timecourses[0]
        for tc in timecourses[1:]:
            if tc.time_unit != first.time_unit or tc.unit != first.unit:
                raise ValueError(
                    f"All timecourses need the same units: '{first.time_unit}'/'{first.unit}' "
                    f"and '{tc.time_unit}'/'{tc.unit}'"
                )
            if tc.tissue != first.tissue:
                raise ValueError(
                    f"All timecourses need the same tissue: '{first.tissue}' "
                    f"and '{tc.tissue}'"
                )

        if labels is None:
            labels = [
                tc.label if tc.label is not None else i
                for i, tc in enumerate(timecourses)
            ]
        n_time = max(tc.size for tc in timecourses)
        shared = all(
            tc.size == first.size and np.array_equal(tc.time, first.time)
            for tc in timecourses
        )

        def padded(arrays: Sequence[np.ndarray | None]) -> np.ndarray | None:
            """Stack ragged 1-D arrays into a `(len(arrays), n_time)` array padded with `NaN`.

            Args:
                arrays: the arrays to stack, one per sample.

            Returns:
                The padded array, or `None` when any element of `arrays` is `None`.
            """
            if any(a is None for a in arrays):
                return None
            return pad_rows([a for a in arrays if a is not None], n_time)

        def warn_partial(name: str, arrays: Sequence[Any]) -> None:
            """Warn when some but not all curves carry an optional field.

            Args:
                name: name of the field
                arrays: the field of every curve, `None` where it is missing.
            """
            without = [
                str(label) for label, a in zip(labels, arrays, strict=True) if a is None
            ]
            if without and len(without) != len(arrays):
                logger.warning(
                    "'%s' is missing for %s, the batch of %d curves carries no '%s'",
                    name,
                    without,
                    len(arrays),
                    name,
                )

        values = padded([tc.value for tc in timecourses])
        assert values is not None
        time: np.ndarray
        if shared:
            time = first.time
        else:
            padded_time = padded([tc.time for tc in timecourses])
            assert padded_time is not None
            time = padded_time
        for name in ("sd", "se", "n"):
            warn_partial(name, [getattr(tc, name) for tc in timecourses])
        sd = padded([tc.sd for tc in timecourses])
        se = padded([tc.se for tc in timecourses])
        n_values = [tc.n for tc in timecourses]
        n: np.ndarray | None = None
        if all(v is not None for v in n_values):
            # a curve carries its count as one number or per time point; the
            # batch keeps the counts per time point when one of them varies
            counts = padded(
                [
                    np.broadcast_to(np.asarray(v, dtype=np.float64), tc.time.shape)
                    for tc, v in zip(timecourses, n_values, strict=True)
                ]
            )
            assert counts is not None
            n = _batch_counts(counts)

        protocols = [tc.dosing for tc in timecourses]
        without_dose = [
            label for label, d in zip(labels, protocols, strict=True) if d is None
        ]
        if without_dose and len(without_dose) != len(protocols):
            raise ValueError(
                "Either all or no timecourses need a dose, there is no dose for "
                f"{[str(label) for label in without_dose]}"
            )
        dose, route = dose_mapping(protocols, allow_mixed_routes=True)

        # the limit of quantification is metadata of a curve and travels as a
        # coordinate along the sample dimension, as the readers write it
        limits = np.array(
            [np.nan if tc.lloq is None else float(tc.lloq) for tc in timecourses]
        )
        coords: dict[str, Any] = {dim: list(labels)}
        if np.isfinite(limits).any():
            coords[LLOQ_VAR] = (dim, limits)

        # the substance and the route become coordinates when the curves differ
        # in them, so that a batch of several analytes or of several routes is
        # one batch and every sample keeps its own value
        substances = [tc.substance for tc in timecourses]
        if len(set(substances)) > 1:
            coords[SUBSTANCE_VAR] = (dim, np.array(substances, dtype=object))
        # either all or no curve carries a protocol, checked above
        sample_routes = [
            protocol.route for protocol in protocols if protocol is not None
        ]
        if len(set(sample_routes)) > 1:
            coords[ROUTE_VAR] = (
                dim,
                np.array([str(r) for r in sample_routes], dtype=object),
            )

        return cls.from_arrays(
            time,
            values,
            time_unit=first.time_unit,
            unit=first.unit,
            dims=(dim,),
            coords=coords,
            sd=sd,
            se=se,
            n=n,
            dose=dose,
            route=route,
            substance=first.substance,
            tissue=first.tissue,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        sample: Sequence[str],
        time_unit: str,
        unit: str,
        time: str = "time",
        value: str = "value",
        sd: str | None = None,
        se: str | None = None,
        n: str | None = None,
        nominal_time: str | None = None,
        lloq: str | None = None,
        dose_amount: str | None = None,
        dose_unit: str | None = None,
        dose_time: str | None = None,
        route: Route | str | None = None,
        substance: str = "substance",
        tissue: str | None = None,
    ) -> "Timecourses":
        """Create a batch from a long data frame, one row per sample and time point.

        Several sample columns span their cartesian product; a combination
        without rows in `df` becomes a sample with `NaN` values, which iteration
        and `sel`/`isel` return as a `Timecourse` with `NaN` values and
        `dose=None`.

        Args:
            df: the data frame
            sample: the columns which identify a sample; they become the sample
                dimensions, several columns give their cartesian product with
                `NaN` for missing combinations
            time_unit: unit of the time column
            unit: unit of the value column
            time: name of the time column
            value: name of the value column
            sd: name of the standard deviation column
            se: name of the standard error column
            n: name of the column with the number of subjects; the batch keeps
                one number per sample, and the counts per time point when the
                column varies within a sample
            nominal_time: name of the column with the nominal (scheduled) time
                of the point, in `time_unit`; it becomes the variable
                `nominal_time` of the batch and the actual `time` column stays
                what the analyses read
            lloq: name of the column with the limit of quantification, which
                has to be constant within a sample; it becomes the coordinate
                `lloq` along the sample dimension
            dose_amount: name of the dose column (constant per sample without
                `dose_time`, one value per dose time with it)
            dose_unit: unit of the doses, required with `dose_amount`
            dose_time: name of the dose time column, 0 by default; every
                distinct `(dose_time, dose_amount)` pair of a sample is one
                dose of its protocol, rows with `NaN` dose columns are
                observations only
            route: route of the doses, required with `dose_amount`, a `Route`
                or a string it coerces (`"oral"`, `"IV_BOLUS"`)
            substance: name of the substance or effect
            tissue: tissue or matrix the values were measured in, e.g.
                `"plasma"`

        The batch is built from the padded arrays of the frame, not from one
        `Timecourse` per sample: the checks of a single curve (at least two
        time points, no `NaN` and no duplicate times) and of a single protocol
        (finite, non-negative amounts, one amount per dose time) are made for
        every sample at once and name the sample they fail for.

        Returns:
            The batch.

        Raises:
            ValueError: if `sample` is empty, if the frame holds no sample, if
                a column holds a value which is neither missing nor a number,
                if a sample has fewer than two time points, a `NaN` time or
                duplicate times, if the limit of quantification is not constant
                within a sample, or if the doses of a sample are not a valid
                protocol; every one of them names the sample.
        """
        sample = list(sample)
        if not sample:
            raise ValueError("'sample' needs at least one column")
        if dose_amount is not None and DOSE_DIM in sample:
            raise ValueError(
                f"A sample column must not be named '{DOSE_DIM}' in a batch "
                f"with doses: '{DOSE_DIM}' is the dimension of the doses of "
                "'dose_amount', 'dose_time' and 'dose_duration'"
            )
        route = None if route is None else Route(route)
        # the samples in the order of their first appearance, the order
        # `to_dataframe` and the readers write them in
        codes, keys = _sample_codes(df, sample)
        if not keys:
            raise ValueError("At least one timecourse is required")
        columns = {
            name: _numeric_column(df, name, codes=codes, labels=keys).to_numpy()
            for name in dict.fromkeys(
                name
                for name in (time, value, sd, se, n, nominal_time, lloq)
                if name is not None
            )
        }
        times = columns[time]
        order, row, column, counts = _rows_by_sample(codes, times, len(keys))
        _check_sample_times(times[order], row, counts, keys)

        def padded(name: str | None) -> np.ndarray | None:
            """Stack one column of the frame into the `(n_samples, n_time)` array.

            Args:
                name: name of the column, `None` for a column the caller
                    switched off.

            Returns:
                The padded array, `NaN` where a sample has fewer time points,
                or `None` when `name` is `None`.
            """
            if name is None:
                return None
            out = np.full((len(keys), int(counts.max())), np.nan)
            out[row, column] = columns[name][order]
            return out

        grid = padded(time)
        assert grid is not None
        values = padded(value)
        spread = {"sd": padded(sd), "se": padded(se)}
        subjects = padded(n)
        if subjects is not None:
            # `sd` and `se` are derived from each other per time point, as a
            # single curve does; the batch keeps one `n` per sample unless a
            # sample counts its time points separately
            with np.errstate(invalid="ignore"):
                root = np.sqrt(subjects)
            if spread["se"] is None and spread["sd"] is not None:
                spread["se"] = spread["sd"] / root
            elif spread["sd"] is None and spread["se"] is not None:
                spread["sd"] = spread["se"] * root
        dose: dict[str, Any] | None = None
        if dose_amount is not None:
            dose = _frame_doses(
                df,
                codes=codes,
                labels=keys,
                dose_amount=dose_amount,
                dose_unit=dose_unit,
                dose_time=dose_time,
                route=route,
            )

        shared = bool(
            (counts == counts[0]).all()
            and np.array_equal(grid, np.broadcast_to(grid[0], grid.shape))
        )
        dim = sample[0] if len(sample) == 1 else "_sample"
        coordinates: dict[str, Any] = {
            dim: (list(keys) if len(sample) == 1 else list(range(len(keys))))
        }
        if lloq is not None:
            coordinates[LLOQ_VAR] = (
                dim,
                _constant_per_sample(
                    columns[lloq], name=lloq, codes=codes, labels=keys
                ),
            )
        flat = cls.from_arrays(
            grid[0] if shared else grid,
            values,
            time_unit=time_unit,
            unit=unit,
            dims=(dim,),
            coords=coordinates,
            sd=spread["sd"],
            se=spread["se"],
            n=None if subjects is None else _batch_counts(subjects),
            nominal_time=padded(nominal_time),
            dose=dose,
            route=route,
            substance=substance,
            tissue=tissue,
        )
        if len(sample) == 1:
            return flat

        # several sample columns: built along one flat dimension, then unstacked
        tuple_keys: list[tuple[Any, ...]] = []
        for k in keys:
            assert isinstance(k, tuple)
            tuple_keys.append(k)
        index = pd.MultiIndex.from_tuples(tuple_keys, names=sample)
        mindex_coords = xr.Coordinates.from_pandas_multiindex(index, "_sample")
        ds = (
            flat.ds.drop_vars("_sample").assign_coords(mindex_coords).unstack("_sample")
        )
        ds.attrs.update(flat.ds.attrs)
        for name in ds.data_vars:
            ds[name].attrs.update(flat.ds[name].attrs)
        if TIME_DIM in ds.coords:
            ds[TIME_DIM].attrs.update(flat.ds[TIME_DIM].attrs)
        return cls(ds.transpose(*sample, TIME_DIM, ...))

    @classmethod
    def from_dataset(
        cls,
        ds: xr.Dataset,
        value: str,
        *,
        unit: str,
        time_unit: str,
        time_dim: str = "_time",
        time: str | None = None,
        dose: Dose | Dosing | Mapping[str, Any] | None = None,
        route: Route | str | None = None,
        substance: str | None = None,
        tissue: str | None = None,
    ) -> "Timecourses":
        """Create a batch from a dataset of a simulation, e.g. a parameter scan.

        Args:
            ds: dataset with the time dimension `time_dim` and the variable `value`
                over it and the scan dimensions
            value: name of the variable with the values
            unit: unit of the values
            time_unit: unit of the times
            time_dim: name of the time dimension
            time: name of the variable with the time values, the coordinate of
                `time_dim` by default
            dose: the doses, as in `from_arrays`: one `Dose` or one `Dosing`
                protocol for all samples, or a mapping of arrays whose rows are
                sorted by dose time with the `NaN` padding trailing
            route: route of the doses when `dose` is a mapping, a `Route` or
                a string it coerces (`"oral"`, `"IV_BOLUS"`)
            substance: name of the substance, `value` by default
            tissue: tissue or matrix the values were measured in, e.g.
                `"plasma"`

        Returns:
            The batch with the scan dimensions as sample dimensions.

        Raises:
            ValueError: if `value` has no dimension `time_dim`, if the time
                values are not one dimensional, or if `dose` does not fit the
                batch (`from_arrays`).
        """
        da = ds[value]
        if time_dim not in da.dims:
            raise ValueError(f"'{value}' has no dimension '{time_dim}'")
        sample_dims = tuple(str(d) for d in da.dims if d != time_dim)
        values = da.transpose(*sample_dims, time_dim).to_numpy()
        grid = (
            (ds[time] if time is not None else ds[time_dim])
            .to_numpy()
            .astype(np.float64)
        )
        if grid.ndim != 1:
            raise ValueError("The time values must be one dimensional")
        coords = {d: ds[d].to_numpy() for d in sample_dims if d in ds.coords}
        return cls.from_arrays(
            grid,
            values,
            time_unit=time_unit,
            unit=unit,
            dims=sample_dims,
            coords=coords,
            dose=dose,
            route=route,
            substance=value if substance is None else substance,
            tissue=tissue,
        )

    @classmethod
    def from_xresult(
        cls,
        xres: Any,
        selection: str,
        *,
        dose: Dose | Dosing | None = None,
        substance: str | None = None,
    ) -> "Timecourses":
        """Create a batch from the result of an sbmlsim simulation.

        sbmlsim is not a dependency; an `XResult` is used by its attributes:
        `xres.xds` is the dataset with the `_time` dimension and `xres.uinfo`
        maps `selection` and `"time"` to unit strings.

        Args:
            xres: the `sbmlsim.result.XResult`
            selection: the variable of the result, e.g. `"[Cve_mid]"`
            dose: one dose or one dosing protocol for all samples
            substance: name of the substance, `selection` by default

        Returns:
            The batch with the scan dimensions as sample dimensions.
        """
        return cls.from_dataset(
            xres.xds,
            selection,
            unit=str(xres.uinfo[selection]),
            time_unit=str(xres.uinfo["time"]),
            dose=dose,
            substance=substance,
        )

    # --- access -------------------------------------------------------------

    def __len__(self) -> int:
        """Number of timecourses."""
        return self.n_samples

    def __eq__(self, other: object) -> bool:
        """Compare two batches by their datasets, `NaN` equals `NaN`.

        `xarray.Dataset.identical` compares the variables, the coordinates and
        their values, the names and every `attrs` of the dataset and of its
        variables; values are compared with `NaN` equal to `NaN`, so the
        padding of the ragged grids and of the dosing protocols compares equal.

        Args:
            other: the object to compare with.

        Returns:
            Whether `other` is a batch with an identical dataset;
            `NotImplemented` for any other type, so that python falls back to
            the identity comparison.
        """
        if not isinstance(other, Timecourses):
            return NotImplemented
        return bool(self.ds.identical(other.ds))

    #: a batch is a mutable wrapper of its dataset, so it is not hashable
    #: (python would set this implicitly, it is spelled out to say so)
    __hash__ = None

    def _sample_arrays(self) -> "_SampleArrays":
        """The arrays of the batch, aligned once for building single timecourses.

        `isel`, `sel` and the iteration take the row of a sample out of these
        arrays instead of indexing the dataset, which rebuilds every
        `DataArray` of the batch per sample.

        Returns:
            The arrays, transposed to `(*sample_dims, time)` and
            `(*sample_dims, dose_index)`; `times` is the shared grid itself
            when the samples share one.
        """
        variables = self.ds.variables
        sample_dims = self.sample_dims
        order = (
            *sample_dims,
            *(d for d in (TIME_DIM, DOSE_DIM) if d not in sample_dims),
        )

        def aligned(name: str) -> np.ndarray | None:
            """The data of one variable in the dimension order of the batch.

            Args:
                name: name of the variable.

            Returns:
                The array, or `None` when the variable is not in the dataset.
            """
            variable = variables.get(name)
            if variable is None:
                return None
            dims = tuple(str(d) for d in variable.dims)
            target = tuple(d for d in order if d in dims)
            data = np.asarray(variable.values)
            if target != dims:
                data = data.transpose([dims.index(d) for d in target])
            return data

        per_sample = aligned(TIMES_VAR)
        shared = per_sample is None
        times = (
            np.asarray(variables[TIME_DIM].values, dtype=np.float64)
            if per_sample is None
            else per_sample
        )
        values = aligned("value")
        assert values is not None
        sd, se = aligned("sd"), aligned("se")
        n = aligned("n")
        # `Timecourse` derives the missing one of `sd` and `se` from `n`: a
        # batch which carries only one of them needs that validation
        derives = n is not None and (sd is None) != (se is None)
        # the substance and the route are either one value for the batch or a
        # coordinate along the sample dimensions; the single value is read only
        # when there is no coordinate, since the property raises for a mixed one
        substances = self.substances
        routes = self.routes
        substance = "" if substances is not None else self.substance
        route = None if routes is not None else self.route
        return _SampleArrays(
            sample_dims=sample_dims,
            time_unit=self.time_unit,
            unit=self.unit,
            substance=substance,
            substances=substances,
            tissue=self.tissue,
            route=route,
            routes=routes,
            dose_unit=self.dose_unit,
            times=times,
            shared_grid=shared,
            values=values,
            sd=sd,
            se=se,
            n=n,
            dose_amount=aligned("dose_amount"),
            dose_time=aligned("dose_time"),
            dose_duration=aligned("dose_duration"),
            lloq=self.lloq,
            labels={
                d: self.ds[d].to_numpy() for d in sample_dims if d in self.ds.coords
            },
            complete=not derives,
        )

    def _check_units_once(self, arrays: "_SampleArrays") -> None:
        """Check the unit strings of the batch, once per set of units.

        The units of a batch are three strings, not per sample data, so the
        pint parse of `Timecourse` and `Dosing` is done once for the batch
        instead of once per sample built from it.

        Args:
            arrays: the arrays of the batch, which carry its units.

        Raises:
            ValueError: if a unit is not a unit, or the dose unit is not an
                amount (`pkpdutils.units.check_dose_unit`).
        """
        units = (arrays.time_unit, arrays.unit, arrays.dose_unit)
        if units == self._checked_units:
            return
        parse_unit(units[0])
        parse_unit(units[1])
        if units[2] is not None:
            check_dose_unit(units[2])
        self._checked_units = units

    def _timecourse_at(
        self, arrays: "_SampleArrays", index: tuple[int, ...], label: Any
    ) -> Timecourse:
        """Build the `Timecourse` of one sample from the arrays of the batch.

        The curve is built with `model_construct`, which skips the validation
        of `Timecourse`, when the row of the sample already holds every
        invariant it validates: at least two time points which are strictly
        increasing (so no `NaN`, no duplicates and no sorting), `float64`
        arrays and nothing to derive (`_SampleArrays.complete`). The units are
        checked once per batch by `_check_units_once`. Every other row goes
        through the validating constructor, which sorts it or raises as before.

        Args:
            arrays: the arrays of the batch.
            index: the position of the sample along the sample dimensions.
            label: label of the sample, `None` without sample dimensions.

        Returns:
            The timecourse.
        """
        time = arrays.times if arrays.shared_grid else arrays.times[index]
        mask = ~np.isnan(time)
        data: dict[str, Any] = {
            "time": np.asarray(time[mask], dtype=np.float64),
            "value": np.asarray(arrays.values[index][mask], dtype=np.float64),
            "time_unit": arrays.time_unit,
            "unit": arrays.unit,
            "substance": (
                arrays.substance
                if arrays.substances is None
                else str(arrays.substances[index])
            ),
            "tissue": arrays.tissue,
            "label": None if label is None else str(label),
        }
        for name in ("sd", "se"):
            array = getattr(arrays, name)
            if array is not None:
                data[name] = np.asarray(array[index][mask], dtype=np.float64)
        if arrays.n is not None:
            row = arrays.n[index]
            # one count per time point (the group curve of `mean`) travels
            # into the curve as it is, one count per sample as the number
            data["n"] = (
                np.asarray(row[mask], dtype=np.float64)
                if np.ndim(row) > 0
                else float(row)
            )
        if arrays.lloq is not None:
            limit = float(arrays.lloq[index])
            if np.isfinite(limit):
                data["lloq"] = limit
        dosing = self._dosing_at(arrays, index)
        if dosing is not None:
            data["dosing"] = dosing
        curve = data["time"]
        if arrays.complete and curve.size >= 2 and bool((np.diff(curve) > 0).all()):
            self._check_units_once(arrays)
            return Timecourse.model_construct(**data)
        return Timecourse(**data)

    def _dosing_at(
        self, arrays: "_SampleArrays", index: tuple[int, ...]
    ) -> Dosing | None:
        """Rebuild the dosing protocol of one sample from the arrays of the batch.

        The doses of a sample are the finite entries of its dose row; a row of
        `NaN` marks a sample combination which is not in the batch (its values
        are `NaN` as well), it has no protocol. The protocol is built with
        `model_construct`, which skips the validation of `Dosing`, when the row
        already holds every invariant it validates: finite, non-negative
        amounts at strictly increasing times and durations which fit the route.
        Every other row goes through the validating constructor.

        Args:
            arrays: the arrays of the batch.
            index: the position of the sample along the sample dimensions.

        Returns:
            The protocol, or `None` without doses.
        """
        # the three dose variables are present together
        if (
            arrays.dose_amount is None
            or arrays.dose_time is None
            or arrays.dose_duration is None
        ):
            return None
        amounts = np.atleast_1d(np.asarray(arrays.dose_amount[index], dtype=np.float64))
        times = np.atleast_1d(np.asarray(arrays.dose_time[index], dtype=np.float64))
        durations = np.atleast_1d(
            np.asarray(arrays.dose_duration[index], dtype=np.float64)
        )
        mask = np.isfinite(amounts) & np.isfinite(times)
        if not mask.any():
            return None
        amounts, times, durations = amounts[mask], times[mask], durations[mask]
        # both are set with the dose variables, which were checked above; a
        # batch of several routes carries the route of the sample in `routes`
        route = arrays.route if arrays.routes is None else Route(arrays.routes[index])
        dose_unit = arrays.dose_unit
        assert route is not None and dose_unit is not None
        given = None if bool(np.isnan(durations).all()) else durations
        infusion = route is Route.IV_INFUSION
        fits = (
            given is not None and bool((np.isfinite(given) & (given > 0)).all())
            if infusion
            else given is None
        )
        if (
            fits
            and bool((amounts >= 0).all())
            and bool((np.diff(times) > 0).all() if times.size > 1 else True)
        ):
            self._check_units_once(arrays)
            return Dosing.model_construct(
                amounts=amounts,
                times=times,
                durations=given,
                unit=dose_unit,
                route=route,
            )
        return Dosing(
            amounts=amounts,
            times=times,
            durations=durations,
            unit=dose_unit,
            route=route,
        )

    def _position(
        self, indexers: Mapping[str, Any], *, by_label: bool
    ) -> tuple[int, ...]:
        """The position of one sample along the sample dimensions.

        A label of a dimension which carries no coordinate is taken as the
        position itself, so that `sel` selects such a dimension positionally as
        `xarray.sel` does (which hands the label to `isel` unchanged); the
        label is coerced with `int`, so a label which is not a number raises
        here rather than in the indexing.

        Args:
            indexers: one index or coordinate label per sample dimension.

        Keyword Args:
            by_label: whether the indexers are coordinate labels (`sel`) or
                integer positions (`isel`).

        Returns:
            The position of the sample.

        Raises:
            ValueError: for an unknown dimension, or for a label of a dimension
                without a coordinate which is not an integer.
            KeyError: for a label which is not a coordinate of its dimension.
        """
        unknown = set(indexers) - set(self.sample_dims)
        if unknown:
            raise ValueError(
                f"{sorted(unknown)} are not sample dimensions of the batch, "
                f"which has {list(self.sample_dims)}"
            )
        position: list[int] = []
        for dim in self.sample_dims:
            label = indexers[dim]
            index = self.ds.indexes.get(dim) if by_label else None
            position.append(
                int(index.get_loc(label)) if index is not None else int(label)
            )
        return tuple(position)

    def _label_at(self, arrays: "_SampleArrays", index: tuple[int, ...]) -> Any:
        """Label of a sample: the coordinates of the sample dimensions joined by `|`.

        Args:
            arrays: the arrays of the batch.
            index: the position of the sample along the sample dimensions.

        Returns:
            The label, `None` when no sample dimension carries a coordinate.
        """
        parts = [
            str(arrays.labels[dim][position])
            for dim, position in zip(arrays.sample_dims, index, strict=True)
            if dim in arrays.labels
        ]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else "|".join(parts)

    def relative_to_dose(
        self, which: Literal["first", "last"] = "first"
    ) -> "Timecourses":
        """Copy with the times of every sample relative to a dose of its own protocol.

        Every sample is shifted by the time of its first (or last) dose, so
        that this dose is at time 0; the dose times of its protocol are
        shifted with it and a sample without a protocol stays where it is.
        The batch is returned unchanged when it carries no doses or when
        every dose time is already 0.

        Equal shifts keep the layout of the batch, a shared sampling grid
        included. Shifts which differ from sample to sample move the samples
        against each other: the times of every sample are then placed on the
        union of the shifted grids, with `NaN` values where a sample has no
        point at a time of another sample.

        Args:
            which: `"first"` shifts every sample by the time of its first
                dose, `"last"` by the time of its last dose.

        Returns:
            The shifted batch, or `self` when there is nothing to shift.
        """
        dose_time = self.first_dose_time if which == "first" else self.last_dose_time
        if dose_time is None:
            return self
        shift = np.where(np.isfinite(dose_time), dose_time, 0.0).astype(np.float64)
        if not shift.any():
            return self
        flat_shift = shift.reshape(-1)
        uniform = bool(np.all(flat_shift == flat_shift[0]))
        ds = self._shifted_times(shift, uniform=uniform)
        offsets = xr.DataArray(shift, dims=self.sample_dims)
        # the dose times and the nominal sampling times move with the actual
        # times, so that every time of the batch keeps the same reference
        for name in ("dose_time", NOMINAL_TIMES_VAR):
            if name in ds:
                attrs = dict(ds[name].attrs)
                ds[name] = ds[name] - offsets
                ds[name].attrs.update(attrs)
        return Timecourses(ds)

    def _shifted_times(self, shift: np.ndarray, *, uniform: bool) -> xr.Dataset:
        """The dataset with the sampling times shifted by one offset per sample.

        Args:
            shift: the offset of every sample, of shape `sample_shape`.
            uniform: whether every offset is the same, in which case the
                layout of the batch is kept and only the times are moved.

        Returns:
            The dataset with the shifted times; a non-uniform shift puts the
            values on the union of the shifted grids.
        """
        ds = self.ds.copy()
        if uniform:
            offset = float(shift.reshape(-1)[0])
            if TIMES_VAR in ds:
                attrs = dict(ds[TIMES_VAR].attrs)
                ds[TIMES_VAR] = ds[TIMES_VAR] - offset
                ds[TIMES_VAR].attrs.update(attrs)
            else:
                attrs = dict(ds[TIME_DIM].attrs)
                ds = ds.assign_coords(
                    {TIME_DIM: ds[TIME_DIM].to_numpy().astype(np.float64) - offset}
                )
                ds[TIME_DIM].attrs.update(attrs)
            return ds

        n_rows, n_time = self.n_samples, self.n_time
        times = (self.times - shift[..., None]).reshape(n_rows, n_time)
        finite = np.isfinite(times)
        grid = np.unique(times[finite])
        rows = np.repeat(np.arange(n_rows), finite.sum(axis=1))
        columns = np.searchsorted(grid, times[finite])
        time_vars = [
            str(name)
            for name, da in self.ds.data_vars.items()
            if TIME_DIM in da.dims and str(name) != TIMES_VAR
        ]
        regridded: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, Any]]] = {}
        for name in time_vars:
            values = (
                self.ds[name]
                .transpose(*self.sample_dims, TIME_DIM)
                .to_numpy()
                .astype(np.float64)
                .reshape(n_rows, n_time)
            )
            placed = np.full((n_rows, grid.size), np.nan)
            placed[rows, columns] = values[finite]
            regridded[name] = (
                (*self.sample_dims, TIME_DIM),
                placed.reshape(*self.sample_shape, grid.size),
                dict(self.ds[name].attrs),
            )
        time_attrs = dict(
            (self.ds[TIMES_VAR] if TIMES_VAR in self.ds else self.ds[TIME_DIM]).attrs
        )
        ds = ds.drop_vars([*time_vars, *([TIMES_VAR] if TIMES_VAR in ds else [])])
        ds = ds.drop_dims(TIME_DIM) if TIME_DIM in ds.dims else ds
        ds = ds.assign_coords({TIME_DIM: grid})
        ds[TIME_DIM].attrs.update(time_attrs)
        for name, (dims, values, attrs) in regridded.items():
            ds[name] = xr.DataArray(values, dims=dims, attrs=attrs)
        return ds

    def _coordinate_dim(self, name: str) -> str:
        """The sample dimension a coordinate of the batch lives on.

        Args:
            name: name of a sample dimension or of a coordinate along one.

        Returns:
            The sample dimension.

        Raises:
            ValueError: if `name` is neither a sample dimension nor a
                coordinate along exactly one of them.
        """
        if name in self.sample_dims:
            return name
        if name not in self.ds.coords:
            raise ValueError(
                f"'{name}' is neither a sample dimension {self.sample_dims} nor a "
                "coordinate of the batch"
            )
        dims = tuple(str(d) for d in self.ds[name].dims)
        if len(dims) != 1 or dims[0] not in self.sample_dims:
            raise ValueError(
                f"coordinate '{name}' has the dimensions {dims}, a selection needs "
                f"a coordinate along one sample dimension {self.sample_dims}"
            )
        return dims[0]

    def select(self, **indexers: Any) -> "Timecourses":
        """A sub-batch by label, list or slice on the sample dimensions and their coordinates.

        The counterpart of `sel`, which returns a single `Timecourse` and needs
        a label for every sample dimension: `select` keeps the dimensions and
        returns a batch, so that the arm of a study, a dose group or the
        subjects of a period can be analysed on their own. A single label
        therefore does not drop its dimension, it keeps it with one sample.

        The name of an indexer is a sample dimension or a coordinate along one
        (`treatment`, `sex`, the dose group of the individuals, as the readers
        of `pkpdutils.io` build them); its value is a label, a list of labels
        or a `slice` of labels, whose bounds are both included, as in
        `xarray.Dataset.sel`. The selected samples keep the order of the
        batch. A sample dimension without labels is selected by integer
        position instead, where a `slice` is the usual python slice with an
        exclusive stop.

        A list names the samples the caller expects, so every label of it has
        to be in the batch: a list holding a label which no sample carries
        raises and names it, rather than quietly returning the samples of the
        other labels. A `slice` is a range and is not checked that way.

        Args:
            **indexers: label, list of labels or slice per sample dimension or
                coordinate along one.

        Returns:
            The sub-batch.

        Raises:
            ValueError: if a name is neither a sample dimension nor a
                coordinate along one, if a label of a list is not in the
                batch, or if no sample of the batch matches.
        """
        ds = self.ds
        for name, value in indexers.items():
            dim = self._coordinate_dim(name)
            if name == dim and name not in ds.coords:
                # a sample dimension without labels selects by integer position
                selector = (
                    value if isinstance(value, slice | list | np.ndarray) else [value]
                )
                ds = ds.isel({dim: selector})
            else:
                # the labels of the dimension itself and the values of a
                # coordinate along it are matched the same way, so that a
                # label which no sample carries is a `ValueError` naming it
                # and not a `KeyError` of the index
                labels = ds[name].to_numpy()
                missing = _missing_labels(labels, value)
                if missing:
                    raise ValueError(
                        f"no sample of the batch has {name} = "
                        + ", ".join(repr(label) for label in missing)
                    )
                mask = _label_mask(labels, value)
                ds = ds.isel({dim: np.flatnonzero(mask)})
            if ds.sizes[dim] == 0:
                raise ValueError(f"no sample of the batch has {name} = {value!r}")
        return Timecourses(ds)

    def groupby(self, coord: str) -> Iterator[tuple[Any, "Timecourses"]]:
        """Iterate over the groups of a coordinate as sub-batches.

        The groups come in the order of their first appearance along the
        dimension of the coordinate, so that a study keeps the order of its
        table; every group is a `Timecourses` with the same layout as the
        batch.

        Args:
            coord: a sample dimension or a coordinate along one, e.g. the dose
                group or the treatment of the individuals.

        Yields:
            The value of the coordinate and the sub-batch of the samples
            carrying it.

        Raises:
            ValueError: if `coord` is neither a sample dimension nor a
                coordinate along one.
        """
        dim = self._coordinate_dim(coord)
        if coord not in self.ds.coords:
            raise ValueError(f"dimension '{coord}' of the batch has no labels")
        labels = self.ds[coord].to_numpy()
        for value in pd.unique(labels):
            positions = np.flatnonzero(labels == value)
            yield value, Timecourses(self.ds.isel({dim: positions}))

    def mean(
        self,
        dim: str,
        *,
        spread: Literal["sd", "se"] = "sd",
        min_n: int = 1,
    ) -> "Timecourses":
        r"""The mean curve over one sample dimension, with its spread and count.

        The group curve a publication reports: at every time point the
        arithmetic mean \(\bar c_j\) of the samples with a finite value there,
        their standard deviation \(s_j\) (\(n_j - 1\) degrees of freedom) and
        the standard error \(s_j / \sqrt{n_j}\); a point covered by fewer than
        `min_n` samples is `NaN`.

        `n` is the count \(n_j\) of its own time point, not one number for the
        curve, so that \(\mathrm{se}_j = s_j/\sqrt{n_j}\) holds at every point
        of a ragged group as well, where the late points carry fewer subjects
        than the early ones. `Timecourses.n_subjects` is the number of
        subjects of the group, the largest of the counts.

        The group curve carries `sd` and `se`: the standard deviation is the
        scatter of the samples at the point and the standard error follows
        from it through the count of the point. `spread` names the statistic
        which is computed from the curves and is kept for the symmetry with
        `plot_mean_timecourse`; since `n` is the count of the point itself,
        the two statistics imply each other and the result is the same either
        way.

        The samples need a shared sampling grid; a ragged batch is placed on
        the union of the grids of its samples first, with `NaN` where a sample
        has no point at the time of another. An existing `sd`, `se` or `n` of
        the samples is not propagated: the spread of the group curve is the
        scatter of the curves which were reduced.

        The dosing protocol of the group is the protocol of its samples when
        they share one, and the protocol of the first sample with a warning
        when they do not; `relative_to_dose` aligns the samples beforehand
        when they were dosed at different times.

        Args:
            dim: the sample dimension to reduce.
            spread: the statistic which is computed from the curves, the other
                one is derived from it through `n`; both give the same pair.
            min_n: fewest samples a time point must be covered by.

        Returns:
            The batch of group curves over the remaining sample dimensions.

        Raises:
            ValueError: if `dim` is not a sample dimension or `min_n` is not
                positive.
        """
        if dim not in self.sample_dims:
            raise ValueError(f"'{dim}' is not a sample dimension {self.sample_dims}")
        if min_n < 1:
            raise ValueError(f"'min_n' must be positive, got {min_n}")
        if spread not in ("sd", "se"):
            raise ValueError(f"'spread' must be 'sd' or 'se', got '{spread}'")
        batch = self
        if TIMES_VAR in self.ds:
            batch = Timecourses(
                self._shifted_times(np.zeros(self.sample_shape), uniform=False)
            )
        rest = tuple(d for d in batch.sample_dims if d != dim)
        values = batch.ds["value"].transpose(dim, *rest, TIME_DIM).to_numpy()
        finite = np.isfinite(values)
        count = finite.sum(axis=0).astype(np.float64)
        enough = count >= min_n
        with (
            np.errstate(invalid="ignore", divide="ignore"),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.where(enough, np.nanmean(values, axis=0), np.nan)
            scatter = np.where(
                enough & (count > 1), np.nanstd(values, axis=0, ddof=1), np.nan
            )
            # `n` is the count of the point, so the two statistics imply each
            # other wherever the count is the same
            root_n = np.sqrt(count)
            if spread == "sd":
                sd = scatter
                se = sd / root_n
            else:
                se = scatter / root_n
                sd = se * root_n

        unit = batch.unit
        data_vars: dict[str, Any] = {
            "value": ((*rest, TIME_DIM), mean, {"units": unit}),
            "sd": ((*rest, TIME_DIM), sd, {"units": unit}),
            "se": ((*rest, TIME_DIM), se, {"units": unit}),
            "n": _count_layout(rest, count),
        }
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                TIME_DIM: batch.ds[TIME_DIM],
                **{
                    str(name): coord
                    for name, coord in batch.ds.coords.items()
                    if str(name) not in (TIME_DIM, dim)
                    and dim not in tuple(str(d) for d in coord.dims)
                    and DOSE_DIM not in tuple(str(d) for d in coord.dims)
                },
            },
            attrs=dict(batch.ds.attrs),
        )
        dose = batch._group_dosing(dim)
        if dose is not None:
            ds = ds.assign(dose)
            ds = ds.assign_coords({DOSE_DIM: batch.ds[DOSE_DIM]})
        return Timecourses(ds)

    def dose_normalized(
        self, reference: float | Quantity | None = None
    ) -> "Timecourses":
        r"""The values divided by the dose, for the overlay of several dose levels.

        Dose normalization removes the dose from the curves of a dose
        escalation: with linear kinetics the normalized curves
        \(c(t) / D\) of every dose level fall on top of each other, and a
        deviation from that overlay is the figure of a dose dependency.
        Every sample is divided by the amount of its first dose, or by
        `reference` when one is given, and `sd` and `se` are divided with it;
        the unit of the values becomes `unit / dose_unit`, simplified by pint
        (`"nanogram / milliliter"` per `"milligram"` gives
        `"nanogram / milligram / milliliter"`). The doses themselves are kept,
        so that a figure still draws them, and a sample without a dose amount
        becomes `NaN`.

        Args:
            reference: the amount every sample is divided by, as a number in
                the dose unit of the batch or as a pint quantity converted to
                it; `None` divides every sample by its own first dose.

        Returns:
            The normalized batch.

        Raises:
            ValueError: without doses, if `reference` is not positive or
                carries a unit which is not a dose unit of the batch.
        """
        amounts = self.first_dose_amount
        dose_unit = self.dose_unit
        if amounts is None or dose_unit is None:
            raise ValueError("The batch carries no doses to normalize with")
        n_doses = self.n_doses
        if n_doses is not None and int(np.nanmax(n_doses)) > 1:
            logger.warning(
                "the protocol of a sample has more than one dose; the values are "
                "normalized with the first dose of every sample"
            )
        if reference is None:
            divisor = np.asarray(amounts, dtype=np.float64)
        else:
            if isinstance(reference, Quantity):
                reference = float(reference.to(dose_unit).magnitude)
            if not float(reference) > 0:
                raise ValueError(f"'reference' must be positive, got {reference}")
            divisor = np.full(self.sample_shape, float(reference))
        unit = str((Q_(1.0, self.unit) / Q_(1.0, dose_unit)).units)
        ds = self.ds.copy()
        factor = xr.DataArray(divisor, dims=self.sample_dims)
        for name in ("value", "sd", "se"):
            if name in ds:
                attrs = dict(ds[name].attrs)
                attrs["units"] = unit
                ds[name] = ds[name] / factor
                ds[name].attrs.update(attrs)
        ds.attrs["unit"] = unit
        return Timecourses(ds)

    def _group_dosing(self, dim: str) -> dict[str, xr.DataArray] | None:
        """The dose variables of a group curve reduced over one sample dimension.

        Args:
            dim: the reduced sample dimension.

        Returns:
            `dose_amount`, `dose_time` and `dose_duration` of the group,
            `None` for a batch without doses; the protocol of the first
            sample, with a warning, when the samples of a group do not share
            one protocol.
        """
        if not self.has_dose:
            return None
        names = ("dose_amount", "dose_time", "dose_duration")
        first = {name: self.ds[name].isel({dim: 0}, drop=True) for name in names}
        shared = all(
            bool(
                np.all(
                    np.isclose(
                        rows := self.ds[name].transpose(dim, ...).to_numpy(),
                        rows[:1],
                        equal_nan=True,
                    )
                )
            )
            for name in names
        )
        if not shared:
            logger.warning(
                "the samples of the group differ in their dosing protocol; the "
                "mean curve carries the protocol of the first sample along '%s'",
                dim,
            )
        return first

    def dosing_of(self, **indexers: Any) -> Dosing | None:
        """The dosing protocol of one sample, selected by coordinate label.

        Args:
            **indexers: one label per sample dimension, as for `sel`.

        Returns:
            The protocol, `None` without doses or for a sample combination
            which is not in the batch.

        Raises:
            ValueError: without a label for every sample dimension.
        """
        missing = set(self.sample_dims) - set(indexers)
        if missing:
            raise ValueError(
                f"dosing_of needs a label for every sample dimension, missing {sorted(missing)}"
            )
        return self._dosing_at(
            self._sample_arrays(), self._position(indexers, by_label=True)
        )

    def isel(self, **indexers: int) -> Timecourse:
        """One timecourse by integer position on every sample dimension."""
        missing = set(self.sample_dims) - set(indexers)
        if missing:
            raise ValueError(
                f"isel needs an index for every sample dimension, missing {sorted(missing)}"
            )
        arrays = self._sample_arrays()
        index = self._position(indexers, by_label=False)
        return self._timecourse_at(arrays, index, self._label_at(arrays, index))

    def sel(self, **indexers: Any) -> Timecourse:
        """One timecourse by coordinate label on every sample dimension."""
        missing = set(self.sample_dims) - set(indexers)
        if missing:
            raise ValueError(
                f"sel needs a label for every sample dimension, missing {sorted(missing)}"
            )
        arrays = self._sample_arrays()
        index = self._position(indexers, by_label=True)
        return self._timecourse_at(arrays, index, self._label_at(arrays, index))

    def __iter__(self) -> Iterator[Timecourse]:
        """Iterate over the timecourses in C order of the sample dimensions."""
        arrays = self._sample_arrays()
        for index in np.ndindex(*self.sample_shape):
            position = tuple(int(i) for i in index)
            yield self._timecourse_at(
                arrays, position, self._label_at(arrays, position)
            )

    def to_dataframe(self) -> pd.DataFrame:
        """The batch as a long data frame: the sample coordinates, `time`, `value` and the optional columns.

        One row per sample and time point. The limit of quantification of a
        sample, which is one number per sample, is repeated in every row of it
        (`from_dataframe(lloq="lloq")` reads it back). The dose variables are
        not part of the frame: they live over the dose dimension, not over the
        time dimension, and there is no one dose per row; `to_events` writes
        the dosing protocol as its own rows.

        Returns:
            The long data frame.
        """
        names = ["value", *[v for v in ("sd", "se", "n") if v in self.ds]]
        dim_order = [*self.sample_dims, TIME_DIM]
        # a coordinate along the sample dimensions is a column of the frame of
        # `xarray` already, a data variable has to be selected
        has_lloq = self.lloq is not None
        in_coords = LLOQ_VAR in self.ds.coords
        sub = self.ds[[*names, LLOQ_VAR] if has_lloq and not in_coords else names]
        if has_lloq:
            names = [*names, LLOQ_VAR]
        if DOSE_DIM in sub.dims and DOSE_DIM not in self.sample_dims:
            sub = sub.drop_dims(DOSE_DIM)
        df = sub.to_dataframe(dim_order=dim_order).reset_index()
        if TIMES_VAR in self.ds:
            # the rows are in C order of `dim_order`, as are the padded times;
            # the padding of a shorter grid is not a row of the frame, so that
            # `from_dataframe` reads the frame back (a `NaN` time is not a time)
            df[TIME_DIM] = self.times.ravel()
            df = df[df[TIME_DIM].notna()].reset_index(drop=True)
        return df[[*dim_order, *names]]

    # --- exchange formats ---------------------------------------------------
    #
    # `pkpdutils.io` imports this module, so it is imported inside the methods

    @classmethod
    def from_events(cls, df: pd.DataFrame, **kwargs: Any) -> "Timecourses":
        """Read a batch from event records, `pkpdutils.io.read_events`.

        Args:
            df: the event table, one row per dose or observation
            **kwargs: the arguments of `pkpdutils.io.read_events`

        Returns:
            The batch.
        """
        from pkpdutils.io import read_events

        return read_events(df, **kwargs)

    def to_events(self, **kwargs: Any) -> pd.DataFrame:
        """Write the batch as event records, `pkpdutils.io.write_events`.

        Args:
            **kwargs: the arguments of `pkpdutils.io.write_events`

        Returns:
            The event table.
        """
        from pkpdutils.io import write_events

        return write_events(self, **kwargs)

    @classmethod
    def from_pknca(
        cls, conc: pd.DataFrame, dose: pd.DataFrame, **kwargs: Any
    ) -> "Timecourses":
        """Read a batch from the two tables of `PKNCA`, `pkpdutils.io.read_pknca`.

        Args:
            conc: the concentration table
            dose: the dose table
            **kwargs: the arguments of `pkpdutils.io.read_pknca`

        Returns:
            The batch.
        """
        from pkpdutils.io import read_pknca

        return read_pknca(conc, dose, **kwargs)

    def to_pknca(self, *args: Any, **kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Write the batch as the two tables of `PKNCA`, `pkpdutils.io.write_pknca`.

        Args:
            *args: the paths of `pkpdutils.io.write_pknca`
            **kwargs: its keyword arguments

        Returns:
            The concentration table and the dose table.
        """
        from pkpdutils.io import write_pknca

        return write_pknca(self, *args, **kwargs)

    @classmethod
    def from_adnca(cls, df: pd.DataFrame, **kwargs: Any) -> "Timecourses":
        """Read a batch from a CDISC ADaM ADNCA dataset, `pkpdutils.io.read_adnca`.

        Args:
            df: the ADNCA dataset
            **kwargs: the arguments of `pkpdutils.io.read_adnca`

        Returns:
            The batch.
        """
        from pkpdutils.io import read_adnca

        return read_adnca(df, **kwargs)

    def to_adnca(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Write the batch as a CDISC ADaM ADNCA dataset, `pkpdutils.io.write_adnca`.

        Args:
            *args: the path of `pkpdutils.io.write_adnca`
            **kwargs: its keyword arguments

        Returns:
            The dataset.
        """
        from pkpdutils.io import write_adnca

        return write_adnca(self, *args, **kwargs)
