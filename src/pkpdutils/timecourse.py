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
from collections.abc import Iterator, Mapping, Sequence
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
    """

    IV_BOLUS = "iv_bolus"
    IV_INFUSION = "iv_infusion"
    ORAL = "oral"

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
            required for `Route.IV_INFUSION`, not allowed otherwise
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
            if self.duration is None or self.duration <= 0:
                raise ValueError("An infusion needs a positive 'duration'")
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
            every value positive for `Route.IV_INFUSION`, not allowed
            otherwise
        unit: unit of the amounts, see `pkpdutils.units.check_dose_unit`
        route: route of administration, shared by every dose of the protocol
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    amounts: Any
    times: Any
    durations: Any = None
    unit: str
    route: Route = Route.ORAL

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """Convert the arrays, sort by time and check the doses against the route.

        Returns:
            The validated protocol.

        Raises:
            ValueError: if `amounts`, `times` or `durations` mismatch in
                length, if there is no dose, if an amount is negative, if the
                dose times contain duplicates, or if `durations` does not fit
                `route`.
        """
        amounts = _as_float_array("amounts", self.amounts)
        times = _as_float_array("times", self.times)
        if amounts.size != times.size:
            raise ValueError(
                f"'amounts' has length {amounts.size}, 'times' has length {times.size}"
            )
        if amounts.size == 0:
            raise ValueError("A dosing protocol needs at least one dose")
        if (amounts < 0).any():
            raise ValueError("'amounts' must be non-negative")
        check_dose_unit(self.unit)

        durations: np.ndarray | None = None
        if self.durations is not None:
            durations = _as_float_array("durations", self.durations)
            if durations.size != times.size:
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
            if durations is None or (durations <= 0).any():
                raise ValueError(
                    "An infusion needs a positive 'duration' for every dose"
                )
        elif durations is not None:
            if np.isnan(durations).all():
                durations = None
            else:
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
        """First dose of the protocol, `None` without `dosing`."""
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


def _dose_of_group(
    g: pd.DataFrame,
    dose_amount: str | None,
    dose_unit: str | None,
    dose_time: str | None,
    route: Route | None,
) -> Dose | None:
    """The dose of the rows of one sample of a long data frame.

    Args:
        g: the rows of one sample.
        dose_amount: name of the dose column, `None` for no dose.
        dose_unit: unit of the doses, required with `dose_amount`.
        dose_time: name of the dose time column, 0 by default.
        route: route of the doses, required with `dose_amount`.

    Returns:
        The dose, or `None` when `dose_amount` is `None`.

    Raises:
        ValueError: if `dose_unit` or `route` is missing, or the dose (or its
            time) is not constant per sample.
    """
    if dose_amount is None:
        return None
    if dose_unit is None or route is None:
        raise ValueError("'dose_unit' and 'route' are required with 'dose_amount'")
    amounts = g[dose_amount].dropna().unique()
    if amounts.size != 1:
        raise ValueError(f"The dose must be constant per sample, found {amounts}")
    time = 0.0
    if dose_time is not None:
        times = g[dose_time].dropna().unique()
        if times.size != 1:
            raise ValueError(
                f"The dose time must be constant per sample, found {times}"
            )
        time = float(times[0])
    return Dose(amount=float(amounts[0]), unit=dose_unit, route=route, time=time)


class Timecourses:
    """A batch of timecourses as an `xarray.Dataset`.

    The dataset has the dimension `time` and any number of sample dimensions,
    e.g. `individual`, `group`, `study`, or the dimensions of a simulation
    scan. Its variables are

    - `value` over `(*sample_dims, time)`, the values; `NaN` marks missing points,
    - `sd`, `se` over the same dimensions and `n` over the sample dimensions,
      for group data (optional),
    - `dose_amount`, `dose_time`, `dose_duration` over the sample dimensions
      (optional; `dose_duration` is `NaN` without infusion),
    - the coordinate `time` with the shared sampling grid, or, when the samples
      have different sampling times, the variable `times` over
      `(*sample_dims, time)` padded with `NaN` and an integer coordinate `time`.

    Every variable carries its unit in `attrs["units"]`; the dataset carries
    `substance`, `time_unit` and `unit` in its `attrs`, and `route` only when
    doses are present. The properties `times` and `values` return the
    `(*sample_shape, n_time)` arrays every analysis of the package works on;
    iteration and `sel`/`isel` give single `Timecourse` objects.

    A batch has one route: curves with different routes go into separate
    batches (a deliberate restriction of the 1.0.0 data model). `n` is one
    number per sample, not one per time point.

    Several sample dimensions span their cartesian product, which can have
    combinations without data (no curve was measured for them). Such a sample
    is all `NaN`; iteration and `sel`/`isel` return a `Timecourse` with `NaN`
    values and `dose=None` for it.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap a dataset, see the class documentation for its layout.

        Every variable with the `time` dimension (`value`, `sd`, `se`, `times`)
        is transposed to `(*sample_dims, time)`, so that the arrays and the data
        frame of the batch are built from one layout.

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
        if any(
            tuple(ds[name].dims) != tuple(d for d in layout if d in ds[name].dims)
            for name in ds.data_vars
        ):
            ds = ds.transpose(*layout, ...)
        if "units" not in ds["value"].attrs:
            raise ValueError("'value' needs attrs['units']")
        time_var = ds[TIMES_VAR] if TIMES_VAR in ds else ds[TIME_DIM]
        if "units" not in time_var.attrs:
            raise ValueError("the time coordinate needs attrs['units']")
        self.ds: xr.Dataset = ds

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

    @property
    def substance(self) -> str:
        """Name of the substance or effect."""
        return str(self.ds.attrs.get("substance", "substance"))

    @property
    def route(self) -> Route | None:
        """Route of the doses, `None` without dose information."""
        route = self.ds.attrs.get("route")
        return None if route is None else Route(route)

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

    def _optional(self, name: str) -> np.ndarray | None:
        """Return a variable as an array transposed to `(*sample_dims[, time])`, or `None` if absent.

        Args:
            name: name of the data variable.

        Returns:
            The array, or `None` when `name` is not a variable of the dataset.
        """
        if name not in self.ds:
            return None
        da = self.ds[name]
        if TIME_DIM in da.dims:
            return da.transpose(*self.sample_dims, TIME_DIM).to_numpy()
        return da.transpose(*self.sample_dims).to_numpy()

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
        """Number of subjects per sample, `None` without."""
        return self._optional("n")

    @property
    def dose_amount(self) -> np.ndarray | None:
        """Dose amounts per sample, `None` without doses."""
        return self._optional("dose_amount")

    @property
    def dose_time(self) -> np.ndarray | None:
        """Dose times per sample, `None` without doses."""
        return self._optional("dose_time")

    @property
    def dose_duration(self) -> np.ndarray | None:
        """Infusion durations per sample (`NaN` without infusion), `None` without doses."""
        return self._optional("dose_duration")

    @property
    def dose_unit(self) -> str | None:
        """Unit of the doses, `None` without doses."""
        if not self.has_dose:
            return None
        return str(self.ds["dose_amount"].attrs["units"])

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
        dose: Dose | Mapping[str, Any] | None = None,
        route: Route | None = None,
        substance: str = "substance",
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
            n: number of subjects, one number or an array of shape `sample_shape`
            dose: one `Dose` for all samples, or a mapping with `amount`
                (array of shape `sample_shape`), `unit`, and optionally `time`
                and `duration` arrays; the route is then given by `route`
            route: route of the doses when `dose` is a mapping
            substance: name of the substance or effect

        Returns:
            The batch.

        Raises:
            ValueError: if the shapes do not fit.
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
        if n is not None:
            n_arr = np.broadcast_to(
                np.asarray(n, dtype=np.float64), sample_shape
            ).copy()
            data_vars["n"] = (dims, n_arr, {"units": "dimensionless"})
            if "sd" in data_vars and "se" not in data_vars:
                data_vars["se"] = (
                    all_dims,
                    data_vars["sd"][1] / np.sqrt(n_arr)[..., None],
                    {"units": unit},
                )
            elif "se" in data_vars and "sd" not in data_vars:
                data_vars["sd"] = (
                    all_dims,
                    data_vars["se"][1] * np.sqrt(n_arr)[..., None],
                    {"units": unit},
                )

        attrs: dict[str, Any] = {
            "substance": substance,
            "time_unit": time_unit,
            "unit": unit,
        }
        if isinstance(dose, Dose):
            attrs["route"] = dose.route.value
            amount = np.full(sample_shape, dose.amount)
            dose_time = np.full(sample_shape, dose.time)
            duration = np.full(
                sample_shape, np.nan if dose.duration is None else dose.duration
            )
            data_vars["dose_amount"] = (dims, amount, {"units": dose.unit})
            data_vars["dose_time"] = (dims, dose_time, {"units": time_unit})
            data_vars["dose_duration"] = (dims, duration, {"units": time_unit})
        elif dose is not None:
            if route is None:
                raise ValueError("'route' is required when 'dose' is given as arrays")
            dose_unit = str(dose["unit"])
            check_dose_unit(dose_unit)
            attrs["route"] = route.value
            amount = np.broadcast_to(
                np.asarray(dose["amount"], dtype=np.float64), sample_shape
            ).copy()
            dose_time = np.broadcast_to(
                np.asarray(dose.get("time", 0.0), dtype=np.float64), sample_shape
            ).copy()
            duration = np.broadcast_to(
                np.asarray(dose.get("duration", np.nan), dtype=np.float64), sample_shape
            ).copy()
            data_vars["dose_amount"] = (dims, amount, {"units": dose_unit})
            data_vars["dose_time"] = (dims, dose_time, {"units": time_unit})
            data_vars["dose_duration"] = (dims, duration, {"units": time_unit})

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

        The timecourses must share `time_unit`, `unit`, `substance` and the route
        of their doses. If all sampling grids are equal the grid becomes the
        `time` coordinate, otherwise the times are stored per sample and shorter
        curves are padded with `NaN`.

        Either all or no curves carry a dose, and all doses need the same route
        and unit; curves with different routes go into separate batches. A batch
        keeps one `n` per sample: an `n` which varies over the time points of a
        curve is reduced to its maximum and logs a warning. `sd`, `se` and `n`
        are kept only when every curve carries them; a field which some curves
        are missing is dropped for the whole batch and logs a warning.

        Args:
            timecourses: the curves
            dim: name of the sample dimension
            labels: coordinate labels of the samples, the `label` of every
                timecourse (or its index when missing) by default

        Returns:
            The batch.

        Raises:
            ValueError: for an empty sequence, differing units, doses on some but
                not all curves, or doses with different routes or units.
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
            if tc.substance != first.substance:
                raise ValueError("All timecourses need the same substance")

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
            out = np.full((len(arrays), n_time), np.nan)
            for i, a in enumerate(arrays):
                assert a is not None
                out[i, : a.size] = a
            return out

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
            maxima: list[float] = []
            for label, v in zip(labels, n_values, strict=True):
                assert v is not None
                maximum = float(np.nanmax(v))
                if np.ndim(v) > 0 and float(np.nanmin(v)) != maximum:
                    logger.warning(
                        "'n' varies over the time points of '%s', the batch keeps "
                        "one number per sample, the maximum %s",
                        label,
                        maximum,
                    )
                maxima.append(maximum)
            n = np.array(maxima)

        doses = [tc.dose for tc in timecourses]
        without_dose = [
            label for label, d in zip(labels, doses, strict=True) if d is None
        ]
        if without_dose and len(without_dose) != len(doses):
            raise ValueError(
                "Either all or no timecourses need a dose, there is no dose for "
                f"{[str(label) for label in without_dose]}"
            )
        dose: Mapping[str, Any] | None = None
        route: Route | None = None
        if all(d is not None for d in doses):
            routes = {d.route for d in doses if d is not None}
            units = {d.unit for d in doses if d is not None}
            if len(routes) != 1:
                raise ValueError(
                    "A batch has one route, found "
                    f"{sorted(r.value for r in routes)}; build separate batches, "
                    "one per route"
                )
            if len(units) != 1:
                raise ValueError(f"All doses need the same unit, found {sorted(units)}")
            route = routes.pop()
            dose = {
                "amount": np.array([d.amount for d in doses if d is not None]),
                "unit": units.pop(),
                "time": np.array([d.time for d in doses if d is not None]),
                "duration": np.array(
                    [
                        np.nan if d.duration is None else d.duration
                        for d in doses
                        if d is not None
                    ]
                ),
            }

        return cls.from_arrays(
            time,
            values,
            time_unit=first.time_unit,
            unit=first.unit,
            dims=(dim,),
            coords={dim: list(labels)},
            sd=sd,
            se=se,
            n=n,
            dose=dose,
            route=route,
            substance=first.substance,
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
        dose_amount: str | None = None,
        dose_unit: str | None = None,
        dose_time: str | None = None,
        route: Route | None = None,
        substance: str = "substance",
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
            n: name of the column with the number of subjects (constant per sample)
            dose_amount: name of the dose column (constant per sample)
            dose_unit: unit of the doses, required with `dose_amount`
            dose_time: name of the dose time column, 0 by default
            route: route of the doses, required with `dose_amount`
            substance: name of the substance or effect

        Returns:
            The batch.

        Raises:
            ValueError: if `sample` is empty.
        """
        sample = list(sample)
        if not sample:
            raise ValueError("'sample' needs at least one column")
        # a single column groups by the column itself (scalar keys, no
        # deprecation warning); several columns need the list form (tuple keys)
        groups = df.groupby(
            sample[0] if len(sample) == 1 else sample, sort=True, dropna=False
        )
        keys = list(groups.groups)
        # from_timecourses decides between a shared grid and per sample grids
        timecourses = [
            Timecourse.from_dataframe(
                g.sort_values(time),
                time_unit=time_unit,
                unit=unit,
                time=time,
                value=value,
                sd=sd,
                se=se,
                n=n,
                substance=substance,
                label=str(key),
                dose=_dose_of_group(g, dose_amount, dose_unit, dose_time, route),
            )
            for key, g in groups
        ]

        if len(sample) == 1:
            return cls.from_timecourses(timecourses, dim=sample[0], labels=list(keys))

        # several sample columns: build along one flat dimension, then unstack
        flat = cls.from_timecourses(
            timecourses, dim="_sample", labels=list(range(len(keys)))
        )
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
        return cls(ds.transpose(*sample, TIME_DIM))

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
        dose: Dose | None = None,
        substance: str | None = None,
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
            dose: one dose for all samples
            substance: name of the substance, `value` by default

        Returns:
            The batch with the scan dimensions as sample dimensions.

        Raises:
            ValueError: if `value` has no dimension `time_dim`, or the time
                values are not one dimensional.
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
            substance=value if substance is None else substance,
        )

    @classmethod
    def from_xresult(
        cls,
        xres: Any,
        selection: str,
        *,
        dose: Dose | None = None,
        substance: str | None = None,
    ) -> "Timecourses":
        """Create a batch from the result of an sbmlsim simulation.

        sbmlsim is not a dependency; an `XResult` is used by its attributes:
        `xres.xds` is the dataset with the `_time` dimension and `xres.uinfo`
        maps `selection` and `"time"` to unit strings.

        Args:
            xres: the `sbmlsim.result.XResult`
            selection: the variable of the result, e.g. `"[Cve_mid]"`
            dose: one dose for all samples
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

    def _timecourse(self, sample: xr.Dataset, label: Any) -> Timecourse:
        """Build the `Timecourse` of a dataset without sample dimensions.

        Args:
            sample: the dataset indexed down to a single sample.
            label: label of the sample, `None` without sample dimensions.

        Returns:
            The timecourse.
        """
        time = (
            sample[TIMES_VAR].to_numpy()
            if TIMES_VAR in sample
            else sample[TIME_DIM].to_numpy().astype(np.float64)
        )
        value = sample["value"].to_numpy()
        mask = ~np.isnan(time)
        data: dict[str, Any] = {
            "time": time[mask],
            "value": value[mask],
            "time_unit": self.time_unit,
            "unit": self.unit,
            "substance": self.substance,
            "label": None if label is None else str(label),
        }
        for name in ("sd", "se"):
            if name in sample:
                data[name] = sample[name].to_numpy()[mask]
        if "n" in sample:
            data["n"] = float(sample["n"].to_numpy())
        if self.has_dose:
            amount = float(sample["dose_amount"].to_numpy())
            if not np.isnan(amount):
                # a NaN amount marks a sample combination which is not in the
                # batch (its values are NaN as well), it has no dose
                duration = float(sample["dose_duration"].to_numpy())
                route = self.route
                assert route is not None
                data["dose"] = Dose(
                    amount=amount,
                    unit=self.dose_unit or "mg",
                    route=route,
                    time=float(sample["dose_time"].to_numpy()),
                    duration=None if np.isnan(duration) else duration,
                )
        return Timecourse(**data)

    def isel(self, **indexers: int) -> Timecourse:
        """One timecourse by integer position on every sample dimension."""
        missing = set(self.sample_dims) - set(indexers)
        if missing:
            raise ValueError(
                f"isel needs an index for every sample dimension, missing {sorted(missing)}"
            )
        sample = self.ds.isel(indexers)
        label = self._label(sample)
        return self._timecourse(sample, label)

    def sel(self, **indexers: Any) -> Timecourse:
        """One timecourse by coordinate label on every sample dimension."""
        missing = set(self.sample_dims) - set(indexers)
        if missing:
            raise ValueError(
                f"sel needs a label for every sample dimension, missing {sorted(missing)}"
            )
        sample = self.ds.sel(indexers)
        label = self._label(sample)
        return self._timecourse(sample, label)

    def _label(self, sample: xr.Dataset) -> Any:
        """Label of a selected sample: the coordinates of the sample dimensions joined by `|`."""
        parts = [str(sample[d].values) for d in self.sample_dims if d in sample.coords]
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else "|".join(parts)

    def __iter__(self) -> Iterator[Timecourse]:
        """Iterate over the timecourses in C order of the sample dimensions."""
        for index in np.ndindex(*self.sample_shape):
            yield self.isel(
                **dict(zip(self.sample_dims, (int(i) for i in index), strict=True))
            )

    def to_dataframe(self) -> pd.DataFrame:
        """The batch as a long data frame: the sample coordinates, `time`, `value` and the optional columns."""
        names = ["value", *[v for v in ("sd", "se", "n") if v in self.ds]]
        dim_order = [*self.sample_dims, TIME_DIM]
        df = self.ds[names].to_dataframe(dim_order=dim_order).reset_index()
        if TIMES_VAR in self.ds:
            # the rows are in C order of `dim_order`, as are the padded times
            df[TIME_DIM] = self.times.ravel()
        return df[[*dim_order, *names]]
