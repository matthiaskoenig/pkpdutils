"""Timecourses, doses and dosing regimens.

The data model of the package:

- `Timecourse` is one curve, i.e. values over time with units, an optional
  uncertainty (`sd`/`se` and `n` for group data), a `Dose` and metadata.
- `Timecourses` is a batch of curves as an `xarray.Dataset` with a `time`
  dimension and any number of sample dimensions (individuals, groups, studies,
  the dimensions of a simulation scan). Every analysis of the package works on
  a `Timecourses` object and returns an `xarray.Dataset` over the same sample
  dimensions.
- `DosingRegimen` describes repeated dosing for steady state analyses.

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
from enum import StrEnum
from typing import Any, Self

import numpy as np
import pandas as pd
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
        dose: the dose of the substance, `None` without dose information
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
    dose: Dose | None = None
    substance: str = "substance"
    label: str | None = None
    tissue: str | None = None

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

    @property
    def size(self) -> int:
        """Number of time points."""
        return int(self.time.size)

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

    def relative_to_dose(self) -> "Timecourse":
        """Copy with the time shifted so that the dose is given at time 0.

        Returns the timecourse itself when it has no dose or the dose is at 0.

        Returns:
            The shifted timecourse, or `self` when there is nothing to shift.
        """
        if self.dose is None or self.dose.time == 0.0:
            return self
        shift = self.dose.time
        return self.model_copy(
            update={
                "time": self.time - shift,
                "dose": self.dose.model_copy(update={"time": 0.0}),
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
