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

from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pkpdutils.units import Q_, Quantity, check_dose_unit, is_per_bodyweight


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
