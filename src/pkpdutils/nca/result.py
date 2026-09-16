"""Units of the parameters and the result container of the NCA.

An `NCAResult` wraps an `xarray.Dataset` with one variable per parameter over
the sample dimensions of the analysed batch, `attrs["units"]` on every
variable and the integer variable `flags` (`pkpdutils.nca.options.NCAFlag`).
The units are derived from the units of the input with pint: an area carries
`unit * time_unit`, a rate `1 / time_unit`, a clearance `dose_unit / (unit *
time_unit)` converted to `liter / hour` (or per kilogram), a volume converted
to `liter` (or per kilogram), see `pkpdutils.units`.
"""

from functools import lru_cache

import pandas as pd

from pkpdutils.nca.intervals import INTERVAL_DIM, INTERVAL_PREFIX, INTERVAL_UNITS
from pkpdutils.nca.options import NCAFlag
from pkpdutils.nca.uncertainty import DISCRETE_PARAMETERS, LOGNORMAL_PARAMETERS
from pkpdutils.result import ParameterResult
from pkpdutils.units import (
    CACHE_SIZE,
    Q_,
    normalize_clearance,
    normalize_volume,
    ureg,
)


@lru_cache(maxsize=CACHE_SIZE)
def parameter_unit(
    expression: str, *, unit: str, time_unit: str, dose_unit: str | None
) -> tuple[str, float]:
    """Unit of a parameter and the factor from its raw unit to the reported unit.

    The unit of a parameter depends only on the four strings of the signature,
    of which an analysis has a handful (`PARAMETER_UNITS` holds 29 expressions),
    while the derivation costs several pint conversions; the result is therefore
    cached (`pkpdutils.units.CACHE_SIZE` entries).

    Args:
        expression: pint expression with the placeholders `{unit}`, `{time}`
            and `{dose}`, e.g. `"({unit}) * ({time})"`
        unit: unit of the values
        time_unit: unit of the times
        dose_unit: unit of the doses, `None` without doses (an expression
            with `{dose}` then raises `ValueError`)

    Returns:
        The canonical unit string and the factor a magnitude in the raw unit is
        multiplied with to be in the canonical unit (volumes to liter,
        clearances to liter per hour, else 1).

    Raises:
        ValueError: if `expression` uses `{dose}` without a `dose_unit`.
    """
    if "{dose}" in expression and dose_unit is None:
        raise ValueError(f"'{expression}' needs a dose unit")
    raw = expression.format(unit=unit, time=time_unit, dose=dose_unit or "")
    quantity = Q_(1.0, ureg.parse_units(raw))
    converted = normalize_clearance(normalize_volume(quantity))
    return str(converted.units), float(converted.magnitude)


class NCAResult(ParameterResult):
    """Parameters of a non-compartmental analysis as an `xarray.Dataset`.

    One variable per parameter over the sample dimensions of the analysed
    `Timecourses`, `attrs["units"]` on every variable, the uncertainty
    variables of `pkpdutils.nca.uncertainty` and the integer variable `flags`
    (`NCAFlag`). See `pkpdutils.result.ParameterResult` for the interface.
    """

    flag_type = NCAFlag
    lognormal_parameters = LOGNORMAL_PARAMETERS
    discrete_parameters = DISCRETE_PARAMETERS
    #: the per-interval parameters are point variables (the dimension
    #: `interval`) which `summarize` reduces over the sample dimension, so
    #: that a multiple dose study reports the mean trough per interval over
    #: its subjects
    summarized_point_variables = frozenset(INTERVAL_UNITS)

    @property
    def has_intervals(self) -> bool:
        """Whether the result carries the parameters of the single dosing intervals."""
        return INTERVAL_DIM in self.ds.dims and bool(self._interval_variables)

    @property
    def _interval_variables(self) -> list[str]:
        """Names of the per-interval variables, in the order of the dataset."""
        return [
            str(name)
            for name in self.ds.data_vars
            if str(name).startswith(INTERVAL_PREFIX)
        ]

    def intervals(self) -> pd.DataFrame:
        """The per-interval parameters as one row per sample and dosing interval.

        The interval variables carry the dimension `interval` beyond the sample
        dimensions and are therefore point variables, which `to_dataframe`
        leaves out; this frame reports them with the sample coordinates and the
        number of the interval.

        Returns:
            One row per sample and interval with the sample coordinates (the
            dimension coordinates and the coordinates along them, such as the
            weight of a subject), `interval` and every `interval_*` variable;
            an empty frame for a single dose result.
        """
        names = self._interval_variables
        if not self.has_intervals:
            return pd.DataFrame()
        df = self.ds[names].to_dataframe().reset_index()
        leading = [*self.sample_dims, INTERVAL_DIM]
        # the non-dimension coordinates along the sample dimensions travel with
        # the samples, as they do in the result itself
        coordinates = [
            column for column in df.columns if column not in (*leading, *names)
        ]
        return df[[*leading, *coordinates, *names]]
