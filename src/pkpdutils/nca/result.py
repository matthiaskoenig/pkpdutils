"""Units of the parameters and the result container of the NCA.

An `NCAResult` wraps an `xarray.Dataset` with one variable per parameter over
the sample dimensions of the analysed batch, `attrs["units"]` on every
variable and the integer variable `flags` (`pkpdutils.nca.options.NCAFlag`).
The units are derived from the units of the input with pint: an area carries
`unit * time_unit`, a rate `1 / time_unit`, a clearance `dose_unit / (unit *
time_unit)` converted to `liter / hour` (or per kilogram), a volume converted
to `liter` (or per kilogram), see `pkpdutils.units`.
"""

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

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

#: name of the coordinate carrying the dose amount of every sample of a result
DOSE_COORDINATE = "dose_amount"

#: suffix of a dose normalized variable (`NCAResult.dose_normalized`)
DOSE_NORMALIZED_SUFFIX = "_dn"

#: unit expressions of the parameters `NCAResult.dose_normalized` normalizes by
#: default: the concentrations and the exposures
DOSE_NORMALIZED_EXPRESSIONS: tuple[str, ...] = ("{unit}", "({unit}) * ({time})")

#: parameters whose dose normalized variable carries a name of its own, from
#: before the general rule existed
DOSE_NORMALIZED_NAMES: dict[str, str] = {"auc_inf_obs": "auc_inf_dn"}


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


@lru_cache(maxsize=CACHE_SIZE)
def _per_dose(unit: str, dose_unit: str) -> tuple[str, float]:
    """Unit of a parameter per dose and the factor to it.

    Args:
        unit: unit of the parameter, as the result reports it
        dose_unit: unit of the doses

    Returns:
        The canonical unit string of the parameter per dose and the factor a
        magnitude in the raw unit is multiplied with, as `parameter_unit`.
    """
    quantity = Q_(1.0, ureg.parse_units(f"({unit}) / ({dose_unit})"))
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
    #: the parameters the console table shows with one row per sample: the
    #: exposure, the peak, the terminal phase, clearance and volume, and the
    #: steady state parameters when the analysis has them
    console_parameters = (
        "cmax",
        "tmax",
        "auc_last",
        "auc_inf_obs",
        "auc_extrap_fraction",
        "thalf",
        "cl",
        "cl_f",
        "vz",
        "vz_f",
        "mrt",
        "auc_tau",
        "cmax_ss",
        "ctrough",
        "accumulation_ratio",
    )

    @property
    def dose(self) -> xr.DataArray | None:
        """The dose amount of every sample, `None` for a result without doses.

        The coordinate `dose_amount` the analysis carries over from the batch:
        the first dose of a single dose sample and the last dose of a multiple
        dose one, the dose its parameters are divided by.
        """
        if DOSE_COORDINATE not in self.ds.coords:
            return None
        return self.ds.coords[DOSE_COORDINATE]

    def dose_normalized_parameters(self) -> list[str]:
        """The parameters `dose_normalized` normalizes without being asked.

        Returns:
            The concentration and exposure parameters of the result, those
            whose unit expression is `{unit}` or `({unit}) * ({time})`
            (`DOSE_NORMALIZED_EXPRESSIONS`), in the order of the dataset;
            a parameter which is itself dose normalized is left out.
        """
        from pkpdutils.nca.nca import PARAMETER_UNITS

        return [
            name
            for name in self.parameters
            if not name.endswith(DOSE_NORMALIZED_SUFFIX)
            and PARAMETER_UNITS.get(name) in DOSE_NORMALIZED_EXPRESSIONS
        ]

    def dose_normalized(self, parameters: Sequence[str] | None = None) -> "NCAResult":
        r"""A copy of the result with the dose normalized variables of its parameters.

        The dose normalized variable of a parameter is the parameter divided by
        the dose of its sample,

        $$x_\mathrm{dn} = \frac{x}{D},$$

        with the unit of the parameter per dose unit; `NaN` where the sample
        has no positive dose (a placebo arm). It is the form ICH M13A (2024)
        asks for when strengths are compared, and the `*D` family of the CDISC
        codelist (Phoenix `AUClast_D`, `Cmax_D`; PKNCA `pk.calc.dn`).

        The variable is named `x_dn`, except for `auc_inf_obs`, whose
        normalized variable is the `auc_inf_dn` every analysis already reports
        (`DOSE_NORMALIZED_NAMES`). Normalize before summarizing: the summary of
        a dimension carries no dose coordinate any more.

        Args:
            parameters: the parameters to normalize, the concentrations and
                exposures of the result by default
                (`dose_normalized_parameters`).

        Returns:
            A copy of the result with one dose normalized variable per
            parameter added.

        Raises:
            ValueError: if the result carries no dose (an analysis of a batch
                without doses), or if a name is not a parameter of the result.
        """
        dose = self.dose
        if dose is None:
            raise ValueError(
                "the result carries no dose, so no parameter can be normalized "
                "by it; the analysed batch has no dose amounts"
            )
        names = list(
            self.dose_normalized_parameters() if parameters is None else parameters
        )
        missing = [name for name in names if name not in self.ds.data_vars]
        if missing:
            raise ValueError(f"{missing} are no parameters of the result")
        dose_unit = str(dose.attrs.get("units", ""))
        with np.errstate(divide="ignore", invalid="ignore"):
            amount = dose.where(dose > 0.0)
        ds = self.ds.copy()
        for name in names:
            unit, factor = _per_dose(self.units(name), dose_unit)
            normalized = (self.ds[name] / amount) * factor
            normalized.attrs = {"units": unit}
            ds[DOSE_NORMALIZED_NAMES.get(name, f"{name}{DOSE_NORMALIZED_SUFFIX}")] = (
                normalized
            )
        return NCAResult(ds)

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
