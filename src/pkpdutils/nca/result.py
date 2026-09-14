"""Units of the parameters and the result container of the NCA.

An `NCAResult` wraps an `xarray.Dataset` with one variable per parameter over
the sample dimensions of the analysed batch, `attrs["units"]` on every
variable and the integer variable `flags` (`pkpdutils.nca.options.NCAFlag`).
The units are derived from the units of the input with pint: an area carries
`unit * time_unit`, a rate `1 / time_unit`, a clearance `dose_unit / (unit *
time_unit)` converted to `liter / hour` (or per kilogram), a volume converted
to `liter` (or per kilogram), see `pkpdutils.units`.
"""

from typing import Any

import pandas as pd
import xarray as xr

from pkpdutils.nca.options import NCAFlag, decode_flags
from pkpdutils.nca.uncertainty import base_name
from pkpdutils.units import Q_, Quantity, normalize_clearance, normalize_volume, ureg


def parameter_unit(
    expression: str, *, unit: str, time_unit: str, dose_unit: str | None
) -> tuple[str, float]:
    """Unit of a parameter and the factor from its raw unit to the reported unit.

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


class NCAResult:
    """Parameters of a non-compartmental analysis as an `xarray.Dataset`.

    The dataset has one variable per parameter over the sample dimensions of
    the analysed `Timecourses`, `attrs["units"]` on every variable and the
    integer variable `flags`. A parameter which does not apply to a sample
    (no dose, no terminal phase, ...) is `NaN`. The analysis of group curves
    adds the derived variables of the uncertainty (`x_se`, `x_ci_low`, ...,
    see `pkpdutils.nca.uncertainty`) and the number of subjects `n`;
    `parameters` lists the parameters themselves, `derived_variables` the
    derived ones.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap a result dataset.

        Args:
            ds: dataset with a `flags` variable and `attrs["units"]` on every
                data variable.

        Raises:
            ValueError: without a `flags` variable or without units on a variable.
        """
        if "flags" not in ds:
            raise ValueError("The dataset needs a 'flags' variable")
        for name in ds.data_vars:
            if "units" not in ds[name].attrs:
                raise ValueError(f"'{name}' needs attrs['units']")
        self.ds: xr.Dataset = ds

    @property
    def sample_dims(self) -> tuple[str, ...]:
        """The sample dimensions."""
        return tuple(str(d) for d in self.ds["flags"].dims)

    @property
    def parameters(self) -> list[str]:
        """Names of the parameters (the data variables except `flags`, `n` and the derived variables)."""
        return [
            str(name)
            for name in self.ds.data_vars
            if name not in ("flags", "n") and base_name(str(name)) is None
        ]

    @property
    def derived_variables(self) -> list[str]:
        """Names of the uncertainty and summary variables (`x_se`, `x_ci_low`, ...)."""
        return [
            str(name) for name in self.ds.data_vars if base_name(str(name)) is not None
        ]

    @property
    def has_uncertainty(self) -> bool:
        """Whether any parameter carries a standard error."""
        return any(name.endswith("_se") for name in self.derived_variables)

    @property
    def _variables(self) -> list[str]:
        """Names of the data variables except `flags`, in the order of the dataset."""
        return [str(name) for name in self.ds.data_vars if name != "flags"]

    def units(self, name: str) -> str:
        """Unit string of a parameter.

        Args:
            name: name of the parameter.

        Returns:
            The unit string of the parameter.
        """
        return str(self.ds[name].attrs["units"])

    def __getitem__(self, name: str) -> xr.DataArray:
        """The array of a parameter.

        Args:
            name: name of the parameter.

        Returns:
            The data array of the parameter.
        """
        return self.ds[name]

    def __contains__(self, name: object) -> bool:
        """Whether a parameter is part of the result.

        Args:
            name: name to check.

        Returns:
            `True` if `name` is a data variable of the result.
        """
        return name in self.ds.data_vars

    def _sample(self, indexers: dict[str, Any]) -> xr.Dataset:
        """Select one sample of the dataset.

        Args:
            indexers: coordinate label per sample dimension.

        Returns:
            The dataset reduced to one sample.

        Raises:
            ValueError: if a sample dimension has no indexer.
        """
        missing = set(self.sample_dims) - set(indexers)
        if missing:
            raise ValueError(
                f"A label for every sample dimension is needed, missing {sorted(missing)}"
            )
        return self.ds.sel(indexers)

    def to_quantities(self, **indexers: Any) -> dict[str, Quantity]:
        """The variables of one sample as pint quantities.

        Args:
            **indexers: coordinate label per sample dimension.

        Returns:
            Variable name to quantity, for the parameters, the derived
            variables and `n` (every data variable except `flags`).
        """
        sample = self._sample(indexers)
        return {
            name: Q_(float(sample[name].values), self.units(name))
            for name in self._variables
        }

    def flags(self, **indexers: Any) -> list[str]:
        """Names of the flags set for one sample.

        Args:
            **indexers: coordinate label per sample dimension.

        Returns:
            The names of the flags set for the sample.
        """
        sample = self._sample(indexers)
        return decode_flags(int(sample["flags"].values))

    def to_dataframe(self) -> pd.DataFrame:
        """One row per sample: the sample coordinates, every variable and the decoded flags.

        `xarray.Dataset.to_dataframe` refuses a 0-dimensional dataset (no
        sample dimensions), so that case is built as a single-row frame
        directly.

        Returns:
            The dataframe.
        """
        if not self.sample_dims:
            row: dict[str, Any] = {
                name: float(self.ds[name].values) for name in self._variables
            }
            row["flags"] = "|".join(decode_flags(int(self.ds["flags"].values)))
            return pd.DataFrame([row])
        df = self.ds.to_dataframe().reset_index()
        df["flags"] = ["|".join(decode_flags(int(v))) for v in df["flags"]]
        columns = [*self.sample_dims, *self._variables, "flags"]
        return df[columns]

    def flag_table(self) -> pd.DataFrame:
        """One row per sample with a boolean column per flag.

        Returns:
            The dataframe.
        """
        if not self.sample_dims:
            value = int(self.ds["flags"].values)
            row = {
                flag.name: (value & flag.value) != 0 for flag in NCAFlag if flag.value
            }
            return pd.DataFrame([row])
        df = self.ds[["flags"]].to_dataframe().reset_index()
        df = df[[*self.sample_dims, "flags"]]
        values = df["flags"].to_numpy().astype(int)
        for flag in NCAFlag:
            if flag.value:
                df[flag.name] = (values & flag.value) != 0
        return df.drop(columns=["flags"])
