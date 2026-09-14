"""Units of the parameters and the result container of the NCA.

An `NCAResult` wraps an `xarray.Dataset` with one variable per parameter over
the sample dimensions of the analysed batch, `attrs["units"]` on every
variable and the integer variable `flags` (`pkpdutils.nca.options.NCAFlag`).
The units are derived from the units of the input with pint: an area carries
`unit * time_unit`, a rate `1 / time_unit`, a clearance `dose_unit / (unit *
time_unit)` converted to `liter / hour` (or per kilogram), a volume converted
to `liter` (or per kilogram), see `pkpdutils.units`.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import t as student_t

from pkpdutils.nca.options import NCAFlag, decode_flags
from pkpdutils.nca.uncertainty import LOGNORMAL_PARAMETERS, base_name
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

    def summarize(self, dim: str, ci_level: float = 0.95) -> "NCAResult":
        """Summarize the parameters of individual samples over one sample dimension.

        For every parameter `x` the summary carries the arithmetic mean `x`,
        `x_sd`, `x_se`, the t-based confidence interval `x_ci_low`/`x_ci_high`
        at `ci_level`, `x_median`, `x_q25`, `x_q75`, the count of finite values
        `x_n` and, for log-normal parameters, `x_geomean` and `x_geocv`;
        `flags` is the union of the flags of the samples. Derived variables of
        the input are dropped.

        The two counts differ: `n` is the number of samples along `dim`,
        `x_n` the number of them at which `x` is finite, and every statistic of
        `x` uses `x_n` (`x_se = x_sd / sqrt(x_n)`, the interval uses
        `t` with `x_n - 1` degrees of freedom). A parameter which does not
        apply to every sample (no terminal phase, no dose) therefore has
        `x_n < n`.

        Args:
            dim: the sample dimension to reduce
            ci_level: level of the confidence interval of the mean

        Returns:
            The summary over the remaining sample dimensions.

        Raises:
            ValueError: if `dim` is not a sample dimension of the result.
        """
        if dim not in self.sample_dims:
            raise ValueError(f"'{dim}' is not a sample dimension {self.sample_dims}")
        alpha = 1.0 - ci_level
        data_vars: dict[str, Any] = {}
        for name in self.parameters:
            da = self.ds[name].transpose(..., dim)
            values = da.to_numpy().astype(np.float64)
            dims = tuple(str(d) for d in da.dims if d != dim)
            units = self.units(name)
            with (
                np.errstate(invalid="ignore", divide="ignore"),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", RuntimeWarning)
                finite = np.isfinite(values)
                count = finite.sum(axis=-1).astype(np.float64)
                filled = np.where(finite, values, np.nan)
                mean = np.nanmean(filled, axis=-1)
                sd = np.where(count > 1, np.nanstd(filled, axis=-1, ddof=1), np.nan)
                se = sd / np.sqrt(count)
                tq = student_t.ppf(1.0 - alpha / 2.0, np.maximum(count - 1.0, 1.0))
                low, high = mean - tq * se, mean + tq * se
                median = np.nanmedian(filled, axis=-1)
                q25, q75 = np.nanpercentile(filled, [25, 75], axis=-1)
                mean = np.where(count > 0, mean, np.nan)
                median = np.where(count > 0, median, np.nan)
            data_vars[name] = (dims, mean, {"units": units})
            data_vars[f"{name}_sd"] = (dims, sd, {"units": units})
            data_vars[f"{name}_se"] = (dims, se, {"units": units})
            data_vars[f"{name}_ci_low"] = (
                dims,
                np.where(count > 1, low, np.nan),
                {"units": units},
            )
            data_vars[f"{name}_ci_high"] = (
                dims,
                np.where(count > 1, high, np.nan),
                {"units": units},
            )
            data_vars[f"{name}_median"] = (dims, median, {"units": units})
            data_vars[f"{name}_q25"] = (
                dims,
                np.where(count > 0, q25, np.nan),
                {"units": units},
            )
            data_vars[f"{name}_q75"] = (
                dims,
                np.where(count > 0, q75, np.nan),
                {"units": units},
            )
            data_vars[f"{name}_n"] = (dims, count, {"units": "dimensionless"})
            if name in LOGNORMAL_PARAMETERS:
                with (
                    np.errstate(invalid="ignore", divide="ignore"),
                    warnings.catch_warnings(),
                ):
                    warnings.simplefilter("ignore", RuntimeWarning)
                    logs = np.where(
                        finite & (values > 0),
                        np.log(np.where(values > 0, values, 1.0)),
                        np.nan,
                    )
                    n_pos = np.isfinite(logs).sum(axis=-1)
                    geomean = np.where(
                        n_pos > 0, np.exp(np.nanmean(logs, axis=-1)), np.nan
                    )
                    geocv = np.where(
                        n_pos > 1,
                        np.sqrt(np.expm1(np.nanvar(logs, axis=-1, ddof=1))),
                        np.nan,
                    )
                data_vars[f"{name}_geomean"] = (dims, geomean, {"units": units})
                data_vars[f"{name}_geocv"] = (dims, geocv, {"units": "dimensionless"})
        flags = self.ds["flags"].transpose(..., dim)
        remaining = tuple(str(d) for d in flags.dims if d != dim)
        data_vars["n"] = (
            remaining,
            np.full(flags.shape[:-1], float(self.ds.sizes[dim])),
            {"units": "dimensionless"},
        )
        data_vars["flags"] = (
            remaining,
            np.bitwise_or.reduce(flags.to_numpy().astype(np.int64), axis=-1),
            {"units": "dimensionless"},
        )
        coords = {d: self.ds[d] for d in remaining if d in self.ds.coords}
        return NCAResult(
            xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(self.ds.attrs))
        )
