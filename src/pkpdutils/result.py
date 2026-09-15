"""Shared container of parameter results (`NCAResult`, `FitResult`): an `xarray.Dataset` over sample dimensions, units per variable, an integer `flags` variable, quantities, data frames and summaries."""

import warnings
from collections.abc import Iterable, Mapping, Sequence
from enum import IntFlag
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import t as student_t

from pkpdutils.units import Q_, Quantity

if TYPE_CHECKING:
    from pkpdutils.stats.sample import ParameterSample


def sample_coordinates(
    ds: xr.Dataset, sample_dims: Sequence[str]
) -> dict[str, xr.DataArray]:
    """The coordinates of a dataset which live on the sample dimensions.

    The dimension coordinates of the sample dimensions and every
    non-dimension coordinate along them (the `period` or the `sequence` of
    the individuals of a crossover study, the weight of the subjects) are
    carried from the analysed batch to its result, so that
    `ParameterResult.sample` finds them.

    Args:
        ds: the dataset of the batch.
        sample_dims: the sample dimensions.

    Returns:
        Coordinate name to coordinate.
    """
    dims = set(sample_dims)
    return {
        str(name): coord
        for name, coord in ds.coords.items()
        if {str(d) for d in coord.dims} <= dims
    }


def check_coordinate_collision(
    coords: Mapping[str, Any], variables: Iterable[str]
) -> None:
    """Raise if a coordinate of the batch shares its name with a result variable.

    `xr.Dataset` and `xr.DataArray` refuse a name that is both a coordinate
    and a data variable (`ValueError: variables {...} are found in both
    data_vars and coords`); a batch coordinate carried over by
    `sample_coordinates` (e.g. an individual attribute happening to be named
    `n` or after a parameter such as `cmax`) would otherwise only surface as
    that opaque error deep inside the construction of the result. Calling
    this first turns it into a clear message that names the batch coordinate
    to rename.

    Args:
        coords: the coordinates that are about to be attached to the result.
        variables: the names of the data variables of the result.

    Raises:
        ValueError: if a name is both a coordinate and a data variable.
    """
    clash = set(coords) & set(variables)
    if clash:
        raise ValueError(
            f"coordinate {sorted(clash)} of the batch collides with a result "
            "variable; rename the coordinate"
        )


#: suffixes of the uncertainty variables of a parameter (`_cv` is the
#: coefficient of variation of a fitted parameter, `pkpdutils.fit`)
UNCERTAINTY_SUFFIXES: tuple[str, ...] = (
    "_sd",
    "_se",
    "_ci_low",
    "_ci_high",
    "_pi_low",
    "_pi_high",
    "_geomean",
    # `_geocv` before `_cv`: `base_name` returns on the first match, so the
    # shorter suffix would turn `auc_geocv` into `auc_geo`
    "_geocv",
    "_cv",
)

#: suffixes of the summary variables of a parameter (`ParameterResult.summarize`)
SUMMARY_SUFFIXES: tuple[str, ...] = ("_median", "_q25", "_q75", "_n")


def base_name(name: str) -> str | None:
    """The parameter a derived variable belongs to, `None` for a parameter itself.

    Args:
        name: name of a result variable, e.g. `"auc_last_se"`.

    Returns:
        The name of the parameter the variable is derived from, `None` for a
        parameter.
    """
    for suffix in (*UNCERTAINTY_SUFFIXES, *SUMMARY_SUFFIXES):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return None


class ParameterResult:
    """Parameters of an analysis as an `xarray.Dataset`.

    The dataset has one variable per parameter over the sample dimensions, an
    `attrs["units"]` on every variable and the integer variable `flags`. A
    parameter which does not apply to a sample is `NaN`. The analysis of
    group curves adds the derived variables of the uncertainty (`x_se`,
    `x_ci_low`, ...) and the number of subjects `n`; `parameters` lists the
    parameters themselves, `derived_variables` the derived ones,
    `statistics` the statistics of the analysis itself (`statistic_variables`,
    e.g. the goodness of fit of a curve fit) and `point_variables` the
    variables carrying an extra, non-sample dimension (such as predicted
    curves).
    """

    #: the IntFlag type that decodes the `flags` variable
    flag_type: ClassVar[type[IntFlag]] = IntFlag
    #: parameters reported with geometric statistics in `summarize`
    lognormal_parameters: ClassVar[frozenset[str]] = frozenset()
    #: parameters without uncertainty variables
    discrete_parameters: ClassVar[frozenset[str]] = frozenset()
    #: variables describing the analysis of a sample rather than a parameter
    #: of it (the goodness of fit and the counts of a fit); they are no
    #: parameters and are not summarized over samples
    statistic_variables: ClassVar[frozenset[str]] = frozenset()

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
    def point_variables(self) -> list[str]:
        """Names of the data-point variables (an extra dimension beyond the sample dims)."""
        sample_dims = set(self.sample_dims)
        return [
            str(name)
            for name in self.ds.data_vars
            if not set(self.ds[name].dims) <= sample_dims
        ]

    @property
    def parameters(self) -> list[str]:
        """Names of the parameters (the data variables except `flags`, `n`, the statistics, the derived and the point variables)."""
        point = set(self.point_variables)
        return [
            str(name)
            for name in self.ds.data_vars
            if name not in ("flags", "n")
            and str(name) not in self.statistic_variables
            and base_name(str(name)) is None
            and name not in point
        ]

    @property
    def statistics(self) -> list[str]:
        """Names of the statistics of the analysis present in the result (`statistic_variables`)."""
        return [
            str(name)
            for name in self.ds.data_vars
            if str(name) in self.statistic_variables
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
        """Names of the scalar data variables except `flags`, in the order of the dataset."""
        point = set(self.point_variables)
        return [
            str(name)
            for name in self.ds.data_vars
            if name != "flags" and name not in point
        ]

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

    def decode_flags(self, value: int) -> list[str]:
        """Names of the flags set in an integer flag value, in bit order.

        Args:
            value: an integer combination of `flag_type` values.

        Returns:
            The names of the set flags, in the declaration order of `flag_type`.
        """
        names: list[str] = []
        for flag in self.flag_type:
            if flag.value and value & flag.value and flag.name is not None:
                names.append(str(flag.name))
        return names

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
        """The scalar variables of one sample as pint quantities.

        Args:
            **indexers: coordinate label per sample dimension.

        Returns:
            Variable name to quantity, for the parameters, the derived
            variables and `n` (every scalar data variable except `flags`).
        """
        sample = self._sample(indexers)
        return {
            name: Q_(float(sample[name].values), self.units(name))
            for name in self._variables
        }

    def sample(
        self, name: str, dim: str | None = None, **indexers: Any
    ) -> "ParameterSample":
        """A parameter as a `ParameterSample` for the statistics of `pkpdutils.stats`.

        With `dim`, the individual values of `name` along `dim`, after the
        other sample dimensions were selected with `indexers`; the labels are
        the coordinate of `dim` and the coordinates along `dim` (`period`,
        `sequence`, ...) travel with the sample. Without `dim`, the summary
        statistics of a group result: `x` as the mean, `x_sd` (or `x_se`
        times the square root of `n`) as the standard deviation, `x_n` when
        present else `n` as the number of individuals, and `x_geomean`,
        `x_geocv` when present.

        Args:
            name: name of the parameter.
            dim: the sample dimension the values run over, `None` for a
                summary sample.
            **indexers: coordinate label per remaining sample dimension.

        Returns:
            The sample.

        Raises:
            ValueError: if `name` is not a variable, `dim` is not a sample
                dimension, a sample dimension besides `dim` is not indexed,
                or the summary sample has no group statistics.
        """
        from pkpdutils.stats.sample import ParameterSample

        if name not in self.ds.data_vars:
            raise ValueError(f"'{name}' is not a variable of the result")
        if dim is not None and dim not in self.sample_dims:
            raise ValueError(f"'{dim}' is not a sample dimension {self.sample_dims}")
        selected = self.ds.sel(indexers) if indexers else self.ds
        remaining = [d for d in selected["flags"].dims if d != dim]
        if remaining:
            raise ValueError(
                f"The remaining sample dimensions {remaining} need an indexer each"
            )
        unit = self.units(name)
        if dim is not None:
            da = selected[name]
            labels = (
                selected[dim].to_numpy()
                if dim in selected.coords
                else np.arange(da.sizes[dim])
            )
            coords = {
                str(key): coord.to_numpy()
                for key, coord in selected.coords.items()
                if key != dim and tuple(coord.dims) == (dim,)
            }
            return ParameterSample(
                values=da.to_numpy().astype(np.float64),
                labels=labels,
                coords=coords,
                name=name,
                unit=unit,
            )
        has_sd = f"{name}_sd" in selected
        has_se = f"{name}_se" in selected
        count_name = f"{name}_n" if f"{name}_n" in selected else "n"
        if not (has_sd or has_se) or count_name not in selected:
            raise ValueError(
                f"'{name}' has no group data ('{name}_sd' or '{name}_se' with 'n'); "
                "give 'dim' for individual values"
            )
        n = float(selected[count_name].values)
        sd = (
            float(selected[f"{name}_sd"].values)
            if has_sd
            else float(selected[f"{name}_se"].values) * np.sqrt(n)
        )
        geomean = (
            float(selected[f"{name}_geomean"].values)
            if f"{name}_geomean" in selected
            else None
        )
        geocv = (
            float(selected[f"{name}_geocv"].values)
            if f"{name}_geocv" in selected
            else None
        )
        return ParameterSample(
            mean=float(selected[name].values),
            sd=sd,
            n=n,
            geomean=geomean,
            geocv=geocv,
            name=name,
            unit=unit,
        )

    def flags(self, **indexers: Any) -> list[str]:
        """Names of the flags set for one sample.

        Args:
            **indexers: coordinate label per sample dimension.

        Returns:
            The names of the flags set for the sample.
        """
        sample = self._sample(indexers)
        return self.decode_flags(int(sample["flags"].values))

    def to_dataframe(self) -> pd.DataFrame:
        """One row per sample: the sample coordinates, every scalar variable and the decoded flags.

        The point variables (the data and the predictions of a fit, the
        correlation matrix) are left out: they carry a dimension beyond the
        sample dimensions, and `xarray.Dataset.to_dataframe` of the whole
        dataset would repeat every sample once per point and per parameter
        pair. `xarray.Dataset.to_dataframe` also refuses a 0-dimensional
        dataset (no sample dimensions), so that case is built as a single-row
        frame directly.

        Returns:
            The dataframe.
        """
        if not self.sample_dims:
            row: dict[str, Any] = {
                name: float(self.ds[name].values) for name in self._variables
            }
            row["flags"] = "|".join(self.decode_flags(int(self.ds["flags"].values)))
            return pd.DataFrame([row])
        df = self.ds[[*self._variables, "flags"]].to_dataframe().reset_index()
        df["flags"] = ["|".join(self.decode_flags(int(v))) for v in df["flags"]]
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
                str(flag.name): (value & flag.value) != 0
                for flag in self.flag_type
                if flag.value
            }
            return pd.DataFrame([row])
        df = self.ds[["flags"]].to_dataframe().reset_index()
        df = df[[*self.sample_dims, "flags"]]
        values = df["flags"].to_numpy().astype(int)
        for flag in self.flag_type:
            if flag.value:
                df[str(flag.name)] = (values & flag.value) != 0
        return df.drop(columns=["flags"])

    def _new(self, ds: xr.Dataset) -> Self:
        """Wrap a dataset in the concrete result type.

        Subclasses with extra constructor arguments override this hook.

        Args:
            ds: the dataset of the new result.

        Returns:
            The new result.
        """
        return type(self)(ds)

    def summarize(self, dim: str, ci_level: float = 0.95) -> Self:
        """Summarize the parameters of individual samples over one sample dimension.

        For every parameter `x` the summary carries the arithmetic mean `x`,
        `x_sd`, `x_se`, the t-based confidence interval `x_ci_low`/`x_ci_high`
        at `ci_level`, `x_median`, `x_q25`, `x_q75`, the count of finite values
        `x_n` and, for log-normal parameters (`lognormal_parameters`),
        `x_geomean` and `x_geocv`; `flags` is the union of the flags of the
        samples. The derived, the point and the statistic variables of the
        input are dropped: a statistic (`statistic_variables`, the goodness
        of fit and the counts of a fit) describes the analysis of one sample,
        not a quantity of which a mean over samples would mean anything, and
        is read from the unsummarized result.

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
            if name in self.lognormal_parameters:
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
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(self.ds.attrs))
        return self._new(ds)
