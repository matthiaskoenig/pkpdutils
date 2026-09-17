"""Shared container of parameter results (`NCAResult`, `FitResult`): an `xarray.Dataset` over sample dimensions, units per variable, an integer `flags` variable, quantities, data frames and summaries."""

import itertools
import warnings
from collections.abc import Iterable, Mapping, Sequence
from enum import IntFlag
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import t as student_t

from pkpdutils.units import Q_, Quantity, parse_unit, short_unit

if TYPE_CHECKING:
    from rich.table import Table

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


def nan_percentile(
    values: np.ndarray, q: float | Sequence[float], axis: int = -1
) -> np.ndarray:
    """Percentiles along one axis ignoring `NaN`, vectorized over the other axes.

    Same result as `numpy.nanpercentile(values, q, axis=axis)` with the default
    linear interpolation, down to the last bit, but without its fallback to
    `numpy.apply_along_axis`, which is a python level loop over the reduced
    slices as soon as the array holds a single `NaN`. The slices are sorted
    instead (`NaN` sorts last), the finite count `k` of every slice gives the
    virtual index `(k - 1) q / 100` and the two neighbouring order statistics
    are gathered and interpolated in one vectorized step, as numpy does for an
    array without `NaN`.

    A slice without a single non-`NaN` value is `NaN`, without the
    `RuntimeWarning` numpy emits for it.

    Args:
        values: the values; `NaN` is ignored, `+-inf` is an ordinary value, as
            in `numpy.nanpercentile`.
        q: percentile in `[0, 100]`, or a sequence of them.
        axis: the axis to reduce.

    Returns:
        The percentiles: the shape of `values` without `axis` for a single `q`,
        with the number of percentiles prepended for a sequence of them.
    """
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.true_divide(np.asarray(q, dtype=np.float64), 100)
    single = quantiles.ndim == 0
    levels = np.atleast_1d(quantiles)[:, None]
    moved = np.moveaxis(array, axis, -1)
    shape = moved.shape[:-1]
    if moved.shape[-1] == 0:
        empty = np.full((levels.size, *shape), np.nan)
        return empty[0] if single else empty
    flat = np.ascontiguousarray(moved).reshape(-1, moved.shape[-1])
    ordered = np.sort(flat, axis=-1)
    count = np.count_nonzero(~np.isnan(flat), axis=-1)
    n = count.astype(np.float64)[None, :]
    # numpy's "linear" method: the percentile sits at (k - 1) q / 100 of the
    # sorted values and is interpolated between its neighbours; an index at or
    # beyond the last value takes the last value; a slice of only NaN has
    # k = 0 and every index of it is NaN
    virtual = (n - 1.0) * levels
    last = np.maximum(count - 1, 0)[None, :]
    below = np.floor(virtual).astype(np.intp)
    above = below + 1
    beyond = virtual >= n - 1.0
    before = virtual < 0.0
    below = np.where(beyond, last, below)
    above = np.where(beyond, last, above)
    below = np.where(before, 0, below)
    above = np.where(before, 0, above)
    gamma = virtual - below
    rows = np.arange(flat.shape[0])[None, :]
    low, high = ordered[rows, below], ordered[rows, above]
    # the two branches of numpy's interpolation, which anchors at the closer of
    # the two values so that the result stays inside the interval; `errstate`
    # covers the slices which hold `NaN` or an infinity, where numpy warns too
    with np.errstate(invalid="ignore"):
        span = high - low
        out = np.where(gamma >= 0.5, high - span * (1.0 - gamma), low + span * gamma)
    out = out.reshape((levels.size, *shape))
    return out[0] if single else out


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


def decode_flags(flag_type: type[IntFlag], value: int) -> list[str]:
    """Names of the flags set in an integer flag value, in bit order.

    Args:
        flag_type: the `IntFlag` type the value belongs to.
        value: an integer combination of its members.

    Returns:
        The names of the set flags, in the declaration order of `flag_type`;
        the zero member and unnamed members are left out.
    """
    return [
        str(flag.name)
        for flag in flag_type
        if flag.value and value & flag.value and flag.name is not None
    ]


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
SUMMARY_SUFFIXES: tuple[str, ...] = (
    "_median",
    "_q25",
    "_q75",
    "_min",
    "_max",
    "_n",
)


#: the boolean variable which marks a sample the analyses and the statistics
#: leave out (`pkpdutils.nca.NCAResult.exclude`, `Acceptance(exclude=True)`)
EXCLUDED_VARIABLE: str = "excluded"

#: the summary variables every statistic of `summary_table` reads, as the
#: suffixes of the parameter (`""` is the parameter itself, the mean)
TABLE_STATISTICS: dict[str, tuple[str, ...]] = {
    "n": ("_n",),
    "mean": ("",),
    "sd": ("_sd",),
    "se": ("_se",),
    "cv": ("_cv",),
    "geomean": ("_geomean",),
    "geocv": ("_geocv",),
    "median": ("_median",),
    "q25": ("_q25",),
    "q75": ("_q75",),
    "min": ("_min",),
    "max": ("_max",),
    "range": ("_min", "_max"),
}

#: statistics of `summary_table` which are fractions and are reported in percent
PERCENT_STATISTICS: frozenset[str] = frozenset({"cv", "geocv"})

#: the statistics `summary_table` reports by default
DEFAULT_STATISTICS: tuple[str, ...] = (
    "n",
    "mean",
    "sd",
    "cv",
    "geomean",
    "geocv",
    "median",
    "min",
    "max",
)


def format_number(value: float, digits: int = 3) -> str:
    """A number rounded to significant digits, without an exponent where one is not needed.

    The shared formatting of the publication tables
    (`pkpdutils.result.summary_table`, `pkpdutils.stats.ratio_table`,
    `pkpdutils.stats.ddi_table`, `pkpdutils.fit.proportionality_table`): the
    value is rounded to `digits` significant digits and written in plain
    notation while its exponent lies in `[-4, digits + 3)`, the range in which
    the plain form is no longer than the scientific one, and in scientific
    notation outside it.

    Args:
        value: the number; `NaN` and `None` give an empty cell.
        digits: significant digits.

    Returns:
        The formatted number; `""` for a missing value.

    Raises:
        ValueError: if `digits` is not positive.
    """
    if digits < 1:
        raise ValueError(f"'digits' must be positive, got {digits}")
    if value is None:
        return ""
    number = float(value)
    if np.isnan(number):
        return ""
    if np.isinf(number):
        return "inf" if number > 0 else "-inf"
    if number == 0.0:
        return "0"
    rounded = float(f"{number:.{digits}g}")
    exponent = int(np.floor(np.log10(abs(rounded))))
    if -4 <= exponent < digits + 3:
        return f"{rounded:.{max(digits - 1 - exponent, 0)}f}"
    return f"{rounded:.{digits - 1}e}"


def _conversion(name: str, current: str, target: str) -> tuple[float, str]:
    """The factor from one unit to another and the canonical target unit.

    Args:
        name: name of the variable, for the error message.
        current: the unit the variable carries.
        target: the unit it is converted to.

    Returns:
        The factor a magnitude in `current` is multiplied with, and the
        canonical spelling of `target` (`"hour * nanogram / milliliter"` for
        `"h*ng/mL"`).

    Raises:
        ValueError: if `target` is not a unit of the registry or does not have
            the dimensionality of `current`.
    """
    unit = parse_unit(target)
    quantity = Q_(1.0, current)
    if not quantity.is_compatible_with(unit):
        raise ValueError(
            f"'{name}' is in '{current}' and cannot be converted to '{target}'"
        )
    return float(quantity.to(unit).magnitude), str(unit)


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
    #: point variables (an extra dimension beyond the sample dimensions) which
    #: `summarize` reduces over the sample dimension like a parameter, keeping
    #: their extra dimension (the `interval_*` parameters of a multiple dose
    #: analysis: the mean trough per dosing interval over the subjects)
    summarized_point_variables: ClassVar[frozenset[str]] = frozenset()
    #: variables describing the status of a sample rather than a quantity of
    #: it: the boolean `accepted` and `excluded` and the text
    #: `excluded_reason` of an NCA. They are no parameters, carry no unit and
    #: no statistics, and are reported as columns of `to_dataframe` only
    status_variables: ClassVar[frozenset[str]] = frozenset()
    #: the parameters `rich_table` shows by default with one row per sample,
    #: in this order, the ones a reader looks for first; a subclass names its
    #: headline parameters, an empty tuple shows every parameter
    console_parameters: ClassVar[tuple[str, ...]] = ()
    #: the dose normalized variable of a parameter whose name is not
    #: `f"{parameter}{DOSE_NORMALIZED_SUFFIX}"` (`auc_inf_obs` is reported as
    #: `auc_inf_dn`); read by `to_units`, which converts the dose normalized
    #: companion of a parameter with it
    dose_normalized_names: ClassVar[Mapping[str, str]] = {}

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
        """Names of the parameters (the data variables except `flags`, `n`, the statistics, the status, the derived and the point variables)."""
        point = set(self.point_variables)
        return [
            str(name)
            for name in self.ds.data_vars
            if name not in ("flags", "n")
            and str(name) not in self.statistic_variables
            and str(name) not in self.status_variables
            and base_name(str(name)) is None
            and name not in point
        ]

    @property
    def status(self) -> list[str]:
        """Names of the status variables present in the result (`status_variables`)."""
        return [
            str(name)
            for name in self.ds.data_vars
            if str(name) in self.status_variables
        ]

    @property
    def excluded(self) -> xr.DataArray | None:
        """The boolean `excluded` variable of the result, `None` without one.

        A sample which is marked here is left out of `summarize`,
        `summary_table`, `sample` and every statistic which reads a result,
        unless `include_excluded=True` asks for it.
        """
        if EXCLUDED_VARIABLE not in self.ds.data_vars:
            return None
        return self.ds[EXCLUDED_VARIABLE].astype(bool)

    def _keep(self, include_excluded: bool) -> xr.DataArray | None:
        """The samples an analysis reads: `None` when every sample is read.

        Args:
            include_excluded: whether the excluded samples are read as well.

        Returns:
            The boolean array of the samples to keep over the sample
            dimensions, `None` when the result marks none or every sample is
            read.
        """
        if include_excluded:
            return None
        excluded = self.excluded
        if excluded is None or not bool(excluded.any()):
            return None
        return ~excluded

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
        """Names of the numeric scalar data variables except `flags`, in the order of the dataset."""
        point = set(self.point_variables)
        return [
            str(name)
            for name in self.ds.data_vars
            if name != "flags"
            and name not in point
            and str(name) not in self.status_variables
        ]

    def units(self, name: str) -> str:
        """Unit string of a parameter.

        Args:
            name: name of the parameter.

        Returns:
            The unit string of the parameter.
        """
        return str(self.ds[name].attrs["units"])

    def _companions(self, name: str) -> list[str]:
        """The variables derived from one parameter, in the order of the dataset.

        The uncertainty and summary variables of the parameter (`x_sd`, `x_se`,
        `x_ci_low`, `x_median`, ..., every variable whose `base_name` is
        `name`) and its dose normalized variable (`x_dn`, or the name of
        `dose_normalized_names`).

        Args:
            name: name of the parameter.

        Returns:
            The names of the derived variables present in the result.
        """
        normalized = self.dose_normalized_names.get(name, f"{name}_dn")
        return [
            str(variable)
            for variable in self.ds.data_vars
            if base_name(str(variable)) == name or str(variable) == normalized
        ]

    def to_units(self, units: Mapping[str, str]) -> Self:
        """Convert named variables of the result to other units.

        The reporting units of a submission are not the units the data was
        measured in: an exposure in `hour * nanogram / milliliter` is reported
        as `h*ng/mL`, a clearance in `liter / hour` as `mL/min`. Every named
        variable is converted with pint, together with the variables derived
        from it, which carry the same quantity: the uncertainty and the summary
        variables (`x_sd`, `x_se`, `x_ci_low`, `x_ci_high`, `x_geomean`,
        `x_median`, `x_min`, ...) and the dose normalized variable (`x_dn`,
        which is converted per dose, so `auc_inf_obs` in `h*ng/mL` reports
        `auc_inf_dn` in `h*ng/mL/mg`). The dimensionless companions of a
        parameter (`x_cv`, `x_geocv`, `x_n`) are left as they are.

        The values are multiplied by the conversion factor and
        `attrs["units"]` is rewritten with the canonical spelling of the
        target unit; the result is a new object, the one it was called on is
        unchanged. `NCAOptions.units` applies the conversion to the result of
        `pkpdutils.nca.nca` directly.

        Args:
            units: variable name to the unit to convert it to, e.g.
                `{"auc_inf_obs": "h*ng/mL", "cl_f": "mL/min"}`.

        Returns:
            The result with the named variables and their companions in the
            new units.

        Raises:
            KeyError: if a name is not a variable of the result.
            ValueError: if a unit is not a unit of the registry, or does not
                have the dimensionality of the variable.
        """
        ds = self.ds.copy()
        for name, target in units.items():
            if name not in ds.data_vars:
                raise KeyError(f"'{name}' is not a variable of the result")
            current = str(ds[name].attrs["units"])
            normalized = self.dose_normalized_names.get(name, f"{name}_dn")
            family = [name, *self._companions(name)]
            if normalized in family:
                # the dose normalized variable brings its own companions
                family += [
                    variable
                    for variable in self._companions(normalized)
                    if variable not in family
                ]
            normalized_unit = (
                str(ds[normalized].attrs["units"])
                if normalized in ds.data_vars
                else None
            )
            for variable in family:
                variable_unit = str(ds[variable].attrs.get("units", "dimensionless"))
                quantity = Q_(1.0, variable_unit)
                if quantity.is_compatible_with(Q_(1.0, current)):
                    # the parameter itself and every companion which carries
                    # the same quantity (`x_sd`, `x_ci_low`, `x_median`, ...)
                    variable_target = target
                elif normalized_unit is not None and quantity.is_compatible_with(
                    Q_(1.0, normalized_unit)
                ):
                    # the dose normalized variable and its own companions are
                    # the parameter per dose: the dose keeps its unit and the
                    # numerator follows
                    dose = (
                        Q_(1.0, current) / Q_(1.0, normalized_unit)
                    ).to_reduced_units()
                    variable_target = str((Q_(1.0, target) / dose).units)
                else:
                    # a dimensionless companion (`x_cv`, `x_geocv`, `x_n`)
                    continue
                factor, unit = _conversion(variable, variable_unit, variable_target)
                attrs = dict(ds[variable].attrs)
                ds[variable] = ds[variable] * factor
                ds[variable].attrs = {**attrs, "units": unit}
        return self._new(ds)

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
        return decode_flags(self.flag_type, value)

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
        self,
        name: str,
        dim: str | None = None,
        *,
        include_excluded: bool = False,
        **indexers: Any,
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

        A sample which the result marks `excluded`
        (`pkpdutils.nca.NCAResult.exclude`) is left out, so that every
        statistic of `pkpdutils.stats` which reads a result reads the same
        individuals as the summary of it.

        Args:
            name: name of the parameter.
            dim: the sample dimension the values run over, `None` for a
                summary sample.
            include_excluded: read the excluded samples as well.
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
        keep = self._keep(include_excluded)
        if keep is not None and dim is not None:
            kept = keep.sel(indexers) if indexers else keep
            if tuple(str(d) for d in kept.dims) == (dim,):
                selected = selected.isel({dim: np.flatnonzero(kept.to_numpy())})
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
            n=int(n),
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

    def rich_table(
        self,
        *,
        parameters: Sequence[str] | None = None,
        digits: int = 3,
        transpose: bool | None = None,
        title: str | None = None,
    ) -> "Table":
        """The result as a rich table for the console, the units in short symbols.

        A result of a few samples is shown with one row per variable (name,
        unit, one column per sample), which fits a console; a result of many
        samples with one row per sample and one column per parameter, as
        `to_dataframe` lays it out. Every number is rounded to `digits`
        significant digits, the flags are written by name, and the frame of
        `to_dataframe` itself is unchanged and keeps the full precision.

        Args:
            parameters: the variables to show, in this order; by default every
                variable with one row per variable, and the headline parameters
                of the result type (`console_parameters`, every parameter if it
                names none) with one row per sample.
            digits: significant digits of the numbers.
            transpose: one row per variable (`True`) or one row per sample
                (`False`); by default one row per variable up to eight samples.
            title: the title above the table; by default the type of the result,
                the number of samples and the sample dimensions.

        Returns:
            The table, built by `pkpdutils.console.rich_table`.

        Raises:
            ValueError: if a name of `parameters` is not a variable of the result.
        """
        # imported here: pkpdutils.console imports the formatting of this module
        from pkpdutils.console import rich_table

        frame = self.to_dataframe()
        count = len(frame)
        if transpose is None:
            transpose = count <= 8
        if parameters is None:
            if transpose:
                names = list(self._variables)
            else:
                headline = [p for p in self.console_parameters if p in frame.columns]
                names = headline or self.parameters
        else:
            names = list(parameters)
        missing = [name for name in names if name not in frame.columns]
        if missing:
            raise ValueError(f"{missing} are no variables of the result")
        if title is None:
            samples = "1 sample" if count == 1 else f"{count} samples"
            dims = (
                ", ".join(self.sample_dims)
                if self.sample_dims
                else "no sample dimension"
            )
            title = f"{type(self).__name__}: {samples} ({dims})"
        if transpose:
            labels = [
                "|".join(str(frame.iloc[i][dim]) for dim in self.sample_dims) or "value"
                for i in range(count)
            ]
            rows = {
                "variable": names,
                "unit": [short_unit(self.units(name)) for name in names],
            }
            for i, label in enumerate(labels):
                rows[label] = [frame.iloc[i][name] for name in names]
            rows["variable"].append("flags")
            rows["unit"].append("")
            for i, label in enumerate(labels):
                rows[label].append(frame.iloc[i]["flags"])
            return rich_table(pd.DataFrame(rows), title=title, digits=digits)
        headers = {}
        for name in names:
            unit = short_unit(self.units(name))
            headers[name] = f"{name} [{unit}]" if unit else name
        shown = frame[[*self.sample_dims, *names, "flags"]].rename(columns=headers)
        return rich_table(shown, title=title, digits=digits)

    def __rich__(self) -> "Table":
        """The rich console protocol: `console.print(result)` shows `rich_table()`.

        Returns:
            The table of `rich_table` with its defaults.
        """
        return self.rich_table()

    def to_dataframe(self) -> pd.DataFrame:
        """One row per sample: the sample coordinates, every scalar variable and the decoded flags.

        Every sample is a row, the excluded ones included, and the status
        variables of the result (`status_variables`: `accepted`, `excluded` and
        `excluded_reason` of an NCA) are columns between the parameters and the
        flags.

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
        status = self.status
        if not self.sample_dims:
            row: dict[str, Any] = {
                name: float(self.ds[name].values) for name in self._variables
            }
            for name in status:
                row[name] = self.ds[name].values.item()
            row["flags"] = "|".join(self.decode_flags(int(self.ds["flags"].values)))
            return pd.DataFrame([row])
        df = self.ds[[*self._variables, *status, "flags"]].to_dataframe().reset_index()
        df["flags"] = ["|".join(self.decode_flags(int(v))) for v in df["flags"]]
        columns = [*self.sample_dims, *self._variables, *status, "flags"]
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

    def summary_table(
        self,
        dim: str,
        *,
        by: str | Sequence[str] | None = None,
        parameters: Sequence[str] | None = None,
        stats: Sequence[str] = DEFAULT_STATISTICS,
        digits: int | Mapping[str, int] = 3,
        units: Literal["column", "header"] = "column",
        unit_style: Literal["long", "short"] = "long",
        layout: Literal[
            "parameters_rows", "parameters_columns", "long"
        ] = "parameters_rows",
        include_excluded: bool = False,
    ) -> pd.DataFrame:
        """The publication parameter table of this result, see `pkpdutils.result.summary_table`.

        Args:
            dim: the sample dimension the statistics are taken over.
            by: coordinate along `dim` to group the samples by, or several.
            parameters: the parameters of the table, every parameter of the
                result by default.
            stats: the statistics of the table, see
                `pkpdutils.result.TABLE_STATISTICS`.
            digits: significant digits of the numbers, one number for the
                whole table or one per parameter.
            units: whether the unit is a column of its own or part of the
                parameter name.
            unit_style: the long form of pint or its short symbols.
            layout: parameters as rows, as columns, or one row per parameter,
                group and statistic.
            include_excluded: report the excluded samples as well.

        Returns:
            The table, every cell a formatted string.

        Raises:
            ValueError: as `pkpdutils.result.summary_table`.
        """
        # the module level function of the same name, the method delegates
        return summary_table(
            self,
            dim,
            by=by,
            parameters=parameters,
            stats=stats,
            digits=digits,
            units=units,
            unit_style=unit_style,
            layout=layout,
            include_excluded=include_excluded,
        )

    def _new(self, ds: xr.Dataset) -> Self:
        """Wrap a dataset in the concrete result type.

        Subclasses with extra constructor arguments override this hook.

        Args:
            ds: the dataset of the new result.

        Returns:
            The new result.
        """
        return type(self)(ds)

    def summarize(
        self, dim: str, ci_level: float = 0.95, *, include_excluded: bool = False
    ) -> Self:
        """Summarize the parameters of individual samples over one sample dimension.

        For every parameter `x` the summary carries the arithmetic mean `x`,
        `x_sd`, `x_se`, the coefficient of variation `x_cv` as a fraction, the
        t-based confidence interval `x_ci_low`/`x_ci_high` at `ci_level`,
        `x_median`, `x_q25`, `x_q75`, `x_min`, `x_max`, the count of finite
        values `x_n` and, for log-normal parameters (`lognormal_parameters`),
        `x_geomean` and `x_geocv`; `flags` is the union of the flags of the
        samples. `pkpdutils.result.summary_table` formats these numbers into
        the parameter table of a publication.

        The derived and the statistic variables of the input are dropped: a
        statistic (`statistic_variables`, the goodness of fit and the counts
        of a fit) describes the analysis of one sample, not a quantity of
        which a mean over samples would mean anything, and is read from the
        unsummarized result. A point variable is dropped as well, unless it is
        listed in `summarized_point_variables` (the `interval_*` parameters of
        a multiple dose analysis), in which case it is reduced over `dim` like
        a parameter and keeps its extra dimension.

        A discrete parameter (`discrete_parameters`: an observed time, a point
        count, a diagnostic of the terminal regression) carries no uncertainty:
        a standard error, a coefficient of variation or a confidence interval
        of a point count is not a quantity, so only `x`, `x_median`, `x_q25`,
        `x_q75`, `x_min`, `x_max` and `x_n` are reported for it, the same set
        the uncertainty of an analysis of group curves reports
        (`pkpdutils.nca.uncertainty`).

        The two counts differ: `n` is the number of samples along `dim`,
        `x_n` the number of them at which `x` is finite, and every statistic of
        `x` uses `x_n` (`x_se = x_sd / sqrt(x_n)`, the interval uses
        `t` with `x_n - 1` degrees of freedom). A parameter which does not
        apply to every sample (no terminal phase, no dose) therefore has
        `x_n < n`.

        A sample which the result marks `excluded`
        (`pkpdutils.nca.NCAResult.exclude`, `Acceptance(exclude=True)`) enters
        no statistic and is not counted, neither in `n` nor in `x_n`, and its
        flags are not part of the union; `include_excluded=True` summarizes
        every sample. The status variables themselves (`status_variables`) are
        dropped, as the derived and the statistic variables are.

        Args:
            dim: the sample dimension to reduce
            ci_level: level of the confidence interval of the mean
            include_excluded: summarize the excluded samples as well

        Returns:
            The summary over the remaining sample dimensions.

        Raises:
            ValueError: if `dim` is not a sample dimension of the result.
        """
        if dim not in self.sample_dims:
            raise ValueError(f"'{dim}' is not a sample dimension {self.sample_dims}")
        alpha = 1.0 - ci_level
        keep = self._keep(include_excluded)
        data_vars: dict[str, Any] = {}
        summarized = [
            *self.parameters,
            *(
                name
                for name in self.point_variables
                if name in self.summarized_point_variables and dim in self.ds[name].dims
            ),
        ]
        for name in summarized:
            da = self.ds[name]
            if keep is not None:
                # an excluded sample is dropped from every statistic by making
                # it a missing value, which works whatever further dimensions
                # the variable carries
                da = da.where(keep)
            da = da.transpose(..., dim)
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
                q25, q75 = nan_percentile(filled, (25.0, 75.0))
                minimum = np.nanmin(filled, axis=-1)
                maximum = np.nanmax(filled, axis=-1)
                cv = sd / np.abs(mean)
                mean = np.where(count > 0, mean, np.nan)
                median = np.where(count > 0, median, np.nan)
                minimum = np.where(count > 0, minimum, np.nan)
                maximum = np.where(count > 0, maximum, np.nan)
            data_vars[name] = (dims, mean, {"units": units})
            if name not in self.discrete_parameters:
                data_vars[f"{name}_sd"] = (dims, sd, {"units": units})
                data_vars[f"{name}_se"] = (dims, se, {"units": units})
                data_vars[f"{name}_cv"] = (dims, cv, {"units": "dimensionless"})
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
            data_vars[f"{name}_min"] = (dims, minimum, {"units": units})
            data_vars[f"{name}_max"] = (dims, maximum, {"units": units})
            data_vars[f"{name}_n"] = (dims, count, {"units": "dimensionless"})
            if (
                name in self.lognormal_parameters
                and name not in self.discrete_parameters
            ):
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
        kept = (
            None
            if keep is None
            else keep.broadcast_like(self.ds["flags"]).transpose(..., dim).to_numpy()
        )
        flags = self.ds["flags"].transpose(..., dim)
        remaining = tuple(str(d) for d in flags.dims if d != dim)
        data_vars["n"] = (
            remaining,
            (
                np.full(flags.shape[:-1], float(self.ds.sizes[dim]))
                if kept is None
                else kept.sum(axis=-1).astype(np.float64)
            ),
            {"units": "dimensionless"},
        )
        flag_values = flags.to_numpy().astype(np.int64)
        if kept is not None:
            flag_values = np.where(kept, flag_values, 0)
        data_vars["flags"] = (
            remaining,
            np.bitwise_or.reduce(flag_values, axis=-1),
            {"units": "dimensionless"},
        )
        # the dimension coordinates of the remaining sample dimensions and of
        # the extra dimensions of the summarized point variables (`interval`)
        extra = {
            str(d)
            for name in summarized
            for d in self.ds[name].dims
            if str(d) != dim and str(d) not in remaining
        }
        coords = {
            d: self.ds[d] for d in (*remaining, *sorted(extra)) if d in self.ds.coords
        }
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(self.ds.attrs))
        return self._new(ds)


def _table_groups(
    result: ParameterResult,
    dim: str,
    by: Sequence[str],
    *,
    include_excluded: bool = False,
) -> list[tuple[dict[str, Any], ParameterResult]]:
    """The summaries of a result, one per group of the grouping coordinates.

    Args:
        result: the result of the individual samples.
        dim: the sample dimension the statistics are taken over.
        by: coordinates along `dim`, empty for one group of everything.
        include_excluded: summarize the excluded samples as well.

    Returns:
        The group labels (coordinate name to value, empty without `by`) and the
        summary of the group over `dim`, in the order the groups first appear
        along `dim`.

    Raises:
        ValueError: if a coordinate of `by` is not a coordinate along `dim`.
    """
    if not by:
        return [({}, result.summarize(dim, include_excluded=include_excluded))]
    along = sorted(
        str(name)
        for name, coord in result.ds.coords.items()
        if tuple(str(d) for d in coord.dims) == (dim,) and str(name) != dim
    )
    for name in by:
        if name != dim and name in result.sample_dims:
            raise ValueError(
                f"'{name}' is a sample dimension of the result, not a coordinate "
                f"along '{dim}': the table already carries one row per '{name}' "
                "and names its value in a column of its own; 'by' groups the "
                f"samples of '{dim}' by one of its coordinates {along}"
            )
        if name not in result.ds.coords or tuple(result.ds[name].dims) != (dim,):
            raise ValueError(f"'{name}' is not a coordinate along '{dim}': {along}")
    columns = [result.ds[name].to_numpy() for name in by]
    keys = list(zip(*columns, strict=True))
    groups: list[tuple[dict[str, Any], ParameterResult]] = []
    for key in dict.fromkeys(keys):
        positions = [i for i, other in enumerate(keys) if other == key]
        subset = result._new(result.ds.isel({dim: positions}))
        groups.append(
            (
                dict(zip(by, key, strict=True)),
                subset.summarize(dim, include_excluded=include_excluded),
            ),
        )
    return groups


def _statistic_cell(
    summary: ParameterResult,
    name: str,
    stat: str,
    index: Mapping[str, int],
    digits: int,
) -> str:
    """One cell of a summary table: a statistic of a parameter, formatted.

    Args:
        summary: the summary of one group.
        name: name of the parameter.
        stat: name of the statistic (`TABLE_STATISTICS`).
        index: integer position per remaining sample dimension.
        digits: significant digits of the numbers.

    Returns:
        The formatted cell, `""` where the statistic is missing (a discrete
        parameter has no `sd`, a parameter which is not log-normal no
        `geomean`) or not a number.
    """
    values: list[float] = []
    for suffix in TABLE_STATISTICS[stat]:
        variable = f"{name}{suffix}"
        if variable not in summary.ds:
            return ""
        da = summary.ds[variable]
        values.append(float(da.isel(index).values if index else da.values))
    if stat == "n":
        return "" if np.isnan(values[0]) else str(int(values[0]))
    if stat in PERCENT_STATISTICS:
        cell = format_number(values[0] * 100.0, digits)
        return f"{cell} %" if cell else ""
    if stat == "range":
        low, high = format_number(values[0], digits), format_number(values[1], digits)
        return f"{low} - {high}" if low and high else ""
    return format_number(values[0], digits)


def summary_table(
    result: ParameterResult,
    dim: str,
    *,
    by: str | Sequence[str] | None = None,
    parameters: Sequence[str] | None = None,
    stats: Sequence[str] = DEFAULT_STATISTICS,
    digits: int | Mapping[str, int] = 3,
    units: Literal["column", "header"] = "column",
    unit_style: Literal["long", "short"] = "long",
    layout: Literal[
        "parameters_rows", "parameters_columns", "long"
    ] = "parameters_rows",
    include_excluded: bool = False,
) -> pd.DataFrame:
    """The parameter table of a publication: one row per parameter, formatted.

    The statistics are those of `ParameterResult.summarize(dim)`, read from the
    summary and formatted with `digits` significant digits as strings, so that
    the frame goes into a manuscript (`to_csv`, `to_markdown`, `to_latex`)
    without further rounding. `cv` and `geocv` are fractions in the summary and
    are written as percentages (`"12.3 %"`); `range` is the two order
    statistics in one cell (`"10.2 - 14.8"`); a statistic a parameter does not
    carry (the `sd` of a discrete parameter, the `geomean` of a parameter which
    is not log-normal) is an empty cell. The flags are not part of the table,
    `ParameterResult.flag_table` reports them.

    The convention of the pharmacokinetic literature, "geometric mean [CV %]",
    is `stats=("n", "geomean", "geocv")`; the arithmetic convention
    "mean (SD)" is `stats=("n", "mean", "sd")`.

    Args:
        result: the result of the individual samples (not a summary).
        dim: the sample dimension the statistics are taken over, e.g.
            `"individual"`.
        by: coordinate along `dim` to group the samples by (the dose group,
            the treatment), or several of them; one group of everything by
            default.
        parameters: the parameters of the table, in this order; every
            parameter of the result by default.
        stats: the statistics, in this order, see `TABLE_STATISTICS`.
        digits: significant digits of the numbers, one number for the whole
            table or a mapping of parameter name to its own number, in which
            case a parameter the mapping does not name keeps the default 3
            (`{"tmax": 1}` writes the time of the maximum with one digit and
            every other parameter with three).
        units: `"column"` gives the unit a column of its own (a row in the
            `"parameters_columns"` layout), `"header"` appends it to the
            parameter name (`"cmax [milligram / liter]"`).
        unit_style: `"long"` writes the unit as the result stores it, the
            canonical long form of pint (`milligram / liter`); `"short"`
            writes its short symbols (`mg/l`, `pkpdutils.units.short_unit`),
            the form a manuscript prints.
        layout: `"parameters_rows"` (one row per parameter and group, one
            column per statistic), `"parameters_columns"` (the transpose: one
            column per parameter, one row per statistic and group) or
            `"long"` (one row per parameter, group and statistic).
        include_excluded: report the excluded samples as well; by default a
            sample which the result marks `excluded`
            (`pkpdutils.nca.NCAResult.exclude`) enters no statistic of the
            table and is not counted in `n`.

    Returns:
        The table, every cell a string.

    Raises:
        ValueError: if `dim` is not a sample dimension, a parameter is not a
            variable of the result, a statistic is unknown, or `units`,
            `unit_style` or `layout` is not one of the values above.
    """
    if dim not in result.sample_dims:
        raise ValueError(f"'{dim}' is not a sample dimension {result.sample_dims}")
    unknown = [stat for stat in stats if stat not in TABLE_STATISTICS]
    if unknown:
        raise ValueError(
            f"unknown statistics {unknown}, known are {sorted(TABLE_STATISTICS)}"
        )
    if units not in ("column", "header"):
        raise ValueError(f"'units' must be 'column' or 'header', got '{units}'")
    if unit_style not in ("long", "short"):
        raise ValueError(f"'unit_style' must be 'long' or 'short', got '{unit_style}'")
    if layout not in ("parameters_rows", "parameters_columns", "long"):
        raise ValueError(
            "'layout' must be 'parameters_rows', 'parameters_columns' or 'long', "
            f"got '{layout}'"
        )
    names = list(result.parameters if parameters is None else parameters)
    missing = [name for name in names if name not in result.ds.data_vars]
    if missing:
        raise ValueError(f"{missing} are no variables of the result")
    sample_dims = set(result.sample_dims)
    extra = [
        name for name in names if not set(map(str, result.ds[name].dims)) <= sample_dims
    ]
    if extra:
        raise ValueError(
            f"{extra} carry a dimension beyond the sample dimensions and have no "
            "row in the table; the per-interval parameters are reported by "
            "'NCAResult.intervals()' of the summary"
        )
    group_columns = [by] if isinstance(by, str) else list(by or [])
    groups = _table_groups(
        result, dim, group_columns, include_excluded=include_excluded
    )

    records: list[dict[str, Any]] = []
    for labels, summary in groups:
        rest = summary.sample_dims
        sizes = [range(int(summary.ds.sizes[d])) for d in rest]
        for position in itertools.product(*sizes):
            index = dict(zip(rest, position, strict=True))
            sample = {
                d: (summary.ds[d].to_numpy()[i] if d in summary.ds.coords else int(i))
                for d, i in index.items()
            }
            for name in names:
                unit = result.units(name)
                if unit_style == "short":
                    unit = short_unit(unit)
                parameter_digits = (
                    digits if isinstance(digits, int) else digits.get(name, 3)
                )
                row: dict[str, Any] = {
                    "parameter": f"{name} [{unit}]" if units == "header" else name
                }
                if units == "column":
                    row["unit"] = unit
                row.update(labels)
                row.update(sample)
                for stat in stats:
                    row[stat] = _statistic_cell(
                        summary, name, stat, index, parameter_digits
                    )
                records.append(row)

    df = pd.DataFrame.from_records(records)
    index_columns = [column for column in df.columns if column not in set(stats)]
    if layout == "parameters_rows":
        return df
    if layout == "long":
        # `melt` groups the frame by statistic; `_row` restores the order of
        # the records, so that the statistics of a parameter stay together
        long = df.assign(_row=np.arange(len(df))).melt(
            id_vars=[*index_columns, "_row"],
            value_vars=list(stats),
            var_name="statistic",
            value_name="value",
        )
        long = long.sort_values("_row", kind="stable").drop(columns="_row")
        return long.reset_index(drop=True)
    return _parameters_as_columns(df, index_columns, list(stats), units)


def _parameters_as_columns(
    df: pd.DataFrame, index_columns: Sequence[str], stats: Sequence[str], units: str
) -> pd.DataFrame:
    """Transpose a `"parameters_rows"` table to one column per parameter.

    Args:
        df: the table with one row per parameter, group and sample.
        index_columns: the columns of `df` which are not statistics.
        stats: the statistics, in the order of the rows of the result.
        units: `"column"` writes the units into a row of their own.

    Returns:
        One block per group and sample, with `"statistic"` and the group
        columns as the leading columns and one column per parameter.
    """
    keys = [column for column in index_columns if column not in ("parameter", "unit")]
    blocks = df.groupby(keys, sort=False) if keys else [((), df)]
    frames: list[pd.DataFrame] = []
    for key, block in blocks:
        values = key if isinstance(key, tuple) else (key,)
        table = block.set_index("parameter")[list(stats)].T
        if units == "column":
            unit_row = block.set_index("parameter")["unit"].to_frame().T
            unit_row.index = pd.Index(["unit"])
            table = pd.concat([unit_row, table])
        table.index.name = "statistic"
        table = table.reset_index()
        for column, value in reversed(list(zip(keys, values, strict=True))):
            table.insert(0, column, value)
        frames.append(table)
    out = pd.concat(frames, ignore_index=True)
    out.columns.name = None
    return out
