"""Front ends of the engine for one timecourse, for batches of timecourses and for tables of parameters."""

from typing import Any

import numpy as np
import xarray as xr

from pkpdutils.fit.engine import _named, build_result, fit_rows
from pkpdutils.fit.model import Model
from pkpdutils.fit.options import FitOptions
from pkpdutils.fit.result import FitResult
from pkpdutils.result import sample_coordinates
from pkpdutils.timecourse import Timecourse, Timecourses

#: the placeholder name of a batch which does not name its substance; the
#: figures then label the value axis `value` instead of `substance`
_UNNAMED_SUBSTANCE = "substance"


def fit_timecourse(
    model: Model, timecourse: Timecourse, *, options: FitOptions | None = None
) -> FitResult:
    """Fit a model to one timecourse, with the times relative to the first dose.

    The curve is fitted as a batch of one (`fit_timecourses`) and the single
    sample is dropped from the result, so the result has no sample dimension
    and `to_quantities`, `flags`, `predict` and `correlation` need no
    indexer.

    Args:
        model: the model (`x` is the time relative to the first dose, `y` the value)
        timecourse: the curve; its `sd` is used for `Weighting.INV_SD`

    Keyword Args:
        options: the options, defaults for `None`

    Returns:
        The result of the single curve, without a sample dimension.
    """
    batch = fit_timecourses(
        model, Timecourses.from_timecourses([timecourse]), options=options
    )
    dim = batch.sample_dims[0]
    return FitResult(batch.ds.isel({dim: 0}, drop=True), model)


def fit_timecourses(
    model: Model, timecourses: Timecourses, *, options: FitOptions | None = None
) -> FitResult:
    """Fit a model to every curve of a batch, with the times relative to the first dose.

    The batch is flattened to `(N, n_time)` rows over its sample dimensions
    (any number of them) and fitted with `fit_rows`; the result is then
    reshaped back to `timecourses.sample_shape`. `x` is the time relative to
    the first dose of the protocol when the batch carries doses, else `times`
    unchanged; `NaN`-padded
    (ragged) times stay `NaN` and are dropped by the engine. Parameters are
    reported in the raw units of the batch, `x_unit = timecourses.time_unit`
    and `y_unit = timecourses.unit`.

    The result names what was fitted in `attrs["x_name"] = "time"` and
    `attrs["y_name"]`, the substance of the batch (`"value"` when it does not
    name one), which the figures of the fit use to label their axes.

    Args:
        model: the model (`x` is the time relative to the first dose, `y` the value)
        timecourses: the batch; `sd` is used for `Weighting.INV_SD`

    Keyword Args:
        options: the options, defaults for `None`

    Returns:
        The result over the sample dimensions of the batch.

    Raises:
        ValueError: if a sample dimension or a coordinate of the batch
            collides with a variable or a dimension (`point`, `parameter`,
            `parameter_`) of the result.
    """
    options = options or FitOptions()
    n_rows, n_time = timecourses.n_samples, timecourses.n_time
    x = timecourses.times.reshape(n_rows, n_time)
    first_dose_time = timecourses.first_dose_time
    if first_dose_time is not None:
        x = x - np.asarray(first_dose_time, dtype=np.float64).reshape(n_rows)[:, None]
    y = timecourses.values.reshape(n_rows, n_time)
    sd = None if timecourses.sd is None else timecourses.sd.reshape(n_rows, n_time)
    rows = fit_rows(model, x, y, sd, options)
    coords = sample_coordinates(timecourses.ds, timecourses.sample_dims)
    result = build_result(
        model,
        rows,
        x=x,
        y=y,
        sd=sd,
        x_unit=timecourses.time_unit,
        y_unit=timecourses.unit,
        dims=timecourses.sample_dims,
        coords=coords,
        options=options,
        shape=timecourses.sample_shape,
    )
    # a batch of several analytes has no single name for its values
    substance = (
        _UNNAMED_SUBSTANCE
        if timecourses.substances is not None
        else timecourses.substance
    )
    return _named(
        result,
        "time",
        "value" if substance == _UNNAMED_SUBSTANCE else substance,
    )


def _check_broadcastable(
    da: xr.DataArray, y_da: xr.DataArray, name: str, y_name: str
) -> None:
    """Reject a variable with a dimension `y` does not have.

    Such a dimension would silently survive the broadcast against `y_da` and
    inflate the row count of the fit with a dimension the caller never
    intended as a sample dimension.

    Args:
        da: the variable to check (`x` or `sd`).
        y_da: the dependent variable, already transposed to its own order.
        name: name of `da` in the dataset, for the message.
        y_name: name of `y` in the dataset, for the message.

    Raises:
        ValueError: if `da` has a dimension `y_da` does not have.
    """
    extra = {str(d) for d in da.dims} - {str(d) for d in y_da.dims}
    if extra:
        raise ValueError(
            f"'{name}' has the dimension '{sorted(extra)[0]}' that '{y_name}' does not have"
        )


def fit_table(
    model: Model,
    ds: xr.Dataset,
    x: str,
    y: str,
    *,
    dim: str,
    sd: str | None = None,
    options: FitOptions | None = None,
) -> FitResult:
    """Fit `y` against `x` along one dimension of a dataset, for every combination of the other dimensions.

    `x` may be a coordinate or a variable of `ds`; it is broadcast to the
    dimensions of `y` (e.g. a `dose` coordinate against an `auc_inf_obs`
    variable that also carries an `individual` dimension). The units are read
    from `attrs["units"]` of `x` and `y`, `"dimensionless"` when absent (a
    coordinate such as the `dose` of an `NCAResult.ds` need not carry units;
    a model whose parameter units depend on `[x]` then reports that
    parameter without the `x` part of its unit). This works directly on the
    dataset of another result, e.g.
    `fit_table(Power(), result.ds, "dose", "auc_inf_obs", dim="dose")` on an
    `NCAResult`. `NaN` in `x` or `y` drops the point, as does any point the
    engine already drops as non-finite.

    The result names what was fitted against what in `attrs["x_name"]` and
    `attrs["y_name"]` (`x` and `y`), which the figures of the fit use to
    label their axes.

    Args:
        model: the model
        ds: dataset with the variable `y` and the coordinate or variable `x`
        x: name of the independent variable
        y: name of the dependent variable
        dim: the dimension along which the points of one fit lie (e.g. `"dose"`)
        sd: name of the standard deviation variable of `y`, for `Weighting.INV_SD`
        options: the options, defaults for `None`

    Returns:
        The result over the remaining dimensions of `y` (0-D when `y` has only `dim`).

    Raises:
        ValueError: if `y` has no dimension `dim`, if `x` (or `sd`) has a
            dimension `y` does not have, or if a remaining dimension of `y`
            collides with a variable or a dimension (`point`, `parameter`,
            `parameter_`) of the result.
    """
    options = options or FitOptions()
    if dim not in ds[y].dims:
        raise ValueError(f"'{y}' has no dimension '{dim}'")
    y_da = ds[y].transpose(..., dim)
    _check_broadcastable(ds[x], y_da, x, y)
    # broadcasting can reorder the dimensions (the union of both arguments'
    # dimensions, in first-seen order), so the result is transposed back to
    # `y_da`'s own dimension order rather than merely moving `dim` to the
    # end; otherwise a sample dimension whose order differs between `x` and
    # `y` (e.g. `x` over `("b", "a", dim)` against `y` over `("a", "b", dim)`)
    # reshapes into rows that pair the wrong `x` with the wrong `y`.
    x_da = xr.broadcast(ds[x], y_da)[0].transpose(*y_da.dims)
    sample_dims = tuple(str(d) for d in y_da.dims if d != dim)
    shape = tuple(int(y_da.sizes[d]) for d in sample_dims)
    n = int(y_da.sizes[dim])
    n_rows = int(np.prod(shape, dtype=int)) if shape else 1
    x_arr = x_da.to_numpy().astype(np.float64).reshape(n_rows, n)
    y_arr = y_da.to_numpy().astype(np.float64).reshape(n_rows, n)
    sd_arr = None
    if sd is not None:
        _check_broadcastable(ds[sd], y_da, sd, y)
        sd_da = xr.broadcast(ds[sd], y_da)[0].transpose(*y_da.dims)
        sd_arr = sd_da.to_numpy().astype(np.float64).reshape(n_rows, n)
    rows = fit_rows(model, x_arr, y_arr, sd_arr, options)
    coords: dict[str, Any] = {d: ds[d] for d in sample_dims if d in ds.coords}
    result = build_result(
        model,
        rows,
        x=x_arr,
        y=y_arr,
        sd=sd_arr,
        x_unit=str(ds[x].attrs.get("units", "dimensionless")),
        y_unit=str(ds[y].attrs.get("units", "dimensionless")),
        dims=sample_dims,
        coords=coords,
        options=options,
        shape=shape,
    )
    return _named(result, x, y)
