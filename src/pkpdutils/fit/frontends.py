"""Front ends of the engine for batches of timecourses and for tables of parameters."""

from typing import Any

import numpy as np
import xarray as xr

from pkpdutils.fit.engine import build_result, fit_rows
from pkpdutils.fit.model import Model
from pkpdutils.fit.options import FitOptions
from pkpdutils.fit.result import FitResult
from pkpdutils.timecourse import Timecourses


def fit_timecourses(
    model: Model, timecourses: Timecourses, options: FitOptions | None = None
) -> FitResult:
    """Fit a model to every curve of a batch, with the times relative to the dose.

    The batch is flattened to `(N, n_time)` rows over its sample dimensions
    (any number of them) and fitted with `fit_rows`; the result is then
    reshaped back to `timecourses.sample_shape`. `x` is `times - dose_time`
    when the batch carries a dose, else `times` unchanged; `NaN`-padded
    (ragged) times stay `NaN` and are dropped by the engine. Parameters are
    reported in the raw units of the batch, `x_unit = timecourses.time_unit`
    and `y_unit = timecourses.unit`.

    Args:
        model: the model (`x` is the time relative to the dose, `y` the value)
        timecourses: the batch; `sd` is used for `Weighting.INV_SD`
        options: the options, defaults for `None`

    Returns:
        The result over the sample dimensions of the batch.
    """
    options = options or FitOptions()
    n_rows, n_time = timecourses.n_samples, timecourses.n_time
    x = timecourses.times.reshape(n_rows, n_time)
    dose_time = timecourses.dose_time
    if dose_time is not None:
        x = x - np.asarray(dose_time, dtype=np.float64).reshape(n_rows)[:, None]
    y = timecourses.values.reshape(n_rows, n_time)
    sd = None if timecourses.sd is None else timecourses.sd.reshape(n_rows, n_time)
    rows = fit_rows(model, x, y, sd, options)
    coords = {
        d: timecourses.ds[d]
        for d in timecourses.sample_dims
        if d in timecourses.ds.coords
    }
    return build_result(
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
    from `attrs["units"]` of `x` and `y` (`"dimensionless"` when absent), so
    this works directly on the dataset of another result, e.g.
    `fit_table(Power(), result.ds, "dose", "auc_inf_obs", dim="dose")` on an
    `NCAResult`. `NaN` in `x` or `y` drops the point, as does any point the
    engine already drops as non-finite.

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
        ValueError: if `y` has no dimension `dim`.
    """
    options = options or FitOptions()
    if dim not in ds[y].dims:
        raise ValueError(f"'{y}' has no dimension '{dim}'")
    y_da = ds[y].transpose(..., dim)
    x_da = xr.broadcast(ds[x], y_da)[0].transpose(..., dim)
    sample_dims = tuple(str(d) for d in y_da.dims if d != dim)
    shape = tuple(int(y_da.sizes[d]) for d in sample_dims)
    n = int(y_da.sizes[dim])
    n_rows = int(np.prod(shape, dtype=int)) if shape else 1
    x_arr = x_da.to_numpy().astype(np.float64).reshape(n_rows, n)
    y_arr = y_da.to_numpy().astype(np.float64).reshape(n_rows, n)
    sd_arr = None
    if sd is not None:
        sd_da = xr.broadcast(ds[sd], y_da)[0].transpose(..., dim)
        sd_arr = sd_da.to_numpy().astype(np.float64).reshape(n_rows, n)
    rows = fit_rows(model, x_arr, y_arr, sd_arr, options)
    coords: dict[str, Any] = {d: ds[d] for d in sample_dims if d in ds.coords}
    return build_result(
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
