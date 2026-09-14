"""The non-compartmental analysis.

`nca` analyses a `Timecourses` batch, `nca_single` one `Timecourse`. The
numerics run in `compute_parameters` on `(N, n)` arrays, one row per curve,
with the times relative to the dose; the rows are the flattened sample
dimensions of the batch and the results are reshaped back into an
`xarray.Dataset` over the same dimensions (`NCAResult`).

Definitions follow Gabrielsson & Weiner (2016, ch. 2.8) and the Phoenix
WinNonlin NCA, see `docs/nca.md`:

- `AUC(0-tlast)` and `AUMC(0-tlast)` by the trapezoid rule of `AUCMethod`
- `lambda_z` from the terminal log-linear regression (`TerminalPhase`),
  `t½ = ln 2 / lambda_z`
- `AUC(0-inf) = AUC(0-tlast) + Clast / lambda_z` (observed or predicted `Clast`)
- `AUMC(0-inf) = AUMC(0-tlast) + Clast tlast / lambda_z + Clast / lambda_z²`
- `MRT = AUMC(0-inf) / AUC(0-inf)`, minus half the infusion duration
- `CL = Dose / AUC(0-inf)`, `Vz = CL / lambda_z`, `Vss = CL MRT` (intravenous)
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import xarray as xr

from pkpdutils.nca.auc import auc_aumc, insert_point, interpolate_at, pack_valid
from pkpdutils.nca.options import (
    AUCMethod,
    BLQHandling,
    C0Method,
    Kind,
    NCAFlag,
    NCAOptions,
    TerminalMethod,
    UncertaintyMethod,
)
from pkpdutils.nca.result import NCAResult, parameter_unit
from pkpdutils.nca.terminal import terminal_fit
from pkpdutils.nca.uncertainty import base_name, bootstrap
from pkpdutils.timecourse import Route, Timecourse, Timecourses

logger = logging.getLogger(__name__)

#: unit expression per parameter, see `pkpdutils.nca.result.parameter_unit`
PARAMETER_UNITS: dict[str, str] = {
    "cmax": "{unit}",
    "tmax": "{time}",
    "cmin": "{unit}",
    "tmin": "{time}",
    "clast": "{unit}",
    "tlast": "{time}",
    "c0": "{unit}",
    "cmax_half": "{unit}",
    "tmax_half": "{time}",
    "auc_last": "({unit}) * ({time})",
    "auc_inf_obs": "({unit}) * ({time})",
    "auc_inf_pred": "({unit}) * ({time})",
    "auc_extrap_fraction": "dimensionless",
    "aumc_last": "({unit}) * ({time}) ** 2",
    "aumc_inf": "({unit}) * ({time}) ** 2",
    "mrt": "{time}",
    "lambda_z": "1 / ({time})",
    "lambda_z_stderr": "1 / ({time})",
    "lambda_z_intercept": "dimensionless",
    "lambda_z_r2": "dimensionless",
    "lambda_z_r2_adj": "dimensionless",
    "lambda_z_n_points": "dimensionless",
    "lambda_z_t_first": "{time}",
    "thalf": "{time}",
    "cl": "({dose}) / (({unit}) * ({time}))",
    "cl_f": "({dose}) / (({unit}) * ({time}))",
    "vz": "({dose}) / ({unit})",
    "vz_f": "({dose}) / ({unit})",
    "vss": "({dose}) / ({unit})",
    "auc_inf_dn": "(({unit}) * ({time})) / ({dose})",
    "cmax_dn": "({unit}) / ({dose})",
    "e0": "{unit}",
    "emax_obs": "{unit}",
    "temax": "{time}",
    "auec_last": "({unit}) * ({time})",
    "auec_baseline": "({unit}) * ({time})",
    "emax_baseline": "{unit}",
    "time_above": "{time}",
    "auc_tau": "({unit}) * ({time})",
    "cmin_ss": "{unit}",
    "cmax_ss": "{unit}",
    "ctrough": "{unit}",
    "cavg": "{unit}",
    "fluctuation": "dimensionless",
    "swing": "dimensionless",
    "accumulation_ratio": "dimensionless",
    "cl_ss": "({dose}) / (({unit}) * ({time}))",
    "flags": "dimensionless",
}


def unit_expression(name: str) -> str:
    """Unit expression of a result variable, derived variables from their parameter.

    Args:
        name: name of a variable of the result, e.g. `"auc_last"`, `"auc_last_se"`
            or `"n"`.

    Returns:
        The unit expression of `PARAMETER_UNITS`, the one of the parameter a
        derived variable belongs to, or `"dimensionless"` for `n` and the
        dimensionless derived variables.

    Raises:
        KeyError: if the name belongs to no known parameter.
    """
    if name in PARAMETER_UNITS:
        return PARAMETER_UNITS[name]
    if name == "n" or name.endswith(("_geocv", "_n")):
        return "dimensionless"
    base = base_name(name)
    if base is None or base not in PARAMETER_UNITS:
        raise KeyError(f"No unit expression for '{name}'")
    return PARAMETER_UNITS[base]


def _take(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Element `idx[i]` of row `i`.

    Args:
        a: array `(N, n)`
        idx: one column index per row `(N,)`

    Returns:
        The selected elements `(N,)`.
    """
    return np.take_along_axis(a, idx[:, None], axis=1)[:, 0]


def _apply_lloq(c: np.ndarray, options: NCAOptions) -> tuple[np.ndarray, np.ndarray]:
    """Remove values below the limit of quantification.

    Args:
        c: values `(N, n)`
        options: the options, `lloq` and `blq` are used

    Returns:
        The values and a boolean array of the rows with removed values.
    """
    if options.lloq is None:
        return c, np.zeros(c.shape[0], dtype=bool)
    with np.errstate(invalid="ignore"):
        below = c < options.lloq
    truncated = below.any(axis=1)
    if options.blq is BLQHandling.NAN:
        return np.where(below, np.nan, c), truncated
    kept = np.where(below, np.nan, c)
    all_nan = np.isnan(kept).all(axis=1)
    masked = np.where(np.isnan(kept), -np.inf, kept)
    tmax_idx = np.where(all_nan, 0, masked.argmax(axis=1))
    before = np.arange(c.shape[1])[None, :] < tmax_idx[:, None]
    return np.where(below & before, 0.0, kept), truncated


def _effect_parameters(
    t: np.ndarray, c: np.ndarray, options: NCAOptions
) -> dict[str, np.ndarray]:
    """Parameters of effect timecourses (`Kind.EFFECT`).

    Args:
        t: times `(N, n)`
        c: values `(N, n)`
        options: the options, `effect_threshold` is used

    Returns:
        One `(N,)` array per parameter and `flags`.
    """
    tp, cp, n_valid = pack_valid(t, c)
    _, n = tp.shape
    has_data = n_valid >= 2
    flags = np.where(has_data, 0, NCAFlag.NO_DATA).astype(np.int64)
    idx = np.arange(n)[None, :]
    in_row = idx < n_valid[:, None]
    e0 = np.where(n_valid > 0, cp[:, 0], np.nan)
    masked = np.where(in_row, cp, -np.inf)
    imax = masked.argmax(axis=1)
    emax = np.where(has_data, _take(cp, imax), np.nan)
    temax = np.where(has_data, _take(tp, imax), np.nan)
    last_idx = np.clip(n_valid - 1, 0, n - 1)
    tlast = np.where(has_data, _take(tp, last_idx), np.nan)
    auec, _ = auc_aumc(tp, cp, n_valid, AUCMethod.LINEAR)
    auec_base, _ = auc_aumc(tp, cp - e0[:, None], n_valid, AUCMethod.LINEAR)
    out: dict[str, np.ndarray] = {
        "e0": e0,
        "emax_obs": emax,
        "temax": temax,
        "tlast": tlast,
        "auec_last": np.where(has_data, auec, np.nan),
        "auec_baseline": np.where(has_data, auec_base, np.nan),
        "emax_baseline": emax - e0,
    }
    if options.effect_threshold is not None:
        out["time_above"] = np.where(
            has_data, _time_above(tp, cp, n_valid, options.effect_threshold), np.nan
        )
    out["flags"] = flags
    return out


def _time_above(
    tp: np.ndarray, cp: np.ndarray, n_valid: np.ndarray, threshold: float
) -> np.ndarray:
    """Total time the linearly interpolated curve is above a threshold, per row.

    Args:
        tp: packed times `(N, n)`
        cp: packed values `(N, n)`
        n_valid: valid points per row
        threshold: the threshold

    Returns:
        The total time above the threshold `(N,)`.
    """
    t1, t2 = tp[:, :-1], tp[:, 1:]
    c1, c2 = cp[:, :-1], cp[:, 1:]
    in_curve = np.arange(tp.shape[1] - 1)[None, :] < (n_valid - 1)[:, None]
    dt = t2 - t1
    with np.errstate(invalid="ignore"):
        a1 = c1 > threshold
        a2 = c2 > threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (threshold - c1) / (c2 - c1)  # position of the crossing in the segment
    both = np.where(a1 & a2, dt, 0.0)
    rising = np.where(~a1 & a2, dt * (1.0 - frac), 0.0)
    falling = np.where(a1 & ~a2, dt * frac, 0.0)
    total = np.where(in_curve, both + rising + falling, 0.0)
    return np.nansum(total, axis=1)


def compute_parameters(
    t: np.ndarray,
    c: np.ndarray,
    *,
    dose_amount: np.ndarray | None,
    dose_time: np.ndarray | None,
    dose_duration: np.ndarray | None,
    route: Route | None,
    options: NCAOptions,
) -> dict[str, np.ndarray]:
    """Single dose parameters of every row of `(N, n)` time and value arrays.

    Args:
        t: times `(N, n)`, `NaN` for missing points
        c: values `(N, n)`, `NaN` for missing values
        dose_amount: dose per row `(N,)`, `None` without doses
        dose_time: time of the dose per row, `None` for 0
        dose_duration: infusion duration per row (`NaN` without infusion), `None` for none
        route: route of the batch, `None` without doses
        options: the options

    Returns:
        One `(N,)` array per parameter (see `PARAMETER_UNITS`) and `flags`.
    """
    t = np.asarray(t, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if dose_time is not None:
        t = t - dose_time[:, None]
    if options.kind is Kind.EFFECT:
        return _effect_parameters(t, c, options)

    c, truncated = _apply_lloq(c, options)
    tp, cp, n_valid = pack_valid(t, c)
    n_rows, n = tp.shape
    idx = np.arange(n)[None, :]
    in_row = idx < n_valid[:, None]
    has_data = n_valid >= 2
    flags = np.where(has_data, 0, NCAFlag.NO_DATA).astype(np.int64)
    flags |= np.where(truncated, NCAFlag.BLQ_TRUNCATED, 0)
    nan = np.full(n_rows, np.nan)

    # observed maxima and minima
    masked_max = np.where(in_row, cp, -np.inf)
    imax = masked_max.argmax(axis=1)
    cmax = np.where(has_data, _take(cp, imax), nan)
    tmax = np.where(has_data, _take(tp, imax), nan)
    masked_min = np.where(in_row, cp, np.inf)
    imin = masked_min.argmin(axis=1)
    cmin = np.where(has_data, _take(cp, imin), nan)
    tmin = np.where(has_data, _take(tp, imin), nan)
    flags |= np.where(has_data & (imax == n_valid - 1), NCAFlag.NO_MAX, 0)
    if route is Route.ORAL:
        flags |= np.where(has_data & (imax == 0), NCAFlag.NO_ABSORPTION, 0)

    # last measurable point
    with np.errstate(invalid="ignore"):
        positive = in_row & (cp > 0)
    has_positive = positive.any(axis=1)
    ilast = np.where(has_positive, n - 1 - positive[:, ::-1].argmax(axis=1), 0)
    clast = np.where(has_data & has_positive, _take(cp, ilast), nan)
    tlast = np.where(has_data & has_positive, _take(tp, ilast), nan)

    # C0 of an intravenous bolus, inserted at t = 0 for the areas
    c0 = nan.copy()
    tp_area, cp_area, n_area = tp, cp, n_valid
    if route is Route.IV_BOLUS:
        t1, c1 = tp[:, 0], cp[:, 0]
        t2 = np.where(n_valid > 1, tp[:, 1], np.nan)
        c2 = np.where(n_valid > 1, cp[:, 1], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            back = np.exp(np.log(c1) - (np.log(c2) - np.log(c1)) / (t2 - t1) * t1)
            usable = (c1 > 0) & (c2 > 0) & (c2 < c1) & (t2 > t1)
        if options.c0_method is C0Method.LOG_BACK_EXTRAPOLATION:
            c0 = np.where(usable, back, c1)
        else:
            c0 = c1
        c0 = np.where(has_data, c0, nan)
        with np.errstate(invalid="ignore"):
            insert = has_data & (t1 > 0)
        tp_area, cp_area, n_area = insert_point(
            tp, cp, n_valid, np.where(insert, 0.0, np.nan), np.where(insert, c0, np.nan)
        )

    auc_last, aumc_last = auc_aumc(
        tp_area, cp_area, n_area, options.auc_method, t_end=tlast
    )
    auc_last = np.where(has_data & has_positive, auc_last, nan)
    aumc_last = np.where(has_data & has_positive, aumc_last, nan)

    # terminal phase
    manual_mask = None
    if options.terminal.method is TerminalMethod.MANUAL:
        assert options.terminal.points is not None
        original = np.zeros_like(c, dtype=bool)
        original[:, list(options.terminal.points)] = True
        # the packed position of the selected original points
        valid = np.isfinite(t) & np.isfinite(c)
        order = np.argsort(~valid, axis=1, kind="stable")
        manual_mask = np.take_along_axis(original & valid, order, axis=1)
    fit = terminal_fit(tp, cp, n_valid, imax, options.terminal, manual_mask=manual_mask)
    flags |= np.where(has_data, fit.flags, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lambda_z = -fit.slope
        thalf = np.log(2.0) / lambda_z
        auc_inf_obs = auc_last + clast / lambda_z
        clast_pred = np.exp(fit.intercept - lambda_z * tlast)
        auc_inf_pred = auc_last + clast_pred / lambda_z
        extrap = (auc_inf_obs - auc_last) / auc_inf_obs
        aumc_inf = aumc_last + clast * tlast / lambda_z + clast / (lambda_z * lambda_z)
        mrt = aumc_inf / auc_inf_obs
        if route is Route.IV_INFUSION and dose_duration is not None:
            mrt = mrt - np.where(np.isnan(dose_duration), 0.0, dose_duration) / 2.0
        flags |= np.where(
            extrap > options.extrapolation_warning, NCAFlag.EXTRAPOLATION_HIGH, 0
        )

    # half maximum during absorption
    before_max = in_row & (idx < imax[:, None])
    with np.errstate(invalid="ignore"):
        distance = np.where(before_max, np.abs(cp - 0.5 * cmax[:, None]), np.inf)
    ihalf = distance.argmin(axis=1)
    has_half = has_data & (imax > 0) & (route is Route.ORAL)
    cmax_half = np.where(has_half, _take(cp, ihalf), nan)
    tmax_half = np.where(has_half, _take(tp, ihalf), nan)

    out: dict[str, np.ndarray] = {
        "cmax": cmax,
        "tmax": tmax,
        "cmin": cmin,
        "tmin": tmin,
        "clast": clast,
        "tlast": tlast,
        "auc_last": auc_last,
        "auc_inf_obs": auc_inf_obs,
        "auc_inf_pred": auc_inf_pred,
        "auc_extrap_fraction": extrap,
        "aumc_last": aumc_last,
        "aumc_inf": aumc_inf,
        "mrt": mrt,
        "lambda_z": lambda_z,
        "lambda_z_stderr": fit.se_slope,
        "lambda_z_intercept": fit.intercept,
        "lambda_z_r2": fit.r2,
        "lambda_z_r2_adj": fit.r2_adj,
        "lambda_z_n_points": fit.n_points,
        "lambda_z_t_first": fit.t_first,
        "thalf": thalf,
    }
    if route is Route.IV_BOLUS:
        out["c0"] = c0
    if route is Route.ORAL:
        out["cmax_half"] = cmax_half
        out["tmax_half"] = tmax_half
    if dose_amount is not None and route is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            cl = dose_amount / auc_inf_obs
            vz = cl / lambda_z
            suffix = "" if route.is_iv else "_f"
            out[f"cl{suffix}"] = cl
            out[f"vz{suffix}"] = vz
            if route.is_iv:
                out["vss"] = cl * mrt
            out["auc_inf_dn"] = auc_inf_obs / dose_amount
            out["cmax_dn"] = cmax / dose_amount
    out["flags"] = flags
    return out


def _compute_chunk(args: tuple[Any, ...]) -> dict[str, np.ndarray]:
    """Worker entry: the parameters of a chunk of rows.

    The steady state analysis runs through the same entry, so a batch with a
    `regimen` is chunked and parallelized like a single dose batch.

    Args:
        args: the arguments of `compute_parameters` as a tuple.

    Returns:
        The parameters of the rows of the chunk.
    """
    t, c, dose_amount, dose_time, dose_duration, route, options = args
    if options.regimen is not None:
        # the steady state analysis imports this module, so the import is local
        from pkpdutils.nca.steady_state import compute_steady_state

        return compute_steady_state(
            t,
            c,
            dose_amount=dose_amount,
            dose_time=dose_time,
            dose_duration=dose_duration,
            route=route,
            options=options,
        )
    return compute_parameters(
        t,
        c,
        dose_amount=dose_amount,
        dose_time=dose_time,
        dose_duration=dose_duration,
        route=route,
        options=options,
    )


def run_rows(
    t: np.ndarray,
    c: np.ndarray,
    *,
    dose_amount: np.ndarray | None,
    dose_time: np.ndarray | None,
    dose_duration: np.ndarray | None,
    route: Route | None,
    options: NCAOptions,
) -> dict[str, np.ndarray]:
    """Run the core on `(N, n)` arrays in chunks, serially or in the worker pool.

    The rows are analysed in chunks of at most `options.chunk_rows` rows, which
    bounds the memory of the vectorized core; with `options.n_workers > 1` the
    chunks are mapped in order over a `ProcessPoolExecutor`.

    Args:
        t: times `(N, n)`
        c: values `(N, n)`
        dose_amount: dose per row, `None` without doses
        dose_time: dose time per row, `None` for 0
        dose_duration: infusion duration per row, `None` for none
        route: route of the batch
        options: the options

    Returns:
        One `(N,)` array per parameter and `flags`.
    """
    n_rows = t.shape[0]
    # the rows are analysed in chunks of at most `chunk_rows` rows, which bounds
    # the memory of the vectorized core; the worker pool maps the chunks in order
    n_chunks = max(1, -(-n_rows // options.chunk_rows))
    jobs = [
        (
            t[rows],
            c[rows],
            None if dose_amount is None else dose_amount[rows],
            None if dose_time is None else dose_time[rows],
            None if dose_duration is None else dose_duration[rows],
            route,
            options,
        )
        for rows in np.array_split(np.arange(n_rows), n_chunks)
    ]
    if options.n_workers is not None and options.n_workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=options.n_workers) as pool:
            parts = list(pool.map(_compute_chunk, jobs))
    else:
        parts = [_compute_chunk(job) for job in jobs]
    names = list(parts[0])
    assert all(list(part) == names for part in parts), (
        "the chunks returned different parameters"
    )
    return {name: np.concatenate([part[name] for part in parts]) for name in names}


def nca(timecourses: Timecourses, options: NCAOptions | None = None) -> NCAResult:
    """Non-compartmental analysis of a batch of timecourses.

    The rows are analysed in chunks of `options.chunk_rows` rows, in the calling
    process or, with `options.n_workers`, in a pool of worker processes; a
    steady state analysis (`options.regimen`) is chunked the same way.

    A batch of group curves (`sd` or `se` per point) also carries the
    uncertainty of every parameter, by default from the parametric bootstrap
    (`options.uncertainty`, `pkpdutils.nca.uncertainty`): `x_sd`, `x_se`,
    `x_ci_low`, `x_ci_high` and, for log-normal parameters, `x_geomean`,
    `x_geocv`.

    Args:
        timecourses: the batch
        options: the options, defaults for `None`

    Returns:
        The parameters, their uncertainty variables and the number of subjects
        `n` over the sample dimensions of the batch.
    """
    options = options or NCAOptions()
    shape = timecourses.sample_shape
    n_rows = timecourses.n_samples
    t = timecourses.times.reshape(n_rows, timecourses.n_time)
    c = timecourses.values.reshape(n_rows, timecourses.n_time)

    def flat(a: np.ndarray | None) -> np.ndarray | None:
        """Flatten an optional per sample array to `(n_rows,)`.

        Args:
            a: the array, or `None`.

        Returns:
            The flattened array, or `None`.
        """
        return None if a is None else np.asarray(a, dtype=np.float64).reshape(n_rows)

    dose_amount = flat(timecourses.dose_amount)
    dose_time = flat(timecourses.dose_time)
    dose_duration = flat(timecourses.dose_duration)
    route = timecourses.route

    values = run_rows(
        t,
        c,
        dose_amount=dose_amount,
        dose_time=dose_time,
        dose_duration=dose_duration,
        route=route,
        options=options,
    )

    # `flags` is the last variable of the result, the uncertainty variables and
    # `n` go before it
    flags = values.pop("flags")
    method = options.resolve_uncertainty(timecourses.has_uncertainty)
    if method is UncertaintyMethod.BOOTSTRAP:
        values.update(bootstrap(timecourses, options, values))
    n_subjects = timecourses.n
    values["n"] = (
        np.full(n_rows, np.nan)
        if n_subjects is None
        else np.asarray(n_subjects, dtype=np.float64).reshape(n_rows)
    )
    values["flags"] = flags

    n_flagged = int((flags != 0).sum())
    if n_flagged:
        logger.info(
            "NCA: %d of %d samples carry flags, see NCAResult.flag_table()",
            n_flagged,
            n_rows,
        )
    return _to_result(values, timecourses, shape)


def _to_result(
    values: dict[str, np.ndarray], timecourses: Timecourses, shape: tuple[int, ...]
) -> NCAResult:
    """Build the result dataset over the sample dimensions of the batch.

    Args:
        values: one `(N,)` array per parameter
        timecourses: the analysed batch
        shape: the sample shape the arrays are reshaped to

    Returns:
        The result.
    """
    coords = {
        d: timecourses.ds[d]
        for d in timecourses.sample_dims
        if d in timecourses.ds.coords
    }
    data_vars: dict[str, Any] = {}
    for name, array in values.items():
        unit, factor = parameter_unit(
            unit_expression(name),
            unit=timecourses.unit,
            time_unit=timecourses.time_unit,
            dose_unit=timecourses.dose_unit,
        )
        if name == "flags":
            data_vars[name] = (
                timecourses.sample_dims,
                array.reshape(shape).astype(np.int64),
                {"units": unit},
            )
        else:
            data_vars[name] = (
                timecourses.sample_dims,
                (array * factor).reshape(shape),
                {"units": unit},
            )
    ds = xr.Dataset(
        data_vars=data_vars, coords=coords, attrs={"substance": timecourses.substance}
    )
    return NCAResult(ds)


def nca_single(timecourse: Timecourse, options: NCAOptions | None = None) -> NCAResult:
    """Non-compartmental analysis of one timecourse.

    Args:
        timecourse: the curve
        options: the options, defaults for `None`

    Returns:
        The parameters, without sample dimensions.
    """
    batch = Timecourses.from_timecourses([timecourse], dim="_single")
    result = nca(batch, options)
    return NCAResult(result.ds.isel(_single=0).drop_vars("_single"))


def partial_auc(
    timecourses: Timecourses,
    t_start: float,
    t_end: float,
    options: NCAOptions | None = None,
) -> xr.DataArray:
    """Area under the curve of every sample between two times relative to the dose.

    The values at the bounds are interpolated with the trapezoid rule of
    `options.auc_method` (`pkpdutils.nca.auc.interpolate_at`) and the area is
    summed with the same rule; a sample whose observed range does not cover
    `[t_start, t_end]` gives `NaN`.

    Args:
        timecourses: the batch
        t_start: start of the interval, in the time unit of the batch, relative to the dose
        t_end: end of the interval, greater than `t_start`
        options: the options, defaults for `None`

    Returns:
        The areas over the sample dimensions, named `auc_partial`, with the unit of `auc_last`.

    Raises:
        ValueError: if `t_end <= t_start`.
    """
    if t_end <= t_start:
        raise ValueError(
            f"'t_end' ({t_end}) must be greater than 't_start' ({t_start})"
        )
    options = options or NCAOptions()
    n_rows = timecourses.n_samples
    t = timecourses.times.reshape(n_rows, timecourses.n_time)
    c = timecourses.values.reshape(n_rows, timecourses.n_time)
    if timecourses.dose_time is not None:
        t = (
            t
            - np.asarray(timecourses.dose_time, dtype=np.float64).reshape(n_rows)[
                :, None
            ]
        )
    tp, cp, n_valid = pack_valid(t, c)
    start = np.full(n_rows, float(t_start))
    end = np.full(n_rows, float(t_end))
    c_start = interpolate_at(tp, cp, n_valid, start, options.auc_method)
    c_end = interpolate_at(tp, cp, n_valid, end, options.auc_method)
    tp, cp, n_valid = insert_point(tp, cp, n_valid, start, c_start)
    tp, cp, n_valid = insert_point(tp, cp, n_valid, end, c_end)
    area, _ = auc_aumc(tp, cp, n_valid, options.auc_method, t_start=start, t_end=end)
    area = np.where(np.isfinite(c_start) & np.isfinite(c_end), area, np.nan)
    unit, factor = parameter_unit(
        PARAMETER_UNITS["auc_last"],
        unit=timecourses.unit,
        time_unit=timecourses.time_unit,
        dose_unit=timecourses.dose_unit,
    )
    coords = {
        d: timecourses.ds[d]
        for d in timecourses.sample_dims
        if d in timecourses.ds.coords
    }
    return xr.DataArray(
        (area * factor).reshape(timecourses.sample_shape),
        dims=timecourses.sample_dims,
        coords=coords,
        name="auc_partial",
        attrs={"units": unit},
    )
