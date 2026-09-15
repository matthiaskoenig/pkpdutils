"""Steady state parameters and superposition.

At steady state under repeated dosing every dosing interval `tau` looks the
same; the exposure over one interval, `AUC(0-tau)`, equals the single dose
`AUC(0-inf)` when the kinetics are linear (Gabrielsson & Weiner 2016, ch.
2.8; Rowland & Tozer 2011, ch. 11). The parameters of one interval are

- `AUC(0-tau)` with the values at the dose and at `tau` interpolated, so that
  samples before the dose do not add area,
- `Ctrough = C(tau)`, `Cmin,ss` and `Cmax,ss` the smallest and the largest value
  in the interval,
- `Cavg = AUC(0-tau) / tau`,
- `fluctuation = (Cmax,ss - Cmin,ss) / Cavg`,
  `swing = (Cmax,ss - Cmin,ss) / Cmin,ss`,
- `CLss = Dose / AUC(0-tau)`,
- the accumulation ratio `R = 1 / (1 - exp(-lambda_z tau))` predicted from the
  terminal phase, or observed as `AUC(0-tau)ss / AUC(0-tau)single`.

`superposition` predicts the multiple dose curve from a single dose curve by
adding the shifted single dose curves (linear superposition), which is valid
for linear kinetics.
"""

import numpy as np
import xarray as xr

from pkpdutils.nca.auc import auc_aumc, insert_point, interpolate_at, pack_valid
from pkpdutils.nca.nca import compute_parameters, nca_single
from pkpdutils.nca.options import NCAOptions
from pkpdutils.nca.result import NCAResult
from pkpdutils.timecourse import DosingRegimen, Route, Timecourse


def compute_steady_state(
    t: np.ndarray,
    c: np.ndarray,
    *,
    dose_amount: np.ndarray | None,
    dose_time: np.ndarray | None,
    dose_duration: np.ndarray | None,
    route: Route | None,
    options: NCAOptions,
) -> dict[str, np.ndarray]:
    """Single dose and steady state parameters of every row over the dosing interval.

    The dose arrays are one dose per row, the reference dose of the interval:
    `pkpdutils.nca.nca.reference_dose` picks the last dose of the protocol of
    every row of the batch before the chunk reaches this function.

    Args:
        t: times `(N, n)`
        c: values `(N, n)`
        dose_amount: amount of the reference dose per row `(N,)`, `None`
            without doses
        dose_time: time of the reference dose per row `(N,)`, `None` for the
            time of `options.regimen.dose`
        dose_duration: infusion duration of the reference dose per row `(N,)`
        route: route of the batch
        options: the options; `options.regimen` must be set

    Returns:
        The parameters of `compute_parameters` plus the steady state parameters.
    """
    regimen = options.regimen
    if regimen is None:
        raise ValueError("A steady state analysis needs 'options.regimen'")
    t = np.asarray(t, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if dose_time is None:
        dose_time = np.full(t.shape[0], regimen.dose.time)
    out = compute_parameters(
        t,
        c,
        dose_amount=dose_amount,
        dose_time=dose_time,
        dose_duration=dose_duration,
        route=route,
        options=options,
    )
    tau = regimen.interval
    tp, cp, n_valid = pack_valid(t - dose_time[:, None], c)
    n_rows = tp.shape[0]
    tau_row = np.full(n_rows, tau)
    zero_row = np.zeros(n_rows)
    if route is Route.IV_BOLUS and "c0" in out:
        # the interval starts at the dose: insert (0, C0) like the single dose areas do
        first_after_zero = (n_valid > 0) & (tp[:, 0] > 0) & np.isfinite(out["c0"])
        tp, cp, n_valid = insert_point(
            tp,
            cp,
            n_valid,
            np.where(first_after_zero, 0.0, np.nan),
            np.where(first_after_zero, out["c0"], np.nan),
        )
    # a curve with pre-dose samples starts before the dose: insert the interpolated
    # value at the dose, so a segment straddling the dose contributes only its
    # part after the dose to `auc_tau`
    c_zero = interpolate_at(tp, cp, n_valid, zero_row, options.auc_method)
    with np.errstate(invalid="ignore"):
        straddles = (n_valid > 0) & (tp[:, 0] < 0) & np.isfinite(c_zero)
    tp, cp, n_valid = insert_point(
        tp,
        cp,
        n_valid,
        np.where(straddles, 0.0, np.nan),
        np.where(straddles, c_zero, np.nan),
    )
    ctrough = interpolate_at(tp, cp, n_valid, tau_row, options.auc_method)
    tp2, cp2, n2 = insert_point(
        tp, cp, n_valid, np.where(np.isnan(ctrough), np.nan, tau_row), ctrough
    )
    in_interval = (
        (np.arange(tp2.shape[1])[None, :] < n2[:, None]) & (tp2 >= 0) & (tp2 <= tau)
    )
    # segments before the dose or after tau do not count
    auc_tau, _ = auc_aumc(
        tp2, cp2, n2, options.auc_method, t_start=zero_row, t_end=tau_row
    )
    with np.errstate(invalid="ignore"):
        cmin_ss = np.where(in_interval, cp2, np.inf).min(axis=1)
        cmax_ss = np.where(in_interval, cp2, -np.inf).max(axis=1)
    has_tau = ~np.isnan(ctrough)
    nan = np.full(n_rows, np.nan)
    auc_tau = np.where(has_tau, auc_tau, nan)
    cmin_ss = np.where(has_tau & np.isfinite(cmin_ss), cmin_ss, nan)
    cmax_ss = np.where(has_tau & np.isfinite(cmax_ss), cmax_ss, nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        cavg = auc_tau / tau
        fluctuation = (cmax_ss - cmin_ss) / cavg
        swing = (cmax_ss - cmin_ss) / cmin_ss
        accumulation = 1.0 / (1.0 - np.exp(-out["lambda_z"] * tau))
    flags = out.pop("flags")
    out["auc_tau"] = auc_tau
    out["cmin_ss"] = cmin_ss
    out["cmax_ss"] = cmax_ss
    out["ctrough"] = ctrough
    out["cavg"] = cavg
    out["fluctuation"] = fluctuation
    out["swing"] = swing
    out["accumulation_ratio"] = accumulation
    if dose_amount is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["cl_ss"] = dose_amount / auc_tau
    out["flags"] = flags
    return out


def accumulation_ratio(steady_state: NCAResult, single_dose: NCAResult) -> xr.DataArray:
    """Observed accumulation ratio `AUC(0-tau) at steady state / AUC(0-tau) after a single dose`.

    Both results come from analyses with the same `regimen`, so both carry `auc_tau`.

    Args:
        steady_state: result of the analysis of the steady state curve
        single_dose: result of the analysis of the single dose curve

    Returns:
        The observed accumulation ratio, `attrs["units"]` is `"dimensionless"`.

    Raises:
        ValueError: if either result has no `auc_tau`.
    """
    for result in (steady_state, single_dose):
        if "auc_tau" not in result:
            raise ValueError("Both results need 'auc_tau': analyse with a regimen")
    ratio = steady_state["auc_tau"] / single_dose["auc_tau"]
    ratio.attrs["units"] = "dimensionless"
    return ratio.rename("accumulation_ratio")


def superposition(
    timecourse: Timecourse,
    regimen: DosingRegimen,
    options: NCAOptions | None = None,
    t_end: float | None = None,
) -> Timecourse:
    """Predict the multiple dose curve from a single dose curve by superposition.

    Args:
        timecourse: the single dose curve (its dose time is the origin)
        regimen: the doses to superpose; `n_doses` is required
        options: NCA options for the interpolation and the terminal phase
        t_end: end of the predicted curve, the last dose time plus the last
            observed time by default

    Returns:
        The predicted curve with the dose of the regimen.

    Raises:
        ValueError: without `n_doses` or without a terminal phase of the curve.
    """
    if regimen.n_doses is None:
        raise ValueError("'regimen.n_doses' is required for the superposition")
    options = options or NCAOptions()
    single = timecourse.relative_to_dose()
    q = nca_single(single, options).to_quantities()
    lambda_z = float(q["lambda_z"].magnitude)
    if not np.isfinite(lambda_z):
        raise ValueError(
            "The single dose curve has no terminal phase (lambda_z is NaN)"
        )
    tlast = float(q["tlast"].magnitude)
    clast = float(q["clast"].magnitude)

    dose_times = regimen.dose_times() - regimen.dose.time
    end = t_end if t_end is not None else float(dose_times[-1] + single.time[-1])
    grid = np.unique(np.concatenate([single.time + d for d in dose_times]))
    grid = grid[(grid >= 0) & (grid <= end)]

    tp, cp, n_valid = pack_valid(single.time[None, :], single.value[None, :])
    total = np.zeros_like(grid)
    for d in dose_times:
        tau_rel = grid - d
        inside = (tau_rel >= single.time[0]) & (tau_rel <= tlast)
        rows = np.repeat(tp, grid.size, axis=0)
        cols = np.repeat(cp, grid.size, axis=0)
        interp = interpolate_at(
            rows, cols, np.repeat(n_valid, grid.size), tau_rel, options.auc_method
        )
        values = np.where(inside, np.nan_to_num(interp), 0.0)
        beyond = tau_rel > tlast
        values = np.where(beyond, clast * np.exp(-lambda_z * (tau_rel - tlast)), values)
        if single.time[0] > 0:
            # before the first observed point after a dose: linear rise from 0
            before = (tau_rel < single.time[0]) & (tau_rel >= 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rise = single.value[0] * tau_rel / single.time[0]
            values = np.where(before, rise, values)
        total = total + values
    return Timecourse(
        time=grid,
        value=total,
        time_unit=single.time_unit,
        unit=single.unit,
        dose=regimen.dose,
        substance=single.substance,
        label=single.label,
        tissue=single.tissue,
    )
