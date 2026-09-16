"""Steady state parameters of the last dosing interval and superposition.

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
- `CLss = Dose / AUC(0-tau)` (`cl_ss`, `cl_ss_f` for an extravascular route),
  the clearance of a multiple dose analysis: the single dose `CL`, `Vz`, `Vss`,
  `auc_inf_dn` and `cmax_dn` are `NaN` there (`SINGLE_DOSE_PARAMETERS`),
- the accumulation ratio `R = 1 / (1 - exp(-lambda_z tau))` predicted from the
  terminal phase, and `accumulation_ratio_obs`, the observed ratio of the
  exposure of the last and of the first dosing interval of the protocol.

`compute_steady_state` analyses the last dosing interval of the protocol of
every row, `[t_K, t_K + tau]`, where `tau` is `NCAOptions.tau` or the distance
of the last two doses; the parameters of every interval come from
`pkpdutils.nca.intervals`. The point parameters of the same rows are computed
from the last dose on: the values before it are dropped and the times are
relative to it, so that `cmax`, `tmax`, the terminal phase and the
extrapolated areas describe the last dosing interval and its decline; the
parameters which would read that slice as a single dose curve are dropped, see
`compute_steady_state`.

`superposition` predicts the multiple dose curve of a dosing protocol from a
single dose curve by adding the shifted, dose-scaled single dose curves
(linear superposition), which is valid for linear kinetics.
"""

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from pkpdutils.nca.auc import interpolate_at, pack_valid, take_rows
from pkpdutils.nca.intervals import compute_intervals
from pkpdutils.nca.nca import (
    compute_parameters,
    dose_counts,
    nca_single,
    positive_dose,
    reference_dose,
)
from pkpdutils.nca.options import Kind, NCAFlag, NCAOptions
from pkpdutils.nca.result import NCAResult
from pkpdutils.timecourse import Dosing, DosingRegimen, Route, Timecourse

#: single dose parameters which a multiple dose analysis does not report: they
#: divide the dose by the exposure of the slice after the last dose, which also
#: carries the exposure of the earlier doses (`compute_steady_state`)
SINGLE_DOSE_PARAMETERS: tuple[str, ...] = (
    "cl",
    "cl_f",
    "vz",
    "vz_f",
    "vss",
    "auc_inf_dn",
    "cmax_dn",
)


def _interval_length(times: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Length of the last dosing interval of every protocol, `t_K - t_{K-1}`.

    Args:
        times: dose times `(N, K)`, `NaN` padded
        counts: number of doses per row `(N,)`

    Returns:
        The length `(N,)`, `NaN` for a protocol of one dose.
    """
    n_dose = times.shape[1]
    last = np.clip(counts - 1, 0, n_dose - 1)
    previous = np.clip(counts - 2, 0, n_dose - 1)
    return np.where(
        counts >= 2, take_rows(times, last) - take_rows(times, previous), np.nan
    )


def compute_steady_state(
    t: np.ndarray,
    c: np.ndarray,
    *,
    dose_amount: np.ndarray | None,
    dose_time: np.ndarray | None,
    dose_duration: np.ndarray | None,
    route: Route | None,
    options: NCAOptions,
    lloq: np.ndarray | None = None,
    windows: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Point, per-interval and steady state parameters of every row of a batch.

    The dose arrays carry the dosing protocol of every row, `(N, K)` padded
    with `NaN` (`pkpdutils.timecourse.Timecourses`). The point parameters are
    computed from the last dose of every protocol on (the values before it are
    dropped), the per-interval parameters over every dosing interval
    (`pkpdutils.nca.intervals.compute_intervals`) and the steady state
    parameters from the last interval `[t_K, t_K + tau]`. A row whose last
    interval is not covered by the data carries
    `NCAFlag.INCOMPLETE_INTERVAL` and `NaN` steady state parameters.

    A multiple dose analysis reports no single dose quantities: the slice after
    the last dose carries the exposure of every earlier dose as well, so the
    parameters which divide the dose by it (`SINGLE_DOSE_PARAMETERS`: `cl`,
    `cl_f`, `vz`, `vz_f`, `vss`, `auc_inf_dn`, `cmax_dn`) are `NaN`. The
    clearance is `cl_ss` (`cl_ss_f` for an extravascular route), the dose over
    the exposure of the dosing interval. `auc_inf_obs`, `auc_inf_pred`,
    `aumc_inf` and `mrt` are reported and are the areas of that slice
    extrapolated with its terminal phase, i.e. the exposure after the last
    dose, not the single dose exposure of the substance.

    Args:
        t: times `(N, n)`
        c: values `(N, n)`

    Keyword Args:
        dose_amount: dose amounts `(N, K)`, `None` without doses
        dose_time: dose times `(N, K)`, `None` without doses (the interval of
            `options.tau` then starts at time 0)
        dose_duration: infusion durations `(N, K)`, `None` for none
        route: route of the batch
        options: the options; `tau` gives the length of the last interval when
            the protocol has one dose
        lloq: limit of quantification per row `(N,)`, `None` for none; it
            applies to the point parameters, as `NCAOptions.lloq` does
        windows: the terminal window of single rows `(N, 2)`, `NaN` for a row
            without one; it applies to the point parameters, whose times are
            relative to the last dose

    Returns:
        The parameters of `pkpdutils.nca.nca.compute_parameters` plus the
        steady state parameters, the per-interval parameters (with
        `options.intervals`) and `flags`.
    """
    t = np.asarray(t, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    n_rows = t.shape[0]
    amount, time, duration = reference_dose(
        dose_amount, dose_time, dose_duration, last=True
    )

    # the point parameters describe the curve from the last dose on
    values = c
    if time is not None:
        with np.errstate(invalid="ignore"):
            before = np.isfinite(time)[:, None] & (t < time[:, None])
        values = np.where(before, np.nan, c)
    out = compute_parameters(
        t,
        values,
        dose_amount=amount,
        dose_time=time,
        dose_duration=duration,
        route=route,
        options=options,
        lloq=lloq,
        windows=windows,
    )
    flags = out.pop("flags")

    # the dose-dependent parameters of a single dose analysis do not describe a
    # multiple dose curve: the slice after the last dose carries the exposure of
    # every earlier dose as well, so `CL = D / AUC(0-inf)` of that slice is
    # biased low. The clearance of the analysis is `cl_ss` over the dosing
    # interval; the extrapolated areas stay and describe the decline after the
    # last dose. Only the multiple dose rows of a batch reach this function
    # (`pkpdutils.nca.nca.is_multiple_dose`), so every row is blanked
    for name in SINGLE_DOSE_PARAMETERS:
        if name in out:
            out[name] = np.full(n_rows, np.nan)

    # a batch without a protocol but with `tau` is one interval from time 0
    times = (
        np.zeros((n_rows, 1))
        if dose_time is None
        else np.asarray(dose_time, dtype=np.float64).reshape(n_rows, -1)
    )
    amounts = (
        None
        if dose_amount is None
        else np.asarray(dose_amount, dtype=np.float64).reshape(n_rows, -1)
    )
    counts = np.isfinite(times).sum(axis=1)
    tau = (
        np.full(n_rows, float(options.tau))
        if options.tau is not None
        else _interval_length(times, counts)
    )
    intervals, extrapolated = compute_intervals(
        t,
        c,
        dose_amount=amounts,
        dose_time=times,
        tau=tau,
        route=route,
        options=options,
    )

    n_dose = times.shape[1]
    last = np.clip(counts - 1, 0, n_dose - 1)
    has_interval = counts > 0

    def last_interval(name: str) -> np.ndarray:
        """The value of the last dosing interval of every row.

        Args:
            name: name of the interval variable.

        Returns:
            The column of the last interval `(N,)`.
        """
        return np.where(has_interval, take_rows(intervals[name], last), np.nan)

    def first_interval(name: str) -> np.ndarray:
        """The value of the first dosing interval of every row.

        Args:
            name: name of the interval variable.

        Returns:
            The first column `(N,)`, `NaN` for a protocol of one dose.
        """
        return np.where(counts >= 2, intervals[name][:, 0], np.nan)

    area_name = (
        "interval_auc" if options.kind is Kind.CONCENTRATION else "interval_auec"
    )
    area = last_interval(area_name)
    with np.errstate(divide="ignore", invalid="ignore"):
        accumulation_obs = area / first_interval(area_name)
    if options.kind is Kind.CONCENTRATION:
        out["auc_tau"] = area
        out["cmin_ss"] = last_interval("interval_cmin")
        out["cmax_ss"] = last_interval("interval_cmax")
        out["ctrough"] = last_interval("interval_ctrough")
        out["cavg"] = last_interval("interval_cavg")
        out["fluctuation"] = last_interval("interval_fluctuation")
        out["swing"] = last_interval("interval_swing")
        with np.errstate(divide="ignore", invalid="ignore"):
            out["accumulation_ratio"] = 1.0 / (1.0 - np.exp(-out["lambda_z"] * tau))
        out["accumulation_ratio_obs"] = accumulation_obs
        if amount is not None and route is not None:
            # as `compute_parameters` does: without the fraction absorbed the
            # clearance of an extravascular dose is `CL/F`
            suffix = "" if route.is_iv else "_f"
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"cl_ss{suffix}"] = positive_dose(amount) / area
    else:
        out["auec_tau"] = area
        out["emin_ss"] = last_interval("interval_emin")
        out["emax_ss"] = last_interval("interval_emax")
        out["eavg"] = last_interval("interval_eavg")
        out["time_above_tau"] = last_interval("interval_time_above")
        out["accumulation_ratio_obs"] = accumulation_obs
    # a batch without a protocol carries no dose: the interval of `tau` only
    out["n_doses"] = dose_counts(dose_time, n_rows)
    out["tau"] = np.where(has_interval, tau, np.nan)

    with np.errstate(invalid="ignore"):
        incomplete = has_interval & np.isfinite(tau) & ~np.isfinite(area)
    flags = flags | np.where(incomplete, int(NCAFlag.INCOMPLETE_INTERVAL), 0).astype(
        flags.dtype
    )
    flags = flags | np.where(extrapolated, int(NCAFlag.EXTRAPOLATED_TROUGH), 0).astype(
        flags.dtype
    )
    if options.intervals:
        out.update(intervals)
    out["flags"] = flags
    return out


def accumulation_ratio(steady_state: NCAResult, single_dose: NCAResult) -> xr.DataArray:
    """Observed accumulation ratio `AUC(0-tau) at steady state / AUC(0-tau) after a single dose`.

    Both results come from analyses over the same dosing interval, so both
    carry `auc_tau`. Within one multiple dose curve the ratio of the last and
    the first dosing interval is reported as `accumulation_ratio_obs`.

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
            raise ValueError(
                "Both results need 'auc_tau': analyse with a dosing protocol or 'tau'"
            )
    ratio = steady_state["auc_tau"] / single_dose["auc_tau"]
    ratio.attrs["units"] = "dimensionless"
    return ratio.rename("accumulation_ratio")


def superposition(
    timecourse: Timecourse,
    dosing: Dosing | DosingRegimen,
    *,
    options: NCAOptions | None = None,
    t_end: float | None = None,
    grid: ArrayLike | None = None,
) -> Timecourse:
    r"""Predict the multiple dose curve of a protocol from a single dose curve.

    Every dose of the protocol contributes the single dose curve shifted to its
    time and scaled by `amount_k / amount_single`, the linear superposition
    which holds for linear kinetics (Gabrielsson & Weiner 2016, ch. 2.8). The
    curve is interpolated on the union of the shifted time grids, or on `grid`,
    and continued beyond its last observed point with its terminal phase.

    Before the first observed point after a dose the curve runs in a straight
    line from the value at the dose, the back-extrapolated \(C_0\) of a bolus
    (`c0` of the analysis) and 0 for every other route, to that point. The
    predicted curve carries a sample right before every dose after the
    first, a thousandth of the shortest dosing interval ahead of the dose
    time: the sample at the dose time carries the post-dose value, so without
    the pre-dose sample the curve of a bolus would rise to the next peak in a
    straight line from the last sample of the interval instead of falling to
    the trough and jumping. The trough of every interval is therefore in the
    curve, and a figure of the prediction shows the sawtooth of a bolus.

    The reference amount is the dose of the single dose curve; a curve whose
    dose amount is 0 carries no scale, so every dose of the protocol then
    contributes the curve unscaled.

    Args:
        timecourse: the single dose curve (its dose is the reference amount)
        dosing: the protocol to superpose, or a `DosingRegimen` with `n_doses`

    Keyword Args:
        options: NCA options for the interpolation and the terminal phase
        t_end: end of the predicted curve, the last dose time plus the last
            observed time by default
        grid: the times to predict at, from the first dose to `t_end`; by
            default the union of the observed times shifted to every dose,
            which is as sparse as the observed curve. A fine grid
            (`np.arange(0, 120, 0.25)`) gives a smooth curve of a figure. The
            pre-dose samples are added either way.

    Returns:
        The predicted curve carrying the protocol, without a label: the label
        of the single dose curve describes that curve, not the prediction.

    Raises:
        ValueError: without `n_doses` of a regimen, without a dose of the
            single dose curve or without a terminal phase of the curve.
    """
    protocol = dosing.dosing() if isinstance(dosing, DosingRegimen) else dosing
    options = options or NCAOptions()
    if timecourse.dose is None:
        raise ValueError("The single dose curve needs a dose to scale the protocol")
    amount_single = timecourse.dose.amount
    single = timecourse.relative_to_dose()
    q = nca_single(single, options=options).to_quantities()
    lambda_z = float(q["lambda_z"].magnitude)
    if not np.isfinite(lambda_z):
        raise ValueError(
            "The single dose curve has no terminal phase (lambda_z is NaN)"
        )
    tlast = float(q["tlast"].magnitude)
    clast = float(q["clast"].magnitude)
    # a bolus starts at its back-extrapolated C0, every other route at 0
    c_start = 0.0
    if single.dose is not None and single.dose.route is Route.IV_BOLUS:
        c0 = float(q["c0"].magnitude) if "c0" in q else np.nan
        c_start = c0 if np.isfinite(c0) else 0.0

    dose_times = protocol.times
    factors = (
        protocol.amounts / amount_single
        if amount_single
        else np.ones_like(protocol.amounts)
    )
    end = t_end if t_end is not None else float(dose_times[-1] + single.time[-1])
    if grid is None:
        times = np.concatenate([single.time + d for d in dose_times])
    else:
        times = np.asarray(grid, dtype=float).ravel()
    if dose_times.size > 1:
        # the trough right before every later dose, see the docstring
        ahead = 1e-3 * float(np.min(np.diff(dose_times)))
        times = np.concatenate([times, dose_times[1:] - ahead])
    grid = np.unique(times)
    grid = grid[(grid >= dose_times[0]) & (grid <= end)]

    tp, cp, n_valid = pack_valid(single.time[None, :], single.value[None, :])
    total = np.zeros_like(grid)
    for d, factor in zip(dose_times, factors, strict=True):
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
            # before the first observed point after a dose: a straight line
            # from the start value at the dose (C0 of a bolus, 0 otherwise)
            before = (tau_rel < single.time[0]) & (tau_rel >= 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rise = c_start + (single.value[0] - c_start) * tau_rel / single.time[0]
            values = np.where(before, rise, values)
        total = total + factor * values
    return Timecourse(
        time=grid,
        value=total,
        time_unit=single.time_unit,
        unit=single.unit,
        dosing=protocol,
        substance=single.substance,
        tissue=single.tissue,
    )
