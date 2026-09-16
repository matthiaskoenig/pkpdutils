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
- `thalf_eff = ln 2 * MRT`, the effective half-life. The formula is the one of
  PKNCA (Denney et al. 2015), whose `pk.calc.thalf.eff` reads

  ```r
  #' @details thalf.eff is `log(2)*mrt`.
  pk.calc.thalf.eff <- function(mrt) {
    log(2)*mrt
  }
  ```

  and whose interval columns `thalf.eff.obs`, `thalf.eff.pred` and
  `thalf.eff.iv.*` all evaluate it with the mean residence time they name. It
  is reported by every concentration analysis, single dose and multiple dose,
  and it uses the `MRT` of the row, the infusion correction included
- `CL = Dose / AUC(0-inf)`, `Vz = CL / lambda_z`, `Vss = CL MRT` (intravenous)
"""

import itertools
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from pkpdutils.nca.auc import (
    auc_aumc,
    insert_point,
    interpolate_at,
    pack_valid,
    segment_areas,
    take_rows,
    time_above_threshold,
)
from pkpdutils.nca.intervals import INTERVAL_DIM, INTERVAL_UNITS
from pkpdutils.nca.options import (
    C0_BACK_EXTRAPOLATION,
    C0_FIRST_VALUE,
    C0_NONE,
    WINDOW_DEFAULT_KEY,
    Acceptance,
    AUCMethod,
    BLQAction,
    BLQRules,
    C0Method,
    Kind,
    NCAFlag,
    NCAOptions,
    TerminalMethod,
    TerminalPhase,
    UncertaintyMethod,
)
from pkpdutils.nca.result import (
    DOSE_COORDINATE,
    DOSE_NORMALIZED_SUFFIX,
    PARTIAL_AUCS_ATTR,
    NCAResult,
    parameter_unit,
)
from pkpdutils.nca.terminal import CANDIDATE_DIM, CANDIDATE_PREFIX, terminal_fit
from pkpdutils.nca.uncertainty import bootstrap, delta
from pkpdutils.parallel import (
    NCA_WORKER_THRESHOLD,
    executor,
    resolve_workers,
    split_rows,
)
from pkpdutils.result import (
    SUMMARY_SUFFIXES,
    UNCERTAINTY_SUFFIXES,
    base_name,
    check_coordinate_collision,
    sample_coordinates,
)
from pkpdutils.timecourse import Route, Timecourse, Timecourses

logger = logging.getLogger(__name__)

#: half-lives the terminal phase must cover for `lambda_z_span` to be accepted;
#: below it the row is flagged `NCAFlag.SPAN_LOW`
SPAN_MINIMUM: float = 2.0

#: variables of a result which are integer codes and not measurements; a row
#: group which does not report one of them carries its 0 (`merge_rows`)
INTEGER_VARIABLES: frozenset[str] = frozenset({"flags", "c0_method"})

#: variables of a result which are booleans and not measurements
BOOLEAN_VARIABLES: frozenset[str] = frozenset({"accepted", "excluded"})

#: variables of a result which carry text
TEXT_VARIABLES: frozenset[str] = frozenset({"excluded_reason"})

#: `excluded_reason` of a sample which `Acceptance(exclude=True)` excluded
ACCEPTANCE_REASON: str = "acceptance criteria"

#: unit expression per parameter, see `pkpdutils.nca.result.parameter_unit`
PARAMETER_UNITS: dict[str, str] = {
    "cmax": "{unit}",
    "tmax": "{time}",
    "cmin": "{unit}",
    "tmin": "{time}",
    "clast": "{unit}",
    "clast_pred": "{unit}",
    "tlast": "{time}",
    "tlag": "{time}",
    "c0": "{unit}",
    "c0_method": "dimensionless",
    "cmax_half": "{unit}",
    "tmax_half": "{time}",
    "auc_last": "({unit}) * ({time})",
    "auc_all": "({unit}) * ({time})",
    "auc_partial": "({unit}) * ({time})",
    "auc_inf_obs": "({unit}) * ({time})",
    "auc_inf_pred": "({unit}) * ({time})",
    "auc_extrap_fraction": "dimensionless",
    "auc_back_extrap_fraction": "dimensionless",
    "aumc_back_extrap_fraction": "dimensionless",
    "aumc_last": "({unit}) * ({time}) ** 2",
    "aumc_all": "({unit}) * ({time}) ** 2",
    "aumc_inf": "({unit}) * ({time}) ** 2",
    "mrt": "{time}",
    "thalf_eff": "{time}",
    "lambda_z": "1 / ({time})",
    "lambda_z_stderr": "1 / ({time})",
    "lambda_z_intercept": "dimensionless",
    "lambda_z_r2": "dimensionless",
    "lambda_z_r2_adj": "dimensionless",
    "lambda_z_n_points": "dimensionless",
    "lambda_z_t_first": "{time}",
    "lambda_z_t_last": "{time}",
    "lambda_z_span": "dimensionless",
    "thalf": "{time}",
    "cl": "({dose}) / (({unit}) * ({time}))",
    "cl_f": "({dose}) / (({unit}) * ({time}))",
    "vz": "({dose}) / ({unit})",
    "vz_f": "({dose}) / ({unit})",
    "vss": "({dose}) / ({unit})",
    "auc_inf_dn": "(({unit}) * ({time})) / ({dose})",
    "cmax_dn": "({unit}) / ({dose})",
    "auc_last_dn": "(({unit}) * ({time})) / ({dose})",
    "auc_all_dn": "(({unit}) * ({time})) / ({dose})",
    "auc_tau_dn": "(({unit}) * ({time})) / ({dose})",
    "cavg_dn": "({unit}) / ({dose})",
    "cmax_ss_dn": "({unit}) / ({dose})",
    "c0_dn": "({unit}) / ({dose})",
    "e0": "{unit}",
    "emax_obs": "{unit}",
    "temax": "{time}",
    "auec_last": "({unit}) * ({time})",
    "auec_baseline": "({unit}) * ({time})",
    "emax_baseline": "{unit}",
    "time_above": "{time}",
    "auc_tau": "({unit}) * ({time})",
    "auc_tau_extrap_fraction": "dimensionless",
    "cmin_ss": "{unit}",
    "cmax_ss": "{unit}",
    "ctrough": "{unit}",
    "cavg": "{unit}",
    "fluctuation": "dimensionless",
    "swing": "dimensionless",
    "fluctuation_tau": "dimensionless",
    "swing_tau": "dimensionless",
    "ptr": "dimensionless",
    "accumulation_ratio": "dimensionless",
    "accumulation_ratio_obs": "dimensionless",
    "accumulation_ratio_cmax_obs": "dimensionless",
    "accumulation_ratio_cmin_obs": "dimensionless",
    "accumulation_ratio_ctrough_obs": "dimensionless",
    "cl_ss": "({dose}) / (({unit}) * ({time}))",
    "cl_ss_f": "({dose}) / (({unit}) * ({time}))",
    "auec_tau": "({unit}) * ({time})",
    "emin_ss": "{unit}",
    "emax_ss": "{unit}",
    "eavg": "{unit}",
    "time_above_tau": "{time}",
    "n_doses": "dimensionless",
    "tau": "{time}",
    "accepted": "dimensionless",
    "excluded": "dimensionless",
    "excluded_reason": "dimensionless",
    "flags": "dimensionless",
    **INTERVAL_UNITS,
    # the urinary excretion analysis (`pkpdutils.nca.urine`), whose values are
    # amounts per time rather than concentrations
    "rate": "({amount}) / ({time})",
    "midpoint": "{time}",
    "max_rate": "({amount}) / ({time})",
    "tmax_rate": "{time}",
    "rate_last": "({amount}) / ({time})",
    "mid_pt_last": "{time}",
    "aurc_last": "{amount}",
    "aurc_all": "{amount}",
    "aurc_inf_obs": "{amount}",
    "aurc_inf_pred": "{amount}",
    "amount_recovered": "{amount}",
    "percent_recovered": "percent",
    "vol_ur": "{volume}",
    "clr": "({amount}) / (({unit}) * ({time}))",
    # the sparse sampling analysis (`pkpdutils.nca.sparse`)
    "auc_last_df": "dimensionless",
    "n_animals": "dimensionless",
    # the candidate windows of the terminal regression over the dimension
    # `candidate` (`TerminalPhase.keep_candidates`, `plot_terminal_windows`)
    "candidate_t_first": "{time}",
    "candidate_n_points": "dimensionless",
    "candidate_r2_adj": "dimensionless",
}


def unit_expression(name: str) -> str:
    """Unit expression of a result variable, derived variables from their parameter.

    Args:
        name: name of a variable of the result, e.g. `"auc_last"`, `"auc_last_se"`
            or `"n"`.

    Returns:
        The unit expression of `PARAMETER_UNITS`, the one of the parameter a
        derived variable belongs to, the expression of a parameter per dose for
        a dose normalized variable `x_dn` (`NCAResult.dose_normalized`), or
        `"dimensionless"` for `n` and the dimensionless derived variables.

    Raises:
        KeyError: if the name belongs to no known parameter.
    """
    if name in PARAMETER_UNITS:
        return PARAMETER_UNITS[name]
    if name == "n" or name.endswith(("_geocv", "_n")):
        return "dimensionless"
    if name.endswith(DOSE_NORMALIZED_SUFFIX):
        normalized = name[: -len(DOSE_NORMALIZED_SUFFIX)]
        if normalized in PARAMETER_UNITS:
            return f"({PARAMETER_UNITS[normalized]}) / ({{dose}})"
    base = base_name(name)
    if base is None or base not in PARAMETER_UNITS:
        raise KeyError(f"No unit expression for '{name}'")
    return PARAMETER_UNITS[base]


def positive_dose(dose_amount: np.ndarray) -> np.ndarray:
    """The dose amounts, `NaN` where a row carries no positive dose.

    A dose of 0 is the encoding of a placebo arm (`Dose.amount` is
    non-negative). The parameters which divide by the dose - the clearance, the
    volumes and the dose normalized exposure - are not defined for it, so the
    amount is `NaN` there and every one of them follows; the analysis reports
    this in a debug log and sets no flag, since a zero dose is a property of
    the data and not a finding of the analysis.

    Args:
        dose_amount: the reference dose per row `(N,)`

    Returns:
        The amounts with the non-positive ones replaced by `NaN`.
    """
    with np.errstate(invalid="ignore"):
        positive = dose_amount > 0
        n_zero = int((np.isfinite(dose_amount) & ~positive).sum())
    if n_zero:
        logger.debug(
            "%d of %d rows carry a dose of 0: their dose dependent parameters "
            "(cl, vz, vss, auc_inf_dn, cmax_dn) are NaN",
            n_zero,
            dose_amount.shape[0],
        )
    return np.where(positive, dose_amount, np.nan)


def bolus_c0(
    tp: np.ndarray, cp: np.ndarray, n_valid: np.ndarray, options: NCAOptions
) -> tuple[np.ndarray, np.ndarray]:
    r"""Concentration at time 0 of an intravenous bolus, per row.

    With `C0Method.LOG_BACK_EXTRAPOLATION` the first two samples are
    extrapolated back to the dose,

    $$C_0 = \exp\left(\ln C_1 - \frac{\ln C_2 - \ln C_1}{t_2 - t_1} t_1\right),$$

    the estimate of Gabrielsson & Weiner (2016, ch. 2.8). The back
    extrapolation needs two samples which decline, so it is used when the row
    carries two valid points, both values are positive, the second value is
    below the first and the second time is after the first; in every other case
    the first observed value is used, which is the documented fallback chain of
    Phoenix WinNonlin ("if the regression yields a slope >= 0, or at least one
    of the first two y-values is zero ... then the first observed y-value is
    used"). `C0Method.FIRST_VALUE` always takes the first value and
    `C0Method.NONE` estimates nothing: `c0` is `NaN`, no point is inserted and
    the areas start at the first sample.

    The inserted point never enters the terminal regression, which reads the
    observed values, and the rule of a row is reported in `c0_method`
    (`C0_NONE`, `C0_BACK_EXTRAPOLATION`, `C0_FIRST_VALUE`). For an
    extravascular single dose the value at the dose time is 0 and for a steady
    state interval the minimum observed value of the interval
    (`pkpdutils.nca.intervals`), neither of them an estimate of `C0`.

    Args:
        tp: packed times `(N, n)`, relative to the dose
        cp: packed values `(N, n)`
        n_valid: valid points per row `(N,)`
        options: the options, `c0_method` is used

    Returns:
        The estimate per row `(N,)` and the rule which produced it, one of
        `C0_NONE`, `C0_BACK_EXTRAPOLATION` and `C0_FIRST_VALUE` per row.
    """
    n_rows = tp.shape[0]
    if options.c0_method is C0Method.NONE:
        return np.full(n_rows, np.nan), np.full(n_rows, C0_NONE, dtype=np.int64)
    t1, c1 = tp[:, 0], cp[:, 0]
    has_point = n_valid >= 1
    first = np.where(has_point, c1, np.nan)
    method = np.where(has_point, C0_FIRST_VALUE, C0_NONE).astype(np.int64)
    if options.c0_method is not C0Method.LOG_BACK_EXTRAPOLATION:
        return first, method
    t2 = np.where(n_valid > 1, tp[:, 1], np.nan)
    c2 = np.where(n_valid > 1, cp[:, 1], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        back = np.exp(np.log(c1) - (np.log(c2) - np.log(c1)) / (t2 - t1) * t1)
        usable = has_point & (c1 > 0) & (c2 > 0) & (c2 < c1) & (t2 > t1)
    return (
        np.where(usable, back, first),
        np.where(usable, C0_BACK_EXTRAPOLATION, method).astype(np.int64),
    )


def resolve_lloq(
    options: NCAOptions, lloq: np.ndarray | None, n_rows: int
) -> np.ndarray | None:
    """The limit of quantification of every row.

    Args:
        options: the options, `lloq` is the limit of the whole analysis
        lloq: the limit of every row `(N,)` (the per-sample `lloq` of the
            batch), `None` without one
        n_rows: number of rows `N`

    Returns:
        One limit per row, `None` when neither the options nor the batch name
        one. `NCAOptions.lloq` wins over the per-sample limit; a row whose
        limit is `NaN` has none.
    """
    if options.lloq is not None:
        return np.full(n_rows, float(options.lloq))
    if lloq is None:
        return None
    return np.asarray(lloq, dtype=np.float64).reshape(n_rows)


def _imputed_value(
    action: BLQAction | float, c: np.ndarray, limit: np.ndarray
) -> np.ndarray:
    """The value an action writes in place of a value below the limit.

    Args:
        action: the action of the position, neither `DROP` nor `KEEP`
        c: values `(N, n)`, for the shape
        limit: the limit of quantification per row `(N, 1)`

    Returns:
        The imputed values `(N, n)`.

    Raises:
        ValueError: for an action which imputes nothing.
    """
    if not isinstance(action, BLQAction):
        return np.full(c.shape, float(action))
    if action is BLQAction.ZERO:
        return np.zeros(c.shape)
    if action is BLQAction.LLOQ:
        return np.broadcast_to(limit, c.shape)
    if action is BLQAction.HALF_LLOQ:
        return np.broadcast_to(0.5 * limit, c.shape)
    raise ValueError(f"'{action}' imputes no value")


def apply_blq(
    c: np.ndarray, lloq: np.ndarray | None, rules: BLQRules
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the rules for the values below the limit of quantification.

    The rules are read by position (`first`, `middle`, `last`) or against the
    maximum (`before_tmax`, `after_tmax`), see `BLQRules`; a position without a
    rule drops its values. A row without a single measurable value is `first`
    on the positional axis and `after_tmax` on the tmax axis.

    Args:
        c: values `(N, n)` in the time order of the curve
        lloq: limit of quantification per row `(N,)`, `None` for no limit
        rules: the rules

    Returns:
        The values, the rows in which a value was dropped or imputed
        (`NCAFlag.BLQ_TRUNCATED`) and the mask of the values below the limit
        which are still part of the curve, imputed or kept (`(N, n)`); the
        terminal regression leaves those out unless
        `BLQRules.terminal_regression`.
    """
    n_rows, n = c.shape
    if lloq is None:
        empty = np.zeros(c.shape, dtype=bool)
        return c, np.zeros(n_rows, dtype=bool), empty
    limit = np.asarray(lloq, dtype=np.float64).reshape(n_rows, 1)
    with np.errstate(invalid="ignore"):
        below = np.isfinite(c) & np.isfinite(limit) & (c < limit)
    measurable = np.isfinite(c) & ~below
    any_measurable = measurable.any(axis=1)
    idx = np.arange(n)[None, :]
    groups: list[tuple[np.ndarray, BLQAction | float | None]]
    if rules.by_tmax:
        masked = np.where(measurable, c, -np.inf)
        tmax_idx = np.where(any_measurable, masked.argmax(axis=1), 0)[:, None]
        groups = [
            (below & (idx < tmax_idx), rules.before_tmax),
            (below & (idx >= tmax_idx), rules.after_tmax),
        ]
    else:
        # a row without a measurable value has no first and no last one, so
        # `first` covers all of it (`n` is beyond every column)
        first_idx = np.where(any_measurable, measurable.argmax(axis=1), n)[:, None]
        last_idx = np.where(
            any_measurable, n - 1 - measurable[:, ::-1].argmax(axis=1), n
        )[:, None]
        groups = [
            (below & (idx < first_idx), rules.first),
            (below & (idx > first_idx) & (idx < last_idx), rules.middle),
            (below & (idx > last_idx), rules.last),
        ]
    values = c
    changed = np.zeros(c.shape, dtype=bool)
    in_curve = np.zeros(c.shape, dtype=bool)
    for mask, rule in groups:
        action = BLQAction.DROP if rule is None else rule
        if action is BLQAction.KEEP:
            in_curve |= mask
            continue
        if action is BLQAction.DROP:
            values = np.where(mask, np.nan, values)
            changed |= mask
            continue
        values = np.where(mask, _imputed_value(action, c, limit), values)
        changed |= mask
        in_curve |= mask
    return values, changed.any(axis=1), in_curve


def packed_mask(t: np.ndarray, c: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A mask of the original columns of a row in the layout of `pack_valid`.

    Args:
        t: times `(N, n)`, as they are packed
        c: values `(N, n)`, as they are packed
        mask: the mask over the original columns `(N, n)`

    Returns:
        The mask over the packed columns `(N, n)`; a column which is not a
        valid point is `False`.
    """
    valid = np.isfinite(t) & np.isfinite(c)
    order = np.argsort(~valid, axis=1, kind="stable")
    return np.take_along_axis(mask & valid, order, axis=1)


def _effect_parameters(
    t: np.ndarray, c: np.ndarray, truncated: np.ndarray, options: NCAOptions
) -> dict[str, np.ndarray]:
    """Parameters of effect timecourses (`Kind.EFFECT`).

    The areas of an effect timecourse are always linear: the logarithmic rules
    are exact for a mono-exponential decline of a concentration and need
    positive values, which an effect - a change against a baseline, possibly
    negative - does not have. `NCAOptions.auc_method` therefore does not apply
    here, while `lloq` and `blq` do and are applied by the caller.

    Args:
        t: times `(N, n)`
        c: values `(N, n)`, already cleaned of the values below `lloq`
        truncated: rows in which a value below `lloq` was replaced `(N,)`
        options: the options, `effect_threshold` is used

    Returns:
        One `(N,)` array per parameter and `flags`.
    """
    tp, cp, n_valid = pack_valid(t, c)
    _, n = tp.shape
    has_data = n_valid >= 2
    flags = np.where(has_data, 0, NCAFlag.NO_DATA).astype(np.int64)
    flags |= np.where(truncated, NCAFlag.BLQ_TRUNCATED, 0)
    idx = np.arange(n)[None, :]
    in_row = idx < n_valid[:, None]
    e0 = np.where(n_valid > 0, cp[:, 0], np.nan)
    masked = np.where(in_row, cp, -np.inf)
    imax = masked.argmax(axis=1)
    emax = np.where(has_data, take_rows(cp, imax), np.nan)
    temax = np.where(has_data, take_rows(tp, imax), np.nan)
    last_idx = np.clip(n_valid - 1, 0, n - 1)
    tlast = np.where(has_data, take_rows(tp, last_idx), np.nan)
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
            has_data,
            time_above_threshold(tp, cp, n_valid, options.effect_threshold),
            np.nan,
        )
    out["flags"] = flags
    return out


def _lag_time(
    tp: np.ndarray,
    cp: np.ndarray,
    in_row: np.ndarray,
    route: Route | None,
    blq: np.ndarray,
) -> np.ndarray:
    r"""Lag time of the absorption of an extravascular dose, per row.

    $$t_\mathrm{lag} = t_{j-1}, \qquad j = \min\{\, i : t_i \ge 0,\ C_i > 0 \,\},$$

    the time of the last sample before the first measurable value after the
    dose (Gabrielsson & Weiner 2016, ch. 2.8; Phoenix WinNonlin, which computes
    `Tlag` "only when the dosing type is extravascular"). The times are
    relative to the dose, so only samples at or after it are candidates: a
    pre-dose sample is not a lag of the absorption. A curve whose first sample
    at or after the dose is already measurable has no such sample and its lag
    is 0, the time of the dose, as Phoenix WinNonlin reports it. `NaN` when no
    value is measurable and for every intravenous route, which has no
    absorption phase.

    Args:
        tp: packed times `(N, n)`, relative to the dose
        cp: packed values `(N, n)`
        in_row: which packed columns are points of the row `(N, n)`
        route: route of the batch
        blq: packed values below the limit of quantification which a BLQ rule
            kept or imputed `(N, n)`; they are not measurable values

    Returns:
        The lag time per row `(N,)`.
    """
    n_rows, n = tp.shape
    if route is not Route.ORAL:
        return np.full(n_rows, np.nan)
    with np.errstate(invalid="ignore"):
        after_dose = in_row & (tp >= 0.0)
        measurable = after_dose & (cp > 0) & ~blq
    has_measurable = measurable.any(axis=1)
    first = np.where(has_measurable, measurable.argmax(axis=1), 0)
    # the sample before it, which must itself be a sample after the dose
    previous = np.clip(first - 1, 0, n - 1)
    has_lag = has_measurable & (first > 0) & take_rows(after_dose, previous)
    # without such a sample the absorption started at the dose: the lag is 0
    return np.where(
        has_lag, take_rows(tp, previous), np.where(has_measurable, 0.0, np.nan)
    )


def compute_parameters(
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
    single_dose: bool = True,
) -> dict[str, np.ndarray]:
    """Single dose parameters of every row of `(N, n)` time and value arrays.

    Args:
        t: times `(N, n)`, `NaN` for missing points
        c: values `(N, n)`, `NaN` for missing values
        dose_amount: dose per row `(N,)`, `None` without doses (`NaN` for a row
            without a dose in a batch which has them)
        dose_time: time of the dose per row, `None` for 0; a row without a dose
            carries `NaN` and its times are kept as they are, so that the
            dose-independent parameters of the row are still computed
        dose_duration: infusion duration per row (`NaN` without infusion), `None` for none
        route: route of the batch, `None` without doses
        options: the options
        lloq: limit of quantification per row `(N,)`, `None` for none;
            `NCAOptions.lloq` wins over it (`resolve_lloq`)
        windows: the terminal window `(t_first, t_last)` of single rows
            `(N, 2)` in the times of the analysis, `NaN` for a row without one
            (`TerminalPhase.windows`, `sample_windows`)
        single_dose: whether the rows are single dose curves. An infusion which
            starts at the dose is 0 there, so a zero is inserted at the dose
            time of a single dose row whose first sample comes later (the
            `insert_point` call of the `IV_INFUSION` branch below, which
            `_insert_dose_value` does for a partial area); the same row of a
            steady state interval starts at its trough and nothing is inserted
            (`pkpdutils.nca.steady_state.compute_steady_state` passes `False`)

    Returns:
        One `(N,)` array per parameter (see `PARAMETER_UNITS`) and `flags`.
    """
    t = np.asarray(t, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if dose_time is not None:
        # a row without a dose is not shifted: its times are already the times
        # of the curve and only the dose-dependent parameters stay `NaN`
        shift = np.where(np.isfinite(dose_time), dose_time, 0.0)
        t = t - shift[:, None]
    rules = options.blq_rules
    c, truncated, blq_in_curve = apply_blq(
        c, resolve_lloq(options, lloq, t.shape[0]), rules
    )
    if options.kind is Kind.EFFECT:
        return _effect_parameters(t, c, truncated, options)

    # a value below the limit which is still part of the curve is no quantified
    # value: it enters the areas, but it is neither the last measurable value
    # nor a point of the terminal regression
    blq_packed = (
        packed_mask(t, c, blq_in_curve)
        if blq_in_curve.any()
        else np.zeros(c.shape, dtype=bool)
    )
    exclude = None if rules.terminal_regression else blq_packed
    tp, cp, n_valid = pack_valid(t, c)
    if route is Route.IV_INFUSION and dose_duration is not None:
        # the terminal regression of an infusion starts after the infusion: the
        # concentration still rises while the drug is given, so no sample at or
        # before the end of the infusion is a candidate (Phoenix WinNonlin,
        # which starts the window at the first sample after `dose_duration`).
        # The times are relative to the dose, so the end of the infusion is its
        # duration; a row without one excludes nothing
        with np.errstate(invalid="ignore"):
            during = tp <= np.nan_to_num(dose_duration, nan=0.0)[:, None]
        exclude = during if exclude is None else (exclude | during)
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
    cmax = np.where(has_data, take_rows(cp, imax), nan)
    tmax = np.where(has_data, take_rows(tp, imax), nan)
    masked_min = np.where(in_row, cp, np.inf)
    imin = masked_min.argmin(axis=1)
    cmin = np.where(has_data, take_rows(cp, imin), nan)
    tmin = np.where(has_data, take_rows(tp, imin), nan)
    flags |= np.where(has_data & (imax == n_valid - 1), NCAFlag.NO_MAX, 0)
    if route is Route.ORAL:
        flags |= np.where(has_data & (imax == 0), NCAFlag.NO_ABSORPTION, 0)

    # last measurable point: a value below the limit which a BLQ rule kept or
    # imputed is positive but not measurable
    with np.errstate(invalid="ignore"):
        positive = in_row & (cp > 0) & ~blq_packed
    has_positive = positive.any(axis=1)
    ilast = np.where(has_positive, n - 1 - positive[:, ::-1].argmax(axis=1), 0)
    clast = np.where(has_data & has_positive, take_rows(cp, ilast), nan)
    tlast = np.where(has_data & has_positive, take_rows(tp, ilast), nan)

    # the lag time of an extravascular dose: the last sample at or after the
    # dose before the first measurable value
    tlag = _lag_time(tp, cp, in_row, route, blq_packed)

    # C0 of an intravenous bolus, inserted at t = 0 for the areas
    c0 = nan.copy()
    c0_method = np.full(n_rows, C0_NONE, dtype=np.int64)
    insert = np.zeros(n_rows, dtype=bool)
    tp_area, cp_area, n_area = tp, cp, n_valid
    if route is Route.IV_BOLUS:
        estimate, rule = bolus_c0(tp, cp, n_valid, options)
        c0 = np.where(has_data, estimate, nan)
        c0_method = np.where(has_data, rule, C0_NONE).astype(np.int64)
        with np.errstate(invalid="ignore"):
            insert = has_data & (tp[:, 0] > 0) & np.isfinite(c0)
        tp_area, cp_area, n_area = insert_point(
            tp, cp, n_valid, np.where(insert, 0.0, np.nan), np.where(insert, c0, np.nan)
        )
    elif route is Route.IV_INFUSION and single_dose:
        # an infusion starts at 0 at its dose, so a curve whose first sample
        # comes later starts at the dose with a zero, as Phoenix WinNonlin
        # inserts it; a steady state interval starts at its trough instead
        with np.errstate(invalid="ignore"):
            at_dose = has_data & (tp[:, 0] > 0)
        tp_area, cp_area, n_area = insert_point(
            tp,
            cp,
            n_valid,
            np.where(at_dose, 0.0, np.nan),
            np.where(at_dose, 0.0, np.nan),
        )

    auc_last, aumc_last = auc_aumc(
        tp_area, cp_area, n_area, options.auc_method, t_end=tlast
    )
    auc_last = np.where(has_data & has_positive, auc_last, nan)
    aumc_last = np.where(has_data & has_positive, aumc_last, nan)

    # AUCall = the area over every observation of the row, so the trailing
    # zeros and the values a BLQ rule imputed are part of it, where AUClast
    # ends at the last measurable value: "if the last concentration is
    # positive, AUClast = AUCall; otherwise it includes the additional area
    # from the last measurable concentration down to zero or negative
    # observations" (Phoenix WinNonlin NCA, `AUCall`)
    auc_all, aumc_all = auc_aumc(tp_area, cp_area, n_area, options.auc_method)
    auc_all = np.where(has_data, auc_all, nan)
    aumc_all = np.where(has_data, aumc_all, nan)

    # terminal phase
    manual_mask = None
    if options.terminal.method is TerminalMethod.MANUAL:
        assert options.terminal.points is not None
        original = np.zeros_like(c, dtype=bool)
        original[:, list(options.terminal.points)] = True
        # the packed position of the selected original points
        manual_mask = packed_mask(t, c, original)
    fit = terminal_fit(
        tp,
        cp,
        n_valid,
        imax,
        options.terminal,
        manual_mask=manual_mask,
        exclude=exclude,
        windows=windows,
    )
    flags |= np.where(has_data, fit.flags, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lambda_z = -fit.slope
        thalf = np.log(2.0) / lambda_z
        auc_inf_obs = auc_last + clast / lambda_z
        # Clast_pred = exp(Lambda_z_intercept - Lambda_z tlast), the terminal
        # regression at the last measurable time (Phoenix WinNonlin NCA,
        # `Clast_pred`), which `auc_inf_pred` extrapolates with
        clast_pred = np.exp(fit.intercept - lambda_z * tlast)
        auc_inf_pred = auc_last + clast_pred / lambda_z
        extrap = (auc_inf_obs - auc_last) / auc_inf_obs
        aumc_inf = aumc_last + clast * tlast / lambda_z + clast / (lambda_z * lambda_z)
        mrt = aumc_inf / auc_inf_obs
        if route is Route.IV_INFUSION and dose_duration is not None:
            mrt = mrt - np.where(np.isnan(dose_duration), 0.0, dose_duration) / 2.0
        # the effective half-life, PKNCA `pk.calc.thalf.eff`, verbatim
        # `log(2)*mrt` (see the module docstring)
        thalf_eff = np.log(2.0) * mrt
        flags |= np.where(
            extrap > options.extrapolation_warning, NCAFlag.EXTRAPOLATION_HIGH, 0
        )
        # the terminal phase should cover at least two half-lives
        span = (fit.t_last - fit.t_first) / thalf
        flags |= np.where(span < SPAN_MINIMUM, NCAFlag.SPAN_LOW, 0)

    # half maximum during absorption
    before_max = in_row & (idx < imax[:, None])
    with np.errstate(invalid="ignore"):
        distance = np.where(before_max, np.abs(cp - 0.5 * cmax[:, None]), np.inf)
    ihalf = distance.argmin(axis=1)
    has_half = has_data & (imax > 0) & (route is Route.ORAL)
    cmax_half = np.where(has_half, take_rows(cp, ihalf), nan)
    tmax_half = np.where(has_half, take_rows(tp, ihalf), nan)

    out: dict[str, np.ndarray] = {
        "cmax": cmax,
        "tmax": tmax,
        "cmin": cmin,
        "tmin": tmin,
        "clast": clast,
        "clast_pred": np.where(has_data, clast_pred, nan),
        "tlast": tlast,
        "auc_last": auc_last,
        "auc_all": auc_all,
        "auc_inf_obs": auc_inf_obs,
        "auc_inf_pred": auc_inf_pred,
        "auc_extrap_fraction": extrap,
        "aumc_last": aumc_last,
        "aumc_all": aumc_all,
        "aumc_inf": aumc_inf,
        "mrt": mrt,
        "thalf_eff": thalf_eff,
        "lambda_z": lambda_z,
        "lambda_z_stderr": fit.se_slope,
        "lambda_z_intercept": fit.intercept,
        "lambda_z_r2": fit.r2,
        "lambda_z_r2_adj": fit.r2_adj,
        "lambda_z_n_points": fit.n_points,
        "lambda_z_t_first": fit.t_first,
        "lambda_z_t_last": fit.t_last,
        "lambda_z_span": span,
        "thalf": thalf,
    }
    if route is Route.IV_BOLUS:
        out["c0"] = c0
        out["c0_method"] = c0_method
        # AUC_%Back_Ext = (area of the segment from the dose to the first
        # sample) / AUC(0-inf), the share of the exposure the estimate of C0
        # contributes rather than the data (Phoenix WinNonlin NCA,
        # `AUC_%Back_Ext_obs`); 0 for a row with a sample at the dose, which
        # has no such segment and needs no second pass over the areas
        zero = np.zeros(n_rows)
        area, moment = (
            segment_areas(tp_area, cp_area, n_area, options.auc_method)
            if insert.any()
            else (zero[:, None], zero[:, None])
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out["auc_back_extrap_fraction"] = np.where(
                has_data, np.where(insert, area[:, 0], 0.0) / auc_inf_obs, nan
            )
            out["aumc_back_extrap_fraction"] = np.where(
                has_data, np.where(insert, moment[:, 0], 0.0) / aumc_inf, nan
            )
    if route is Route.ORAL:
        out["tlag"] = tlag
        out["cmax_half"] = cmax_half
        out["tmax_half"] = tmax_half
    if dose_amount is not None and route is not None:
        amount = positive_dose(dose_amount)
        with np.errstate(divide="ignore", invalid="ignore"):
            cl = amount / auc_inf_obs
            vz = cl / lambda_z
            suffix = "" if route.is_iv else "_f"
            out[f"cl{suffix}"] = cl
            out[f"vz{suffix}"] = vz
            if route.is_iv:
                out["vss"] = cl * mrt
            out["auc_inf_dn"] = auc_inf_obs / amount
            out["cmax_dn"] = cmax / amount
    out.update(candidate_variables(fit.candidates, n_rows=n_rows))
    out["flags"] = flags
    return out


def candidate_variables(
    candidates: pd.DataFrame | None, *, n_rows: int
) -> dict[str, np.ndarray]:
    """The candidate windows of the terminal regression as variables of one row.

    The table of `pkpdutils.nca.terminal.candidate_table` becomes the
    `(1, K)` arrays `candidate_t_first`, `candidate_n_points` and
    `candidate_r2_adj` of a single curve, which `_to_result` writes over the
    dimension `candidate`. Only an analysis of one row carries them: the
    windows of a row are a table of their own and the rows of a batch need not
    have equally many of them, so a batch would need a padded extra dimension
    which every later step (the uncertainty, the summary, the tables) would
    have to carry along.

    Args:
        candidates: the table, `None` unless `TerminalPhase.keep_candidates`

    Keyword Args:
        n_rows: number of rows of the analysis

    Returns:
        The three arrays, or nothing for a batch of several rows and for a row
        without a single candidate window.
    """
    if candidates is None or n_rows != 1 or candidates.empty:
        return {}
    return {
        "candidate_t_first": candidates["start_time"].to_numpy(dtype=np.float64)[
            None, :
        ],
        "candidate_n_points": candidates["n"].to_numpy(dtype=np.float64)[None, :],
        "candidate_r2_adj": candidates["r2_adj"].to_numpy(dtype=np.float64)[None, :],
    }


def reserved_variables(values: dict[str, np.ndarray]) -> set[str]:
    """Every name the result of an analysis can carry, for the name of a partial area.

    A named partial area (`NCAOptions.partial_aucs`) becomes a variable of the
    result and may not take a name the analysis writes itself. At the point
    where the areas are computed the parameters are known, while `flags`, `n`,
    the status variables, the uncertainty variables of a group batch and the
    summary variables of `pkpdutils.result.ParameterResult.summarize` are
    written afterwards, so their names are derived here.

    Args:
        values: the parameters of the rows so far

    Returns:
        The names of the parameters, of `flags` and `n`, of the boolean and
        text variables and of every derived variable of a parameter
        (`pkpdutils.result.UNCERTAINTY_SUFFIXES` and `SUMMARY_SUFFIXES`).
    """
    names = set(values) | {"flags", "n"} | BOOLEAN_VARIABLES | TEXT_VARIABLES
    return names | {
        f"{name}{suffix}"
        for name in names
        for suffix in (*UNCERTAINTY_SUFFIXES, *SUMMARY_SUFFIXES)
    }


def evaluate_acceptance(
    values: dict[str, np.ndarray], acceptance: Acceptance, *, n_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    r"""Which rows meet every threshold of `Acceptance`, and the flag of the others.

    A threshold which is `None` is not checked; a row which does not carry the
    value of a threshold which is set (a row without a terminal phase has no
    adjusted \(R^2\) and no span) fails it. Without a single threshold every
    row is accepted, which is the default analysis.

    The extrapolated fraction is checked on the predicted variant,
    \((\mathrm{AUC}_{0\text{-}\infty,\mathrm{pred}} -
    \mathrm{AUC}_{0\text{-}t_\mathrm{last}}) /
    \mathrm{AUC}_{0\text{-}\infty,\mathrm{pred}}\), as PKanalix and Phoenix
    WinNonlin do, while the warning flag `NCAFlag.EXTRAPOLATION_HIGH` of
    `NCAOptions.extrapolation_warning` reads the observed variant
    `auc_extrap_fraction`.

    Args:
        values: the parameters of the rows, which carry `lambda_z_r2_adj`,
            `lambda_z_span`, `lambda_z_n_points`, `auc_last` and `auc_inf_pred`
        acceptance: the thresholds

    Keyword Args:
        n_rows: number of rows `N`

    Returns:
        The accepted rows `(N,)` and the flags of the rows which are not
        (`NCAFlag.NOT_ACCEPTED`).
    """
    accepted = np.ones(n_rows, dtype=bool)
    if not acceptance.any_threshold:
        return accepted, np.zeros(n_rows, dtype=np.int64)
    nan = np.full(n_rows, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc_last = values.get("auc_last", nan)
        auc_inf_pred = values.get("auc_inf_pred", nan)
        extrapolated = (auc_inf_pred - auc_last) / auc_inf_pred
        checks = (
            (acceptance.r2_adj_min, values.get("lambda_z_r2_adj", nan), True),
            (acceptance.extrapolation_max, extrapolated, False),
            (acceptance.span_min, values.get("lambda_z_span", nan), True),
            (acceptance.n_points_min, values.get("lambda_z_n_points", nan), True),
        )
        for threshold, value, at_least in checks:
            if threshold is None:
                continue
            met = value >= threshold if at_least else value <= threshold
            accepted &= np.isfinite(value) & met
    return accepted, np.where(accepted, 0, NCAFlag.NOT_ACCEPTED).astype(np.int64)


def reference_dose(
    dose_amount: np.ndarray | None,
    dose_time: np.ndarray | None,
    dose_duration: np.ndarray | None,
    *,
    last: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Pick one dose per row from the `(N, n_dose)` dose arrays of a batch.

    A row carries the dosing protocol of its sample, the doses at the front and
    the remaining columns `NaN` (`pkpdutils.timecourse.Timecourses`). The core
    of the analysis works with one reference dose per row: the first dose of
    the protocol for the single dose analysis and the last dose for the steady
    state analysis. A 1-D array is taken as one dose per row already.

    Args:
        dose_amount: the amounts `(N, n_dose)`, `None` without doses
        dose_time: the times `(N, n_dose)`, `None` without doses
        dose_duration: the infusion durations `(N, n_dose)`, `None` for none

    Keyword Args:
        last: whether to pick the last dose of every protocol instead of the
            first

    Returns:
        The amount, the time and the duration of the reference dose, each
        `(N,)` or `None` where the input is `None`. A row without a dose - a
        subject of an exchange format whose dose records are missing - gets
        `NaN`: `compute_parameters` then leaves its times unshifted and reports
        its dose-independent parameters, the dose-dependent ones being `NaN`.
    """
    arrays = [
        None if a is None else np.asarray(a, dtype=np.float64)
        for a in (dose_amount, dose_time, dose_duration)
    ]
    shaped = [None if a is None else a.reshape(a.shape[0], -1) for a in arrays]
    if all(a is None for a in shaped):
        return None, None, None
    valid: np.ndarray | None = None
    for a in shaped[:2]:
        if a is not None:
            finite = np.isfinite(a)
            valid = finite if valid is None else (valid & finite)
    assert valid is not None
    counts = valid.sum(axis=1)
    index = np.maximum(counts - 1, 0) if last else np.zeros_like(counts)
    picked = [
        None if a is None else np.where(counts > 0, take_rows(a, index), np.nan)
        for a in shaped
    ]
    return picked[0], picked[1], picked[2]


def is_multiple_dose(
    dose_amount: np.ndarray | None,
    dose_time: np.ndarray | None,
    options: NCAOptions,
    *,
    n_rows: int,
) -> np.ndarray:
    """Which rows of a batch are analysed as multiple dose rows.

    A row whose protocol holds more than one dose is analysed over its dosing
    intervals (`compute_steady_state`), every other row as a single dose curve
    (`compute_parameters`); `options.tau` (a steady state curve given with its
    last dose only) puts every row on the multiple dose path. The decision is
    taken per row, so a batch mixing the protocols reports the single dose
    parameters of its single dose rows and the steady state parameters of its
    multiple dose rows, each row `NaN` in the variables of the other path.

    Args:
        dose_amount: dose amounts `(N, n_dose)`, `None` without doses
        dose_time: dose times `(N, n_dose)`, `None` without doses
        options: the options, `tau` is used

    Keyword Args:
        n_rows: number of rows `N`

    Returns:
        The boolean mask of the multiple dose rows `(N,)`.
    """
    if options.tau is not None:
        return np.ones(n_rows, dtype=bool)
    if dose_amount is None or dose_time is None:
        return np.zeros(n_rows, dtype=bool)
    amounts = np.asarray(dose_amount, dtype=np.float64)
    times = np.asarray(dose_time, dtype=np.float64)
    given = np.isfinite(amounts.reshape(n_rows, -1)) & np.isfinite(
        times.reshape(n_rows, -1)
    )
    return given.sum(axis=1) >= 2


def dose_counts(dose_time: np.ndarray | None, n_rows: int) -> np.ndarray:
    """Number of doses of the protocol of every row.

    Args:
        dose_time: dose times `(N, n_dose)`, `NaN` padded, `None` without doses
        n_rows: number of rows `N`

    Returns:
        The count per row `(N,)`, 0 for a batch without doses.
    """
    if dose_time is None:
        return np.zeros(n_rows)
    times = np.asarray(dose_time, dtype=np.float64).reshape(n_rows, -1)
    return np.isfinite(times).sum(axis=1).astype(np.float64)


def chunk_bounds(n_rows: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split `n_rows` rows into `n_chunks` contiguous ranges of nearly equal size.

    The ranges are the ones `numpy.array_split` cuts (the first `n_rows %
    n_chunks` of them are one row longer) and they are contiguous, so a chunk
    of an array is a slice and therefore a view: a chunked analysis does not
    copy the batch before it starts.

    Args:
        n_rows: number of rows to split, 0 or more
        n_chunks: number of ranges, 1 or more

    Returns:
        The `(start, stop)` of every range, in row order; a range is empty if
        there are fewer rows than chunks.
    """
    base, extra = divmod(n_rows, n_chunks)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(n_chunks):
        stop = start + base + (1 if index < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def merge_rows(
    parts: list[dict[str, np.ndarray]], counts: list[int]
) -> dict[str, np.ndarray]:
    """Stack the parameters of row groups which need not carry the same variables.

    A group which does not report a variable of another group is `NaN` in it
    (0 in the integer variables `flags` and `c0_method`, whose 0 is "none" in
    both cases), so that the result of a batch is the
    union of the variables of its groups: a single dose row of a mixed batch
    carries `NaN` in the steady state variables and a multiple dose row `NaN`
    in `cl`, `vz`, `vss`, `auc_inf_dn` and `cmax_dn`; `n_doses`, which
    describes the protocol of a row and not the path it took, is filled in for
    every row of the batch by `run_rows`. The variables are ordered after the
    group which reports the most of them.

    Args:
        parts: one mapping of variable name to `(n_k,)` or `(n_k, K)` array per
            group, in the row order of the batch
        counts: number of rows `n_k` of every group

    Returns:
        One array per variable of the union, stacked over the rows.

    Raises:
        ValueError: if a variable has a different second dimension in two groups.
    """
    if len(parts) == 1:
        return parts[0]
    names: list[str] = []
    for index in sorted(range(len(parts)), key=lambda i: -len(parts[i])):
        names.extend(name for name in parts[index] if name not in names)
    widths: dict[str, int] = {}
    for part in parts:
        for name, array in part.items():
            width = array.shape[1] if array.ndim > 1 else 0
            if widths.setdefault(name, width) != width:
                raise ValueError(
                    f"'{name}' has {width} and {widths[name]} columns in two row groups"
                )
    out: dict[str, np.ndarray] = {}
    for name in names:
        blocks = []
        for part, n_rows in zip(parts, counts, strict=True):
            if name in part:
                blocks.append(part[name])
                continue
            shape = (n_rows, widths[name]) if widths[name] else (n_rows,)
            missing = (
                np.zeros(shape, dtype=np.int64)
                if name in INTEGER_VARIABLES
                else np.full(shape, np.nan)
            )
            blocks.append(missing)
        out[name] = np.concatenate(blocks)
    return out


def _compute_chunk(args: tuple[Any, ...]) -> dict[str, np.ndarray]:
    """Worker entry: the parameters of a chunk of rows.

    The chunk carries the dose arrays of the batch, `(N, n_dose)`, and the mask
    of its multiple dose rows (`is_multiple_dose`). A single dose row is
    reduced to the one dose of its protocol (`reference_dose`) and analysed by
    `compute_parameters`; a multiple dose row keeps the whole protocol and is
    analysed by `compute_steady_state`, which picks the last dose itself. A
    chunk holding both is split, both parts are analysed and the rows are put
    back in order by `merge_rows`. Both paths run through the same entry, so a
    multiple dose batch is chunked and parallelized like a single dose batch.

    Args:
        args: the times, the values, the dose arrays over the dose dimension,
            the route, the options, the mask of the multiple dose rows, the
            limit of quantification per row and the terminal window per row.

    Returns:
        The parameters of the rows of the chunk, in their order.
    """
    (
        t,
        c,
        dose_amount,
        dose_time,
        dose_duration,
        route,
        options,
        multiple,
        lloq,
        windows,
    ) = args
    if multiple.any() and not multiple.all():
        groups = [~multiple, multiple]
        order = np.concatenate([np.flatnonzero(mask) for mask in groups])
        parts = [
            _compute_chunk(
                (
                    t[mask],
                    c[mask],
                    None if dose_amount is None else dose_amount[mask],
                    None if dose_time is None else dose_time[mask],
                    None if dose_duration is None else dose_duration[mask],
                    route,
                    options,
                    multiple[mask],
                    None if lloq is None else lloq[mask],
                    None if windows is None else windows[mask],
                )
            )
            for mask in groups
        ]
        merged = merge_rows(parts, [int(mask.sum()) for mask in groups])
        # the two groups are concatenated in the order of `groups`, the rows go
        # back into the order of the chunk
        back = np.empty_like(order)
        back[order] = np.arange(order.size)
        return {name: array[back] for name, array in merged.items()}
    if multiple.all() and multiple.size:
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
            lloq=lloq,
            windows=windows,
        )
    amount, time, duration = reference_dose(
        dose_amount, dose_time, dose_duration, last=False
    )
    return compute_parameters(
        t,
        c,
        dose_amount=amount,
        dose_time=time,
        dose_duration=duration,
        route=route,
        options=options,
        lloq=lloq,
        windows=windows,
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
    lloq: np.ndarray | None = None,
    windows: np.ndarray | None = None,
    routes: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run the core on `(N, n)` arrays in chunks, serially or in the worker pool.

    The rows are cut into about one chunk per worker, none of them longer than
    `options.chunk_rows` rows, which bounds the memory of the vectorized core
    (`pkpdutils.parallel.split_rows`). A chunk is a contiguous range of rows,
    so it is a slice of the input arrays and not a copy of them.

    `options.n_workers` decides how many workers run them
    (`pkpdutils.parallel.resolve_workers`): `None` is automatic and stays in
    the calling thread below `pkpdutils.parallel.NCA_WORKER_THRESHOLD` rows,
    `1` is serial and any other number is taken as given. The chunks of a
    parallel run are mapped in order over the shared thread pool
    (`pkpdutils.parallel.executor`), since the core is vectorized numpy and
    releases the GIL for most of its time: the chunks are neither pickled nor
    copied and the pool starts in half a millisecond. The
    temporaries of the core then live for as many chunks as run at once, so a
    run holds up to `min(n_workers, len(chunks)) * options.chunk_rows` rows of
    them instead of `chunk_rows`.

    The dose arrays carry the dosing protocol of every row, `(N, n_dose)`
    padded with `NaN`. A row whose protocol holds more than one dose, and every
    row of an analysis with `options.tau`, is analysed over the dosing
    intervals (`is_multiple_dose`,
    `pkpdutils.nca.steady_state.compute_steady_state`); a single dose row is
    reduced to the one dose of its protocol (`reference_dose`). A 1-D array
    `(N,)` is one dose per row. A batch mixing the two carries the union of the
    variables, every row `NaN` in the variables of the other path
    (`merge_rows`).

    Args:
        t: times `(N, n)`
        c: values `(N, n)`
        dose_amount: dose amounts per row `(N, n_dose)`, `None` without doses
        dose_time: dose times per row `(N, n_dose)`, `None` for 0
        dose_duration: infusion durations per row `(N, n_dose)`, `None` for none
        route: route of the batch
        options: the options
        lloq: limit of quantification per row `(N,)`, `None` for none;
            `NCAOptions.lloq` wins over it (`resolve_lloq`)
        windows: the terminal window of single rows `(N, 2)`, `NaN` for a row
            without one (`TerminalPhase.windows`, `sample_windows`)
        routes: the route of every row `(N,)`, for a batch whose samples were
            given different ones (`Timecourses.routes`); `None` for the one
            route of `route`. The rows are grouped by route and every group is
            run on its own, so that the parameters which depend on the route
            (`c0`, `cl` against `cl_f`, `tlag`, the value at the dose time)
            follow the row; the result is the union of the variables of the
            groups (`merge_rows`), every row `NaN` in the variables of the
            other routes.

    Returns:
        One `(N,)` array per parameter and `flags`, and one `(N, K)` array per
        per-interval parameter of a multiple dose batch (`K` dosing intervals).
    """
    if routes is not None and len(routes) > 0:
        # one group per route, in the order of their first appearance, and the
        # rows back in their own order afterwards
        given = [Route(value) for value in routes]
        order_of_route = list(dict.fromkeys(given))
        masks = [
            np.array([value is one for value in given], dtype=bool)
            for one in order_of_route
        ]
        order = np.concatenate([np.flatnonzero(mask) for mask in masks])
        parts = [
            run_rows(
                t[mask],
                c[mask],
                dose_amount=None if dose_amount is None else dose_amount[mask],
                dose_time=None if dose_time is None else dose_time[mask],
                dose_duration=None if dose_duration is None else dose_duration[mask],
                route=one,
                options=options,
                lloq=None if lloq is None else lloq[mask],
                windows=None if windows is None else windows[mask],
            )
            for one, mask in zip(order_of_route, masks, strict=True)
        ]
        merged = merge_rows(parts, [int(mask.sum()) for mask in masks])
        back = np.empty_like(order)
        back[order] = np.arange(order.size)
        return {name: array[back] for name, array in merged.items()}
    n_rows = t.shape[0]
    n_workers = resolve_workers(
        options.n_workers, n_rows, threshold=NCA_WORKER_THRESHOLD
    )
    # about one chunk per worker, none longer than `chunk_rows`, which bounds
    # the memory of the vectorized core; an empty batch keeps its one empty
    # chunk, so that the result carries the variables of the analysis
    chunks = split_rows(n_rows, n_workers, max_rows=options.chunk_rows) or [slice(0, 0)]
    multiple = is_multiple_dose(dose_amount, dose_time, options, n_rows=n_rows)
    jobs = [
        (
            t[rows],
            c[rows],
            None if dose_amount is None else dose_amount[rows],
            None if dose_time is None else dose_time[rows],
            None if dose_duration is None else dose_duration[rows],
            route,
            options,
            multiple[rows],
            None if lloq is None else lloq[rows],
            None if windows is None else windows[rows],
        )
        for rows in chunks
    ]
    if n_workers > 1 and len(jobs) > 1:
        logger.debug(
            "NCA: %d rows in %d chunks over %d threads", n_rows, len(jobs), n_workers
        )
        parts = list(executor("thread", n_workers).map(_compute_chunk, jobs))
    else:
        parts = [_compute_chunk(job) for job in jobs]
    # a chunk of single dose rows reports fewer variables than one holding a
    # multiple dose row, so the chunks are merged into their union
    values = merge_rows(parts, [rows.stop - rows.start for rows in chunks])
    if "n_doses" in values:
        # the number of doses describes the protocol of a row and not the path
        # it took: it is filled in for every row of the batch, so that a single
        # dose row reports its own count wherever the chunks happen to cut
        values["n_doses"] = dose_counts(dose_time, n_rows)
    return values


def nca(timecourses: Timecourses, *, options: NCAOptions | None = None) -> NCAResult:
    """Non-compartmental analysis of a batch of timecourses.

    The rows are analysed in chunks of at most `options.chunk_rows` rows, in
    the calling thread or, for a large batch or an explicit
    `options.n_workers`, in the shared thread pool (`run_rows`); a multiple
    dose analysis is chunked the same way.

    A sample whose dosing protocol holds more than one dose (and every sample
    of an analysis with `options.tau`) is analysed over its dosing intervals:
    the point parameters are computed from the last dose on, the per-interval
    parameters (`interval_*` over the dimension `interval`) over every dosing
    interval and the steady state parameters from the last one, see
    `pkpdutils.nca.steady_state`. The decision is taken per sample, so a batch
    mixing single dose and multiple dose subjects reports `cl`/`cl_f` for the
    single dose samples and `cl_ss`/`cl_ss_f` for the multiple dose ones; every
    sample is `NaN` in the variables of the other path.

    A batch of group curves (`sd` or `se` per point) also carries the
    uncertainty of every parameter, by default from the parametric bootstrap
    (`options.uncertainty`, `pkpdutils.nca.uncertainty`): `x_sd`, `x_se`,
    `x_ci_low`, `x_ci_high`, under `BootstrapSpread.SD` draws also
    `x_pi_low`, `x_pi_high`, and, for log-normal parameters, `x_geomean`,
    `x_geocv`. The delta method can add `NCAFlag.DELTA_WINDOW_CHANGE` to the
    flags of a sample.

    A batch whose samples were given by different routes (the coordinate
    `route`, `Timecourses.routes`) is analysed per route: the rows are grouped
    and every group runs on its own, so that `c0`, `cl` against `cl_f`, `tlag`
    and the value at the dose time follow the row rather than the batch. The
    result carries the union of the variables, every sample `NaN` in the
    variables of the other routes.

    `NCAOptions.units` converts the named variables of the result to the
    reporting units at the end (`pkpdutils.result.ParameterResult.to_units`);
    the analysis itself runs in the units of the batch.

    Args:
        timecourses: the batch

    Keyword Args:
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

    n_dose = timecourses.n_dose

    def flat(a: np.ndarray | None) -> np.ndarray | None:
        """Flatten an optional dose array to `(n_rows, n_dose)`.

        Args:
            a: the array of shape `(*sample_shape, n_dose)`, or `None`.

        Returns:
            The flattened array, or `None`.
        """
        return (
            None
            if a is None
            else np.asarray(a, dtype=np.float64).reshape(n_rows, n_dose)
        )

    dose_amount = flat(timecourses.dose_amount)
    dose_time = flat(timecourses.dose_time)
    dose_duration = flat(timecourses.dose_duration)
    route, routes = row_routes(timecourses, n_rows)
    batch_lloq = timecourses.lloq
    lloq = None if batch_lloq is None else batch_lloq.reshape(n_rows)
    windows = sample_windows(timecourses, options.terminal)

    values = run_rows(
        t,
        c,
        dose_amount=dose_amount,
        dose_time=dose_time,
        dose_duration=dose_duration,
        route=route,
        options=options,
        lloq=lloq,
        windows=windows,
        routes=routes,
    )

    # `flags` is the last variable of the result, the uncertainty variables and
    # `n` go before it
    flags = values.pop("flags")

    units: dict[str, str] = {}
    if options.partial_aucs:
        collision = sorted(set(options.partial_aucs) & reserved_variables(values))
        if collision:
            raise ValueError(
                f"the partial areas {collision} carry the name of a variable of "
                "the result; name them differently"
            )
        first_time, reference_time = dose_times(
            dose_amount, dose_time, dose_duration, options, n_rows=n_rows
        )
        areas, extrapolated = named_partial_aucs(
            t - first_time[:, None],
            c,
            values,
            route=route,
            options=options,
            shift=reference_time - first_time,
            routes=routes,
        )
        values.update(areas)
        units.update(
            dict.fromkeys(options.partial_aucs, PARAMETER_UNITS["auc_partial"])
        )
        flags = flags | np.where(extrapolated, NCAFlag.PARTIAL_EXTRAPOLATED, 0).astype(
            flags.dtype
        )

    method = options.resolve_uncertainty(timecourses.has_uncertainty)
    if method is UncertaintyMethod.BOOTSTRAP:
        values.update(bootstrap(timecourses, options, values))
    elif method is UncertaintyMethod.DELTA:
        uncertainty = delta(timecourses, options, values)
        # the delta method reports the rows whose terminal window moved
        flags = flags | uncertainty.pop("flags").astype(flags.dtype)
        values.update(uncertainty)
    n_subjects = timecourses.n_subjects
    values["n"] = (
        np.full(n_rows, np.nan)
        if n_subjects is None
        else np.asarray(n_subjects, dtype=np.float64).reshape(n_rows)
    )

    accepted, not_accepted = evaluate_acceptance(
        values, options.acceptance, n_rows=n_rows
    )
    flags = flags | not_accepted
    values["accepted"] = accepted
    excluded = ~accepted if options.acceptance.exclude else np.zeros(n_rows, dtype=bool)
    values["excluded"] = excluded
    if excluded.any():
        # the reason is a text variable and only a result which excludes a
        # sample carries it; `NCAResult.exclude` adds it when it marks one
        values["excluded_reason"] = np.where(excluded, ACCEPTANCE_REASON, "")
    values["flags"] = flags

    n_flagged = int((flags != 0).sum())
    if n_flagged:
        logger.info(
            "NCA: %d of %d samples carry flags, see NCAResult.flag_table()",
            n_flagged,
            n_rows,
        )
    if excluded.any():
        logger.info(
            "NCA: %d of %d samples are excluded, see NCAResult.exclude()",
            int(excluded.sum()),
            n_rows,
        )
    result = _to_result(
        values,
        timecourses,
        shape,
        dose=reference_dose_amount(
            dose_amount, dose_time, dose_duration, options, n_rows=n_rows
        ),
        units=units,
    )
    if options.partial_aucs:
        # the intervals travel with the result, so that a figure can shade a
        # named area without being given the options again (`NCAResult.partial_aucs`)
        result.ds.attrs[PARTIAL_AUCS_ATTR] = {
            name: (float(start), float(end))
            for name, (start, end) in options.partial_aucs.items()
        }
    for name, (t_start, t_end) in options.partial_aucs.items():
        # the interval of a named partial area travels with its variable, so
        # that a writer of the result (`pkpdutils.cdisc`) can name it
        result.ds[name].attrs["window"] = [float(t_start), float(t_end)]
    # the analysis runs in the units of the batch; the reporting units of
    # `NCAOptions.units` are applied to the finished result
    return result.to_units(options.units) if options.units else result


def row_routes(
    timecourses: Timecourses, n_rows: int
) -> tuple[Route | None, np.ndarray | None]:
    """The route of a batch, or the route of every one of its rows.

    A batch which carries the coordinate `route` along a sample dimension was
    given by several routes (`Timecourses.routes`), and the analysis follows
    the route of every row rather than one route of the batch. A batch with one
    route keeps the fast path: the route is one value and the rows run in one
    group.

    Args:
        timecourses: the batch.
        n_rows: number of rows of the flattened batch.

    Returns:
        The one route of the batch and `None`, or `None` and the route of every
        row `(N,)`.
    """
    routes = timecourses.routes
    if routes is None:
        return timecourses.route, None
    return None, np.asarray(routes, dtype=object).reshape(n_rows)


def dose_times(
    dose_amount: np.ndarray | None,
    dose_time: np.ndarray | None,
    dose_duration: np.ndarray | None,
    options: NCAOptions,
    *,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The time of the first and of the reference dose of every row.

    The named partial areas are relative to the first dose of the protocol
    while the point parameters of a row are relative to its reference dose (the
    last dose of a multiple dose row, `reference_dose_amount`), so the two
    times are what translates between the two frames.

    Args:
        dose_amount: dose amounts `(N, n_dose)`, `None` without doses
        dose_time: dose times `(N, n_dose)`, `None` without doses
        dose_duration: infusion durations `(N, n_dose)`, `None` for none
        options: the options, `tau` is used by `is_multiple_dose`

    Keyword Args:
        n_rows: number of rows `N`

    Returns:
        The time of the first dose and the time of the reference dose per row,
        both 0 where the row carries no dose.
    """
    if dose_time is None:
        zero = np.zeros(n_rows)
        return zero, zero
    _, first, _ = reference_dose(dose_amount, dose_time, dose_duration, last=False)
    _, last, _ = reference_dose(dose_amount, dose_time, dose_duration, last=True)
    assert first is not None and last is not None
    multiple = is_multiple_dose(dose_amount, dose_time, options, n_rows=n_rows)
    return (
        np.nan_to_num(first, nan=0.0),
        np.nan_to_num(np.where(multiple, last, first), nan=0.0),
    )


def reference_dose_amount(
    dose_amount: np.ndarray | None,
    dose_time: np.ndarray | None,
    dose_duration: np.ndarray | None,
    options: NCAOptions,
    *,
    n_rows: int,
) -> np.ndarray | None:
    """The dose amount every row is analysed against.

    The first dose of the protocol for a single dose row and the last one for a
    multiple dose row (`is_multiple_dose`), the dose the parameters of the row
    are divided by (`cl`, `cl_ss`, the dose normalized variables of
    `pkpdutils.nca.result.NCAResult.dose_normalized`).

    Args:
        dose_amount: dose amounts `(N, n_dose)`, `None` without doses
        dose_time: dose times `(N, n_dose)`, `None` without doses
        dose_duration: infusion durations `(N, n_dose)`, `None` for none
        options: the options, `tau` is used by `is_multiple_dose`

    Keyword Args:
        n_rows: number of rows `N`

    Returns:
        The amount per row `(N,)`, `None` for a batch without doses.
    """
    if dose_amount is None:
        return None
    multiple = is_multiple_dose(dose_amount, dose_time, options, n_rows=n_rows)
    first, _, _ = reference_dose(dose_amount, dose_time, dose_duration, last=False)
    last, _, _ = reference_dose(dose_amount, dose_time, dose_duration, last=True)
    assert first is not None and last is not None
    return np.where(multiple, last, first)


def _to_result(
    values: dict[str, np.ndarray],
    timecourses: Timecourses,
    shape: tuple[int, ...],
    *,
    dose: np.ndarray | None = None,
    units: Mapping[str, str] | None = None,
) -> NCAResult:
    """Build the result dataset over the sample dimensions of the batch.

    A parameter is one `(N,)` array over the sample dimensions; a per-interval
    parameter is one `(N, K)` array and gets the extra dimension `interval`.

    Args:
        values: one `(N,)` array per parameter, one `(N, K)` array per
            per-interval parameter
        timecourses: the analysed batch
        shape: the sample shape the arrays are reshaped to

    Keyword Args:
        dose: the dose amount of every row `(N,)`, which travels into the
            result as the coordinate `dose_amount` so that a parameter can be
            normalized by it afterwards (`NCAResult.dose_normalized`); `None`
            for a batch without doses
        units: the unit expression of the variables whose name the analysis
            only knows at run time (the named partial areas of
            `NCAOptions.partial_aucs`); every other variable takes the
            expression of `unit_expression`

    Returns:
        The result.

    Raises:
        ValueError: if a non-dimension coordinate of the batch collides with
            a data variable of the result (`check_coordinate_collision`).
    """
    coords = sample_coordinates(timecourses.ds, timecourses.sample_dims)
    if dose is not None and DOSE_COORDINATE not in coords:
        coords[DOSE_COORDINATE] = xr.DataArray(
            np.asarray(dose, dtype=np.float64).reshape(shape),
            dims=timecourses.sample_dims,
            attrs={"units": timecourses.dose_unit},
        )
    data_vars: dict[str, Any] = {}
    n_intervals = 0
    n_candidates = 0
    overrides = dict(units or {})
    for name, array in values.items():
        unit, factor = parameter_unit(
            overrides.get(name) or unit_expression(name),
            unit=timecourses.unit,
            time_unit=timecourses.time_unit,
            dose_unit=timecourses.dose_unit,
        )
        if name in BOOLEAN_VARIABLES or name in TEXT_VARIABLES:
            data_vars[name] = (
                timecourses.sample_dims,
                array.reshape(shape),
                {"units": unit},
            )
        elif array.ndim > 1:
            # the per-interval parameters carry the extra dimension `interval`
            # and the candidate windows of the terminal regression the extra
            # dimension `candidate`
            is_candidate = name.startswith(CANDIDATE_PREFIX)
            width = array.shape[1]
            if is_candidate:
                n_candidates = width
            else:
                n_intervals = width
            data_vars[name] = (
                (
                    *timecourses.sample_dims,
                    CANDIDATE_DIM if is_candidate else INTERVAL_DIM,
                ),
                (array * factor).reshape((*shape, width)),
                {"units": unit},
            )
        elif name in INTEGER_VARIABLES:
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
    if n_intervals:
        coords[INTERVAL_DIM] = xr.DataArray(
            np.arange(1, n_intervals + 1), dims=INTERVAL_DIM
        )
    if n_candidates:
        coords[CANDIDATE_DIM] = xr.DataArray(
            np.arange(1, n_candidates + 1), dims=CANDIDATE_DIM
        )
    check_coordinate_collision(coords, data_vars)
    # a batch of several analytes or of several routes names them in the
    # coordinates `substance` and `route`, which travel into the result with
    # the other coordinates; a batch with one of each names it in `attrs`, so
    # that a writer of the result (`pkpdutils.cdisc`) finds what it describes
    attrs: dict[str, Any] = {}
    if timecourses.substances is None:
        attrs["substance"] = timecourses.substance
    if timecourses.routes is None and timecourses.route is not None:
        attrs["route"] = timecourses.route.value
    if timecourses.tissue is not None:
        attrs["tissue"] = timecourses.tissue
    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    return NCAResult(ds)


def nca_single(
    timecourse: Timecourse, *, options: NCAOptions | None = None
) -> NCAResult:
    """Non-compartmental analysis of one timecourse.

    Args:
        timecourse: the curve

    Keyword Args:
        options: the options, defaults for `None`

    Returns:
        The parameters, without sample dimensions.
    """
    batch = Timecourses.from_timecourses([timecourse], dim="_single")
    result = nca(batch, options=options)
    return NCAResult(result.ds.isel(_single=0).drop_vars("_single"))


def _insert_dose_value(
    tp: np.ndarray,
    cp: np.ndarray,
    n_valid: np.ndarray,
    *,
    route: Route | None,
    options: NCAOptions,
    routes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add the value at the dose to every curve whose first sample is after it.

    The value at time 0 is the back extrapolated `c0` for an intravenous bolus
    (`bolus_c0`) and 0 for an extravascular dose and for an infusion, both of
    which start at nothing when the dose is given: "for extravascular and
    infusion single dose a concentration of zero is inserted at the dose time"
    (Phoenix WinNonlin NCA). A batch without a route is left as it is, so that
    an area reaching before the first sample stays `NaN` there. The insertion
    describes a single dose curve; a dosing interval of a steady state analysis
    starts at its trough and is handled by `pkpdutils.nca.intervals`.

    Args:
        tp: packed times `(N, n)`, relative to the dose
        cp: packed values `(N, n)`
        n_valid: valid points per row `(N,)`

    Keyword Args:
        route: route of the batch
        options: the options, `c0_method` is used
        routes: the route of every row `(N,)` for a batch of several routes
            (`Timecourses.routes`), which wins over `route`

    Returns:
        The packed times, values and counts, one column wider when a value was
        added.
    """
    if route is None and routes is None:
        return tp, cp, n_valid
    with np.errstate(invalid="ignore"):
        insert = (n_valid >= 1) & (tp[:, 0] > 0)
    if routes is not None:
        bolus = np.array(
            [Route(value) is Route.IV_BOLUS for value in routes], dtype=bool
        )
        value = np.where(bolus, bolus_c0(tp, cp, n_valid, options)[0], 0.0)
    else:
        value = (
            bolus_c0(tp, cp, n_valid, options)[0]
            if route is Route.IV_BOLUS
            else np.zeros(tp.shape[0])
        )
    return insert_point(
        tp,
        cp,
        n_valid,
        np.where(insert & np.isfinite(value), 0.0, np.nan),
        np.where(insert, value, np.nan),
    )


def area_between(
    t: np.ndarray,
    c: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    route: Route | None,
    options: NCAOptions,
    routes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Area of every row between two times, with the bounds interpolated.

    The core of `partial_auc` and of the named partial areas of
    `NCAOptions.partial_aucs`: the values at the two bounds are interpolated
    with the trapezoid rule of `options.auc_method`
    (`pkpdutils.nca.auc.interpolate_at`), the value at the dose is added when
    the route allows it (`_insert_dose_value`) and the area is summed with the
    same rule.

    Args:
        t: times `(N, n)`, relative to the dose of the interval
        c: values `(N, n)`
        start: start of the interval per row `(N,)`
        end: end of the interval per row `(N,)`

    Keyword Args:
        route: route of the batch, which decides the value at the dose
        options: the options, `auc_method` and `c0_method` are used
        routes: the route of every row `(N,)` for a batch of several routes
            (`Timecourses.routes`), which wins over `route`

    Returns:
        The area per row and the rows whose observed range covers both bounds;
        the area of a row which is not covered is meaningless.
    """
    tp, cp, n_valid = pack_valid(t, c)
    tp, cp, n_valid = _insert_dose_value(
        tp, cp, n_valid, route=route, options=options, routes=routes
    )
    c_start = interpolate_at(tp, cp, n_valid, start, options.auc_method)
    c_end = interpolate_at(tp, cp, n_valid, end, options.auc_method)
    tp, cp, n_valid = insert_point(tp, cp, n_valid, start, c_start)
    tp, cp, n_valid = insert_point(tp, cp, n_valid, end, c_end)
    area, _ = auc_aumc(tp, cp, n_valid, options.auc_method, t_start=start, t_end=end)
    return area, np.isfinite(c_start) & np.isfinite(c_end)


def named_partial_aucs(
    t: np.ndarray,
    c: np.ndarray,
    values: dict[str, np.ndarray],
    *,
    route: Route | None,
    options: NCAOptions,
    shift: np.ndarray | None = None,
    routes: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    r"""The named partial areas of `NCAOptions.partial_aucs` of every row.

    The area between the two times of the interval, both relative to the first
    dose of the protocol. An interval which reaches beyond the last measurable
    value is completed with the terminal regression, as Phoenix WinNonlin does
    for a partial area past `Tlast`: the tail from \(t_\mathrm{last}\) to
    \(t_\mathrm{end}\) of

    $$\hat C(t) = \hat C_\mathrm{last}\, e^{-\lambda_z (t - t_\mathrm{last})}
    \quad\text{is}\quad
    \frac{\hat C_\mathrm{last}}{\lambda_z}
    \left(1 - e^{-\lambda_z (t_\mathrm{end} - t_\mathrm{last})}\right),$$

    and the row is reported in the returned mask
    (`NCAFlag.PARTIAL_EXTRAPOLATED`); without a terminal phase such a row is
    `NaN`. `AUC(0-72)`, the primary exposure of a drug with a long half-life in
    ICH M13A (2024), and the `pAUC` of the modified release guidances are
    intervals of this kind.

    Args:
        t: times `(N, n)`, relative to the first dose of the protocol
        c: values `(N, n)`
        values: the parameters of the rows so far, which carry `tlast`,
            `clast_pred` and `lambda_z`

    Keyword Args:
        route: route of the batch
        options: the options, `partial_aucs` and `auc_method` are used
        shift: the time of the reference dose of every row relative to the
            first dose `(N,)`, which puts `tlast` into the times of `t`;
            `None` for a single dose analysis, where they are the same
        routes: the route of every row `(N,)` for a batch of several routes
            (`Timecourses.routes`), which wins over `route`

    Returns:
        One `(N,)` array per named area and the rows whose area was completed
        with the terminal regression; a row which reaches beyond the last
        measurable value without a terminal phase is `NaN` and is not among
        them, since nothing was extrapolated.
    """
    n_rows = t.shape[0]
    nan = np.full(n_rows, np.nan)
    offset = np.zeros(n_rows) if shift is None else np.nan_to_num(shift, nan=0.0)
    tlast = values.get("tlast", nan) + offset
    clast_pred = values.get("clast_pred", nan)
    lambda_z = values.get("lambda_z", nan)
    out: dict[str, np.ndarray] = {}
    extrapolated = np.zeros(n_rows, dtype=bool)
    for name, (t_start, t_end) in options.partial_aucs.items():
        start = np.full(n_rows, float(t_start))
        end = np.full(n_rows, float(t_end))
        with np.errstate(invalid="ignore"):
            observed_end = np.minimum(end, tlast)
            has_observed = observed_end > start
            beyond = end > tlast
        # the rows without an observed part would ask for an empty interval;
        # they are computed with the full interval and discarded afterwards
        area, covered = area_between(
            t,
            c,
            start,
            np.where(has_observed, observed_end, end),
            route=route,
            options=options,
            routes=routes,
        )
        observed = np.where(has_observed, np.where(covered, area, np.nan), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            tail_start = np.maximum(start, tlast)
            tail = (
                clast_pred
                / lambda_z
                * (
                    np.exp(-lambda_z * (tail_start - tlast))
                    - np.exp(-lambda_z * (end - tlast))
                )
            )
        total = observed + np.where(beyond, tail, 0.0)
        out[name] = np.where(np.isfinite(tlast), total, np.nan)
        # a row without a terminal phase has no tail to add: its area is `NaN`
        # and nothing was extrapolated, so it is not flagged either
        extrapolated |= beyond & np.isfinite(tail)
    return out, extrapolated


def sample_windows(timecourses: Timecourses, phase: TerminalPhase) -> np.ndarray | None:
    """The terminal window of every row of a batch, `NaN` for a row without one.

    `TerminalPhase.windows` is keyed by the sample label: the label of a batch
    with one sample dimension, the tuple of labels of a batch with several, and
    the string `"*"` for every sample the mapping does not name.

    Args:
        timecourses: the batch
        phase: the terminal phase options, `windows` is read

    Returns:
        The windows `(N, 2)` in the order of the rows of the analysis, or
        `None` when no window is given.

    Raises:
        ValueError: if a key of `windows` is no label of the batch (and is not
            `"*"`).
    """
    if not phase.windows:
        return None
    labels = sample_keys(timecourses.ds, timecourses.sample_dims)
    known = set(labels)
    unknown = [
        key for key in phase.windows if key != WINDOW_DEFAULT_KEY and key not in known
    ]
    if unknown:
        raise ValueError(
            f"the terminal windows {sorted(map(str, unknown))} name no sample of "
            f"the batch; its samples are {sorted(map(str, known))}"
        )
    default = phase.windows.get(WINDOW_DEFAULT_KEY)
    out = np.full((len(labels), 2), np.nan)
    for row, label in enumerate(labels):
        window = phase.windows.get(label, default)
        if window is not None:
            out[row] = window
    return out


def sample_keys(ds: xr.Dataset, sample_dims: tuple[str, ...]) -> list[Any]:
    """The label of every sample of a dataset, in the row order of the analysis.

    The label of a batch with one sample dimension is the value of its
    coordinate (the integer position without one), the label of a batch with
    several is the tuple of the values, in the order of the dimensions. The
    values are python objects, so that they compare equal to the keys a user
    writes (`TerminalPhase.windows`, `NCAResult.terminal_windows`).

    Args:
        ds: the dataset of the batch or of a result
        sample_dims: the sample dimensions, in the order the samples are
            enumerated

    Returns:
        One label per sample, in C order of `sample_dims`; a batch without
        sample dimensions gives one label `()`.
    """
    per_dim = [
        (
            ds[dim].to_numpy().tolist()
            if dim in ds.coords
            else list(range(int(ds.sizes[dim])))
        )
        for dim in sample_dims
    ]
    if len(sample_dims) == 1:
        return list(per_dim[0])
    return [tuple(combination) for combination in itertools.product(*per_dim)]


def partial_auc(
    timecourses: Timecourses,
    t_start: float,
    t_end: float,
    *,
    options: NCAOptions | None = None,
) -> xr.DataArray:
    """Area under the curve of every sample between two times relative to the first dose.

    The values at the bounds are interpolated with the trapezoid rule of
    `options.auc_method` (`pkpdutils.nca.auc.interpolate_at`) and the area is
    summed with the same rule; a sample whose observed range does not cover
    `[t_start, t_end]` gives `NaN`.

    An interval which starts before the first sample of a curve but not before
    its dose - `AUC(0-12)` of a schedule whose first sample is at 0.5 h - is
    the common request, and the value at the dose comes from the route: 0 for
    an extravascular dose (nothing is absorbed yet, so the area up to the first
    sample is the triangle below it, the convention of Phoenix `AUC(0-t)`), the
    back-extrapolated `c0` for an intravenous bolus (`bolus_c0`, the estimate
    `compute_parameters` uses for the single dose areas) and `NaN` for an
    infusion, whose curve rises over the infusion in a way no extrapolation of
    the samples describes, and for a batch without a route.

    Only `options.auc_method` and `options.c0_method` are used: the area is
    read from the values as they are, so `lloq`, `blq` and `kind` do not apply
    and no uncertainty is propagated.

    Args:
        timecourses: the batch
        t_start: start of the interval, in the time unit of the batch, relative
            to the first dose of the protocol
        t_end: end of the interval, greater than `t_start`

    Keyword Args:
        options: the options, defaults for `None`

    Returns:
        The areas over the sample dimensions, named `auc_partial`, with the unit of `auc_last`.

    Raises:
        ValueError: if `t_end <= t_start`, or if a non-dimension coordinate
            of the batch collides with `auc_partial`
            (`check_coordinate_collision`).
    """
    if t_end <= t_start:
        raise ValueError(
            f"'t_end' ({t_end}) must be greater than 't_start' ({t_start})"
        )
    options = options or NCAOptions()
    n_rows = timecourses.n_samples
    t = timecourses.times.reshape(n_rows, timecourses.n_time)
    c = timecourses.values.reshape(n_rows, timecourses.n_time)
    first_dose_time = timecourses.first_dose_time
    if first_dose_time is not None:
        # the area is relative to the first dose of the protocol
        t = t - np.asarray(first_dose_time, dtype=np.float64).reshape(n_rows)[:, None]
    route, routes = row_routes(timecourses, n_rows)
    area, covered = area_between(
        t,
        c,
        np.full(n_rows, float(t_start)),
        np.full(n_rows, float(t_end)),
        route=route,
        options=options,
        routes=routes,
    )
    area = np.where(covered, area, np.nan)
    unit, factor = parameter_unit(
        PARAMETER_UNITS["auc_partial"],
        unit=timecourses.unit,
        time_unit=timecourses.time_unit,
        dose_unit=timecourses.dose_unit,
    )
    coords = sample_coordinates(timecourses.ds, timecourses.sample_dims)
    check_coordinate_collision(coords, {"auc_partial"})
    return xr.DataArray(
        (area * factor).reshape(timecourses.sample_shape),
        dims=timecourses.sample_dims,
        coords=coords,
        name="auc_partial",
        attrs={"units": unit},
    )
