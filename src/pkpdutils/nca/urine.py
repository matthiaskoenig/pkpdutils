r"""Non-compartmental analysis of urinary excretion data.

A urine study does not sample a concentration over time, it collects the
urine of a subject over intervals \([s_k, e_k]\) and measures the volume and
the concentration of the substance in every collection. The analysis runs on
the *excretion rate curve*: the rate

$$\dot A_k = \frac{A_k}{e_k - s_k}$$

of every collection, plotted against the midpoint \(\bar t_k = (s_k + e_k)/2\)
of its interval. The rate curve is analysed like a concentration curve
(Phoenix WinNonlin urine models 210 to 212, which mirror the plasma models 200
to 202), so the areas, the peak and the terminal regression come from the same
vectorized core as every other analysis of the package
(`pkpdutils.nca.nca.compute_parameters`) and only the names differ:
`aurc_last` is the `auc_last` of the rate curve, `max_rate` its `cmax` and
`mid_pt_last` its `tlast`.

The area under the rate curve is an amount, since a rate times a time is an
amount, and \(\mathrm{AURC}_{0\text{-}\infty}\) is the amount the subject would
excrete in total. What the subject did excrete over the collections is
`amount_recovered`, the plain sum \(A_e = \sum_k A_k\), and
`percent_recovered` is that amount as a percentage of the dose. With the
plasma curve of the same subject the renal clearance

$$\mathrm{CL}_R = \frac{A_e}{\mathrm{AUC}}$$

follows over the same window, the one parameter Phoenix, PKanalix and Pumas
all leave to the user.
"""

import logging
from typing import Any

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, model_validator

from pkpdutils.nca.nca import PARAMETER_UNITS, compute_parameters
from pkpdutils.nca.options import NCAOptions
from pkpdutils.nca.result import NCAResult, parameter_unit
from pkpdutils.timecourse import Dose, Route, Timecourse, Timecourses
from pkpdutils.units import Q_, parse_unit

logger = logging.getLogger(__name__)

#: dimension of the collection intervals of the excretion rate curve
COLLECTION_DIM: str = "collection"

#: the parameters of the rate curve, under the name they carry in the result,
#: and the name the core computes them under (Phoenix WinNonlin urine models
#: 210 to 212 use the names on the left)
RATE_PARAMETERS: dict[str, str] = {
    "max_rate": "cmax",
    "tmax_rate": "tmax",
    "rate_last": "clast",
    "mid_pt_last": "tlast",
    "aurc_last": "auc_last",
    "aurc_all": "auc_all",
    "aurc_inf_obs": "auc_inf_obs",
    "aurc_inf_pred": "auc_inf_pred",
    "lambda_z": "lambda_z",
    "lambda_z_stderr": "lambda_z_stderr",
    "lambda_z_intercept": "lambda_z_intercept",
    "lambda_z_r2": "lambda_z_r2",
    "lambda_z_r2_adj": "lambda_z_r2_adj",
    "lambda_z_n_points": "lambda_z_n_points",
    "lambda_z_t_first": "lambda_z_t_first",
    "lambda_z_t_last": "lambda_z_t_last",
    "lambda_z_span": "lambda_z_span",
    "thalf": "thalf",
}


class Excretion(BaseModel):
    r"""Amounts of a substance collected over urine collection intervals.

    One subject and one substance: the collection intervals \([s_k, e_k]\) with
    the volume of every collection and either the concentration measured in it
    or the amount it contains. The intervals are sorted by their start on
    construction, have to be strictly increasing (\(s_k < e_k\)) and may not
    overlap (\(e_k \le s_{k+1}\)); a gap between two collections is allowed,
    since a subject does not void continuously.

    The amount and the concentration are two views of the same measurement and
    the volume connects them, \(A_k = c_k V_k\): give either one together with
    the volume and the other is derived, or give the amount alone when the
    volumes were not recorded (`vol_ur` is then `NaN`). The product is taken as
    it is given, so the concentration has to be in `unit / volume_unit`: with
    `unit="mg"` and `volume_unit="ml"` it is in mg/ml, not in mg/l.

    Attributes:
        start: start of every collection interval, in `time_unit`
        end: end of every collection interval, in `time_unit`
        unit: unit of the amount, e.g. `"mg"`
        time_unit: unit of `start` and `end`, e.g. `"hr"`
        volume: volume of every collection, in `volume_unit`; `None` when the
            volumes were not recorded
        concentration: concentration in every collection, in
            `unit / volume_unit`; derived from `amount` and `volume` when it is
            not given
        amount: amount in every collection, in `unit`; derived from
            `concentration` and `volume` when it is not given
        volume_unit: unit of `volume`, e.g. `"ml"`; required with a volume
        dose: the dose the collections follow, whose amount `percent_recovered`
            refers to and whose route decides the value of the rate curve at
            the dose time
        substance: name of the substance
        label: label of the subject or of the profile
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    start: np.ndarray
    end: np.ndarray
    unit: str
    time_unit: str
    volume: np.ndarray | None = None
    concentration: np.ndarray | None = None
    amount: np.ndarray | None = None
    volume_unit: str | None = None
    dose: Dose | None = None
    substance: str = "substance"
    label: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _prepare(cls, data: Any) -> Any:
        """Convert the arrays, sort by the start of the interval and derive the third of amount, concentration and volume."""
        if not isinstance(data, dict):
            return data
        values = dict(data)
        for name in ("start", "end", "volume", "concentration", "amount"):
            raw = values.get(name)
            if raw is not None:
                values[name] = np.asarray(raw, dtype=np.float64).reshape(-1)
        start = values.get("start")
        if start is None:
            return values
        order = np.argsort(start, kind="stable")
        for name in ("start", "end", "volume", "concentration", "amount"):
            array = values.get(name)
            if array is not None and array.shape == start.shape:
                values[name] = array[order]
        volume = values.get("volume")
        if volume is not None:
            if values.get("amount") is None and values.get("concentration") is not None:
                values["amount"] = values["concentration"] * volume
            elif (
                values.get("concentration") is None and values.get("amount") is not None
            ):
                with np.errstate(divide="ignore", invalid="ignore"):
                    values["concentration"] = values["amount"] / volume
        return values

    @model_validator(mode="after")
    def _validate(self) -> "Excretion":
        """Check the units, the lengths and the ordering of the intervals.

        Returns:
            The validated model.

        Raises:
            ValueError: if the units are unknown, if the arrays do not have the
                same length, if an interval is empty or two intervals overlap,
                if no amount could be derived or if a volume carries no unit.
        """
        parse_unit(self.unit)
        parse_unit(self.time_unit)
        n = self.start.size
        if self.end.size != n:
            raise ValueError(
                f"'start' has {n} entries, 'end' has {self.end.size}; every "
                "collection interval needs a start and an end"
            )
        if n == 0:
            raise ValueError("an excretion needs at least one collection interval")
        for name in ("volume", "concentration", "amount"):
            array = getattr(self, name)
            if array is not None and array.size != n:
                raise ValueError(
                    f"'{name}' has {array.size} entries, the {n} collection "
                    "intervals need one each"
                )
        if not np.all(self.end > self.start):
            raise ValueError("every collection interval needs 'end' above 'start'")
        if n > 1 and not np.all(self.end[:-1] <= self.start[1:]):
            raise ValueError("the collection intervals may not overlap")
        if self.amount is None:
            raise ValueError(
                "give 'amount', or 'concentration' together with 'volume' so "
                "that the amount of every collection is known"
            )
        if self.volume is not None:
            if self.volume_unit is None:
                raise ValueError("'volume' needs a 'volume_unit'")
            parse_unit(self.volume_unit)
        return self

    @property
    def n_intervals(self) -> int:
        """Number of collection intervals."""
        return int(self.start.size)

    @property
    def duration(self) -> np.ndarray:
        r"""Length \(e_k - s_k\) of every collection interval."""
        return self.end - self.start

    @property
    def midpoint(self) -> np.ndarray:
        r"""Midpoint \(\bar t_k = (s_k + e_k)/2\) of every collection interval.

        The time the excretion rate of an interval is reported at, the
        convention of Phoenix WinNonlin, PKanalix, Pumas and PKNCA.
        """
        return 0.5 * (self.start + self.end)

    @property
    def rate(self) -> np.ndarray:
        r"""Excretion rate \(\dot A_k = A_k / (e_k - s_k)\) of every collection interval."""
        assert self.amount is not None
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.amount / self.duration

    @property
    def cumulative(self) -> np.ndarray:
        r"""Amount recovered up to the end of every collection interval, \(\sum_{j \le k} A_j\)."""
        assert self.amount is not None
        return np.nancumsum(self.amount)

    @property
    def total_amount(self) -> float:
        r"""Amount recovered over every collection interval, \(A_e = \sum_k A_k\)."""
        assert self.amount is not None
        return float(np.nansum(self.amount))

    @property
    def total_volume(self) -> float:
        """Volume collected over every interval, `NaN` without volumes."""
        if self.volume is None:
            return float("nan")
        return float(np.nansum(self.volume))

    @property
    def route(self) -> Route | None:
        """Route of the dose, `None` without a dose."""
        return None if self.dose is None else self.dose.route

    @property
    def rate_unit(self) -> str:
        """Unit of the excretion rate, the amount unit per time unit."""
        return f"({self.unit}) / ({self.time_unit})"

    @property
    def concentration_unit(self) -> str:
        """Unit of the concentration in the collections, `None` without volumes."""
        if self.volume_unit is None:
            return ""
        return f"({self.unit}) / ({self.volume_unit})"


def _plasma_area(
    plasma: "NCAResult | Timecourse | Timecourses",
    excretion: Excretion,
    options: NCAOptions,
) -> tuple[float, str]:
    r"""The plasma area the renal clearance divides by, with its unit.

    A curve is integrated over the collection span \\([s_1, e_K]\\) with
    `pkpdutils.nca.nca.partial_auc`, which is the window the recovered amount
    belongs to; a result contributes its `auc_last`, which is the same window
    only when the curve ends with the last collection.

    Args:
        plasma: the plasma curve of the same subject, or the result of its
            analysis
        excretion: the collections, whose span the curve is integrated over
        options: the options of the area (`auc_method`, `c0_method`)

    Returns:
        The area and its unit string.

    Raises:
        ValueError: if a result carries no `auc_last`, or if a batch holds more
            than one sample.
    """
    from pkpdutils.nca.nca import partial_auc

    if isinstance(plasma, NCAResult):
        if "auc_last" not in plasma.ds.data_vars:
            raise ValueError("the plasma result carries no 'auc_last'")
        values = np.asarray(plasma["auc_last"].to_numpy(), dtype=np.float64).reshape(-1)
        if values.size != 1:
            raise ValueError(
                f"the plasma result holds {values.size} samples, the excretion "
                "of one subject needs the plasma curve of that subject"
            )
        return float(values[0]), plasma.units("auc_last")
    batch = plasma.to_batch() if isinstance(plasma, Timecourse) else plasma
    if batch.n_samples != 1:
        raise ValueError(
            f"the plasma batch holds {batch.n_samples} samples, the excretion "
            "of one subject needs the plasma curve of that subject"
        )
    area = partial_auc(
        batch,
        float(excretion.start[0]),
        float(excretion.end[-1]),
        options=options,
    )
    return float(np.asarray(area.to_numpy()).reshape(-1)[0]), str(area.attrs["units"])


def nca_urine(
    excretion: Excretion,
    *,
    options: NCAOptions | None = None,
    plasma: "NCAResult | Timecourse | Timecourses | None" = None,
) -> NCAResult:
    r"""Non-compartmental analysis of the urinary excretion of one subject.

    The excretion rate \(\dot A_k = A_k / (e_k - s_k)\) of every collection is
    analysed against the midpoint \(\bar t_k\) of its interval, with the same
    trapezoid rules and the same terminal regression as a concentration curve
    (Phoenix WinNonlin urine models 210 to 212): the area under the rate curve
    is an amount, so

    $$\mathrm{AURC}_{0\text{-}t_\mathrm{last}} = \sum_k \int_{\bar t_k}^{\bar
    t_{k+1}} \dot A(t)\, \mathrm dt, \qquad \mathrm{AURC}_{0\text{-}\infty} =
    \mathrm{AURC}_{0\text{-}t_\mathrm{last}} + \frac{\dot
    A_\mathrm{last}}{\lambda_z},$$

    and \(\lambda_z\) is the negative slope of \(\ln \dot A_k\) against
    \(\bar t_k\), which is the elimination rate constant of the substance when
    the renal elimination follows the plasma. The area starts at the dose
    (\(t = 0\)) the way it does for a concentration curve: at 0 for an
    extravascular dose, at the back-extrapolated rate of an intravenous bolus
    (`NCAOptions.c0_method`) and at the first midpoint without a dose.

    What was collected is reported as it was measured:
    \(A_e = \sum_k A_k\) (`amount_recovered`), \(100\,A_e/D\)
    (`percent_recovered`) and \(\sum_k V_k\) (`vol_ur`). With the plasma curve
    of the same subject the renal clearance

    $$\mathrm{CL}_R = \frac{A_e}{\mathrm{AUC}_{s_1\text{-}e_K}}$$

    is computed over the collection span, the window the amount belongs to
    (EMA CPMP/EWP/QWP/1401/98 Rev. 1 asks for \(A_e\) and, where it applies,
    the maximum rate; the CDISC codelist names the parameters `AURC*`,
    `RCAMINT`, `RCPCINT`, `VOLPK` and `RENALCL`).

    Only `auc_method`, `c0_method`, `terminal` and `extrapolation_warning` of
    the options are read: the rules which read a concentration
    (`lloq`, `blq`, `kind`, `partial_aucs`, `acceptance`, the uncertainty) do
    not apply to a rate curve and are ignored.

    Args:
        excretion: the collections of one subject

    Keyword Args:
        options: the options, defaults for `None`
        plasma: the plasma curve of the same subject (a `Timecourse` or a
            batch of one sample), whose area over the collection span
            \([s_1, e_K]\) the renal clearance divides by, or the `NCAResult`
            of that curve, whose `auc_last` is taken instead; without it the
            result carries no `clr`

    Returns:
        The parameters of the excretion without sample dimensions, with the
        rate curve as the point variables `rate` and `midpoint` over the
        dimension `collection`.
    """
    options = options or NCAOptions()
    # the rate curve is read with the rules of a curve, not with the rules
    # which read a concentration (a rate has no limit of quantification)
    curve_options = NCAOptions(
        auc_method=options.auc_method,
        c0_method=options.c0_method,
        terminal=options.terminal,
        extrapolation_warning=options.extrapolation_warning,
    )
    midpoint = excretion.midpoint
    rate = excretion.rate
    values = compute_parameters(
        midpoint[None, :],
        rate[None, :],
        dose_amount=None,
        dose_time=None,
        dose_duration=None,
        route=excretion.route,
        options=curve_options,
    )

    data_vars: dict[str, Any] = {}
    # a study without recorded volumes reports `vol_ur` as `NaN` liter, the
    # canonical volume of the package, rather than without a unit
    volume_unit = excretion.volume_unit or "liter"

    def resolve(expression: str) -> tuple[str, float]:
        """The unit of an expression of the excretion and the factor to it."""
        return parameter_unit(
            expression,
            unit="dimensionless",
            time_unit=excretion.time_unit,
            dose_unit=None if excretion.dose is None else excretion.dose.unit,
            amount_unit=excretion.unit,
            volume_unit=volume_unit,
        )

    def write(name: str, value: float, expression: str) -> None:
        """Write one scalar variable of the result, converted to its canonical unit."""
        unit, factor = resolve(expression)
        data_vars[name] = xr.DataArray(float(value) * factor, attrs={"units": unit})

    for name, source in RATE_PARAMETERS.items():
        write(name, float(values[source][0]), PARAMETER_UNITS[name])
    write(
        "amount_recovered", excretion.total_amount, PARAMETER_UNITS["amount_recovered"]
    )
    write("vol_ur", excretion.total_volume, PARAMETER_UNITS["vol_ur"])

    percent = np.nan
    if excretion.dose is not None:
        try:
            fraction = (
                Q_(excretion.total_amount, excretion.unit)
                / Q_(excretion.dose.amount, excretion.dose.unit)
            ).to("dimensionless")
        except Exception as error:  # pint raises several exception types
            raise ValueError(
                f"the recovered amount in '{excretion.unit}' cannot be compared "
                f"with the dose in '{excretion.dose.unit}'"
            ) from error
        percent = 100.0 * float(fraction.magnitude)
    write("percent_recovered", percent, PARAMETER_UNITS["percent_recovered"])

    if plasma is not None:
        area, area_unit = _plasma_area(plasma, excretion, options)
        # `({amount}) / (({unit}) * ({time}))` of `PARAMETER_UNITS["clr"]`,
        # spelled with the concrete units of the two quantities, since the
        # concentration unit of the plasma curve is not a unit of the excretion
        clr_unit, factor = parameter_unit(
            f"({excretion.unit}) / ({area_unit})",
            unit="dimensionless",
            time_unit=excretion.time_unit,
            dose_unit=None,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            clr = excretion.total_amount / area
        data_vars["clr"] = xr.DataArray(clr * factor, attrs={"units": clr_unit})

    for name, array in (("midpoint", midpoint), ("rate", rate)):
        unit, factor = resolve(PARAMETER_UNITS[name])
        data_vars[name] = xr.DataArray(
            array * factor, dims=COLLECTION_DIM, attrs={"units": unit}
        )
    data_vars["flags"] = xr.DataArray(
        np.int64(values["flags"][0]), attrs={"units": "dimensionless"}
    )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={COLLECTION_DIM: np.arange(1, excretion.n_intervals + 1)},
        attrs={"substance": excretion.substance},
    )
    logger.debug(
        "urine analysis of %s: %d collections, %s recovered",
        excretion.label or excretion.substance,
        excretion.n_intervals,
        Q_(excretion.total_amount, excretion.unit),
    )
    return NCAResult(ds)
