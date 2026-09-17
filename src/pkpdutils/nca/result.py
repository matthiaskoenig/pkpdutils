"""Units of the parameters and the result container of the NCA.

An `NCAResult` wraps an `xarray.Dataset` with one variable per parameter over
the sample dimensions of the analysed batch, `attrs["units"]` on every
variable and the integer variable `flags` (`pkpdutils.nca.options.NCAFlag`).
The units are derived from the units of the input with pint: an area carries
`unit * time_unit`, a rate `1 / time_unit`, a clearance `dose_unit / (unit *
time_unit)` converted to `liter / hour` (or per kilogram), a volume converted
to `liter` (or per kilogram), see `pkpdutils.units`.
"""

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from pkpdutils.nca.intervals import INTERVAL_DIM, INTERVAL_PREFIX, INTERVAL_UNITS
from pkpdutils.nca.options import NCAFlag
from pkpdutils.nca.uncertainty import DISCRETE_PARAMETERS, LOGNORMAL_PARAMETERS
from pkpdutils.result import EXCLUDED_VARIABLE, ParameterResult
from pkpdutils.units import (
    CACHE_SIZE,
    Q_,
    normalize_clearance,
    normalize_volume,
    ureg,
)

#: name of the coordinate carrying the dose amount of every sample of a result
DOSE_COORDINATE = "dose_amount"

#: name of the text variable carrying why a sample was excluded
REASON_VARIABLE = "excluded_reason"

#: attribute of the result dataset carrying the intervals of the named partial
#: areas of the analysis (`NCAOptions.partial_aucs`, `NCAResult.partial_aucs`)
PARTIAL_AUCS_ATTR = "partial_aucs"

#: suffix of a dose normalized variable (`NCAResult.dose_normalized`)
DOSE_NORMALIZED_SUFFIX = "_dn"

#: unit expressions of the parameters `NCAResult.dose_normalized` normalizes by
#: default: the concentrations and the exposures
DOSE_NORMALIZED_EXPRESSIONS: tuple[str, ...] = ("{unit}", "({unit}) * ({time})")

#: parameters whose dose normalized variable carries a name of its own, from
#: before the general rule existed
DOSE_NORMALIZED_NAMES: dict[str, str] = {"auc_inf_obs": "auc_inf_dn"}


@lru_cache(maxsize=CACHE_SIZE)
def parameter_unit(
    expression: str,
    *,
    unit: str,
    time_unit: str,
    dose_unit: str | None,
    amount_unit: str | None = None,
    volume_unit: str | None = None,
) -> tuple[str, float]:
    """Unit of a parameter and the factor from its raw unit to the reported unit.

    The unit of a parameter depends only on the strings of the signature, of
    which an analysis has a handful (`PARAMETER_UNITS` holds one expression per
    parameter), while the derivation costs several pint conversions; the result
    is therefore cached (`pkpdutils.units.CACHE_SIZE` entries).

    Args:
        expression: pint expression with the placeholders `{unit}`, `{time}`,
            `{dose}`, `{amount}` and `{volume}`, e.g. `"({unit}) * ({time})"`
        unit: unit of the values
        time_unit: unit of the times
        dose_unit: unit of the doses, `None` without doses (an expression
            with `{dose}` then raises `ValueError`)
        amount_unit: unit of an excreted amount, `None` outside the urinary
            excretion analysis (`pkpdutils.nca.urine`)
        volume_unit: unit of a collected volume, `None` outside the urinary
            excretion analysis

    Returns:
        The canonical unit string and the factor a magnitude in the raw unit is
        multiplied with to be in the canonical unit (volumes to liter,
        clearances to liter per hour, else 1).

    Raises:
        ValueError: if `expression` uses a placeholder whose unit is `None`.
    """
    for placeholder, value in (
        ("{dose}", dose_unit),
        ("{amount}", amount_unit),
        ("{volume}", volume_unit),
    ):
        if placeholder in expression and value is None:
            raise ValueError(f"'{expression}' needs a {placeholder[1:-1]} unit")
    raw = expression.format(
        unit=unit,
        time=time_unit,
        dose=dose_unit or "",
        amount=amount_unit or "",
        volume=volume_unit or "",
    )
    quantity = Q_(1.0, ureg.parse_units(raw))
    converted = normalize_clearance(normalize_volume(quantity))
    return str(converted.units), float(converted.magnitude)


@lru_cache(maxsize=CACHE_SIZE)
def _per_dose(unit: str, dose_unit: str) -> tuple[str, float]:
    """Unit of a parameter per dose and the factor to it.

    Args:
        unit: unit of the parameter, as the result reports it
        dose_unit: unit of the doses

    Returns:
        The canonical unit string of the parameter per dose and the factor a
        magnitude in the raw unit is multiplied with, as `parameter_unit`.
    """
    quantity = Q_(1.0, ureg.parse_units(f"({unit}) / ({dose_unit})"))
    converted = normalize_clearance(normalize_volume(quantity))
    return str(converted.units), float(converted.magnitude)


class NCAResult(ParameterResult):
    """Parameters of a non-compartmental analysis as an `xarray.Dataset`.

    One variable per parameter over the sample dimensions of the analysed
    `Timecourses`, `attrs["units"]` on every variable, the uncertainty
    variables of `pkpdutils.nca.uncertainty` and the integer variable `flags`
    (`NCAFlag`). See `pkpdutils.result.ParameterResult` for the interface.
    """

    flag_type = NCAFlag
    lognormal_parameters = LOGNORMAL_PARAMETERS
    discrete_parameters = DISCRETE_PARAMETERS
    #: `auc_inf_obs` is reported dose normalized as `auc_inf_dn`, every other
    #: parameter as `x_dn` (`to_units` converts the companion with it)
    dose_normalized_names = DOSE_NORMALIZED_NAMES
    #: the acceptance and the exclusion of a sample, no parameters of it
    status_variables = frozenset({"accepted", EXCLUDED_VARIABLE, REASON_VARIABLE})
    #: the per-interval parameters are point variables (the dimension
    #: `interval`) which `summarize` reduces over the sample dimension, so
    #: that a multiple dose study reports the mean trough per interval over
    #: its subjects
    summarized_point_variables = frozenset(INTERVAL_UNITS)
    #: the parameters the console table shows with one row per sample: the
    #: exposure, the peak, the terminal phase, clearance and volume, and the
    #: steady state parameters when the analysis has them
    console_parameters = (
        "cmax",
        "tmax",
        "auc_last",
        "auc_inf_obs",
        "auc_extrap_fraction",
        "thalf",
        "cl",
        "cl_f",
        "vz",
        "vz_f",
        "mrt",
        "auc_tau",
        "cmax_ss",
        "ctrough",
        "accumulation_ratio",
    )

    @property
    def dose(self) -> xr.DataArray | None:
        """The dose amount of every sample, `None` for a result without doses.

        The coordinate `dose_amount` the analysis carries over from the batch:
        the first dose of a single dose sample and the last dose of a multiple
        dose one, the dose its parameters are divided by.
        """
        if DOSE_COORDINATE not in self.ds.coords:
            return None
        return self.ds.coords[DOSE_COORDINATE]

    def dose_normalized_parameters(self) -> list[str]:
        """The parameters `dose_normalized` normalizes without being asked.

        Returns:
            The concentration and exposure parameters of the result, those
            whose unit expression is `{unit}` or `({unit}) * ({time})`
            (`DOSE_NORMALIZED_EXPRESSIONS`), in the order of the dataset;
            a parameter which is itself dose normalized is left out.
        """
        from pkpdutils.nca.nca import PARAMETER_UNITS

        return [
            name
            for name in self.parameters
            if not name.endswith(DOSE_NORMALIZED_SUFFIX)
            and PARAMETER_UNITS.get(name) in DOSE_NORMALIZED_EXPRESSIONS
        ]

    def dose_normalized(self, parameters: Sequence[str] | None = None) -> "NCAResult":
        r"""A copy of the result with the dose normalized variables of its parameters.

        The dose normalized variable of a parameter is the parameter divided by
        the dose of its sample,

        $$x_\mathrm{dn} = \frac{x}{D},$$

        with the unit of the parameter per dose unit; `NaN` where the sample
        has no positive dose (a placebo arm). It is the form ICH M13A (2024)
        asks for when strengths are compared, and the `*D` family of the CDISC
        codelist (Phoenix `AUClast_D`, `Cmax_D`; PKNCA `pk.calc.dn`).

        The variable is named `x_dn`, except for `auc_inf_obs`, whose
        normalized variable is the `auc_inf_dn` every analysis already reports
        (`DOSE_NORMALIZED_NAMES`). Normalize before summarizing: the summary of
        a dimension carries no dose coordinate any more.

        Args:
            parameters: the parameters to normalize, the concentrations and
                exposures of the result by default
                (`dose_normalized_parameters`).

        Returns:
            A copy of the result with one dose normalized variable per
            parameter added.

        Raises:
            ValueError: if the result carries no dose (an analysis of a batch
                without doses), or if a name is not a parameter of the result.
        """
        dose = self.dose
        if dose is None:
            raise ValueError(
                "the result carries no dose, so no parameter can be normalized "
                "by it; the analysed batch has no dose amounts"
            )
        names = list(
            self.dose_normalized_parameters() if parameters is None else parameters
        )
        missing = [name for name in names if name not in self.ds.data_vars]
        if missing:
            raise ValueError(f"{missing} are no parameters of the result")
        dose_unit = str(dose.attrs.get("units", ""))
        with np.errstate(divide="ignore", invalid="ignore"):
            amount = dose.where(dose > 0.0)
        ds = self.ds.copy()
        for name in names:
            unit, factor = _per_dose(self.units(name), dose_unit)
            normalized = (self.ds[name] / amount) * factor
            normalized.attrs = {"units": unit}
            ds[DOSE_NORMALIZED_NAMES.get(name, f"{name}{DOSE_NORMALIZED_SUFFIX}")] = (
                normalized
            )
        return NCAResult(ds)

    def exclude(
        self,
        mask: "npt.ArrayLike | xr.DataArray | None" = None,
        *,
        reason: str = "",
        **indexers: Any,
    ) -> "NCAResult":
        """A copy of the result with further samples marked as excluded.

        The excluded samples stay in the result - `to_dataframe` reports every
        row and the `excluded` column says which - and are left out of
        `summarize`, `summary_table`, `ParameterResult.sample` and therefore of
        every statistic of `pkpdutils.stats` which reads a result, unless
        `include_excluded=True` asks for them. It is the record-level and
        subject-level exclusion a regulatory analysis documents (CDISC ADNCA
        carries the subject-level exclusion flags; PKNCA the
        `exclude_nca_*` rules), and the same mechanism
        `pkpdutils.nca.options.Acceptance(exclude=True)` uses.

        Args:
            mask: the samples to exclude, a boolean array over the sample
                dimensions or a boolean `xarray.DataArray` along them;
                `None` with `indexers` to name single samples.
            reason: the text written into `excluded_reason` of the newly
                excluded samples; the reason of a sample which was already
                excluded is kept.
            **indexers: coordinate label per sample dimension of one sample, as
                `xarray.Dataset.sel` takes them; a dimension without an indexer
                is excluded as a whole.

        Returns:
            A copy of the result with `excluded` set and `excluded_reason`
            written.

        Raises:
            ValueError: if neither `mask` nor `indexers` are given, if both
                are, or if the mask does not have the shape of the samples.
        """
        if (mask is None) == (not indexers):
            raise ValueError(
                "give either 'mask', a boolean array over the sample dimensions, "
                "or the indexers of the samples to exclude"
            )
        template = xr.zeros_like(self.ds["flags"], dtype=bool)
        if mask is None:
            selected = template.copy()
            selected.loc[indexers] = True
        elif isinstance(mask, xr.DataArray):
            selected = (
                mask.astype(bool).broadcast_like(template).transpose(*template.dims)
            )
        else:
            values = np.asarray(mask, dtype=bool)
            if values.shape != template.shape:
                raise ValueError(
                    f"'mask' has the shape {values.shape}, the samples "
                    f"{template.dims} have {template.shape}"
                )
            selected = xr.DataArray(values, dims=template.dims, coords=template.coords)
        ds = self.ds.copy()
        before = (
            ds[EXCLUDED_VARIABLE].astype(bool)
            if EXCLUDED_VARIABLE in ds.data_vars
            else template
        )
        excluded = before | selected
        excluded.attrs = {"units": "dimensionless"}
        ds[EXCLUDED_VARIABLE] = excluded
        reasons = (
            ds[REASON_VARIABLE]
            if REASON_VARIABLE in ds.data_vars
            else xr.full_like(template, "", dtype=object)
        )
        # a sample which was already excluded keeps the reason it carries
        written = xr.where(selected & ~before, reason, reasons.astype(str))
        written.attrs = {"units": "dimensionless"}
        ds[REASON_VARIABLE] = written
        return NCAResult(ds)

    def terminal_windows(self) -> dict[Any, tuple[float, float]]:
        """The terminal window of every sample, keyed as `TerminalPhase.windows`.

        `lambda_z_t_first` and `lambda_z_t_last` of every sample with a
        terminal phase, keyed by the sample label (the label of a result with
        one sample dimension, the tuple of labels of a result with several), so
        that

        ```python
        # not executed
        reviewed = nca(batch, options=options.model_copy(
            update={"terminal": TerminalPhase(windows=result.terminal_windows())}
        ))
        ```

        re-runs the analysis with exactly the windows of `result`. A sample
        without a terminal phase carries no window and follows
        `TerminalPhase.method` again, which reproduces its result as well.

        Returns:
            Sample label to `(t_first, t_last)`, in the times of the analysis
            (relative to the reference dose of the sample).
        """
        from pkpdutils.nca.nca import sample_keys

        if "lambda_z_t_first" not in self.ds.data_vars:
            return {}
        first = self.ds["lambda_z_t_first"].to_numpy().reshape(-1)
        last = self.ds["lambda_z_t_last"].to_numpy().reshape(-1)
        labels = sample_keys(self.ds, self.sample_dims)
        return {
            label: (float(t_first), float(t_last))
            for label, t_first, t_last in zip(labels, first, last, strict=True)
            if np.isfinite(t_first) and np.isfinite(t_last)
        }

    @property
    def partial_aucs(self) -> dict[str, tuple[float, float]]:
        """The named partial areas of the analysis with their intervals.

        `NCAOptions.partial_aucs` as the analysis ran it, stored by `nca` in
        the attributes of the dataset: the name of every area to its
        `(t_start, t_end)`, both relative to the first dose of the protocol.
        The figures read it to shade an area (`pkpdutils.plot.plot_nca`), and
        it is empty for an analysis which computed none.

        Returns:
            Name to interval, empty without named areas.
        """
        stored = self.ds.attrs.get(PARTIAL_AUCS_ATTR) or {}
        return {
            str(name): (float(bounds[0]), float(bounds[1]))
            for name, bounds in dict(stored).items()
        }

    @property
    def has_intervals(self) -> bool:
        """Whether the result carries the parameters of the single dosing intervals."""
        return INTERVAL_DIM in self.ds.dims and bool(self._interval_variables)

    @property
    def _interval_variables(self) -> list[str]:
        """Names of the per-interval variables, in the order of the dataset."""
        return [
            str(name)
            for name in self.ds.data_vars
            if str(name).startswith(INTERVAL_PREFIX)
        ]

    def intervals(self) -> pd.DataFrame:
        """The per-interval parameters as one row per sample and dosing interval.

        The interval variables carry the dimension `interval` beyond the sample
        dimensions and are therefore point variables, which `to_dataframe`
        leaves out; this frame reports them with the sample coordinates and the
        number of the interval.

        Returns:
            One row per sample and interval with the sample coordinates (the
            dimension coordinates and the coordinates along them, such as the
            weight of a subject), `interval` and every `interval_*` variable;
            an empty frame for a single dose result.
        """
        names = self._interval_variables
        if not self.has_intervals:
            return pd.DataFrame()
        df = self.ds[names].to_dataframe().reset_index()
        leading = [*self.sample_dims, INTERVAL_DIM]
        # the non-dimension coordinates along the sample dimensions travel with
        # the samples, as they do in the result itself
        coordinates = [
            column for column in df.columns if column not in (*leading, *names)
        ]
        return df[[*leading, *coordinates, *names]]
