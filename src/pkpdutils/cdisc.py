"""CDISC map of the parameters: `PKPARMCD` codes, `PKUNIT` spellings and the `PP` domain.

A submission does not carry the variable names of an analysis package: every
parameter of the `PP` domain (and of the ADaM `ADPP` dataset derived from it)
is named by a code of the CDISC controlled terminology, `PKPARMCD`, and its
unit by a value of `PKUNIT`. This module holds the crosswalk from the variables
`pkpdutils` reports to those codes and writes the domain.

The codes are read from `pkpdutils/data/pkparmcd.csv`, which was extracted from
the tab-delimited NCI EVS package of the CDISC SDTM controlled terminology
(codelist `C85839`, "PK Parameters Code"); the header of the file names the
source, the date and the checksum of the package it was taken from. Nothing is
transcribed by hand, and a variable the terminology has no code for maps to
`None` and is left out of the domain with a warning. Several codes that are
commonly assumed do not exist (`CLSTP` for the predicted last concentration,
`CMAXSS` and `CMINSS` for the steady state peak and trough, `ACCIND`,
`PTROUGH`, `AEAMT`, `FE`): the steady state peak and trough are `CMAX` and
`CMIN` with `PPSCAT = "STEADY STATE"`, the accumulation index is `AILAMZ` and
the peak trough ratio `PTROUGHR`.

```python
from pkpdutils.cdisc import to_pp

pp = to_pp(result, subject_dim="individual")
```
"""

import csv
import logging
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from pkpdutils.result import ParameterResult
from pkpdutils.timecourse import Route
from pkpdutils.units import parse_unit

logger = logging.getLogger(__name__)

#: the data file with the `PKPARMCD` crosswalk
PKPARMCD_FILE: str = "data/pkparmcd.csv"

#: the substance of a batch which does not name one
#: (`pkpdutils.timecourse.Timecourses.substance`); `PPCAT` is empty for it
UNNAMED_SUBSTANCE: str = "substance"


def _read_codes() -> tuple[dict[str, str], dict[str, dict[Route, str]], dict[str, str]]:
    """Read the crosswalk of `PKPARMCD_FILE`.

    Returns:
        The code of every variable with one code, the codes of every variable
        whose code depends on the route, and the CDISC name of every code.
    """
    text = files("pkpdutils").joinpath(PKPARMCD_FILE).read_text(encoding="utf-8")
    rows = csv.DictReader(
        line for line in text.splitlines() if not line.startswith("#")
    )
    codes: dict[str, str] = {}
    by_route: dict[str, dict[Route, str]] = {}
    names: dict[str, str] = {}
    for row in rows:
        variable, code = row["variable"], row["code"]
        names[code] = row["name"]
        condition = row["condition"].strip()
        if condition:
            by_route.setdefault(variable, {})[Route(condition)] = code
        else:
            codes[variable] = code
    return codes, by_route, names


#: `PKPARMCD` code of every variable of a result which has one, e.g.
#: `PKPARMCD["auc_inf_obs"] == "AUCIFO"`; a variable without a code is not a
#: key (`pkparmcd` returns `None` for it)
PKPARMCD: dict[str, str]

#: `PKPARMCD` codes of the variables whose code depends on the route of
#: administration: the mean residence time is `MRTEVIFO` after an extravascular
#: dose, `MRTIBIFO` after a bolus and `MRTICIFO` after an infusion
PKPARMCD_BY_ROUTE: dict[str, dict[Route, str]]

#: the CDISC name of every code of the crosswalk, the `PPTEST` of the domain
PKPARMCD_NAMES: dict[str, str]

PKPARMCD, PKPARMCD_BY_ROUTE, PKPARMCD_NAMES = _read_codes()

#: the variables which carry `PPSCAT = "STEADY STATE"` in the domain whatever
#: the sample they belong to; the peak and the trough of a dosing interval are
#: `CMAX` and `CMIN` like those of a single dose, and this is what tells the
#: two apart (there is no `CMAXSS` code). Every parameter of a sample which
#: was analysed over its dosing intervals is `STEADY STATE` as well
#: (`_steady_state_samples`)
STEADY_STATE_VARIABLES: frozenset[str] = frozenset(
    {
        "auc_tau",
        "auc_tau_dn",
        "auc_tau_extrap_fraction",
        "cmax_ss",
        "cmax_ss_dn",
        "cmin_ss",
        "ctrough",
        "cavg",
        "cavg_dn",
        "fluctuation",
        "fluctuation_tau",
        "swing",
        "swing_tau",
        "ptr",
        "tau",
        "cl_ss",
        "cl_ss_f",
        "accumulation_ratio",
        "accumulation_ratio_obs",
        "accumulation_ratio_cmax_obs",
        "accumulation_ratio_cmin_obs",
        "accumulation_ratio_ctrough_obs",
        "stationarity_ratio",
    }
)

#: the variables of the urinary excretion analysis (`pkpdutils.nca.urine`),
#: whose `PPSPEC` is `URINE` rather than the specimen of the batch
URINE_VARIABLES: frozenset[str] = frozenset(
    {
        "rate",
        "midpoint",
        "max_rate",
        "tmax_rate",
        "rate_last",
        "mid_pt_last",
        "aurc_last",
        "aurc_all",
        "aurc_inf_obs",
        "aurc_inf_pred",
        "amount_recovered",
        "percent_recovered",
        "vol_ur",
        "clr",
    }
)

#: the variables whose CDISC code is defined as a percentage while the package
#: reports a fraction (`AUCPEO` is "AUC ... as a percentage of the area under
#: the curve extrapolated to infinity", `FLUCP` is `Fluctuation%`): the domain
#: carries the value multiplied by 100 with the unit `%`
PERCENT_VARIABLES: frozenset[str] = frozenset(
    {"auc_extrap_fraction", "auc_back_extrap_fraction", "fluctuation"}
)

#: `PKUNIT` spelling of the units an analysis derives, the canonical pint
#: string of the unit to the CDISC submission value (`"hour * nanogram /
#: milliliter"` is `"h*ng/mL"`). A unit which is not in the table is written in
#: the same symbols in the order pint spells it (`pkunit`), which is a
#: `PKUNIT` value for most units and a readable approximation for the rest;
#: `pkpdutils.result.ParameterResult.to_units` converts a result to a unit the
#: terminology spells exactly.
PKUNIT: dict[str, str] = {}

#: the CDISC symbol of every unit the spellings are composed of, for a unit
#: which is not in `PKUNIT` itself
PKUNIT_SYMBOLS: dict[str, str] = {
    "day": "day",
    "hour": "h",
    "minute": "min",
    "second": "s",
    "liter": "L",
    "milliliter": "mL",
    "deciliter": "dL",
    "microliter": "uL",
    "kilogram": "kg",
    "gram": "g",
    "milligram": "mg",
    "microgram": "ug",
    "nanogram": "ng",
    "picogram": "pg",
    "femtogram": "fg",
    "mole": "mol",
    "millimole": "mmol",
    "micromole": "umol",
    "nanomole": "nmol",
    "picomole": "pmol",
    "IU": "IU",
    "percent": "%",
    "meter": "m",
}

#: the spellings of `PKUNIT`, as the unit expression and the submission value;
#: the expression is parsed with pint, so that the table is keyed by the
#: canonical unit string of the registry whatever the data spelled
_PKUNIT_SPELLINGS: tuple[tuple[str, str], ...] = (
    ("hour", "h"),
    ("minute", "min"),
    ("day", "day"),
    ("1 / hour", "/h"),
    ("1 / minute", "/min"),
    ("1 / day", "/day"),
    ("liter", "L"),
    ("milliliter", "mL"),
    ("milligram", "mg"),
    ("microgram", "ug"),
    ("nanogram", "ng"),
    ("millimole", "mmol"),
    ("micromole", "umol"),
    ("nanomole", "nmol"),
    ("gram / milliliter", "g/mL"),
    ("milligram / milliliter", "mg/mL"),
    ("microgram / milliliter", "ug/mL"),
    ("nanogram / milliliter", "ng/mL"),
    ("picogram / milliliter", "pg/mL"),
    ("mole / liter", "mol/L"),
    ("millimole / liter", "mmol/L"),
    ("micromole / liter", "umol/L"),
    ("nanomole / liter", "nmol/L"),
    ("picomole / liter", "pmol/L"),
    ("hour * milligram / milliliter", "h*mg/mL"),
    ("hour * microgram / milliliter", "h*ug/mL"),
    ("hour * nanogram / milliliter", "h*ng/mL"),
    ("hour * picogram / milliliter", "h*pg/mL"),
    ("hour * millimole / liter", "h*mmol/L"),
    ("hour * micromole / liter", "h*umol/L"),
    ("hour * nanomole / liter", "h*nmol/L"),
    ("minute * microgram / milliliter", "min*ug/mL"),
    ("minute * nanogram / milliliter", "min*ng/mL"),
    ("day * microgram / milliliter", "day*ug/mL"),
    ("day * nanogram / milliliter", "day*ng/mL"),
    ("hour ** 2 * milligram / milliliter", "h2*mg/mL"),
    ("hour ** 2 * microgram / milliliter", "h2*ug/mL"),
    ("hour ** 2 * nanogram / milliliter", "h2*ng/mL"),
    ("hour ** 2 * millimole / liter", "h2*mmol/L"),
    ("hour ** 2 * micromole / liter", "h2*umol/L"),
    ("liter / hour", "L/h"),
    ("liter / minute", "L/min"),
    ("liter / day", "L/day"),
    ("milliliter / hour", "mL/h"),
    ("milliliter / minute", "mL/min"),
    ("liter / hour / kilogram", "(L/h)/kg"),
    ("milliliter / hour / kilogram", "(mL/h)/kg"),
    ("milliliter / minute / kilogram", "(mL/min)/kg"),
    ("liter / kilogram", "L/kg"),
    ("milliliter / kilogram", "mL/kg"),
    ("hour * nanogram / milligram / milliliter", "h*ng/mL/mg"),
    ("hour * microgram / milligram / milliliter", "h*ug/mL/mg"),
    ("nanogram / milligram / milliliter", "ng/mL/mg"),
    ("microgram / milligram / milliliter", "ug/mL/mg"),
    ("percent", "%"),
)

PKUNIT = {str(parse_unit(expression)): value for expression, value in _PKUNIT_SPELLINGS}

#: the submission values of `PKUNIT`, which is how `to_pp` tells a unit the
#: terminology spells from one it wrote in the CDISC symbols itself
PKUNIT_VALUES: frozenset[str] = frozenset(PKUNIT.values())


def pkunit(unit: str) -> str:
    """The `PKUNIT` spelling of a unit of a result.

    The exact submission value of `PKUNIT` when the table knows the unit,
    otherwise the same unit written in the CDISC symbols of `PKUNIT_SYMBOLS`
    in the order pint spells it (`milligram / liter` becomes `mg/L`, which the
    terminology does not have, since it spells mass concentrations per
    milliliter; `ParameterResult.to_units` converts such a result to a unit the
    terminology spells). A dimensionless unit is the empty string, a unit which
    is not a unit of the registry is passed through unchanged.

    Args:
        unit: the unit string of a variable of a result, e.g.
            `"hour * nanogram / milliliter"`.

    Returns:
        The `PKUNIT` submission value.
    """
    if not unit.strip() or unit == "dimensionless":
        return ""
    try:
        canonical = str(parse_unit(unit))
    except ValueError:
        return unit
    if canonical in PKUNIT:
        return PKUNIT[canonical]
    if canonical == "dimensionless":
        return ""
    spelled = _spell(canonical)
    logger.debug("'%s' is not a PKUNIT value, written as '%s'", unit, spelled)
    return spelled


def _spell(canonical: str) -> str:
    """Write a canonical pint unit in the CDISC symbols.

    Args:
        canonical: the canonical unit string of the registry, e.g.
            `"hour ** 2 * nanogram / milliliter"`.

    Returns:
        The unit in the symbols of `PKUNIT_SYMBOLS` without blanks, e.g.
        `"h2*ng/mL"`.
    """
    out = ""
    tokens = canonical.split(" ")
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in ("*", "/"):
            out += token
        elif token == "**":
            # the exponent follows the unit it belongs to, `h2` for `hour ** 2`
            out += tokens[index + 1]
            index += 1
        else:
            out += PKUNIT_SYMBOLS.get(token, token)
        index += 1
    return out


def pkparmcd(variable: str, route: Route | str | None = None) -> str | None:
    """The `PKPARMCD` code of a variable of a result.

    Args:
        variable: name of the variable, e.g. `"auc_inf_obs"` or `"mrt"`.
        route: the route of administration, for a variable whose code depends
            on it (`mrt`); `None` when it is not known.

    Returns:
        The code, or `None` when the terminology has none for the variable or
        when the code needs a route which was not given.
    """
    if variable in PKPARMCD:
        return PKPARMCD[variable]
    by_route = PKPARMCD_BY_ROUTE.get(variable)
    if by_route is None or route is None:
        return None
    return by_route.get(Route(route))


def _usubjid_of(
    usubjid: Mapping[Any, str] | Sequence[str] | None, label: Any, position: int
) -> str:
    """The `USUBJID` of one subject.

    Args:
        usubjid: the identifiers given by the caller: a mapping from the label
            of the subject to its `USUBJID`, a sequence in the order of the
            subject dimension, or `None` for the label itself.
        label: the label of the subject along the subject dimension.
        position: the position of the subject along the subject dimension.

    Returns:
        The identifier.
    """
    if usubjid is None:
        return str(label)
    if isinstance(usubjid, Mapping):
        return str(usubjid.get(label, label))
    return str(usubjid[position])


def _coordinate_at(
    result: ParameterResult, name: str, index: tuple[int, ...]
) -> str | None:
    """The value of a coordinate of the result at one sample, `None` without it.

    Args:
        result: the result.
        name: name of the coordinate (`substance`, `route`).
        index: the position of the sample along the sample dimensions.

    Returns:
        The value as a string, or `None` when the result carries no such
        coordinate.
    """
    if name not in result.ds.coords:
        return None
    coord = result.ds.coords[name]
    dims = result.sample_dims
    position = tuple(index[dims.index(str(d))] for d in coord.dims)
    return str(coord.to_numpy()[position] if position else coord.to_numpy())


def _steady_state_samples(result: ParameterResult) -> np.ndarray | None:
    """The samples whose parameters were computed from the last dose on.

    A sample of more than one dose (`n_doses`), and every sample of an analysis
    with `NCAOptions.tau` (`auc_tau`), is analysed over its dosing intervals,
    so its peak, its exposure and its clearance describe the steady state
    rather than a single dose (`pkpdutils.nca.steady_state`); `PPSCAT` says so
    for every parameter of such a sample.

    Args:
        result: the result.

    Returns:
        The boolean array over the sample dimensions, `None` when the result
        holds nothing of a multiple dose analysis.
    """
    dims = result.sample_dims
    steady: np.ndarray | None = None
    if "n_doses" in result.ds.data_vars:
        steady = (
            np.asarray(
                result.ds["n_doses"].transpose(*dims).to_numpy(), dtype=np.float64
            )
            > 1.0
        )
    if "auc_tau" in result.ds.data_vars:
        # a single dose profile analysed with `NCAOptions.tau` carries one dose
        # and the parameters of a dosing interval
        covered = np.isfinite(
            np.asarray(
                result.ds["auc_tau"].transpose(*dims).to_numpy(), dtype=np.float64
            )
        )
        steady = covered if steady is None else (steady | covered)
    return steady


def _time_unit(result: ParameterResult) -> str:
    """The time unit of a result, read from a parameter which is a time.

    Args:
        result: the result.

    Returns:
        The `PKUNIT` spelling of the time unit, the empty string when the
        result carries no time parameter.
    """
    for name in ("tmax", "tlast", "tau", "thalf"):
        if name in result.ds.data_vars:
            return pkunit(result.units(name))
    return ""


def to_pp(
    result: ParameterResult,
    *,
    subject_dim: str,
    usubjid: Mapping[Any, str] | Sequence[str] | None = None,
    spec: Literal["SDTM", "ADaM"] = "SDTM",
    studyid: str | None = None,
    route: Route | str | None = None,
    ppspec: str | None = None,
    digits: int = 6,
) -> pd.DataFrame:
    """Lay a result out as the CDISC `PP` domain, one row per subject and parameter.

    Every parameter of the result which the terminology has a code for
    (`pkparmcd`) becomes one row per sample: `PPTESTCD` the code, `PPTEST` its
    CDISC name, `PPORRES` the value as it was reported and `PPORRESU` its unit
    in the `PKUNIT` spelling (`pkunit`), `PPSTRESN` and `PPSTRESU` the same
    value as a number. `PPCAT` is the substance the analysis was run on and
    `PPSPEC` the specimen; `PPSEQ` numbers the rows of a subject from 1.

    `PPSCAT` tells a single dose parameter from a steady state one, and the
    analysis of the sample decides it: every parameter of a sample which was
    analysed over its dosing intervals (more than one dose, or
    `NCAOptions.tau`, `_steady_state_samples`) describes the steady state,
    since its peak, its exposure and its clearance are computed from the last
    dose on; a variable of `STEADY_STATE_VARIABLES` is `STEADY STATE` whatever
    the sample, which is what tells the steady state peak and trough from a
    single dose one (both are `CMAX` and `CMIN`, there is no `CMAXSS`). `PPRFTDTC`, the
    reference date-time of the analysis, is empty: the analysis works on
    elapsed times and never sees a date.

    A parameter without a code is left out and named in a warning. The
    variables which CDISC defines as a percentage while the package reports a
    fraction (`PERCENT_VARIABLES`) are multiplied by 100 and carry the unit
    `%`. A sample whose parameter is `NaN` (a parameter of the other route or
    of the other dosing path) gets no row. The uncertainty and summary
    variables of a parameter, the per-interval point variables and the status
    variables are not part of the domain.

    The substance and the route are read from the coordinates `substance` and
    `route` of the result when it carries them (a batch of several analytes or
    of several routes) and from its attributes otherwise; `route` names the
    route of a result which carries neither and is needed for the mean
    residence time, whose code depends on it.

    Args:
        result: the result, e.g. of `pkpdutils.nca.nca`

    Keyword Args:
        subject_dim: the sample dimension whose labels are the subjects
        usubjid: the `USUBJID` of every subject, a mapping from the label of
            the subject or a sequence in the order of the dimension; the label
            itself by default
        spec: `"SDTM"` writes the `PP` domain, `"ADaM"` adds the analysis
            variables `PARAMCD`, `PARAM`, `AVAL` and `AVALU` of an `ADPP`
            dataset and leaves the `DOMAIN` column out
        studyid: the `STUDYID` of the study, left out when it is not given
        route: route of administration, when the result names none itself
        ppspec: the specimen, `"PLASMA"` by default and `"URINE"` for the
            variables of a urinary excretion analysis (`URINE_VARIABLES`); the
            `tissue` of the analysed batch when it names one
        digits: significant digits of the character result `PPORRES`; the
            numeric `PPSTRESN` carries the value itself

    Returns:
        The domain, one row per subject and parameter.

    Raises:
        ValueError: if `subject_dim` is not a sample dimension of the result.
    """
    dims = result.sample_dims
    if subject_dim not in dims:
        raise ValueError(
            f"'{subject_dim}' is not a sample dimension of the result {list(dims)}"
        )
    ds = result.ds
    subject_axis = dims.index(subject_dim)
    labels = (
        ds[subject_dim].to_numpy()
        if subject_dim in ds.coords
        else np.arange(ds.sizes[subject_dim])
    )
    default_spec = str(ds.attrs.get("tissue", "plasma")).upper()
    # `parameters` leaves out the derived, the point and the status variables
    parameters = result.parameters
    without_code: set[str] = set()
    spelled_by_hand: set[str] = set()
    rows: list[dict[str, Any]] = []
    shape = tuple(int(ds.sizes[d]) for d in dims)
    sequence: dict[str, int] = {}
    time_unit = _time_unit(result)
    steady = _steady_state_samples(result)

    # everything a parameter carries whatever the sample is: its values in the
    # dimension order of the result, its unit and the test it names. Only the
    # code of a variable whose code depends on the route (`mrt`) is left to the
    # sample, and it is resolved once per route.
    static: dict[str, tuple[np.ndarray, str, str | None, bool]] = {}
    for name in parameters:
        raw_unit = result.units(name)
        unit = pkunit(raw_unit)
        if unit and unit not in PKUNIT_VALUES:
            spelled_by_hand.add(raw_unit)
        window = ds[name].attrs.get("window")
        test = (
            None
            if window is None
            else f"AUC from {window[0]:g} to {window[1]:g} {time_unit}".strip()
        )
        static[name] = (
            np.asarray(ds[name].transpose(*dims).to_numpy(), dtype=np.float64),
            unit,
            test,
            name in STEADY_STATE_VARIABLES,
        )
    codes: dict[Any, dict[str, str]] = {}

    def resolved(sample_route: Any) -> dict[str, str]:
        """The code of every parameter which has one, for one route.

        Args:
            sample_route: the route of the sample, `None` when it names none.

        Returns:
            Parameter name to its `PKPARMCD` code; a parameter without one is
            not a key and is collected in `without_code`.
        """
        if sample_route not in codes:
            found: dict[str, str] = {}
            for name in parameters:
                code = pkparmcd(name, sample_route)
                if code is None and static[name][2] is not None:
                    # a named partial area (`NCAOptions.partial_aucs`) is
                    # `AUCINT` and names its interval in `PPTEST`
                    code = "AUCINT"
                if code is None:
                    without_code.add(name)
                else:
                    found[name] = code
            codes[sample_route] = found
        return codes[sample_route]

    for index in np.ndindex(*shape) if shape else [()]:
        subject = _usubjid_of(usubjid, labels[index[subject_axis]], index[subject_axis])
        substance = _coordinate_at(result, "substance", index) or str(
            ds.attrs.get("substance", "")
        )
        if substance == UNNAMED_SUBSTANCE:
            # the placeholder of a batch which does not name its substance
            substance = ""
        sample_route = _coordinate_at(result, "route", index) or ds.attrs.get("route")
        sample_route = route if sample_route is None else sample_route
        sample_codes = resolved(sample_route)
        multiple_dose = bool(steady[index]) if steady is not None else False
        for name in parameters:
            code = sample_codes.get(name)
            if code is None:
                continue
            values, unit, test, steady_state = static[name]
            value = float(values[index])
            if not np.isfinite(value):
                continue
            if name in PERCENT_VARIABLES:
                value, unit = value * 100.0, "%"
            sequence[subject] = sequence.get(subject, 0) + 1
            row: dict[str, Any] = {
                "STUDYID": studyid,
                "DOMAIN": "PP",
                "USUBJID": subject,
                "PPSEQ": sequence[subject],
                "PPTESTCD": code,
                "PPTEST": PKPARMCD_NAMES[code] if test is None else test,
                "PPCAT": substance,
                "PPSCAT": (
                    "STEADY STATE" if steady_state or multiple_dose else "SINGLE DOSE"
                ),
                "PPORRES": f"{value:.{digits}g}",
                "PPORRESU": unit,
                "PPSTRESN": value,
                "PPSTRESU": unit,
                "PPSPEC": (
                    ppspec
                    if ppspec is not None
                    else ("URINE" if name in URINE_VARIABLES else default_spec)
                ),
                "PPRFTDTC": "",
            }
            if spec == "ADaM":
                row.update(
                    {
                        "PARAMCD": code,
                        "PARAM": row["PPTEST"],
                        "AVAL": value,
                        "AVALU": unit,
                    }
                )
            rows.append(row)
    if without_code:
        logger.warning(
            "no PKPARMCD code for %s, left out of the domain", sorted(without_code)
        )
    if spelled_by_hand:
        logger.warning(
            "%s is no PKUNIT value and was written in the CDISC symbols; "
            "convert the result with ParameterResult.to_units to a unit the "
            "terminology spells",
            sorted(spelled_by_hand),
        )
    columns = [
        "STUDYID",
        "DOMAIN",
        "USUBJID",
        "PPSEQ",
        "PPTESTCD",
        "PPTEST",
        "PPCAT",
        "PPSCAT",
        "PPORRES",
        "PPORRESU",
        "PPSTRESN",
        "PPSTRESU",
        "PPSPEC",
        "PPRFTDTC",
    ]
    if spec == "ADaM":
        # an ADaM dataset has no DOMAIN and carries the analysis variables
        columns = [column for column in columns if column != "DOMAIN"]
        columns[4:4] = ["PARAMCD", "PARAM"]
        columns += ["AVAL", "AVALU"]
    if studyid is None:
        columns = [column for column in columns if column != "STUDYID"]
    return pd.DataFrame(rows, columns=columns)


def write_pp(result: ParameterResult, path: str | Path, **options: Any) -> pd.DataFrame:
    """Write a result as the `PP` domain into a csv file.

    Args:
        result: the result
        path: the file to write
        **options: the keyword arguments of `to_pp` (`subject_dim` is required)

    Returns:
        The domain that was written.
    """
    frame = to_pp(result, **options)
    frame.to_csv(path, index=False)
    logger.info("wrote %d rows of the PP domain to %s", len(frame), path)
    return frame
