"""Exchange formats of pharmacokinetic data.

Readers and a writer for the table formats the field exchanges timecourses and
dosing protocols in. Every reader takes a pandas `DataFrame` (the caller reads
the csv, sas or xpt file) and returns a `pkpdutils.timecourse.Timecourses`
batch with one sample dimension, the times as given (a reader never shifts the
time axis), one route and the dosing protocol of every subject:

- **event records** (`read_events`, `write_events`): the one row per event
  format of NONMEM and Monolix, a row being a dose (`EVID 1`/`4`) or an
  observation (`EVID 0`). Repeated doses are given explicitly, as `ADDL`
  additional doses at the interdose interval `II`, or as a steady state dose
  (`SS 1`); see Bauer (2019) and the Monolix data format documentation.
- **PKNCA tables** (`read_pknca`): the two table layout of the R package
  `PKNCA`, the concentrations and the doses, joined on the subject and the
  grouping columns; see Denney et al. (2015).
- **CDISC ADNCA** (`read_adnca`): the analysis dataset of a non-compartmental
  analysis of the ADaM standard, one row per concentration record with the
  time since the first dose (`AFRLT`) and since the reference dose (`ARRLT`),
  see the CDISC ADaM ADNCA implementation guide (2024).

The readers are also reachable as the constructors `Timecourses.from_events`,
`Timecourses.from_pknca` and `Timecourses.from_adnca`, the writer as
`Timecourses.to_events`.

```python
import pandas as pd

from pkpdutils import Route, Timecourses

df = pd.read_csv("study.csv", na_values=".")
batch = Timecourses.from_events(
    df, time_unit="hr", unit="ng/ml", dose_unit="mg", route=Route.ORAL
)
```

Columns are looked up case-insensitively, a column which is not in the table is
treated as absent (a missing required column raises). Compartment columns
(`CMT`, `ADM`) are not interpreted and modelled rates (`RATE -1`, `RATE -2`)
are not data: both are out of scope, a batch has one route and the caller
filters the table before reading it.

The references of the formats are the "Data formats" section of
`docs/references.md`.
"""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

# `_pad_protocols` is the padding of the dose variables of a batch, shared with
# the constructors of `Timecourses` (the readers do not go through
# `Timecourse`: a subject of an exchange format may have a single observation
# or no dose records at all, which a single curve does not allow)
from pkpdutils.timecourse import Dosing, Route, Timecourses, _pad_protocols
from pkpdutils.units import parse_unit

logger = logging.getLogger(__name__)

#: Monolix column names of the event record format and the role they fill
EVENT_ALIASES: dict[str, str] = {
    "AMOUNT": "amt",
    "OBSERVATION": "dv",
    "ADDITIONAL DOSES": "addl",
    "INTERDOSE INTERVAL": "ii",
    "STEADY STATE": "ss",
    "INFUSION DURATION": "tinf",
    "INFUSION RATE": "rate",
}

#: columns of an event table which are never covariates: the compartment and
#: occasion columns, which the reader does not interpret
EVENT_IGNORED: frozenset[str] = frozenset({"ADM", "CMT", "DVID", "OCC", "YTYPE"})

#: values of the `ROUTE` column of an ADNCA dataset, upper case, and the route
#: they stand for
ADNCA_ROUTES: dict[str, Route] = {
    "ORAL": Route.ORAL,
    "PO": Route.ORAL,
    "IV": Route.IV_BOLUS,
    "INTRAVENOUS": Route.IV_BOLUS,
    "IV BOLUS": Route.IV_BOLUS,
    "INTRAVENOUS BOLUS": Route.IV_BOLUS,
    "IV INFUSION": Route.IV_INFUSION,
    "INTRAVENOUS INFUSION": Route.IV_INFUSION,
}


def _lookup(df: pd.DataFrame) -> dict[str, str]:
    """Map the lower case name of every column to the column itself.

    Args:
        df: the table.

    Returns:
        The mapping used by `_column` for the case-insensitive lookup.
    """
    return {str(column).strip().lower(): str(column) for column in df.columns}


def _column(
    lookup: dict[str, str],
    name: str | None,
    role: str | None = None,
    *,
    required: bool = False,
) -> str | None:
    """Find a column by name (case-insensitive) or by an alias of its role.

    Args:
        lookup: the mapping of `_lookup`.
        name: the name to look for, `None` for a column the caller switched off.
        role: the role of the column, e.g. `"amt"`; the Monolix aliases of
            `EVENT_ALIASES` with this role are tried after `name`.
        required: whether a missing column is an error.

    Returns:
        The name of the column as it is spelled in the table, or `None` when it
        is not there and not required.

    Raises:
        ValueError: if a required column is not in the table.
    """
    if name is not None:
        candidates = [name, *(a for a, r in EVENT_ALIASES.items() if r == role)]
        for candidate in candidates:
            column = lookup.get(candidate.strip().lower())
            if column is not None:
                return column
    if required:
        raise ValueError(
            f"The table has no '{name}' column, found {sorted(lookup.values())}"
        )
    return None


def _numeric(
    df: pd.DataFrame, column: str | None, default: float = np.nan
) -> pd.Series:
    """A column as a float series, `default` everywhere when the column is absent.

    Args:
        df: the table.
        column: name of the column, `None` when it is absent.
        default: value of the series when the column is absent.

    Returns:
        The series of floats, indexed like `df`; values which are not numbers
        become `NaN`.
    """
    if column is None:
        return pd.Series(default, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[column], errors="coerce").astype(np.float64)


def _constant_per_subject(
    df: pd.DataFrame, subject: str, column: str, labels: Sequence[Any]
) -> np.ndarray | None:
    """The value of a covariate column per subject, `None` when it is not constant.

    Args:
        df: the table.
        subject: name of the subject column.
        column: name of the covariate column.
        labels: the subjects, in the order of the batch.

    Returns:
        The first value of the column per subject in the order of `labels`, or
        `None` when a subject carries more than one value or no value at all.
    """
    grouped = df.groupby(subject, sort=False)[column]
    if (grouped.nunique(dropna=True) != 1).any():
        return None
    return grouped.first().reindex(labels).to_numpy()


def _subjects(df: pd.DataFrame, column: str) -> dict[Any, pd.Index]:
    """Group the rows of a table by subject, in the order of their first appearance.

    Args:
        df: the table.
        column: name of the subject column.

    Returns:
        The index of the rows of every subject, keyed by the subject.
    """
    missing = int(df[column].isna().sum())
    if missing:
        logger.warning("Dropped %d rows without a value in '%s'", missing, column)
    return dict(df.groupby(column, sort=False).groups)


def _pad(arrays: Sequence[np.ndarray], n_time: int) -> np.ndarray:
    """Stack the 1-D arrays of the subjects into `(n_subjects, n_time)`, padded with `NaN`.

    Args:
        arrays: one array per subject.
        n_time: number of columns, at least the length of the longest array.

    Returns:
        The padded array.
    """
    out = np.full((len(arrays), n_time), np.nan)
    for i, array in enumerate(arrays):
        out[i, : array.size] = array
    return out


def _build_batch(
    *,
    labels: Sequence[Any],
    times: Sequence[np.ndarray],
    values: Sequence[np.ndarray],
    protocols: Sequence[Dosing | None],
    time_unit: str,
    unit: str,
    dim: str,
    substance: str,
    coordinates: dict[str, np.ndarray],
) -> Timecourses:
    """Assemble the batch of the subjects a reader has collected.

    The subjects of an exchange format may have different sampling grids,
    different numbers of doses and, unlike a single `Timecourse`, a single
    observation or no dose records at all, so the batch is built from the
    padded arrays rather than from `Timecourses.from_timecourses`. A subject
    without a protocol gets a row of `NaN` dose entries.

    Args:
        labels: the subjects, in the order of the sample dimension.
        times: the sampling times of every subject, ascending.
        values: the values of every subject, with the times.
        protocols: the dosing protocol of every subject, `None` without doses.
        time_unit: unit of the times.
        unit: unit of the values.
        dim: name of the sample dimension.
        substance: name of the substance or effect.
        coordinates: the covariates of the subjects along `dim`.

    Returns:
        The batch, with the shared sampling grid as the `time` coordinate when
        every subject has the same one.

    Raises:
        ValueError: if there is no subject, or if the protocols do not share
            one route and one dose unit.
    """
    if not labels:
        raise ValueError("The table holds no subject")
    n_time = max(array.size for array in times)
    shared = all(
        array.size == times[0].size and np.array_equal(array, times[0])
        for array in times
    )
    grid: np.ndarray = times[0] if shared else _pad(times, n_time)

    dose: dict[str, Any] | None = None
    route: Route | None = None
    given = [protocol for protocol in protocols if protocol is not None]
    if given:
        routes = {protocol.route for protocol in given}
        units = {protocol.unit for protocol in given}
        if len(routes) != 1:
            raise ValueError(
                "A batch has one route, found "
                f"{sorted(r.value for r in routes)}; read the routes into "
                "separate batches"
            )
        if len(units) != 1:
            raise ValueError(f"All doses need the same unit, found {sorted(units)}")
        route = routes.pop()
        amounts, dose_times, durations = _pad_protocols(
            protocols, max(protocol.n_doses for protocol in given)
        )
        dose = {
            "amount": amounts,
            "unit": units.pop(),
            "time": dose_times,
            "duration": durations,
        }

    batch = Timecourses.from_arrays(
        grid,
        _pad(values, n_time),
        time_unit=time_unit,
        unit=unit,
        dims=(dim,),
        coords={dim: list(labels)},
        dose=dose,
        route=route,
        substance=substance,
    )
    if not coordinates:
        return batch
    return Timecourses(
        batch.ds.assign_coords({name: (dim, v) for name, v in coordinates.items()})
    )


def _dose_times(
    time: float,
    *,
    addl: float,
    interval: float,
    steady_state: bool,
    ss_doses: int,
    subject: Any,
) -> list[float]:
    """The times of a dose record, expanded by `ADDL` and `SS`.

    `ADDL` gives `addl` further doses at `time + k * interval`; a steady state
    record (`SS == 1`) stands for a dosing history, represented by `ss_doses`
    preceding doses at `time - k * interval` (Bauer 2019). Both need a positive
    interdose interval `II`: a record which asks for repeated doses without one
    is an incomplete table, not a single dose, and raises.

    Args:
        time: time of the record.
        addl: number of additional doses, `NaN` or 0 for none.
        interval: interdose interval `II`, `NaN` or 0 for none.
        steady_state: whether the record is marked as a steady state dose.
        ss_doses: number of preceding doses a steady state record stands for.
        subject: the subject of the record, for the error message.

    Returns:
        The dose times, ascending.

    Raises:
        ValueError: if the record has `ADDL > 0` or `SS == 1` without a
            positive interdose interval.
    """
    has_interval = np.isfinite(interval) and interval > 0
    repeated = steady_state or (np.isfinite(addl) and addl > 0)
    if repeated and not has_interval:
        raise ValueError(
            f"'ADDL'/'SS' need a positive 'II': subject '{subject}' has the "
            f"dose record at time {time} with ADDL {addl}, SS "
            f"{int(steady_state)} and II {interval}"
        )
    times = [time]
    if steady_state:
        times = [time - k * interval for k in range(ss_doses, 0, -1)] + times
    if np.isfinite(addl) and addl > 0:
        times = times + [time + k * interval for k in range(1, int(addl) + 1)]
    return times


def read_events(
    df: pd.DataFrame,
    *,
    time_unit: str,
    unit: str,
    dose_unit: str,
    route: Route,
    id: str = "ID",
    time: str = "TIME",
    dv: str = "DV",
    amt: str = "AMT",
    evid: str = "EVID",
    mdv: str = "MDV",
    rate: str = "RATE",
    tinf: str = "TINF",
    addl: str = "ADDL",
    ii: str = "II",
    ss: str = "SS",
    ss_doses: int = 5,
    dim: str = "individual",
    substance: str = "substance",
    covariates: Sequence[str] | None = None,
) -> Timecourses:
    """Read a batch from event records, the NONMEM and Monolix format.

    A row of the table is one event of one subject: a dose when `EVID` is 1 or
    4 (without an `EVID` column when `AMT > 0`), an observation when `EVID` is
    0 (or there is no `EVID` column), `MDV` is 0 (or there is no `MDV` column)
    and `DV` is not missing. Rows with `EVID` 2 (other type event) or 3 (reset)
    are dropped and counted in a warning (Bauer 2019). The two rules are read
    independently, so without an `EVID` column a row with `AMT > 0` and a
    value in `DV` is both a dose and an observation: a table which records a
    dose and a sample in one row needs no `EVID` column, a table which does not
    needs one.

    The duration of an infusion is `TINF` (Monolix) when it is positive, else
    `AMT / RATE` for a positive `RATE`; modelled rates (`RATE -1`, `RATE -2`)
    are not data and raise. A dose record with `ADDL` and `II` stands for
    `ADDL` further doses at the interdose interval, a record with `SS == 1`
    for a dosing history of `ss_doses` preceding doses at the interdose
    interval, and the batch is marked with `attrs["steady_state_marker"]`;
    `ADDL > 0` or `SS == 1` without a positive `II` is an incomplete table and
    raises.

    The Monolix column names `AMOUNT`, `OBSERVATION`, `INFUSION DURATION`,
    `INFUSION RATE`, `ADDITIONAL DOSES`, `INTERDOSE INTERVAL` and `STEADY
    STATE` are recognized as aliases (Monolix data format); the lookup of
    every column is case-insensitive.

    Args:
        df: the event table, one row per dose or observation
        time_unit: unit of the `TIME` column
        unit: unit of the `DV` column
        dose_unit: unit of the `AMT` column
        route: route of the doses; the event format has no route column
            (`CMT`/`ADM` are compartments, not routes) and a batch has one
            route, so a table of several routes is filtered by the caller
        id: name of the subject column
        time: name of the time column
        dv: name of the observation column
        amt: name of the dose amount column
        evid: name of the event identifier column
        mdv: name of the missing dependent value column
        rate: name of the infusion rate column
        tinf: name of the infusion duration column (Monolix)
        addl: name of the additional doses column
        ii: name of the interdose interval column
        ss: name of the steady state column
        ss_doses: number of preceding doses a steady state record stands for
        dim: name of the sample dimension of the batch
        substance: name of the substance or effect
        covariates: columns to keep as coordinates along `dim`; by default
            every column which is neither an event column nor a compartment or
            occasion column (`EVENT_IGNORED`) and which is constant within
            every subject, the columns which vary within a subject being
            logged and skipped (a column named here raises instead)

    Returns:
        The batch, the subjects in the order of their first appearance and
        their `ID` as the coordinate of `dim`.

    Raises:
        ValueError: if a required column (`id`, `time`, `dv`) is missing, if a
            rate is negative (a modelled rate), if a dose record asks for
            repeated doses without a positive `II`, if a requested covariate is
            not a column or not constant within a subject, or if the doses of
            the subjects do not share one unit.
    """
    df = df.reset_index(drop=True)
    lookup = _lookup(df)
    c_id = _column(lookup, id, "id", required=True)
    c_time = _column(lookup, time, "time", required=True)
    c_dv = _column(lookup, dv, "dv", required=True)
    c_amt = _column(lookup, amt, "amt")
    c_evid = _column(lookup, evid, "evid")
    c_mdv = _column(lookup, mdv, "mdv")
    c_rate = _column(lookup, rate, "rate")
    c_tinf = _column(lookup, tinf, "tinf")
    c_addl = _column(lookup, addl, "addl")
    c_ii = _column(lookup, ii, "ii")
    c_ss = _column(lookup, ss, "ss")
    assert c_id is not None and c_time is not None and c_dv is not None

    if c_rate is not None and (_numeric(df, c_rate) < 0).any():
        raise ValueError(
            "Modelled rates (RATE -1/-2) are not data: give the infusion "
            f"duration in '{tinf}' or a positive rate in '{rate}'"
        )

    if c_evid is not None:
        other = _numeric(df, c_evid).isin([2.0, 3.0])
        if bool(other.any()):
            logger.warning(
                "Dropped %d rows with EVID 2 or 3 (other type event, reset)",
                int(other.sum()),
            )
            df = df.loc[~other]

    times = _numeric(df, c_time)
    amounts = _numeric(df, c_amt, default=0.0)
    if c_evid is not None:
        evid_values = _numeric(df, c_evid, default=0.0)
        is_dose = evid_values.isin([1.0, 4.0])
        is_observation = evid_values.fillna(0.0) == 0.0
    else:
        is_dose = amounts > 0
        is_observation = pd.Series(True, index=df.index)
    if c_mdv is not None:
        is_observation &= _numeric(df, c_mdv, default=0.0).fillna(0.0) == 0.0
    values = _numeric(df, c_dv)
    is_observation &= values.notna()

    durations = pd.Series(np.nan, index=df.index, dtype=np.float64)
    if c_tinf is not None:
        infusion = _numeric(df, c_tinf)
        durations = durations.where(~(infusion > 0), infusion)
    if c_rate is not None:
        rates = _numeric(df, c_rate)
        from_rate = amounts / rates.where(rates > 0)
        durations = durations.where(durations.notna(), from_rate)

    addl_values = _numeric(df, c_addl, default=0.0)
    ii_values = _numeric(df, c_ii, default=np.nan)
    ss_values = _numeric(df, c_ss, default=0.0)

    subjects = _subjects(df, c_id)
    labels = list(subjects)
    steady_state_marker = False
    sample_times: list[np.ndarray] = []
    sample_values: list[np.ndarray] = []
    protocols: list[Dosing | None] = []
    for label, rows in subjects.items():
        dose_rows = [i for i in rows if bool(is_dose[i])]
        dose_times: list[float] = []
        dose_amounts: list[float] = []
        dose_durations: list[float] = []
        for i in dose_rows:
            steady_state = bool(ss_values[i] == 1)
            steady_state_marker |= steady_state
            expanded = _dose_times(
                float(times[i]),
                addl=float(addl_values[i]),
                interval=float(ii_values[i]),
                steady_state=steady_state,
                ss_doses=ss_doses,
                subject=label,
            )
            dose_times.extend(expanded)
            dose_amounts.extend([float(amounts[i])] * len(expanded))
            dose_durations.extend([float(durations[i])] * len(expanded))
        dosing = None
        if dose_times:
            dosing = Dosing(
                amounts=np.array(dose_amounts),
                times=np.array(dose_times),
                durations=np.array(dose_durations),
                unit=dose_unit,
                route=route,
            )
        protocols.append(dosing)
        observation_rows = [i for i in rows if bool(is_observation[i])]
        observed = times[observation_rows].to_numpy()
        order = np.argsort(observed, kind="stable")
        sample_times.append(observed[order])
        sample_values.append(values[observation_rows].to_numpy()[order])

    known = {
        column
        for column in (
            c_id,
            c_time,
            c_dv,
            c_amt,
            c_evid,
            c_mdv,
            c_rate,
            c_tinf,
            c_addl,
            c_ii,
            c_ss,
        )
        if column is not None
    }
    known |= {
        str(column)
        for column in df.columns
        if str(column).strip().upper() in EVENT_IGNORED
    }
    coordinates: dict[str, np.ndarray] = {}
    if covariates is None:
        for column in df.columns:
            name = str(column)
            if name in known:
                continue
            constant = _constant_per_subject(df, c_id, name, labels)
            if constant is None:
                logger.warning(
                    "column %s varies within a subject and is not a coordinate",
                    name,
                )
                continue
            coordinates[name] = constant
    else:
        for name in covariates:
            column = _column(lookup, name, required=True)
            assert column is not None
            constant = _constant_per_subject(df, c_id, column, labels)
            if constant is None:
                raise ValueError(
                    f"The covariate column '{column}' is not constant within "
                    "every subject"
                )
            coordinates[column] = constant

    batch = _build_batch(
        labels=labels,
        times=sample_times,
        values=sample_values,
        protocols=protocols,
        time_unit=time_unit,
        unit=unit,
        dim=dim,
        substance=substance,
        coordinates=coordinates,
    )
    if steady_state_marker:
        batch.ds.attrs["steady_state_marker"] = True
    return batch


def write_events(
    timecourses: Timecourses,
    *,
    id: str = "ID",
    time: str = "TIME",
    dv: str = "DV",
    amt: str = "AMT",
    evid: str = "EVID",
    mdv: str = "MDV",
    rate: str = "RATE",
) -> pd.DataFrame:
    """Write a batch as event records, the inverse of `read_events`.

    Every sample contributes one row per dose of its protocol (`EVID 1`,
    `MDV 1`, no `DV`, the amount in `AMT` and, for an infusion, the rate
    `AMT / duration` in `RATE`) and one row per observation (`EVID 0`,
    `AMT 0`, `RATE 0`, `MDV 0`, or `MDV 1` for a missing value). The rows of a
    sample are sorted by time, the doses before the observations at the same
    time; the samples keep the order of the batch. Repeated doses are written
    out (no `ADDL`/`II`/`SS`), so the table is read back by `read_events`
    without the expansion rules. The covariate coordinates along the sample
    dimension become columns after the event columns.

    A missing value is written as a row with `MDV 1`, which `read_events` does
    not read back as an observation: the round trip keeps the observed points
    and the protocol, not the missing points.

    Args:
        timecourses: the batch, with exactly one sample dimension
        id: name of the subject column
        time: name of the time column
        dv: name of the observation column
        amt: name of the dose amount column
        evid: name of the event identifier column
        mdv: name of the missing dependent value column
        rate: name of the infusion rate column

    Returns:
        The event table with the columns `id`, `time`, `dv`, `amt`, `evid`,
        `mdv`, `rate` and one column per covariate coordinate.

    Raises:
        ValueError: if the batch does not have exactly one sample dimension.
    """
    dims = timecourses.sample_dims
    if len(dims) != 1:
        raise ValueError(
            "Event records have one subject column: the batch needs exactly "
            f"one sample dimension, not {list(dims)}"
        )
    dim = dims[0]
    ds = timecourses.ds
    n_samples = timecourses.sample_shape[0]
    labels = (
        ds[dim].to_numpy() if dim in ds.coords else np.arange(n_samples, dtype=np.int64)
    )
    covariate_names = [
        str(name)
        for name in ds.coords
        if str(name) != dim and tuple(ds[name].dims) == (dim,)
    ]

    times = timecourses.times
    values = timecourses.values
    amounts = timecourses.dose_amount
    dose_times = timecourses.dose_time
    durations = timecourses.dose_duration

    rows: list[dict[str, Any]] = []
    for index in range(n_samples):
        shared: dict[str, Any] = {id: labels[index]}
        shared.update({name: ds[name].to_numpy()[index] for name in covariate_names})
        sample_rows: list[dict[str, Any]] = []
        if amounts is not None and dose_times is not None and durations is not None:
            for amount, dose_time, duration in zip(
                amounts[index], dose_times[index], durations[index], strict=True
            ):
                if not (np.isfinite(amount) and np.isfinite(dose_time)):
                    continue  # the padding of a shorter protocol
                sample_rows.append(
                    {
                        **shared,
                        time: float(dose_time),
                        dv: np.nan,
                        amt: float(amount),
                        evid: 1,
                        mdv: 1,
                        rate: (
                            float(amount) / float(duration)
                            if np.isfinite(duration) and duration > 0
                            else 0.0
                        ),
                    }
                )
        for t, value in zip(times[index], values[index], strict=True):
            if not np.isfinite(t):
                continue  # the padding of a shorter sampling grid
            sample_rows.append(
                {
                    **shared,
                    time: float(t),
                    dv: float(value),
                    amt: 0.0,
                    evid: 0,
                    mdv: 0 if np.isfinite(value) else 1,
                    rate: 0.0,
                }
            )
        sample_rows.sort(key=lambda row: (row[time], -row[evid]))
        rows.extend(sample_rows)

    columns = [id, time, dv, amt, evid, mdv, rate, *covariate_names]
    return pd.DataFrame(rows, columns=columns)


def read_pknca(
    conc: pd.DataFrame,
    dose: pd.DataFrame,
    *,
    time_unit: str,
    unit: str,
    dose_unit: str,
    route: Route,
    conc_col: str = "conc",
    time_col: str = "time",
    dose_col: str = "dose",
    dose_time_col: str = "time",
    subject: str = "subject",
    groups: Sequence[str] = (),
    duration_col: str | None = None,
    dim: str = "individual",
    substance: str = "substance",
) -> Timecourses:
    """Read a batch from the two tables of the R package `PKNCA`.

    The concentration table holds one row per subject and sampling time, the
    dose table one row per subject and dose; both are joined on the subject
    column (Denney et al. 2015). A subject without a row in the dose table gets
    no protocol. `PKNCA` codes a value below the limit of quantification as 0
    and a missing value as `NA`: both are kept as given (`NaN` for `NA`), the
    `lloq` and `blq` options of the NCA handle them.

    Args:
        conc: the concentration table
        dose: the dose table
        time_unit: unit of the time columns
        unit: unit of the concentration column
        dose_unit: unit of the dose column
        route: route of the doses
        conc_col: name of the concentration column
        time_col: name of the time column of `conc`
        dose_col: name of the dose amount column
        dose_time_col: name of the time column of `dose`, 0 when it is absent
        subject: name of the subject column of both tables
        groups: further grouping columns which are constant within a subject;
            they become coordinates along `dim`
        duration_col: name of the infusion duration column of `dose`, `None`
            without infusions
        dim: name of the sample dimension of the batch
        substance: name of the substance or effect

    Returns:
        The batch, the subjects in the order of their first appearance in
        `conc` and their subject label as the coordinate of `dim`.

    Raises:
        ValueError: if a required column is missing, if a grouping column is in
            neither table or is not constant within a subject, or if the
            concentration table holds no subject.
    """
    conc = conc.reset_index(drop=True)
    dose = dose.reset_index(drop=True)
    c_lookup = _lookup(conc)
    d_lookup = _lookup(dose)
    c_subject = _column(c_lookup, subject, required=True)
    c_time = _column(c_lookup, time_col, required=True)
    c_value = _column(c_lookup, conc_col, required=True)
    d_subject = _column(d_lookup, subject, required=True)
    d_amount = _column(d_lookup, dose_col, required=True)
    d_time = _column(d_lookup, dose_time_col)
    d_duration = _column(d_lookup, duration_col)
    assert c_subject is not None and c_time is not None and c_value is not None
    assert d_subject is not None and d_amount is not None

    subjects = _subjects(conc, c_subject)
    labels = list(subjects)
    dosed = _subjects(dose, d_subject)
    dose_times = _numeric(dose, d_time, default=0.0)
    dose_amounts = _numeric(dose, d_amount)
    dose_durations = _numeric(dose, d_duration)

    conc_times = _numeric(conc, c_time)
    conc_values = _numeric(conc, c_value)
    sample_times: list[np.ndarray] = []
    sample_values: list[np.ndarray] = []
    protocols: list[Dosing | None] = []
    for label, rows in subjects.items():
        times = conc_times[rows].to_numpy()
        order = np.argsort(times, kind="stable")
        sample_times.append(times[order])
        sample_values.append(conc_values[rows].to_numpy()[order])
        dose_rows = dosed.get(label, pd.Index([]))
        protocols.append(
            Dosing(
                amounts=dose_amounts[dose_rows].to_numpy(),
                times=dose_times[dose_rows].to_numpy(),
                durations=dose_durations[dose_rows].to_numpy(),
                unit=dose_unit,
                route=route,
            )
            if len(dose_rows) > 0
            else None
        )

    coordinates: dict[str, np.ndarray] = {}
    for name in groups:
        column = _column(c_lookup, name)
        table, table_subject = conc, c_subject
        if column is None:
            column = _column(d_lookup, name, required=True)
            table, table_subject = dose, d_subject
        assert column is not None
        constant = _constant_per_subject(table, table_subject, column, labels)
        if constant is None:
            raise ValueError(
                f"The grouping column '{column}' is not constant within every subject"
            )
        coordinates[column] = constant

    return _build_batch(
        labels=labels,
        times=sample_times,
        values=sample_values,
        protocols=protocols,
        time_unit=time_unit,
        unit=unit,
        dim=dim,
        substance=substance,
        coordinates=coordinates,
    )


def _adnca_unit(
    df: pd.DataFrame, given: str | None, column: str | None, what: str
) -> str:
    """The unit of the values or of the doses of an ADNCA dataset.

    Args:
        df: the rows of the analyte.
        given: the unit given by the caller, which wins over the table.
        column: name of the unit column of the table, `None` when it is absent.
        what: what the unit describes, for the error message.

    Returns:
        The unit string, as given or as spelled in the table.

    Raises:
        ValueError: if there is no unit at all, or the string is not a unit.
    """
    unit = given
    if unit is None and column is not None:
        values = df[column].dropna()
        if not values.empty:
            unit = str(values.iloc[0])
    if unit is None:
        raise ValueError(
            f"The unit of the {what} is neither given nor in the '{column}' column"
        )
    parse_unit(unit)
    return unit


def read_adnca(
    df: pd.DataFrame,
    *,
    time_unit: str = "hr",
    unit: str | None = None,
    dose_unit: str | None = None,
    route: Route | None = None,
    subject: str = "USUBJID",
    analyte: str | None = None,
    param: str = "PARAMCD",
    value: str = "AVAL",
    value_unit: str = "AVALU",
    time_first: str = "AFRLT",
    time_ref: str = "ARRLT",
    dose: str = "DOSEA",
    dose_unit_col: str = "DOSEU",
    route_col: str = "ROUTE",
    dtype: str = "DTYPE",
    lloq: str = "ALLOQ",
    dim: str = "individual",
    substance: str | None = None,
) -> Timecourses:
    """Read a batch from a CDISC ADaM ADNCA (ADPC) dataset.

    The dataset holds one row per concentration record of one analyte
    (`PARAMCD`), with the time since the first dose (`AFRLT`) and the time
    since the most recent dose (`ARRLT`) (CDISC ADNCA). The time of a record
    is `AFRLT`, so the dose times of a subject are the distinct values of
    `AFRLT - ARRLT` with the amount `DOSEA` of their records. Derived copies of
    a record (`DTYPE == "COPY"`, the pre-dose record duplicated into the
    previous interval) are dropped.

    Args:
        df: the ADNCA dataset
        time_unit: unit of the time columns
        unit: unit of the values, the first `AVALU` of the analyte by default
        dose_unit: unit of the doses, the first `DOSEU` by default
        route: route of the doses, the first `ROUTE` by default
        subject: name of the subject column
        analyte: the analyte to read, the single analyte of the dataset by
            default
        param: name of the parameter code column
        value: name of the value column
        value_unit: name of the unit column of the values
        time_first: name of the column with the time since the first dose
        time_ref: name of the column with the time since the reference dose
        dose: name of the dose amount column
        dose_unit_col: name of the unit column of the doses
        route_col: name of the route column
        dtype: name of the derivation type column
        lloq: name of the column with the lower limit of quantification; it
            becomes the coordinate `lloq` along `dim`
        dim: name of the sample dimension of the batch
        substance: name of the substance, the analyte by default

    Returns:
        The batch, the subjects in the order of their first appearance and
        their `USUBJID` as the coordinate of `dim`.

    Raises:
        ValueError: if a required column is missing, if `analyte` is `None` and
            the dataset holds several analytes, if a unit or a route cannot be
            read, or if the records of one dose time of a subject disagree on
            the dose amount.
    """
    df = df.reset_index(drop=True)
    lookup = _lookup(df)
    c_subject = _column(lookup, subject, required=True)
    c_param = _column(lookup, param, required=True)
    c_value = _column(lookup, value, required=True)
    c_first = _column(lookup, time_first, required=True)
    c_ref = _column(lookup, time_ref, required=True)
    c_dose = _column(lookup, dose, required=True)
    c_value_unit = _column(lookup, value_unit)
    c_dose_unit = _column(lookup, dose_unit_col)
    c_route = _column(lookup, route_col)
    c_dtype = _column(lookup, dtype)
    c_lloq = _column(lookup, lloq)
    assert c_subject is not None and c_param is not None and c_value is not None
    assert c_first is not None and c_ref is not None and c_dose is not None

    analytes = [str(a) for a in pd.unique(df[c_param].dropna())]
    if analyte is None:
        if len(analytes) != 1:
            raise ValueError(
                f"The dataset holds {len(analytes)} analytes in '{c_param}' "
                f"({sorted(analytes)}), give the 'analyte' to read"
            )
        analyte = analytes[0]
    rows = df.loc[df[c_param].astype(str) == analyte]
    if rows.empty:
        raise ValueError(f"No record of the analyte '{analyte}' in '{c_param}'")
    if c_dtype is not None:
        rows = rows.loc[rows[c_dtype].astype(str).str.upper() != "COPY"]

    values_unit = _adnca_unit(rows, unit, c_value_unit, "values")
    doses_unit = _adnca_unit(rows, dose_unit, c_dose_unit, "doses")
    if route is None:
        if c_route is None:
            raise ValueError(
                f"The route is neither given nor in a '{route_col}' column"
            )
        names = rows[c_route].dropna().astype(str).str.strip().str.upper().unique()
        if len(names) != 1:
            raise ValueError(
                f"A batch has one route, '{c_route}' holds {sorted(names)}; "
                "read the routes into separate batches"
            )
        if names[0] not in ADNCA_ROUTES:
            raise ValueError(
                f"Unknown route '{names[0]}', known are {sorted(ADNCA_ROUTES)}"
            )
        route = ADNCA_ROUTES[names[0]]

    times = _numeric(rows, c_first)
    reference = (times - _numeric(rows, c_ref)).round(6)
    amounts = _numeric(rows, c_dose)
    values = _numeric(rows, c_value)

    subjects = _subjects(rows, c_subject)
    labels = list(subjects)
    sample_times: list[np.ndarray] = []
    sample_values: list[np.ndarray] = []
    protocols: list[Dosing | None] = []
    for label, index in subjects.items():
        protocol: dict[float, float] = {}
        for i in index:
            dose_time, amount = float(reference[i]), float(amounts[i])
            if not (np.isfinite(dose_time) and np.isfinite(amount)):
                continue
            if protocol.get(dose_time, amount) != amount:
                raise ValueError(
                    f"Subject '{label}' has the dose amounts "
                    f"{protocol[dose_time]} and {amount} at the dose time "
                    f"{dose_time}"
                )
            protocol[dose_time] = amount
        dosing = None
        if protocol:
            dose_times = sorted(protocol)
            dosing = Dosing(
                amounts=np.array([protocol[t] for t in dose_times]),
                times=np.array(dose_times),
                unit=doses_unit,
                route=route,
            )
        protocols.append(dosing)
        observed = times[index].to_numpy()
        order = np.argsort(observed, kind="stable")
        sample_times.append(observed[order])
        sample_values.append(values[index].to_numpy()[order])

    coordinates: dict[str, np.ndarray] = {}
    if c_lloq is not None and bool(rows[c_lloq].notna().any()):
        constant = _constant_per_subject(rows, c_subject, c_lloq, labels)
        if constant is None:
            raise ValueError(
                f"The limit of quantification '{c_lloq}' is not constant within "
                "every subject"
            )
        coordinates["lloq"] = constant

    return _build_batch(
        labels=labels,
        times=sample_times,
        values=sample_values,
        protocols=protocols,
        time_unit=time_unit,
        unit=values_unit,
        dim=dim,
        substance=analyte if substance is None else substance,
        coordinates=coordinates,
    )
