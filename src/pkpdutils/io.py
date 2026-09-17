"""Exchange formats of pharmacokinetic data.

Readers and writers for the table formats the field exchanges timecourses and
dosing protocols in. Every reader takes a pandas `DataFrame` (the caller reads
the csv, sas or xpt file) and returns a `pkpdutils.timecourse.Timecourses`
batch with one sample dimension, the times as given (a reader never shifts the
time axis), one route and the dosing protocol of every subject; `analytes`
reads several analytes of a table into one batch with a second sample
dimension:

- **event records** (`read_events`, `write_events`): the one row per event
  format of NONMEM and Monolix, a row being a dose (`EVID 1`) or an
  observation (`EVID 0`). Repeated doses are given explicitly, as `ADDL`
  additional doses at the interdose interval `II`, or as a steady state dose
  (`SS 1`); see Bauer (2019) and the Monolix data format documentation.
- **PKNCA tables** (`read_pknca`): the two table layout of the R package
  `PKNCA`, the concentrations and the doses, joined on the subject and the
  grouping columns; see Denney et al. (2015).
- **CDISC ADNCA** (`read_adnca`): the analysis dataset of a non-compartmental
  analysis of the ADaM standard, one row per concentration record with the
  time since the first dose (`AFRLT`) and since the reference dose (`ARRLT`),
  see the CDISC ADaM ADNCA implementation guide (2021).

Every format is written back as well (`write_events`, `write_pknca`,
`write_adnca`), so a study round trips through any of them. The readers are
also reachable as the constructors `Timecourses.from_events`,
`Timecourses.from_pknca` and `Timecourses.from_adnca`, the writers as
`Timecourses.to_events`, `to_pknca` and `to_adnca`; the parameters of an
analysis are written as the CDISC `PP` domain by `pkpdutils.cdisc`.

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
are not data: both are out of scope. A reader reads one route; a study of
several routes is read into one batch per route, which
`Timecourses.from_timecourses` combines into one multi-route batch.

The references of the formats are the "Data formats" section of
`docs/references.md`.
"""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import ValidationError

# `dose_mapping` and `pad_rows` build the padded variables of a batch, shared
# with the constructors of `Timecourses` (the readers do not go through
# `Timecourse`: a subject of an exchange format may have no dose records at
# all, which a single curve does not allow)
from pkpdutils.timecourse import (
    LLOQ_VAR,
    ROUTE_VAR,
    SUBSTANCE_VAR,
    TIME_DIM,
    Dosing,
    Route,
    Timecourses,
    dose_mapping,
    pad_rows,
)
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


def _subject_codes(df: pd.DataFrame, column: str) -> tuple[np.ndarray, list[Any]]:
    """Number the rows of a table by subject, in the order of their first appearance.

    The positional counterpart of `_subjects`: the reader indexes its columns
    as numpy arrays, so the rows of a subject are a group of positions rather
    than an index of labels.

    Args:
        df: the table.
        column: name of the subject column.

    Returns:
        The subject of every row as an index into the labels (`-1` for a row
        without a subject) and the labels in the order of their first
        appearance.
    """
    codes, uniques = pd.factorize(df[column], use_na_sentinel=True)
    missing = int((codes < 0).sum())
    if missing:
        logger.warning("Dropped %d rows without a value in '%s'", missing, column)
    return codes, list(uniques)


def _grouped_rows(codes: np.ndarray, n_subjects: int) -> tuple[np.ndarray, np.ndarray]:
    """The row positions of every subject, grouped and in the order of the table.

    Args:
        codes: the subject of every row, `-1` for a row without one.
        n_subjects: number of subjects.

    Returns:
        The positions of the rows with a subject, ordered by subject and within
        a subject by the table, and the number of rows per subject.
    """
    known = codes >= 0
    positions = np.flatnonzero(known)
    order = np.argsort(codes[positions], kind="stable")
    counts = np.bincount(codes[positions], minlength=n_subjects)
    return positions[order], counts


def _split_by_subject(values: np.ndarray, counts: np.ndarray) -> list[np.ndarray]:
    """Cut an array whose entries are grouped by subject into one array per subject.

    Args:
        values: the entries, ordered by subject.
        counts: number of entries per subject.

    Returns:
        One array per subject, empty for a subject without entries.
    """
    if counts.size == 0:
        return []
    return np.split(values, np.cumsum(counts)[:-1])


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


def _subject_arrays(
    rows: pd.Index,
    time: pd.Series,
    columns: Sequence[pd.Series],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """The times of one subject and its other columns, sorted by time.

    Args:
        rows: the rows of the subject.
        time: the time column of the table.
        columns: the other columns to take, e.g. the values and the uncertainty.

    Returns:
        The times and one array per column, all in the order of the times.
    """
    times = time[rows].to_numpy(dtype=np.float64)
    order = np.argsort(times, kind="stable")
    return times[order], [
        column[rows].to_numpy(dtype=np.float64)[order] for column in columns
    ]


def _constant_n(
    values: np.ndarray,
    *,
    counts: np.ndarray,
    labels: Sequence[Any],
    column: str | None,
) -> np.ndarray:
    """The number of subjects of every group curve, one value per subject.

    Args:
        values: the column of the number of subjects, the rows grouped by
            subject.
        counts: number of rows per subject.

    Keyword Args:
        labels: the subjects, in the order of the batch.
        column: name of the column, for the error message.

    Returns:
        The value of every subject, `NaN` for a subject without one.

    Raises:
        ValueError: if a subject carries more than one value, named with the
            subject.
    """
    starts = np.cumsum(counts) - counts
    known = ~np.isnan(values)
    low = np.minimum.reduceat(np.where(known, values, np.inf), starts)
    high = np.maximum.reduceat(np.where(known, values, -np.inf), starts)
    varies = np.isfinite(low) & np.isfinite(high) & (low != high)
    if varies.any():
        i = int(np.argmax(varies))
        rows = values[starts[i] : starts[i] + counts[i]]
        found = pd.unique(rows[~np.isnan(rows)])
        raise ValueError(
            f"subject {labels[i]}: '{column}' is not constant, found {found}"
        )
    return np.where(np.isfinite(low), low, np.nan)


def _dosing(label: Any, **fields: Any) -> Dosing:
    """Build the dosing protocol of one subject, naming it in every error.

    Args:
        label: the subject, for the error message.
        **fields: the fields of `Dosing`.

    Returns:
        The protocol.

    Raises:
        ValueError: with the message of the first error of `Dosing`, prefixed
            by the subject; a reader never raises a pydantic `ValidationError`.
    """
    try:
        return Dosing(**fields)
    except ValidationError as err:
        message = str(err.errors()[0]["msg"]).removeprefix("Value error, ")
        raise ValueError(f"subject {label}: {message}") from err


def _check_times(label: Any, times: np.ndarray) -> None:
    """Check the sampling times of one subject.

    A subject of an exchange format needs the sampling times of a curve: at
    least two of them, finite and without duplicates. Both were only caught
    later, by `Timecourses.sel` or, for the duplicates, not at all (they skew
    `cmax` and add a zero width trapezoid to the AUC).

    Args:
        label: the subject, for the error message.
        times: the sampling times of the subject.

    Raises:
        ValueError: if the subject has fewer than two observations, a time
            which is not finite, or duplicate times.
    """
    if times.size < 2:
        raise ValueError(
            f"subject {label}: a timecourse needs at least 2 time points, "
            f"found {times.size}"
        )
    if not np.isfinite(times).all():
        raise ValueError(f"subject {label}: the sampling times must be finite")
    unique, counts = np.unique(times, return_counts=True)
    if (counts > 1).any():
        raise ValueError(
            f"subject {label}: duplicate sampling time {unique[counts > 1][0]:g}"
        )


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
    nominal_time: Sequence[np.ndarray] | None = None,
    sd: Sequence[np.ndarray] | None = None,
    se: Sequence[np.ndarray] | None = None,
    n: np.ndarray | None = None,
) -> Timecourses:
    """Assemble the batch of the subjects a reader has collected.

    The subjects of an exchange format may have different sampling grids,
    different numbers of doses and, unlike a single `Timecourse`, no dose
    records at all, so the batch is built from the padded arrays rather than
    from `Timecourses.from_timecourses`. A subject without a protocol gets a
    row of `NaN` dose entries.

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
        nominal_time: the nominal (planned) time of every observation of every
            subject, with the times; it becomes the variable `nominal_time`
            over `(dim, time)`, which the figures of a regulatory report use
            for the mean curves.
        sd: the standard deviations of every subject, with the times.
        se: the standard errors of every subject, with the times.
        n: the number of subjects of every group curve, of shape `(len(labels),)`.

    Returns:
        The batch, with the shared sampling grid as the `time` coordinate when
        every subject has the same one.

    Raises:
        ValueError: if there is no subject, if a subject has fewer than two
            observations, a time which is not finite or duplicate times, or if
            the protocols do not share one route and one dose unit.
    """
    if not labels:
        raise ValueError("The table holds no subject")
    for label, subject_times in zip(labels, times, strict=True):
        _check_times(label, subject_times)
    n_time = max(array.size for array in times)
    shared = all(
        array.size == times[0].size and np.array_equal(array, times[0])
        for array in times
    )
    grid: np.ndarray = times[0] if shared else pad_rows(times, n_time)
    dose, route = dose_mapping(protocols)

    batch = Timecourses.from_arrays(
        grid,
        pad_rows(values, n_time),
        time_unit=time_unit,
        unit=unit,
        dims=(dim,),
        coords={dim: list(labels)},
        sd=None if sd is None else pad_rows(sd, n_time),
        se=None if se is None else pad_rows(se, n_time),
        n=n,
        dose=dose,
        route=route,
        substance=substance,
    )
    ds = batch.ds
    if nominal_time is not None:
        ds = ds.assign(
            nominal_time=(
                (dim, TIME_DIM),
                pad_rows(nominal_time, n_time),
                {"units": time_unit},
            )
        )
    if coordinates:
        ds = ds.assign_coords({name: (dim, v) for name, v in coordinates.items()})
    return Timecourses(ds) if ds is not batch.ds else batch


#: name of the sample dimension of the analytes a reader was asked for
ANALYTE_DIM: str = "analyte"


def _stack_analytes(
    batches: Sequence[Timecourses],
    analytes: Sequence[str],
    *,
    substances: Sequence[str],
    dim: str = ANALYTE_DIM,
) -> Timecourses:
    """Stack the batches of several analytes along a new sample dimension.

    Every analyte is read on its own, as a table of one analyte is, and the
    batches are concatenated along `dim`; the substance of every analyte
    becomes the coordinate `substance` along it, which the analysis and
    `pkpdutils.nca.analytes.metabolite_ratio` read per sample. The subjects are
    aligned by their label, so an analyte which was not measured in a subject
    gives a row of `NaN` there, and the protocols are padded to the longest one
    of the batch.

    Args:
        batches: the batch of every analyte, in the order of `analytes`.
        analytes: the labels of the analytes, the coordinate of `dim`.

    Keyword Args:
        substances: the substance of every analyte.
        dim: name of the new sample dimension.

    Returns:
        The batch over `(dim, *sample_dims, time)`.

    Raises:
        ValueError: if the analytes were measured in different units or on
            different time units.
    """
    units = {(batch.unit, batch.time_unit) for batch in batches}
    if len(units) != 1:
        raise ValueError(
            "the analytes of a batch need the same units, found "
            f"{sorted(units)}; read them into separate batches"
        )
    datasets = []
    for batch in batches:
        ds = batch.ds.copy()
        # the substance travels as the coordinate along `dim` from here on
        ds.attrs.pop("substance", None)
        datasets.append(ds)
    combined = xr.concat(
        datasets,
        dim=pd.Index(list(analytes), name=dim),
        join="outer",
        combine_attrs="drop_conflicts",
    )
    combined = combined.assign_coords(
        {SUBSTANCE_VAR: (dim, np.array(list(substances), dtype=object))}
    )
    return Timecourses(combined)


def _read_analytes(
    analytes: Sequence[str],
    read: "Callable[[str, pd.DataFrame], Timecourses]",
    df: pd.DataFrame,
    *,
    column: str | None,
    dim: str,
    keep_unnamed: bool,
) -> Timecourses:
    """Read every analyte of a table on its own and stack the batches.

    Args:
        analytes: the values of `column` to read, one batch each.
        read: reads one analyte from the rows of the table it is given.
        df: the table.

    Keyword Args:
        column: the column which names the analyte of a row.
        dim: name of the sample dimension of the analytes.
        keep_unnamed: whether a row without a value in `column` belongs to
            every analyte, which a dose record of an event table does.

    Returns:
        The batch over `(dim, *sample_dims, time)`.

    Raises:
        ValueError: if `analytes` is empty or if no row names one of them.
    """
    if not analytes:
        raise ValueError("'analytes' is empty")
    assert column is not None
    named = df[column].astype("object")
    batches: list[Timecourses] = []
    for name in analytes:
        keep = named.astype(str).str.strip() == name
        if not bool(keep.any()):
            raise ValueError(
                f"No row of the analyte '{name}' in '{column}', found "
                f"{sorted({str(v) for v in named.dropna().unique()})}"
            )
        if keep_unnamed:
            keep = keep | named.isna()
        batches.append(read(name, df.loc[keep]))
    return _stack_analytes(
        batches, analytes, substances=[batch.substance for batch in batches], dim=dim
    )


def _expand_doses(
    *,
    time: np.ndarray,
    addl: np.ndarray,
    interval: np.ndarray,
    steady_state: np.ndarray,
    ss_doses: int,
    subjects: Sequence[Any],
    codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The times of the dose records of a table, expanded by `ADDL` and `SS`.

    `ADDL` gives `addl` further doses at `time + k * interval`; a steady state
    record (`SS == 1`) stands for a dosing history, represented by `ss_doses`
    preceding doses at `time - k * interval` (Bauer 2019). Both need a positive
    interdose interval `II`: a record which asks for repeated doses without one
    is an incomplete table, not a single dose, and raises. The doses of one
    record are therefore `time + k * interval` for `k` from `-ss_doses` (or 0)
    to `addl`, which the whole table is expanded with in one `numpy.repeat`.

    Args:
        time: time of every record.
        addl: number of additional doses per record, `NaN` or 0 for none.
        interval: interdose interval `II` per record, `NaN` or 0 for none.
        steady_state: whether the record is marked as a steady state dose.
        ss_doses: number of preceding doses a steady state record stands for.
        subjects: the subjects, in the order of the batch, for the error message.
        codes: the subject of every record, an index into `subjects`.

    Returns:
        The dose times, ascending per record, and the number of doses every
        record expands into.

    Raises:
        ValueError: if a record has `ADDL > 0` or `SS == 1` without a positive
            interdose interval; the first such record of the first such subject
            is named.
    """
    with np.errstate(invalid="ignore"):
        has_interval = np.isfinite(interval) & (interval > 0)
        additional = np.where(np.isfinite(addl) & (addl > 0), addl, 0.0).astype(
            np.int64
        )
    incomplete = (steady_state | (additional > 0)) & ~has_interval
    if incomplete.any():
        i = int(np.argmax(incomplete))
        raise ValueError(
            f"'ADDL'/'SS' need a positive 'II': subject '{subjects[codes[i]]}' has "
            f"the dose record at time {float(time[i])} with ADDL {float(addl[i])}, "
            f"SS {int(steady_state[i])} and II {float(interval[i])}"
        )
    first = np.where(steady_state, -ss_doses, 0)
    counts = additional - first + 1
    total = int(counts.sum())
    # `k` runs from `first` to `additional` within every record
    within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    k = within + np.repeat(first, counts)
    step = np.where(has_interval, interval, 0.0)
    return np.repeat(time, counts) + k * np.repeat(step, counts), counts


def read_events(
    df: pd.DataFrame,
    *,
    time_unit: str,
    unit: str,
    dose_unit: str,
    route: Route | str,
    id_col: str = "ID",
    time_col: str = "TIME",
    dv_col: str = "DV",
    amt_col: str = "AMT",
    evid_col: str = "EVID",
    mdv_col: str = "MDV",
    rate_col: str = "RATE",
    tinf_col: str = "TINF",
    addl_col: str = "ADDL",
    ii_col: str = "II",
    ss_col: str = "SS",
    sd_col: str = "SD",
    se_col: str = "SE",
    n_col: str = "N",
    ss_doses: int = 5,
    keep_missing: bool = True,
    dim: str = "individual",
    analyte_col: str | None = None,
    analytes: Sequence[str] | None = None,
    analyte_dim: str = ANALYTE_DIM,
    substance: str = "substance",
    covariates: Sequence[str] | None = None,
) -> Timecourses:
    """Read a batch from event records, the NONMEM and Monolix format.

    A row of the table is one event of one subject: a dose when `EVID` is 1, an
    observation when `EVID` is 0. An observation whose `DV` is missing or whose
    `MDV` is 1 is a missing value: with `keep_missing` it keeps its time and is
    read as `NaN` (the sampling grid of the table is the grid of the batch,
    which is what a table of values below the limit of quantification needs),
    without it the row is dropped. Rows with `EVID` 2 (other type event) or 3 (reset) are
    dropped and counted in a warning; `EVID` 4 (reset and dose) raises, since a
    reset starts a new period which the reader would silently merge into the
    protocol of the subject (Bauer 2019). A row with `EVID` 1 and a value in
    `DV` is a dose and an observation, which is how a table records a dose and
    a sample at the same time.

    Without an `EVID` column a row with `AMT > 0` is a dose and nothing else,
    the NM-TRAN semantics of a table without event identifiers; every other row
    with a value in `DV` is an observation. A `DV` on such a dose row is
    ignored and counted in a warning: a table which records a dose and a sample
    in one row needs an `EVID` column.

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
        route: route of the doses, a `Route` or a string it coerces
            (`"oral"`, `"IV_BOLUS"`); the event format has no route column
            (`CMT`/`ADM` are compartments, not routes) and a batch has one
            route, so a table of several routes is filtered by the caller
        id_col: name of the subject column
        time_col: name of the time column
        dv_col: name of the observation column
        amt_col: name of the dose amount column
        evid_col: name of the event identifier column
        mdv_col: name of the missing dependent value column
        rate_col: name of the infusion rate column
        tinf_col: name of the infusion duration column (Monolix)
        addl_col: name of the additional doses column
        ii_col: name of the interdose interval column
        ss_col: name of the steady state column
        sd_col: name of the standard deviation column of a group curve
        se_col: name of the standard error column of a group curve
        n_col: name of the column with the number of subjects of a group curve
            (constant within a subject)
        ss_doses: number of preceding doses a steady state record stands for
        keep_missing: whether a missing observation (`MDV` 1 or no `DV`) is
            read as a `NaN` value at its time instead of being dropped
        dim: name of the sample dimension of the batch
        analyte_col: name of the column which names the analyte of an
            observation (`DVID`, `YTYPE`, `CMT` or a column of the study),
            required with `analytes`
        analytes: the analytes to read into one batch, which gives the sample
            dimension `analyte_dim` and the coordinate `substance` along it; a
            row which names no analyte (a dose record) belongs to every one of
            them
        analyte_dim: name of the sample dimension of `analytes`
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
        ValueError: if a required column (`id_col`, `time_col`, `dv_col`) is missing, if a
            row carries `EVID` 4, if a row carries an `SS` value other than 0
            or 1, if a rate is negative (a modelled rate), if a dose record
            asks for repeated doses without a positive `II`, if a subject has
            fewer than two observations, duplicate sampling times or a dosing
            protocol which is not valid (named with the subject), if a
            requested covariate is not a column or not constant within a
            subject, or if the doses of the subjects do not share one unit.
    """
    if analytes is not None:
        return _read_analytes(
            [str(name) for name in analytes],
            lambda name, rows: read_events(
                rows,
                time_unit=time_unit,
                unit=unit,
                dose_unit=dose_unit,
                route=route,
                id_col=id_col,
                time_col=time_col,
                dv_col=dv_col,
                amt_col=amt_col,
                evid_col=evid_col,
                mdv_col=mdv_col,
                rate_col=rate_col,
                tinf_col=tinf_col,
                addl_col=addl_col,
                ii_col=ii_col,
                ss_col=ss_col,
                sd_col=sd_col,
                se_col=se_col,
                n_col=n_col,
                ss_doses=ss_doses,
                keep_missing=keep_missing,
                dim=dim,
                analyte_col=analyte_col,
                substance=name,
                covariates=covariates,
            ),
            df,
            column=_column(_lookup(df), analyte_col, required=True),
            dim=analyte_dim,
            keep_unnamed=True,
        )
    route = Route(route)
    df = df.reset_index(drop=True)
    lookup = _lookup(df)
    c_id = _column(lookup, id_col, "id", required=True)
    c_time = _column(lookup, time_col, "time", required=True)
    c_dv = _column(lookup, dv_col, "dv", required=True)
    c_amt = _column(lookup, amt_col, "amt")
    c_evid = _column(lookup, evid_col, "evid")
    c_mdv = _column(lookup, mdv_col, "mdv")
    c_rate = _column(lookup, rate_col, "rate")
    c_tinf = _column(lookup, tinf_col, "tinf")
    c_addl = _column(lookup, addl_col, "addl")
    c_ii = _column(lookup, ii_col, "ii")
    c_ss = _column(lookup, ss_col, "ss")
    c_sd = _column(lookup, sd_col)
    c_se = _column(lookup, se_col)
    c_n = _column(lookup, n_col)
    assert c_id is not None and c_time is not None and c_dv is not None

    if c_rate is not None and (_numeric(df, c_rate) < 0).any():
        raise ValueError(
            "Modelled rates (RATE -1/-2) are not data: give the infusion "
            f"duration in '{tinf_col}' or a positive rate in '{rate_col}'"
        )

    if c_evid is not None:
        evid_column = _numeric(df, c_evid)
        reset_dose = evid_column == 4.0
        if bool(reset_dose.any()):
            subject = df[c_id][reset_dose.idxmax()]
            raise ValueError(
                f"EVID 4 (reset and dose) at subject {subject}: split the "
                "periods into separate tables"
            )
        other = evid_column.isin([2.0, 3.0])
        if bool(other.any()):
            logger.warning(
                "Dropped %d rows with EVID 2 or 3 (other type event, reset)",
                int(other.sum()),
            )
            df = df.loc[~other]

    times = _numeric(df, c_time)
    amounts = _numeric(df, c_amt, default=0.0)
    values = _numeric(df, c_dv)
    if c_evid is not None:
        evid_values = _numeric(df, c_evid, default=0.0)
        is_dose = evid_values == 1.0
        is_observation = evid_values.fillna(0.0) == 0.0
    else:
        # without an `EVID` column a row with an amount is a dose and nothing
        # else (NM-TRAN): a `DV` on it is not an observation of the subject
        is_dose = amounts > 0
        is_observation = ~is_dose
        ignored = int((is_dose & values.notna()).sum())
        if ignored:
            logger.warning(
                "%d dose rows carry a DV value which is ignored (no EVID column)",
                ignored,
            )
    missing = values.isna()
    if c_mdv is not None:
        missing |= _numeric(df, c_mdv, default=0.0).fillna(0.0) != 0.0
    if keep_missing:
        # the row keeps its time; a `DV` on an `MDV 1` row is not observed
        values = values.where(~missing)
    else:
        is_observation &= ~missing

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
    # `SS` 2 (the dose is added to the steady state of the other doses) and the
    # remaining NONMEM codes describe a history this reader does not represent
    unknown_ss = ss_values.notna() & ~ss_values.isin([0.0, 1.0])
    if bool(unknown_ss.any()):
        row = unknown_ss.idxmax()
        raise ValueError(
            f"Unknown 'SS' value {ss_values[row]:g} at subject {df[c_id][row]}: "
            "only 0 (no steady state dose) and 1 (a dosing history of "
            "'ss_doses' doses) are read"
        )

    codes, labels = _subject_codes(df, c_id)
    rows_by_subject, rows_per_subject = _grouped_rows(codes, len(labels))

    # the doses: every dose record of the table is expanded at once, the rows
    # grouped by subject and in the order of the table within a subject
    dose_rows = rows_by_subject[is_dose.to_numpy(dtype=bool)[rows_by_subject]]
    dose_of_row = codes[dose_rows]
    dose_time = times.to_numpy()[dose_rows]
    steady = ss_values.to_numpy()[dose_rows] == 1.0
    steady_state_marker = bool(steady.any())
    expanded_times, expanded_counts = _expand_doses(
        time=dose_time,
        addl=addl_values.to_numpy()[dose_rows],
        interval=ii_values.to_numpy()[dose_rows],
        steady_state=steady,
        ss_doses=ss_doses,
        subjects=labels,
        codes=dose_of_row,
    )
    expanded_amounts = np.repeat(amounts.to_numpy()[dose_rows], expanded_counts)
    expanded_durations = np.repeat(durations.to_numpy()[dose_rows], expanded_counts)
    doses_per_subject = np.bincount(
        np.repeat(dose_of_row, expanded_counts), minlength=len(labels)
    )
    protocols: list[Dosing | None] = [
        None
        if count == 0
        else _dosing(
            label,
            amounts=subject_amounts,
            times=subject_times,
            durations=subject_durations,
            unit=dose_unit,
            route=route,
        )
        for label, count, subject_amounts, subject_times, subject_durations in zip(
            labels,
            doses_per_subject,
            _split_by_subject(expanded_amounts, doses_per_subject),
            _split_by_subject(expanded_times, doses_per_subject),
            _split_by_subject(expanded_durations, doses_per_subject),
            strict=True,
        )
    ]

    # the observations: sorted by time within every subject, which the stable
    # sort by time followed by the stable sort by subject leaves in the order
    # of the table for equal times
    observation_rows = rows_by_subject[
        is_observation.to_numpy(dtype=bool)[rows_by_subject]
    ]
    observed_times = times.to_numpy()[observation_rows]
    by_time = np.argsort(observed_times, kind="stable")
    observation_rows = observation_rows[
        by_time[np.argsort(codes[observation_rows][by_time], kind="stable")]
    ]
    observed = np.bincount(codes[observation_rows], minlength=len(labels))
    sample_times = _split_by_subject(times.to_numpy()[observation_rows], observed)
    sample_values = _split_by_subject(values.to_numpy()[observation_rows], observed)
    sample_sd = _split_by_subject(
        _numeric(df, c_sd).to_numpy()[observation_rows], observed
    )
    sample_se = _split_by_subject(
        _numeric(df, c_se).to_numpy()[observation_rows], observed
    )

    sample_n = _constant_n(
        _numeric(df, c_n).to_numpy()[rows_by_subject],
        counts=rows_per_subject,
        labels=labels,
        column=c_n,
    )

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
            c_sd,
            c_se,
            c_n,
        )
        if column is not None
    }
    known |= {
        str(column)
        for column in df.columns
        if str(column).strip().upper() in EVENT_IGNORED
    }
    analyte_column = _column(lookup, analyte_col)
    if analyte_column is not None:
        known.add(analyte_column)
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

    subject_counts = sample_n if c_n is not None else None
    if subject_counts is not None and not np.isfinite(subject_counts).all():
        logger.warning(
            "'%s' is missing for some subjects, the batch carries no 'n'", c_n
        )
        subject_counts = None

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
        sd=sample_sd if c_sd is not None else None,
        se=sample_se if c_se is not None else None,
        n=subject_counts,
    )
    if steady_state_marker:
        batch.ds.attrs["steady_state_marker"] = True
    return batch


def write_events(
    timecourses: Timecourses,
    *,
    id_col: str = "ID",
    time_col: str = "TIME",
    dv_col: str = "DV",
    amt_col: str = "AMT",
    evid_col: str = "EVID",
    mdv_col: str = "MDV",
    rate_col: str = "RATE",
    sd_col: str = "SD",
    se_col: str = "SE",
    n_col: str = "N",
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

    A missing value is written as a row with `MDV 1` and no `DV`, which
    `read_events` reads back as a missing value at its time: the round trip
    keeps the sampling grid, the observed points and the protocol.

    A group curve carries its uncertainty in the columns `sd_col`, `se_col` and
    `n_col`, written only when the batch has them: `sd` and `se` on the
    observation rows, the number of subjects `n` on every row of the sample.

    Args:
        timecourses: the batch, with exactly one sample dimension
        id_col: name of the subject column
        time_col: name of the time column
        dv_col: name of the observation column
        amt_col: name of the dose amount column
        evid_col: name of the event identifier column
        mdv_col: name of the missing dependent value column
        rate_col: name of the infusion rate column
        sd_col: name of the standard deviation column of a group curve
        se_col: name of the standard error column of a group curve
        n_col: name of the column with the number of subjects of a group curve

    Returns:
        The event table with the subject, time, observation, amount, event
        identifier, missing value and rate columns, the uncertainty columns of
        a group curve and one column per covariate coordinate.

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
    sd = timecourses.sd
    se = timecourses.se
    # the event format carries one number of subjects per subject; a batch
    # which counts every time point on its own (the group curve of a ragged
    # batch) is written with the number of subjects of the group
    if timecourses.n is not None and timecourses.ds["n"].dims != (dim,):
        logger.debug(
            "'%s' varies over the time points of a curve, the event records "
            "carry the number of subjects of every curve",
            n_col,
        )
    n = timecourses.n_subjects

    rows: list[dict[str, Any]] = []
    for index in range(n_samples):
        shared: dict[str, Any] = {id_col: labels[index]}
        if n is not None:
            shared[n_col] = float(n[index])
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
                        time_col: float(dose_time),
                        dv_col: np.nan,
                        amt_col: float(amount),
                        evid_col: 1,
                        mdv_col: 1,
                        rate_col: (
                            float(amount) / float(duration)
                            if np.isfinite(duration) and duration > 0
                            else 0.0
                        ),
                    }
                )
        for point, (t, value) in enumerate(
            zip(times[index], values[index], strict=True)
        ):
            if not np.isfinite(t):
                continue  # the padding of a shorter sampling grid
            observation = {
                **shared,
                time_col: float(t),
                dv_col: float(value),
                amt_col: 0.0,
                evid_col: 0,
                mdv_col: 0 if np.isfinite(value) else 1,
                rate_col: 0.0,
            }
            if sd is not None:
                observation[sd_col] = float(sd[index][point])
            if se is not None:
                observation[se_col] = float(se[index][point])
            sample_rows.append(observation)
        sample_rows.sort(key=lambda row: (row[time_col], -row[evid_col]))
        rows.extend(sample_rows)

    uncertainty = [
        column
        for column, present in (
            (sd_col, sd is not None),
            (se_col, se is not None),
            (n_col, n is not None),
        )
        if present
    ]
    columns = [
        id_col,
        time_col,
        dv_col,
        amt_col,
        evid_col,
        mdv_col,
        rate_col,
        *uncertainty,
        *covariate_names,
    ]
    return pd.DataFrame(rows, columns=columns)


def read_pknca(
    conc: pd.DataFrame,
    dose: pd.DataFrame,
    *,
    time_unit: str,
    unit: str,
    dose_unit: str,
    route: Route | str,
    conc_col: str = "conc",
    time_col: str = "time",
    dose_col: str = "dose",
    dose_time_col: str = "time",
    subject_col: str = "subject",
    duration_col: str | None = "duration",
    covariates: Sequence[str] = (),
    dim: str = "individual",
    analyte_col: str | None = None,
    analytes: Sequence[str] | None = None,
    analyte_dim: str = ANALYTE_DIM,
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
        route: route of the doses, a `Route` or a string it coerces
            (`"oral"`, `"IV_BOLUS"`)
        conc_col: name of the concentration column
        time_col: name of the time column of `conc`
        dose_col: name of the dose amount column
        dose_time_col: name of the time column of `dose`, 0 when it is absent
        subject_col: name of the subject column of both tables
        duration_col: name of the infusion duration column of `dose`, absent
            allowed (a table of another format carries none); `None` reads no
            duration. `write_pknca` writes it under this name
        covariates: further columns of either table which are constant within a
            subject; they become coordinates along `dim`
        dim: name of the sample dimension of the batch
        analyte_col: name of the column which names the analyte of a row of
            the concentration table (and of the dose table when it has one),
            required with `analytes`
        analytes: the analytes to read into one batch, which gives the sample
            dimension `analyte_dim` and the coordinate `substance` along it
        analyte_dim: name of the sample dimension of `analytes`
        substance: name of the substance or effect

    Returns:
        The batch, the subjects in the order of their first appearance in
        `conc` and their subject label as the coordinate of `dim`.

    Raises:
        ValueError: if a required column is missing, if a covariate column is
            in neither table or is not constant within a subject, if the
            concentration table holds no subject, if a subject has fewer than
            two concentrations or duplicate times, or if the dose rows of a
            subject are not a valid protocol (a dose time which is not a
            number, a negative amount, a missing infusion duration), named with
            the subject.
    """
    if analytes is not None:
        c_analyte = _column(_lookup(conc), analyte_col, required=True)
        d_analyte = _column(_lookup(dose), analyte_col)
        return _read_analytes(
            [str(name) for name in analytes],
            lambda name, rows: read_pknca(
                rows,
                (
                    dose
                    if d_analyte is None
                    else dose.loc[dose[d_analyte].astype(str).str.strip() == name]
                ),
                time_unit=time_unit,
                unit=unit,
                dose_unit=dose_unit,
                route=route,
                conc_col=conc_col,
                time_col=time_col,
                dose_col=dose_col,
                dose_time_col=dose_time_col,
                subject_col=subject_col,
                duration_col=duration_col,
                covariates=covariates,
                dim=dim,
                substance=name,
            ),
            conc,
            column=c_analyte,
            dim=analyte_dim,
            keep_unnamed=False,
        )
    route = Route(route)
    conc = conc.reset_index(drop=True)
    dose = dose.reset_index(drop=True)
    c_lookup = _lookup(conc)
    d_lookup = _lookup(dose)
    c_subject = _column(c_lookup, subject_col, required=True)
    c_time = _column(c_lookup, time_col, required=True)
    c_value = _column(c_lookup, conc_col, required=True)
    d_subject = _column(d_lookup, subject_col, required=True)
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
        times, columns = _subject_arrays(rows, conc_times, [conc_values])
        sample_times.append(times)
        sample_values.append(columns[0])
        dose_rows = dosed.get(label, pd.Index([]))
        protocols.append(
            _dosing(
                label,
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
    for name in covariates:
        column = _column(c_lookup, name)
        table, table_subject = conc, c_subject
        if column is None:
            column = _column(d_lookup, name, required=True)
            table, table_subject = dose, d_subject
        assert column is not None
        constant = _constant_per_subject(table, table_subject, column, labels)
        if constant is None:
            raise ValueError(
                f"The covariate column '{column}' is not constant within every subject"
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
    subject_col: str = "USUBJID",
    analyte: str | None = None,
    analytes: Sequence[str] | None = None,
    param_col: str = "PARAMCD",
    value_col: str = "AVAL",
    value_unit_col: str = "AVALU",
    time_first_col: str = "AFRLT",
    time_ref_col: str = "ARRLT",
    dose_col: str = "DOSEA",
    dose_unit_col: str = "DOSEU",
    duration_col: str | None = "ADUR",
    nominal_time_col: str | None = "NRRLT",
    route_col: str = "ROUTE",
    dtype_col: str = "DTYPE",
    lloq_col: str = "ALLOQ",
    dim: str = "individual",
    analyte_dim: str = ANALYTE_DIM,
    substance: str | None = None,
    covariates: Sequence[str] = (),
) -> Timecourses:
    """Read a batch from a CDISC ADaM ADNCA (ADPC) dataset.

    The dataset holds one row per concentration record of one analyte
    (`PARAMCD`), with the time since the first dose (`AFRLT`) and the time
    since the most recent dose (`ARRLT`) (CDISC ADNCA). The time of a record
    is `AFRLT`, so the dose times of a subject are the distinct values of
    `AFRLT - ARRLT` with the amount `DOSEA` of their records. Derived copies of
    a record (`DTYPE == "COPY"`, the pre-dose record duplicated into the
    previous interval) are dropped.

    The infusion duration is the `ADUR` of the records of a dose
    (`duration_col`), which not every dataset carries: without the column an
    infusion protocol cannot be read and a route of `Route.IV_INFUSION` raises
    (`Dosing` requires a positive duration for every dose), and such a study is
    read from the event records or from the PKNCA tables instead, which carry
    the duration or the rate.

    `analytes` reads several analytes of the dataset into one batch: every
    analyte is read on its own and the batches are stacked along the sample
    dimension `analyte_dim`, whose coordinate `substance` names the analyte of
    every row (`_stack_analytes`). The analysis then follows the substance of
    every sample and `pkpdutils.nca.analytes.metabolite_ratio` divides one by
    the other.

    Args:
        df: the ADNCA dataset
        time_unit: unit of the time columns
        unit: unit of the values, the first `AVALU` of the analyte by default
        dose_unit: unit of the doses, the first `DOSEU` by default
        route: route of the doses, a `Route` or a string it coerces, the
            first `ROUTE` by default
        subject_col: name of the subject column
        analyte: the analyte to read, the single analyte of the dataset by
            default
        analytes: the analytes to read into one batch, which gives the sample
            dimension `analyte_dim` and the coordinate `substance` along it;
            `None` reads the single analyte of `analyte`
        param_col: name of the parameter code column
        value_col: name of the value column
        value_unit_col: name of the unit column of the values
        time_first_col: name of the column with the time since the first dose
        time_ref_col: name of the column with the time since the reference dose
        dose_col: name of the dose amount column
        dose_unit_col: name of the unit column of the doses
        duration_col: name of the infusion duration column, absent in most
            datasets; the records of one dose must agree on it, `None` reads
            no duration
        nominal_time_col: name of the nominal (planned) time column, absent in
            many datasets; it becomes the variable `nominal_time` over
            `(dim, time)`, in the time frame the column itself uses (`NRRLT` is
            the nominal time within the dosing interval, `NFRLT` the one since
            the first dose, which is the frame of the observation times the
            reader writes); `None` reads no nominal time
        route_col: name of the route column
        dtype_col: name of the derivation type column
        lloq_col: name of the column with the lower limit of quantification; it
            becomes the coordinate `lloq` along `dim`
        dim: name of the sample dimension of the batch
        analyte_dim: name of the sample dimension of `analytes`
        substance: name of the substance, the analyte by default
        covariates: further columns which are constant within a subject; they
            become coordinates along `dim`

    Returns:
        The batch, the subjects in the order of their first appearance and
        their `USUBJID` as the coordinate of `dim`.

    Raises:
        ValueError: if a required column is missing, if both `analyte` and
            `analytes` are given, if both are `None` and the dataset holds
            several analytes, if a unit or a route cannot be read, if the route
            is `Route.IV_INFUSION` without a duration column (the error names
            the subject), if a subject has fewer than two records or duplicate
            times, if the records of one dose time of a subject disagree on the
            dose amount or on the duration, or if a covariate column is not in
            the dataset or not constant within a subject.
    """
    if analytes is not None:
        if analyte is not None:
            raise ValueError("give either 'analyte' or 'analytes', not both")
        names = [str(name) for name in analytes]
        if not names:
            raise ValueError("'analytes' is empty")
        batches = [
            read_adnca(
                df,
                time_unit=time_unit,
                unit=unit,
                dose_unit=dose_unit,
                route=route,
                subject_col=subject_col,
                analyte=name,
                param_col=param_col,
                value_col=value_col,
                value_unit_col=value_unit_col,
                time_first_col=time_first_col,
                time_ref_col=time_ref_col,
                dose_col=dose_col,
                dose_unit_col=dose_unit_col,
                duration_col=duration_col,
                nominal_time_col=nominal_time_col,
                route_col=route_col,
                dtype_col=dtype_col,
                lloq_col=lloq_col,
                dim=dim,
                covariates=covariates,
            )
            for name in names
        ]
        return _stack_analytes(
            batches,
            names,
            substances=[batch.substance for batch in batches],
            dim=analyte_dim,
        )
    route = None if route is None else Route(route)
    df = df.reset_index(drop=True)
    lookup = _lookup(df)
    c_subject = _column(lookup, subject_col, required=True)
    c_param = _column(lookup, param_col, required=True)
    c_value = _column(lookup, value_col, required=True)
    c_first = _column(lookup, time_first_col, required=True)
    c_ref = _column(lookup, time_ref_col, required=True)
    c_dose = _column(lookup, dose_col, required=True)
    c_value_unit = _column(lookup, value_unit_col)
    c_dose_unit = _column(lookup, dose_unit_col)
    c_route = _column(lookup, route_col)
    c_dtype = _column(lookup, dtype_col)
    c_lloq = _column(lookup, lloq_col)
    c_duration = _column(lookup, duration_col)
    c_nominal = _column(lookup, nominal_time_col)
    assert c_subject is not None and c_param is not None and c_value is not None
    assert c_first is not None and c_ref is not None and c_dose is not None

    present = [str(a) for a in pd.unique(df[c_param].dropna())]
    if analyte is None:
        if len(present) != 1:
            raise ValueError(
                f"The dataset holds {len(present)} analytes in '{c_param}' "
                f"({sorted(present)}), give the 'analyte' or the 'analytes' to read"
            )
        analyte = present[0]
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
    durations = _numeric(rows, c_duration)
    nominal = _numeric(rows, c_nominal)

    subjects = _subjects(rows, c_subject)
    labels = list(subjects)
    sample_times: list[np.ndarray] = []
    sample_values: list[np.ndarray] = []
    sample_nominal: list[np.ndarray] = []
    protocols: list[Dosing | None] = []
    for label, index in subjects.items():
        protocol: dict[float, float] = {}
        infusion: dict[float, float] = {}
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
            duration = float(durations[i])
            if np.isfinite(duration):
                if infusion.get(dose_time, duration) != duration:
                    raise ValueError(
                        f"Subject '{label}' has the infusion durations "
                        f"{infusion[dose_time]} and {duration} at the dose "
                        f"time {dose_time}"
                    )
                infusion[dose_time] = duration
        dosing = None
        if protocol:
            dose_times = sorted(protocol)
            dosing = _dosing(
                label,
                amounts=np.array([protocol[t] for t in dose_times]),
                times=np.array(dose_times),
                durations=np.array([infusion.get(t, np.nan) for t in dose_times]),
                unit=doses_unit,
                route=route,
            )
        protocols.append(dosing)
        observed, columns = _subject_arrays(index, times, [values, nominal])
        sample_times.append(observed)
        sample_values.append(columns[0])
        sample_nominal.append(columns[1])

    coordinates: dict[str, np.ndarray] = {}
    if c_lloq is not None and bool(rows[c_lloq].notna().any()):
        constant = _constant_per_subject(rows, c_subject, c_lloq, labels)
        if constant is None:
            raise ValueError(
                f"The limit of quantification '{c_lloq}' is not constant within "
                "every subject"
            )
        coordinates["lloq"] = constant
    for name in covariates:
        column = _column(lookup, name, required=True)
        assert column is not None
        constant = _constant_per_subject(rows, c_subject, column, labels)
        if constant is None:
            raise ValueError(
                f"The covariate column '{column}' is not constant within every subject"
            )
        coordinates[column] = constant

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
        nominal_time=None if c_nominal is None else sample_nominal,
    )


#: the `ROUTE` value `write_adnca` writes for every route, one of the values
#: `ADNCA_ROUTES` reads back
ADNCA_ROUTE_NAMES: dict[Route, str] = {
    Route.ORAL: "ORAL",
    Route.IV_BOLUS: "IV BOLUS",
    Route.IV_INFUSION: "IV INFUSION",
}


def _writer_dims(timecourses: Timecourses) -> tuple[str, str | None]:
    """The subject dimension and the analyte dimension of a batch to write.

    A batch of one sample dimension is one analyte; a batch of two is the
    subjects and the analytes, the analyte dimension being the one the
    `substance` coordinate lies along.

    Args:
        timecourses: the batch.

    Returns:
        The name of the subject dimension and of the analyte dimension, the
        latter `None` for a batch of one sample dimension.

    Raises:
        ValueError: if the batch has no sample dimension, more than two, or two
            without a `substance` coordinate along one of them.
    """
    dims = timecourses.sample_dims
    if len(dims) == 1:
        return dims[0], None
    if len(dims) == 2 and SUBSTANCE_VAR in timecourses.ds.coords:
        coord = timecourses.ds.coords[SUBSTANCE_VAR]
        analyte = [str(d) for d in coord.dims]
        if len(analyte) == 1 and analyte[0] in dims:
            return next(d for d in dims if d != analyte[0]), analyte[0]
    raise ValueError(
        "the table formats have one subject column: the batch needs exactly "
        f"one sample dimension, or two with a '{SUBSTANCE_VAR}' coordinate "
        f"along one of them, not {list(dims)}"
    )


def _sample_labels(timecourses: Timecourses, dim: str) -> np.ndarray:
    """The labels of one sample dimension, its positions when it has no coordinate.

    Args:
        timecourses: the batch.
        dim: name of the sample dimension.

    Returns:
        The labels.
    """
    ds = timecourses.ds
    if dim in ds.coords:
        return ds[dim].to_numpy()
    return np.arange(int(ds.sizes[dim]))


def _covariate_columns(
    timecourses: Timecourses, dim: str, skip: Sequence[str]
) -> list[str]:
    """The coordinates along one sample dimension a writer carries over as columns.

    Args:
        timecourses: the batch.
        dim: the sample dimension.
        skip: names never written as a covariate column.

    Returns:
        The names of the coordinates.
    """
    ds = timecourses.ds
    return [
        str(name)
        for name in ds.coords
        if str(name) != dim
        and str(name) not in skip
        and tuple(str(d) for d in ds[name].dims) == (dim,)
    ]


def write_pknca(
    timecourses: Timecourses,
    conc_path: str | Path | None = None,
    dose_path: str | Path | None = None,
    *,
    conc_col: str = "conc",
    time_col: str = "time",
    dose_col: str = "dose",
    dose_time_col: str = "time",
    subject_col: str = "subject",
    duration_col: str = "duration",
    analyte_col: str = "analyte",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write a batch as the two tables of `PKNCA`, the inverse of `read_pknca`.

    The concentration table holds one row per sample and observation (the
    subject, the time and the value, `NaN` for a missing one) and the dose
    table one row per sample and dose of its protocol (the subject, the dose
    time, the amount and, for an infusion, its duration). The coordinates along
    the subject dimension become further columns of the concentration table,
    which `read_pknca` reads back as `covariates`; a batch of several analytes
    (a `substance` coordinate along a second sample dimension) writes the
    analyte of every row into `analyte_col` in both tables, which `read_pknca`
    reads back as `analytes`.

    Args:
        timecourses: the batch, with one sample dimension or with an analyte
            dimension besides it
        conc_path: file to write the concentration table to, `None` to write
            no file
        dose_path: file to write the dose table to, `None` to write no file

    Keyword Args:
        conc_col: name of the concentration column
        time_col: name of the time column of the concentration table
        dose_col: name of the dose amount column
        dose_time_col: name of the time column of the dose table
        subject_col: name of the subject column of both tables
        duration_col: name of the infusion duration column, written only when
            the batch carries a duration
        analyte_col: name of the analyte column, written only for a batch of
            several analytes

    Returns:
        The concentration table and the dose table.

    Raises:
        ValueError: if the batch does not have the sample dimensions of a
            table (`_writer_dims`).
    """
    subject_dim, analyte_dim = _writer_dims(timecourses)
    ds = timecourses.ds
    subjects = _sample_labels(timecourses, subject_dim)
    substances = (
        None
        if analyte_dim is None
        else [str(value) for value in ds.coords[SUBSTANCE_VAR].to_numpy().ravel()]
    )
    covariates = _covariate_columns(
        timecourses, subject_dim, skip=[SUBSTANCE_VAR, ROUTE_VAR]
    )
    dims = timecourses.sample_dims
    times = timecourses.times
    values = timecourses.values
    amounts = timecourses.dose_amount
    dose_times = timecourses.dose_time
    durations = timecourses.dose_duration
    has_duration = durations is not None and bool(np.isfinite(durations).any())
    conc_rows: list[dict[str, Any]] = []
    dose_rows: list[dict[str, Any]] = []
    for index in np.ndindex(*timecourses.sample_shape):
        position = dict(zip(dims, index, strict=True))
        shared: dict[str, Any] = {subject_col: subjects[position[subject_dim]]}
        if analyte_dim is not None and substances is not None:
            shared[analyte_col] = substances[position[analyte_dim]]
        for name in covariates:
            shared[name] = ds[name].to_numpy()[position[subject_dim]]
        for t, value in zip(times[index], values[index], strict=True):
            if not np.isfinite(t):
                continue  # the padding of a shorter sampling grid
            conc_rows.append({**shared, time_col: float(t), conc_col: float(value)})
        if amounts is None or dose_times is None:
            continue
        for dose_index in range(amounts.shape[-1]):
            amount = float(amounts[index][dose_index])
            dose_time = float(dose_times[index][dose_index])
            if not (np.isfinite(amount) and np.isfinite(dose_time)):
                continue  # the padding of a shorter protocol
            row = {**shared, dose_time_col: dose_time, dose_col: amount}
            if has_duration and durations is not None:
                row[duration_col] = float(durations[index][dose_index])
            dose_rows.append(row)
    conc = pd.DataFrame(conc_rows)
    dose = pd.DataFrame(dose_rows)
    if conc_path is not None:
        conc.to_csv(conc_path, index=False)
    if dose_path is not None:
        dose.to_csv(dose_path, index=False)
    return conc, dose


def write_adnca(
    timecourses: Timecourses,
    path: str | Path | None = None,
    *,
    subject_col: str = "USUBJID",
    param_col: str = "PARAMCD",
    value_col: str = "AVAL",
    value_unit_col: str = "AVALU",
    time_first_col: str = "AFRLT",
    time_ref_col: str = "ARRLT",
    dose_col: str = "DOSEA",
    dose_unit_col: str = "DOSEU",
    duration_col: str = "ADUR",
    nominal_time_col: str = "NRRLT",
    route_col: str = "ROUTE",
    lloq_col: str = "ALLOQ",
) -> pd.DataFrame:
    """Write a batch as a CDISC ADaM ADNCA (ADPC) dataset, the inverse of `read_adnca`.

    One row per sample and observation: `AFRLT` the time of the record,
    `ARRLT` its time since the reference dose (the last dose at or before it,
    the first dose for a record before it) and `DOSEA` the amount of that dose,
    which is how `read_adnca` recovers the protocol of a subject. `ADUR` is the
    duration of the reference dose of an infusion, `ALLOQ` the limit of
    quantification of the subject and `PARAMCD` its analyte. The coordinates
    along the subject dimension become further columns, which `read_adnca`
    reads back as `covariates`. No row is a `DTYPE == "COPY"` duplicate.

    A dose which is not followed by an observation is not the reference dose of
    any record and is therefore not in the dataset, which is a property of the
    format rather than of the writer: the protocol of a subject lives in the
    concentration records.

    Args:
        timecourses: the batch, with one sample dimension or with an analyte
            dimension besides it
        path: file to write to, `None` to write no file

    Keyword Args:
        subject_col: name of the subject column
        param_col: name of the analyte column
        value_col: name of the value column
        value_unit_col: name of the unit column of the values
        time_first_col: name of the column with the time since the first dose
        time_ref_col: name of the column with the time since the reference dose
        dose_col: name of the dose amount column
        dose_unit_col: name of the unit column of the doses
        duration_col: name of the infusion duration column, written only when
            the batch carries a duration
        nominal_time_col: name of the nominal (planned) time column, written
            only when the batch carries the variable `nominal_time`
        route_col: name of the route column
        lloq_col: name of the column with the limit of quantification, written
            only when the batch carries one

    Returns:
        The dataset.

    Raises:
        ValueError: if the batch does not have the sample dimensions of a
            table (`_writer_dims`).
    """
    subject_dim, analyte_dim = _writer_dims(timecourses)
    ds = timecourses.ds
    subjects = _sample_labels(timecourses, subject_dim)
    substances = (
        None
        if analyte_dim is None
        else [str(value) for value in ds.coords[SUBSTANCE_VAR].to_numpy().ravel()]
    )
    covariates = _covariate_columns(
        timecourses, subject_dim, skip=[SUBSTANCE_VAR, ROUTE_VAR, LLOQ_VAR]
    )
    dims = timecourses.sample_dims
    times = timecourses.times
    values = timecourses.values
    amounts = timecourses.dose_amount
    dose_times = timecourses.dose_time
    durations = timecourses.dose_duration
    has_duration = durations is not None and bool(np.isfinite(durations).any())
    limits = timecourses.lloq
    nominal = (
        ds["nominal_time"].transpose(*dims, TIME_DIM).to_numpy()
        if "nominal_time" in ds.data_vars
        else None
    )
    routes = timecourses.routes
    one_route = None if routes is not None else timecourses.route
    rows: list[dict[str, Any]] = []
    for index in np.ndindex(*timecourses.sample_shape):
        position = dict(zip(dims, index, strict=True))
        row_route = one_route if routes is None else Route(routes[index])
        shared: dict[str, Any] = {
            subject_col: subjects[position[subject_dim]],
            param_col: (
                timecourses.substance
                if substances is None or analyte_dim is None
                else substances[position[analyte_dim]]
            ),
            value_unit_col: timecourses.unit,
            dose_unit_col: timecourses.dose_unit,
            route_col: (
                None if row_route is None else ADNCA_ROUTE_NAMES[Route(row_route)]
            ),
        }
        if limits is not None and np.isfinite(limits[index]):
            shared[lloq_col] = float(limits[index])
        for name in covariates:
            shared[name] = ds[name].to_numpy()[position[subject_dim]]
        protocol_times = (
            np.array([])
            if dose_times is None
            else np.asarray(dose_times[index], dtype=np.float64)
        )
        protocol_amounts = (
            np.array([])
            if amounts is None
            else np.asarray(amounts[index], dtype=np.float64)
        )
        protocol_durations = (
            np.array([])
            if durations is None
            else np.asarray(durations[index], dtype=np.float64)
        )
        given = np.isfinite(protocol_times) & np.isfinite(protocol_amounts)
        protocol_times = protocol_times[given]
        protocol_amounts = protocol_amounts[given]
        protocol_durations = (
            protocol_durations[given] if protocol_durations.size else protocol_durations
        )
        for point, (t, value) in enumerate(
            zip(times[index], values[index], strict=True)
        ):
            if not np.isfinite(t):
                continue  # the padding of a shorter sampling grid
            row = {**shared, time_first_col: float(t), value_col: float(value)}
            if nominal is not None:
                row[nominal_time_col] = float(nominal[index][point])
            if protocol_times.size:
                # the reference dose is the last one at or before the record,
                # the first dose for a record before every dose (ARRLT < 0)
                at_or_before = np.flatnonzero(protocol_times <= t)
                reference = int(at_or_before[-1]) if at_or_before.size else 0
                row[time_ref_col] = float(t - protocol_times[reference])
                row[dose_col] = float(protocol_amounts[reference])
                if has_duration and protocol_durations.size:
                    row[duration_col] = float(protocol_durations[reference])
            rows.append(row)
    frame = pd.DataFrame(rows)
    if path is not None:
        frame.to_csv(path, index=False)
    return frame
