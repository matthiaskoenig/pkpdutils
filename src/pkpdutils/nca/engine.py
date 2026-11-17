"""The rust kernel of the non-compartmental analysis.

`pkpdutils-core` carries the vectorized core of the analysis as compiled code.
This module is the boundary between the two: it decides whether a batch can go
there at all (`unsupported_reason`), lays the batch and the options out as the
dicts of the abi (`batch_dict`, `options_dict`) and runs them (`run_kernel`),
giving back the same mapping of one array per variable that
`pkpdutils.nca.nca.run_rows` gives back, so that `nca` assembles the result of
either engine with the same code.

The dependency is hard: `pkpdutils-core` is installed with `pkpdutils` and
this module imports it at import time. What is *not* hard is the analysis a
build of it carries, and that is the asymmetry the whole module is built
around:

- `pkpdutils_core.is_supported` and the `pkpdutils_core.UnsupportedError` of
  `pkpdutils_core.analyze_batch` are the **authority** on what the kernel
  analyses. `unsupported_reason` forwards to the first of them and holds no
  refusal of its own, so a kernel release which changes one needs no change
  here; the three conditions it does answer itself are about the python
  objects and are no refusal of the kernel (see its docstring).
- A batch which only `analyze_batch` refuses (a unit the registry of the
  kernel cannot read) raises `UnsupportedError` and `nca` falls back, so the
  reason is the kernel's own sentence, word for word, either way, and
  `tests/nca/test_engine.py` asserts the two agree on every refusal rather
  than asserting a text of its own.

The two engines report the same variables with the same numbers, and one
analysis in a different order: a batch of several routes. The python core runs
one group per route and takes the union of their variables in the order the
routes appear in the batch (`pkpdutils.nca.nca.merge_rows`), where the kernel
reports the union in its canonical order. Nothing of the result but the order
of its variables differs, and the order of the kernel does not depend on which
route the batch names first.

Because the two engines are held to one answer rather than trusted to agree,
this module also carries the comparison itself: `compare_engines` is every way
in which two mappings of one analysis can disagree and `compare_with_python`
is the gate `pkpdutils.nca.nca.nca` runs it from, armed by the environment
variable `COMPARE_VARIABLE`. The gate compares the answer of a batch which
reached the kernel, so by construction it cannot see the other direction, a
batch which quietly never got there; that one is covered by the cross-check of
`tests/conftest.py`, which asks both predicates about every batch of the test
suite. The two are complements, not alternatives.

What the gate adds over the reference comparisons of the suite is the argument
for having it, so it is named here rather than left to a judgement.
`tests/nca/test_benchmark_datasets.py` and `tests/nca/test_validation.py` hold
the main parameters of theophylline and indomethacin against WinNonlin, PKNCA
and NonCompart within 1e-8, single dose, bolus, infusion and extravascular:
dense cover of those variables on those batches, and the measure of it is that
a mutation of `aumc_inf_pred` by 1e-5 for every batch of three samples or more
fails 25 tests with the gate disarmed. The gate compares **every** batch the
caller builds, and these are the parts of an answer no reference table of the
suite reaches, the multiple dose analysis among them (the `interval_*` point
variables, the steady state, `n_doses`, `tau`, the flags `INCOMPLETE_INTERVAL`
and `EXTRAPOLATED_TROUGH`, and a last interval completed within `tau_tolerance`):

- `flags` and `c0_method` as exact integers, which no table of another tool
  carries and which a tolerance would not compare anyway (`_compare_columns`),
- the point variables of a kept candidate table (`candidate_t_first`,
  `candidate_n_points`, `candidate_r2_adj`), which are no column of a
  parameter table,
- a limit of quantification and a terminal window given by hand, neither of
  which the benchmark scenarios exercise,
- a batch of several routes and the order of its variables, which is the order
  of the columns of `NCAResult.to_dataframe` and of every rendered table,
- the positions of `NaN`, the dtype of every column and the unit string the
  kernel reports for every variable, which the host neither adopts nor drops.

What the gate does **not** reach is every analysis the dispatch refuses, since
a batch which never goes to the kernel is held against nothing: the
propagation of the uncertainty of a group curve, the acceptance criteria, the
named partial areas, the effect curve and the unit conversion of a result
(`unsupported_reason`, the table of `docs/benchmark.md`). The urine analysis is out of reach for another
reason: `pkpdutils.nca.urine.nca_urine` calls `compute_parameters` of the
python core directly and never passes through `nca`, so `rate` and `midpoint`
are no answer of an engine at all. Every one of those is held to the python
core alone, by the reference tests of the suite, and the day the kernel carries
one of them the gate compares it without a word being changed here.

Where a reference table does cover a variable, the gate's value is breadth and
diagnosis rather than a sensitivity the suite lacks: of that `aumc_inf_pred`
mutation the armed run says which variable, which batch and both numbers in
its first failure, where the disarmed run says that 25 published numbers no
longer match.

`UnsupportedError` is the one exception a caller answers by taking its own
path, and it is re-exported here so that `nca` catches it without importing
the kernel itself. Every other exception of the bindings is a disagreement
about the contract (a `KeyError` naming a key of the batch or of the options, a
`TypeError`, a `ValueError` the kernel refused the batch with) and must
propagate: the day `NCAOptions` gains a field without a release of
`pkpdutils-core`, that `KeyError` is what says so, and a `except ValueError`
anywhere on this path would turn it into every batch silently taking the python
path for good.
"""

import logging
import os
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

import numpy as np
import xarray as xr

from pkpdutils.nca.options import (
    SCALAR_KEY_TYPES,
    TEXT_KEY_TYPES,
    WINDOW_DEFAULT_KEY,
    NCAOptions,
    encode_window_key,
)
from pkpdutils.nca.result import parameter_unit
from pkpdutils.result import sample_coordinates
from pkpdutils.timecourse import Route, Timecourses

try:
    import pkpdutils_core
except ImportError as missing:
    raise ImportError(
        "the rust kernel `pkpdutils-core` is required by `pkpdutils` and is "
        "not importable; install it with `pip install pkpdutils-core` or "
        "reinstall `pkpdutils`, which depends on it"
    ) from missing

logger = logging.getLogger(__name__)

# the wheel of `pkpdutils-core` (0.2.0 to 0.3.0) carries no stub beside its compiled
# module, so a type checker sees none of its three names and every lookup of
# one is suppressed by name
UnsupportedError = pkpdutils_core.UnsupportedError  # ty: ignore[missing-attribute]
"""The refusal of an analysis this build of the kernel does not carry.

A subclass of `ValueError`. It is raised for exactly the two outcomes of
"this build does not support this analysis", the refusal of
`pkpdutils_core.is_supported` and a refusal the kernel itself reports, and for
nothing else, so it is the one exception a caller answers by taking the python
path (`pkpdutils.nca.nca.nca`).
"""

_force_python: bool = False
"""Whether `unsupported_reason` refuses every batch.

For the tests and the benchmark only, which run the two engines against each
other in one process and need the python path of a batch the kernel carries.
It is no part of the public API and no environment variable.
"""

FORCED_REASON: str = "the python path is forced (`pkpdutils.nca.engine._force_python`)"
"""The reason `unsupported_reason` gives while `_force_python` is set."""

EMPTY_REASON: str = "a batch of no samples, whose variables the kernel does not report"
"""The reason `unsupported_reason` gives for a batch without a single sample.

Together with `FORCED_REASON` the only reason of this module which is not the
kernel's own, and the only one of the two about a real batch. `pkpdutils-core`
(0.2.0 to 0.3.0) derives the variables of its result from the routes of the rows it ran,
so a batch of no rows comes back with one column instead of with the empty
column of every variable python reports (`docs/carry-forward.md` of the kernel
names the fix, which belongs on its side). The host therefore answers an empty
batch itself, which costs nothing: there is no row to analyse.
"""

MIXED_LABELS_REASON: str = (
    "a terminal window which names a sample of a batch whose labels do not "
    "share one type, which one type per label column cannot carry"
)
"""The reason `unsupported_reason` gives for a batch of labels of several types.

The third reason of this module which is not the kernel's own and the second
one about a real batch. A coordinate travels under one type for the whole
column where a window key travels under its own, so a batch whose labels are
of several types and whose options name one of them by hand would be refused
by the kernel with a plain `ValueError` - no `UnsupportedError`, so no
fallback - where the python core matches the label and answers. The host
therefore refuses it itself, which routes it to the python core, and the
refusal is the narrowest one which covers the defect (`mixed_label_dim`,
`_names_a_sample`): a batch of such labels without a window given by hand is
analysed by the kernel as before, since nothing of its labels is ever read.
"""

HOST_VARIABLES: frozenset[str] = frozenset({"n"})
"""The variables of a result which the host writes itself, not the kernel.

`nca` writes the number of subjects behind every sample from
`Timecourses.n_subjects` after the rows are run, so `run_rows` does not report
it and `run_kernel` drops the column of the kernel rather than reporting it one
step early, which would place it in front of the uncertainty variables of the
result instead of behind them. The kernel's own column is compared against the
one of the host rather than adopted.
"""


class KernelInput:
    """A batch and its options laid out once as the dicts of the abi.

    `unsupported_reason` and `run_kernel` both need `batch_dict` and
    `options_dict`, and the conversion copies the whole batch, so `nca` builds
    one `KernelInput` and hands it to both rather than converting twice. The
    dicts are built on first use, so an analysis the host answers itself
    (`FORCED_REASON`, `EMPTY_REASON`, `MIXED_LABELS_REASON`) converts nothing.
    """

    def __init__(self, timecourses: Timecourses, options: NCAOptions) -> None:
        """Hold the batch and the options, converting nothing yet.

        Args:
            timecourses: the batch
            options: the options of the analysis
        """
        self.timecourses = timecourses
        self.options = options
        self._dicts: tuple[dict[str, Any], dict[str, Any]] | None = None

    @property
    def dicts(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """The batch and the options as the two dicts of the abi, built once."""
        if self._dicts is None:
            self._dicts = (
                batch_dict(self.timecourses),
                options_dict(self.options),
            )
        return self._dicts


def unsupported_reason(
    timecourses: Timecourses,
    options: NCAOptions,
    *,
    prepared: KernelInput | None = None,
) -> str | None:
    """Why the kernel would refuse this analysis, `None` when it would run it.

    The kernel decides, the host forwards: every refusal of the kernel (an
    effect curve, the acceptance criteria, the named partial areas, the
    propagation of the uncertainty of a group curve, the unit conversion of a
    result, and whatever a later release adds) is the answer of
    `pkpdutils_core.is_supported`, text and order included, so that a release
    which changes a refusal changes it here without a change of the host.

    The host answers before it asks only what the kernel cannot see, which is
    a statement about the python objects and no refusal of the kernel's, and
    each of them is named rather than written here (`FORCED_REASON`,
    `EMPTY_REASON`, `MIXED_LABELS_REASON`). `is_supported` itself accepts all
    three: it says nothing about the python path being pinned, and it answers
    `None` for an empty batch and for a batch of mixed labels, whose analysis
    the kernel then answers with one column or rejects with a plain
    `ValueError`. One refusal of the kernel is invisible to `is_supported`
    too, a unit string its registry cannot read, and it is raised by
    `run_kernel` as an `UnsupportedError`, which `nca` answers with the
    fallback.

    Args:
        timecourses: the batch
        options: the options of the analysis

    Keyword Args:
        prepared: the conversion of this batch and these options which
            `run_kernel` is given as well, so that it is made once; built here
            when `None`

    Returns:
        The reason, or `None` for an analysis the kernel carries.
    """
    if _force_python:
        return FORCED_REASON
    if timecourses.n_samples == 0:
        return EMPTY_REASON
    if (
        _names_a_sample(options.terminal.windows)
        and mixed_label_dim(timecourses) is not None
    ):
        return MIXED_LABELS_REASON
    is_supported = pkpdutils_core.is_supported  # ty: ignore[missing-attribute]
    batch, kernel_options = (prepared or KernelInput(timecourses, options)).dicts
    reason: str | None = is_supported(batch, kernel_options)
    return reason


def dose_arrays(
    timecourses: Timecourses,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The dosing protocol of every row as the three `(N, max_doses)` arrays of the abi.

    A batch without doses has `max_doses == 0` and three empty arrays rather
    than the `None` of `run_rows`, since the abi always carries the three
    fields. A batch which gives amounts without times is dosed at time 0, as
    `run_rows` reads it, and a dose without a duration is no infusion and is
    `NaN`.

    Args:
        timecourses: the batch

    Returns:
        The amounts, the times and the durations, `NaN` padded at the back of
        every row.
    """
    n_rows = timecourses.n_samples
    amounts = timecourses.dose_amount
    if amounts is None:
        empty = np.zeros((n_rows, 0))
        return empty, empty.copy(), empty.copy()
    n_dose = timecourses.n_dose
    times = timecourses.dose_time
    durations = timecourses.dose_duration
    return (
        _rows(amounts, n_rows, n_dose),
        np.zeros((n_rows, n_dose)) if times is None else _rows(times, n_rows, n_dose),
        np.full((n_rows, n_dose), np.nan)
        if durations is None
        else _rows(durations, n_rows, n_dose),
    )


def _rows(values: np.ndarray, n_rows: int, width: int) -> np.ndarray:
    """One row per sample as a C-contiguous array of doubles.

    Args:
        values: the array of shape `(*sample_shape, width)`
        n_rows: number of samples, the product of the sample shape
        width: number of entries per sample

    Returns:
        The array of shape `(n_rows, width)`.
    """
    return np.ascontiguousarray(values, dtype=np.float64).reshape(n_rows, width)


def batch_dict(timecourses: Timecourses) -> dict[str, Any]:
    """The batch as the dict of `batch/1`, one key per field of the abi `Batch`.

    The arrays travel as C-contiguous doubles of shape `(n_samples,
    n_time)`, the dose arrays as `(n_samples, max_doses)` and `lloq` as
    `(n_samples,)`, every one of them in the row order of the analysis. The
    enumerations travel as the strings both sides spell them with and the
    coordinates as records which carry their dtype (`_coordinate`), since the
    kernel labels the samples of a batch with them
    (`TerminalPhase.windows`).

    `contract` is no key of it: the bindings speak one contract and write it
    themselves.

    Args:
        timecourses: the batch

    Returns:
        The dict `pkpdutils_core.analyze_batch` and
        `pkpdutils_core.is_supported` take.
    """
    n_rows = timecourses.n_samples
    n_time = timecourses.n_time
    amounts, times, durations = dose_arrays(timecourses)
    lloq = timecourses.lloq
    nominal_time = timecourses.nominal_times
    counts = timecourses.n
    substances = timecourses.substances
    routes = timecourses.routes
    return {
        "sample_dims": list(timecourses.sample_dims),
        "sample_shape": list(timecourses.sample_shape),
        "n_time": n_time,
        "times": _rows(timecourses.times, n_rows, n_time),
        "values": _rows(timecourses.values, n_rows, n_time),
        "sd": _optional_points(timecourses.sd, n_rows, n_time),
        "se": _optional_points(timecourses.se, n_rows, n_time),
        # the counts of a batch which keeps one per sample instead of one per
        # time point are no field of `batch/1`, which has the second form
        # only; `nca` writes `n` from `Timecourses.n_subjects` itself either way
        "n": _optional_points(counts, n_rows, n_time),
        "nominal_time": _optional_points(nominal_time, n_rows, n_time),
        "lloq": None
        if lloq is None
        else np.ascontiguousarray(lloq, dtype=np.float64).reshape(n_rows),
        "max_doses": amounts.shape[1],
        "dose_amount": amounts,
        "dose_time": times,
        "dose_duration": durations,
        "time_unit": timecourses.time_unit,
        "unit": timecourses.unit,
        "dose_unit": timecourses.dose_unit,
        "substance": None if substances is not None else timecourses.substance,
        "substances": None
        if substances is None
        else [str(value) for value in substances.ravel().tolist()],
        "route": None if routes is not None else _route_name(timecourses.route),
        "routes": None
        if routes is None
        else [Route(value).value for value in routes.ravel().tolist()],
        "tissue": timecourses.tissue,
        "coords": [
            _coordinate(name, coord, timecourses.sample_dims)
            for name, coord in sample_coordinates(
                timecourses.ds, timecourses.sample_dims
            ).items()
        ],
    }


def _optional_points(
    values: np.ndarray | None, n_rows: int, n_time: int
) -> np.ndarray | None:
    """One value per sample and time point, `None` for a variable the batch lacks.

    A variable the batch carries per sample instead of per time point is left
    out: `batch/1` has the per-point form only and nothing the kernel analyses
    reads the per-sample one.

    Args:
        values: the array of shape `(*sample_shape, n_time)`, or `None`
        n_rows: number of samples
        n_time: number of time points per sample

    Returns:
        The array of shape `(n_rows, n_time)`, or `None`.
    """
    if values is None or values.size != n_rows * n_time:
        return None
    return _rows(values, n_rows, n_time)


def _route_name(route: Route | None) -> str | None:
    """The name `batch/1` spells a route with, `None` for a batch without doses.

    Args:
        route: the route of the batch.

    Returns:
        The value of the member, which is the name of the abi `Route`.
    """
    return None if route is None else Route(route).value


def _coordinate(
    name: str, coord: xr.DataArray, sample_dims: tuple[str, ...]
) -> dict[str, Any]:
    """One sample coordinate as the record of `batch/1`.

    Args:
        name: the name of the coordinate
        coord: the coordinate, on sample dimensions only
        sample_dims: the sample dimensions in the order the samples are
            enumerated

    Returns:
        Its name, the sample dimensions it lives on in their order and its
        values row major over them, under their dtype.
    """
    dims = tuple(dim for dim in sample_dims if dim in coord.dims)
    values = (coord.transpose(*dims) if dims else coord).to_numpy()
    return {"name": name, "dims": list(dims), "values": _coordinate_values(values)}


def _coordinate_values(values: np.ndarray) -> dict[str, list[Any]]:
    """The values of a coordinate under the key of their dtype.

    The dtype travels because the kernel labels the samples of a batch with
    the coordinates of its dimensions and matches those labels against the
    keys of `TerminalPhase.windows`, which `encode_window_key` tags the same
    way: a subject labelled `"1"` must not become the number 1 on the way
    over. An object array is read like a key of the windows, by the type of
    its values, with a numpy scalar unwrapped first, and anything which is no
    number, no truth value and no string travels as its text, which is how
    the kernel compares a date to a label of a batch.

    Args:
        values: the values of the coordinate

    Returns:
        The one entry `{"str" | "int" | "float" | "bool": values}` of
        `Coordinate`.
    """
    items = values.ravel().tolist()
    if values.dtype.kind == "b":
        return {"bool": [bool(value) for value in items]}
    if values.dtype.kind in "iu":
        return {"int": [int(value) for value in items]}
    if values.dtype.kind == "f":
        return {"float": [float(value) for value in items]}
    if values.dtype.kind == "O":
        items = _unwrapped(items)
        tagged = _object_tag(items)
        if tagged is not None:
            tag, cls = tagged
            return {tag: [cls(value) for value in items]}
    return {"str": [_label_text(value) for value in items]}


def _unwrapped(items: list[Any]) -> list[Any]:
    """The values of an object column as the python objects they hold.

    A numpy scalar is no instance of the python type it holds, so it is
    unwrapped before it is read, exactly as `encode_window_key` unwraps the key
    of a terminal window: an object array of `np.int64` is what `xarray` hands
    over for a coordinate of integers it was given as an object array, and it
    has to read as the integers it holds and not as their text.

    Args:
        items: the values of the column.

    Returns:
        The values, a numpy scalar replaced by the object it holds.
    """
    return [value.item() if isinstance(value, np.generic) else value for value in items]


def _object_tag(items: list[Any]) -> tuple[str, type] | None:
    """The one type every value of an object column shares, with its tag.

    The types are read in the order of `SCALAR_KEY_TYPES`, which
    `encode_window_key` reads the key of a terminal window with, so a column
    and a key which names one of its values cannot be tagged by two different
    rules.

    Args:
        items: the values of the column, numpy scalars unwrapped
            (`_unwrapped`).

    Returns:
        The tag of `Coordinate` and the type to convert with, or `None` for an
        empty column and for one whose values share no type of the table.
    """
    for tag, cls in SCALAR_KEY_TYPES:
        if items and all(isinstance(value, cls) for value in items):
            return tag, cls
    return None


def _label_text(value: Any) -> str:
    """The text of a coordinate value which is no number, truth value or string.

    A date and a datetime are spelled as `encode_window_key` spells the key of
    a terminal window, so that a key of either type matches the label of the
    batch it names.

    Args:
        value: the value.

    Returns:
        Its text.
    """
    if isinstance(value, date | datetime):
        return value.isoformat()
    return str(value)


def mixed_label_dim(timecourses: Timecourses) -> str | None:
    """The sample dimension whose labels the two encoders of the host disagree about.

    A coordinate travels under one type for the whole column
    (`_coordinate_values`), as the kernel's `Coordinate` carries one, where
    `encode_window_key` keeps the type of every single key of
    `TerminalPhase.windows`. For a column whose values share one type the two
    agree, and for one whose values do not, the column travels as text while a
    key of it travels as its own type: the kernel then holds `Str("1")` against
    `Int(1)`, finds no sample of that name and refuses the batch with a plain
    `ValueError`, where the python core matches the label (`sample_keys`).

    Such a batch is therefore the host's own refusal
    (`MIXED_LABELS_REASON`) rather than a batch the kernel may see, and the
    refusal is as narrow as the defect: a value whose record is the text of
    the label itself (`TEXT_KEY_TYPES`, a string as itself and a date or a
    datetime as its isoformat) meets the text of the column whichever way the
    column was tagged, so a column of dates and a column of dates and strings
    are both carried faithfully and are not refused. Only the sample
    dimensions are read, since the label of a sample is the value of its
    dimension coordinate and nothing else (`sample_keys`), and a window keyed
    by anything which is no label of the batch is refused by the host before
    either engine is reached (`sample_windows`).

    Args:
        timecourses: the batch.

    Returns:
        The name of the first sample dimension whose labels do not survive the
        crossing, or `None` when every one of them does.
    """
    for dim in timecourses.sample_dims:
        coord = timecourses.ds.coords.get(dim)
        if coord is None or coord.dtype.kind != "O":
            continue
        items = _unwrapped(coord.to_numpy().ravel().tolist())
        if _object_tag(items) is not None:
            # the column carries the one type of its values, as a key does
            continue
        if any(_key_type(value) not in TEXT_KEY_TYPES for value in items):
            return str(dim)
    return None


def _key_type(value: Any) -> str | None:
    """The record type `encode_window_key` would give a label, `None` for none.

    Args:
        value: the value of a label column.

    Returns:
        The `type` of the record, or `None` for a value no record spells, which
        is no text either and makes the column of it a refusal.
    """
    try:
        return str(encode_window_key(value)["type"])
    except ValueError:
        return None


def _names_a_sample(windows: Mapping[Any, Any] | None) -> bool:
    """Whether a terminal window is keyed by the label of a sample.

    The default key names every sample the mapping does not name
    (`WINDOW_DEFAULT_KEY`) and is no label, so a mapping of that key alone
    holds no label against the samples of the batch and nothing of the labels
    is compared.

    Args:
        windows: the windows of `TerminalPhase`, `None` when none are given.

    Returns:
        Whether at least one key is the label of a sample.
    """
    return any(key != WINDOW_DEFAULT_KEY for key in windows or ())


def options_dict(options: NCAOptions) -> dict[str, Any]:
    """The options as the dict of the nca options of version 1.

    One key per field of `NCAOptions`, which the bindings read by name: a
    field the model gains and the kernel does not map is refused by the
    kernel, naming the key, and a field it loses is refused the same way, so
    an option cannot go silently inert. `_nca_options_version` is no key of
    it, as `contract` is no key of the batch.

    The terminal windows travel as the typed records of `encode_window_key`
    and not as a mapping keyed by the label, because a python dict key says no
    more about whether it is a string, a number or a tuple than a JSON key
    does. An interval of two times is a tuple on both sides, where
    `BaseModel.model_dump` writes a list for one.

    Args:
        options: the options of the analysis

    Returns:
        The dict `pkpdutils_core.analyze_batch` and
        `pkpdutils_core.is_supported` take.
    """
    payload = options.model_dump(mode="json", exclude={"terminal": {"windows"}})
    windows = options.terminal.windows
    payload["terminal"]["windows"] = (
        None
        if windows is None
        else [
            {"key": encode_window_key(key), "window": _interval(window)}
            for key, window in windows.items()
        ]
    )
    payload["partial_aucs"] = {
        name: _interval(window) for name, window in options.partial_aucs.items()
    }
    return payload


def _interval(window: tuple[float, float]) -> tuple[float, float]:
    """Two times as the pair of the abi, which is a tuple and no list.

    Args:
        window: the start and the end of the interval.

    Returns:
        The pair.
    """
    start, end = window
    return float(start), float(end)


def kernel_result(
    timecourses: Timecourses,
    options: NCAOptions,
    *,
    prepared: KernelInput | None = None,
) -> dict[str, Any]:
    """Run the kernel on a batch and give back its result as it comes.

    The six sections of the bindings: `parameters`, `units` over the variables
    of every section, `flags`, `point_variables`, `booleans` and `texts`.
    `run_kernel` flattens the ones the host analyses with; the others are read
    from here by whatever compares the two engines.

    Args:
        timecourses: the batch
        options: the options of the analysis

    Keyword Args:
        prepared: the conversion of the batch and the options, made once by the
            caller which asked `unsupported_reason` as well; built here when
            `None`

    Returns:
        The result of `pkpdutils_core.analyze_batch`.

    Raises:
        UnsupportedError: if this build of the kernel does not carry the
            analysis, which `unsupported_reason` answers without running
            anything.
    """
    analyze = pkpdutils_core.analyze_batch  # ty: ignore[missing-attribute]
    batch, kernel_options = (prepared or KernelInput(timecourses, options)).dicts
    return dict(
        analyze(
            batch,
            kernel_options,
            n_workers=options.n_workers,
        )
    )


def _as_computed(values: dict[str, np.ndarray], timecourses: Timecourses) -> None:
    """The columns of the kernel as the python core would have computed them.

    Two things separate the two: the normalization of the units and the dtype
    of the integer variables.

    `run_rows` reports a parameter in the unit the batch was measured in and
    `pkpdutils.nca.nca._to_result` converts it to the reported unit of
    `pkpdutils.nca.result.parameter_unit` (a volume to `liter`, a clearance to
    `liter/hour`, every other parameter by a factor of 1). The kernel applies
    the same factor to its own columns, so they arrive converted, and
    `_to_result` would convert them a second time.

    The factor is therefore divided back out here rather than skipped in
    `_to_result`, for two reasons: the mapping of an engine then means one
    thing, the numbers as they were computed, which is what makes the two
    engines comparable at all, and the unit of a result stays pint's answer
    alone. A factor is no power of ten in general - the ones the analysis
    really reaches include 60, for a clearance of a batch measured per minute,
    and about 4.17e-8, for a volume of a batch in nanomolar units - but it is
    applied once in each direction, so `x * f / f` costs at most an ulp of the
    column against the tolerance of 1e-6 the two engines are compared with.

    An integer variable is cast and never divided: it counts rather than
    measures, so its expression is dimensionless and its factor is 1, which is
    checked rather than assumed, and with a raise rather than with an `assert`,
    which `python -O` removes from the one place the invariant is read.

    Args:
        values: the columns of the kernel, converted in place
        timecourses: the analysed batch, which carries the units

    Raises:
        RuntimeError: if a counted variable carries a unit factor, which would
            report it as a number of another unit than the host believes.
    """
    # `pkpdutils.nca.nca` imports this module, so the table of the unit
    # expressions is read where it is needed and not at the top
    from pkpdutils.nca.nca import INTEGER_VARIABLES, unit_expression

    for name, column in values.items():
        _, factor = parameter_unit(
            unit_expression(name),
            unit=timecourses.unit,
            time_unit=timecourses.time_unit,
            dose_unit=timecourses.dose_unit,
        )
        if name in INTEGER_VARIABLES:
            # every column of the kernel is a double, where the python core
            # counts in integers what is counted (`flags`, the rule of
            # `c0_method`); `_to_result` casts such a column and never
            # multiplies it, so there is no factor to divide back out
            if factor != 1.0:
                raise RuntimeError(
                    f"the counted variable '{name}' carries the unit factor {factor}"
                )
            values[name] = column.astype(np.int64)
            continue
        if factor != 1.0:
            values[name] = column / factor


_ORDER_CACHE: dict[tuple[Any, ...], list[str]] = {}
"""The variable order of `_python_order` per shape of a batch of several routes.

The order the python core reports a batch of several routes in is a function
of three things and of nothing else: the routes in the order of their first
appearance, whether the batch carries doses and the options. It does not
depend on the data, on the number of rows or on the number of time points, so
it is computed once per shape and not once per analysis: the dry run of
`_dry_order` costs about a millisecond per route, which is more than the whole
analysis of the small batch of two arms every bioavailability study produces
and would make the kernel path the slower one for that shape. A key which is
already known costs tens of microseconds.

The key carries `repr(options)`, which is every field of `NCAOptions` and of
its nested models, rather than a list of the fields which are thought to
matter: a key which omits one serves a wrong order from the cache, which is
worse than the dry run it saves. `model_dump_json` is no alternative, it
refuses a window keyed by a numpy scalar and spells the keys `1` and `"1"`
the same way.
"""

_ORDER_CACHE_MAX: int = 64
"""How many shapes `_ORDER_CACHE` holds before it is cleared.

A process analyses a handful of shapes; a flood of keys is a caller which
varies its options per call, and dropping the whole table costs that caller
one dry run per batch and every other caller nothing.
"""


def _order_key(
    routes: tuple[Route, ...], dosed: int, options: NCAOptions
) -> tuple[Any, ...]:
    """The key of `_ORDER_CACHE`: the shape the variable order depends on.

    The routes in the order of their first appearance, the width of the
    protocol and the options, as `repr(options)`, which is every field of
    `NCAOptions` and of its nested models: a key which names the fields
    thought to matter would serve a wrong order from the cache for the one
    field it forgets, and a wrong order is worse than the dry run it saves.

    Args:
        routes: the routes of the batch, in the order of their first
            appearance, every one of them once
        dosed: the number of doses per row of the protocol, 0 without one
        options: the options of the analysis

    Returns:
        The key, a tuple of hashable values.
    """
    return (routes, dosed, repr(options))


def _dry_order(
    routes: tuple[Route, ...], *, dosed: int, n_time: int, options: NCAOptions
) -> list[str]:
    """The variables of a batch of several routes, in the order the python core reports.

    The core is run once per route, in the order given, on no rows at all,
    which costs a pass of the vectorized core over empty arrays and reports
    the variables of that route, and the mappings are merged by `merge_rows`,
    the very function the analysis merges its groups with. That matters: the
    order of a merge is not the order of the groups but the group which
    reports the most variables first, ties keeping the order of the groups, a
    rule which lives in `merge_rows` and is read from it here rather than
    repeated.

    Args:
        routes: the routes of the batch, in the order of their first
            appearance, every one of them once

    Keyword Args:
        dosed: the number of doses per row of the protocol, 0 for a batch
            without doses, whose rows report no dose dependent variable
        n_time: the number of time points of the batch
        options: the options of the analysis

    Returns:
        The names of the variables in the order `run_rows` reports them in.
    """
    # `pkpdutils.nca.nca` imports this module, so the core is reached where it
    # is needed and not at the top
    from pkpdutils.nca.nca import merge_rows, run_rows

    time = np.zeros((0, n_time))
    amounts = np.zeros((0, dosed)) if dosed else None
    durations = np.full((0, dosed), np.nan) if dosed else None
    parts = [
        run_rows(
            time,
            time,
            dose_amount=amounts,
            dose_time=amounts,
            dose_duration=durations,
            route=route,
            options=options,
        )
        for route in routes
    ]
    return list(merge_rows(parts, [0] * len(parts)))


def _python_order(
    routes: np.ndarray,
    options: NCAOptions,
    names: Iterable[str],
    *,
    dosed: int,
    n_time: int,
) -> list[str]:
    """The order the python core reports the variables of a batch of several routes in.

    The kernel analyses a batch of several routes in one call and reports one
    union of variables, while `run_rows` runs one group per route and merges
    their mappings (`merge_rows`), so the two agree on the variables of such a
    batch and not on their order: `cl` and `cl_f` of a batch of a bolus arm
    and an oral arm stand next to each other in the kernel's answer and in two
    blocks, one per route, in python's. The mapping of an engine is the mapping
    of the rows, so the kernel's answer is sorted into python's order here
    rather than python's order being left to whichever engine ran.

    The order is python's own answer and no table of this module (`_dry_order`),
    read from `_ORDER_CACHE` for a shape which was analysed before. A variable
    the dry run does not name keeps the place the kernel gave it, behind the
    ones it does.

    The routes are taken as the array rather than read from the batch, since
    the property builds it (one `Route` per sample) and the caller has it in
    hand already.

    Args:
        routes: the route of every row of the batch
            (`pkpdutils.nca.nca.row_routes`, `Timecourses.routes`)
        options: the options of the analysis
        names: the variables to sort, the keys of the kernel's mapping

    Keyword Args:
        dosed: the number of doses per row of the protocol (`Timecourses.n_dose`,
            0 for a batch without a protocol, exactly where `dose_arrays`
            reports an empty row)
        n_time: the number of time points of the batch

    Returns:
        The names in the order `run_rows` would report them in.
    """
    # the routes of a batch are a handful of values repeated over its rows, so
    # they are deduplicated before they are read as routes and not after
    given = tuple(Route(value) for value in dict.fromkeys(routes.ravel().tolist()))
    key = _order_key(given, dosed, options)
    order = _ORDER_CACHE.get(key)
    if order is None:
        order = _dry_order(given, dosed=dosed, n_time=n_time, options=options)
        if len(_ORDER_CACHE) >= _ORDER_CACHE_MAX:
            _ORDER_CACHE.clear()
        _ORDER_CACHE[key] = order
    wanted = set(names)
    named = set(order)
    return [name for name in order if name in wanted] + [
        name for name in names if name not in named
    ]


def run_kernel(
    timecourses: Timecourses,
    options: NCAOptions,
    *,
    units: dict[str, str] | None = None,
    prepared: KernelInput | None = None,
) -> dict[str, np.ndarray]:
    """Run the kernel on a batch, as `run_rows` runs the python core.

    The parameters of the kernel, its point variables as `(N, K)` arrays and
    its flags, in the layout and the order `pkpdutils.nca.nca.run_rows`
    reports: one `(N,)` array per parameter, the variables the host writes
    itself left out (`HOST_VARIABLES`) and the numbers in the units the batch
    was measured in (`_as_computed`). `flags` is the last variable of a batch
    of one route; a batch of several is sorted into python's order by
    `_python_order`, which carries `flags` inside the block of the group
    `merge_rows` reports first. The units of the kernel are not adopted
    either: `nca` derives the unit of every variable of its result with pint,
    from the units of the batch, whichever engine computed the numbers.

    Args:
        timecourses: the batch
        options: the options of the analysis

    Keyword Args:
        units: a dict the unit the kernel reports for every variable of its
            result is written into, the one part of its answer which
            `run_kernel` neither adopts nor drops. The differential gate reads
            them from here and holds them against pint's own derivation
            (`compare_with_python`), so that the kernel's units are compared
            without the analysis running twice; `None` outside the gate, which
            is every ordinary call.
        prepared: the conversion of the batch and the options, see
            `unsupported_reason`; built here when `None`

    Returns:
        One array per variable, `flags` among the keys.

    Raises:
        UnsupportedError: if this build of the kernel does not carry the
            analysis, which `unsupported_reason` answers without running
            anything.
    """
    result = kernel_result(timecourses, options, prepared=prepared)
    if units is not None:
        units.update(result["units"])
    n_rows = timecourses.n_samples
    values: dict[str, np.ndarray] = {
        name: np.asarray(column, dtype=np.float64)
        for name, column in result["parameters"].items()
        if name not in HOST_VARIABLES
    }
    for name, variable in result["point_variables"].items():
        width = int(variable["size"])
        values[name] = np.asarray(variable["values"], dtype=np.float64).reshape(
            n_rows, width
        )
    # `flags` goes last, where the python core reports it for a batch of one
    # route; `_python_order` moves it for a batch of several
    values["flags"] = np.asarray(result["flags"], dtype=np.float64)
    _as_computed(values, timecourses)
    routes = timecourses.routes
    if routes is None:
        # a batch of one route ran in one group and is in python's order
        return values
    order = _python_order(
        routes,
        options,
        values,
        dosed=timecourses.n_dose,
        n_time=timecourses.n_time,
    )
    return {name: values[name] for name in order}


# --- the differential gate ----------------------------------------------------

COMPARE_VARIABLE: str = "PKPDUTILS_COMPARE_ENGINES"
"""The environment variable which arms the differential gate.

Set to anything but an empty string, `0`, `false`, `no` or `off`, every batch
the kernel analyses is analysed by the python core as well and the two answers
are compared, which costs more than either engine alone and is meant for a
test run and for a bisection, never for production. The suite runs with it in
continuous integration (`.github/workflows/ci-cd.yml`), which is the acceptance
evidence that the two engines agree on every batch the tests build.
"""

COMPARE_REL: float = 1e-6
"""The relative tolerance the values of the two engines are compared with.

The tolerance of `tests/nca/test_reference.py`, which the analysis is held to
against `pkdb_analysis` 0.3.1; the two engines are held to the same one.
"""

COMPARE_ABS: float = 1e-12
"""The absolute tolerance the values of the two engines are compared with."""

KERNEL_UNITLESS: frozenset[str] = frozenset({"flags"})
"""The variables of the mapping whose unit the kernel reports for nobody.

`pkpdutils-core` (0.2.0 to 0.3.0) reports a unit for every parameter of its result and
none for its flags, which carry no unit on either side (`PARAMETER_UNITS`
spells them `dimensionless`). A name of this set is left out of the unit
comparison **only while the kernel really reports nothing for it**: the day it
reports one, the comparison takes it like any other, so the exemption cannot go
stale. Every other variable of the mapping is compared, which is what makes a
unit the kernel stops reporting a disagreement rather than a silence.
"""

FALSE_WORDS: frozenset[str] = frozenset({"", "0", "false", "no", "off"})
"""The values of `COMPARE_VARIABLE` which leave the gate disarmed."""


class EngineMismatch(RuntimeError):
    """The two engines disagreed on a batch while the differential gate was armed.

    No `ValueError`, so that nothing on the dispatch of `pkpdutils.nca.nca.nca`
    can mistake it for the refusal of an analysis and answer it by taking the
    python path: a disagreement is reported to whoever armed the gate and
    never swallowed.
    """


def comparing() -> bool:
    """Whether the differential gate is armed (`COMPARE_VARIABLE`).

    One lookup of the environment per analysis, which is what an ordinary run
    of `pkpdutils.nca.nca.nca` pays for the gate. The value is read every time
    rather than once at import, so that a test arms and disarms it with
    `monkeypatch.setenv`.

    Returns:
        Whether the two engines are to be compared.
    """
    return os.environ.get(COMPARE_VARIABLE, "").strip().lower() not in FALSE_WORDS


_comparisons: int = 0
"""How many batches the gate has compared, see `comparisons`."""


def note_comparison() -> None:
    """Record that the gate compared the two engines on one batch.

    Called by the dispatch of `pkpdutils.nca.nca.nca` after a comparison and
    deliberately **not** by `compare_with_python` itself: the tests of the gate
    call that function directly with `COMPARE_VARIABLE` set by `monkeypatch`,
    so a counter kept inside it could never be zero and the sentinel which
    reads it would be satisfied by the gate's own unit tests instead of by the
    batch of an analysis.
    """
    global _comparisons
    _comparisons += 1


def comparisons() -> int:
    """How many batches the gate compared since this module was imported.

    The witness `tests/conftest.py` fails a test session on: a run which says
    it armed the gate and compared nothing took the whole suite through one
    engine while reporting the agreement of two, which is the one failure of
    this gate that nothing else can see - every test passes, in the ordinary
    way, and the evidence is a job that was never armed.

    Read as a floor and not as an exact count: nothing is asserted against the
    number itself, only against its being above zero. `importlib.reload` of
    this module sets it back to zero, which one test of
    `tests/nca/test_engine.py` does on purpose and repairs afterwards by
    replaying the count through `note_comparison`, so an armed run whose last
    comparison came before that test still reports what it compared.

    Returns:
        The number of batches `pkpdutils.nca.nca.nca` compared.
    """
    return _comparisons


def compare_engines(
    python: Mapping[str, np.ndarray],
    kernel: Mapping[str, np.ndarray],
    *,
    units_python: Mapping[str, str],
    units_kernel: Mapping[str, str],
) -> list[str]:
    """Every way in which two mappings of the same analysis disagree.

    The mappings are the ones the two engines report
    (`pkpdutils.nca.nca.run_analysis_rows` and `run_kernel`), in the layout the
    result is assembled from, and the two unit mappings are the unit of every
    variable as each engine spells it.

    What is compared, and how:

    - **the names**, in both directions: a variable one engine reports and the
      other does not is named rather than skipped, which is the one way a
      comparison of two mappings can pass by looking at nothing
    - **the order** of the variables they share, since it is the order of the
      columns of `NCAResult.to_dataframe` and of every table rendered from a
      result, so a swap is observable to a user
    - **the shape and the dtype** of every column, a counted variable being an
      integer on both sides and a measured one a double
    - **the positions of `NaN`**, exactly: a parameter which is `NaN` in one
      engine and a number in the other is a disagreement about whether the
      analysis has an answer at all, and a tolerance must never hide it
    - **the numbers**, within `COMPARE_REL` and `COMPARE_ABS`, except a column
      which counts rather than measures: a flag word and the rule of
      `c0_method` are compared exactly, since a relative tolerance on a bit
      field would let a high bit differ unnoticed
    - **the units**, as strings, exactly

    Args:
        python: one array per variable, as the python core reports them
        kernel: one array per variable, as the kernel reports them

    Keyword Args:
        units_python: the unit of every variable as the host derives it with
            pint
        units_kernel: the unit of every variable as the kernel reports it

    Returns:
        One sentence per disagreement, naming the variable and both values;
        empty when the two engines agree.
    """
    disagreements: list[str] = []
    missing = [name for name in python if name not in kernel]
    if missing:
        disagreements.append(f"the kernel reports none of the variables {missing}")
    extra = [name for name in kernel if name not in python]
    if extra:
        disagreements.append(f"the python core reports none of the variables {extra}")
    shared = [name for name in python if name in kernel]
    if shared != [name for name in kernel if name in python]:
        disagreements.append(_order_disagreement(python, kernel))
    for name in shared:
        disagreements.extend(_compare_columns(name, python[name], kernel[name]))
    for name in dict.fromkeys([*units_python, *units_kernel]):
        left, right = units_python.get(name), units_kernel.get(name)
        if left != right:
            disagreements.append(
                f"'{name}': the python core reports the unit {left!r} and the "
                f"kernel {right!r}"
            )
    return disagreements


def _order_disagreement(
    python: Mapping[str, np.ndarray], kernel: Mapping[str, np.ndarray]
) -> str:
    """The first position at which the two engines name a different variable.

    The names they share, in the order of each mapping; a variable only one of
    them reports is left out, since it is reported on its own and would
    otherwise shift every name behind it.

    Args:
        python: the mapping of the python core
        kernel: the mapping of the kernel

    Returns:
        The sentence.
    """
    left = [name for name in python if name in kernel]
    right = [name for name in kernel if name in python]
    for position, (first, second) in enumerate(zip(left, right, strict=True)):
        if first != second:
            return (
                f"the variables are in a different order: at position "
                f"{position} the python core reports '{first}' and the kernel "
                f"'{second}'"
            )
    # unreachable: the caller compares the two lists before it asks
    return "the variables are in a different order"


def _compare_columns(name: str, python: np.ndarray, kernel: np.ndarray) -> list[str]:
    """The disagreements about one variable of the two mappings.

    Args:
        name: the name of the variable
        python: its column as the python core reports it
        kernel: its column as the kernel reports it

    Returns:
        One sentence per disagreement, empty when the two agree.
    """
    if python.shape != kernel.shape:
        return [
            f"'{name}': the python core reports the shape {python.shape} and "
            f"the kernel {kernel.shape}"
        ]
    disagreements: list[str] = []
    if python.dtype != kernel.dtype:
        disagreements.append(
            f"'{name}': the python core reports the dtype {python.dtype} and "
            f"the kernel {kernel.dtype}"
        )
    if python.dtype.kind in "iub" and kernel.dtype.kind in "iub":
        # a counted variable is exact on both sides: a flag word differing in a
        # high bit alone passes any relative tolerance
        if not np.array_equal(python, kernel):
            disagreements.append(_exact_disagreement(name, python, kernel))
        return disagreements
    left = np.asarray(python, dtype=np.float64)
    right = np.asarray(kernel, dtype=np.float64)
    nan_left, nan_right = np.isnan(left), np.isnan(right)
    if not np.array_equal(nan_left, nan_right):
        disagreements.append(
            _position_disagreement(
                name,
                left,
                right,
                np.flatnonzero((nan_left != nan_right).ravel()),
                "are NaN in one engine and a number in the other",
            )
        )
    both = np.flatnonzero(~(nan_left | nan_right).ravel())
    flat_left, flat_right = left.ravel()[both], right.ravel()[both]
    apart = both[~np.isclose(flat_left, flat_right, rtol=COMPARE_REL, atol=COMPARE_ABS)]
    if apart.size:
        disagreements.append(
            _position_disagreement(
                name,
                left,
                right,
                apart,
                f"differ by more than rel={COMPARE_REL:g}, abs={COMPARE_ABS:g}",
                worst=True,
            )
        )
    return disagreements


def _exact_disagreement(name: str, python: np.ndarray, kernel: np.ndarray) -> str:
    """The sentence of a counted variable whose values are not equal.

    Args:
        name: the name of the variable
        python: its column as the python core reports it
        kernel: its column as the kernel reports it

    Returns:
        The sentence, naming the first value which differs.
    """
    apart = np.flatnonzero((python != kernel).ravel())
    first = int(apart[0])
    return (
        f"'{name}': {apart.size} of {python.size} counted values are not "
        f"equal, the first at the flat index {first}, where the python core "
        f"reports {python.ravel()[first].item()!r} and the kernel "
        f"{kernel.ravel()[first].item()!r}"
    )


def _position_disagreement(
    name: str,
    python: np.ndarray,
    kernel: np.ndarray,
    positions: np.ndarray,
    what: str,
    *,
    worst: bool = False,
) -> str:
    """The sentence of a variable whose values disagree at the given positions.

    Args:
        name: the name of the variable
        python: its column as the python core reports it, as doubles
        kernel: its column as the kernel reports it, as doubles
        positions: the flat indices at which the two disagree
        what: how they disagree, read after the count
        worst: whether to name the position of the largest relative difference
            rather than the first; for a difference of two numbers, where the
            largest one is what says whether a tolerance is nearly met or
            missed by a mile

    Returns:
        The sentence, naming the count, one position and both values there.
    """
    flat_left, flat_right = python.ravel()[positions], kernel.ravel()[positions]
    at = 0
    if worst:
        scale = np.maximum(np.abs(flat_left), np.abs(flat_right))
        error = np.abs(flat_left - flat_right) / np.where(scale > 0.0, scale, 1.0)
        at = int(np.argmax(error))
    return (
        f"'{name}': {positions.size} of {python.size} values {what}, at the "
        f"flat index {int(positions[at])} the python core reports "
        f"{float(flat_left[at])!r} and the kernel {float(flat_right[at])!r}"
    )


def host_units(names: Iterable[str], timecourses: Timecourses) -> dict[str, str]:
    """The unit of every named variable as the host derives it with pint.

    `pkpdutils.nca.nca._to_result` derives the unit of every variable of a
    result this way, whichever engine computed the numbers, so these are the
    units a caller of `nca` really sees. A name the host knows no expression
    for is left out rather than raising: it is a variable of the kernel alone,
    which `compare_engines` reports by name.

    Args:
        names: the variables to derive the units of
        timecourses: the analysed batch, which carries the units

    Returns:
        The unit of every name, in the order of the names.
    """
    # `pkpdutils.nca.nca` imports this module, so the table of the unit
    # expressions is read where it is needed and not at the top
    from pkpdutils.nca.nca import unit_expression

    units: dict[str, str] = {}
    for name in names:
        try:
            expression = unit_expression(name)
        except KeyError:
            continue
        units[name] = parameter_unit(
            expression,
            unit=timecourses.unit,
            time_unit=timecourses.time_unit,
            dose_unit=timecourses.dose_unit,
        )[0]
    return units


def compare_with_python(
    timecourses: Timecourses,
    *,
    kernel: Mapping[str, np.ndarray],
    python: Mapping[str, np.ndarray],
    units_kernel: Mapping[str, str],
) -> None:
    """Hold the kernel's answer against the python core's, and raise if they differ.

    The gate of `pkpdutils.nca.nca.nca`, armed by `COMPARE_VARIABLE`: the two
    mappings are the answers of the two engines to one batch and the units are
    the kernel's own, which are held against pint's derivation (`host_units`) -
    the one part of the kernel's answer the host does not adopt, so a kernel
    which computed a parameter in another unit than the host believes is a
    disagreement here rather than a wrong number nobody can see.

    Args:
        timecourses: the analysed batch, which carries the units and names the
            shape in the message

    Keyword Args:
        kernel: the mapping of the kernel
        python: the mapping of the python core on the same batch
        units_kernel: the unit of every variable as the kernel reports it
            (`run_kernel(..., units=)`)

    Raises:
        EngineMismatch: if the two engines disagree in any way
            `compare_engines` compares.
    """
    # a variable the kernel reports no unit for is left out of the unit
    # comparison only while that is really so, see `KERNEL_UNITLESS`
    names = [
        name
        for name in dict.fromkeys([*python, *kernel])
        if name in units_kernel or name not in KERNEL_UNITLESS
    ]
    disagreements = compare_engines(
        python,
        kernel,
        units_python=host_units(names, timecourses),
        units_kernel={
            name: units_kernel[name] for name in names if name in units_kernel
        },
    )
    if not disagreements:
        return
    routes = timecourses.routes
    raise EngineMismatch(
        "the two engines disagree on a batch of "
        f"{timecourses.n_samples} sample(s) by {timecourses.n_time} time "
        f"point(s) in {timecourses.unit!r} over {timecourses.time_unit!r}, "
        f"{'several routes' if routes is not None else timecourses.route}:\n"
        + "\n".join(f"- {sentence}" for sentence in disagreements)
    )
