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
  analyses. `unsupported_reason` is an **optimization** which mirrors them, so
  that a batch the kernel would refuse is never converted at all.
- A mirror which is too narrow is the dangerous direction: the batch quietly
  takes the python path, the numbers are right and nobody learns that the fast
  path never fired. A mirror which is too wide corrects itself, since the
  kernel refuses with `UnsupportedError` and the caller falls back. The reason
  this module reports is therefore the kernel's own sentence, word for word,
  and `tests/nca/test_engine.py` asserts the two agree on every refusal rather
  than asserting a text of its own.

The two engines report the same variables with the same numbers, and one
analysis in a different order: a batch of several routes. The python core runs
one group per route and takes the union of their variables in the order the
routes appear in the batch (`pkpdutils.nca.nca.merge_rows`), where the kernel
reports the union in its canonical order. Nothing of the result but the order
of its variables differs, and the order of the kernel does not depend on which
route the batch names first.

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
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

import numpy as np
import xarray as xr

from pkpdutils.nca.options import (
    Kind,
    NCAOptions,
    UncertaintyMethod,
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

# the wheel of `pkpdutils-core` 0.2.0 carries no stub beside its compiled
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
0.2.0 derives the variables of its result from the routes of the rows it ran,
so a batch of no rows comes back with one column instead of with the empty
column of every variable python reports (`docs/carry-forward.md` of the kernel
names the fix, which belongs on its side). The host therefore answers an empty
batch itself, which costs nothing: there is no row to analyse.
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


def unsupported_reason(timecourses: Timecourses, options: NCAOptions) -> str | None:
    """Why the kernel would refuse this analysis, `None` when it would run it.

    The conditions are `unsupported_reason` of `crates/python/src/options.rs`
    in its order, since an analysis may trip over several of them and only the
    first is reported, and every sentence is the kernel's own (the `what` of
    its `KernelError::Unsupported`), so that a reason of this module and a
    reason of the kernel cannot drift apart unnoticed. Two reasons are the
    host's own and are no refusal of the kernel, `FORCED_REASON` and
    `EMPTY_REASON`, which is why they are named rather than written here:

    - the analysis of an effect curve (`NCAOptions.kind`)
    - a multiple dose analysis (`NCAOptions.tau`)
    - a protocol of more than one dose in any row of the batch
    - the acceptance criteria, whenever one of their four thresholds is set
      (`NCAOptions.acceptance`)
    - the named partial areas (`NCAOptions.partial_aucs`)
    - the propagation of the uncertainty of a group curve, of what the options
      *resolve* to and not of what they say: a batch with `sd` or `se` is
      refused although the options name no method, and one is analysed
      although it carries `sd` when they name `UncertaintyMethod.NONE`. The
      counts `n` play no part in it, as they play none in
      `NCAOptions.resolve_uncertainty`
    - the unit conversion of a result (`NCAOptions.units`)

    Args:
        timecourses: the batch
        options: the options of the analysis

    Returns:
        The reason, or `None` for an analysis the kernel carries.
    """
    if _force_python:
        return FORCED_REASON
    if timecourses.n_samples == 0:
        return EMPTY_REASON
    if options.kind is Kind.EFFECT:
        return "the analysis of an effect curve (Kind::Effect)"
    if options.tau is not None:
        return "a multiple dose analysis (NcaOptions::tau)"
    amounts, times, _ = dose_arrays(timecourses)
    # the doses of a row are the columns whose amount and whose time are both a
    # number, counted on the very arrays `batch_dict` sends, so that the
    # predicate and the batch cannot read one protocol differently
    if np.any((np.isfinite(amounts) & np.isfinite(times)).sum(axis=1) >= 2):
        return "a protocol of more than one dose (the multiple dose analysis)"
    acceptance = options.acceptance
    if any(
        threshold is not None
        for threshold in (
            acceptance.r2_adj_min,
            acceptance.extrapolation_max,
            acceptance.span_min,
            acceptance.n_points_min,
        )
    ):
        return "the acceptance criteria (NcaOptions::acceptance)"
    if options.partial_aucs:
        return "the named partial areas (NcaOptions::partial_aucs)"
    if resolved_uncertainty(timecourses, options) is not UncertaintyMethod.NONE:
        return "the propagation of the uncertainty of a group curve"
    if options.units:
        return "the unit conversion of a result (NcaOptions::units)"
    return None


def resolved_uncertainty(
    timecourses: Timecourses, options: NCAOptions
) -> UncertaintyMethod:
    """The uncertainty the analysis would really propagate.

    `resolved_uncertainty` of `crates/python/src/options.rs`, which is
    `NCAOptions.resolve_uncertainty` of the spread the batch carries: the
    bootstrap for a batch with `sd` or `se` per point, nothing for a batch of
    individual curves. The counts `n` are no spread and play no part.

    Args:
        timecourses: the batch
        options: the options of the analysis

    Returns:
        The method, `UncertaintyMethod.NONE` for an analysis which propagates
        nothing.
    """
    spread = timecourses.sd is not None or timecourses.se is not None
    return options.resolve_uncertainty(spread)


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
        # a numpy scalar is no instance of the python type it holds, so it is
        # unwrapped before it is read, exactly as `encode_window_key` unwraps
        # the key of a terminal window: an object array of `np.int64` is what
        # `xarray` hands over for a coordinate of integers it was given as an
        # object array, and it has to read as the integers it holds and not as
        # their text
        items = [
            value.item() if isinstance(value, np.generic) else value for value in items
        ]
        # the order of `encode_window_key`: a truth value is an integer in
        # python and the text of a number is no number
        for tag, cls in (("bool", bool), ("str", str), ("int", int), ("float", float)):
            if items and all(isinstance(value, cls) for value in items):
                return {tag: [cls(value) for value in items]}
    return {"str": [_label_text(value) for value in items]}


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


def kernel_result(timecourses: Timecourses, options: NCAOptions) -> dict[str, Any]:
    """Run the kernel on a batch and give back its result as it comes.

    The six sections of the bindings: `parameters`, `units` over the variables
    of every section, `flags`, `point_variables`, `booleans` and `texts`.
    `run_kernel` flattens the ones the host analyses with; the others are read
    from here by whatever compares the two engines.

    Args:
        timecourses: the batch
        options: the options of the analysis

    Returns:
        The result of `pkpdutils_core.analyze_batch`.

    Raises:
        UnsupportedError: if this build of the kernel does not carry the
            analysis, which `unsupported_reason` answers without running
            anything.
    """
    analyze = pkpdutils_core.analyze_batch  # ty: ignore[missing-attribute]
    return dict(
        analyze(
            batch_dict(timecourses),
            options_dict(options),
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
    asserted rather than assumed.

    Args:
        values: the columns of the kernel, converted in place
        timecourses: the analysed batch, which carries the units
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
            assert factor == 1.0, (
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


def run_kernel(timecourses: Timecourses, options: NCAOptions) -> dict[str, np.ndarray]:
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

    Returns:
        One array per variable, `flags` among the keys.

    Raises:
        UnsupportedError: if this build of the kernel does not carry the
            analysis, which `unsupported_reason` answers without running
            anything.
    """
    result = kernel_result(timecourses, options)
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
