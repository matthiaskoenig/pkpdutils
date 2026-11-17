import importlib
import os
import subprocess
import sys
import warnings
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pkpdutils_core
import pytest

from pkpdutils import Dose, Dosing, Route, Timecourse, Timecourses
from pkpdutils.nca import (
    Acceptance,
    AUCMethod,
    BLQHandling,
    BootstrapDistribution,
    BootstrapSpread,
    C0Method,
    Kind,
    NCAFlag,
    NCAOptions,
    NCAResult,
    TerminalPhase,
    UncertaintyMethod,
    engine,
    nca,
    nca_single,
)

K, C0 = 0.5, 10.0
TIMES = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0])
IV_DOSE = Dose(amount=100.0, unit="mg", route=Route.IV_BOLUS)
ORAL_DOSE = Dose(amount=100.0, unit="mg", route=Route.ORAL)
INFUSION_DOSE = Dose(amount=100.0, unit="mg", route=Route.IV_INFUSION, duration=0.5)
# the tolerance of `tests/nca/test_reference.py`, which the two engines are
# held to as well
REL, ABS = 1e-6, 1e-12

# the wheel of `pkpdutils-core` carries no stub beside its compiled module,
# so the one lookup of the kernel's own predicate is suppressed here rather
# than at every call of it
is_supported = pkpdutils_core.is_supported  # ty: ignore[missing-attribute]


def curve(scale: float = 1.0, dose: Dose | Dosing = IV_DOSE) -> Timecourse:
    return Timecourse(
        time=TIMES,
        value=scale * C0 * np.exp(-K * TIMES),
        time_unit="hr",
        unit="mg/l",
        dose=dose,
        substance="x",
    )


def batch(**kwargs: Any) -> Timecourses:
    return Timecourses.from_timecourses(
        [curve(), curve(2.0)], dim="subject", labels=["A", "B"], **kwargs
    )


def group_batch(*, sd: bool = True, counts: bool = False) -> Timecourses:
    """A batch of one mean curve, with a spread per point, counts, or both."""
    values = C0 * np.exp(-K * TIMES)
    return Timecourses.from_arrays(
        TIMES,
        values[None, :],
        time_unit="hr",
        unit="mg/l",
        dims=("subject",),
        coords={"subject": ["mean"]},
        sd=0.1 * values[None, :] if sd else None,
        n=np.full((1, TIMES.size), 6.0) if counts else None,
        dose=IV_DOSE,
    )


def multiple_dose_batch() -> Timecourses:
    """One curve of three doses twelve hours apart, sampled over every interval."""
    # every interval is sampled to within `NCAOptions.tau_tolerance` of its end
    offsets = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 11.5])
    times = np.concatenate([offsets + 12.0 * i for i in range(3)])
    values = sum(
        np.where(times >= 12.0 * i, C0 * np.exp(-K * (times - 12.0 * i)), 0.0)
        for i in range(3)
    )
    return Timecourses.from_arrays(
        times[None, :],
        np.asarray(values)[None, :],
        time_unit="hr",
        unit="mg/l",
        dims=("subject",),
        coords={"subject": ["A"]},
        dose={
            "amount": [[100.0, 100.0, 100.0]],
            "unit": "mg",
            "time": [[0.0, 12.0, 24.0]],
        },
        route=Route.IV_BOLUS,
    )


def python_result(
    timecourses: Timecourses, options: NCAOptions | None = None
) -> NCAResult:
    """The result of the python core of the same batch and the same options."""
    forced = engine._force_python
    engine._force_python = True
    try:
        return nca(timecourses, options=options)
    finally:
        engine._force_python = forced


def assert_same_result(kernel: NCAResult, python: NCAResult) -> None:
    """Every variable of two results, their units and their flags, in order."""
    assert list(kernel.ds.data_vars) == list(python.ds.data_vars)
    assert {name: kernel.ds[name].attrs["units"] for name in kernel.ds.data_vars} == {
        name: python.ds[name].attrs["units"] for name in python.ds.data_vars
    }
    for name in python.ds.data_vars:
        left = kernel.ds[name].to_numpy()
        right = python.ds[name].to_numpy()
        assert left.shape == right.shape, name
        if left.dtype.kind in "fc":
            # the positions of NaN are compared exactly, a difference there is
            # a disagreement and not a rounding
            assert np.array_equal(np.isnan(left), np.isnan(right)), name
            assert left[~np.isnan(left)] == pytest.approx(
                right[~np.isnan(right)], rel=REL, abs=ABS
            ), name
        else:
            assert np.array_equal(left, right), name


def assert_same_mapping(
    timecourses: Timecourses, options: NCAOptions | None = None
) -> None:
    """The mapping of `run_kernel` against the mapping of the python core.

    The two engines below `nca`, on the variables as they were computed: their
    names, their order, their shapes, their dtypes and their numbers.

    A test which compares two results of `nca` alone cannot tell a batch the
    kernel really analysed from one it refused: the fallback is silent, so
    `nca` would compare the answer of the python core with itself and pass
    whatever the kernel does with a shape, a unit or a route it has never
    seen. `run_kernel` is the kernel and nothing else, and the eligibility of
    the batch is asserted here rather than assumed.
    """
    from pkpdutils.nca.nca import row_routes, run_analysis_rows, sample_windows

    options = options or NCAOptions()
    assert engine.unsupported_reason(timecourses, options) is None
    n_rows, n_time = timecourses.n_samples, timecourses.n_time
    amounts, times, durations = engine.dose_arrays(timecourses)
    dosed = amounts.shape[1] > 0
    route, routes = row_routes(timecourses, n_rows)
    lloq = timecourses.lloq
    rows = run_analysis_rows(
        timecourses.times.reshape(n_rows, n_time),
        timecourses.values.reshape(n_rows, n_time),
        dose_amount=amounts if dosed else None,
        dose_time=times if dosed else None,
        dose_duration=durations if dosed else None,
        route=route,
        options=options,
        lloq=None if lloq is None else lloq.reshape(n_rows),
        windows=sample_windows(timecourses, options.terminal),
        routes=routes,
    )
    kernel = engine.run_kernel(timecourses, options)
    assert list(kernel) == list(rows)
    for name, column in rows.items():
        assert kernel[name].shape == column.shape, name
        assert kernel[name].dtype == column.dtype, name
        assert kernel[name] == pytest.approx(column, nan_ok=True, rel=REL, abs=ABS), (
            name
        )


# --- the eligibility predicate ------------------------------------------------


def test_a_single_dose_batch_is_eligible() -> None:
    assert engine.unsupported_reason(batch(), NCAOptions()) is None


def test_a_multiple_dose_batch_is_eligible() -> None:
    assert engine.unsupported_reason(multiple_dose_batch(), NCAOptions()) is None
    assert (
        engine.unsupported_reason(multiple_dose_batch(), NCAOptions(tau=12.0)) is None
    )


def test_a_group_batch_is_not() -> None:
    reason = engine.unsupported_reason(group_batch(), NCAOptions())
    assert reason is not None
    assert "uncertainty" in reason


def test_the_counts_alone_do_not_refuse_a_batch() -> None:
    # the kernel resolves the uncertainty of `sd` and `se`, as
    # `NCAOptions.resolve_uncertainty` does; `n` is no spread and a predicate
    # which read it too would send every group mean curve down the slow path
    counted = group_batch(sd=False, counts=True)
    assert counted.n is not None
    assert engine.unsupported_reason(counted, NCAOptions()) is None


def test_one_count_per_sample_is_no_field_of_the_contract() -> None:
    # `batch/1` carries a count per time point only, and a count per sample
    # sent under that key is a length the kernel refuses; `nca` writes `n`
    # from `Timecourses.n_subjects` itself either way
    per_sample = Timecourses.from_arrays(
        TIMES,
        (C0 * np.exp(-K * TIMES))[None, :],
        time_unit="hr",
        unit="mg/l",
        dims=("subject",),
        coords={"subject": ["mean"]},
        n=6.0,
        dose=IV_DOSE,
    )
    assert per_sample.n is not None
    assert per_sample.n.shape == (1,)
    assert engine.batch_dict(per_sample)["n"] is None
    assert engine.unsupported_reason(per_sample, NCAOptions()) is None
    result = nca(per_sample)
    assert result.ds["n"].to_numpy() == pytest.approx(6.0)
    assert_same_result(result, python_result(per_sample))


def test_a_spread_with_an_explicit_method_of_none_is_eligible() -> None:
    # the refusal is of what the options resolve to and not of what the batch
    # carries
    options = NCAOptions(uncertainty=UncertaintyMethod.NONE)
    assert engine.unsupported_reason(group_batch(), options) is None


def test_a_batch_without_a_spread_and_an_explicit_method_is_not() -> None:
    options = NCAOptions(uncertainty=UncertaintyMethod.BOOTSTRAP)
    assert engine.unsupported_reason(batch(), options) is not None


REFUSED: list[tuple[str, NCAOptions]] = [
    ("effect curve", NCAOptions(kind=Kind.EFFECT, effect_threshold=1.0)),
    ("acceptance", NCAOptions(acceptance=Acceptance(span_min=3.0))),
    ("partial_aucs", NCAOptions(partial_aucs={"auc_0_4": (0.0, 4.0)})),
    ("units", NCAOptions(units={"auc_last": "mg*min/l"})),
]


@pytest.mark.parametrize(("what", "options"), REFUSED, ids=[r[0] for r in REFUSED])
def test_each_refused_option_is_named(what: str, options: NCAOptions) -> None:
    reason = engine.unsupported_reason(batch(), options)
    assert reason is not None, what
    # the reason names the option it refused, so a fallback says which
    # analysis of a batch the kernel does not carry
    assert what in reason, reason


def test_the_acceptance_criteria_without_a_threshold_are_eligible() -> None:
    # `exclude` alone excludes nothing, as the kernel reads it
    options = NCAOptions(acceptance=Acceptance(exclude=True))
    assert engine.unsupported_reason(batch(), options) is None


AGREEMENT: list[tuple[str, Timecourses, NCAOptions]] = [
    ("a single dose batch", batch(), NCAOptions()),
    ("a group batch", group_batch(), NCAOptions()),
    ("the counts alone", group_batch(sd=False, counts=True), NCAOptions()),
    (
        "a spread with no method",
        group_batch(),
        NCAOptions(uncertainty=UncertaintyMethod.NONE),
    ),
    (
        "no spread with a method",
        batch(),
        NCAOptions(uncertainty=UncertaintyMethod.DELTA),
    ),
    ("a multiple dose batch", multiple_dose_batch(), NCAOptions()),
    ("a given dosing interval", batch(), NCAOptions(tau=12.0)),
    *[(what, batch(), options) for what, options in REFUSED],
    (
        "the acceptance exclusion",
        batch(),
        NCAOptions(acceptance=Acceptance(exclude=True)),
    ),
    (
        "a window by hand",
        batch(),
        NCAOptions(terminal=TerminalPhase(windows={"A": (2.0, 12.0)})),
    ),
]


@pytest.mark.parametrize(
    ("what", "timecourses", "options"), AGREEMENT, ids=[a[0] for a in AGREEMENT]
)
def test_the_reason_is_the_kernels_own(
    what: str, timecourses: Timecourses, options: NCAOptions
) -> None:
    # the predicate of the host and the predicate of the kernel are two lists
    # which look alike, so they are held to the same answer, word for word: a
    # host predicate which is wider raises at the call site of a user and one
    # which is narrower sends work the kernel carries down the slow path for
    # good, without a sound
    reason = engine.unsupported_reason(timecourses, options)
    kernel = is_supported(engine.batch_dict(timecourses), engine.options_dict(options))
    assert reason == kernel, what


COMBINED: list[tuple[str, Timecourses, NCAOptions]] = [
    (
        "effect curve and units, the kernel's first",
        batch(),
        NCAOptions(
            kind=Kind.EFFECT, effect_threshold=1.0, units={"auc_last": "mg*min/l"}
        ),
    ),
    (
        "partial areas and units, the kernel's first",
        batch(),
        NCAOptions(
            partial_aucs={"auc_0_4": (0.0, 4.0)}, units={"auc_last": "mg*min/l"}
        ),
    ),
    (
        "a group batch and units, the kernel's first",
        group_batch(),
        NCAOptions(units={"auc_last": "mg*min/l"}),
    ),
    (
        "a multiple dose batch with a threshold of the acceptance",
        multiple_dose_batch(),
        NCAOptions(tau=12.0, acceptance=Acceptance(r2_adj_min=0.9)),
    ),
]


@pytest.mark.parametrize(
    ("what", "timecourses", "options"), COMBINED, ids=[c[0] for c in COMBINED]
)
def test_the_host_agrees_with_the_kernel_on_several_refusals_at_once(
    what: str, timecourses: Timecourses, options: NCAOptions
) -> None:
    # the host holds no order of refusals of its own: an analysis which trips
    # over several is answered with the one the kernel names first, which is
    # compared against the kernel on the real batch and not against a table
    kernel = is_supported(engine.batch_dict(timecourses), engine.options_dict(options))
    assert kernel is not None, what
    assert engine.unsupported_reason(timecourses, options) == kernel, what


def test_the_host_does_not_decide_what_the_kernel_refuses(
    engine_cross_check: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the host predicate forwards to the kernel's `is_supported` and holds no
    # condition of its own about the options, so a refusal added in, or lifted
    # from, a kernel release needs no change here. Asked through the original
    # predicate, since the cross-check of the suite stands in for the name
    # `engine.unsupported_reason` and holds it against the unpatched kernel
    host = engine_cross_check._predicate
    refused = [options for _, options in REFUSED]
    seen: list[tuple[Any, Any]] = []

    def says(answer: str | None) -> Any:
        def stand_in(batch_dict: Any, options_dict: Any) -> str | None:
            seen.append((batch_dict, options_dict))
            return answer

        return stand_in

    # a refusal the kernel invents, for an analysis the host has never heard of
    monkeypatch.setattr(pkpdutils_core, "is_supported", says("a refusal of 9.9"))
    assert host(batch(), NCAOptions()) == "a refusal of 9.9"
    assert len(seen) == 1
    # and every refusal the kernel lifts, which a restating host would still make
    monkeypatch.setattr(pkpdutils_core, "is_supported", says(None))
    for options in refused:
        assert host(batch(), options) is None
    assert host(group_batch(), NCAOptions()) is None
    # an ordinary batch asks the kernel exactly once
    seen.clear()
    assert host(batch(), NCAOptions()) is None
    assert len(seen) == 1


def test_the_host_s_own_answers_neither_ask_nor_convert(
    engine_cross_check: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the forced path, an empty batch and a batch of mixed labels are answered
    # before the kernel is asked and before anything is converted for it
    host = engine_cross_check._predicate
    asked: list[str] = []
    monkeypatch.setattr(
        pkpdutils_core, "is_supported", lambda *args: asked.append("kernel")
    )
    for name in ("batch_dict", "options_dict"):
        real = getattr(engine, name)
        monkeypatch.setattr(
            engine, name, lambda *a, _n=name, _r=real: (asked.append(_n), _r(*a))[1]
        )
    empty = Timecourses.from_arrays(
        TIMES,
        np.zeros((0, TIMES.size)),
        time_unit="hr",
        unit="mg/l",
        dims=("subject",),
        coords={"subject": np.array([], dtype=object)},
    )
    mixed = labelled_batch([1, "a"])
    named = NCAOptions(terminal=TerminalPhase(windows={1: (2.0, 8.0)}))
    assert host(empty, NCAOptions()) == engine.EMPTY_REASON
    assert host(mixed, named) == engine.MIXED_LABELS_REASON
    monkeypatch.setattr(engine, "_force_python", True)
    assert host(batch(), NCAOptions()) == engine.FORCED_REASON
    assert asked == []
    # and the same batches do reach the kernel when the host does not answer
    monkeypatch.setattr(engine, "_force_python", False)
    host(mixed, NCAOptions())
    assert asked == ["batch_dict", "options_dict", "kernel"]


def test_a_conversion_of_another_batch_is_refused() -> None:
    other = engine.KernelInput(group_batch(), NCAOptions())
    same_options = NCAOptions()
    for call in (engine.unsupported_reason, engine.kernel_result, engine.run_kernel):
        with pytest.raises(ValueError, match="another batch"):
            call(batch(), same_options, prepared=other)
    shared = engine.KernelInput(batch(), same_options)
    with pytest.raises(ValueError, match="another batch"):
        engine.unsupported_reason(shared.timecourses, NCAOptions(), prepared=shared)


def test_the_source_of_the_host_predicate_names_no_refusal_of_the_kernel(
    engine_cross_check: Any,
) -> None:
    # the structural half of the same claim: nothing of the function body reads
    # the option or the spread of the batch a refusal of the kernel is about.
    # The only conditions it holds are the three the host answers itself
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(engine_cross_check._predicate)))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    body = function.body[1:]  # without the docstring
    read = {
        node.attr
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Attribute)
    }
    strings = {
        node.value
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert strings == set()
    assert read.isdisjoint(
        {"kind", "acceptance", "partial_aucs", "units", "sd", "se", "uncertainty"}
    ), read
    assert read <= {"terminal", "windows", "n_samples", "is_supported", "dicts"}, read


def test_the_forced_python_path_is_no_refusal_of_the_kernel() -> None:
    forced = engine._force_python
    engine._force_python = True
    try:
        reason = engine.unsupported_reason(batch(), NCAOptions())
    finally:
        engine._force_python = forced
    assert reason == engine.FORCED_REASON
    assert (
        is_supported(engine.batch_dict(batch()), engine.options_dict(NCAOptions()))
        is None
    )


# --- the conversion -----------------------------------------------------------


def test_string_coordinates_survive_the_conversion() -> None:
    # a label the conversion drops or turns into a number is a sample the
    # kernel cannot find the terminal window of
    one = batch()
    assert engine.batch_dict(one)["coords"] == [
        {"name": "subject", "dims": ["subject"], "values": {"str": ["A", "B"]}}
    ]
    numbered = Timecourses.from_timecourses(
        [curve(), curve(2.0)], dim="subject", labels=[1, 2]
    )
    assert engine.batch_dict(numbered)["coords"] == [
        {"name": "subject", "dims": ["subject"], "values": {"int": [1, 2]}}
    ]


def test_the_coordinates_of_a_study_travel_with_their_dtype() -> None:
    values = C0 * np.exp(-K * TIMES)
    crossover = Timecourses.from_arrays(
        TIMES,
        np.stack([values, 1.1 * values]),
        time_unit="hr",
        unit="mg/l",
        dims=("subject",),
        coords={
            "subject": ["S1", "S2"],
            "period": ("subject", [1, 2]),
            "weight": ("subject", [70.5, 80.0]),
            "fasted": ("subject", [True, False]),
        },
        dose=IV_DOSE,
    )
    coords = {
        record["name"]: record["values"]
        for record in engine.batch_dict(crossover)["coords"]
    }
    assert coords == {
        "subject": {"str": ["S1", "S2"]},
        "period": {"int": [1, 2]},
        "weight": {"float": [70.5, 80.0]},
        "fasted": {"bool": [True, False]},
    }


SCALAR_LABELS: list[tuple[str, list[Any], dict[str, list[Any]], Any]] = [
    ("int64", [np.int64(1), np.int64(2)], {"int": [1, 2]}, 1),
    ("bool_", [np.bool_(True), np.bool_(False)], {"bool": [True, False]}, True),
]


@pytest.mark.parametrize(
    ("what", "labels", "sent", "key"), SCALAR_LABELS, ids=[s[0] for s in SCALAR_LABELS]
)
def test_a_coordinate_of_numpy_scalars_reads_as_its_values(
    what: str, labels: list[Any], sent: dict[str, list[Any]], key: Any
) -> None:
    # a coordinate given as an object array keeps its dtype, and the objects
    # in it are numpy scalars: `np.int64` is no instance of `int` and
    # `np.bool_` none of `bool`, so a conversion which reads the python type
    # alone sends those labels as their text while `encode_window_key` sends
    # the key of a window as a number or a truth value, and the kernel refuses
    # a window which names a sample of the batch. `np.float64` and `np.str_`
    # need no unwrapping, they are a `float` and a `str`
    values = C0 * np.exp(-K * TIMES)
    labelled = Timecourses.from_arrays(
        TIMES,
        np.stack([values, 1.1 * values]),
        time_unit="hr",
        unit="mg/l",
        dims=("subject",),
        coords={"subject": np.array(labels, dtype=object)},
        dose=IV_DOSE,
    )
    assert labelled.ds["subject"].dtype == np.dtype("O")
    coords = {
        record["name"]: record["values"]
        for record in engine.batch_dict(labelled)["coords"]
    }
    assert coords == {"subject": sent}, what
    options = NCAOptions(terminal=TerminalPhase(windows={key: (0.5, 4.0)}))
    result = nca(labelled, options=options)
    assert result.terminal_windows()[key] == (0.5, 4.0)
    assert_same_result(result, python_result(labelled, options))
    assert_same_mapping(labelled, options)


def labelled_batch(labels: Sequence[Any]) -> Timecourses:
    """Two curves labelled by an object coordinate of the given values."""
    values = C0 * np.exp(-K * TIMES)
    return Timecourses.from_arrays(
        TIMES,
        np.stack([values, 1.1 * values]),
        time_unit="hr",
        unit="mg/l",
        dims=("subject",),
        coords={"subject": np.array(labels, dtype=object)},
        dose=IV_DOSE,
    )


MIXED_LABELS: list[tuple[str, list[Any], Any]] = [
    ("int and str", [1, "a"], 1),
    ("int and float", [1, 2.5], 2.5),
    ("bool and str", [True, "a"], True),
]


@pytest.mark.parametrize(
    ("what", "labels", "key"), MIXED_LABELS, ids=[m[0] for m in MIXED_LABELS]
)
def test_a_window_on_labels_of_several_types_is_the_host_s_own_refusal(
    what: str, labels: list[Any], key: Any
) -> None:
    # a coordinate travels under one type for the whole column, where
    # `encode_window_key` keeps the type of every key: the kernel then holds
    # `Str("1")` against `Int(1)`, finds no sample of that name and refuses the
    # batch with a plain `ValueError` - no `UnsupportedError`, so no fallback -
    # where the released library matches the label and answers. Both shapes are
    # what pandas hands over for a column of mixed identifiers
    labelled = labelled_batch(labels)
    options = NCAOptions(terminal=TerminalPhase(windows={key: (2.0, 8.0)}))
    # the kernel's own predicate accepts the batch, which is why the refusal is
    # the host's own and not the kernel's sentence; and the refusal of its
    # analysis is pinned here, so that the day the kernel carries such a column
    # this goes red and the reason is removed rather than kept forever
    assert is_supported(engine.batch_dict(labelled), engine.options_dict(options)) is (
        None
    )
    with pytest.raises(ValueError, match="names no sample") as refusal:
        engine.kernel_result(labelled, options)
    assert not isinstance(refusal.value, engine.UnsupportedError), (
        "an `UnsupportedError` would be answered by the fallback of `nca` and "
        "would need no reason of the host at all"
    )
    assert engine.unsupported_reason(labelled, options) == engine.MIXED_LABELS_REASON
    # and the analysis is the answer of the python core rather than that
    # exception: the window is on the row its key names
    result = nca(labelled, options=options)
    row = labels.index(key)
    assert result.ds["lambda_z_t_first"].to_numpy()[row] == pytest.approx(2.0)
    assert result.ds["lambda_z_t_last"].to_numpy()[row] == pytest.approx(8.0)
    assert_same_result(result, python_result(labelled, options))


CARRIED_LABELS: list[tuple[str, list[Any], dict[Any, tuple[float, float]] | None]] = [
    ("several types and no window", [1, "a"], None),
    ("several types and the default window alone", [1, "a"], {"*": (2.0, 8.0)}),
    ("one type and a window", [1, 2], {1: (2.0, 8.0)}),
    (
        "dates and a window",
        [date(2024, 1, 1), date(2024, 1, 2)],
        {date(2024, 1, 1): (2.0, 8.0)},
    ),
]


@pytest.mark.parametrize(
    ("what", "labels", "windows"), CARRIED_LABELS, ids=[c[0] for c in CARRIED_LABELS]
)
def test_the_kernel_keeps_every_batch_whose_labels_survive_the_crossing(
    what: str, labels: list[Any], windows: dict[Any, tuple[float, float]] | None
) -> None:
    # the refusal above is as narrow as the defect, which is the comparison of
    # a key against a column that lost its type: labels are read for nothing
    # else, so a batch of several types without a window and one whose only key
    # is the default name no label at all, and a date travels as the isoformat
    # text of `_label_text` on one side and of `encode_window_key` on the
    # other, so it meets its label either way. A refusal wider than the defect
    # is a permanent slowdown nothing would report
    labelled = labelled_batch(labels)
    options = NCAOptions(
        terminal=TerminalPhase(windows=windows) if windows else TerminalPhase()
    )
    assert engine.unsupported_reason(labelled, options) is None, what
    assert_same_result(nca(labelled, options=options), python_result(labelled, options))
    assert_same_mapping(labelled, options)


def test_a_covariate_of_several_types_is_no_refusal() -> None:
    # the label of a sample is the value of its dimension coordinate and
    # nothing else (`pkpdutils.nca.nca.sample_keys`), and a window keyed by
    # anything else is refused by the host before either engine is reached, so
    # a covariate of several types names no sample and costs the batch nothing
    values = C0 * np.exp(-K * TIMES)
    labelled = Timecourses.from_arrays(
        TIMES,
        np.stack([values, 1.1 * values]),
        time_unit="hr",
        unit="mg/l",
        dims=("subject",),
        coords={
            "subject": ["S1", "S2"],
            "weight": ("subject", np.array([70, "not reported"], dtype=object)),
        },
        dose=IV_DOSE,
    )
    options = NCAOptions(terminal=TerminalPhase(windows={"S1": (2.0, 8.0)}))
    assert engine.mixed_label_dim(labelled) is None
    assert engine.unsupported_reason(labelled, options) is None
    assert_same_result(nca(labelled, options=options), python_result(labelled, options))
    assert_same_mapping(labelled, options)


def test_the_batch_carries_every_field_of_the_contract() -> None:
    one = batch()
    sent = engine.batch_dict(one)
    assert sent["sample_dims"] == ["subject"]
    assert sent["sample_shape"] == [2]
    assert sent["n_time"] == TIMES.size
    assert sent["times"].shape == (2, TIMES.size)
    assert sent["times"].dtype == np.float64
    assert sent["times"].flags["C_CONTIGUOUS"]
    assert sent["values"] == pytest.approx(one.values)
    assert sent["max_doses"] == 1
    assert sent["dose_amount"] == pytest.approx(np.full((2, 1), 100.0))
    assert sent["dose_time"] == pytest.approx(np.zeros((2, 1)))
    assert np.isnan(sent["dose_duration"]).all()
    assert (sent["time_unit"], sent["unit"], sent["dose_unit"]) == ("hr", "mg/l", "mg")
    assert sent["route"] == "iv_bolus"
    assert sent["substance"] == "x"
    assert (sent["sd"], sent["se"], sent["n"], sent["nominal_time"]) == (None,) * 4
    assert "contract" not in sent


def test_a_batch_without_doses_carries_empty_dose_arrays() -> None:
    values = C0 * np.exp(-K * TIMES)
    without = Timecourses.from_arrays(
        TIMES, values[None, :], time_unit="hr", unit="mg/l", dims=("subject",)
    )
    sent = engine.batch_dict(without)
    assert sent["max_doses"] == 0
    for name in ("dose_amount", "dose_time", "dose_duration"):
        assert sent[name].shape == (1, 0), name
    assert sent["dose_unit"] is None
    assert sent["route"] is None
    assert engine.unsupported_reason(without, NCAOptions()) is None


def test_a_batch_of_two_routes_travels_per_sample() -> None:
    # the route decides whether a row reports `cl` or `cl_f`, `c0` and the
    # value at the dose time, so a batch whose route is lost is analysed as
    # another study
    mixed = Timecourses.from_timecourses(
        [curve(), curve(dose=ORAL_DOSE)], dim="arm", labels=["iv", "oral"]
    )
    assert mixed.routes is not None
    assert engine.unsupported_reason(mixed, NCAOptions()) is None
    sent = engine.batch_dict(mixed)
    assert sent["route"] is None
    assert sent["routes"] == ["iv_bolus", "oral"]
    result = nca(mixed)
    assert np.isfinite(result.ds["cl"].to_numpy()[0])
    assert np.isfinite(result.ds["cl_f"].to_numpy()[1])
    # the kernel analyses the batch in one call and reports the union of the
    # variables of the two routes in its own order, where the python core runs
    # one group per route and appends what the second adds; `_python_order`
    # sorts the kernel's answer into python's, so a result is the same mapping
    # whichever engine ran it, down to the order of its variables
    assert_same_result(result, python_result(mixed))
    assert_same_mapping(mixed)


# the one value per field of `NCAOptions` the cache key has to tell apart, by
# the name of the field; a field added to the options has to be added here as
# well, which `test_the_key_of_the_order_cache_separates_every_option` asserts
OPTION_VARIANTS: dict[str, Any] = {
    "kind": Kind.EFFECT,
    "auc_method": AUCMethod.LOG,
    "terminal": TerminalPhase(min_points=4),
    "lloq": 0.05,
    "lloq_relative": 0.1,
    "blq": BLQHandling.ZERO_BEFORE_TMAX,
    "c0_method": C0Method.FIRST_VALUE,
    "extrapolation_warning": 0.3,
    "acceptance": Acceptance(exclude=True),
    "partial_aucs": {"auc_0_4": (0.0, 4.0)},
    "tau": 12.0,
    "tau_tolerance": 0.2,
    "intervals": False,
    "units": {"auc_last": "mg*min/l"},
    "effect_threshold": 1.0,
    "n_workers": 2,
    "chunk_rows": 1000,
    "uncertainty": UncertaintyMethod.DELTA,
    "n_boot": 100,
    "seed": 7,
    "ci_level": 0.9,
    "bootstrap_spread": BootstrapSpread.SD,
    "bootstrap_distribution": BootstrapDistribution.LOGNORMAL,
    "delta_step": 0.02,
}


def test_the_key_of_the_order_cache_separates_every_option() -> None:
    # the key carries `repr(options)`, so two option sets which differ in any
    # one field cannot collide; a key which named a few fields would serve the
    # order of one analysis for another
    assert set(OPTION_VARIANTS) == set(NCAOptions.model_fields)
    routes = (Route.IV_BOLUS, Route.ORAL)
    keys = {engine._order_key(routes, 1, NCAOptions())}
    for name, value in OPTION_VARIANTS.items():
        options = NCAOptions(**{name: value})
        assert options != NCAOptions(), name
        keys.add(engine._order_key(routes, 1, options))
    assert len(keys) == len(OPTION_VARIANTS) + 1


def test_the_order_cache_tells_three_shapes_apart() -> None:
    # the three shapes which really report different variables: the routes
    # decide whether `c0` and the back extrapolation appear (a bolus against
    # an infusion) and a batch without a protocol reports no dose dependent
    # variable at all. Every one of them is compared against the python core
    # with the cache warm from the other two, so a key which dropped the
    # routes or the protocol would serve one order for another shape
    bolus_oral = Timecourses.from_timecourses(
        [curve(), curve(dose=ORAL_DOSE)], dim="arm", labels=["a", "b"]
    )
    infusion_oral = Timecourses.from_timecourses(
        [curve(dose=INFUSION_DOSE), curve(dose=ORAL_DOSE)], dim="arm", labels=["a", "b"]
    )
    values = C0 * np.exp(-K * TIMES)
    undosed = Timecourses.from_arrays(
        TIMES,
        np.stack([values, 1.1 * values]),
        time_unit="hr",
        unit="mg/l",
        dims=("arm",),
        coords={"arm": ["a", "b"], "route": ("arm", ["iv_bolus", "oral"])},
    )
    shapes = [bolus_oral, infusion_oral, undosed]
    orders = {tuple(engine.run_kernel(one, NCAOptions())) for one in shapes}
    assert len(orders) == 3
    # every one of them against the python core with the cache warm from the
    # other two, in both directions: an order of a shape with more variables
    # filtered down to a shape with fewer can come out right by luck, where
    # the other direction appends what it does not know at the end
    for index, one in enumerate(shapes):
        engine._ORDER_CACHE.clear()
        for other in shapes[:index] + shapes[index + 1 :]:
            engine.run_kernel(other, NCAOptions())
        assert_same_mapping(one)


def test_the_order_of_a_shape_is_read_from_the_cache() -> None:
    # the dry run costs about a millisecond, which a batch of two arms would
    # pay on every call: it is paid once per shape
    core = importlib.import_module("pkpdutils.nca.nca")

    mixed = Timecourses.from_timecourses(
        [curve(), curve(dose=ORAL_DOSE)], dim="arm", labels=["a", "b"]
    )
    routes = mixed.routes
    assert routes is not None
    runs = 0
    real = core.run_rows

    def counted(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        nonlocal runs
        runs += 1
        return real(*args, **kwargs)

    engine._ORDER_CACHE.clear()
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(core, "run_rows", counted)
        first = engine._python_order(
            routes, NCAOptions(), [], dosed=mixed.n_dose, n_time=mixed.n_time
        )
        assert runs == 2, "one dry run per route of the batch"
        second = engine._python_order(
            routes, NCAOptions(), [], dosed=mixed.n_dose, n_time=mixed.n_time
        )
        assert runs == 2, "the second call reads the order from the cache"
    assert first == second == []


def test_the_order_of_a_mixed_batch_is_the_order_of_the_merge() -> None:
    # `merge_rows` orders the variables of a batch of several routes after the
    # group which reports the most of them and not after the row order of the
    # groups, so a batch whose first row is the oral one still reports the
    # variables of the bolus group first. The batch above cannot tell the two
    # rules apart, its bolus row being both the first and the larger group:
    # this one has them disagree
    oral_first = Timecourses.from_timecourses(
        [curve(dose=ORAL_DOSE), curve()], dim="arm", labels=["oral", "iv"]
    )
    assert oral_first.routes is not None
    assert list(oral_first.routes) == [Route.ORAL, Route.IV_BOLUS]
    assert_same_mapping(oral_first)
    assert_same_result(nca(oral_first), python_result(oral_first))


def test_the_limit_of_quantification_of_a_sample_travels() -> None:
    limited = Timecourses.from_timecourses(
        [
            Timecourse(
                time=TIMES,
                value=C0 * np.exp(-K * TIMES),
                time_unit="hr",
                unit="mg/l",
                dose=IV_DOSE,
                substance="x",
                lloq=1.0,
            ),
            curve(),
        ],
        dim="subject",
        labels=["A", "B"],
    )
    sent = engine.batch_dict(limited)
    assert sent["lloq"] == pytest.approx([1.0, np.nan], nan_ok=True)
    assert engine.unsupported_reason(limited, NCAOptions()) is None
    result = nca(limited)
    # the limit truncates the curve of the first subject and not of the second
    assert result.ds["tlast"].to_numpy()[0] < result.ds["tlast"].to_numpy()[1]
    assert_same_result(result, python_result(limited))
    assert_same_mapping(limited)


def test_a_sliced_batch_is_converted_and_not_reinterpreted() -> None:
    # a view and a float32 array are no buffer of doubles in C order
    every_other = batch().ds.isel(time=slice(None, None, 2))
    sliced = Timecourses(every_other)
    sent = engine.batch_dict(sliced)
    assert sent["values"].flags["C_CONTIGUOUS"]
    assert sent["values"] == pytest.approx(sliced.values)
    assert engine.unsupported_reason(sliced, NCAOptions()) is None
    assert_same_result(nca(sliced), python_result(sliced))
    assert_same_mapping(sliced)


def test_the_options_carry_every_field_and_the_typed_windows() -> None:
    options = NCAOptions(terminal=TerminalPhase(windows={"A": (2.0, 12.0)}))
    sent = engine.options_dict(options)
    assert set(sent) == set(NCAOptions.model_fields)
    assert "_nca_options_version" not in sent
    assert sent["kind"] == "concentration"
    assert sent["terminal"]["method"] == "best_fit"
    assert sent["terminal"]["windows"] == [
        {"key": {"type": "str", "value": "A"}, "window": (2.0, 12.0)}
    ]


def test_a_missing_kernel_fails_plainly() -> None:
    # the reload resets every attribute of the module and with it the count of
    # the differential gate, which the sentinel of tests/conftest.py reads as a
    # floor: an armed run whose last comparison happened before this test would
    # otherwise report that it compared nothing, so the count is carried over
    # the reload rather than left where the reload put it
    compared = engine.comparisons()
    try:
        with pytest.MonkeyPatch.context() as patched:
            patched.setitem(sys.modules, "pkpdutils_core", None)
            with pytest.raises(ImportError, match="pkpdutils-core"):
                importlib.reload(engine)
    finally:
        importlib.reload(engine)
        for _ in range(compared):
            engine.note_comparison()
    assert engine.comparisons() == compared
    assert engine.unsupported_reason(batch(), NCAOptions()) is None


# --- the dispatch -------------------------------------------------------------


def test_an_eligible_batch_gives_the_same_result() -> None:
    one = batch()
    assert engine.unsupported_reason(one, NCAOptions()) is None
    assert_same_result(nca(one), python_result(one))


@pytest.mark.parametrize(
    ("what", "timecourses", "options"), AGREEMENT, ids=[a[0] for a in AGREEMENT]
)
def test_the_two_engines_agree_on_every_analysis(
    what: str, timecourses: Timecourses, options: NCAOptions
) -> None:
    if options.resolve_uncertainty(timecourses.has_uncertainty) is not (
        UncertaintyMethod.NONE
    ):
        # a bootstrap draws its replicates and the delta method needs a spread
        # the batch of this case does not carry; both run in the python core
        pytest.skip("the propagation of an uncertainty is the python core's own")
    assert_same_result(
        nca(timecourses, options=options), python_result(timecourses, options)
    )
    if engine.unsupported_reason(timecourses, options) is None:
        # the comparison of the two results above passes whether or not the
        # kernel ran, the fallback being silent; this one is the kernel itself
        assert_same_mapping(timecourses, options)


def test_a_normalized_unit_is_converted_once() -> None:
    # the kernel converts a volume to `liter` and a clearance to `liter/hour`
    # itself, as `_to_result` does, so a column which arrived converted and
    # was converted again would be off by that factor
    nanomolar = Timecourses.from_arrays(
        TIMES,
        (C0 * np.exp(-K * TIMES))[None, :],
        time_unit="min",
        unit="nmol/l",
        dims=("subject",),
        dose={"amount": [1.0], "unit": "nmol"},
        route=Route.IV_BOLUS,
    )
    assert engine.unsupported_reason(nanomolar, NCAOptions()) is None
    result = nca(nanomolar)
    assert result.ds["vz"].attrs["units"] == "liter"
    assert_same_result(result, python_result(nanomolar))
    # the comparison which sees the kernel's own columns: a result compared
    # with a result would pass with the factor applied twice if the batch had
    # never reached the kernel at all
    assert_same_mapping(nanomolar)


def mixed_dose_batch(
    route: Route = Route.IV_BOLUS,
    *,
    last: float = 11.5,
    at_dose: bool = False,
    duration: float | None = None,
) -> Timecourses:
    """Four subjects of 1, 3, 1 and 3 doses twelve hours apart, in one batch.

    `last` is the last offset sampled in every interval (a value short of the
    interval completes by the terminal regression or is flagged) and `at_dose`
    adds a sample at every dose time, which a bolus carries post-dose.
    """
    protocols = (1, 3, 1, 3)
    offsets = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, last])
    if at_dose:
        offsets = np.concatenate([[0.0], offsets])
    curves = []
    for n in protocols:
        t = np.concatenate([offsets + 12.0 * i for i in range(n)])
        v = sum(
            np.where(t >= 12.0 * i, C0 * np.exp(-K * (t - 12.0 * i)), 0.0)
            for i in range(n)
        )
        curves.append((t, np.asarray(v)))
    n_time = max(t.size for t, _ in curves)
    times = np.full((4, n_time), np.nan)
    values = np.full((4, n_time), np.nan)
    amounts = np.full((4, 3), np.nan)
    dose_times = np.full((4, 3), np.nan)
    for i, ((t, v), n) in enumerate(zip(curves, protocols, strict=True)):
        times[i, : t.size] = t
        values[i, : t.size] = v
        amounts[i, :n] = 100.0
        dose_times[i, :n] = 12.0 * np.arange(n)
    dose: dict[str, Any] = {"amount": amounts, "unit": "mg", "time": dose_times}
    if duration is not None:
        dose["duration"] = np.where(np.isfinite(amounts), duration, np.nan)
    return Timecourses.from_arrays(
        times,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("subject",),
        coords={"subject": list("ABCD")},
        dose=dose,
        route=route,
    )


def irregular_dose_batch() -> Timecourses:
    """A subject of doses 0, 24, 30, 42 h of unequal amounts beside a regular one.

    The gaps of the first are 24, 6 and 12 h, so that its last gap differs from
    the maximum (24), the mean (14), the first gap (24) and the span over the
    number of gaps (14) and `tau` discriminates every other rule than "the last
    gap". The amounts differ within and across the rows.
    """
    protocols = [
        (np.array([0.0, 24.0, 30.0, 42.0]), np.array([100.0, 50.0, 100.0, 75.0])),
        (np.array([0.0, 12.0, 24.0]), np.array([80.0, 80.0, 80.0])),
    ]
    ends = [54.0, 36.0]
    rows = []
    for (dose_t, amount), end in zip(protocols, ends, strict=True):
        t = np.unique(
            np.concatenate(
                [
                    dose_t[i] + np.array([0.25, 1.0, 2.0, 4.0, 8.0])
                    for i in range(dose_t.size)
                ]
                + [dose_t[1:] - 0.5, [end - 0.5]]
            )
        )
        v = sum(
            np.where(t >= d, a / 10.0 * np.exp(-K * (t - d)), 0.0)
            for d, a in zip(dose_t, amount, strict=True)
        )
        rows.append((t, np.asarray(v)))
    n_time = max(t.size for t, _ in rows)
    times = np.full((2, n_time), np.nan)
    values = np.full((2, n_time), np.nan)
    amounts = np.full((2, 4), np.nan)
    dose_times = np.full((2, 4), np.nan)
    for i, ((t, v), (dose_t, amount)) in enumerate(zip(rows, protocols, strict=True)):
        times[i, : t.size] = t
        values[i, : t.size] = v
        amounts[i, : amount.size] = amount
        dose_times[i, : dose_t.size] = dose_t
    return Timecourses.from_arrays(
        times,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("subject",),
        coords={"subject": ["A", "B"]},
        dose={"amount": amounts, "unit": "mg", "time": dose_times},
        route=Route.IV_BOLUS,
    )


def test_the_tau_of_an_irregular_protocol_is_its_last_gap_in_both_engines() -> None:
    timecourses = irregular_dose_batch()
    options = NCAOptions()
    assert engine.unsupported_reason(timecourses, options) is None
    python = python_result(timecourses, options)
    assert python.ds["n_doses"].to_numpy().tolist() == [4, 3]
    assert python.ds["tau"].to_numpy().tolist() == [12.0, 12.0]
    assert_same_mapping(timecourses, options)
    assert_same_result(nca(timecourses, options=options), python)


def mixed_route_dose_batch(
    protocols: Sequence[tuple[Route, int]] = (
        (Route.ORAL, 3),
        (Route.IV_BOLUS, 3),
        (Route.ORAL, 1),
        (Route.IV_BOLUS, 4),
    ),
) -> Timecourses:
    """Oral with 3 doses, bolus with 3, oral with 1 and bolus with 4, in one batch."""
    curves = []
    for route, n in protocols:
        offsets = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 11.5])
        t = np.concatenate([offsets + 12.0 * i for i in range(n)])
        v = sum(
            np.where(t >= 12.0 * i, C0 * np.exp(-K * (t - 12.0 * i)), 0.0)
            for i in range(n)
        )
        curves.append(
            Timecourse(
                time=t,
                value=np.asarray(v),
                time_unit="hr",
                unit="mg/l",
                dosing=Dosing.regimen(
                    Dose(amount=100.0, unit="mg", route=route), 12.0, n
                ),
                substance="x",
            )
        )
    return Timecourses.from_timecourses(curves, dim="subject", labels=list("ABCD"))


MIXED_ROUTE_BATCH = mixed_route_dose_batch()
# a route which has no multiple dose row at all, first and last: the group which
# reports the steady state variables is the other one, and the merge of the
# groups follows which that is
ONE_ROUTE_SINGLE_DOSE = mixed_route_dose_batch(
    ((Route.ORAL, 1), (Route.ORAL, 1), (Route.IV_BOLUS, 3), (Route.IV_BOLUS, 3))
)
ONE_ROUTE_SINGLE_DOSE_LAST = mixed_route_dose_batch(
    ((Route.IV_BOLUS, 3), (Route.IV_BOLUS, 3), (Route.ORAL, 1), (Route.ORAL, 1))
)


MULTIPLE_DOSE_CASES: list[tuple[str, Timecourses, NCAOptions]] = [
    ("bolus", mixed_dose_batch(), NCAOptions()),
    ("bolus without intervals", mixed_dose_batch(), NCAOptions(intervals=False)),
    ("bolus a given tau", mixed_dose_batch(), NCAOptions(tau=12.0)),
    ("bolus at the dose times", mixed_dose_batch(at_dose=True), NCAOptions()),
    ("oral", mixed_dose_batch(Route.ORAL), NCAOptions()),
    (
        "infusion",
        mixed_dose_batch(Route.IV_INFUSION, duration=0.5),
        NCAOptions(),
    ),
    # the routes differ per sample, so the python core runs one group per route
    # and the kernel's variables have to be put in its order
    ("mixed routes", MIXED_ROUTE_BATCH, NCAOptions()),
    ("mixed routes a given tau", MIXED_ROUTE_BATCH, NCAOptions(tau=12.0)),
    # `tau` puts every row of every route on the multiple dose path
    (
        "mixed routes, single dose rows with a given tau",
        ONE_ROUTE_SINGLE_DOSE,
        NCAOptions(tau=12.0),
    ),
    # a last sample short of the interval end, completed by the regression
    ("a last interval within the tolerance", mixed_dose_batch(last=11.0), NCAOptions()),
    # and one too short for it, which is flagged
    ("an incomplete last interval", mixed_dose_batch(last=8.0), NCAOptions()),
]


@pytest.mark.parametrize(
    ("what", "timecourses", "options"),
    MULTIPLE_DOSE_CASES,
    ids=[c[0] for c in MULTIPLE_DOSE_CASES],
)
def test_a_multiple_dose_batch_gives_the_same_result_from_either_engine(
    what: str, timecourses: Timecourses, options: NCAOptions
) -> None:
    # the differential gate covers this for every batch the suite sends through
    # nca(), but this test names the case directly: every interval_* variable,
    # every steady state variable and the flags agree within rel=1e-6
    assert engine.unsupported_reason(timecourses, options) is None, what
    python = python_result(timecourses, options)
    # the layer below `nca` as well, since the result of `nca` alone would
    # compare the python core with itself if the kernel had refused the batch
    assert_same_mapping(timecourses, options)
    assert_same_result(nca(timecourses, options=options), python)
    # the numbers of the python core, exactly where they are counted
    counts = np.isfinite(engine.dose_arrays(timecourses)[0]).sum(axis=1).tolist()
    assert python.ds["n_doses"].to_numpy().tolist() == counts
    tau = python.ds["tau"].to_numpy()
    if options.tau is None:
        single = [n == 1 for n in counts]
        assert np.isnan(tau[single]).all()
        assert (tau[[not one for one in single]] == 12.0).all()
    else:
        # a given interval makes every row a multiple dose row
        assert tau.tolist() == [12.0] * 4
    assert ("interval_auc" in python.ds.data_vars) == options.intervals


@pytest.mark.parametrize(
    "timecourses", [ONE_ROUTE_SINGLE_DOSE, ONE_ROUTE_SINGLE_DOSE_LAST]
)
def test_routes_which_differ_in_their_dosing_take_the_python_path(
    timecourses: Timecourses, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the kernel reports one union of variables, the python core reports them
    # per route, and the two differ in a variable (`cl_ss_f`), not in order
    assert (
        engine.unsupported_reason(timecourses, NCAOptions())
        == engine.MIXED_ROUTE_DOSING_REASON
    )
    kernel = set(engine.run_kernel(timecourses, NCAOptions()))
    assert kernel - set(python_result(timecourses).ds.data_vars) == {"cl_ss_f"}
    # so `nca` answers from the python core, armed or not
    monkeypatch.setenv(engine.COMPARE_VARIABLE, "1")
    before = engine.comparisons()
    assert_same_result(nca(timecourses), python_result(timecourses))
    assert engine.comparisons() == before


def test_the_order_cache_tells_a_multiple_dose_batch_from_a_single_dose_one() -> None:
    # the same routes in the same order and the same options, once with no
    # multiple dose row and once with one in every route: the cached order of
    # the first must not be served for the second
    single = mixed_route_dose_batch(
        ((Route.ORAL, 1), (Route.IV_BOLUS, 1), (Route.ORAL, 1), (Route.IV_BOLUS, 1))
    )
    multiple = mixed_route_dose_batch(
        ((Route.ORAL, 3), (Route.IV_BOLUS, 3), (Route.ORAL, 3), (Route.IV_BOLUS, 3))
    )
    engine._ORDER_CACHE.clear()
    assert_same_mapping(single)
    assert_same_mapping(multiple)
    engine._ORDER_CACHE.clear()
    assert_same_mapping(multiple)
    assert_same_mapping(single)


def test_the_single_dose_rows_of_a_mixed_batch_have_no_interval_values() -> None:
    result = nca(mixed_dose_batch())
    for name in result.ds.data_vars:
        if str(name).startswith("interval_") or name in {"auc_tau", "cl_ss", "tau"}:
            column = result.ds[name].to_numpy()
            single = column[[0, 2]]
            assert np.isnan(single).all(), name
            if name != "tau":
                assert np.isfinite(column[[1, 3]]).any(), name
    assert result.ds["interval_auc"].dims == ("subject", "interval")


def test_the_flags_of_a_multiple_dose_batch_are_raised_by_the_kernel() -> None:
    # the flags are read from the kernel alone (`run_kernel`), and are raised
    # whether or not the intervals are part of the result
    for intervals in (True, False):
        options = NCAOptions(intervals=intervals)
        short = engine.run_kernel(mixed_dose_batch(last=8.0), options)["flags"]
        assert (short & int(NCAFlag.INCOMPLETE_INTERVAL)).tolist() == [0, 256, 0, 256]
        post = engine.run_kernel(mixed_dose_batch(at_dose=True), options)["flags"]
        assert (post & int(NCAFlag.EXTRAPOLATED_TROUGH)).tolist() == [0, 512, 0, 512]
        clean = engine.run_kernel(mixed_dose_batch(), options)["flags"]
        assert (clean & 768).tolist() == [0, 0, 0, 0]


def test_an_ineligible_batch_takes_the_python_path() -> None:
    # the acceptance criteria are an analysis the kernel does not carry
    options = NCAOptions(acceptance=Acceptance(span_min=3.0))
    assert engine.unsupported_reason(batch(), options) is not None
    result = nca(batch(), options=options)
    assert_same_result(result, python_result(batch(), options))


def test_a_zero_sample_batch_matches() -> None:
    # the kernel of 0.2.0 derives the variables of a result from the routes of
    # the rows it ran and reports one column for a batch with no rows, where
    # python reports the empty column of every variable; the host answers an
    # empty batch itself, so a result does not lose its variables
    empty = Timecourses.from_arrays(
        TIMES,
        np.zeros((0, TIMES.size)),
        time_unit="hr",
        unit="mg/l",
        dims=("subject",),
        coords={"subject": np.array([], dtype=object)},
    )
    assert empty.n_samples == 0
    assert engine.unsupported_reason(empty, NCAOptions()) == engine.EMPTY_REASON
    assert (
        is_supported(engine.batch_dict(empty), engine.options_dict(NCAOptions()))
        is None
    )
    # the answer of the kernel this refusal exists for, pinned rather than
    # described: `flags` alone, where python reports the empty column of every
    # variable of the analysis. The day the kernel reports them this goes red,
    # and `EMPTY_REASON` is removed together with it; a test which only
    # compared the two results would pass either way, since the host sends
    # this batch to the python core and would compare its answer with itself
    assert list(engine.run_kernel(empty, NCAOptions())) == ["flags"]
    result = nca(empty)
    assert result.ds.sizes["subject"] == 0
    assert "auc_last" in result.ds.data_vars
    assert_same_result(result, python_result(empty))


def test_nca_single_uses_the_kernel() -> None:
    one = curve()
    assert (
        nca_single(one)
        .to_dataframe()
        .equals(
            python_result(one.to_batch(dim="_single"))
            .ds.isel(_single=0)
            .drop_vars("_single")
            .pipe(NCAResult)
            .to_dataframe()
        )
    )


def test_the_kernel_is_what_runs_an_eligible_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the one test which fails if the dispatch never happens: a result which is
    # right proves nothing about which engine computed it
    def refuse(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        raise AssertionError("the kernel was not called")

    monkeypatch.setattr(engine, "run_kernel", refuse)
    with pytest.raises(AssertionError, match="the kernel was not called"):
        nca(batch())


def test_an_error_which_is_no_refusal_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `UnsupportedError` is a `ValueError`, and an `except ValueError` here
    # would answer a disagreement about the contract - an option the kernel
    # does not map, a batch it cannot read - by taking the python path for
    # good, with nothing to notice it by
    def mismatch(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        raise ValueError("the kernel and the host disagree")

    monkeypatch.setattr(engine, "run_kernel", mismatch)
    with pytest.raises(ValueError, match="disagree"):
        nca(batch())


@pytest.mark.engine_fallback
def test_a_refusal_of_the_kernel_falls_back_to_the_python_path(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    engine_cross_check: Any,
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        raise engine.UnsupportedError("an analysis of a later build")

    one = batch()
    expected = python_result(one)
    monkeypatch.setattr(engine, "run_kernel", refuse)
    with caplog.at_level("WARNING", logger="pkpdutils.nca.nca"):
        result = nca(one)
    assert_same_result(result, expected)
    assert any(
        "an analysis of a later build" in record.message for record in caplog.records
    )
    # the only test of the suite which makes the kernel refuse a batch on
    # purpose, and therefore the one which proves that the watch of
    # `tests/conftest.py` sees a real fallback: every other test fails if this
    # record is logged, and the marker is what exempts this one
    assert [record.getMessage() for record in engine_cross_check.fallbacks] == [
        record.getMessage() for record in caplog.records if "refused" in record.message
    ]
    assert engine_cross_check.fallbacks


def test_the_forced_python_path_reaches_the_python_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        raise AssertionError("the kernel ran a batch of the python path")

    monkeypatch.setattr(engine, "run_kernel", refuse)
    monkeypatch.setattr(engine, "_force_python", True)
    assert np.isfinite(nca(batch()).ds["auc_last"].to_numpy()).all()


def test_the_mapping_of_the_kernel_is_the_mapping_of_the_rows() -> None:
    assert_same_mapping(batch())


# --- the differential gate ----------------------------------------------------


def two_mappings() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Two equal mappings of one analysis, as the two engines report them."""
    python: dict[str, np.ndarray] = {
        "cmax": np.array([8.0, 4.0]),
        "tmax": np.array([0.25, 0.5]),
        "flags": np.array([0, 2], dtype=np.int64),
    }
    return python, {name: column.copy() for name, column in python.items()}


# the unit of every variable of `two_mappings` both engines name; `flags` is
# named by neither, as the kernel of 0.2.0 reports none for it
MAPPING_UNITS: dict[str, str] = {"cmax": "milligram / liter", "tmax": "hour"}


def disagreements(
    python: dict[str, np.ndarray],
    kernel: dict[str, np.ndarray],
    *,
    units_python: dict[str, str] | None = None,
    units_kernel: dict[str, str] | None = None,
) -> list[str]:
    """`compare_engines` of two mappings, the units of both engines by default."""
    return engine.compare_engines(
        python,
        kernel,
        units_python=MAPPING_UNITS if units_python is None else units_python,
        units_kernel=MAPPING_UNITS if units_kernel is None else units_kernel,
    )


def test_agreement_is_reported_as_empty() -> None:
    python, kernel = two_mappings()
    assert disagreements(python, kernel) == []


def test_a_value_difference_is_reported() -> None:
    python, kernel = two_mappings()
    kernel["cmax"][1] += 1e-3
    (said,) = disagreements(python, kernel)
    # the variable and both numbers, so that a mismatch is read without a
    # second run
    assert "'cmax'" in said
    assert "4.0" in said
    assert "4.001" in said


def test_a_difference_inside_the_tolerance_is_no_disagreement() -> None:
    python, kernel = two_mappings()
    # the last bits of `x * f / f`, which `_as_computed` divides the unit
    # factor back out with
    kernel["cmax"][1] *= 1.0 + 1e-9
    assert disagreements(python, kernel) == []


def test_a_nan_position_difference_is_reported() -> None:
    python, kernel = two_mappings()
    kernel["tmax"][0] = np.nan
    (said,) = disagreements(python, kernel)
    assert "'tmax'" in said
    assert "NaN" in said
    assert "0.25" in said


def test_a_missing_variable_is_reported() -> None:
    python, kernel = two_mappings()
    del kernel["tmax"]
    (said,) = disagreements(python, kernel)
    assert "kernel reports none" in said
    assert "tmax" in said


def test_a_variable_of_the_kernel_alone_is_reported() -> None:
    python, kernel = two_mappings()
    del python["tmax"]
    (said,) = disagreements(python, kernel)
    assert "python core reports none" in said
    assert "tmax" in said


def test_a_unit_difference_is_reported() -> None:
    python, kernel = two_mappings()
    (said,) = disagreements(
        python, kernel, units_kernel={**MAPPING_UNITS, "cmax": "mg/l"}
    )
    assert "'cmax'" in said
    assert "'milligram / liter'" in said
    assert "'mg/l'" in said


def test_a_unit_only_one_engine_names_is_reported() -> None:
    python, kernel = two_mappings()
    (said,) = disagreements(python, kernel, units_kernel={"cmax": "milligram / liter"})
    assert "'tmax'" in said
    assert "None" in said


def test_a_flag_difference_is_reported() -> None:
    python, kernel = two_mappings()
    kernel["flags"][1] = 3
    (said,) = disagreements(python, kernel)
    assert "'flags'" in said
    assert "counted values" in said
    assert "2" in said
    assert "3" in said


def test_a_flag_word_differing_in_a_high_bit_is_reported() -> None:
    python, kernel = two_mappings()
    python["flags"] = np.array([2**40, 0], dtype=np.int64)
    kernel["flags"] = np.array([2**40 + 1, 0], dtype=np.int64)
    # the reason a counted variable is compared exactly: the relative
    # tolerance of the values would call these two bit fields equal
    assert np.isclose(
        float(2**40), float(2**40 + 1), rtol=engine.COMPARE_REL, atol=engine.COMPARE_ABS
    )
    (said,) = disagreements(python, kernel)
    assert "'flags'" in said
    assert "1099511627776" in said
    assert "1099511627777" in said


def test_an_order_difference_is_reported() -> None:
    python, kernel = two_mappings()
    reordered = {name: kernel[name] for name in reversed(list(kernel))}
    (said,) = disagreements(python, reordered)
    # the order of the variables is the order of the columns of
    # `NCAResult.to_dataframe`, so a swap is observable to a caller
    assert "different order" in said
    assert "'cmax'" in said
    assert "'flags'" in said


def test_a_shape_difference_is_reported() -> None:
    python, kernel = two_mappings()
    kernel["cmax"] = np.array([8.0, 4.0, 1.0])
    (said,) = disagreements(python, kernel)
    assert "'cmax'" in said
    assert "shape" in said


def test_a_dtype_difference_is_reported() -> None:
    python, kernel = two_mappings()
    # the same numbers, counted as doubles: `to_dataframe` reports the column
    # of a flag word as a float and `NCAResult.flag_table` reads bits of it
    kernel["flags"] = kernel["flags"].astype(np.float64)
    (said,) = disagreements(python, kernel)
    assert "'flags'" in said
    assert "dtype" in said


def test_a_mismatch_is_no_value_error() -> None:
    # `nca` answers a `ValueError` of the kernel by taking the python path; a
    # disagreement of the two engines must reach whoever armed the gate
    assert not issubclass(engine.EngineMismatch, ValueError)


@pytest.mark.parametrize(
    ("value", "armed"),
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("yes", True),
        ("on", True),
        ("", False),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
    ],
)
def test_the_gate_is_armed_by_the_environment(
    monkeypatch: pytest.MonkeyPatch, value: str, armed: bool
) -> None:
    monkeypatch.setenv(engine.COMPARE_VARIABLE, value)
    assert engine.comparing() is armed


def test_the_gate_is_disarmed_without_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(engine.COMPARE_VARIABLE, raising=False)
    assert engine.comparing() is False


def changing_kernel(
    change: Any,
) -> Any:
    """A `run_kernel` which runs the kernel and then changes its answer."""
    real = engine.run_kernel

    def run(
        timecourses: Timecourses,
        options: NCAOptions,
        *,
        units: dict[str, str] | None = None,
        prepared: engine.KernelInput | None = None,
    ) -> dict[str, np.ndarray]:
        values = real(timecourses, options, units=units, prepared=prepared)
        return change(values, units)

    return run


def scaled_cmax(
    values: dict[str, np.ndarray], units: dict[str, str] | None
) -> dict[str, np.ndarray]:
    """The answer of the kernel with `cmax` off by a thousandth."""
    values["cmax"] = values["cmax"] * 1.001
    return values


def test_the_armed_gate_raises_on_a_value_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(engine.COMPARE_VARIABLE, "1")
    monkeypatch.setattr(engine, "run_kernel", changing_kernel(scaled_cmax))
    with pytest.raises(engine.EngineMismatch, match="'cmax'"):
        nca(batch())


def test_the_gate_lets_a_difference_through_when_it_is_not_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the gate costs an ordinary analysis one lookup of the environment and
    # compares nothing, which is what makes this test the proof that the gate
    # above is what raises
    monkeypatch.delenv(engine.COMPARE_VARIABLE, raising=False)
    monkeypatch.setattr(engine, "run_kernel", changing_kernel(scaled_cmax))
    one = batch()
    expected = python_result(one).ds["cmax"].to_numpy() * 1.001
    assert nca(one).ds["cmax"].to_numpy() == pytest.approx(expected)


def test_the_armed_gate_raises_on_a_different_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reversed_order(
        values: dict[str, np.ndarray], units: dict[str, str] | None
    ) -> dict[str, np.ndarray]:
        return {name: values[name] for name in reversed(list(values))}

    monkeypatch.setenv(engine.COMPARE_VARIABLE, "1")
    monkeypatch.setattr(engine, "run_kernel", changing_kernel(reversed_order))
    with pytest.raises(engine.EngineMismatch, match="different order"):
        nca(batch())


def test_the_armed_gate_raises_on_a_unit_of_the_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def shortened_unit(
        values: dict[str, np.ndarray], units: dict[str, str] | None
    ) -> dict[str, np.ndarray]:
        assert units is not None, "the armed gate collects the units of the kernel"
        units["cmax"] = "mg/l"
        return values

    monkeypatch.setenv(engine.COMPARE_VARIABLE, "1")
    monkeypatch.setattr(engine, "run_kernel", changing_kernel(shortened_unit))
    # the numbers of the two engines agree and their units do not, which is the
    # defect class of a kernel computing a parameter in another unit than the
    # host believes
    with pytest.raises(engine.EngineMismatch, match="mg/l"):
        nca(batch())


def test_the_armed_gate_names_the_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(engine.COMPARE_VARIABLE, "1")
    monkeypatch.setattr(engine, "run_kernel", changing_kernel(scaled_cmax))
    with pytest.raises(engine.EngineMismatch, match="2 sample") as raised:
        nca(batch())
    assert "mg/l" in str(raised.value)
    assert "iv_bolus" in str(raised.value)


@pytest.mark.parametrize(
    ("what", "timecourses", "options"), AGREEMENT, ids=[a[0] for a in AGREEMENT]
)
def test_the_armed_gate_agrees_on_every_analysis(
    monkeypatch: pytest.MonkeyPatch,
    what: str,
    timecourses: Timecourses,
    options: NCAOptions,
) -> None:
    if options.resolve_uncertainty(timecourses.has_uncertainty) is not (
        UncertaintyMethod.NONE
    ):
        # a bootstrap draws its replicates and the delta method needs a spread
        # the batch of this case does not carry; both run in the python core
        pytest.skip("the propagation of an uncertainty is the python core's own")
    # the gate itself as the comparison, on the analyses of `AGREEMENT`: it
    # raises on a name, an order, a shape, a dtype, a NaN position, a number or
    # a unit the two engines do not share
    monkeypatch.setenv(engine.COMPARE_VARIABLE, "1")
    assert engine.comparing()
    nca(timecourses, options=options)


def test_the_armed_gate_compares_nothing_on_a_batch_of_the_python_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(engine.COMPARE_VARIABLE, "1")
    options = NCAOptions(acceptance=Acceptance(span_min=3.0))
    assert engine.unsupported_reason(batch(), options) is not None
    before = engine.comparisons()
    nca(batch(), options=options)
    assert engine.comparisons() == before


def test_the_armed_gate_compares_a_multiple_dose_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(engine.COMPARE_VARIABLE, "1")
    many = multiple_dose_batch()
    assert engine.unsupported_reason(many, NCAOptions()) is None
    before = engine.comparisons()
    assert np.isfinite(nca(many).ds["cl_ss"].to_numpy()).all()
    assert engine.comparisons() == before + 1


def blq_batch() -> Timecourses:
    """One row entirely below its limit of quantification, beside a measured one."""
    return Timecourses.from_timecourses(
        [
            Timecourse(
                time=TIMES,
                value=C0 * np.exp(-K * TIMES),
                time_unit="hr",
                unit="mg/l",
                dose=IV_DOSE,
                substance="x",
                lloq=20.0,
            ),
            curve(),
        ],
        dim="subject",
        labels=["blq", "measured"],
    )


def test_an_all_blq_row_agrees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(engine.COMPARE_VARIABLE, "1")
    assert engine.comparing()
    below = blq_batch()
    # the gate compares the two engines on this batch, the positions of NaN
    # exactly
    result = nca(below)
    for name in ("cmax", "tmax", "auc_last", "lambda_z", "clast", "cl"):
        values = result.ds[name].to_numpy()
        assert np.isnan(values[0]), name
        assert np.isfinite(values[1]), name
    assert_same_mapping(below)
    assert_same_result(result, python_result(below))


def test_the_kernel_reports_the_units_the_host_derives() -> None:
    # the units are the one part of the kernel's answer the host neither adopts
    # nor drops: a result is labelled by pint either way, so a kernel which
    # computed a parameter in another unit would be a wrong number with a right
    # label. They are therefore compared, which is only worth anything while
    # the two spell a unit the same way
    units: dict[str, str] = {}
    nanomolar = Timecourses.from_arrays(
        TIMES,
        (C0 * np.exp(-K * TIMES))[None, :],
        time_unit="min",
        unit="nmol/l",
        dims=("subject",),
        dose={"amount": [1.0], "unit": "nmol"},
        route=Route.IV_BOLUS,
    )
    engine.run_kernel(nanomolar, NCAOptions(), units=units)
    assert len(units) > 40
    assert engine.host_units(units, nanomolar) == units


def test_the_kernel_reports_a_unit_for_every_variable_but_the_flags() -> None:
    units: dict[str, str] = {}
    options = NCAOptions(terminal=TerminalPhase(keep_candidates=True))
    values = engine.run_kernel(batch(), options, units=units)
    # the exemption of the unit comparison is pinned to what the kernel really
    # does: the day it reports a unit for its flags this goes red and
    # `KERNEL_UNITLESS` is removed, and the day it stops reporting one for a
    # parameter the gate reports that parameter rather than passing
    assert set(values) - set(units) == engine.KERNEL_UNITLESS


# --- the flag which requires the gate -----------------------------------------

ROOT = Path(__file__).resolve().parents[2]
GATE_FLAG = "--require-engine-gate"


def collect_with(*args: str, armed: bool) -> subprocess.CompletedProcess[str]:
    """Collect one file of the suite in a child session, armed or not.

    The flag is the one thing continuous integration says on the command line
    and the variable the one thing it says through the environment, so the two
    are given to a real session here rather than asserted about the hook: a
    flag which stopped being read, or a session which stopped refusing to run
    without the variable, is what the `engines` job would otherwise report as
    the agreement of two engines it never compared.
    """
    # the variable the gate reads is the one thing this controls, and the
    # `PYTEST_*` of the running session are dropped so that a child session is
    # a session of its own and not a second worker of this one
    env = {
        name: value
        for name, value in os.environ.items()
        if name != engine.COMPARE_VARIABLE and not name.startswith("PYTEST_")
    }
    if armed:
        env[engine.COMPARE_VARIABLE] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "-q",
            "--collect-only",
            "-p",
            "no:cacheprovider",
            "tests/test_units.py",
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=False,
    )


def test_the_flag_refuses_a_session_which_is_not_armed() -> None:
    refused = collect_with(GATE_FLAG, armed=False)
    assert refused.returncode != 0
    assert GATE_FLAG in refused.stderr
    assert "the gate is not armed" in refused.stderr
    assert engine.COMPARE_VARIABLE in refused.stderr
    # nothing ran: the session is refused before it collects a test
    assert "test session starts" not in refused.stdout


def test_the_flag_is_satisfied_by_the_variable() -> None:
    armed = collect_with(GATE_FLAG, armed=True)
    assert armed.returncode == 0, armed.stderr
    assert "the gate is not armed" not in armed.stderr


def test_a_misspelled_flag_is_no_silent_pass() -> None:
    # the lesson of the step this flag replaced: a check which can fail for a
    # reason of its own is no check, and a check which can pass for one is
    # worse. A flag can only be read or be unknown
    misspelled = collect_with(GATE_FLAG + "s", armed=True)
    assert misspelled.returncode != 0
    assert "unrecognized arguments" in misspelled.stderr


def test_a_session_without_the_flag_needs_no_variable() -> None:
    # the ordinary local run, which the flag must leave alone
    plain = collect_with(armed=False)
    assert plain.returncode == 0, plain.stderr


# --- the cross-check of the two predicates ------------------------------------


def test_the_cross_check_sees_every_batch_of_a_test(
    engine_cross_check: Any,
) -> None:
    # the hook of `tests/conftest.py` is armed for every test of the suite and
    # asks both predicates about every batch which reaches `nca`; a test which
    # trusted it without looking would pass under a hook which was never armed
    before = engine_cross_check.compared
    nca(batch())
    nca(multiple_dose_batch())
    assert engine_cross_check.compared == before + 2


def test_the_cross_check_counts_the_batches_of_the_three_host_reasons(
    engine_cross_check: Any,
) -> None:
    empty = Timecourses.from_arrays(
        TIMES,
        np.zeros((0, TIMES.size)),
        time_unit="hr",
        unit="mg/l",
        dims=("subject",),
        coords={"subject": np.array([], dtype=object)},
    )
    before = engine_cross_check.compared, engine_cross_check.empty
    nca(empty)
    # a batch of no samples is the host's own refusal and no question of the
    # kernel, which is read from the batch and not from the answer
    assert (engine_cross_check.compared, engine_cross_check.empty) == (
        before[0],
        before[1] + 1,
    )
    forced = engine_cross_check.forced
    python_result(batch())
    assert engine_cross_check.forced == forced + 1
    # and the third, a window given by hand on labels of several types, which
    # the kernel's predicate accepts and its analysis refuses
    mixed = engine_cross_check.mixed
    nca(
        labelled_batch([1, "a"]),
        options=NCAOptions(terminal=TerminalPhase(windows={1: (2.0, 8.0)})),
    )
    assert engine_cross_check.mixed == mixed + 1


def test_the_cross_check_catches_a_predicate_which_refuses_too_much(
    monkeypatch: pytest.MonkeyPatch, engine_cross_check: Any
) -> None:
    def narrow(
        timecourses: Timecourses, options: NCAOptions, **kwargs: Any
    ) -> str | None:
        # the mutation the review measured: a predicate which refuses every
        # batch whose unit is not `mg/l` leaves this file green and the
        # analyses of those batches never reach the kernel again
        return "a unit the kernel does not carry"

    monkeypatch.setattr(engine_cross_check, "_predicate", narrow)
    with pytest.raises(AssertionError, match="the two predicates disagree"):
        nca(batch())


def test_the_cross_check_catches_a_predicate_which_refuses_too_little(
    monkeypatch: pytest.MonkeyPatch, engine_cross_check: Any
) -> None:
    def wide(
        timecourses: Timecourses, options: NCAOptions, **kwargs: Any
    ) -> str | None:
        return None

    monkeypatch.setattr(engine_cross_check, "_predicate", wide)
    with pytest.raises(AssertionError, match="the two predicates disagree"):
        # an analysis the kernel does not carry: the predicate of the host is
        # the only thing between it and an `UnsupportedError` at the call site
        # of a user
        nca(batch(), options=NCAOptions(acceptance=Acceptance(span_min=3.0)))


def unit_batch(*, time_unit: str = "hr", unit: str = "mg/l") -> Timecourses:
    """Two curves in the given time and concentration units."""
    values = C0 * np.exp(-K * TIMES)
    return Timecourses.from_arrays(
        TIMES,
        np.stack([values, 1.1 * values]),
        time_unit=time_unit,
        unit=unit,
        dims=("subject",),
        coords={"subject": ["A", "B"]},
        dose=IV_DOSE,
    )


@pytest.mark.engine_fallback
@pytest.mark.parametrize(
    ("time_unit", "unit"),
    [("week", "mg/l"), ("hr", "U/l")],
    ids=["week", "U/l"],
)
def test_a_unit_only_pint_reads_falls_back_instead_of_raising(
    time_unit: str, unit: str, engine_cross_check: Any
) -> None:
    # the kernel's registry is narrower than pint's and 0.2.0 reported such a
    # unit as an error of the batch, which `nca` does not catch. Its predicate
    # stays silent on purpose (answering needs a copy of the registry in the
    # kernel and in the host, which the cross-check would have to mirror), so
    # the batch takes the fallback and the marker says that is expected
    one = unit_batch(time_unit=time_unit, unit=unit)
    result = nca(one)
    assert_same_result(result, python_result(one))
    assert engine_cross_check.refusals, "the kernel was expected to refuse"
    assert engine_cross_check.fallbacks


NUMERIC_LABELS: list[tuple[str, list[Any], Any]] = [
    ("float label and int key", [1.0, 2.0], 1),
    ("int label and float key", [1, 2], 1.0),
    ("bool label and int key", [True, False], 1),
]


@pytest.mark.parametrize(
    ("what", "labels", "key"), NUMERIC_LABELS, ids=[n[0] for n in NUMERIC_LABELS]
)
def test_a_window_key_of_another_numeric_type_matches_on_the_kernel(
    what: str,
    labels: list[Any],
    key: Any,
    monkeypatch: pytest.MonkeyPatch,
    engine_cross_check: Any,
) -> None:
    # python matches `1 == 1.0 == True`; 0.2.0 found no sample of that name and
    # raised, 0.2.1 matches and runs the batch. No marker: the batch must run on
    # the kernel, so the kernel is spied on and a quiet fallback fails the test
    values = C0 * np.exp(-K * TIMES)
    labelled = Timecourses.from_arrays(
        TIMES,
        np.stack([values, 1.1 * values]),
        time_unit="hr",
        unit="mg/l",
        dims=("subject",),
        coords={"subject": np.array(labels)},
        dose=IV_DOSE,
    )
    options = NCAOptions(terminal=TerminalPhase(windows={key: (2.0, 8.0)}))
    expected = python_result(labelled, options)
    calls: list[dict[str, np.ndarray]] = []
    kernel = engine.run_kernel

    def spy(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        calls.append(kernel(*args, **kwargs))
        return calls[-1]

    monkeypatch.setattr(engine, "run_kernel", spy)
    result = nca(labelled, options=options)
    assert len(calls) == 1, what
    assert not engine_cross_check.refusals, what
    assert not engine_cross_check.fallbacks, what
    assert result.terminal_windows()[key] == (2.0, 8.0)
    row = 0
    assert result.ds["lambda_z_t_first"].to_numpy()[row] == pytest.approx(2.0)
    assert_same_result(result, expected)


def test_the_python_core_interpolates_outside_its_points_without_a_warning() -> None:
    # a row whose last interval is not covered queries a time beyond its points
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        python_result(mixed_dose_batch(last=8.0))
