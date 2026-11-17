import importlib
import sys
from typing import Any

import numpy as np
import pkpdutils_core
import pytest

from pkpdutils import Dose, Dosing, Route, Timecourse, Timecourses
from pkpdutils.nca import (
    Acceptance,
    Kind,
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


def test_a_multiple_dose_batch_is_not() -> None:
    reason = engine.unsupported_reason(multiple_dose_batch(), NCAOptions())
    assert reason is not None
    assert "more than one dose" in reason


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
    ("tau", NCAOptions(tau=12.0)),
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


def test_a_coordinate_of_numpy_scalars_reads_as_its_numbers() -> None:
    # a coordinate given as an object array keeps its dtype, and the objects
    # in it are numpy scalars: `np.int64` is no instance of `int`, so a
    # conversion which reads the python type alone sends those labels as their
    # text while `encode_window_key` sends the key of a window as a number,
    # and the kernel refuses a window which names a sample of the batch
    values = C0 * np.exp(-K * TIMES)
    labelled = Timecourses.from_arrays(
        TIMES,
        np.stack([values, 1.1 * values]),
        time_unit="hr",
        unit="mg/l",
        dims=("subject",),
        coords={"subject": np.array([np.int64(1), np.int64(2)], dtype=object)},
        dose=IV_DOSE,
    )
    assert labelled.ds["subject"].dtype == np.dtype("O")
    coords = {
        record["name"]: record["values"]
        for record in engine.batch_dict(labelled)["coords"]
    }
    assert coords == {"subject": {"int": [1, 2]}}
    options = NCAOptions(terminal=TerminalPhase(windows={1: (0.5, 4.0)}))
    result = nca(labelled, options=options)
    assert result.terminal_windows()[1] == (0.5, 4.0)
    assert_same_result(result, python_result(labelled, options))
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
    try:
        with pytest.MonkeyPatch.context() as patched:
            patched.setitem(sys.modules, "pkpdutils_core", None)
            with pytest.raises(ImportError, match="pkpdutils-core"):
                importlib.reload(engine)
    finally:
        importlib.reload(engine)
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


def test_an_ineligible_batch_takes_the_python_path() -> None:
    result = nca(multiple_dose_batch())
    assert "cl_ss" in result.ds.data_vars
    assert np.isfinite(result.ds["cl_ss"].to_numpy()).all()


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


def test_a_refusal_of_the_kernel_falls_back_to_the_python_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
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
