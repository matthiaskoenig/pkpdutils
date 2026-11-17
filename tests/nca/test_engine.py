import importlib
import sys
from collections.abc import Sequence
from datetime import date
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
    ) -> dict[str, np.ndarray]:
        values = real(timecourses, options, units=units)
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
    many = multiple_dose_batch()
    assert engine.unsupported_reason(many, NCAOptions()) is not None
    assert np.isfinite(nca(many).ds["cl_ss"].to_numpy()).all()


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
    def narrow(timecourses: Timecourses, options: NCAOptions) -> str | None:
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
    def wide(timecourses: Timecourses, options: NCAOptions) -> str | None:
        return None

    monkeypatch.setattr(engine_cross_check, "_predicate", wide)
    with pytest.raises(AssertionError, match="the two predicates disagree"):
        # a multiple dose analysis the kernel does not carry: the predicate of
        # the host is the only thing between it and an `UnsupportedError` at
        # the call site of a user
        nca(multiple_dose_batch())


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
