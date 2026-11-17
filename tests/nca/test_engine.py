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


def assert_same_result(
    kernel: NCAResult, python: NCAResult, *, ordered: bool = True
) -> None:
    """Every variable of two results, their units and their flags.

    `ordered` is `False` for a batch of several routes, the one analysis whose
    variables the two engines report in a different order (`engine`).
    """
    if ordered:
        assert list(kernel.ds.data_vars) == list(python.ds.data_vars)
    else:
        assert set(kernel.ds.data_vars) == set(python.ds.data_vars)
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
    assert reason


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
    sent = engine.batch_dict(mixed)
    assert sent["route"] is None
    assert sent["routes"] == ["iv_bolus", "oral"]
    result = nca(mixed)
    assert np.isfinite(result.ds["cl"].to_numpy()[0])
    assert np.isfinite(result.ds["cl_f"].to_numpy()[1])
    # the two engines report the same variables with the same numbers and in a
    # different order, the one difference of the dispatch: the python core runs
    # one group per route and appends what the second route adds, the kernel
    # reports the union in its canonical order, which does not depend on which
    # route the batch names first
    expected = python_result(mixed)
    assert_same_result(result, expected, ordered=False)
    assert list(result.ds.data_vars) != list(expected.ds.data_vars)
    # the order of a batch of one route is the order of the python core, so
    # the difference stays what it is said to be
    assert_same_result(nca(batch()), python_result(batch()))


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
    result = nca(limited)
    # the limit truncates the curve of the first subject and not of the second
    assert result.ds["tlast"].to_numpy()[0] < result.ds["tlast"].to_numpy()[1]
    assert_same_result(result, python_result(limited))


def test_a_sliced_batch_is_converted_and_not_reinterpreted() -> None:
    # a view and a float32 array are no buffer of doubles in C order
    every_other = batch().ds.isel(time=slice(None, None, 2))
    sliced = Timecourses(every_other)
    sent = engine.batch_dict(sliced)
    assert sent["values"].flags["C_CONTIGUOUS"]
    assert sent["values"] == pytest.approx(sliced.values)
    assert_same_result(nca(sliced), python_result(sliced))


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
    result = nca(nanomolar)
    assert result.ds["vz"].attrs["units"] == "liter"
    assert_same_result(result, python_result(nanomolar))


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
    from pkpdutils.nca.nca import run_analysis_rows

    one = batch()
    n_rows, n_time = one.n_samples, one.n_time
    options = NCAOptions()
    rows = run_analysis_rows(
        one.times.reshape(n_rows, n_time),
        one.values.reshape(n_rows, n_time),
        dose_amount=one.dose_amount,
        dose_time=one.dose_time,
        dose_duration=one.dose_duration,
        route=one.route,
        options=options,
    )
    kernel = engine.run_kernel(one, options)
    assert list(kernel) == list(rows)
    for name, column in rows.items():
        assert kernel[name].shape == column.shape, name
        assert kernel[name].dtype == column.dtype, name
        assert kernel[name] == pytest.approx(column, nan_ok=True, rel=REL, abs=ABS), (
            name
        )
