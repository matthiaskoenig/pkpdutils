"""Urinary excretion analysis against the closed forms of a mono-exponential rate.

Two fixtures, because two different things are exact. `collected` holds the
amounts a subject with the excretion rate `A k exp(-k t)` really voids, the
integral of the rate over every collection interval, so the sum of the amounts
is the analytic cumulative amount and the renal clearance is the analytic one.
`midpoint_rates` holds amounts whose *rate* is the instantaneous rate at the
midpoint of the interval, so the rate curve is an exact exponential and
`lambda_z`, `aurc_last` and `aurc_inf_obs` have closed forms. The average rate
over an interval is the midpoint rate times `sinh(k d / 2) / (k d / 2)`, a
factor of the interval length alone, which is why `lambda_z` is exact in both
fixtures as long as the intervals have the same length.
"""

from typing import Any

import numpy as np
import pytest

from pkpdutils import (
    AUCMethod,
    Dose,
    NCAOptions,
    Route,
    Timecourse,
    nca_single,
)
from pkpdutils.nca import engine
from pkpdutils.nca.result import NCAResult
from pkpdutils.nca.urine import Excretion, nca_urine

#: total amount excreted over an infinite collection, in mg
AMOUNT = 80.0

#: elimination rate constant, in 1/hr
K = 0.25

#: the collection intervals, all of the same length
EDGES = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0])

#: volume of every collection, in ml
VOLUME = np.array([220.0, 180.0, 260.0, 310.0, 250.0, 400.0])

#: the dose the collections follow
DOSE = Dose(amount=100.0, unit="mg", route=Route.IV_BOLUS)


def collected() -> Excretion:
    """The amounts the analytic rate `A k exp(-k t)` voids over the intervals."""
    start, end = EDGES[:-1], EDGES[1:]
    amount = AMOUNT * (np.exp(-K * start) - np.exp(-K * end))
    return Excretion(
        start=start,
        end=end,
        amount=amount,
        volume=VOLUME,
        unit="mg",
        time_unit="hr",
        volume_unit="ml",
        dose=DOSE,
        substance="drug",
        label="S1",
    )


def midpoint_rates() -> Excretion:
    """Amounts whose rate is the instantaneous rate at the midpoint of the interval."""
    start, end = EDGES[:-1], EDGES[1:]
    midpoint = 0.5 * (start + end)
    amount = AMOUNT * K * np.exp(-K * midpoint) * (end - start)
    return Excretion(
        start=start,
        end=end,
        amount=amount,
        volume=VOLUME,
        unit="mg",
        time_unit="hr",
        volume_unit="ml",
        dose=DOSE,
        substance="drug",
    )


def plasma_curve(c0: float = 5.0) -> Timecourse:
    """A mono-exponential plasma curve `c0 exp(-k t)` with the same rate constant."""
    time = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0])
    return Timecourse(
        time=time,
        value=c0 * np.exp(-K * time),
        time_unit="hr",
        unit="mg/l",
        dose=DOSE,
        substance="drug",
    )


def test_excretion_derives_the_amount_from_concentration_and_volume() -> None:
    """The amount of a collection is the concentration times the volume."""
    exc = collected()
    concentration = exc.amount / VOLUME
    derived = Excretion(
        start=EDGES[:-1],
        end=EDGES[1:],
        concentration=concentration,
        volume=VOLUME,
        unit="mg",
        time_unit="hr",
        volume_unit="ml",
    )
    assert derived.amount is not None
    np.testing.assert_allclose(np.asarray(derived.amount), np.asarray(exc.amount))
    np.testing.assert_allclose(np.asarray(exc.concentration), concentration)


def test_excretion_sorts_and_validates_the_intervals() -> None:
    """The intervals are sorted by their start and may neither be empty nor overlap."""
    exc = Excretion(
        start=[4.0, 0.0],
        end=[8.0, 4.0],
        amount=[2.0, 5.0],
        unit="mg",
        time_unit="hr",
    )
    np.testing.assert_allclose(exc.start, [0.0, 4.0])
    np.testing.assert_allclose(np.asarray(exc.amount), [5.0, 2.0])
    np.testing.assert_allclose(exc.midpoint, [2.0, 6.0])
    np.testing.assert_allclose(exc.rate, [5.0 / 4.0, 2.0 / 4.0])

    with pytest.raises(ValueError, match="'end' above 'start'"):
        Excretion(start=[0.0], end=[0.0], amount=[1.0], unit="mg", time_unit="hr")
    with pytest.raises(ValueError, match="may not overlap"):
        Excretion(
            start=[0.0, 2.0],
            end=[4.0, 6.0],
            amount=[1.0, 1.0],
            unit="mg",
            time_unit="hr",
        )
    with pytest.raises(ValueError, match="give 'amount'"):
        Excretion(start=[0.0], end=[4.0], unit="mg", time_unit="hr")
    with pytest.raises(ValueError, match="needs a 'volume_unit'"):
        Excretion(
            start=[0.0],
            end=[4.0],
            amount=[1.0],
            volume=[10.0],
            unit="mg",
            time_unit="hr",
        )
    with pytest.raises(ValueError, match="need one each"):
        Excretion(
            start=[0.0, 4.0],
            end=[4.0, 8.0],
            amount=[1.0],
            unit="mg",
            time_unit="hr",
        )


def test_amount_recovered_is_the_analytic_cumulative_amount(urine_engine: str) -> None:
    """The recovered amount is the sum of the collections, the analytic cumulative amount."""
    exc = collected()
    result = nca_urine(exc)
    analytic = AMOUNT * (1.0 - np.exp(-K * EDGES[-1]))
    assert float(result["amount_recovered"]) == pytest.approx(analytic)
    assert float(result["percent_recovered"]) == pytest.approx(
        100.0 * analytic / DOSE.amount
    )
    # the volumes are reported in liter, the canonical volume of the package
    assert result.units("vol_ur") == "liter"
    assert float(result["vol_ur"]) == pytest.approx(1e-3 * VOLUME.sum())
    assert result.units("amount_recovered") == "milligram"
    assert result.units("percent_recovered") == "percent"


def test_lambda_z_recovers_the_rate_constant(urine_engine: str) -> None:
    """The regression of the log rate against the midpoint recovers `k` and its half-life."""
    for excretion in (collected(), midpoint_rates()):
        result = nca_urine(excretion)
        assert float(result["lambda_z"]) == pytest.approx(K)
        assert float(result["thalf"]) == pytest.approx(np.log(2.0) / K)
        assert result.units("lambda_z") == "1 / hour"


def test_the_areas_of_the_rate_curve_are_the_excreted_amounts(
    urine_engine: str,
) -> None:
    """`aurc_inf_obs` is the total amount and `aurc_last` the amount up to the last midpoint."""
    exc = midpoint_rates()
    result = nca_urine(exc, options=NCAOptions(auc_method=AUCMethod.LOG))
    last = float(exc.midpoint[-1])
    assert float(result["mid_pt_last"]) == pytest.approx(last)
    assert float(result["rate_last"]) == pytest.approx(AMOUNT * K * np.exp(-K * last))
    # the bolus back extrapolation inserts the rate at the dose, so the area
    # runs from 0 and the logarithmic rule integrates the exponential exactly
    assert float(result["aurc_last"]) == pytest.approx(
        AMOUNT * (1.0 - np.exp(-K * last))
    )
    assert float(result["aurc_inf_obs"]) == pytest.approx(AMOUNT)
    assert float(result["aurc_inf_pred"]) == pytest.approx(AMOUNT)
    # every rate is positive, so the area to the last observation is the same
    assert float(result["aurc_all"]) == pytest.approx(float(result["aurc_last"]))
    assert result.units("aurc_last") == "milligram"
    assert result.units("max_rate") == "milligram / hour"


def test_the_peak_of_the_rate_curve(urine_engine: str) -> None:
    """The peak rate is the first collection, whose midpoint is `tmax_rate`."""
    exc = midpoint_rates()
    result = nca_urine(exc)
    assert float(result["max_rate"]) == pytest.approx(float(exc.rate[0]))
    assert float(result["tmax_rate"]) == pytest.approx(float(exc.midpoint[0]))


def test_the_rate_curve_is_a_point_variable_of_the_result(urine_engine: str) -> None:
    """`rate` and `midpoint` are point variables over the collection dimension."""
    exc = collected()
    result = nca_urine(exc)
    assert set(result.point_variables) == {"rate", "midpoint"}
    assert result.ds["rate"].dims == ("collection",)
    np.testing.assert_allclose(result["rate"].to_numpy(), exc.rate)
    np.testing.assert_allclose(result["midpoint"].to_numpy(), exc.midpoint)
    np.testing.assert_array_equal(
        result.ds["collection"].to_numpy(), np.arange(1, exc.n_intervals + 1)
    )
    assert "rate" not in result.to_dataframe().columns


def test_renal_clearance_against_the_analytic_plasma_area(urine_engine: str) -> None:
    """`clr` is the recovered amount over the plasma area of the collection span."""
    exc = collected()
    curve = plasma_curve()
    result = nca_urine(exc, options=NCAOptions(auc_method=AUCMethod.LOG), plasma=curve)
    span = EDGES[-1] - EDGES[0]
    area = 5.0 * (1.0 - np.exp(-K * span)) / K  # mg/l * hr
    amount = AMOUNT * (1.0 - np.exp(-K * span))
    # mg / (mg/l * hr) is a liter per hour, the canonical clearance
    assert result.units("clr") == "liter / hour"
    assert float(result["clr"]) == pytest.approx(amount / area)


def test_renal_clearance_from_a_plasma_result(urine_engine: str) -> None:
    """A plasma `NCAResult` contributes its `auc_last` instead of the partial area."""
    exc = collected()
    plasma = nca_single(plasma_curve(), options=NCAOptions(auc_method=AUCMethod.LOG))
    result = nca_urine(exc, plasma=plasma)
    assert float(result["clr"]) == pytest.approx(
        float(result["amount_recovered"]) / float(plasma["auc_last"])
    )


def test_without_a_dose_there_is_no_percentage_and_without_plasma_no_clearance(
    urine_engine: str,
) -> None:
    """A collection without a dose has no `percent_recovered`, one without plasma no `clr`."""
    start, end = EDGES[:-1], EDGES[1:]
    exc = Excretion(
        start=start,
        end=end,
        amount=AMOUNT * (np.exp(-K * start) - np.exp(-K * end)),
        unit="mg",
        time_unit="hr",
    )
    result = nca_urine(exc)
    assert np.isnan(float(result["percent_recovered"]))
    assert np.isnan(float(result["vol_ur"]))
    assert "clr" not in result.ds.data_vars


def test_a_dose_in_an_incompatible_unit_is_rejected(urine_engine: str) -> None:
    """The recovered amount and the dose have to be comparable."""
    start, end = EDGES[:-1], EDGES[1:]
    exc = Excretion(
        start=start,
        end=end,
        amount=AMOUNT * (np.exp(-K * start) - np.exp(-K * end)),
        unit="mg",
        time_unit="hr",
        dose=Dose(amount=1.0, unit="mmole", route=Route.ORAL),
    )
    with pytest.raises(ValueError, match="cannot be compared"):
        nca_urine(exc)


def placebo(dose_unit: str = "mg") -> Excretion:
    """The collections of a placebo arm: a dose of 0 and nothing excreted."""
    start, end = EDGES[:-1], EDGES[1:]
    return Excretion(
        start=start,
        end=end,
        amount=np.zeros_like(start),
        volume=VOLUME,
        unit="mg",
        time_unit="hr",
        volume_unit="ml",
        dose=Dose(amount=0.0, unit=dose_unit, route=Route.ORAL),
        substance="drug",
        label="placebo",
    )


def test_a_placebo_arm_has_no_percentage_and_is_no_error(urine_engine: str) -> None:
    """A dose of 0 gives `percent_recovered` as `NaN`, as every dose divided parameter."""
    result = nca_urine(placebo())
    assert np.isnan(float(result["percent_recovered"]))
    assert result.units("percent_recovered") == "percent"
    # what was collected is reported as it was measured, dose or no dose
    assert float(result["amount_recovered"]) == 0.0
    assert float(result["vol_ur"]) == pytest.approx(1e-3 * VOLUME.sum())


def test_a_placebo_arm_in_an_incompatible_unit_is_still_rejected(
    urine_engine: str,
) -> None:
    """The dose of 0 does not skip the unit check, which is about the units."""
    with pytest.raises(ValueError, match="cannot be compared"):
        nca_urine(placebo(dose_unit="mmole"))


def test_a_contradicting_amount_concentration_and_volume_is_rejected() -> None:
    """All three given: the amount has to be the concentration times the volume."""
    exc = collected()
    assert exc.concentration is not None
    Excretion(
        start=EDGES[:-1],
        end=EDGES[1:],
        amount=exc.amount,
        concentration=exc.concentration,
        volume=VOLUME,
        unit="mg",
        time_unit="hr",
        volume_unit="ml",
    )
    with pytest.raises(ValueError, match="not 'concentration' times 'volume'"):
        Excretion(
            start=EDGES[:-1],
            end=EDGES[1:],
            amount=exc.amount,
            # the classic mistake: mg/l instead of mg/ml
            concentration=1000.0 * exc.concentration,
            volume=VOLUME,
            unit="mg",
            time_unit="hr",
            volume_unit="ml",
        )


def test_a_plasma_result_reaching_past_the_collections_is_reported(
    urine_engine: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`clr` from a result divides by the whole curve, which is logged when the windows differ."""
    curve = Timecourse(
        time=np.array([0.0, 4.0, 8.0, 12.0, 24.0, 48.0]),
        value=5.0 * np.exp(-K * np.array([0.0, 4.0, 8.0, 12.0, 24.0, 48.0])),
        time_unit="hr",
        unit="mg/l",
        dose=DOSE,
        substance="drug",
    )
    plasma = nca_single(curve, options=NCAOptions(auc_method=AUCMethod.LOG))
    with caplog.at_level("INFO", logger="pkpdutils.nca.urine"):
        nca_urine(collected(), plasma=plasma)
    assert "collections end at 24.0 hr" in caplog.text
    # the curve itself is integrated over the span, so nothing is logged
    caplog.clear()
    with caplog.at_level("INFO", logger="pkpdutils.nca.urine"):
        nca_urine(collected(), plasma=curve)
    assert "collections end at" not in caplog.text


def run_both(
    monkeypatch: pytest.MonkeyPatch,
    excretion: Excretion,
    **kwargs: Any,
) -> tuple[NCAResult, NCAResult]:
    """The urine analysis through the kernel and, forced, through python."""
    kernel = nca_urine(excretion, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_force_python", True)
        python = nca_urine(excretion, **kwargs)
    return kernel, python


def assert_same(kernel: NCAResult, python: NCAResult) -> None:
    """Every variable: names, dtype, NaN positions exact, numbers at 1e-12, units."""
    assert list(kernel.ds.data_vars) == list(python.ds.data_vars)
    for name in map(str, python.ds.data_vars):
        a, b = kernel[name].to_numpy(), python[name].to_numpy()
        assert a.dtype == b.dtype, name
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b), err_msg=name)
        np.testing.assert_allclose(a, b, rtol=1e-12, equal_nan=True, err_msg=name)
        assert kernel.units(name) == python.units(name), name


def test_both_engines_agree_variable_by_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every variable with a dose and a plasma curve: NaN positions exact, numbers at 1e-12."""
    if not hasattr(engine.pkpdutils_core, "nca_urine"):
        pytest.skip("this build of pkpdutils-core carries no `nca_urine`")
    kernel, python = run_both(monkeypatch, collected(), plasma=plasma_curve())
    assert "clr" in python.ds.data_vars
    assert_same(kernel, python)
    # an excretion without a dose, whose route the kernel is sent as oral
    start, end = EDGES[:-1], EDGES[1:]
    nodose = Excretion(
        start=start,
        end=end,
        amount=AMOUNT * (np.exp(-K * start) - np.exp(-K * end)),
        unit="mg",
        time_unit="hr",
    )
    assert_same(*run_both(monkeypatch, nodose))


def test_the_kernel_is_really_used_when_it_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the kernel present and not forced off the call goes through it."""
    if not hasattr(engine.pkpdutils_core, "nca_urine"):
        pytest.skip("this build of pkpdutils-core carries no `nca_urine`")
    calls: list[int] = []
    real = engine.pkpdutils_core.nca_urine
    monkeypatch.setattr(
        engine.pkpdutils_core, "nca_urine", lambda r: (calls.append(1), real(r))[1]
    )
    monkeypatch.setattr(engine, "_force_python", False)
    nca_urine(collected())
    assert calls == [1]


def test_a_forced_python_run_does_not_touch_the_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_force_python` keeps the urine analysis on the python path."""

    def refuse(request: object) -> None:
        raise AssertionError("the kernel was called")

    monkeypatch.setattr(engine.pkpdutils_core, "nca_urine", refuse, raising=False)
    monkeypatch.setattr(engine, "_force_python", True)
    result = nca_urine(collected())
    assert float(result["lambda_z"]) == pytest.approx(K)


def test_the_placebo_arm_agrees_between_the_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dose of 0 and nothing excreted: `percent_recovered` is NaN on both engines."""
    if not hasattr(engine.pkpdutils_core, "nca_urine"):
        pytest.skip("this build of pkpdutils-core carries no `nca_urine`")
    kernel, python = run_both(monkeypatch, placebo())
    assert_same(kernel, python)
    assert np.isnan(float(kernel["percent_recovered"]))


def test_a_unit_the_kernel_cannot_read_takes_the_python_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The refusal as `UnsupportedError` is answered by python, with the forced answer."""
    if not hasattr(engine.pkpdutils_core, "nca_urine"):
        pytest.skip("this build of pkpdutils-core carries no `nca_urine`")
    start, end = EDGES[:-1], EDGES[1:]
    exc = Excretion(
        start=start,
        end=end,
        amount=AMOUNT * (np.exp(-K * start) - np.exp(-K * end)),
        unit="grain",
        time_unit="hr",
    )
    with caplog.at_level("INFO", logger="pkpdutils.nca.urine"):
        kernel, python = run_both(monkeypatch, exc)
    # the refusal really happened, else the test would pass without a fallback
    assert "the urine analysis takes the python path" in caplog.text
    assert kernel.units("amount_recovered") == "grain"
    assert_same(kernel, python)


def test_a_malformed_request_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `ValueError` of the kernel is no refusal and is not answered by python."""

    def malformed(request: object) -> None:
        raise ValueError("malformed")

    monkeypatch.setattr(engine.pkpdutils_core, "nca_urine", malformed, raising=False)
    monkeypatch.setattr(engine, "_force_python", False)
    with pytest.raises(ValueError, match="malformed"):
        nca_urine(collected())


def zero_plasma() -> Timecourse:
    """A plasma curve of zero concentrations, whose area is exactly 0."""
    return Timecourse(
        time=np.array([0.0, 6.0, 12.0, 24.0]),
        value=np.zeros(4),
        time_unit="hr",
        unit="mg/l",
        dose=DOSE,
        substance="drug",
    )


def test_a_zero_plasma_area_gives_an_infinite_clearance(
    urine_engine: str,
) -> None:
    """The recovered amount over an area of 0 is `inf`, not a `ZeroDivisionError`."""
    result = nca_urine(collected(), plasma=zero_plasma())
    assert float(result["clr"]) == np.inf
    assert result.units("clr") == "liter / hour"


def test_nothing_recovered_over_a_zero_plasma_area_is_nan(
    urine_engine: str,
) -> None:
    """0 over 0 is `nan` on both engines."""
    result = nca_urine(placebo(), plasma=zero_plasma())
    assert np.isnan(float(result["clr"]))
