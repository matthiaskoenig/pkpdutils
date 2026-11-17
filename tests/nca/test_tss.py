"""Time to steady state from the troughs of the dosing intervals."""

import numpy as np
import pkpdutils_core
import pytest

from pkpdutils import (
    AUCMethod,
    Dose,
    Dosing,
    NCAOptions,
    Route,
    Timecourse,
    Timecourses,
    nca,
    nca_single,
)
from pkpdutils.nca import engine, superposition, time_to_steady_state
from pkpdutils.nca import tss as tss_module

C0, TAU = 10.0, 12.0
DOSE = Dose(amount=100, unit="mg", route=Route.IV_BOLUS)
LOG = NCAOptions(auc_method=AUCMethod.LOG)


def single_dose(k: float) -> Timecourse:
    t = np.array([0.0, 0.5, 1, 2, 4, 6, 8, 12, 16, 24, 36, 48, 72, 96])
    return Timecourse(
        time=t,
        value=C0 * np.exp(-k * t),
        time_unit="hr",
        unit="mg/l",
        dose=DOSE,
        substance="x",
    )


def superposed(k: float, n_doses: int) -> Timecourse:
    protocol = Dosing.regimen(DOSE, interval=TAU, n_doses=n_doses)
    return superposition(
        single_dose(k),
        protocol,
        options=LOG,
        grid=np.arange(0.0, n_doses * TAU + 1e-9, 0.25),
    )


def test_the_monoexponential_estimate_recovers_the_rate_constant(
    tss_engine: str,
) -> None:
    k = 0.2
    result = nca_single(superposed(k, 10), options=LOG)
    estimate = time_to_steady_state(result)
    # the trough of interval n is c_ss (1 - exp(-k n tau)) exactly, so the fit
    # recovers k and the time to 90 % of the plateau is ln(10) / k
    assert float(estimate.tss) == pytest.approx(np.log(10.0) / k, rel=1e-6)
    assert estimate.method == "monoexponential"
    assert estimate.fraction == 0.9
    assert estimate.tss.attrs["units"] == "hour"
    # the plateau is the trough of the last interval
    assert estimate.c_ss is not None
    assert float(estimate.c_ss) == pytest.approx(
        C0 * np.exp(-k * TAU) / (1 - np.exp(-k * TAU)), rel=1e-6
    )
    assert estimate.c_ss.attrs["units"] == "milligram / liter"


def test_another_fraction_of_the_plateau(tss_engine: str) -> None:
    k = 0.05
    result = nca_single(superposed(k, 16), options=LOG)
    estimate = time_to_steady_state(result, fraction=0.95)
    assert float(estimate.tss) == pytest.approx(-np.log(0.05) / k, rel=1e-6)


def test_the_stepwise_estimate_finds_the_plateau_of_a_noisy_study(
    tss_engine: str,
) -> None:
    k = 0.05
    predicted = superposed(k, 16)
    rng = np.random.default_rng(7)
    noisy = Timecourse(
        time=predicted.time,
        value=predicted.value * rng.normal(1.0, 0.05, predicted.value.size),
        time_unit="hr",
        unit="mg/l",
        dosing=predicted.dosing,
        substance="x",
    )
    result = nca_single(noisy, options=LOG)
    estimate = time_to_steady_state(result, method="stepwise")
    # the troughs rise over the first three intervals and scatter around the
    # plateau afterwards: the trend is no longer significant from 24 h on
    assert float(estimate.tss) == 24.0
    assert estimate.c_ss is None
    assert estimate.method == "stepwise"
    starts = result.intervals()["interval_start"].to_numpy()
    assert float(estimate.tss) in set(starts.tolist())


def test_the_estimate_of_a_batch_and_its_frame(tss_engine: str) -> None:
    batch = Timecourses.from_timecourses(
        [
            superposed(k, 10).model_copy(update={"label": label})
            for k, label in ((0.2, "s1"), (0.1, "s2"))
        ],
        dim="individual",
    )
    estimate = time_to_steady_state(nca(batch, options=LOG))
    assert estimate.tss.dims == ("individual",)
    assert estimate.tss.to_numpy() == pytest.approx(
        [np.log(10.0) / 0.2, np.log(10.0) / 0.1], rel=1e-5
    )
    frame = estimate.to_dataframe()
    # the sample coordinates travel with the estimate, the dose among them
    assert list(frame.columns) == ["individual", "dose_amount", "tss", "c_ss"]
    assert frame["individual"].tolist() == ["s1", "s2"]


def periodic(trough, n_doses: int = 12) -> Timecourse:
    """A bolus regimen whose interval `i` ends at the trough `trough(i + 1)`."""
    times, values = [], []
    for i in range(n_doses):
        for x in (0.0, 1.0, 2.0, 4.0, 8.0, 12.0):
            if x == 0.0 and i > 0:
                continue
            times.append(i * TAU + x)
            if x == 0.0:
                values.append(trough(0))
            elif x == TAU:
                values.append(trough(i + 1))
            else:
                values.append(trough(i) + 10.0 * np.exp(-0.3 * x))
    return Timecourse(
        time=np.array(times),
        value=np.array(values),
        time_unit="hr",
        unit="mg/l",
        dosing=Dosing.regimen(DOSE, interval=TAU, n_doses=n_doses),
        substance="x",
    )


@pytest.mark.parametrize(
    "trough",
    [
        pytest.param(lambda i: 5.0, id="flat"),
        pytest.param(lambda i: 20.0 - i, id="falling"),
        pytest.param(lambda i: 5.0 + 3.0 * i, id="linear"),
    ],
)
def test_troughs_without_a_plateau_give_no_estimate(trough, tss_engine: str) -> None:
    # the optimum in k is unbounded: the cost keeps falling towards k = 0 or infinity, and
    # where the search stops is an accident of its starts, so there is nothing to report
    result = nca_single(periodic(trough))
    estimate = time_to_steady_state(result)
    assert np.isnan(float(estimate.tss))
    assert estimate.c_ss is not None and np.isnan(float(estimate.c_ss))


def test_a_plateau_next_to_a_flat_series_is_still_estimated(tss_engine: str) -> None:
    result = nca_single(periodic(lambda i: 10.0 * (1.0 - np.exp(-0.2 * i))))
    estimate = time_to_steady_state(result)
    assert estimate.c_ss is not None
    assert np.isfinite(float(estimate.tss)) and np.isfinite(float(estimate.c_ss))


def test_a_single_dose_result_has_no_troughs() -> None:
    result = nca_single(single_dose(0.2), options=LOG)
    with pytest.raises(ValueError, match="no per-interval troughs"):
        time_to_steady_state(result)


def test_the_bounds_of_the_fraction_and_the_level() -> None:
    result = nca_single(superposed(0.2, 10), options=LOG)
    with pytest.raises(ValueError, match=r"'fraction' is 1\.0"):
        time_to_steady_state(result, fraction=1.0)
    with pytest.raises(ValueError, match=r"'alpha' is 0\.0"):
        time_to_steady_state(result, method="stepwise", alpha=0.0)


def test_too_few_intervals_give_no_estimate(tss_engine: str) -> None:
    result = nca_single(superposed(0.2, 2), options=LOG)
    assert np.isnan(float(time_to_steady_state(result).tss))
    assert np.isnan(float(time_to_steady_state(result, method="stepwise").tss))


def test_the_frame_of_a_single_curve_result(tss_engine: str) -> None:
    result = nca_single(superposed(0.2, 10), options=LOG)
    frame = time_to_steady_state(result).to_dataframe()
    # a result without sample dimensions is one row; the scalar coordinates of
    # the result travel with it
    assert len(frame) == 1
    assert list(frame.columns) == ["dose_amount", "tss", "c_ss"]
    assert frame["tss"].to_numpy()[0] == pytest.approx(np.log(10.0) / 0.2, rel=1e-6)
    stepwise = time_to_steady_state(result, method="stepwise").to_dataframe()
    assert list(stepwise.columns) == ["dose_amount", "tss"]


def test_a_flat_series_has_no_estimate_whatever_the_starts_of_the_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the python reference reported tss = 0.038, 6e-14, 0.097, 0.101 and 0.028 for these start
    # sets, with a correct plateau: k is unidentifiable and nothing of it was an estimate
    monkeypatch.setattr(engine, "_force_python", True)
    time = np.arange(12.0, 145.0, 12.0)
    value = np.full(time.size, 5.0)
    for starts in [
        (0.5, 2.0, 8.0),
        (0.3, 1.0, 3.0),
        (1.0, 5.0, 20.0),
        (0.1, 4.0, 40.0),
    ]:
        monkeypatch.setattr(tss_module, "_STARTS", starts)
        tss, plateau = tss_module._monoexponential_row(time, value, 0.9)
        assert np.isnan(tss) and np.isnan(plateau), starts


def test_a_rising_and_a_falling_series_have_no_estimate_whatever_the_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(engine, "_force_python", True)
    time = np.arange(12.0, 145.0, 12.0)
    for value in (0.5 * time, 20.0 - time / 10.0):
        for starts in [(0.5, 2.0, 8.0), (0.3, 1.0, 3.0), (1.0, 5.0, 20.0)]:
            monkeypatch.setattr(tss_module, "_STARTS", starts)
            tss, plateau = tss_module._monoexponential_row(time, value, 0.9)
            assert np.isnan(tss) and np.isnan(plateau), (value[0], starts)


class FakeFit:
    """A fit which reports success wherever it is told it stopped."""

    def __init__(self, x: float, cost: float) -> None:
        self.x = np.array([x])
        self.cost = cost
        self.success = True


def stopped_at(monkeypatch: pytest.MonkeyPatch, x: float, cost: float) -> None:
    """Make the search of the python reference stop at `log10(k) = x` with `cost`."""
    monkeypatch.setattr(tss_module, "least_squares", lambda *a, **k: FakeFit(x, cost))
    monkeypatch.setattr(engine, "_force_python", True)


def test_a_stop_inside_the_range_does_not_make_an_unbounded_optimum_an_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the search reports success where it stopped, and it can stop inside the range of the
    # starts: the cost beyond the range is what tells the optimum is not there
    time = np.arange(12.0, 145.0, 12.0)
    flat = np.full(time.size, 5.0)  # cost 0 at every large k, the high edge
    proportional = 0.5 * time  # cost falls towards k = 0, the low edge
    for value, x in (
        (flat, 0.0),
        (flat, -1.0),
        (proportional, -2.5),
        (proportional, -3.5),
    ):
        stopped_at(monkeypatch, x, 1e6)
        tss, plateau = tss_module._monoexponential_row(time, value, 0.9)
        assert np.isnan(tss) and np.isnan(plateau), (value[0], x)


def test_a_cost_as_low_as_the_edge_is_no_interior_optimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time = np.arange(12.0, 145.0, 12.0)
    value = np.full(time.size, 5.0)
    monkeypatch.setattr(tss_module, "_EDGE_TOLERANCE", 0.0)
    # the cost at the high edge of a flat series is exactly 0: a fit which reports 0 is a tie
    stopped_at(monkeypatch, 0.0, 0.0)
    assert np.isnan(tss_module._monoexponential_row(time, value, 0.9)[0])
    # and a fit which is worse than the edge by a hair is one as well, with the tolerance
    monkeypatch.setattr(tss_module, "_EDGE_TOLERANCE", 1e-12)
    stopped_at(monkeypatch, 0.0, 1e-13 * float(np.sum(value * value)))
    assert np.isnan(tss_module._monoexponential_row(time, value, 0.9)[0])


def profiled_cost(time: np.ndarray, value: np.ndarray, x: float) -> float:
    """Half the residual sum of squares of the profiled model at `k = 10 ** x`."""
    g = 1.0 - np.exp(-(10.0**x) * time)
    plateau = float(np.sum(value * g) / np.sum(g * g))
    return float(np.sum((value - plateau * g) ** 2)) / 2.0


def test_the_cost_at_the_edge_is_compared_with_a_tolerance_of_the_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time = np.arange(12.0, 145.0, 12.0)
    value = 0.5 * time
    low = float(np.log10(0.5 / time[-1])) - 2.0
    edge = profiled_cost(time, value, low)
    scale = float(np.sum(value * value))
    monkeypatch.setattr(tss_module, "_EDGE_TOLERANCE", 1e-12)
    # worse than the edge by half the tolerance: a tie, not an interior optimum
    stopped_at(monkeypatch, -2.5, edge - 0.5e-12 * scale)
    assert np.isnan(tss_module._monoexponential_row(time, value, 0.9)[0])
    # better than the edge by more than the tolerance is one
    stopped_at(monkeypatch, -2.5, edge - 2e-12 * scale)
    assert np.isfinite(tss_module._monoexponential_row(time, value, 0.9)[0])


def test_a_stop_on_the_edge_of_the_range_is_not_inside_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time = np.arange(12.0, 145.0, 12.0)
    value = 10.0 * (1.0 - np.exp(-0.05 * time))  # the edges cost far more than 0
    low = float(np.log10(0.5 / time[-1])) - 2.0
    high = float(np.log10(8.0 / time[-1])) + 2.0
    for x in (low, high):
        stopped_at(monkeypatch, x, 0.0)
        assert np.isnan(tss_module._monoexponential_row(time, value, 0.9)[0]), x
    stopped_at(monkeypatch, np.log10(0.05), 0.0)
    assert np.isfinite(tss_module._monoexponential_row(time, value, 0.9)[0])


def test_a_minimum_two_decades_below_the_starts_is_still_an_estimate() -> None:
    # k = 3e-4 per hour, 1.7 decades below the lowest start: inside the range of the search
    time = np.arange(12.0, 145.0, 12.0)
    value = 1e3 * (1.0 - np.exp(-3e-4 * time))
    engine._force_python = True
    try:
        tss, plateau = tss_module._monoexponential_row(time, value, 0.9)
    finally:
        engine._force_python = False
    assert tss == pytest.approx(-np.log(0.1) / 3e-4, rel=1e-3)
    assert plateau == pytest.approx(1e3, rel=1e-3)


def test_a_minimum_beyond_the_range_of_the_starts_is_no_estimate_in_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # k = 1e-6 per hour is a genuine minimum 4.5 decades below the starts: the kernel searches
    # two decades around them and reports nothing there, and so does the reference
    monkeypatch.setattr(engine, "_force_python", True)
    time = np.arange(12.0, 145.0, 12.0)
    value = 1e6 * (1.0 - np.exp(-1e-6 * time))
    assert np.isnan(tss_module._monoexponential_row(time, value, 0.9)[0])


@pytest.mark.skipif(
    not hasattr(pkpdutils_core, "time_to_steady_state"),
    reason="this build of pkpdutils-core carries no `time_to_steady_state`",
)
class TestTheKernelEntryPoint:
    """The two engines of `time_to_steady_state` against each other."""

    @staticmethod
    def both(result, **options):
        """The estimate of the python path and of the kernel."""
        engine._force_python = True
        try:
            python = time_to_steady_state(result, **options)
        finally:
            engine._force_python = False
        return python, time_to_steady_state(result, **options)

    @pytest.mark.parametrize("method", ["monoexponential", "stepwise"])
    @pytest.mark.parametrize("k", [0.05, 0.2, 0.5])
    def test_the_engines_agree_on_a_batch(self, method, k) -> None:
        rng = np.random.default_rng(11)
        curves = []
        for i in range(6):
            predicted = superposed(k * (1 + 0.1 * i), 14)
            curves.append(
                Timecourse(
                    time=predicted.time,
                    value=predicted.value * rng.normal(1.0, 0.03, predicted.value.size),
                    time_unit="hr",
                    unit="mg/l",
                    dosing=predicted.dosing,
                    substance="x",
                    label=f"s{i}",
                )
            )
        result = nca(
            Timecourses.from_timecourses(curves, dim="individual"), options=LOG
        )
        python, kernel = self.both(result, method=method)
        # noisy troughs have a flat cost, where scipy's Levenberg-Marquardt stops at its own
        # tolerance: it is within 6e-6 of the minimum the kernel finds to 3e-8 (measured
        # over 300 noisy series against a bounded scalar search), so the two agree to 1e-4;
        # the exact series of the other tests hold each engine to 1e-6 of the analytic value
        rtol = 1e-4 if method == "monoexponential" else 0.0
        np.testing.assert_allclose(
            kernel.tss.to_numpy(), python.tss.to_numpy(), rtol=rtol, equal_nan=True
        )
        if method == "monoexponential":
            assert kernel.c_ss is not None and python.c_ss is not None
            np.testing.assert_allclose(
                kernel.c_ss.to_numpy(),
                python.c_ss.to_numpy(),
                rtol=1e-4,
                equal_nan=True,
            )

    def test_an_exact_straight_line_of_troughs_agrees_between_the_engines(self) -> None:
        # a slope of exactly 0 is no trend and every other slope on an exact line is one,
        # whatever the t statistic does at a zero residual
        for trough in (
            lambda i: 5.0,
            lambda i: 5.0 + 3.0 * i,
            lambda i: 40.0 - 2.0 * i,
        ):
            result = nca_single(periodic(trough))
            python, kernel = self.both(result, method="stepwise")
            np.testing.assert_array_equal(kernel.tss.to_numpy(), python.tss.to_numpy())

    def test_a_minimum_beyond_the_range_of_the_starts_is_no_estimate_in_the_kernel(
        self,
    ) -> None:
        time = np.arange(12.0, 145.0, 12.0)[None]
        value = 1e6 * (1.0 - np.exp(-1e-6 * time))
        answer = pkpdutils_core.time_to_steady_state(  # ty: ignore[missing-attribute]
            {
                "time": time,
                "value": value,
                "start": np.zeros_like(time),
                "valid": np.ones(time.shape, dtype=bool),
                "n_rows": 1,
                "n_cols": time.size,
                "method": "monoexponential",
                "fraction": 0.9,
                "alpha": 0.05,
            }
        )
        assert np.isnan(answer["tss"][0]) and np.isnan(answer["c_ss"][0])

    def test_a_malformed_request_is_a_value_error_and_not_a_python_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = nca_single(superposed(0.2, 10), options=LOG)
        real = pkpdutils_core.time_to_steady_state  # ty: ignore[missing-attribute]

        def corrupt(request):
            return real({**request, "value": request["value"][:, :-1]})

        fell_back = []
        real_row = tss_module._monoexponential_row
        monkeypatch.setattr(pkpdutils_core, "time_to_steady_state", corrupt)
        monkeypatch.setattr(
            tss_module,
            "_monoexponential_row",
            lambda *args: (fell_back.append(1), real_row(*args))[1],
        )
        with pytest.raises(ValueError, match="value entries") as raised:
            time_to_steady_state(result)
        assert not isinstance(raised.value, engine.UnsupportedError)
        assert not fell_back
