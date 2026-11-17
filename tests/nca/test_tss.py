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


@pytest.mark.parametrize("thalf", [0.7, 0.6])
def test_a_half_life_far_below_the_dosing_interval_is_still_an_estimate(
    thalf: float, tss_engine: str
) -> None:
    # a drug whose half-life is a twentieth of the 12 h interval is at its plateau to
    # within a part per million at the first trough, so the fit is near exact: its cost
    # is 0 at the optimum and the variance of the troughs at the edges of the search,
    # which is an interior optimum. The plateau is sharp and well determined, so both
    # engines report `-ln(1 - f) / k`. Python used to return NaN at 0.6 h and below,
    # because its tie test compared the cost at the edge with an absolute floor of the
    # sum of squares of the troughs, which a near exact fit falls below; the answer a
    # user got therefore depended on which wheel was installed
    k = np.log(2.0) / thalf
    estimate = time_to_steady_state(nca_single(superposed(k, 10), options=LOG))
    assert float(estimate.tss) == pytest.approx(np.log(10.0) / k, rel=1e-6)
    assert estimate.c_ss is not None
    assert float(estimate.c_ss) == pytest.approx(
        C0 * np.exp(-k * TAU) / (1 - np.exp(-k * TAU)), rel=1e-6
    )


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
    # sets, with a correct plateau: k is unidentifiable and nothing of it was an estimate.
    # Troughs which are all the same value have no rise at all and are answered before the
    # search runs, so the answer cannot depend on the starts
    monkeypatch.setattr(engine, "_force_python", True)
    searched: list[int] = []
    real = tss_module.least_squares
    monkeypatch.setattr(
        tss_module,
        "least_squares",
        lambda *args, **kwargs: (searched.append(1), real(*args, **kwargs))[1],
    )
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
    assert not searched


def test_a_flat_series_on_an_irregular_grid_has_no_estimate_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the input which makes the guard load bearing rather than tidy, found by a differential
    # sweep of 800 noise-free series against the kernel: 23 troughs of one and the same double
    # on a grid whose first time is a twenty-fifth of its last. The cost at the high edge of
    # the range is then not 0 but the rounding of the model there, about 1e-41, and the search
    # stops at an interior point whose cost is 1.7 times lower, so the tie test does not fire
    # and the series was answered with tss = 0.2783 - a number out of the rounding of a
    # constant series. The tolerance is relative and cannot see that, the guard can
    monkeypatch.setattr(engine, "_force_python", True)
    time = np.array(
        [
            4.092234572961332,
            9.732813207153443,
            11.698375397332395,
            11.706181166276703,
            25.99456951343599,
            26.16766364357455,
            35.65941705776162,
            41.17051631505455,
            42.224812446843956,
            44.794893663794916,
            53.61523865561471,
            54.62850377493503,
            63.17089255769438,
            66.97481332378435,
            69.74969693177903,
            78.2895045043537,
            80.61200906800438,
            83.20716595193569,
            90.34865936268163,
            94.16857840298161,
            94.18081950382191,
            95.15203857285773,
            95.63239698232921,
        ]
    )
    value = np.full(time.size, 1.5200989623303864e-06)
    tss, plateau = tss_module._monoexponential_row(time, value, 0.9)
    assert np.isnan(tss) and np.isnan(plateau)


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
    falling = 20.0 - time / 10.0  # the cost falls towards large k, the high edge
    proportional = 0.5 * time  # cost falls towards k = 0, the low edge
    for value, x in (
        (falling, 0.0),
        (falling, -1.0),
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
    value = 20.0 - time / 10.0  # falling: the cost falls towards large k, the high edge
    edge = profiled_cost(time, value, float(np.log10(8.0 / time[-1])) + 2.0)
    monkeypatch.setattr(tss_module, "_EDGE_TOLERANCE", 0.0)
    # a fit which reports the cost of the edge is a tie, with no tolerance at all
    stopped_at(monkeypatch, 0.0, edge)
    assert np.isnan(tss_module._monoexponential_row(time, value, 0.9)[0])
    # and a fit which is worse than the edge is one as well
    stopped_at(monkeypatch, 0.0, 2.0 * edge)
    assert np.isnan(tss_module._monoexponential_row(time, value, 0.9)[0])
    # a flat series is answered before the search, so there is nothing to tie
    stopped_at(monkeypatch, 0.0, 0.0)
    assert np.isnan(
        tss_module._monoexponential_row(time, np.full(time.size, 5.0), 0.9)[0]
    )


def profiled_cost(time: np.ndarray, value: np.ndarray, x: float) -> float:
    """Half the residual sum of squares of the profiled model at `k = 10 ** x`."""
    g = 1.0 - np.exp(-(10.0**x) * time)
    plateau = float(np.sum(value * g) / np.sum(g * g))
    return float(np.sum((value - plateau * g) ** 2)) / 2.0


def test_the_cost_at_the_edge_is_compared_relative_to_the_larger_of_the_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time = np.arange(12.0, 145.0, 12.0)
    value = 0.5 * time
    low = float(np.log10(0.5 / time[-1])) - 2.0
    edge = profiled_cost(time, value, low)
    scale = float(np.sum(value * value))
    monkeypatch.setattr(tss_module, "_EDGE_TOLERANCE", 1e-12)
    # better than the edge by half the relative tolerance: a tie, not an interior optimum
    stopped_at(monkeypatch, -2.5, edge * (1.0 - 0.5e-12))
    assert np.isnan(tss_module._monoexponential_row(time, value, 0.9)[0])
    # better than it by more than the relative tolerance is one
    stopped_at(monkeypatch, -2.5, edge * (1.0 - 2e-12))
    assert np.isfinite(tss_module._monoexponential_row(time, value, 0.9)[0])
    # the tolerance is relative to the costs and not to the sum of squares of the troughs:
    # a fit a billionth of its own cost below the edge is an interior optimum, although that
    # difference is far below `1e-12 * scale`, where the absolute floor this replaces called
    # it a tie and reported no estimate (finding 1 of the whole-branch review)
    assert edge * 1e-9 < 1e-12 * scale
    stopped_at(monkeypatch, -2.5, edge * (1.0 - 1e-9))
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

    def test_the_level_of_the_stepwise_estimate_reaches_the_kernel(self) -> None:
        rng = np.random.default_rng(7)
        predicted = superposed(0.05, 16)
        noisy = predicted.model_copy(
            update={
                "value": predicted.value * rng.normal(1.0, 0.05, predicted.value.size)
            }
        )
        result = nca_single(noisy, options=LOG)
        found = set()
        for alpha in (0.001, 0.05, 0.5, 0.9):
            python, kernel = self.both(result, method="stepwise", alpha=alpha)
            assert float(kernel.tss) == float(python.tss) or (
                np.isnan(float(kernel.tss)) and np.isnan(float(python.tss))
            )
            found.add(float(kernel.tss))
        # the answer depends on alpha, so a level which did not arrive would show
        assert len(found) > 1

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
        # tolerance well before the minimum the kernel brackets to 3e-8. Measured over 400
        # random series of the shape of this test (14 intervals of 12 h, k from 0.05 to 0.75,
        # 3 % noise) the two engines differ in tss by a median of 3.4e-7, 7.2e-6 at the 95th
        # percentile and 1.55e-4 at the worst; over a wider domain (4 to 30 intervals,
        # k * t_last from 0.003 to 1e4) by 2.2e-4 at 3 % noise and 1.2e-3 at 10 %. So 1e-4,
        # which this test asserted, held only for its own seeds: 2e-3 is what the data
        # support. The plateau is the better determined half of the fit, 8.8e-8 at the worst
        # on this shape, and keeps its 1e-4. The exact series of the other tests hold each
        # engine to 1e-6 of the analytic value
        rtol = 2e-3 if method == "monoexponential" else 0.0
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

    def test_troughs_saturated_to_the_resolution_of_a_double_still_differ(self) -> None:
        # the one input where the two engines are known to disagree, measured rather than
        # left to be found: at a half-life a fortieth of the dosing interval the first
        # trough is 1e-12 from the plateau, so the cost of the whole region above the
        # optimum is zero to the resolution of a double. The kernel brackets and recovers
        # `k` to 3e-4, while python's Levenberg-Marquardt stops anywhere in that region,
        # where the cost ties the edge of the range, and reports no estimate. Giving python
        # the same bracketed search is the recorded follow-up of this wave
        k = np.log(2.0) / 0.3
        python, kernel = self.both(nca_single(superposed(k, 10), options=LOG))
        assert np.isnan(float(python.tss))
        assert float(kernel.tss) == pytest.approx(np.log(10.0) / k, rel=1e-3)

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

    def test_the_engines_agree_when_a_sample_has_fewer_troughs_than_the_others(
        self,
    ) -> None:
        # the troughs of the shorter sample are the padding of its row, which the
        # kernel must not read: every entry of `valid` counts
        curves = [
            superposed(0.2, n).model_copy(update={"label": f"s{n}"})
            for n in (6, 10, 14)
        ]
        result = nca(
            Timecourses.from_timecourses(curves, dim="individual"), options=LOG
        )
        for method in ("monoexponential", "stepwise"):
            python, kernel = self.both(result, method=method)
            assert np.isfinite(python.tss.to_numpy()).sum() >= 2
            np.testing.assert_allclose(
                kernel.tss.to_numpy(), python.tss.to_numpy(), rtol=1e-6, equal_nan=True
            )

    def test_the_stepwise_estimate_has_no_plateau_from_the_kernel(self) -> None:
        end = np.array([[12.0, 24.0, 36.0, 48.0]])
        answer = tss_module._kernel_estimate(
            end,
            np.array([[1.0, 2.0, 2.5, 2.5]]),
            end - 12.0,
            np.ones(end.shape, dtype=bool),
            "stepwise",
            0.9,
            0.05,
        )
        assert answer is not None and answer[1] is None

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
