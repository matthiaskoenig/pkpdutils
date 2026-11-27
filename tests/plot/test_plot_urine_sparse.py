import numpy as np
import pytest
from sbmlsim_core.chart import Chart

from pkpdutils import Dose, NCAOptions, Route, Timecourse, Timecourses, nca_single
from pkpdutils.nca.sparse import nca_sparse, sparse_mean
from pkpdutils.nca.urine import Excretion, nca_urine
from pkpdutils.plot import plot_excretion, plot_sparse

EDGES = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0])
TIMES = np.array([1.0, 2.0, 4.0])


def excretion() -> Excretion:
    start, end = EDGES[:-1], EDGES[1:]
    return Excretion(
        start=start,
        end=end,
        amount=80.0 * (np.exp(-0.25 * start) - np.exp(-0.25 * end)),
        volume=np.array([220.0, 180.0, 260.0, 310.0, 250.0, 400.0]),
        unit="mg",
        time_unit="hr",
        volume_unit="ml",
        dose=Dose(amount=100.0, unit="mg", route=Route.IV_BOLUS),
        substance="drug",
    )


def sparse_values() -> np.ndarray:
    values = np.full((9, 3), np.nan)
    for j, column in enumerate(([1.0, 1.2, 1.4], [2.0, 2.4, 2.2], [0.5, 0.4, 0.6])):
        for i, value in enumerate(column):
            values[3 * j + i, j] = value
    return values


def test_plot_excretion_draws_the_rate_the_regression_and_the_cumulative_amount() -> (
    None
):
    exc = excretion()
    chart = plot_excretion(nca_urine(exc), exc)
    assert isinstance(chart, Chart)
    (plot,) = chart.spec()["plots"]
    # the rate axis and the second axis of the amount recovered
    assert plot["y"]["scale"] == "log"
    assert plot["y2"]["min"] == 0.0
    assert plot["legend_loc"] == "upper_right"
    on_y = [e["label"] for e in plot["elements"] if e["axis"] == "y"]
    assert "excretion rate" in on_y
    assert any(str(label).startswith("lambda_z = ") for label in on_y)
    on_y2 = [e["label"] for e in plot["elements"] if e["axis"] == "y2"]
    assert on_y2 == ["amount recovered"]
    # the plateau of the recovery keeps room above it for the legend
    top = float(np.nanmax(exc.cumulative))
    assert plot["y2"]["max"] == pytest.approx(1.6 * top)


def test_plot_excretion_without_the_second_axis_and_on_a_linear_scale() -> None:
    exc = excretion()
    chart = plot_excretion(nca_urine(exc), exc, cumulative=False, log_rate=False)
    (plot,) = chart.spec()["plots"]
    assert "y2" not in plot
    assert plot["y"]["scale"] == "linear"


def test_plot_excretion_needs_a_urine_result() -> None:
    curve = Timecourse(
        time=[0.0, 1.0, 2.0],
        value=[2.0, 1.0, 0.5],
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=10.0, unit="mg", route=Route.IV_BOLUS),
    )
    with pytest.raises(ValueError, match="no excretion rate curve"):
        plot_excretion(nca_single(curve, options=NCAOptions()), excretion())


def test_plot_sparse_shades_the_area_and_writes_the_estimate() -> None:
    values = sparse_values()
    curve = sparse_mean(TIMES, values, time_unit="hr", unit="mg/l", substance="drug")
    result = nca_sparse(TIMES, values, time_unit="hr", unit="mg/l")
    chart = plot_sparse(curve, result)
    assert isinstance(chart, Chart)
    (plot,) = chart.spec()["plots"]
    areas = [e for e in plot["elements"] if e["kind"] == "area"]
    assert [a["label"] for a in areas] == ["auc_last"]
    # the mean with the standard error of every time point
    (mean,) = [e for e in plot["elements"] if e["kind"] == "curve"]
    assert mean["label"] == "mean of the animals" and "yerr" in mean
    (text,) = [e["text"] for e in plot["elements"] if e["kind"] == "text"]
    assert "auc_last =" in text
    assert "df =" in text
    assert "n animals per time point: 3, 3, 3" in text
    assert plot["legend_loc"] == "center_right"


def test_plot_sparse_takes_a_single_curve_and_rejects_a_batch_of_several() -> None:
    values = sparse_values()
    curve = sparse_mean(TIMES, values, time_unit="hr", unit="mg/l")
    result = nca_sparse(TIMES, values, time_unit="hr", unit="mg/l")
    single = next(iter(curve))
    chart = plot_sparse(single, result, log_y=True)
    (plot,) = chart.spec()["plots"]
    assert plot["y"]["scale"] == "log"
    # the area of a logarithmic panel starts at its bottom, not at zero
    (area,) = [e for e in plot["elements"] if e["kind"] == "area"]
    assert min(area["from"]) == pytest.approx(plot["y"]["min"])

    two = Timecourses.from_timecourses([single, single], labels=["a", "b"])
    with pytest.raises(ValueError, match="holds 2 samples"):
        plot_sparse(two, result)


def test_plot_sparse_needs_a_sparse_result() -> None:
    values = sparse_values()
    curve = sparse_mean(TIMES, values, time_unit="hr", unit="mg/l")
    plain = Timecourse(
        time=[0.0, 1.0, 2.0],
        value=[2.0, 1.0, 0.5],
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=10.0, unit="mg", route=Route.IV_BOLUS),
    )
    with pytest.raises(ValueError, match="auc_last_se"):
        plot_sparse(curve, nca_single(plain, options=NCAOptions()))
