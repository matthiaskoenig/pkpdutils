import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from pkpdutils import Dose, NCAOptions, Route, Timecourse, nca_single
from pkpdutils.nca.sparse import nca_sparse, sparse_mean
from pkpdutils.nca.urine import Excretion, nca_urine
from pkpdutils.plot import plot_excretion, plot_sparse

matplotlib.use("Agg")

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
    fig = plot_excretion(nca_urine(exc), exc)
    assert isinstance(fig, Figure)
    # the rate axis and the twin axis of the amount recovered
    assert len(fig.axes) == 2
    rate_axes = fig.axes[0]
    assert rate_axes.get_yscale() == "log"
    labels = [line.get_label() for line in rate_axes.lines]
    assert "excretion rate" in labels
    assert any(str(label).startswith("lambda_z = ") for label in labels)
    assert "amount recovered" in [line.get_label() for line in fig.axes[1].lines]
    matplotlib.pyplot.close("all")


def test_plot_excretion_without_the_second_axis_and_on_a_linear_scale() -> None:
    exc = excretion()
    fig = plot_excretion(nca_urine(exc), exc, cumulative=False, log_rate=False)
    assert len(fig.axes) == 1
    assert fig.axes[0].get_yscale() == "linear"
    matplotlib.pyplot.close("all")


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
    matplotlib.pyplot.close("all")


def test_plot_sparse_shades_the_area_and_writes_the_estimate() -> None:
    values = sparse_values()
    curve = sparse_mean(TIMES, values, time_unit="hr", unit="mg/l", substance="drug")
    result = nca_sparse(TIMES, values, time_unit="hr", unit="mg/l")
    fig = plot_sparse(curve, result)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert any(isinstance(child, PolyCollection) for child in ax.collections)
    text = ax.texts[0].get_text()
    assert "auc_last =" in text
    assert "df =" in text
    assert "n animals per time point: 3, 3, 3" in text
    matplotlib.pyplot.close("all")


def test_plot_sparse_takes_a_single_curve_and_rejects_a_batch_of_several() -> None:
    values = sparse_values()
    curve = sparse_mean(TIMES, values, time_unit="hr", unit="mg/l")
    result = nca_sparse(TIMES, values, time_unit="hr", unit="mg/l")
    single = next(iter(curve))
    assert isinstance(plot_sparse(single, result, log_y=True), Figure)

    from pkpdutils import Timecourses

    two = Timecourses.from_timecourses([single, single], labels=["a", "b"])
    with pytest.raises(ValueError, match="holds 2 samples"):
        plot_sparse(two, result)
    matplotlib.pyplot.close("all")


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
    matplotlib.pyplot.close("all")
