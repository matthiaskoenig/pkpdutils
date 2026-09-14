import matplotlib
import matplotlib.pyplot
import numpy as np
from matplotlib.figure import Figure

from pkpdutils import Dose, Route, Timecourse, Timecourses
from pkpdutils.nca import NCAOptions, nca, nca_single
from pkpdutils.plot import PlotStyle, plot_nca, plot_nca_grid, plot_timecourse

matplotlib.use("Agg")


def oral(ka: float = 2.0, label: str = "a") -> Timecourse:
    t = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 24])
    c = 10 * ka / (ka - 0.3) * (np.exp(-0.3 * t) - np.exp(-ka * t))
    return Timecourse(
        time=t,
        value=c,
        sd=0.1 * c,
        n=8,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        substance="caffeine",
        label=label,
    )


def test_plot_timecourse_single_and_batch() -> None:
    fig = plot_timecourse(oral())
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "time [hr]" and ax.get_ylabel() == "caffeine [mg/l]"
    batch = Timecourses.from_timecourses([oral(1.0, "slow"), oral(4.0, "fast")])
    fig2 = plot_timecourse(batch, log=True, errorbars=False)
    assert fig2.axes[0].get_yscale() == "log"
    labels = [line.get_label() for line in fig2.axes[0].get_lines()]
    assert "slow" in labels and "fast" in labels
    matplotlib.pyplot.close("all")


def test_plot_nca_single() -> None:
    tc = oral()
    result = nca_single(tc)
    fig = plot_nca(tc, result, title="test")
    assert len(fig.axes) == 2
    assert fig.axes[1].get_yscale() == "log"
    assert "test" in fig.axes[0].get_title()
    matplotlib.pyplot.close(fig)


def test_plot_nca_from_batch_result_with_indexers_and_flags_in_title() -> None:
    batch = Timecourses.from_timecourses([oral(1.0, "slow"), oral(4.0, "fast")])
    result = nca(batch)
    fig = plot_nca(batch.sel(individual="fast"), result, individual="fast")
    assert "fast" in fig.axes[0].get_title()
    short = Timecourse(time=[1, 2, 3], value=[1, 3, 2], time_unit="hr", unit="mg/l")
    fig2 = plot_nca(short, nca_single(short))
    assert "TOO_FEW_POINTS" in fig2.axes[0].get_title()
    matplotlib.pyplot.close("all")


def test_plot_nca_iv_bolus_marks_c0() -> None:
    t = np.array([0.5, 1, 2, 4, 8])
    tc = Timecourse(
        time=t,
        value=10 * np.exp(-0.5 * t),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
    )
    fig = plot_nca(tc, nca_single(tc), style=PlotStyle(fit_color="red"))
    labels = {line.get_label() for line in fig.axes[0].get_lines()}
    assert "C0" in labels
    matplotlib.pyplot.close(fig)


def test_plot_nca_grid() -> None:
    batch = Timecourses.from_timecourses(
        [oral(k, str(k)) for k in (1.0, 2.0, 3.0, 4.0)]
    )
    fig = plot_nca_grid(batch, nca(batch, NCAOptions()), ncols=2)
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 4
    matplotlib.pyplot.close(fig)
