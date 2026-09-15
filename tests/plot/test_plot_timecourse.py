import warnings

import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from pkpdutils import Dose, Dosing, Route, Timecourse, Timecourses
from pkpdutils.plot import plot_mean_timecourse, plot_timecourse

matplotlib.use("Agg")


def multiple_dose_tc(n_doses: int = 4, tau: float = 12.0) -> Timecourse:
    """A bolus curve given every `tau` hours, `n_doses` times."""
    time = np.arange(0, n_doses * tau + 0.01, 1.0)
    value = np.zeros_like(time)
    for k in range(n_doses):
        shifted = time - k * tau
        value += np.where(shifted >= 0, 10.0 * np.exp(-0.15 * shifted), 0.0)
    dosing = Dosing.regimen(
        Dose(amount=100, unit="mg", route=Route.IV_BOLUS), interval=tau, n_doses=n_doses
    )
    return Timecourse(
        time=time, value=value, time_unit="hr", unit="mg/l", dosing=dosing
    )


def test_plot_timecourse_draws_a_dotted_line_per_dose() -> None:
    tc = multiple_dose_tc(n_doses=4)
    fig = plot_timecourse(tc)
    ax = fig.axes[0]
    dose_lines = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
    assert len(dose_lines) == 4
    for line in dose_lines:
        assert line.get_color() == "gray"
    matplotlib.pyplot.close(fig)


def test_plot_timecourse_single_dose_draws_no_dose_line() -> None:
    tc = Timecourse(
        time=np.array([0.5, 1, 2, 4, 8]),
        value=10.0 * np.exp(-0.15 * np.array([0.5, 1, 2, 4, 8])),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
    )
    fig = plot_timecourse(tc)
    ax = fig.axes[0]
    dose_lines = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
    assert len(dose_lines) == 0
    matplotlib.pyplot.close(fig)


def test_plot_timecourse_log_without_positive_values_stays_linear() -> None:
    # B28: a curve with no positive value used to raise "UserWarning: Data
    # has no positive values, and therefore cannot be log-scaled" at draw time.
    tc = Timecourse(
        time=[0, 1, 2, 4],
        value=[0.0, 0.0, 0.0, 0.0],
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = plot_timecourse(tc, log_y=True)
        fig.canvas.draw()
    assert fig.axes[0].get_yscale() == "linear"
    matplotlib.pyplot.close(fig)


def test_plot_timecourse_batch_draws_no_dose_lines() -> None:
    a = multiple_dose_tc(n_doses=3)
    b = multiple_dose_tc(n_doses=3)
    batch = Timecourses.from_timecourses([a, b], labels=["one", "two"])
    fig = plot_timecourse(batch)
    ax = fig.axes[0]
    dose_lines = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
    assert len(dose_lines) == 0
    matplotlib.pyplot.close(fig)


def escalation(n_individual: int = 3) -> Timecourses:
    """A batch over `(dose, individual)` of oral curves, dosed in mg."""
    time = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    doses = np.array([50.0, 100.0])
    values = np.empty((doses.size, n_individual, time.size))
    for i, dose in enumerate(doses):
        for j in range(n_individual):
            values[i, j] = dose / 50.0 * (1.0 + 0.1 * j) * np.exp(-0.2 * time)
    return Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("dose", "individual"),
        coords={"dose": doses, "individual": [f"s{j}" for j in range(n_individual)]},
        dose={
            "amount": np.broadcast_to(doses[:, None], (doses.size, n_individual)),
            "unit": "mg",
        },
        route=Route.ORAL,
        substance="caffeine",
    )


def infusion_tc(duration: float = 1.0) -> Timecourse:
    """One curve of a one hour infusion."""
    time = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    value = np.where(
        time <= duration,
        5.0 * (1.0 - np.exp(-0.2 * time)),
        5.0 * (1.0 - np.exp(-0.2 * duration)) * np.exp(-0.2 * (time - duration)),
    )
    return Timecourse(
        time=time,
        value=value,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_INFUSION, duration=duration),
        substance="drug",
    )


def test_plot_timecourse_by_gives_one_legend_entry_and_one_color_per_group() -> None:
    batch = escalation()
    fig = plot_timecourse(batch, by="dose")
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["50", "100"]
    assert legend.get_title().get_text() == "dose"
    colors = {tuple(line.get_color()) for line in ax.get_lines()}
    assert len(colors) == 2
    matplotlib.pyplot.close(fig)


def test_plot_timecourse_facet_draws_one_panel_per_value() -> None:
    batch = escalation()
    fig = plot_timecourse(batch, facet="dose", by="individual")
    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "dose = 50"
    assert fig.axes[1].get_title() == "dose = 100"
    # the legend is drawn once, on the first panel
    assert fig.axes[0].get_legend() is not None
    assert fig.axes[1].get_legend() is None
    matplotlib.pyplot.close(fig)


def test_plot_timecourse_facet_draws_into_the_given_axes() -> None:
    batch = escalation()
    fig, axes = matplotlib.pyplot.subplots(ncols=2)
    assert plot_timecourse(batch, facet="dose", axes=axes) is fig
    assert all(ax.get_lines() for ax in axes)
    assert fig.get_layout_engine() is None
    matplotlib.pyplot.close(fig)


def test_plot_timecourse_suppresses_a_legend_above_max_legend() -> None:
    batch = escalation(n_individual=5)
    assert plot_timecourse(batch, max_legend=12).axes[0].get_legend() is not None
    fig = plot_timecourse(batch, max_legend=3)
    assert fig.axes[0].get_legend() is None
    matplotlib.pyplot.close("all")


def test_plot_timecourse_draws_an_infusion_as_a_window() -> None:
    fig = plot_timecourse(infusion_tc())
    ax = fig.axes[0]
    spans = [patch for patch in ax.patches if patch.get_label() == "infusion"]
    assert len(spans) == 1
    span = spans[0]
    assert isinstance(span, Rectangle)
    assert (span.get_x(), span.get_x() + span.get_width()) == (0.0, 1.0)
    matplotlib.pyplot.close(fig)


def test_plot_timecourse_rejects_conflicting_axes_arguments() -> None:
    batch = escalation()
    _fig, ax = matplotlib.pyplot.subplots()
    with pytest.raises(ValueError):
        plot_timecourse(batch, ax=ax, axes=[ax])
    with pytest.raises(ValueError):
        plot_timecourse(batch, facet="dose", ax=ax)
    with pytest.raises(ValueError):
        plot_timecourse(infusion_tc(), by="dose")
    matplotlib.pyplot.close("all")


def test_plot_mean_timecourse_panels_bands_and_legend() -> None:
    batch = escalation()
    fig = plot_mean_timecourse(batch, by="dose")
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    linear, log = fig.axes
    assert linear.get_yscale() == "linear" and log.get_yscale() == "log"
    assert linear.get_xlabel() == "time [hr]"
    assert linear.get_ylabel() == "caffeine [mg/l]"
    assert "sd" in linear.get_title()
    legend = linear.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "50 (n = 3)",
        "100 (n = 3)",
    ]  # the number of curves of the group
    assert log.get_legend() is None
    # one spread band per group and panel
    assert len(linear.collections) == 2
    matplotlib.pyplot.close(fig)


def test_plot_mean_timecourse_without_individuals_and_without_a_band() -> None:
    batch = escalation()
    fig = plot_mean_timecourse(
        batch, by="dose", spread=None, individuals=False, panels=("linear",)
    )
    ax = fig.axes[0]
    assert len(fig.axes) == 1
    assert not ax.collections  # no band
    assert len(ax.get_lines()) == 2  # one mean curve per group, no individuals
    matplotlib.pyplot.close(fig)


def test_plot_mean_timecourse_draws_into_the_given_axes() -> None:
    batch = escalation()
    fig, axes = matplotlib.pyplot.subplots(ncols=2)
    assert plot_mean_timecourse(batch, by="dose", axes=axes) is fig
    assert all(ax.get_lines() for ax in axes)
    assert fig.get_layout_engine() is None
    matplotlib.pyplot.close(fig)


def test_plot_mean_timecourse_without_by_is_one_mean_curve() -> None:
    batch = Timecourses.from_timecourses(
        [multiple_dose_tc(n_doses=3), multiple_dose_tc(n_doses=3)],
        labels=["one", "two"],
    )
    fig = plot_mean_timecourse(batch, panels=("linear",))
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["mean (n = 2)"]
    # the dose lines of the shared protocol
    assert len([line for line in ax.get_lines() if line.get_linestyle() == ":"]) == 3
    matplotlib.pyplot.close(fig)


def test_plot_mean_timecourse_rejects_unknown_panels_and_spread() -> None:
    batch = escalation()
    with pytest.raises(ValueError):
        plot_mean_timecourse(batch, panels=("linear", "loglog"))
    with pytest.raises(ValueError):
        plot_mean_timecourse(batch, panels=())
    with pytest.raises(ValueError):
        plot_mean_timecourse(batch, spread="ci")  # ty: ignore[invalid-argument-type]
    matplotlib.pyplot.close("all")


def test_plot_mean_timecourse_draws_without_a_matplotlib_warning() -> None:
    batch = escalation()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = plot_mean_timecourse(batch, by="dose")
        fig.canvas.draw()
    matplotlib.pyplot.close(fig)
