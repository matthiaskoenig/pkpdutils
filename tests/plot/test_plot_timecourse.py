import warnings

import matplotlib
import matplotlib.lines
import matplotlib.pyplot
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from pkpdutils import Dose, Dosing, Route, Timecourse, Timecourses
from pkpdutils.plot import (
    PlotStyle,
    plot_mean_timecourse,
    plot_study_curves,
    plot_timecourse,
)

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
    # the unit of the grouping coordinate is written once, in the title
    assert legend.get_title().get_text() == "dose [mg]"
    colors = {tuple(line.get_color()) for line in ax.get_lines()}
    assert len(colors) == 2
    matplotlib.pyplot.close(fig)


def test_plot_timecourse_facet_draws_one_panel_per_value() -> None:
    batch = escalation()
    fig = plot_timecourse(batch, facet="dose", by="individual")
    assert len(fig.axes) == 2
    # the panel names the value of the coordinate with its unit
    assert fig.axes[0].get_title() == "dose = 50 mg"
    assert fig.axes[1].get_title() == "dose = 100 mg"
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
    spans = [
        patch for patch in ax.patches if str(patch.get_label()).startswith("infusion")
    ]
    assert len(spans) == 1
    assert spans[0].get_label() == "infusion (1 hr)"
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
        "50 mg (n = 3)",
        "100 mg (n = 3)",
    ]  # the dose with its unit and the number of curves of the group
    assert log.get_legend() is None
    # one spread band per group and panel
    assert len(linear.collections) == 2
    matplotlib.pyplot.close(fig)


def test_plot_mean_timecourse_labels_a_group_of_strings_without_a_unit() -> None:
    # a `dose` coordinate of names is a label, not an amount: the dose unit of
    # the batch used to be appended to it ("low mg (n = 2)")
    time = np.array([0.0, 1.0, 2.0, 4.0])
    values = np.stack([(1.0 + i) * np.exp(-0.3 * time) for i in range(4)])
    batch = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={
            "individual": [f"s{j}" for j in range(4)],
            "dose": ("individual", ["low", "low", "high", "high"]),
        },
        dose={"amount": np.full(4, 100.0), "unit": "mg"},
        route=Route.ORAL,
        substance="caffeine",
    )
    fig = plot_mean_timecourse(batch, by="dose")
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "low (n = 2)",
        "high (n = 2)",
    ]
    matplotlib.pyplot.close(fig)
    # and the same for the panel titles and the legend title of `plot_timecourse`
    faceted = plot_timecourse(batch, facet="dose", by="individual")
    assert [ax.get_title() for ax in faceted.axes] == ["dose = low", "dose = high"]
    matplotlib.pyplot.close(faceted)
    grouped = plot_timecourse(batch, by="dose")
    grouped_legend = grouped.axes[0].get_legend()
    assert grouped_legend is not None
    assert grouped_legend.get_title().get_text() == "dose"
    assert [text.get_text() for text in grouped_legend.get_texts()] == ["low", "high"]
    matplotlib.pyplot.close(grouped)


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


def test_plot_timecourse_facet_legend_names_every_group() -> None:
    # a group which no sample of the first panel carries still gets its
    # legend entry: the legend is built from the groups of the whole batch
    batch = escalation(n_individual=3)
    sex = np.array([["m", "m", "m"], ["m", "f", "m"]])  # "f" only at dose 100
    batch.ds = batch.ds.assign_coords(sex=(("dose", "individual"), sex))
    fig = plot_timecourse(batch, facet="dose", by="sex")
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert sorted(text.get_text() for text in legend.get_texts()) == ["f", "m"]
    matplotlib.pyplot.close(fig)


def test_plot_mean_timecourse_group_without_a_unit_is_the_bare_value() -> None:
    time = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    values = np.stack([(1.0 + 0.1 * j) * np.exp(-0.2 * time) for j in range(4)])
    batch = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={
            "individual": [f"s{j}" for j in range(4)],
            "sex": ("individual", ["f", "m", "f", "m"]),
        },
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    )
    fig = plot_mean_timecourse(batch, by="sex", panels=("linear",))
    legend = fig.axes[0].get_legend()
    assert legend is not None
    # a coordinate without a unit is named by its bare value
    assert sorted(text.get_text() for text in legend.get_texts()) == [
        "f (n = 2)",
        "m (n = 2)",
    ]
    matplotlib.pyplot.close(fig)


def test_plot_timecourse_drops_the_markers_of_a_densely_sampled_curve() -> None:
    time = np.linspace(0, 24, 241)
    dense = Timecourse(
        time=time,
        value=10.0 * np.exp(-0.2 * time),
        time_unit="hr",
        unit="mg/l",
    )
    # 241 points: the markers would merge into a band and hide the curve
    line = plot_timecourse(dense).axes[0].get_lines()[0]
    assert line.get_marker() == "none"
    sparse = Timecourse(
        time=time[::30],
        value=10.0 * np.exp(-0.2 * time[::30]),
        time_unit="hr",
        unit="mg/l",
    )
    assert plot_timecourse(sparse).axes[0].get_lines()[0].get_marker() == "o"
    # the threshold is a style, a caller who wants every marker raises it
    style = PlotStyle(marker_max_points=500)
    assert (
        plot_timecourse(dense, style=style).axes[0].get_lines()[0].get_marker() == "o"
    )
    matplotlib.pyplot.close("all")


def test_plot_mean_timecourse_clips_the_band_at_the_axis_bottom() -> None:
    batch = escalation()
    fig = plot_mean_timecourse(batch, by="dose", panels=("log",))
    ax = fig.axes[0]
    bottom = ax.get_ylim()[0]
    assert bottom > 0.0
    for band in ax.collections:
        vertices = np.asarray(band.get_paths()[0].vertices, dtype=float)
        assert float(vertices[:, 1].min()) >= bottom
    matplotlib.pyplot.close(fig)


def study_batch(jitter: float = 0.1) -> Timecourses:
    """Eight subjects of two arms, sampled beside the nominal schedule."""
    nominal = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    rng = np.random.default_rng(7)
    actual = nominal + rng.normal(0.0, jitter, size=(8, nominal.size))
    actual[:, 0] = 0.0
    ke = rng.uniform(0.1, 0.2, 8)
    values = 10.0 * (np.exp(-ke[:, None] * actual) - np.exp(-2.0 * actual))
    values[4:] *= 1.5
    return Timecourses.from_arrays(
        actual,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={
            "individual": [f"s{i}" for i in range(8)],
            "arm": ("individual", np.array(["A"] * 4 + ["B"] * 4)),
        },
        nominal_time=nominal,
        dose={"amount": np.full(8, 100.0), "unit": "mg"},
        route=Route.ORAL,
        substance="drug",
    )


def x_data(line: matplotlib.lines.Line2D) -> np.ndarray:
    """The x values of a line as a float array."""
    return np.asarray(line.get_xdata(), dtype=float)


def test_plot_study_curves_has_four_panels_on_actual_and_nominal_times() -> None:
    batch = study_batch()
    fig = plot_study_curves(batch)
    assert len(fig.axes) == 4
    titles = [ax.get_title() for ax in fig.axes]
    assert titles[0] == "individuals, linear"
    assert titles[1] == "individuals, semi-logarithmic"
    assert all(title.startswith("mean on the nominal times") for title in titles[2:])
    assert fig.axes[1].get_yscale() == "log" and fig.axes[3].get_yscale() == "log"
    assert fig.axes[0].get_yscale() == "linear"
    times = batch.nominal_times
    assert times is not None
    nominal = np.unique(times)
    # the individual panels draw the times the samples were taken at, which no
    # two subjects share, the mean panels the eight nominal times
    individual = x_data(fig.axes[0].get_lines()[0])
    assert not np.allclose(individual, nominal)
    mean_line = fig.axes[2].get_lines()[-1]
    np.testing.assert_allclose(x_data(mean_line), nominal)
    assert fig.axes[0].get_xlabel() == "time [hr]"
    matplotlib.pyplot.close(fig)


def test_plot_study_curves_colors_the_groups_the_same_in_every_panel() -> None:
    batch = study_batch()
    fig = plot_study_curves(batch, by="arm")
    colors = []
    for ax in (fig.axes[0], fig.axes[2]):
        labelled = [
            line
            for line in ax.get_lines()
            if str(line.get_label()).startswith(("A", "B"))
        ]
        colors.append([line.get_color() for line in labelled])
    assert len(colors[0]) == 2 and len(colors[1]) == 2
    assert colors[0] == colors[1]
    # both rows name the groups the same way, the mean row with the number of
    # subjects behind the mean
    for ax, entries in (
        (fig.axes[0], ["A", "B"]),
        (fig.axes[2], ["A (n = 4)", "B (n = 4)"]),
    ):
        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == entries
        assert legend.get_title().get_text() == "arm"
    matplotlib.pyplot.close(fig)


def test_plot_study_curves_names_a_numeric_group_with_its_unit() -> None:
    batch = study_batch()
    dose = np.where(np.arange(8) < 4, 50.0, 100.0)
    ds = batch.ds.assign_coords(dose=("individual", dose))
    fig = plot_study_curves(Timecourses(ds), by="dose")
    for ax in (fig.axes[0], fig.axes[2]):
        legend = ax.get_legend()
        assert legend is not None
        assert legend.get_title().get_text() == "dose"
        labels = [text.get_text() for text in legend.get_texts()]
        assert labels[0].startswith("50 mg") and labels[1].startswith("100 mg")
    # no legend at all below the limit
    bare = plot_study_curves(Timecourses(ds), by="dose", max_legend=1)
    assert bare.axes[0].get_legend() is None
    matplotlib.pyplot.close("all")


def test_plot_study_curves_without_the_logarithmic_panels() -> None:
    batch = study_batch()
    fig = plot_study_curves(batch, log_y_panels=False)
    assert len(fig.axes) == 2
    assert [ax.get_yscale() for ax in fig.axes] == ["linear", "linear"]
    matplotlib.pyplot.close(fig)


def test_plot_study_curves_maps_the_times_to_the_nearest_nominal_time() -> None:
    batch = study_batch()
    without = Timecourses(batch.ds.drop_vars("nominal_time"))
    assert without.nominal_times is None
    schedule = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
    fig = plot_study_curves(without, nominal_times=schedule)
    np.testing.assert_allclose(x_data(fig.axes[2].get_lines()[-1]), schedule)
    # without a schedule the actual times are the grid, so the mean curve of a
    # batch whose subjects were sampled at different times is ragged
    bare = plot_study_curves(without)
    assert (
        x_data(fig.axes[2].get_lines()[-1]).size
        < x_data(bare.axes[2].get_lines()[-1]).size
    )
    matplotlib.pyplot.close("all")


def test_plot_study_curves_draws_into_the_given_axes() -> None:
    batch = study_batch()
    fig, axes = matplotlib.pyplot.subplots(nrows=2, ncols=2)
    same = plot_study_curves(batch, by="arm", axes=axes.ravel())
    assert same is fig
    assert all(ax.get_lines() for ax in axes.ravel())
    matplotlib.pyplot.close("all")
