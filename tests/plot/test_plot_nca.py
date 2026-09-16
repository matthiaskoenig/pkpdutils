import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from pkpdutils import Dose, Dosing, Route, Timecourse, Timecourses
from pkpdutils.nca import AUCMethod, NCAOptions, nca, nca_single
from pkpdutils.plot import (
    PlotStyle,
    plot_intervals,
    plot_nca,
    plot_nca_grid,
    plot_timecourse,
    plot_troughs,
)

matplotlib.use("Agg")


def multiple_dose_tc(
    n_doses: int = 3, tau: float = 12.0, c0: float = 10.0, k: float = 0.2
) -> Timecourse:
    """Superposition of a mono-exponential bolus curve given every `tau` hours."""
    time = np.sort(
        np.concatenate([np.arange(0, n_doses * tau + 0.01, 0.5), [n_doses * tau + 24]])
    )
    value = np.zeros_like(time)
    for k_dose in range(n_doses):
        shifted = time - k_dose * tau
        value += np.where(shifted >= 0, c0 * np.exp(-k * shifted), 0.0)
    dosing = Dosing.regimen(
        Dose(amount=100, unit="mg", route=Route.IV_BOLUS), interval=tau, n_doses=n_doses
    )
    return Timecourse(
        time=time, value=value, time_unit="hr", unit="mg/l", dosing=dosing
    )


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
    fig2 = plot_timecourse(batch, log_y=True, errorbars=False)
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
    assert fig.get_suptitle() == "test"
    matplotlib.pyplot.close(fig)


def test_plot_nca_titles_the_figure_once_and_names_the_scales() -> None:
    tc = oral()
    fig = plot_nca(tc, nca_single(tc), title="subject 1")
    # the sample is named once, over the figure, not in both panels
    assert fig.get_suptitle() == "subject 1"
    assert [ax.get_title() for ax in fig.axes] == ["linear", "semi-logarithmic"]
    matplotlib.pyplot.close(fig)


def test_plot_nca_with_given_axes_titles_the_first_panel() -> None:
    tc = oral()
    fig, axes = matplotlib.pyplot.subplots(ncols=2)
    fig.suptitle("the title of the caller")
    plot_nca(tc, nca_single(tc), title="subject 1", axes=axes)
    # a figure of the caller keeps its own title
    assert fig.get_suptitle() == "the title of the caller"
    assert axes[0].get_title() == "subject 1"
    matplotlib.pyplot.close(fig)


def test_plot_nca_from_batch_result_with_indexers_and_flags_in_title() -> None:
    batch = Timecourses.from_timecourses([oral(1.0, "slow"), oral(4.0, "fast")])
    result = nca(batch)
    fig = plot_nca(batch.sel(individual="fast"), result, individual="fast")
    assert "fast" in fig.get_suptitle()
    short = Timecourse(time=[1, 2, 3], value=[1, 3, 2], time_unit="hr", unit="mg/l")
    fig2 = plot_nca(short, nca_single(short))
    assert "TOO_FEW_POINTS" in fig2.get_suptitle()
    matplotlib.pyplot.close("all")


def test_plot_nca_titles_the_figure_with_the_label_of_the_curve() -> None:
    tc = oral(label="healthy volunteers")
    fig = plot_nca(tc, nca_single(tc))
    assert fig.get_suptitle() == "healthy volunteers"
    matplotlib.pyplot.close(fig)


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
    fig = plot_nca_grid(batch, nca(batch, options=NCAOptions()), ncols=2)
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 4
    matplotlib.pyplot.close(fig)


def test_plot_timecourse_by_broadcasts_coordinate_to_sample_dims() -> None:
    n_dose, n_individual, n_time = 2, 3, 4
    time = np.array([0.5, 1.0, 2.0, 4.0])
    values = np.arange(n_dose * n_individual * n_time, dtype=float).reshape(
        n_dose, n_individual, n_time
    )
    tcs = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("dose", "individual"),
        coords={"dose": ["low", "high"], "individual": [0, 1, 2]},
    )
    # "group" is assigned with dims in the opposite order of the sample
    # dimensions ("dose", "individual") of the batch, to catch a naive
    # flatten that ignores the coordinate's own dimension order.
    group = np.array([[f"g{i}{d}" for d in range(n_dose)] for i in range(n_individual)])
    tcs.ds = tcs.ds.assign_coords(group=(("individual", "dose"), group))
    expected = [str(group[i, d]) for d in range(n_dose) for i in range(n_individual)]
    fig = plot_timecourse(tcs, by="group")
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert labels == expected
    matplotlib.pyplot.close(fig)


def test_plot_nca_breaks_line_at_nan_without_bridging() -> None:
    tc = oral()
    value = tc.value.copy()
    value[4] = np.nan
    tc_nan = tc.model_copy(update={"value": value})
    fig = plot_nca(tc_nan, nca_single(tc_nan))
    data_line = next(
        line for line in fig.axes[0].get_lines() if line.get_label() == "data"
    )
    # the raw (unmasked) values are plotted, matplotlib breaks the line at the
    # NaN itself instead of the previous, gap-bridging masked plot
    assert np.isnan(data_line.get_ydata()).any()
    matplotlib.pyplot.close(fig)


def test_plot_nca_multiple_dose_shades_the_steady_state_interval() -> None:
    tc = multiple_dose_tc()
    result = nca_single(tc, options=NCAOptions(auc_method=AUCMethod.LOG))
    fig = plot_nca(tc, result)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert any("tau" in label for label in legend_labels)
    matplotlib.pyplot.close(fig)


def test_plot_intervals_one_line_per_sample() -> None:
    a = multiple_dose_tc(n_doses=3, c0=10.0)
    b = multiple_dose_tc(n_doses=3, c0=20.0)
    batch = Timecourses.from_timecourses([a, b], labels=["one", "two"])
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    fig = plot_intervals(result)
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 2
    assert ax.get_ylabel().startswith("interval_auc [")
    assert ax.get_xlabel() == "interval"
    fig.canvas.draw()
    tick_labels = [
        label.get_text() for label in ax.xaxis.get_ticklabels() if label.get_text()
    ]
    assert tick_labels and all("." not in label for label in tick_labels), tick_labels
    matplotlib.pyplot.close(fig)


def test_plot_intervals_one_sample_with_indexers() -> None:
    a = multiple_dose_tc(n_doses=3, c0=10.0)
    b = multiple_dose_tc(n_doses=3, c0=20.0)
    batch = Timecourses.from_timecourses([a, b], labels=["one", "two"])
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    fig = plot_intervals(result, individual="two")
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 1
    matplotlib.pyplot.close(fig)


def test_plot_intervals_raises_without_intervals() -> None:
    tc = oral()
    result = nca_single(tc)
    with pytest.raises(ValueError):
        plot_intervals(result)


def test_plot_intervals_raises_for_a_non_interval_name() -> None:
    tc = multiple_dose_tc(n_doses=3)
    result = nca_single(tc, options=NCAOptions(auc_method=AUCMethod.LOG))
    with pytest.raises(ValueError):
        plot_intervals(result, name="cmax")


def trough_batch(n: int = 4) -> Timecourses:
    """A batch of bolus curves of three dosing intervals, two arms."""
    curves = [multiple_dose_tc(n_doses=3, c0=10.0 + 2.0 * i) for i in range(n)]
    batch = Timecourses.from_timecourses(curves, labels=[f"s{i}" for i in range(n)])
    arms = ["A" if i < n // 2 else "B" for i in range(n)]
    batch.ds = batch.ds.assign_coords(arm=("individual", arms))
    return batch


def infusion_tc(duration: float = 2.0) -> Timecourse:
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
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


def test_draw_nca_panel_draws_an_infusion_as_a_window() -> None:
    tc = infusion_tc()
    fig = plot_nca(tc, nca_single(tc))
    spans = [
        patch
        for patch in fig.axes[0].patches
        if str(patch.get_label()).startswith("infusion")
    ]
    assert len(spans) == 1
    assert spans[0].get_label() == "infusion (2 hr)"
    span = spans[0]
    assert isinstance(span, Rectangle)
    assert (span.get_x(), span.get_width()) == (0.0, 2.0)
    # the window sits behind the shaded areas and the data
    assert span.get_zorder() == 0.0
    matplotlib.pyplot.close(fig)


def test_plot_nca_draws_the_legend_once() -> None:
    tc = oral()
    fig = plot_nca(tc, nca_single(tc))
    assert fig.axes[0].get_legend() is not None
    assert fig.axes[1].get_legend() is None
    matplotlib.pyplot.close(fig)


def test_plot_nca_grid_titles_name_the_coordinates_with_units() -> None:
    time = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    doses = np.array([50.0, 100.0])
    values = np.stack(
        [np.stack([d / 50.0 * np.exp(-0.2 * time)]) for d in doses]
    )  # (dose, individual, time)
    batch = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("dose", "individual"),
        coords={"dose": doses, "individual": ["s1"]},
        dose={"amount": doses[:, None], "unit": "mg"},
        route=Route.ORAL,
        substance="caffeine",
    )
    fig = plot_nca_grid(batch, nca(batch, options=NCAOptions()), ncols=2)
    titles = [ax.get_title() for ax in fig.axes if ax.get_visible()]
    assert titles[0].startswith("dose = 50 mg, individual = s1")
    assert titles[1].startswith("dose = 100 mg, individual = s1")
    matplotlib.pyplot.close(fig)


def test_plot_nca_grid_draws_one_legend_for_the_figure() -> None:
    batch = Timecourses.from_timecourses([oral(k, str(k)) for k in (1.0, 2.0)])
    fig = plot_nca_grid(batch, nca(batch, options=NCAOptions()), ncols=2)
    assert all(ax.get_legend() is None for ax in fig.axes)
    assert len(fig.legends) == 1
    labels = [text.get_text() for text in fig.legends[0].get_texts()]
    # one entry per artist, the regression line of one curve not naming its
    # own lambda_z in a legend of the whole figure
    assert "terminal regression" in labels
    assert len(labels) == len(set(labels))
    matplotlib.pyplot.close(fig)


def test_plot_nca_grid_with_given_axes_puts_the_legend_in_the_first_panel() -> None:
    batch = Timecourses.from_timecourses([oral(k, str(k)) for k in (1.0, 2.0)])
    fig, axes = matplotlib.pyplot.subplots(ncols=2)
    assert plot_nca_grid(batch, nca(batch, options=NCAOptions()), axes=axes) is fig
    assert not fig.legends
    assert axes[0].get_legend() is not None
    assert axes[1].get_legend() is None
    assert fig.get_layout_engine() is None
    matplotlib.pyplot.close(fig)


def test_plot_nca_grid_clamps_the_columns_to_the_sample_count() -> None:
    batch = Timecourses.from_timecourses([oral(1.0, "a")])
    fig = plot_nca_grid(batch, nca(batch, options=NCAOptions()), ncols=3)
    assert len(fig.axes) == 1
    matplotlib.pyplot.close(fig)


def test_plot_troughs_against_the_time_with_groups() -> None:
    batch = trough_batch()
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    fig = plot_troughs(result, by="arm")
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "time [h]"  # the long "hour" of a result, shortened
    assert ax.get_ylabel().startswith("trough [")
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == "arm"
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["A, ctrough", "A, cmin", "B, ctrough", "B, cmin"]
    # the troughs are the ends of the intervals, not the dose times
    ctrough = next(bar for bar in ax.containers if bar.get_label() == "A, ctrough")
    assert isinstance(ctrough, ErrorbarContainer)
    np.testing.assert_allclose(
        np.asarray(ctrough.lines[0].get_xdata()), [12.0, 24.0, 36.0]
    )
    matplotlib.pyplot.close(fig)


def test_plot_troughs_against_the_interval_number() -> None:
    batch = trough_batch()
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    fig = plot_troughs(result, x="interval", spread="se")
    ax = fig.axes[0]
    assert ax.get_xlabel() == "interval"
    ctrough = next(bar for bar in ax.containers if bar.get_label() == "ctrough")
    assert isinstance(ctrough, ErrorbarContainer)
    np.testing.assert_allclose(
        np.asarray(ctrough.lines[0].get_xdata()), [1.0, 2.0, 3.0]
    )
    fig.canvas.draw()
    tick_labels = [
        label.get_text() for label in ax.xaxis.get_ticklabels() if label.get_text()
    ]
    assert all("." not in label for label in tick_labels), tick_labels
    matplotlib.pyplot.close(fig)


def test_plot_troughs_draws_into_the_given_ax_and_without_a_spread() -> None:
    batch = trough_batch()
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    fig, ax = matplotlib.pyplot.subplots()
    assert plot_troughs(result, spread=None, ax=ax) is fig
    assert ax.containers
    assert fig.get_layout_engine() is None
    matplotlib.pyplot.close(fig)


def test_plot_troughs_raises_without_intervals_and_for_an_unknown_axis() -> None:
    tc = oral()
    with pytest.raises(ValueError):
        plot_troughs(nca_single(tc))
    batch = trough_batch()
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    with pytest.raises(ValueError):
        plot_troughs(result, x="dose")  # ty: ignore[invalid-argument-type]


def test_draw_nca_panel_marks_the_analysed_dose_only() -> None:
    # the panel starts at the last dose, so the earlier doses of the protocol
    # fall outside it and no invisible line is drawn for them
    tc = multiple_dose_tc(n_doses=3)
    fig = plot_nca(tc, nca_single(tc, options=NCAOptions(auc_method=AUCMethod.LOG)))
    dotted = [line for line in fig.axes[0].get_lines() if line.get_linestyle() == ":"]
    assert not dotted
    matplotlib.pyplot.close(fig)


def test_plot_troughs_places_every_group_on_its_own_interval_times() -> None:
    fast = [multiple_dose_tc(n_doses=3, tau=12.0) for _ in range(2)]
    slow = [multiple_dose_tc(n_doses=3, tau=24.0) for _ in range(2)]
    batch = Timecourses.from_timecourses(
        [*fast, *slow], labels=["f1", "f2", "s1", "s2"]
    )
    batch.ds = batch.ds.assign_coords(
        arm=("individual", ["fast", "fast", "slow", "slow"])
    )
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    fig = plot_troughs(result, by="arm")
    ax = fig.axes[0]
    fast_line = next(bar for bar in ax.containers if bar.get_label() == "fast, ctrough")
    slow_line = next(bar for bar in ax.containers if bar.get_label() == "slow, ctrough")
    assert isinstance(fast_line, ErrorbarContainer)
    assert isinstance(slow_line, ErrorbarContainer)
    np.testing.assert_allclose(
        np.asarray(fast_line.lines[0].get_xdata()), [12.0, 24.0, 36.0]
    )
    np.testing.assert_allclose(
        np.asarray(slow_line.lines[0].get_xdata()), [24.0, 48.0, 72.0]
    )
    matplotlib.pyplot.close(fig)
