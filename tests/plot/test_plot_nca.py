import matplotlib
import matplotlib.lines
import matplotlib.pyplot
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from pkpdutils import Dose, Dosing, Route, Timecourse, Timecourses
from pkpdutils.nca import (
    Acceptance,
    AUCMethod,
    NCAOptions,
    TerminalPhase,
    nca,
    nca_single,
)
from pkpdutils.plot import (
    DEFAULT_STYLE,
    PlotStyle,
    draw_nca_panel,
    plot_intervals,
    plot_nca,
    plot_nca_grid,
    plot_terminal_windows,
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
    # the linear panel, the logarithmic panel and the parameter table
    assert len(fig.axes) == 3
    assert fig.axes[1].get_yscale() == "log"
    assert fig.get_suptitle() == "test"
    assert len(plot_nca(tc, result, annotate=False).axes) == 2
    matplotlib.pyplot.close("all")


def test_plot_nca_titles_the_figure_once_and_names_the_scales() -> None:
    tc = oral()
    fig = plot_nca(tc, nca_single(tc), title="subject 1")
    # the sample is named once, over the figure, not in both panels
    assert fig.get_suptitle() == "subject 1"
    assert [ax.get_title() for ax in fig.axes[:2]] == ["linear", "semi-logarithmic"]
    assert fig.axes[2].get_title().startswith("parameters")
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
    assert any(
        text.get_text().startswith("C0 = ") and "ng/ml" in text.get_text()
        for text in fig.axes[0].texts
    ) or any(text.get_text().startswith("C0 = ") for text in fig.axes[0].texts)
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


def test_plot_troughs_names_the_statistic_in_the_title() -> None:
    batch = trough_batch()
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    assert plot_troughs(result).axes[0].get_title() == "mean ± sd over individual"
    assert (
        plot_troughs(result, spread="se", by="arm").axes[0].get_title()
        == "mean ± se over individual"
    )
    assert (
        plot_troughs(result, spread=None).axes[0].get_title() == "mean over individual"
    )
    # a result of a single curve draws that curve, there is no statistic
    single = nca_single(
        multiple_dose_tc(n_doses=3), options=NCAOptions(auc_method=AUCMethod.LOG)
    )
    assert plot_troughs(single).axes[0].get_title() == ""
    # and `plot_intervals` draws one line per sample, not a reduction
    assert plot_intervals(result, "interval_ctrough").axes[0].get_title() == ""
    matplotlib.pyplot.close("all")


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


def individual() -> Timecourse:
    """An individual oral curve, no spread, so no uncertainty analysis."""
    t = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 24])
    return Timecourse(
        time=t,
        value=[0.9, 1.7, 2.6, 2.9, 2.8, 2.5, 2.2, 1.6, 1.2, 0.6, 0.1],
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        substance="caffeine",
    )


def group_curve() -> Timecourse:
    return Timecourse(
        time=[0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 24],
        value=[0.9, 1.7, 2.6, 2.9, 2.8, 2.5, 2.2, 1.6, 1.2, 0.6, 0.1],
        sd=[0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.2, 0.1, 0.03],
        n=12,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        substance="caffeine",
    )


def test_plot_nca_annotates_the_parameters_on_the_plot() -> None:
    tc = group_curve()
    fig = plot_nca(tc, nca_single(tc))
    texts = [text.get_text() for text in fig.axes[0].texts]
    assert any(t.startswith("Cmax = 2.9") and "tmax = 1.5 hr" in t for t in texts)
    assert any(
        t.startswith("lambda_z = ") and "t1/2 = " in t and "n = " in t for t in texts
    )
    assert any(t.startswith("clast = ") and "tlast = 24 hr" in t for t in texts)
    # the areas carry their values, the tail its share
    assert any(t.startswith("AUC(0-tlast) = 22.6") and "h⋅mg/l" in t for t in texts)
    assert any(t.startswith("AUC(tlast-inf) = ") and "%)" in t for t in texts)
    # the intervals of the uncertainty analysis are in the annotation
    assert any("Cmax = 2.9 [" in t for t in texts)
    # the logarithmic panel names the peak as well, with a leader line
    log_texts = [text.get_text() for text in fig.axes[1].texts]
    assert any(t.startswith("Cmax = 2.9") for t in log_texts)
    # the parameter table of the third panel
    table = fig.axes[2].texts[0].get_text()
    assert "cmax" in table and "mg/l" in table and "[" in table
    assert "extrapolated" in table and "thalf" in table and "cl_f" in table
    assert fig.axes[2].get_title() == "parameters, 95 % interval"
    # nothing on the plot without annotations
    bare = plot_nca(tc, nca_single(tc), annotate=False)
    assert not bare.axes[0].texts
    matplotlib.pyplot.close("all")


def test_plot_nca_draws_the_spread_of_the_data() -> None:
    tc = group_curve()
    result = nca_single(tc)
    fig = plot_nca(tc, result)
    containers = [c for c in fig.axes[0].containers if isinstance(c, ErrorbarContainer)]
    assert len(containers) == 1 and containers[0].get_label() == "data ± sd"
    none = plot_nca(tc, result, spread=None)
    assert not [c for c in none.axes[0].containers if isinstance(c, ErrorbarContainer)]
    # a curve without a spread draws no error bars either
    plain = individual()
    bare = plot_nca(plain, nca_single(plain))
    assert not [c for c in bare.axes[0].containers if isinstance(c, ErrorbarContainer)]
    matplotlib.pyplot.close("all")


def band_of(ax) -> PolyCollection | None:
    for collection in ax.collections:
        if isinstance(collection, PolyCollection) and "band" in collection.get_label():
            return collection
    return None


def test_plot_nca_draws_the_confidence_band_of_the_regression() -> None:
    tc = oral()
    result = nca_single(tc)
    fig = plot_nca(tc, result)
    band = band_of(fig.axes[0])
    assert band is not None and band.get_label() == "95 % band of the regression"
    # the band widens away from the centre of the regression window
    vertices = np.asarray(band.get_paths()[0].vertices)
    values = _band_width(tc, result, 0.95)
    assert values[-1] > values[0] > 0
    assert vertices.shape[0] > 10
    # a wider level gives a wider band
    assert _band_width(tc, result, 0.99)[-1] > values[-1]
    # a regression on two points has no band
    two = Timecourse(time=[1, 2, 4, 8], value=[4, 3, 2, 1], time_unit="hr", unit="mg/l")
    from pkpdutils import TerminalMethod, TerminalPhase

    short = nca_single(
        two,
        options=NCAOptions(
            terminal=TerminalPhase(method=TerminalMethod.LAST_N, n_points=3)
        ),
    )
    assert band_of(plot_nca(two, short).axes[0]) is not None
    matplotlib.pyplot.close("all")


def _band_width(tc: Timecourse, result, ci_level: float) -> np.ndarray:
    from pkpdutils.plot.nca import _terminal_band

    values = {name: float(v.magnitude) for name, v in result.to_quantities().items()}
    grid = np.linspace(
        values["lambda_z_t_first"], values["tlast"] + 3 * values["thalf"], 5
    )
    band = _terminal_band(
        np.asarray(tc.time), np.asarray(tc.value), values, grid, ci_level
    )
    assert band is not None
    return np.log(band[1]) - np.log(band[0])


def test_thalf_interval_follows_the_regression_without_an_uncertainty_analysis() -> (
    None
):
    from scipy.stats import t as student_t

    from pkpdutils.plot.nca import _thalf_interval, parameter_rows

    tc = individual()
    result = nca_single(tc)
    values = {name: float(v.magnitude) for name, v in result.to_quantities().items()}
    low, high = _thalf_interval(values, 0.95)
    n = int(values["lambda_z_n_points"])
    q = student_t.ppf(0.975, n - 2)
    assert np.isclose(
        low, np.log(2) / (values["lambda_z"] + q * values["lambda_z_stderr"])
    )
    assert np.isclose(
        high, np.log(2) / (values["lambda_z"] - q * values["lambda_z_stderr"])
    )
    assert low < values["thalf"] < high
    rows = parameter_rows(
        values, {n: result.units(n) for n in values}, ("thalf", "cmax"), 0.95
    )
    assert rows[0][0] == "thalf" and rows[0][2].startswith("[") and rows[0][3] == "h"
    assert rows[1] == ("cmax", f"{values['cmax']:.3g}", "", "mg/l")
    # an uncertainty analysis supplies the interval itself
    both = dict(values, thalf_ci_low=1.0, thalf_ci_high=2.0)
    assert _thalf_interval(both, 0.95) == (1.0, 2.0)
    matplotlib.pyplot.close("all")


def test_plot_nca_with_three_axes_draws_the_table_into_the_third() -> None:
    tc = oral()
    _, axes = matplotlib.pyplot.subplots(ncols=3)
    plot_nca(tc, nca_single(tc), axes=axes)
    assert axes[2].get_title().startswith("parameters") and axes[2].texts
    fig2, two = matplotlib.pyplot.subplots(ncols=2)
    plot_nca(tc, nca_single(tc), axes=two)
    assert len(fig2.axes) == 2
    matplotlib.pyplot.close("all")


def test_plot_nca_grid_has_the_band_but_no_annotations_by_default() -> None:
    batch = Timecourses.from_timecourses([oral(1.0, "a"), oral(4.0, "b")])
    fig = plot_nca_grid(batch, nca(batch))
    assert band_of(fig.axes[0]) is not None
    assert not fig.axes[0].texts
    annotated = plot_nca_grid(batch, nca(batch), annotate=True)
    assert annotated.axes[0].texts
    matplotlib.pyplot.close("all")


def terminal_options(acceptance: Acceptance | None = None) -> NCAOptions:
    """Options which keep the candidate windows of the terminal regression."""
    return NCAOptions(
        terminal=TerminalPhase(keep_candidates=True),
        acceptance=acceptance or Acceptance(),
    )


def x_data(line: matplotlib.lines.Line2D) -> np.ndarray:
    """The x values of a line as a float array."""
    return np.asarray(line.get_xdata(), dtype=float)


def y_data(line: matplotlib.lines.Line2D) -> np.ndarray:
    """The y values of a line as a float array."""
    return np.asarray(line.get_ydata(), dtype=float)


def area_x(patch: PolyCollection) -> np.ndarray:
    """The x range of a filled area."""
    return np.asarray(patch.get_paths()[0].vertices, dtype=float)[:, 0]


def test_plot_terminal_windows_draws_one_marker_per_candidate() -> None:
    tc = oral()
    options = terminal_options()
    result = nca_single(tc, options=options)
    fig = plot_terminal_windows(tc, result, options=options)
    assert len(fig.axes) == 2
    curve_panel, candidates = fig.axes
    n_candidates = int(result.ds.sizes["candidate"])
    assert n_candidates >= 3
    line = next(
        line
        for line in candidates.get_lines()
        if line.get_label() == "candidate window"
    )
    np.testing.assert_allclose(x_data(line), result.ds["candidate_t_first"].to_numpy())
    np.testing.assert_allclose(y_data(line), result.ds["candidate_r2_adj"].to_numpy())
    # the chosen window is marked once, at the start time of the regression
    chosen = next(
        line for line in candidates.get_lines() if line.get_label() == "chosen window"
    )
    assert list(x_data(chosen)) == [float(result["lambda_z_t_first"].values)]
    # the number of points of every window is written above its marker
    assert len(candidates.texts) == n_candidates
    assert {text.get_text() for text in candidates.texts} == {
        str(int(n)) for n in result.ds["candidate_n_points"].to_numpy()
    }
    # the curve panel is logarithmic and brackets the window with two lines
    assert curve_panel.get_yscale() == "log"
    bounds = [
        float(x_data(line)[0])
        for line in curve_panel.get_lines()
        if line.get_label() == "chosen window"
    ]
    assert bounds == [float(result["lambda_z_t_first"].values)]
    matplotlib.pyplot.close(fig)


def test_plot_terminal_windows_draws_the_acceptance_threshold() -> None:
    tc = oral()
    options = terminal_options(acceptance=Acceptance(r2_adj_min=0.9))
    result = nca_single(tc, options=options)
    with_threshold = plot_terminal_windows(tc, result, options=options)
    labels = [line.get_label() for line in with_threshold.axes[1].get_lines()]
    assert any(str(label).startswith("acceptance") for label in labels)
    lines = [
        line
        for line in with_threshold.axes[1].get_lines()
        if str(line.get_label()).startswith("acceptance")
    ]
    np.testing.assert_allclose(y_data(lines[0]), [0.9, 0.9])
    # no threshold without the options, and none without an acceptance rule
    without = plot_terminal_windows(tc, result)
    assert not any(
        str(line.get_label()).startswith("acceptance")
        for line in without.axes[1].get_lines()
    )
    plain = plot_terminal_windows(tc, result, options=terminal_options())
    assert not any(
        str(line.get_label()).startswith("acceptance")
        for line in plain.axes[1].get_lines()
    )
    matplotlib.pyplot.close("all")


def test_plot_terminal_windows_takes_the_axes_and_the_sample_of_a_batch() -> None:
    batch = Timecourses.from_timecourses([oral(2.0, "a")], labels=["a"])
    options = terminal_options()
    result = nca(batch, options=options)
    fig, axes = matplotlib.pyplot.subplots(ncols=2)
    same = plot_terminal_windows(
        batch.sel(individual="a"), result, axes=axes, individual="a", options=options
    )
    assert same is fig
    assert len(axes[1].get_lines()) >= 2
    matplotlib.pyplot.close("all")


def test_plot_terminal_windows_raises_without_the_candidate_windows() -> None:
    tc = oral()
    result = nca_single(tc)
    with pytest.raises(ValueError, match="keep_candidates"):
        plot_terminal_windows(tc, result)
    matplotlib.pyplot.close("all")


def test_plot_terminal_windows_raises_for_a_batch_of_several_samples() -> None:
    batch = Timecourses.from_timecourses([oral(2.0, "a"), oral(3.0, "b")])
    result = nca(batch, options=terminal_options())
    with pytest.raises(ValueError, match="keep_candidates"):
        plot_terminal_windows(batch.sel(individual="a"), result, individual="a")
    matplotlib.pyplot.close("all")


def partial_polygon(ax: Axes, label: str) -> PolyCollection:
    """The filled area of `ax` carrying `label`."""
    return next(
        artist
        for artist in ax.collections
        if isinstance(artist, PolyCollection) and artist.get_label() == label
    )


def test_plot_nca_shades_a_named_partial_area() -> None:
    tc = oral()
    options = NCAOptions(partial_aucs={"auc_0_12": (0.0, 12.0)})
    result = nca_single(tc, options=options)
    fig = plot_nca(tc, result, partial="auc_0_12")
    for ax in fig.axes[:2]:
        patch = partial_polygon(ax, "auc_0_12")
        x = area_x(patch)
        assert x.min() == pytest.approx(0.0) and x.max() == pytest.approx(12.0)
        color = np.asarray(patch.get_facecolor(), dtype=float)[0]
        assert matplotlib.colors.to_hex(color[:3]) == DEFAULT_STYLE.partial_color
    value = float(result["auc_0_12"].values)
    texts = [text.get_text() for text in fig.axes[0].texts]
    assert any(t == f"auc_0_12 = {value:.3g} h⋅mg/l" for t in texts)
    # the area to the last point is still shaded, the partial area sits on it
    assert partial_polygon(fig.axes[0], "AUC(0-tlast)") is not None
    matplotlib.pyplot.close(fig)


def test_plot_nca_grid_shades_the_partial_area_of_every_panel() -> None:
    batch = Timecourses.from_timecourses([oral(2.0, "a"), oral(3.0, "b")])
    options = NCAOptions(partial_aucs={"auc_2_8": (2.0, 8.0)})
    result = nca(batch, options=options)
    fig = plot_nca_grid(batch, result, partial="auc_2_8")
    for ax in fig.axes[:2]:
        x = area_x(partial_polygon(ax, "auc_2_8"))
        assert x.min() == pytest.approx(2.0) and x.max() == pytest.approx(8.0)
    matplotlib.pyplot.close(fig)


def test_plot_nca_raises_for_an_unknown_partial_area() -> None:
    tc = oral()
    result = nca_single(tc, options=NCAOptions(partial_aucs={"auc_0_12": (0.0, 12.0)}))
    with pytest.raises(ValueError, match="auc_0_24"):
        plot_nca(tc, result, partial="auc_0_24")
    matplotlib.pyplot.close("all")


def test_draw_nca_panel_needs_the_range_of_a_partial_area() -> None:
    tc = oral()
    result = nca_single(tc, options=NCAOptions(partial_aucs={"auc_0_12": (0.0, 12.0)}))
    values = {name: float(result[name].values) for name in result.parameters}
    with pytest.raises(ValueError, match="partial_range"):
        draw_nca_panel(tc, values, [], partial="auc_0_12")
    matplotlib.pyplot.close("all")


def test_the_partial_area_of_a_multiple_dose_curve_sits_in_its_interval() -> None:
    # the interval is relative to the first dose, the panel to the last one
    tc = multiple_dose_tc(n_doses=3, tau=12.0)
    options = NCAOptions(
        auc_method=AUCMethod.LOG, partial_aucs={"auc_24_36": (24.0, 36.0)}
    )
    result = nca_single(tc, options=options)
    fig = plot_nca(tc, result, partial="auc_24_36")
    x = area_x(partial_polygon(fig.axes[0], "auc_24_36"))
    assert x.min() == pytest.approx(0.0) and x.max() == pytest.approx(12.0)
    matplotlib.pyplot.close(fig)


def test_a_partial_area_past_the_last_point_follows_the_terminal_regression() -> None:
    tc = oral()  # the last sample is at 24 h
    options = NCAOptions(partial_aucs={"auc_0_48": (0.0, 48.0)})
    result = nca_single(tc, options=options)
    fig = plot_nca(tc, result, partial="auc_0_48")
    vertices = np.asarray(
        partial_polygon(fig.axes[0], "auc_0_48").get_paths()[0].vertices, dtype=float
    )
    x, y = vertices[:, 0], vertices[:, 1]
    assert x.max() == pytest.approx(48.0)
    # the shaded tail is the regression, not the last value carried forward
    lambda_z = float(result["lambda_z"].values)
    clast_pred = float(result["clast_pred"].values)
    tail = (x > 30.0) & (y > 0.0)
    np.testing.assert_allclose(
        y[tail], clast_pred * np.exp(-lambda_z * (x[tail] - 24.0)), rtol=1e-6
    )
    matplotlib.pyplot.close(fig)


def test_the_partial_annotation_stays_inside_the_panel() -> None:
    tc = oral()  # the peak sits at 2 h of a panel which runs to about 40 h
    early = nca_single(tc, options=NCAOptions(partial_aucs={"auc_0_6": (0.0, 6.0)}))
    late = nca_single(tc, options=NCAOptions(partial_aucs={"auc_8_24": (8.0, 24.0)}))
    for result, name, align in (
        (early, "auc_0_6", "left"),
        (late, "auc_8_24", "center"),
    ):
        fig = plot_nca(tc, result, partial=name)
        text = next(t for t in fig.axes[0].texts if t.get_text().startswith(name))
        assert text.get_horizontalalignment() == align
        matplotlib.pyplot.close(fig)
