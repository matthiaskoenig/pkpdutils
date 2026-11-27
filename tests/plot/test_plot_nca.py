from typing import Any

import numpy as np
import pytest
from sbmlsim_core.chart import Chart

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
    new_chart,
    plot_intervals,
    plot_nca,
    plot_nca_grid,
    plot_terminal_windows,
    plot_timecourse,
    plot_troughs,
)


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


def spec(chart: Chart) -> dict[str, Any]:
    """The spec of a chart with every point of its lines, unthinned."""
    chart.thinning = None
    return chart.spec()


def plots(chart: Chart) -> list[dict[str, Any]]:
    return spec(chart)["plots"]


def kinds(plot: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in plot["elements"] if e["kind"] == kind]


def labelled(plot: dict[str, Any], label: str) -> list[dict[str, Any]]:
    return [e for e in plot["elements"] if e.get("label") == label]


def labels(plot: dict[str, Any]) -> list[str]:
    """The labels of the elements of a plot, every one once, in order."""
    found: list[str] = []
    for e in plot["elements"]:
        if "label" in e and e["label"] not in found:
            found.append(e["label"])
    return found


def legend(plot: dict[str, Any]) -> list[str]:
    """The entries of the legend of a plot: its labels in the order of its legend."""
    if not plot["legend"]:
        return []
    order = [g for g in plot.get("legend_order") or [] if g in labels(plot)]
    return [*order, *(g for g in labels(plot) if g not in order)]


def texts(plot: dict[str, Any]) -> list[str]:
    return [e["text"] for e in kinds(plot, "text")]


def floats(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def test_plot_timecourse_single_and_batch() -> None:
    chart = plot_timecourse(oral())
    assert isinstance(chart, Chart)
    plot = plots(chart)[0]
    assert plot["x"]["label"] == "time [hr]"
    assert plot["y"]["label"] == "caffeine [mg/l]"
    batch = Timecourses.from_timecourses([oral(1.0, "slow"), oral(4.0, "fast")])
    second = plots(plot_timecourse(batch, log_y=True, errorbars=False))[0]
    assert second["y"]["scale"] == "log"
    assert "slow" in labels(second) and "fast" in labels(second)


def test_plot_nca_single() -> None:
    tc = oral()
    result = nca_single(tc)
    chart = plot_nca(tc, result, title="test")
    # the linear panel, the logarithmic panel and the parameter table
    assert len(plots(chart)) == 3
    assert plots(chart)[1]["y"]["scale"] == "log"
    assert spec(chart)["title"] == "test"
    assert len(plots(plot_nca(tc, result, annotate=False))) == 2


def test_plot_nca_titles_the_figure_once_and_names_the_scales() -> None:
    tc = oral()
    chart = plot_nca(tc, nca_single(tc), title="subject 1")
    # the sample is named once, over the figure, not in both panels
    assert spec(chart)["title"] == "subject 1"
    linear, log, table = plots(chart)
    assert [linear["title"], log["title"]] == ["linear", "semi-logarithmic"]
    assert table["title"].startswith("parameters")


def test_plot_nca_with_a_given_chart_titles_the_first_panel() -> None:
    tc = oral()
    chart = new_chart(11, 4, columns=2, title="the title of the caller")
    assert (
        plot_nca(tc, nca_single(tc), title="subject 1", annotate=False, chart=chart)
        is chart
    )
    # a chart of the caller keeps its own title
    assert spec(chart)["title"] == "the title of the caller"
    assert plots(chart)[0]["title"] == "subject 1"


def test_plot_nca_from_batch_result_with_indexers_and_flags_in_title() -> None:
    batch = Timecourses.from_timecourses([oral(1.0, "slow"), oral(4.0, "fast")])
    result = nca(batch)
    chart = plot_nca(batch.sel(individual="fast"), result, individual="fast")
    assert "fast" in spec(chart)["title"]
    short = Timecourse(time=[1, 2, 3], value=[1, 3, 2], time_unit="hr", unit="mg/l")
    assert "TOO_FEW_POINTS" in spec(plot_nca(short, nca_single(short)))["title"]


def test_plot_nca_titles_the_figure_with_the_label_of_the_curve() -> None:
    tc = oral(label="healthy volunteers")
    assert spec(plot_nca(tc, nca_single(tc)))["title"] == "healthy volunteers"


def test_plot_nca_iv_bolus_marks_c0() -> None:
    t = np.array([0.5, 1, 2, 4, 8])
    tc = Timecourse(
        time=t,
        value=10 * np.exp(-0.5 * t),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
    )
    plot = plots(plot_nca(tc, nca_single(tc), style=PlotStyle(fit_color="red")))[0]
    (c0,) = labelled(plot, "C0")
    assert c0["style"]["color"] == "#ff0000"
    assert c0["style"]["marker"]["type"] == "diamond"
    assert any(t.startswith("C0 = ") for t in texts(plot))


def test_plot_nca_grid() -> None:
    batch = Timecourses.from_timecourses(
        [oral(k, str(k)) for k in (1.0, 2.0, 3.0, 4.0)]
    )
    chart = plot_nca_grid(batch, nca(batch, options=NCAOptions()), ncols=2)
    assert [(p["row"], p["column"]) for p in plots(chart)] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


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
    tcs = Timecourses(tcs.ds.assign_coords(group=(("individual", "dose"), group)))
    expected = [str(group[i, d]) for d in range(n_dose) for i in range(n_individual)]
    plot = plots(plot_timecourse(tcs, by="group"))[0]
    assert [e["label"] for e in kinds(plot, "curve")] == expected


def test_plot_nca_breaks_line_at_nan_without_bridging() -> None:
    tc = oral()
    value = tc.value.copy()
    value[4] = np.nan
    tc_nan = tc.model_copy(update={"value": value})
    plot = plots(plot_nca(tc_nan, nca_single(tc_nan)))[0]
    (data,) = labelled(plot, "data")
    # the value which is not finite is a gap of the line, not bridged over
    assert np.isnan(floats(data["y"])).any()


def test_plot_nca_multiple_dose_shades_the_steady_state_interval() -> None:
    tc = multiple_dose_tc()
    result = nca_single(tc, options=NCAOptions(auc_method=AUCMethod.LOG))
    plot = plots(plot_nca(tc, result))[0]
    assert any("tau" in label for label in legend(plot))


def test_plot_intervals_one_line_per_sample() -> None:
    a = multiple_dose_tc(n_doses=3, c0=10.0)
    b = multiple_dose_tc(n_doses=3, c0=20.0)
    batch = Timecourses.from_timecourses([a, b], labels=["one", "two"])
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    plot = plots(plot_intervals(result))[0]
    assert len(kinds(plot, "curve")) == 2
    assert plot["y"]["label"].startswith("interval_auc [")
    assert plot["x"]["label"] == "interval"
    # integer ticks only, never a fractional interval
    assert plot["x"]["ticks"] and all(float(t).is_integer() for t in plot["x"]["ticks"])


def test_plot_intervals_one_sample_with_indexers() -> None:
    a = multiple_dose_tc(n_doses=3, c0=10.0)
    b = multiple_dose_tc(n_doses=3, c0=20.0)
    batch = Timecourses.from_timecourses([a, b], labels=["one", "two"])
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    plot = plots(plot_intervals(result, individual="two"))[0]
    assert len(kinds(plot, "curve")) == 1
    assert plot["legend"] is False


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
    return Timecourses(batch.ds.assign_coords(arm=("individual", arms)))


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
    plot = plots(plot_nca(tc, nca_single(tc)))[0]
    spans = [e for e in plot["elements"] if str(e.get("label")).startswith("infusion")]
    assert len(spans) == 1
    span = spans[0]
    assert span["label"] == "infusion (2 hr)"
    assert span["kind"] == "area"
    assert list(floats(span["x"])) == [0.0, 2.0]
    # the window spans the panel and sits behind the shaded areas and the data
    assert list(floats(span["from"])) == [plot["y"]["min"]] * 2
    assert list(floats(span["to"])) == [plot["y"]["max"]] * 2
    assert plot["elements"][0] is span


def test_plot_nca_draws_the_legend_once() -> None:
    tc = oral()
    linear, log, table = plots(plot_nca(tc, nca_single(tc)))
    assert linear["legend"] is True
    assert log["legend"] is False and table["legend"] is False


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
    chart = plot_nca_grid(batch, nca(batch, options=NCAOptions()), ncols=2)
    titles = [p["title"] for p in plots(chart)]
    assert titles[0].startswith("dose = 50 mg, individual = s1")
    assert titles[1].startswith("dose = 100 mg, individual = s1")


def test_plot_nca_grid_draws_one_legend_for_the_figure() -> None:
    batch = Timecourses.from_timecourses([oral(k, str(k)) for k in (1.0, 2.0)])
    chart = plot_nca_grid(batch, nca(batch, options=NCAOptions()), ncols=2)
    # one legend above the grid of the entries of every panel, matplotlib's
    # `fig.legend(loc="outside upper center", ncols=6)`
    (legend,) = spec(chart)["legends"]
    assert legend["position"] == "top" and legend["columns"] == 6
    svg = chart.svg()
    assert svg.count('data-figure="legend"') == 1
    entries = svg.count("sbmlsim-legend-entry")
    panel = plots(chart)[0]
    # the regression line of one curve does not name its own lambda_z in a
    # legend of the whole figure: one entry per kind of element
    assert "terminal regression" in labels(panel)
    assert not any(label.startswith("lambda_z") for label in labels(panel))
    union = {label for p in plots(chart) for label in labels(p)}
    assert entries == len(union)


def test_plot_nca_grid_with_a_given_chart_puts_the_legend_in_the_first_panel() -> None:
    batch = Timecourses.from_timecourses([oral(k, str(k)) for k in (1.0, 2.0)])
    chart = new_chart(9, 3.5, columns=2)
    assert plot_nca_grid(batch, nca(batch, options=NCAOptions()), chart=chart) is chart
    first, second = plots(chart)
    assert spec(chart)["legend"]["position"] == "inside"
    assert first["legend"] is True and second["legend"] is False
    # a legend of one panel names the regression of that curve
    assert any(label.startswith("lambda_z = ") for label in labels(first))


def test_plot_nca_grid_clamps_the_columns_to_the_sample_count() -> None:
    batch = Timecourses.from_timecourses([oral(1.0, "a")])
    chart = plot_nca_grid(batch, nca(batch, options=NCAOptions()), ncols=3)
    assert len(plots(chart)) == 1
    assert spec(chart)["grid"]["columns"] == 1


def trough(plot: dict[str, Any], label: str) -> dict[str, Any]:
    (element,) = labelled(plot, label)
    return element


def test_plot_troughs_against_the_time_with_groups() -> None:
    batch = trough_batch()
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    chart = plot_troughs(result, by="arm")
    assert isinstance(chart, Chart)
    plot = plots(chart)[0]
    assert plot["x"]["label"] == "time [h]"  # the long "hour" of a result, shortened
    assert plot["y"]["label"].startswith("trough [")
    assert plot["legend_title"] == "arm"
    assert legend(plot) == ["A, ctrough", "A, cmin", "B, ctrough", "B, cmin"]
    # the troughs are the ends of the intervals, not the dose times
    ctrough = trough(plot, "A, ctrough")
    assert "yerr" in ctrough
    np.testing.assert_allclose(floats(ctrough["x"]), [12.0, 24.0, 36.0])
    # the minimum is a hollow marker on a dashed line
    cmin = trough(plot, "A, cmin")
    assert cmin["style"]["line"]["type"] == "dashed"
    assert cmin["style"]["marker"]["fill"] == "none"


def test_plot_troughs_against_the_interval_number() -> None:
    batch = trough_batch()
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    plot = plots(plot_troughs(result, x="interval", spread="se"))[0]
    assert plot["x"]["label"] == "interval"
    np.testing.assert_allclose(floats(trough(plot, "ctrough")["x"]), [1.0, 2.0, 3.0])
    assert all(float(t).is_integer() for t in plot["x"]["ticks"])


def test_plot_troughs_draws_into_a_given_chart_and_without_a_spread() -> None:
    batch = trough_batch()
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    chart = new_chart(12, 4, columns=2)
    assert plot_troughs(result, spread=None, chart=chart, column=1) is chart
    (plot,) = plots(chart)
    assert plot["column"] == 1
    assert kinds(plot, "curve") and not any("yerr" in e for e in kinds(plot, "curve"))


def test_plot_troughs_names_the_statistic_in_the_title() -> None:
    batch = trough_batch()
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    assert plots(plot_troughs(result))[0]["title"] == "mean ± sd over individual"
    assert (
        plots(plot_troughs(result, spread="se", by="arm"))[0]["title"]
        == "mean ± se over individual"
    )
    assert (
        plots(plot_troughs(result, spread=None))[0]["title"] == "mean over individual"
    )
    # a result of a single curve draws that curve, there is no statistic
    single = nca_single(
        multiple_dose_tc(n_doses=3), options=NCAOptions(auc_method=AUCMethod.LOG)
    )
    assert plots(plot_troughs(single))[0].get("title") is None
    # and `plot_intervals` draws one line per sample, not a reduction
    assert plots(plot_intervals(result, "interval_ctrough"))[0].get("title") is None


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
    plot = plots(
        plot_nca(tc, nca_single(tc, options=NCAOptions(auc_method=AUCMethod.LOG)))
    )[0]
    assert not kinds(plot, "rule")


def test_plot_troughs_places_every_group_on_its_own_interval_times() -> None:
    fast = [multiple_dose_tc(n_doses=3, tau=12.0) for _ in range(2)]
    slow = [multiple_dose_tc(n_doses=3, tau=24.0) for _ in range(2)]
    batch = Timecourses.from_timecourses(
        [*fast, *slow], labels=["f1", "f2", "s1", "s2"]
    )
    batch = Timecourses(
        batch.ds.assign_coords(arm=("individual", ["fast", "fast", "slow", "slow"]))
    )
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LOG))
    plot = plots(plot_troughs(result, by="arm"))[0]
    np.testing.assert_allclose(
        floats(trough(plot, "fast, ctrough")["x"]), [12.0, 24.0, 36.0]
    )
    np.testing.assert_allclose(
        floats(trough(plot, "slow, ctrough")["x"]), [24.0, 48.0, 72.0]
    )


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
    linear, log, table = plots(plot_nca(tc, nca_single(tc)))
    written = texts(linear)
    assert any(t.startswith("Cmax = 2.9") and "tmax = 1.5 hr" in t for t in written)
    assert any(
        t.startswith("lambda_z = ") and "t1/2 = " in t and "n = " in t for t in written
    )
    assert any(t.startswith("clast = ") and "tlast = 24 hr" in t for t in written)
    # the areas carry their values, the tail its share
    assert any(t.startswith("AUC(0-tlast) = 22.7") and "h⋅mg/l" in t for t in written)
    assert any(t.startswith("AUC(tlast-inf) = ") and "%)" in t for t in written)
    # the intervals of the uncertainty analysis are in the annotation
    assert any("Cmax = 2.9 [" in t for t in written)
    # the logarithmic panel names the peak as well, with a leader line
    assert any(t.startswith("Cmax = 2.9") for t in texts(log))
    (leader,) = kinds(log, "arrow")
    assert leader["head"] == "none"
    # the parameter table of the third panel, one monospace text per row
    rows = texts(table)
    assert all(e["family"] == "monospace" for e in kinds(table, "text"))
    joined = "\n".join(rows)
    assert "cmax" in joined and "mg/l" in joined and "[" in joined
    assert "extrapolated" in joined and "thalf" in joined and "cl_f" in joined
    assert table["title"] == "parameters, 95 % interval"
    # nothing on the plot without annotations
    bare = plots(plot_nca(tc, nca_single(tc), annotate=False))[0]
    assert not texts(bare)


def test_plot_nca_draws_the_spread_of_the_data() -> None:
    tc = group_curve()
    result = nca_single(tc)
    plot = plots(plot_nca(tc, result))[0]
    bars = [e for e in kinds(plot, "curve") if "yerr" in e]
    assert len(bars) == 1 and bars[0]["label"] == "data ± sd"
    none = plots(plot_nca(tc, result, spread=None))[0]
    assert not [e for e in kinds(none, "curve") if "yerr" in e]
    # a curve without a spread draws no error bars either
    plain = individual()
    bare = plots(plot_nca(plain, nca_single(plain)))[0]
    assert not [e for e in kinds(bare, "curve") if "yerr" in e]


def band_of(plot: dict[str, Any]) -> dict[str, Any] | None:
    return next((e for e in kinds(plot, "band") if "band" in e.get("label", "")), None)


def test_plot_nca_draws_the_confidence_band_of_the_regression() -> None:
    tc = oral()
    result = nca_single(tc)
    band = band_of(plots(plot_nca(tc, result))[0])
    assert band is not None and band["label"] == "95 % band of the regression"
    # the band widens away from the centre of the regression window
    widths = _band_width(tc, result, 0.95)
    assert widths[-1] > widths[0] > 0
    assert len(band["x"]) > 10
    drawn = np.log(floats(band["high"])) - np.log(floats(band["low"]))
    assert drawn[-1] > drawn.min() > 0
    # a wider level gives a wider band
    assert _band_width(tc, result, 0.99)[-1] > widths[-1]
    # a regression on three points has a band
    two = Timecourse(time=[1, 2, 4, 8], value=[4, 3, 2, 1], time_unit="hr", unit="mg/l")
    from pkpdutils import TerminalMethod, TerminalPhase

    short = nca_single(
        two,
        options=NCAOptions(
            terminal=TerminalPhase(method=TerminalMethod.LAST_N, n_points=3)
        ),
    )
    assert band_of(plots(plot_nca(two, short))[0]) is not None


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


def test_plot_nca_with_a_given_chart_draws_the_table_into_the_third_cell() -> None:
    tc = oral()
    chart = new_chart(14, 4, columns=3)
    plot_nca(tc, nca_single(tc), chart=chart)
    table = plots(chart)[2]
    assert table["column"] == 2
    assert table["title"].startswith("parameters") and texts(table)
    two = new_chart(11, 4, columns=2)
    plot_nca(tc, nca_single(tc), annotate=False, chart=two)
    assert len(plots(two)) == 2
    with pytest.raises(ValueError, match="need a grid"):
        plot_nca(tc, nca_single(tc), chart=new_chart(11, 4, columns=2))


def test_plot_nca_of_its_own_chart_gives_the_table_a_narrower_panel() -> None:
    tc = oral()
    linear, log, table = plots(plot_nca(tc, nca_single(tc)))
    assert linear["column_span"] == log["column_span"] > table["column_span"]


def test_plot_nca_grid_has_the_band_but_no_annotations_by_default() -> None:
    batch = Timecourses.from_timecourses([oral(1.0, "a"), oral(4.0, "b")])
    first = plots(plot_nca_grid(batch, nca(batch)))[0]
    assert band_of(first) is not None
    assert not texts(first)
    annotated = plots(plot_nca_grid(batch, nca(batch), annotate=True))[0]
    assert texts(annotated)


def terminal_options(acceptance: Acceptance | None = None) -> NCAOptions:
    """Options which keep the candidate windows of the terminal regression."""
    return NCAOptions(
        terminal=TerminalPhase(keep_candidates=True),
        acceptance=acceptance or Acceptance(),
    )


def test_plot_terminal_windows_draws_one_marker_per_candidate() -> None:
    tc = oral()
    options = terminal_options()
    result = nca_single(tc, options=options)
    found = plots(plot_terminal_windows(tc, result, options=options))
    assert len(found) == 2
    curve_panel, candidates = found
    n_candidates = int(result.ds.sizes["candidate"])
    assert n_candidates >= 3
    (line,) = labelled(candidates, "candidate window")
    np.testing.assert_allclose(
        floats(line["x"]), result.ds["candidate_t_first"].to_numpy()
    )
    np.testing.assert_allclose(
        floats(line["y"]), result.ds["candidate_r2_adj"].to_numpy()
    )
    # the chosen window is marked once, at the start time of the regression
    (chosen,) = labelled(candidates, "chosen window")
    assert list(floats(chosen["x"])) == [float(result["lambda_z_t_first"].values)]
    # the number of points of every window is written above its marker
    assert len(texts(candidates)) == n_candidates
    assert set(texts(candidates)) == {
        str(int(n)) for n in result.ds["candidate_n_points"].to_numpy()
    }
    # the curve panel is logarithmic and brackets the window with two lines
    assert curve_panel["y"]["scale"] == "log"
    bounds = [e["value"] for e in kinds(curve_panel, "rule")]
    assert bounds == [
        float(result["lambda_z_t_first"].values),
        float(result["lambda_z_t_last"].values),
    ]
    assert [e.get("label") for e in kinds(curve_panel, "rule")] == [
        "chosen window",
        "chosen window",
    ]
    assert "chosen window" in legend(curve_panel)


def test_plot_terminal_windows_draws_the_acceptance_threshold() -> None:
    tc = oral()
    options = terminal_options(acceptance=Acceptance(r2_adj_min=0.9))
    result = nca_single(tc, options=options)
    candidates = plots(plot_terminal_windows(tc, result, options=options))[1]
    rules = [
        e
        for e in kinds(candidates, "rule")
        if e.get("label", "").startswith("acceptance")
    ]
    assert len(rules) == 1 and rules[0]["value"] == 0.9
    # no threshold without the options, and none without an acceptance rule
    for chart in (
        plot_terminal_windows(tc, result),
        plot_terminal_windows(tc, result, options=terminal_options()),
    ):
        assert not any(
            label.startswith("acceptance") for label in labels(plots(chart)[1])
        )


def test_plot_terminal_windows_takes_a_chart_and_the_sample_of_a_batch() -> None:
    batch = Timecourses.from_timecourses([oral(2.0, "a")], labels=["a"])
    options = terminal_options()
    result = nca(batch, options=options)
    chart = new_chart(11, 4, columns=2)
    same = plot_terminal_windows(
        batch.sel(individual="a"), result, chart=chart, individual="a", options=options
    )
    assert same is chart
    assert len(kinds(plots(chart)[1], "curve")) >= 2


def test_plot_terminal_windows_raises_without_the_candidate_windows() -> None:
    tc = oral()
    result = nca_single(tc)
    with pytest.raises(ValueError, match="keep_candidates"):
        plot_terminal_windows(tc, result)


def test_plot_terminal_windows_raises_for_a_batch_of_several_samples() -> None:
    batch = Timecourses.from_timecourses([oral(2.0, "a"), oral(3.0, "b")])
    result = nca(batch, options=terminal_options())
    with pytest.raises(ValueError, match="keep_candidates"):
        plot_terminal_windows(batch.sel(individual="a"), result, individual="a")


def area(plot: dict[str, Any], label: str) -> dict[str, Any]:
    """The filled area of a plot carrying `label`."""
    return next(e for e in kinds(plot, "area") if e.get("label") == label)


def test_plot_nca_shades_a_named_partial_area() -> None:
    tc = oral()
    options = NCAOptions(partial_aucs={"auc_0_12": (0.0, 12.0)})
    result = nca_single(tc, options=options)
    linear, log, _ = plots(plot_nca(tc, result, partial="auc_0_12"))
    for plot in (linear, log):
        shaded = area(plot, "auc_0_12")
        x = floats(shaded["x"])
        assert x.min() == pytest.approx(0.0) and x.max() == pytest.approx(12.0)
        assert shaded["style"]["color"] == DEFAULT_STYLE.partial_color
    value = float(result["auc_0_12"].values)
    assert f"auc_0_12 = {value:.3g} h⋅mg/l" in texts(linear)
    # the area to the last point is still shaded, the partial area sits on it
    elements = linear["elements"]
    assert elements.index(area(linear, "AUC(0-tlast)")) < elements.index(
        area(linear, "auc_0_12")
    )


def test_plot_nca_grid_shades_the_partial_area_of_every_panel() -> None:
    batch = Timecourses.from_timecourses([oral(2.0, "a"), oral(3.0, "b")])
    options = NCAOptions(partial_aucs={"auc_2_8": (2.0, 8.0)})
    result = nca(batch, options=options)
    for plot in plots(plot_nca_grid(batch, result, partial="auc_2_8")):
        x = floats(area(plot, "auc_2_8")["x"])
        assert x.min() == pytest.approx(2.0) and x.max() == pytest.approx(8.0)


def test_plot_nca_raises_for_an_unknown_partial_area() -> None:
    tc = oral()
    result = nca_single(tc, options=NCAOptions(partial_aucs={"auc_0_12": (0.0, 12.0)}))
    with pytest.raises(ValueError, match="auc_0_24"):
        plot_nca(tc, result, partial="auc_0_24")


def test_draw_nca_panel_needs_the_range_of_a_partial_area() -> None:
    tc = oral()
    result = nca_single(tc, options=NCAOptions(partial_aucs={"auc_0_12": (0.0, 12.0)}))
    values = {name: float(result[name].values) for name in result.parameters}
    with pytest.raises(ValueError, match="partial_range"):
        draw_nca_panel(tc, values, [], partial="auc_0_12")


def test_the_partial_area_of_a_multiple_dose_curve_sits_in_its_interval() -> None:
    # the interval is relative to the first dose, the panel to the last one
    tc = multiple_dose_tc(n_doses=3, tau=12.0)
    options = NCAOptions(
        auc_method=AUCMethod.LOG, partial_aucs={"auc_24_36": (24.0, 36.0)}
    )
    result = nca_single(tc, options=options)
    plot = plots(plot_nca(tc, result, partial="auc_24_36"))[0]
    x = floats(area(plot, "auc_24_36")["x"])
    assert x.min() == pytest.approx(0.0) and x.max() == pytest.approx(12.0)


def test_a_partial_area_past_the_last_point_follows_the_terminal_regression() -> None:
    tc = oral()  # the last sample is at 24 h
    options = NCAOptions(partial_aucs={"auc_0_48": (0.0, 48.0)})
    result = nca_single(tc, options=options)
    plot = plots(plot_nca(tc, result, partial="auc_0_48"))[0]
    shaded = area(plot, "auc_0_48")
    x, y = floats(shaded["x"]), floats(shaded["to"])
    assert x.max() == pytest.approx(48.0)
    # the shaded tail is the regression, not the last value carried forward
    lambda_z = float(result["lambda_z"].values)
    clast_pred = float(result["clast_pred"].values)
    tail = (x > 30.0) & (y > 0.0)
    assert tail.any()
    np.testing.assert_allclose(
        y[tail], clast_pred * np.exp(-lambda_z * (x[tail] - 24.0)), rtol=1e-6
    )


def test_the_partial_annotation_stays_inside_the_panel() -> None:
    tc = oral()  # the peak sits at 2 h of a panel which runs to about 40 h
    early = nca_single(tc, options=NCAOptions(partial_aucs={"auc_0_6": (0.0, 6.0)}))
    late = nca_single(tc, options=NCAOptions(partial_aucs={"auc_8_24": (8.0, 24.0)}))
    for result, name, anchor in (
        (early, "auc_0_6", "start"),
        (late, "auc_8_24", "middle"),
    ):
        plot = plots(plot_nca(tc, result, partial=name))[0]
        text = next(e for e in kinds(plot, "text") if e["text"].startswith(name))
        assert text["anchor"] == anchor


@pytest.mark.parametrize("route", [Route.ORAL, Route.IV_BOLUS])
def test_the_area_starts_at_the_dose(route: Route) -> None:
    # the first sample is at 0.25 h: the shaded area starts at the dose with
    # the value the analysis inserts there, 0 or the back-extrapolated C0
    t = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    value = 10.0 * np.exp(-0.3 * t) - (
        8.0 * np.exp(-2.0 * t) if route is Route.ORAL else 0.0
    )
    tc = Timecourse(
        time=t,
        value=value,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=route),
        substance="drug",
    )
    result = nca_single(tc)
    plot = plots(plot_nca(tc, result))[0]
    shaded = area(plot, "AUC(0-tlast)")
    x, top = floats(shaded["x"]), floats(shaded["to"])
    assert x.min() == 0.0
    start = 0.0 if route is Route.ORAL else float(result["c0"])
    assert np.isclose(top[x == 0.0], start).any()
    # the area reaches down to the axis
    assert np.all(floats(shaded["from"]) == 0.0)
