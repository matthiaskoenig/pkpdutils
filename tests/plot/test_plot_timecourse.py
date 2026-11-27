import numpy as np
import pytest
from sbmlsim_core.chart import Chart

from pkpdutils import Dose, Dosing, Route, Timecourse, Timecourses
from pkpdutils.plot import (
    PlotStyle,
    plot_mean_timecourse,
    plot_study_curves,
    plot_timecourse,
)


def plots(chart: Chart) -> list[dict]:
    """The plots of a chart, from its spec."""
    return chart.spec()["plots"]


def kinds(plot: dict, kind: str) -> list[dict]:
    """The elements of one kind of a plot."""
    return [e for e in plot["elements"] if e["kind"] == kind]


def legend_labels(plot: dict) -> list[str] | None:
    """The entries of the legend of a plot, `None` for a plot without one."""
    if not plot["legend"]:
        return None
    labels: list[str] = []
    for element in plot["elements"]:
        label = element.get("label")
        if label is not None and label not in labels:
            labels.append(label)
    return labels


def dose_rules(plot: dict) -> list[dict]:
    """The dotted vertical lines of the doses of a plot."""
    return [
        e
        for e in kinds(plot, "rule")
        if e["orientation"] == "vertical" and e["style"]["line"]["type"] == "dotted"
    ]


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
    chart = plot_timecourse(multiple_dose_tc(n_doses=4))
    rules = dose_rules(plots(chart)[0])
    assert len(rules) == 4
    assert all(rule["style"]["color"] == "#808080" for rule in rules)  # gray


def test_plot_timecourse_single_dose_draws_no_dose_line() -> None:
    tc = Timecourse(
        time=np.array([0.5, 1, 2, 4, 8]),
        value=10.0 * np.exp(-0.15 * np.array([0.5, 1, 2, 4, 8])),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
    )
    assert dose_rules(plots(plot_timecourse(tc))[0]) == []


def test_plot_timecourse_log_without_positive_values_stays_linear() -> None:
    # a curve without a positive value cannot be drawn on a logarithmic axis
    tc = Timecourse(
        time=[0, 1, 2, 4],
        value=[0.0, 0.0, 0.0, 0.0],
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg"),
    )
    chart = plot_timecourse(tc, log_y=True)
    assert plots(chart)[0]["y"]["scale"] == "linear"
    assert chart.svg().startswith("<svg")


def test_plot_timecourse_log_axis_has_plain_ticks_over_its_data() -> None:
    chart = plot_timecourse(multiple_dose_tc(n_doses=2), log_y=True)
    y = plots(chart)[0]["y"]
    assert y["scale"] == "log"
    assert y["min"] > 0.0 and y["max"] > 10.0
    assert "10" in y["tick_labels"] and all("^" not in t for t in y["tick_labels"])


def test_plot_timecourse_batch_draws_no_dose_lines() -> None:
    a = multiple_dose_tc(n_doses=3)
    b = multiple_dose_tc(n_doses=3)
    batch = Timecourses.from_timecourses([a, b], labels=["one", "two"])
    assert dose_rules(plots(plot_timecourse(batch))[0]) == []


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
    time = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0, 12.0])
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


def test_plot_timecourse_single_and_batch() -> None:
    chart = plot_timecourse(escalation().isel(dose=0, individual=0))
    assert isinstance(chart, Chart)
    plot = plots(chart)[0]
    assert plot["x"]["label"] == "time [hr]"
    assert plot["y"]["label"] == "caffeine [mg/l]"
    assert plot["y"]["scale"] == "linear"


def test_plot_timecourse_by_gives_one_legend_entry_and_one_color_per_group() -> None:
    chart = plot_timecourse(escalation(), by="dose")
    plot = plots(chart)[0]
    assert legend_labels(plot) == ["50", "100"]
    # the unit of the grouping coordinate is written once, in the title
    assert plot["legend_title"] == "dose [mg]"
    colors = {curve["style"]["color"] for curve in kinds(plot, "curve")}
    assert len(colors) == 2
    # every curve of a group is in the legend group of its label
    assert {curve["legend"] for curve in kinds(plot, "curve")} == {"50", "100"}


def test_plot_timecourse_facet_draws_one_panel_per_value() -> None:
    chart = plot_timecourse(escalation(), facet="dose", by="individual")
    first, second = plots(chart)
    assert (first["column"], second["column"]) == (0, 1)
    # the panel names the value of the coordinate with its unit
    assert first["title"] == "dose = 50 mg"
    assert second["title"] == "dose = 100 mg"
    # the legend is drawn once, on the first panel
    assert first["legend"] and not second["legend"]


def test_plot_timecourse_facet_draws_into_the_given_chart() -> None:
    chart = Chart(rows=2, columns=3)
    assert plot_timecourse(
        escalation(), facet="dose", chart=chart, row=1, column=1
    ) is (chart)
    spec = plots(chart)
    assert [(p["row"], p["column"]) for p in spec] == [(1, 1), (1, 2)]
    assert all(kinds(p, "curve") for p in spec)
    # the chart of the caller keeps its size and has no title of the figure
    assert chart.title is None
    with pytest.raises(ValueError, match="grid"):
        plot_timecourse(escalation(), facet="dose", chart=chart, row=0, column=2)


def test_plot_timecourse_suppresses_a_legend_above_max_legend() -> None:
    batch = escalation(n_individual=5)
    assert plots(plot_timecourse(batch, max_legend=12))[0]["legend"]
    assert not plots(plot_timecourse(batch, max_legend=3))[0]["legend"]


def test_plot_timecourse_draws_an_infusion_as_a_window() -> None:
    plot = plots(plot_timecourse(infusion_tc()))[0]
    spans = [
        e for e in kinds(plot, "area") if str(e.get("label", "")).startswith("infusion")
    ]
    assert len(spans) == 1
    span = spans[0]
    assert span["label"] == "infusion (1 hr)"
    assert span["x"] == [0.0, 1.0]
    # the window spans the y axis, whose ends are fixed
    assert span["from"] == [plot["y"]["min"]] * 2
    assert span["to"] == [plot["y"]["max"]] * 2
    # behind the curve
    assert plot["elements"][0]["id"] == span["id"]


def test_plot_timecourse_rejects_by_and_facet_of_a_single_curve() -> None:
    with pytest.raises(ValueError):
        plot_timecourse(infusion_tc(), by="dose")
    with pytest.raises(ValueError):
        plot_timecourse(infusion_tc(), facet="dose")
    with pytest.raises(ValueError, match="given 'chart'"):
        plot_timecourse(infusion_tc(), row=1)


def test_plot_mean_timecourse_panels_bands_and_legend() -> None:
    chart = plot_mean_timecourse(escalation(), by="dose")
    assert isinstance(chart, Chart)
    linear, log = plots(chart)
    assert linear["y"]["scale"] == "linear" and log["y"]["scale"] == "log"
    assert linear["x"]["label"] == "time [hr]"
    assert linear["y"]["label"] == "caffeine [mg/l]"
    assert "sd" in linear["title"]
    # the dose with its unit and the number of curves of the group
    assert legend_labels(linear) == ["50 mg (n = 3)", "100 mg (n = 3)"]
    assert linear["legend_title"] == "dose"
    assert not log["legend"]
    # one spread band per group and panel
    assert len(kinds(linear, "band")) == 2 and len(kinds(log, "band")) == 2


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
    chart = plot_mean_timecourse(batch, by="dose")
    assert legend_labels(plots(chart)[0]) == ["low (n = 2)", "high (n = 2)"]
    # and the same for the panel titles and the legend title of `plot_timecourse`
    faceted = plot_timecourse(batch, facet="dose", by="individual")
    assert [p["title"] for p in plots(faceted)] == ["dose = low", "dose = high"]
    grouped = plots(plot_timecourse(batch, by="dose"))[0]
    assert grouped["legend_title"] == "dose"
    assert legend_labels(grouped) == ["low", "high"]


def test_plot_mean_timecourse_without_individuals_and_without_a_band() -> None:
    chart = plot_mean_timecourse(
        escalation(), by="dose", spread=None, individuals=False, panels=("linear",)
    )
    (plot,) = plots(chart)
    assert not kinds(plot, "band")
    assert len(kinds(plot, "curve")) == 2  # one mean curve per group, no individuals


def test_plot_mean_timecourse_draws_the_individuals_faint_behind_the_mean() -> None:
    chart = plot_mean_timecourse(
        escalation(), by="dose", panels=("linear",), individual_alpha=0.3
    )
    curves = kinds(plots(chart)[0], "curve")
    individuals = [c for c in curves if "label" not in c]
    means = [c for c in curves if "label" in c]
    assert len(individuals) == 6 and len(means) == 2
    assert all(c["style"]["opacity"] == pytest.approx(0.3) for c in individuals)
    assert all(
        c["style"].get("marker", {}).get("type", "none") == "none" for c in individuals
    )
    # the means are drawn last, on top
    assert curves[-2:] == means


def test_plot_mean_timecourse_draws_into_the_given_chart() -> None:
    chart = Chart(columns=3)
    assert plot_mean_timecourse(escalation(), by="dose", chart=chart, column=1) is chart
    spec = plots(chart)
    assert [p["column"] for p in spec] == [1, 2]
    assert all(kinds(p, "curve") for p in spec)


def test_plot_mean_timecourse_without_by_is_one_mean_curve() -> None:
    batch = Timecourses.from_timecourses(
        [multiple_dose_tc(n_doses=3), multiple_dose_tc(n_doses=3)],
        labels=["one", "two"],
    )
    (plot,) = plots(plot_mean_timecourse(batch, panels=("linear",)))
    assert legend_labels(plot) == ["mean (n = 2)"]
    # the dose lines of the shared protocol
    assert len(dose_rules(plot)) == 3


def test_plot_mean_timecourse_rejects_unknown_panels_and_spread() -> None:
    batch = escalation()
    with pytest.raises(ValueError):
        plot_mean_timecourse(batch, panels=("linear", "loglog"))
    with pytest.raises(ValueError):
        plot_mean_timecourse(batch, panels=())
    with pytest.raises(ValueError):
        plot_mean_timecourse(batch, spread="ci")  # ty: ignore[invalid-argument-type]


def test_plot_mean_timecourse_renders() -> None:
    chart = plot_mean_timecourse(escalation(), by="dose")
    svg = chart.svg()
    assert "50 mg (n = 3)" in svg and "semi-logarithmic" in svg


def test_plot_timecourse_facet_legend_names_every_group() -> None:
    # a group which no sample of the first panel carries still gets its
    # legend entry: the legend is built from the groups of the whole batch
    batch = escalation(n_individual=3)
    sex = np.array([["m", "m", "m"], ["m", "f", "m"]])  # "f" only at dose 100
    batch = Timecourses(batch.ds.assign_coords(sex=(("dose", "individual"), sex)))
    first, second = plots(plot_timecourse(batch, facet="dose", by="sex"))
    labels = legend_labels(first)
    assert labels is not None and sorted(labels) == ["f", "m"]
    # the entry of the group of the second panel alone has the color of its curves
    proxy = next(e for e in kinds(first, "curve") if e.get("label") == "f")
    drawn = next(e for e in kinds(second, "curve") if e.get("label") == "f")
    assert proxy["style"]["color"] == drawn["style"]["color"]


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
    (plot,) = plots(plot_mean_timecourse(batch, by="sex", panels=("linear",)))
    # a coordinate without a unit is named by its bare value
    labels = legend_labels(plot)
    assert labels is not None and sorted(labels) == ["f (n = 2)", "m (n = 2)"]


def marker_of(chart: Chart) -> str:
    """The marker of the first curve of a chart."""
    style = kinds(plots(chart)[0], "curve")[0]["style"]
    return style.get("marker", {}).get("type", "none")


def test_plot_timecourse_drops_the_markers_of_a_densely_sampled_curve() -> None:
    time = np.linspace(0, 24, 241)
    dense = Timecourse(
        time=time,
        value=10.0 * np.exp(-0.2 * time),
        time_unit="hr",
        unit="mg/l",
    )
    # 241 points: the markers would merge into a band and hide the curve
    assert marker_of(plot_timecourse(dense)) == "none"
    sparse = Timecourse(
        time=time[::30],
        value=10.0 * np.exp(-0.2 * time[::30]),
        time_unit="hr",
        unit="mg/l",
    )
    assert marker_of(plot_timecourse(sparse)) == "circle"
    # the threshold is a style, a caller who wants every marker raises it
    style = PlotStyle(marker_max_points=500)
    assert marker_of(plot_timecourse(dense, style=style)) == "circle"
    # a marker of matplotlib's code is the marker of the chart
    assert marker_of(plot_timecourse(sparse, style=PlotStyle(data_marker="s"))) == (
        "square"
    )


def test_plot_mean_timecourse_clips_the_band_at_the_axis_bottom() -> None:
    (plot,) = plots(plot_mean_timecourse(escalation(), by="dose", panels=("log",)))
    bottom = plot["y"]["min"]
    assert bottom > 0.0
    for band in kinds(plot, "band"):
        low = np.asarray(band["low"], dtype=float)
        assert float(low.min()) >= bottom


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


def x_data(curve: dict) -> np.ndarray:
    """The x values of a curve as a float array."""
    return np.asarray(curve["x"], dtype=float)


def test_plot_study_curves_has_four_panels_on_actual_and_nominal_times() -> None:
    batch = study_batch()
    spec = plots(plot_study_curves(batch))
    assert len(spec) == 4
    assert [(p["row"], p["column"]) for p in spec] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    titles = [p["title"] for p in spec]
    assert titles[0] == "individuals, linear"
    assert titles[1] == "individuals, semi-logarithmic"
    assert all(title.startswith("mean on the nominal times") for title in titles[2:])
    assert spec[1]["y"]["scale"] == "log" and spec[3]["y"]["scale"] == "log"
    assert spec[0]["y"]["scale"] == "linear"
    times = batch.nominal_times
    assert times is not None
    nominal = np.unique(times)
    # the individual panels draw the times the samples were taken at, which no
    # two subjects share, the mean panels the eight nominal times
    assert not np.allclose(x_data(kinds(spec[0], "curve")[0]), nominal)
    np.testing.assert_allclose(x_data(kinds(spec[2], "curve")[-1]), nominal)
    assert spec[0]["x"]["label"] == "time [hr]"


def test_plot_study_curves_colors_the_groups_the_same_in_every_panel() -> None:
    spec = plots(plot_study_curves(study_batch(), by="arm"))
    colors = []
    for plot in (spec[0], spec[2]):
        by_label: dict[str, str] = {}
        for curve in kinds(plot, "curve"):
            if "label" in curve:
                by_label.setdefault(curve["label"][0], curve["style"]["color"])
        colors.append(by_label)
    assert sorted(colors[0]) == ["A", "B"]
    assert colors[0] == colors[1]
    # both rows name the groups the same way, the mean row with the number of
    # subjects behind the mean
    for plot, entries in (
        (spec[0], ["A", "B"]),
        (spec[2], ["A (n = 4)", "B (n = 4)"]),
    ):
        assert legend_labels(plot) == entries
        assert plot["legend_title"] == "arm"


def test_plot_study_curves_names_a_numeric_group_with_its_unit() -> None:
    batch = study_batch()
    dose = np.where(np.arange(8) < 4, 50.0, 100.0)
    ds = batch.ds.assign_coords(dose=("individual", dose))
    spec = plots(plot_study_curves(Timecourses(ds), by="dose"))
    for plot in (spec[0], spec[2]):
        assert plot["legend_title"] == "dose"
        labels = legend_labels(plot)
        assert labels is not None
        assert labels[0].startswith("50 mg") and labels[1].startswith("100 mg")
    # no legend at all below the limit
    bare = plots(plot_study_curves(Timecourses(ds), by="dose", max_legend=1))
    assert not bare[0]["legend"]


def test_plot_study_curves_without_the_logarithmic_panels() -> None:
    spec = plots(plot_study_curves(study_batch(), log_y_panels=False))
    assert len(spec) == 2
    assert [p["y"]["scale"] for p in spec] == ["linear", "linear"]


def test_plot_study_curves_maps_the_times_to_the_nearest_nominal_time() -> None:
    batch = study_batch()
    without = Timecourses(batch.ds.drop_vars("nominal_time"))
    assert without.nominal_times is None
    schedule = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
    mapped = kinds(
        plots(plot_study_curves(without, nominal_times=schedule))[2], "curve"
    )
    np.testing.assert_allclose(x_data(mapped[-1]), schedule)
    # without a schedule the actual times are the grid, so the mean curve of a
    # batch whose subjects were sampled at different times is ragged
    bare = kinds(plots(plot_study_curves(without))[2], "curve")
    assert x_data(mapped[-1]).size < x_data(bare[-1]).size


def test_plot_study_curves_draws_into_the_given_chart() -> None:
    chart = Chart(rows=3, columns=2)
    assert plot_study_curves(study_batch(), by="arm", chart=chart, row=1) is chart
    spec = plots(chart)
    assert [(p["row"], p["column"]) for p in spec] == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert all(kinds(p, "curve") for p in spec)
    with pytest.raises(ValueError, match="grid"):
        plot_study_curves(study_batch(), chart=chart, row=2)
