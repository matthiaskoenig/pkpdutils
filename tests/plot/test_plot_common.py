import logging
import math

import numpy as np
import pytest
import sbmlsim_core
import xarray as xr
from sbmlsim_core.chart import Chart, ChartAxis

from pkpdutils import Dose, Dosing, Route
from pkpdutils.plot._common import (
    FONTS,
    INCH,
    Frame,
    annotate_column,
    annotation_room,
    axis_label,
    chart_of,
    data_limits,
    dose_markers,
    estimate_text,
    format_value,
    group_colors,
    group_order,
    has_infusion,
    legend_room,
    log_axis,
    make_room_right,
    marker,
    new_chart,
    plain_log_ticks,
    row_axis,
    row_panel_width,
    sample_colors,
    sample_labels,
    sample_title,
    unit_label,
    value_axis,
)


def elements(chart: Chart, plot: int = 0) -> list[dict]:
    """The elements of a plot of a chart, from its spec."""
    return chart.spec()["plots"][plot]["elements"]


def test_new_chart_has_the_size_in_inches_and_the_fonts_of_the_package() -> None:
    chart = new_chart(6, 4, rows=2, columns=3, title="t", fonts={"legend": 7.0})
    assert (chart.width, chart.height) == (6 * INCH, 4 * INCH)
    spec = chart.spec()
    assert spec["grid"]["rows"] == 2 and spec["grid"]["columns"] == 3
    assert spec["title"] == "t"
    assert spec["fonts"]["label"] == FONTS["label"]
    assert spec["fonts"]["title_weight"] == "normal"
    # a font of the caller replaces the one of the package
    assert spec["fonts"]["legend"] == 7.0


def test_chart_of_creates_a_chart_without_one() -> None:
    chart = chart_of(None, 0, 0, rows=1, columns=2, size=(11.0, 4.0), title="x")
    assert chart.rows == 1 and chart.columns == 2
    assert chart.width == 11.0 * INCH
    assert chart.title == "x"
    # a cell places panels into a given chart only
    with pytest.raises(ValueError, match="given 'chart'"):
        chart_of(None, 0, 1, rows=1, columns=1, size=(6.0, 4.0))


def test_chart_of_returns_a_given_chart_and_checks_the_panels_fit() -> None:
    chart = Chart(rows=2, columns=3)
    assert chart_of(chart, 1, 1, rows=1, columns=2, size=(1.0, 1.0)) is chart
    # the title of a new chart does not touch the chart of the caller
    same = chart_of(chart, 0, 0, rows=2, columns=3, size=(1.0, 1.0), title="t")
    assert same is chart and chart.title is None
    with pytest.raises(ValueError, match="need a grid of at least 2x4"):
        chart_of(chart, 1, 2, rows=1, columns=2, size=(1.0, 1.0))
    with pytest.raises(ValueError, match="need a grid of at least 3x1"):
        chart_of(chart, 1, 0, rows=2, columns=1, size=(1.0, 1.0))
    with pytest.raises(ValueError, match="first cell"):
        chart_of(chart, -1, 0, rows=1, columns=1, size=(1.0, 1.0))


def test_marker_takes_the_names_of_the_chart_and_the_codes_of_matplotlib() -> None:
    assert marker("circle") == "circle"
    assert marker("o") == "circle"
    assert marker("s") == "square"
    assert marker("D") == "diamond"
    assert marker("d") == "thin_diamond"
    assert marker("") == "none"
    with pytest.raises(ValueError, match="unknown marker"):
        marker("blob")


def test_data_limits_add_the_margins_of_matplotlib() -> None:
    assert data_limits([0.0, 10.0]) == pytest.approx((-0.5, 10.5))
    # several arrays and values which are not finite
    assert data_limits([0.0, np.nan], [10.0, np.inf]) == pytest.approx((-0.5, 10.5))
    # a fixed end replaces the widened one, as `set_ylim(bottom=0)`
    assert data_limits([2.0, 10.0], low=0.0) == pytest.approx((0.0, 10.4))
    # a logarithmic axis is widened in its decades and ignores non-positive values
    low, high = data_limits([0.0, 1.0, 100.0], log=True)
    assert low == pytest.approx(10.0**-0.1) and high == pytest.approx(10.0**2.1)
    # without data, a default range
    assert data_limits([np.nan]) == pytest.approx((-0.05, 1.05))
    assert data_limits([], log=True)[0] > 0.0


def test_plain_log_ticks_write_plain_numbers() -> None:
    ticks, labels = plain_log_ticks(0.005, 200.0)
    assert ticks == [0.01, 0.1, 1.0, 10.0, 100.0]
    assert labels == ["0.01", "0.1", "1", "10", "100"]
    # two decades or less label 2, 3, 4 and 6 of every decade as well
    ticks, _ = plain_log_ticks(15.0, 450.0)
    assert ticks == [20.0, 30.0, 40.0, 60.0, 100.0, 200.0, 300.0, 400.0]
    # half a decade or less every integer multiple
    ticks, _ = plain_log_ticks(1.5, 4.5)
    assert ticks == [2.0, 3.0, 4.0]
    # every other decade over more than eight
    ticks, labels = plain_log_ticks(1e-6, 1e6)
    assert ticks == [1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6]
    assert all("^" not in label for label in labels)


def test_value_axis_gives_a_logarithmic_axis_its_plain_ticks() -> None:
    log = value_axis("c [mg/l]", log=True, limits=(0.05, 20.0))
    assert log.scale == "log" and (log.min, log.max) == (0.05, 20.0)
    assert log.ticks == (0.1, 1.0, 10.0)
    assert log.tick_labels == ("0.1", "1", "10")
    linear = value_axis("t", limits=(0.0, 5.0))
    assert linear.scale == "linear" and linear.ticks is None
    assert (linear.min, linear.max) == (0.0, 5.0)
    bare = value_axis("t", log=True)
    assert bare.scale == "log" and bare.min is None and bare.ticks is None


def test_log_axis_falls_back_to_linear_without_positive_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert log_axis(True, "y", [0.0, 1.0])
    assert not log_axis(False, "y", [1.0])
    with caplog.at_level(logging.DEBUG, logger="pkpdutils.plot._common"):
        assert not log_axis(True, "y", [0.0, -1.0], None, [np.nan])
    assert any("no positive value" in message for message in caplog.messages)


def test_frame_converts_points_and_fractions_into_values() -> None:
    chart = Chart(width=400, height=300)
    plot = chart.plot(
        x=ChartAxis(min=0.0, max=10.0),
        y=ChartAxis(scale="log", min=0.01, max=100.0),
    )
    plot.curve([1.0, 9.0], [0.1, 10.0])
    frame = Frame.of(plot)
    assert frame.x == (0.0, 10.0) and frame.y == pytest.approx((0.01, 100.0))
    assert frame.log_y and not frame.log_x
    assert 0.0 < frame.width < 400.0 and 0.0 < frame.height < 300.0
    x, y = frame.offset(5.0, 1.0, frame.width / 10.0, frame.height / 4.0)
    assert x == pytest.approx(6.0)
    assert y == pytest.approx(10.0)  # a quarter of four decades
    x, y = frame.at(0.5, 0.75)
    assert x == pytest.approx(5.0) and y == pytest.approx(10.0)


def test_sample_labels_joins_coordinates_per_dimension() -> None:
    ds = xr.Dataset(
        {"value": (("individual", "occasion"), np.zeros((2, 3)))},
        coords={"individual": ["a", "b"], "occasion": [1, 2, 3]},
    )
    labels = sample_labels(ds, ("individual", "occasion"))
    assert labels == ["a|1", "a|2", "a|3", "b|1", "b|2", "b|3"]


def test_sample_labels_uses_the_flat_index_without_a_coordinate() -> None:
    ds = xr.Dataset({"value": (("individual",), np.zeros(3))})
    labels = sample_labels(ds, ("individual",))
    assert labels == ["0", "1", "2"]


def test_sample_labels_empty_for_no_sample_dimensions() -> None:
    ds = xr.Dataset({"value": (("time",), np.zeros(3))})
    assert sample_labels(ds, ()) == []


def test_sample_labels_by_broadcasts_a_coordinate_to_the_sample_dims() -> None:
    group = np.array([["g00", "g01"], ["g10", "g11"], ["g20", "g21"]])
    ds = xr.Dataset(
        {"value": (("individual", "dose"), np.zeros((3, 2)))},
        coords={
            "individual": [0, 1, 2],
            "dose": ["low", "high"],
            "group": (("individual", "dose"), group),
        },
    )
    labels = sample_labels(ds, ("individual", "dose"), by="group")
    assert labels == ["g00", "g01", "g10", "g11", "g20", "g21"]


def test_sample_colors_uses_the_colormap_start_for_a_single_sample() -> None:
    colors = sample_colors(1, "viridis")
    assert colors == list(sbmlsim_core.color_scale("viridis", 0.0, 1.0, [0.0]))
    assert colors == ["#440154"]
    assert sample_colors(0, "viridis") == []


def test_sample_colors_spreads_evenly_across_the_colormap() -> None:
    colors = sample_colors(3, "viridis")
    assert colors == list(sbmlsim_core.color_scale("viridis", 0, 1, [0.0, 0.5, 1.0]))
    assert all(c.startswith("#") and len(c) == 7 for c in colors)


def test_format_value_writes_a_number_without_trailing_zeros() -> None:
    assert format_value(50.0) == "50"
    assert format_value(np.float64(0.125)) == "0.125"
    assert format_value("s1") == "s1"
    assert format_value(np.True_) == "True"


def test_sample_title_names_the_coordinates_with_their_units() -> None:
    ds = xr.Dataset(
        {"value": (("dose", "individual"), np.zeros((2, 2)))},
        coords={"dose": ("dose", [50.0, 100.0]), "individual": ["s1", "s2"]},
    )
    ds["dose"].attrs["units"] = "mg"
    assert (
        sample_title(ds, (0, 1), ("dose", "individual"))
        == "dose = 50 mg, individual = s2"
    )
    # a unit given by the caller for a coordinate which carries none
    plain = xr.Dataset(
        {"value": (("dose",), np.zeros(2))}, coords={"dose": [50.0, 100.0]}
    )
    assert sample_title(plain, (1,), ("dose",)) == "dose = 100"
    assert sample_title(plain, (1,), ("dose",), units={"dose": "mg"}) == "dose = 100 mg"
    assert sample_title(plain, (), ()) == ""


def test_estimate_text_leaves_out_an_incomplete_interval() -> None:
    assert estimate_text(1.2345, 0.9876, 1.5) == "1.23 [0.988, 1.5]"
    assert estimate_text(1.2345, np.nan, 1.5) == "1.23"


def test_group_colors_stop_before_the_pale_end_of_the_colormap() -> None:
    colors = group_colors(3, "viridis")
    assert len(colors) == 3
    assert colors[0] == sample_colors(1, "viridis")[0]
    # the last group is not the very end of the colormap
    assert colors[-1] != sample_colors(2, "viridis")[-1]
    assert colors[-1] == sbmlsim_core.color_scale("viridis", 0, 1, [0.85])[0]
    assert group_colors(0, "viridis") == []


def test_dose_markers_draws_lines_for_doses_and_a_window_for_an_infusion() -> None:
    chart = Chart()
    plot = chart.plot()
    dose_markers(plot, None, y_limits=(0.0, 1.0))
    assert elements(chart) == []
    dosing = Dosing.regimen(
        Dose(amount=100, unit="mg", route=Route.IV_BOLUS), interval=12.0, n_doses=3
    )
    dose_markers(plot, dosing, y_limits=(0.0, 1.0))
    rules = [e for e in elements(chart) if e["kind"] == "rule"]
    assert [r["value"] for r in rules] == [0.0, 12.0, 24.0]
    assert all(r["orientation"] == "vertical" for r in rules)
    assert all(r["style"]["line"]["type"] == "dotted" for r in rules)
    assert all(r["style"]["color"] == "#808080" for r in rules)  # gray
    single_chart = Chart()
    single = single_chart.plot()
    one = Dosing.single(Dose(amount=100, unit="mg", route=Route.IV_BOLUS))
    dose_markers(single, one, y_limits=(0.0, 1.0))
    assert elements(single_chart) == []  # a single dose gets no line
    infusion = Dosing.single(
        Dose(amount=100, unit="mg", route=Route.IV_INFUSION, duration=0.5, time=1.0)
    )
    assert has_infusion(infusion)
    assert not has_infusion(one) and not has_infusion(None)
    dose_markers(single, infusion, y_limits=(0.1, 5.0))
    (area,) = elements(single_chart)
    assert area["kind"] == "area" and area["label"] == "infusion"
    # the window spans the panel, the ends of its y axis
    assert area["x"] == [1.0, 1.5]
    assert area["from"] == [0.1, 0.1] and area["to"] == [5.0, 5.0]


def test_dose_markers_labels_the_window_once_and_not_a_taken_label() -> None:
    chart = Chart()
    plot = chart.plot()
    infusion = Dosing.single(
        Dose(amount=100, unit="mg", route=Route.IV_INFUSION, duration=0.5)
    )
    taken: set[str] = set()
    dose_markers(plot, infusion, y_limits=(0, 1), time_unit="hr", taken=taken)
    dose_markers(plot, infusion, y_limits=(0, 1), time_unit="hr", taken=taken)
    labels = [e.get("label") for e in elements(chart)]
    assert labels == ["infusion (0.5 hr)", None]
    assert taken == {"infusion (0.5 hr)"}
    # a label a curve of the plot already carries is left out
    other = Chart()
    dose_markers(other.plot(), infusion, y_limits=(0, 1), taken={"infusion"})
    assert "label" not in elements(other)[0]


def test_make_room_right_widens_the_axis_by_a_share() -> None:
    assert make_room_right(0.0, 10.0, 1.0) == pytest.approx(20.0)
    # a logarithmic axis by the same share of its decades
    assert make_room_right(1.0, 100.0, 0.5, log=True) == pytest.approx(1000.0)


def test_annotation_room_grows_with_the_length_of_the_texts() -> None:
    assert annotation_room([], 300.0) == 0.0
    short = annotation_room(["1.0"], 300.0)
    long = annotation_room(["1.00 [0.90, 1.10], 12.3 %"], 300.0)
    assert 0.0 < short < long <= 1.5
    # a panel too narrow for the column takes the most room
    assert annotation_room(["1.00 [0.90, 1.10], 12.3 %"], 20.0) == 1.5


def test_row_axis_puts_the_first_row_at_the_top() -> None:
    axis = row_axis(["a", "b", "c"])
    assert axis.reverse
    assert (axis.min, axis.max) == pytest.approx((-0.6, 2.6))
    assert axis.ticks == (0.0, 1.0, 2.0)
    assert axis.tick_labels == ("a", "b", "c")


def test_row_panel_width_leaves_out_the_widest_label() -> None:
    chart = new_chart(6, 4)
    short = row_panel_width(chart, ["a"])
    long = row_panel_width(chart, ["random effects (tau2 = 0.01)"])
    assert long < short < chart.width


def test_annotate_column_writes_one_text_per_row_right_of_the_data() -> None:
    chart = Chart()
    plot = chart.plot(x=ChartAxis(min=0.0, max=10.0), y=row_axis(["a", "b"]))
    plot.curve([1.0, 2.0], [0.0, 1.0])
    annotate_column(plot, ["first", "second"], 5.0)
    texts = [e for e in elements(chart) if e["kind"] == "text"]
    assert [t["text"] for t in texts] == ["first", "second"]
    assert [t["y"] for t in texts] == [0.0, 1.0]
    assert all(5.0 < t["x"] < 6.0 for t in texts)
    annotate_column(plot, [], 5.0)  # nothing to write
    assert len([e for e in elements(chart) if e["kind"] == "text"]) == 2


@pytest.mark.parametrize("log", [False, True])
def test_legend_room_raises_the_top_above_the_data(log: bool) -> None:
    limits = (1.0, 100.0)
    labels = ["individuals", "mean [95 % CI]"]
    top, columns = legend_room(labels, limits, (400.0, 250.0), log=log)
    assert top > limits[1]
    # a wide panel takes the legend in one row
    assert columns == 2
    if log:
        assert math.log10(top) > 2.0
    assert legend_room([], limits, (400.0, 250.0)) == (100.0, 1)


def test_legend_room_is_one_column_in_a_narrow_panel() -> None:
    labels = ["individuals", "geometric mean [95 % CI]"]
    _, columns = legend_room(labels, (0.0, 1.0), (120.0, 250.0))
    assert columns == 1
    wide, _ = legend_room(labels, (0.0, 1.0), (500.0, 250.0))
    narrow, _ = legend_room(labels, (0.0, 1.0), (120.0, 250.0))
    # two rows need more room than one
    assert narrow > wide
    # never more than half of the panel
    top, _ = legend_room(labels * 20, (0.0, 1.0), (120.0, 50.0))
    assert top == pytest.approx(2.0)


def test_sample_labels_write_a_number_without_trailing_zeros() -> None:
    ds = xr.Dataset({"value": (("dose",), np.zeros(2))}, coords={"dose": [50.0, 100.0]})
    assert sample_labels(ds, ("dose",)) == ["50", "100"]


def test_unit_label_shortens_a_canonical_unit_and_keeps_a_spelled_one() -> None:
    assert unit_label("milligram / liter") == "mg/l"
    assert unit_label("hour") == "h"
    assert unit_label("hr") == "hr"  # the user's own spelling of the data
    assert unit_label("ng/ml") == "ng/ml"
    assert unit_label("dimensionless") == ""
    assert unit_label("") == ""
    assert unit_label("not a unit") == "not a unit"


def test_axis_label_leaves_out_an_empty_unit() -> None:
    assert axis_label("trough", "mg/l") == "trough [mg/l]"
    assert axis_label("ratio", "") == "ratio"


def test_group_order_keeps_the_first_appearance_of_every_group() -> None:
    ds = xr.Dataset(
        {"value": (("individual",), np.zeros(4))},
        coords={
            "individual": ["s1", "s2", "s3", "s4"],
            "arm": ("individual", ["B", "A", "B", "A"]),
        },
    )
    assert group_order(ds, ("individual",), "arm") == ["B", "A"]
