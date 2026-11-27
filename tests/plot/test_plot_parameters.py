import math
from typing import Any

import numpy as np
import pytest
import sbmlsim_core
import xarray as xr
from sbmlsim_core.chart import Chart

from pkpdutils import Route, Timecourses, nca
from pkpdutils.plot import new_chart, plot_parameters
from pkpdutils.plot.parameters import box_stats
from pkpdutils.result import ParameterResult
from pkpdutils.stats import Scale


def nca_result():
    time = np.array([0.5, 1, 2, 4, 6, 8, 12, 24])
    rng = np.random.default_rng(2)
    n = 8
    curves = np.stack(
        [rng.lognormal(np.log(10), 0.2) * np.exp(-0.2 * time) for _ in range(n)]
    )
    batch = Timecourses.from_arrays(
        time,
        curves,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={
            "individual": [f"s{i}" for i in range(n)],
            "sex": ("individual", ["F", "M"] * 4),
        },
        dose={"amount": np.full(n, 100.0), "unit": "mg"},
        route=Route.IV_BOLUS,
    )
    return nca(batch)


def plot_of(chart: Chart, k: int = 0) -> dict[str, Any]:
    return chart.spec()["plots"][k]


def legend_labels(plot: dict[str, Any]) -> list[str]:
    """The entries of the legend of a plot, its labelled elements in order."""
    if not plot["legend"]:
        return []
    found: list[str] = []
    for e in plot["elements"]:
        if "label" in e and e["label"] not in found:
            found.append(e["label"])
    return found


def box_values(plot: dict[str, Any], position: int) -> list[float]:
    """The distinct values of the box, whiskers and median at a position."""
    values: set[float] = set()
    for e in plot["elements"]:
        if e["kind"] != "box":
            continue
        for x, stats in zip(e["positions"], e["stats"], strict=True):
            if x == position:
                values.update(
                    stats[key]
                    for key in ("low_whisker", "q1", "median", "q3", "high_whisker")
                )
    return sorted(values)


def _is_strip(e: dict[str, Any]) -> bool:
    return e["kind"] == "curve" and "label" not in e


def strip_values(plot: dict[str, Any], position: int) -> list[float]:
    """The values of the jittered points of the individuals at a position."""
    values: list[float] = []
    for e in plot["elements"]:
        if not _is_strip(e):
            continue
        x = np.asarray(e["x"], dtype=float)
        at = np.abs(x - position) < 0.5
        values.extend(np.asarray(e["y"], dtype=float)[at].tolist())
    return sorted(values)


def markers(plot: dict[str, Any]) -> list[dict[str, Any]]:
    """The statistic of every group, in the order of the groups."""
    return [e for e in plot["elements"] if e["kind"] == "curve" and "label" in e]


def test_plot_parameters_groups() -> None:
    result = nca_result()
    chart = plot_parameters(result, "auc_inf_obs", "individual", by="sex", log_y=True)
    assert isinstance(chart, Chart)
    plot = plot_of(chart)
    assert plot["y"]["scale"] == "log"
    assert plot["x"]["categories"] == ["F", "M"]
    assert plot["x"]["label"] == "sex"
    assert plot["y"]["label"].startswith("auc_inf_obs [")
    assert not any("^" in label for label in plot["y"]["tick_labels"])


def test_plot_parameters_names_the_statistic_in_the_legend_once() -> None:
    plot = plot_of(plot_parameters(nca_result(), "auc_inf_obs", "individual", by="sex"))
    # the groups are the ticks of the x axis, only the marker is named
    assert legend_labels(plot) == ["geometric mean [95 % CI]"]
    assert plot["legend_loc"] == "upper_right"


def test_plot_parameters_legend_follows_the_scale_and_the_level() -> None:
    plot = plot_of(
        plot_parameters(
            nca_result(), "cmax", "individual", scale=Scale.LINEAR, ci_level=0.90
        )
    )
    assert legend_labels(plot) == ["mean [90 % CI]"]


def test_plot_parameters_takes_the_scale_as_a_string() -> None:
    # `scale="log"` is the analysis of `scale=Scale.LOG`, as in `pkpdutils.stats`
    result = nca_result()
    log = plot_of(plot_parameters(result, "cmax", "individual", scale="log"))
    assert legend_labels(log) == ["geometric mean [95 % CI]"]
    linear = plot_of(plot_parameters(result, "cmax", "individual", scale="linear"))
    assert legend_labels(linear) == ["mean [95 % CI]"]
    with pytest.raises(ValueError, match="not a valid Scale"):
        plot_parameters(result, "cmax", "individual", scale="geometric")


def test_plot_parameters_single_group_linear_into_a_given_chart() -> None:
    chart = new_chart(12, 4, columns=2)
    out = plot_parameters(
        nca_result(), "cmax", "individual", scale=Scale.LINEAR, chart=chart, column=1
    )
    assert out is chart
    plot = plot_of(chart)
    assert plot["column"] == 1
    assert plot["x"]["categories"] == ["cmax"]
    # the error bar of the mean
    assert "yerr" in markers(plot)[0]


def _result_with_values(values: np.ndarray) -> ParameterResult:
    n = values.size
    ds = xr.Dataset(
        {
            "value": (("individual",), values, {"units": "mg/l"}),
            "flags": (
                ("individual",),
                np.zeros(n, dtype=np.int64),
                {"units": "dimensionless"},
            ),
        },
        coords={
            "individual": [f"s{i}" for i in range(n)],
            "sex": ("individual", ["F", "M"] * (n // 2)),
        },
    )
    return ParameterResult(ds)


def test_plot_parameters_log_ticks_over_several_decades() -> None:
    values = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0])
    plot = plot_of(
        plot_parameters(_result_with_values(values), "value", "individual", log_y=True)
    )
    labels = plot["y"]["tick_labels"]
    assert len(labels) <= 12, labels
    assert not any("^" in label for label in labels)
    assert "1000" in labels


def _grouped_result(groups: dict[str, list[float]]) -> ParameterResult:
    """A result of one parameter with the individuals of every group in order."""
    values = np.array([v for group in groups.values() for v in group])
    n = values.size
    ds = xr.Dataset(
        {
            "value": (("individual",), values, {"units": "mg/l"}),
            "flags": (
                ("individual",),
                np.zeros(n, dtype=np.int64),
                {"units": "dimensionless"},
            ),
        },
        coords={
            "individual": [f"s{i}" for i in range(n)],
            "group": (
                "individual",
                [key for key, group in groups.items() for _ in group],
            ),
        },
    )
    return ParameterResult(ds)


def test_box_stats_are_the_ones_of_matplotlib() -> None:
    # matplotlib's `cbook.boxplot_stats` of these values, whis=1.5
    values = np.array([1.0, 2.0, 4.0, 8.0, 40.0])
    assert box_stats(values) == {
        "low_whisker": 1.0,
        "q1": 2.0,
        "median": 4.0,
        "q3": 8.0,
        "high_whisker": 8.0,
    }
    assert box_stats(np.array([5.0]))["high_whisker"] == 5.0


def test_plot_parameters_log_box_uses_the_values_of_the_strip() -> None:
    # issue #71: under `log_y=True` the strip dropped the non-positive values
    # but the box was still computed from every finite value, so the box of
    # the mixed group reached below the axis and did not describe its points
    positive = [1.0, 2.0, 4.0, 8.0]
    mixed = [2.0, 4.0, 8.0, -1.0]
    result = _grouped_result({"A": positive, "B": mixed})
    plot = plot_of(
        plot_parameters(result, "value", "individual", by="group", log_y=True)
    )
    assert plot["y"]["scale"] == "log"
    shown = [2.0, 4.0, 8.0]
    assert strip_values(plot, 0) == positive
    assert strip_values(plot, 1) == shown
    for position, values in ((0, positive), (1, shown)):
        expected = set(box_stats(np.array(values)).values())
        assert box_values(plot, position) == pytest.approx(sorted(expected))
    # the statistic is computed from every finite value: the all-positive group
    # keeps its geometric mean, the mixed group falls back to the arithmetic
    # mean with a marker of its own, and the legend names both
    geo, mean = markers(plot)
    assert geo["y"][0] == pytest.approx(np.exp(np.mean(np.log(positive))))
    assert mean["y"][0] == pytest.approx(np.mean(mixed))
    assert geo["style"]["marker"]["type"] != mean["style"]["marker"]["type"]
    assert legend_labels(plot) == ["geometric mean [95 % CI]", "mean [95 % CI]"]


def test_plot_parameters_linear_box_keeps_the_nonpositive_values() -> None:
    mixed = [2.0, 4.0, 8.0, -1.0]
    result = _grouped_result({"A": [1.0, 2.0, 4.0, 8.0], "B": mixed})
    plot = plot_of(plot_parameters(result, "value", "individual", by="group"))
    assert strip_values(plot, 1) == sorted(mixed)
    assert min(box_values(plot, 1)) == -1.0


def test_plot_parameters_log_group_without_a_positive_value() -> None:
    # a group without a positive value has nothing a logarithmic axis can
    # show: no points, no box and no marker, only its tick; a group of a
    # single value is a box of one line and a marker without an interval
    result = _grouped_result(
        {"A": [1.0, 2.0, 4.0, 8.0], "B": [0.0, -1.0, -2.0, 0.0], "C": [5.0]}
    )
    chart = plot_parameters(result, "value", "individual", by="group", log_y=True)
    chart.svg()
    plot = plot_of(chart)
    assert plot["y"]["scale"] == "log"
    assert plot["x"]["categories"] == ["A", "B", "C"]
    assert strip_values(plot, 1) == []
    assert box_values(plot, 1) == []
    assert strip_values(plot, 2) == [5.0]
    assert box_values(plot, 2) == [5.0]
    # the arithmetic mean of B is negative: A and C are the only markers
    found = markers(plot)
    assert len(found) == 2
    assert found[1]["style"]["marker"]["type"] == "diamond"
    assert found[1]["y"][0] == pytest.approx(5.0)
    assert "yerr" not in found[1]
    assert legend_labels(plot) == ["geometric mean [95 % CI]"]


def test_plot_parameters_log_without_positive_values_stays_linear() -> None:
    # B28: an all-non-positive parameter (e.g. an all-zero `cmax` batch) cannot
    # be drawn on a logarithmic axis, which stays linear
    chart = plot_parameters(
        _result_with_values(np.zeros(4)), "value", "individual", log_y=True
    )
    chart.svg()
    plot = plot_of(chart)
    assert plot["y"]["scale"] == "linear"
    # the axis stays linear, so it shows the values a logarithmic one could not
    assert strip_values(plot, 0) == [0.0, 0.0, 0.0, 0.0]
    assert box_values(plot, 0) == [0.0]
    assert legend_labels(plot) == ["mean [95 % CI]"]


def _data_tops_below_the_legend(chart: Chart) -> None:
    """Every point, box and interval of every plot is drawn below its legend."""
    spec = chart.spec()
    geometry = sbmlsim_core.figure_geometry(chart.to_json())
    for plot, frame in zip(spec["plots"], geometry["plots"], strict=True):
        entries = [e for e in geometry["legend_entries"] if e["owner"] == plot["id"]]
        assert entries, "the plot has a legend"
        legend_bottom = max(e["rect"]["y"] + e["rect"]["height"] for e in entries)
        y = frame["y"]
        rect = frame["rect"]

        def pixel(value: float, y=y, rect=rect) -> float:
            low, high = y["low"], y["high"]
            if y["kind"] == "log":
                share = math.log(value / low) / math.log(high / low)
            else:
                share = (value - low) / (high - low)
            return rect["y"] + rect["height"] * (1.0 - share)

        tops: list[float] = []
        for e in plot["elements"]:
            if e["kind"] == "box":
                tops.extend(pixel(s["high_whisker"]) for s in e["stats"])
            elif e["kind"] == "curve":
                values = np.asarray(e["y"], dtype=float)
                if "yerr" in e:
                    values = values + np.asarray(e["yerr"]["high"], dtype=float)
                radius = e["style"]["marker"]["size"] / 2.0
                tops.extend(pixel(float(v)) - radius for v in values if v > 0)
        assert min(tops) > legend_bottom


_LEGEND_GROUPS = {
    "A": [1.0, 2.0, 4.0, 8.0],
    "B": [0.0, -1.0, -2.0, 0.0],
    "C": [5.0],
    "D": [2.0, 4.0, 8.0, -1.0],
}


@pytest.mark.parametrize("shape", [None, (1, 2), (2, 1)], ids=["own", "1x2", "2x1"])
def test_plot_parameters_legend_covers_no_data(shape: tuple[int, int] | None) -> None:
    # the legend used to go where `best` put it, which ignores the error bars:
    # with these groups it sat on the interval of D
    result = _grouped_result(_LEGEND_GROUPS)
    if shape is None:
        chart = plot_parameters(result, "value", "individual", by="group", log_y=True)
    else:
        chart = new_chart(
            6.0 * shape[1], 4.0 * shape[0], rows=shape[0], columns=shape[1]
        )
        for k in range(shape[0] * shape[1]):
            plot_parameters(
                result,
                "value",
                "individual",
                by="group",
                log_y=True,
                chart=chart,
                row=k // shape[1],
                column=k % shape[1],
            )
    _data_tops_below_the_legend(chart)
    # the entries are the markers of the statistics with their error bars
    plot = plot_of(chart)
    assert legend_labels(plot) == ["geometric mean [95 % CI]", "mean [95 % CI]"]
    assert {e["style"]["marker"]["type"] for e in markers(plot)} == {
        "diamond",
        "square",
    }
    assert any("yerr" in e for e in markers(plot))
