import logging
import warnings

import matplotlib
import matplotlib.pyplot
import numpy as np
import pytest
import xarray as xr
from matplotlib.patches import Rectangle

from pkpdutils import Dose, Dosing, Route
from pkpdutils.plot._common import (
    annotate_column,
    annotation_room,
    axes_of,
    dose_markers,
    estimate_text,
    figure_of,
    format_value,
    group_colors,
    log_scale,
    make_room_right,
    plain_log_ticks,
    sample_colors,
    sample_labels,
    sample_title,
)

matplotlib.use("Agg")


def test_figure_of_creates_a_figure_with_constrained_layout() -> None:
    fig, ax = figure_of(None)
    assert fig.get_layout_engine() is not None
    assert ax.get_figure() is fig
    matplotlib.pyplot.close(fig)


def test_figure_of_never_touches_the_layout_engine_of_a_passed_ax() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    assert fig.get_layout_engine() is None
    fig2, ax2 = figure_of(ax)
    assert fig2 is fig and ax2 is ax
    assert fig.get_layout_engine() is None
    matplotlib.pyplot.close(fig)


def test_axes_of_creates_a_grid_with_constrained_layout() -> None:
    fig, axes = axes_of(None, nrows=2, ncols=3, figsize=(6, 4))
    assert fig.get_layout_engine() is not None
    assert axes.shape == (2, 3)
    matplotlib.pyplot.close(fig)


def test_axes_of_never_touches_the_layout_engine_of_passed_axes() -> None:
    fig, axes = matplotlib.pyplot.subplots(nrows=1, ncols=2)
    assert fig.get_layout_engine() is None
    fig2, grid = axes_of(axes, nrows=1, ncols=2, figsize=(6, 4))
    assert fig2 is fig
    assert grid.shape == (1, 2)
    assert fig.get_layout_engine() is None
    matplotlib.pyplot.close(fig)


def test_axes_of_raises_for_a_mismatched_number_of_axes() -> None:
    _fig, axes = matplotlib.pyplot.subplots(nrows=1, ncols=2)
    with pytest.raises(ValueError):
        axes_of(axes, nrows=2, ncols=2, figsize=(6, 4))
    matplotlib.pyplot.close("all")


def test_plain_log_ticks_formats_as_plain_numbers() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot([1, 10, 100], [1, 2, 3])
    ax.set_xscale("log")
    plain_log_ticks(ax.xaxis)
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert labels
    assert all("^" not in label for label in labels)
    matplotlib.pyplot.close(fig)


def test_log_scale_sets_log_and_plain_ticks_when_data_is_positive() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot([0.0, 1.0, 2.0], [1.0, 2.0, 4.0])
    log_scale(ax, "y")
    assert ax.get_yscale() == "log"
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels(which="both") if t.get_text()]
    assert labels
    assert all("^" not in label for label in labels)
    matplotlib.pyplot.close(fig)


def test_log_scale_falls_back_to_linear_without_positive_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot([0.0, 1.0, 2.0, 4.0], [0.0, 0.0, 0.0, 0.0])
    with (
        caplog.at_level(logging.DEBUG, logger="pkpdutils.plot._common"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")
        log_scale(ax, "y")
        fig.canvas.draw()
    assert ax.get_yscale() == "linear"
    assert any("no positive value" in message for message in caplog.messages)
    matplotlib.pyplot.close(fig)


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
    assert len(colors) == 1
    assert colors[0] == matplotlib.pyplot.get_cmap("viridis")(0.0)


def test_sample_colors_spreads_evenly_across_the_colormap() -> None:
    colors = sample_colors(3, "viridis")
    cmap = matplotlib.pyplot.get_cmap("viridis")
    assert colors == [cmap(0.0), cmap(0.5), cmap(1.0)]


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


def test_dose_markers_draws_lines_for_doses_and_a_window_for_an_infusion() -> None:
    _fig, ax = matplotlib.pyplot.subplots()
    dose_markers(ax, None)
    assert not ax.get_lines() and not ax.patches
    dosing = Dosing.regimen(
        Dose(amount=100, unit="mg", route=Route.IV_BOLUS), interval=12.0, n_doses=3
    )
    dose_markers(ax, dosing)
    assert len([line for line in ax.get_lines() if line.get_linestyle() == ":"]) == 3
    _fig2, ax2 = matplotlib.pyplot.subplots()
    single = Dosing.single(Dose(amount=100, unit="mg", route=Route.IV_BOLUS))
    dose_markers(ax2, single)
    assert not ax2.get_lines()  # a single dose gets no line
    infusion = Dosing.single(
        Dose(amount=100, unit="mg", route=Route.IV_INFUSION, duration=0.5, time=1.0)
    )
    dose_markers(ax2, infusion)
    spans = [patch for patch in ax2.patches if patch.get_label() == "infusion"]
    assert len(spans) == 1
    span = spans[0]
    assert isinstance(span, Rectangle)
    assert (span.get_x(), span.get_width()) == (1.0, 0.5)
    matplotlib.pyplot.close("all")


def test_make_room_right_returns_the_fraction_the_data_ends_at() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot([1.0, 2.0], [1.0, 2.0])
    ax.set_xlim(0.0, 10.0)
    fraction = make_room_right(ax, 1.0)
    assert ax.get_xlim() == (0.0, 20.0)
    assert fraction == pytest.approx(0.5)
    ax.set_xscale("log")
    ax.set_xlim(1.0, 100.0)
    make_room_right(ax, 0.5)
    assert ax.get_xlim()[1] == pytest.approx(1000.0)
    matplotlib.pyplot.close(fig)


def test_annotation_room_grows_with_the_length_of_the_texts() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot([1.0, 2.0], [1.0, 2.0])
    assert annotation_room(ax, []) == 0.0
    short = annotation_room(ax, ["1.0"])
    long = annotation_room(ax, ["1.00 [0.90, 1.10], 12.3 %"])
    assert 0.0 < short < long <= 1.5
    matplotlib.pyplot.close(fig)


def test_annotate_column_writes_one_text_per_row() -> None:
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot([1.0, 2.0], [0.0, 1.0])
    annotate_column(ax, [(0.0, "a"), (1.0, "b")], 0.7)
    assert [text.get_text() for text in ax.texts] == ["a", "b"]
    matplotlib.pyplot.close(fig)


def test_sample_labels_write_a_number_without_trailing_zeros() -> None:
    ds = xr.Dataset({"value": (("dose",), np.zeros(2))}, coords={"dose": [50.0, 100.0]})
    assert sample_labels(ds, ("dose",)) == ["50", "100"]
