"""Helpers shared by the figures: charts and their cells, axes, ticks, labels and colors."""

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sbmlsim_core
import xarray as xr
from sbmlsim_core.chart import MARKERS, Chart, ChartAxis, ChartPlot

from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.timecourse import Dosing
from pkpdutils.units import short_unit

logger = logging.getLogger(__name__)

#: pixels per inch of a chart: a chart of `w * INCH` by `h * INCH` pixels has
#: the proportions and the font sizes of a matplotlib figure of `w` by `h`
#: inches, a pixel of the chart being a point
INCH: float = 72.0

#: the font sizes of matplotlib's names, relative to its default of 10 points
FONT_SIZES: dict[str, float] = {
    "xx-small": 5.79,
    "x-small": 6.94,
    "small": 8.33,
    "medium": 10.0,
    "large": 12.0,
    "x-large": 14.4,
}

#: the fonts of a chart of the package: the sizes and weights of matplotlib's
#: defaults, titles of 12 points, labels and ticks of 10 and legends in its
#: `small` (the size the figures of the package give their legends), no bold
FONTS: dict[str, float | str] = {
    "title": FONT_SIZES["large"],
    "title_weight": "normal",
    "plot_title": FONT_SIZES["large"],
    "plot_title_weight": "normal",
    "label": FONT_SIZES["medium"],
    "tick": FONT_SIZES["medium"],
    "legend": FONT_SIZES["small"],
}

#: the markers of the chart by their codes of matplotlib
_MARKER_CODES: dict[str, str] = {
    "o": "circle",
    "s": "square",
    "^": "triangle",
    "D": "diamond",
    "+": "plus",
    "x": "x",
    "v": "triangle_down",
    "<": "triangle_left",
    ">": "triangle_right",
    "*": "star",
    "X": "x_filled",
    "P": "plus_filled",
    "p": "pentagon",
    "h": "hexagon",
    "d": "thin_diamond",
    "8": "octagon",
    "1": "tri_down",
    "2": "tri_up",
    "3": "tri_left",
    "4": "tri_right",
    "": "none",
}


def marker(name: str) -> str:
    """The marker of the chart for a marker of a style.

    Args:
        name: a marker of `sbmlsim_core.chart.MARKERS` (`circle`) or its code of
            matplotlib (`o`).

    Returns:
        The name of the marker in the chart.

    Raises:
        ValueError: for a marker which is neither.
    """
    if name in MARKERS:
        return name
    if name in _MARKER_CODES:
        return _MARKER_CODES[name]
    raise ValueError(
        f"unknown marker {name!r}, a marker is one of {', '.join(MARKERS)} "
        "or its code of matplotlib"
    )


def new_chart(
    width: float,
    height: float,
    *,
    rows: int = 1,
    columns: int = 1,
    title: str | None = None,
    fonts: Mapping[str, float | str] | None = None,
    **kwargs: Any,
) -> Chart:
    """A chart in the size of a matplotlib figure, with the fonts of the package.

    Args:
        width: the width in inches, `INCH` pixels each.
        height: the height in inches.

    Keyword Args:
        rows: the rows of the grid.
        columns: the columns of the grid.
        title: the title of the chart.
        fonts: fonts which replace those of `FONTS`.
        **kwargs: passed on to `Chart` (`wspace`, `legends`, ...).

    Returns:
        The chart.
    """
    return Chart(
        width * INCH,
        height * INCH,
        rows=rows,
        columns=columns,
        title=title,
        fonts={**FONTS, **(fonts or {})},
        **kwargs,
    )


def chart_of(
    chart: Chart | None,
    row: int,
    column: int,
    *,
    rows: int,
    columns: int,
    size: tuple[float, float],
    title: str | None = None,
    fonts: Mapping[str, float | str] | None = None,
    **kwargs: Any,
) -> Chart:
    """The chart a figure draws its panels into, a new one without `chart`.

    A figure of `rows` by `columns` panels is drawn into the cells of the grid
    of `chart` from `(row, column)` on, so that a caller lays out several
    figures in one chart (`Chart(rows=..., columns=...)`); without a chart the
    figure gets a chart of its own, of `size` and with `title`, and its panels
    start at the first cell.

    Args:
        chart: the chart to draw into, `None` for a new one.
        row: the row of the first panel in `chart`.
        column: the column of the first panel in `chart`.

    Keyword Args:
        rows: the rows of panels the figure draws.
        columns: the columns of panels the figure draws.
        size: the width and the height of a new chart in inches.
        title: the title of a new chart; a chart of the caller keeps its own.
        fonts: fonts of a new chart which replace those of `FONTS`.
        **kwargs: passed on to `Chart` for a new chart.

    Returns:
        The chart.

    Raises:
        ValueError: for a cell without `chart`, a cell before the first one,
            or panels which do not fit into the grid of `chart`.
    """
    if row < 0 or column < 0:
        raise ValueError(f"the first cell is (0, 0), got ({row}, {column})")
    if chart is None:
        if row or column:
            raise ValueError(
                "'row' and 'column' place the panels into the grid of a given "
                "'chart'; a new chart starts at its first cell"
            )
        return new_chart(
            *size, rows=rows, columns=columns, title=title, fonts=fonts, **kwargs
        )
    if row + rows > chart.rows or column + columns > chart.columns:
        raise ValueError(
            f"{rows}x{columns} panels from the cell ({row}, {column}) need a grid "
            f"of at least {row + rows}x{column + columns}, the chart has "
            f"{chart.rows}x{chart.columns}"
        )
    return chart


def finite(*arrays: Any) -> np.ndarray:
    """The finite values of arrays, flattened into one.

    Args:
        *arrays: arrays, sequences or numbers; `None` is skipped.

    Returns:
        The finite values.
    """
    parts = [
        np.asarray(a, dtype=float).ravel()
        for a in arrays
        if a is not None and np.size(a)
    ]
    if not parts:
        return np.empty(0)
    values = np.concatenate(parts)
    return values[np.isfinite(values)]


def log_axis(log: bool, which: str, *arrays: Any) -> bool:
    """Whether an axis is logarithmic: asked for, with a positive value to show.

    Without a positive value a logarithmic axis cannot be drawn: it stays
    linear and the fallback is logged at debug level.

    Args:
        log: whether the axis is asked to be logarithmic.
        which: the name of the axis, for the message.
        *arrays: the values drawn on the axis.

    Returns:
        Whether the axis is logarithmic.
    """
    if not log:
        return False
    if bool(np.any(finite(*arrays) > 0.0)):
        return True
    logger.debug("the %s axis has no positive value, staying linear", which)
    return False


#: the share of the data range matplotlib adds at either end of an axis
MARGIN: float = 0.05


def data_limits(
    *arrays: Any,
    log: bool = False,
    low: float | None = None,
    high: float | None = None,
    margin: float = MARGIN,
) -> tuple[float, float]:
    """The limits of an axis over its values, with matplotlib's margins.

    The range of the finite values (the positive ones of a logarithmic axis)
    widened by `margin` of it at either end, in the decades of a logarithmic
    axis, as matplotlib autoscales an axis; a fixed end (`low`, `high`) takes
    the place of the widened one, as `set_ylim(bottom=0)` after autoscaling.

    Args:
        *arrays: the values drawn on the axis.

    Keyword Args:
        log: whether the axis is logarithmic.
        low: a fixed lower end, `None` for the widened one.
        high: a fixed upper end, `None` for the widened one.
        margin: the share added at either end.

    Returns:
        The lower and the upper end.
    """
    data = finite(*arrays)
    if log:
        data = data[data > 0.0]
    if not data.size:
        lo, hi = (1.0, 10.0) if log else (0.0, 1.0)
    else:
        lo, hi = float(data.min()), float(data.max())
    if log:
        a, b = math.log10(lo), math.log10(hi)
        span = b - a or 1.0
        lo, hi = 10.0 ** (a - margin * span), 10.0 ** (b + margin * span)
    else:
        span = hi - lo or (abs(hi) or 1.0)
        lo, hi = lo - margin * span, hi + margin * span
    return (lo if low is None else low), (hi if high is None else high)


def plain_log_ticks(low: float, high: float) -> tuple[list[float], list[str]]:
    """The ticks of a logarithmic axis with plain labels, `0.01` and `100` instead of powers of ten.

    The decades of the range are labelled, every other one over more than
    eight decades; a range of two decades or less labels 2, 3, 4 and 6 of
    every decade as well and a range of half a decade or less every integer
    multiple, as matplotlib's `LogFormatter` labels its minor ticks.

    Args:
        low: the lower end of the axis, positive.
        high: the upper end of the axis.

    Returns:
        The values of the ticks and their labels.
    """
    lo, hi = math.log10(low), math.log10(high)
    span = hi - lo
    first, last = math.floor(lo), math.ceil(hi)
    multiples: tuple[int, ...] = (1,)
    if span <= 0.5:
        multiples = tuple(range(1, 10))
    elif span <= 2.0:
        multiples = (1, 2, 3, 4, 6)
    step = 2 if multiples == (1,) and last - first > 8 else 1
    ticks: list[float] = []
    for decade in range(first, last + 1, step):
        for m in multiples:
            value = float(f"{m * 10.0**decade:.12g}")
            if low * (1 - 1e-9) <= value <= high * (1 + 1e-9):
                ticks.append(value)
    return ticks, [f"{v:g}" for v in ticks]


def value_axis(
    label: str,
    *,
    log: bool = False,
    limits: tuple[float, float] | None = None,
    **kwargs: Any,
) -> ChartAxis:
    """A numeric axis, a logarithmic one with the plain ticks of `plain_log_ticks`.

    Args:
        label: the label of the axis with its unit, `name [unit]`.

    Keyword Args:
        log: whether the axis is logarithmic.
        limits: the ends of the axis, `None` for the limits of the renderer;
            a logarithmic axis needs them for its plain ticks.
        **kwargs: passed on to `ChartAxis`.

    Returns:
        The axis.
    """
    low, high = limits if limits is not None else (None, None)
    if log and limits is not None and "ticks" not in kwargs:
        ticks, labels = plain_log_ticks(*limits)
        if ticks:
            kwargs.update(ticks=ticks, tick_labels=labels)
    return ChartAxis(
        label=label,
        scale="log" if log else "linear",
        min=low,
        max=high,
        **kwargs,
    )


@dataclass(frozen=True)
class Frame:
    """The panel of a plot as the renderer lays it out: its pixels and the ends of its axes.

    It converts an offset in points from a point of the data (matplotlib's
    `textcoords="offset points"`) and a position in fractions of the panel
    into values of the axes, so that a text is placed a fixed distance from
    the point it explains whatever the range of the data.

    Attributes:
        width: the width of the panel in pixels, a point each.
        height: the height of the panel in pixels.
        x: the ends of the x axis.
        y: the ends of the y axis.
        log_x: whether the x axis is logarithmic.
        log_y: whether the y axis is logarithmic.
    """

    width: float
    height: float
    x: tuple[float, float]
    y: tuple[float, float]
    log_x: bool
    log_y: bool

    @classmethod
    def of(cls, plot: ChartPlot) -> "Frame":
        """The frame of a plot with the elements it has, from the geometry of the renderer.

        Args:
            plot: the plot.

        Returns:
            The frame.
        """
        geometry = sbmlsim_core.figure_geometry(plot.chart.to_json())
        entry = next(p for p in geometry["plots"] if p["id"] == plot.id)
        rect, x, y = entry["rect"], entry["x"], entry["y"]
        return cls(
            width=float(rect["width"]),
            height=float(rect["height"]),
            x=(float(x["low"]), float(x["high"])),
            y=(float(y["low"]), float(y["high"])),
            log_x=x["kind"] == "log",
            log_y=y["kind"] == "log",
        )

    @staticmethod
    def _move(
        value: float, ends: tuple[float, float], share: float, log: bool
    ) -> float:
        """A value moved along an axis by a share of the axis."""
        low, high = ends
        if log:
            return value * (high / low) ** share
        return value + share * (high - low)

    def offset(self, x: float, y: float, dx: float, dy: float) -> tuple[float, float]:
        """The point `dx` points right of and `dy` points above a point of the data.

        Args:
            x: the x value of the point.
            y: the y value of the point.
            dx: the offset to the right in points.
            dy: the offset upwards in points.

        Returns:
            The x and the y value of the moved point.
        """
        return (
            self._move(x, self.x, dx / max(self.width, 1.0), self.log_x),
            self._move(y, self.y, dy / max(self.height, 1.0), self.log_y),
        )

    def at(self, fx: float, fy: float) -> tuple[float, float]:
        """The values of the axes at fractions of the panel from its lower left corner.

        Args:
            fx: the share of the width.
            fy: the share of the height.

        Returns:
            The x and the y value.
        """
        return (
            self._move(self.x[0], self.x, fx, self.log_x),
            self._move(self.y[0], self.y, fy, self.log_y),
        )


def sample_labels(
    ds: xr.Dataset, sample_dims: tuple[str, ...] | list[str], *, by: str | None = None
) -> list[str]:
    """One label per sample of the sample dimensions, in C order.

    Without `by`, the label of a sample is the coordinate value of every
    sample dimension joined by `"|"` (the flat index when a dimension
    carries no coordinate), a number written without its trailing zeros
    (`format_value`, `50` rather than `50.0`). With `by`, the label is the value of that
    coordinate broadcast to every sample dimension, used when the samples
    are labelled by one grouping coordinate instead of by their own index
    (`plot_timecourse(by=...)`).

    Args:
        ds: the dataset the coordinates are read from.
        sample_dims: the sample dimensions, in the order the samples are
            enumerated.
        by: name of a coordinate broadcast to the sample dimensions instead
            of the default per-dimension coordinates.

    Returns:
        One label per sample, in C order of `sample_dims`; empty when
        `sample_dims` is empty.
    """
    if not sample_dims:
        return []
    sizes = [ds.sizes[d] for d in sample_dims]
    if by is not None:
        template = xr.DataArray(np.zeros(sizes), dims=list(sample_dims))
        coord, _ = xr.broadcast(ds[by], template)
        values = coord.transpose(*sample_dims).to_numpy().reshape(-1)
        return [format_value(v) for v in values]
    labels = []
    for index in np.ndindex(*sizes):
        parts = [
            format_value(ds.coords[d].to_numpy()[i]) if d in ds.coords else str(i)
            for d, i in zip(sample_dims, index, strict=True)
        ]
        labels.append("|".join(parts))
    return labels


def format_value(value: Any) -> str:
    """One coordinate value as a short string, a number without trailing zeros.

    Args:
        value: the value, a number or anything else.

    Returns:
        `f"{value:g}"` for a number, `str(value)` otherwise.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | float):
        return f"{value:g}"
    return str(value)


def is_number(value: Any) -> bool:
    """Whether a coordinate value is a number, so that a unit belongs to it.

    Args:
        value: the value, a number, a numpy scalar or anything else.

    Returns:
        `True` for an integer or a floating point number, a numpy scalar of
        one included; `False` for a boolean, a string and everything else.
    """
    if isinstance(value, np.generic):
        value = value.item()
    return isinstance(value, int | float) and not isinstance(value, bool)


def value_with_unit(
    ds: xr.Dataset,
    name: str,
    value: Any,
    *,
    units: Mapping[str, str | None] | None = None,
) -> str:
    """One coordinate value with the unit of its coordinate: `50 mg`.

    The unit is `attrs["units"]` of the coordinate `name`, or the entry of
    `units`, which a caller uses for a coordinate whose unit lives elsewhere
    (the `dose` of a batch, whose unit is the dose unit of the batch); a
    dimensionless coordinate and one without a unit give the value alone.

    Only a number carries a unit: a coordinate whose values are the names of
    the groups (`dose` as `"low"` and `"high"`, an arm, a treatment) is a
    label, so it is written as it is instead of as `low mg`.

    Args:
        ds: the dataset the coordinate is read from.
        name: name of the coordinate.
        value: the value to format.

    Keyword Args:
        units: unit per coordinate name, used when the coordinate itself
            carries none.

    Returns:
        The value, followed by its unit when there is one.
    """
    text = format_value(value)
    if name not in ds.coords or not is_number(value):
        return text
    unit = str(ds.coords[name].attrs.get("units", "")) or (
        (units or {}).get(name) or ""
    )
    return f"{text} {unit}" if unit and unit != "dimensionless" else text


def sample_title(
    ds: xr.Dataset,
    index: Sequence[int],
    sample_dims: Sequence[str],
    *,
    units: Mapping[str, str | None] | None = None,
) -> str:
    """The title of one sample: `dose = 50 mg, individual = s1`.

    Every sample dimension contributes `name = value`, the value taken from
    its coordinate (the position when the dimension carries none) and
    followed by the unit of the coordinate when there is one. The unit is
    `attrs["units"]` of the coordinate, or the entry of `units`, which a
    caller uses for a coordinate whose unit lives elsewhere (the `dose` of a
    batch, whose unit is the dose unit of the batch).

    Args:
        ds: the dataset the coordinates are read from.
        index: position of the sample along every sample dimension.
        sample_dims: the sample dimensions, in the order of `index`.

    Keyword Args:
        units: unit per coordinate name, used when the coordinate itself
            carries none.

    Returns:
        The title, empty without sample dimensions.
    """
    parts = []
    for dim, position in zip(sample_dims, index, strict=True):
        if dim in ds.coords:
            text = value_with_unit(
                ds, dim, ds.coords[dim].to_numpy()[position], units=units
            )
        else:
            text = str(position)
        parts.append(f"{dim} = {text}")
    return ", ".join(parts)


def estimate_text(
    estimate: float, ci_low: float, ci_high: float, digits: int = 3
) -> str:
    """`estimate [low, high]` of a point estimate with its interval.

    Args:
        estimate: the point estimate.
        ci_low: lower bound of the interval.
        ci_high: upper bound of the interval.
        digits: significant digits of every number.

    Returns:
        The text; the interval is left out when a bound is not finite.
    """
    text = f"{estimate:.{digits}g}"
    if np.isfinite(ci_low) and np.isfinite(ci_high):
        text = f"{text} [{ci_low:.{digits}g}, {ci_high:.{digits}g}]"
    return text


#: font size of the annotations of a row figure (`plot_ratio`, `plot_forest`)
ANNOTATION_FONTSIZE: float = FONT_SIZES["x-small"]

#: the gap between the data and the column of annotations, in points
ANNOTATION_GAP: float = 4.0


def annotation_room(texts: Sequence[str], panel_width: float) -> float:
    """The share of the x range a column of annotations needs.

    The width of the column is the width of the longest text at
    `ANNOTATION_FONTSIZE` as the renderer measures it
    (`sbmlsim_core.text_width`) with a gap on either side, and the share is
    that width over the width the data keeps in the panel.

    Args:
        texts: the annotations.
        panel_width: the width of the panel in pixels.

    Returns:
        The share of the current x range to add to the right, `0` without
        texts, at most `1.5`.
    """
    if not texts:
        return 0.0
    needed = (
        max(sbmlsim_core.text_width(text, ANNOTATION_FONTSIZE) for text in texts)
        + 2.0 * ANNOTATION_GAP
    )
    if panel_width <= needed:
        return 1.5
    return min(max(needed / (panel_width - needed), 0.15), 1.5)


def make_room_right(
    low: float, high: float, share: float, *, log: bool = False
) -> float:
    """The upper end of an axis widened to the right, for annotations beside the data.

    The annotations of a row figure (`estimate [low, high]`) are written in
    one column to the right of the data, where they neither cover the
    intervals nor run out of the panel. The room is a share of the range the
    data spans: a logarithmic axis is widened by the same share of its
    decades.

    Args:
        low: the lower end of the axis.
        high: the upper end of the axis, where the column starts.
        share: the share of the range to add.

    Keyword Args:
        log: whether the axis is logarithmic.

    Returns:
        The widened upper end.
    """
    if log and low > 0.0:
        return high * (high / low) ** share
    return high + share * (high - low)


def row_axis(labels: Sequence[str]) -> ChartAxis:
    """The y axis of a row figure, one row per label from the top down.

    The rows are at the positions 0, 1, ... with their labels as ticks, the
    first one at the top, and the axis reaches 0.6 beyond the first and the
    last row, as matplotlib's `set_yticks` with `invert_yaxis` of
    `plot_ratio` and `plot_forest`.

    Args:
        labels: the labels of the rows.

    Returns:
        The axis.
    """
    n = len(labels)
    return ChartAxis(
        min=-0.6,
        max=n - 0.4,
        reverse=True,
        ticks=[float(k) for k in range(n)],
        tick_labels=list(labels),
        tick_marks_visible=True,
    )


#: the width of the tick marks, the padding of the tick labels and the margin
#: right of a panel in pixels, which `row_panel_width` leaves out
_ROW_PANEL_PADDING = 36.0


def row_panel_width(chart: Chart, labels: Sequence[str]) -> float:
    """The width of the panel of a row figure in a chart of one column.

    The chart less the widest tick label of the rows, which the renderer
    measures with `sbmlsim_core.text_width`, and the padding around it.

    Args:
        chart: the chart.
        labels: the tick labels of the rows.

    Returns:
        The width in pixels.
    """
    size = float(
        chart.fonts.get("tick", FONTS["tick"]) if chart.fonts else FONTS["tick"]
    )
    widest = max(
        (sbmlsim_core.text_width(label, size) for label in labels), default=0.0
    )
    return chart.width / chart.columns - widest - _ROW_PANEL_PADDING


def annotate_column(plot: ChartPlot, texts: Sequence[str], start: float) -> None:
    """Write one annotation per row in a column right of the data.

    The texts are in the order of the rows of `row_axis`, left aligned
    `ANNOTATION_GAP` points right of `start`, where the data end and the room
    of `make_room_right` begins.

    Args:
        plot: the plot, its x axis widened by `make_room_right`.
        texts: the text of every row, from the top down.
        start: the x value where the data end.
    """
    if not texts:
        return
    frame = Frame.of(plot)
    for y, text in enumerate(texts):
        x, _ = frame.offset(start, float(y), ANNOTATION_GAP, 0.0)
        plot.text(
            x,
            float(y),
            text,
            size=ANNOTATION_FONTSIZE,
            baseline="center",
            color="black",
        )


#: the gap between the data and a legend drawn above it (`legend_room`), as a
#: share of the font size of the legend
LEGEND_GAP: float = 0.8

#: the largest share of the height of a panel `legend_room` gives to the
#: legend, so that the data keeps at least the rest of it
LEGEND_MAX_SHARE: float = 0.5


def legend_room(
    labels: Sequence[str],
    limits: tuple[float, float],
    panel: tuple[float, float],
    *,
    log: bool = False,
    fontsize: float = FONT_SIZES["small"],
) -> tuple[float, int]:
    """Raise the top of a y axis so that a legend in the upper right corner sits above the data.

    The automatic placement of a legend (`best`) avoids points and lines but
    not error bars, so a legend inside the panel can cover what it explains.
    The legend of `labels` is one row when that fits the width of the panel
    and one column otherwise, and the top of the axis is raised until the
    data ends a gap (`LEGEND_GAP`) below it, on a logarithmic axis by the
    same share of its decades. The legend is measured as the renderer lays
    it out: a swatch of two font sizes, the labels in
    `sbmlsim_core.text_width`, entries 1.33 font sizes high in a frame.

    Args:
        labels: the entries of the legend.
        limits: the ends of the y axis over the data.
        panel: the width and the height of the panel in pixels.

    Keyword Args:
        log: whether the y axis is logarithmic.
        fontsize: the font size of the legend.

    Returns:
        The raised upper end of the axis and the columns of the legend, one
        per entry in one row or 1.
    """
    if not labels:
        return limits[1], 1
    width, height = panel
    entries = [
        2.8 * fontsize + sbmlsim_core.text_width(label, fontsize) for label in labels
    ]
    columns = len(labels)
    if sum(entries) + 2.0 * fontsize * len(labels) > width:
        columns = 1
    rows = math.ceil(len(labels) / columns)
    needed = rows * 1.33 * fontsize + 1.6 * fontsize + 6.0 + LEGEND_GAP * fontsize
    share = min(needed / max(height, 1.0), LEGEND_MAX_SHARE)
    low, high = limits
    stretch = 1.0 / (1.0 - share)
    if log and low > 0.0:
        return low * (high / low) ** stretch, columns
    return low + (high - low) * stretch, columns


def dose_markers(
    plot: ChartPlot,
    dosing: Dosing | None,
    *,
    y_limits: tuple[float, float],
    style: PlotStyle = DEFAULT_STYLE,
    min_doses: int = 2,
    time_unit: str | None = None,
    taken: set[str] | None = None,
) -> None:
    """Mark the doses of a protocol on the time axis of a plot.

    An infusion is a window: the span from the dose time to the dose time
    plus the duration of the dose is shaded over the height of the panel
    (`y_limits`, the ends of its y axis), so that the figure shows how long
    the drug went in. Every other route is an instant and gets a thin dotted
    line, drawn only for a protocol of at least `min_doses` doses, since the
    single line at the dose time of a single dose curve carries no
    information. The windows go behind the data: call this before the other
    elements of the plot are added.

    The first window is labelled `infusion (1 hr)` with its duration when
    `time_unit` is given and `infusion` without it, so that a legend of the
    plot explains the shading; the lines are not labelled. A label already
    taken (`taken`, the labels of the plot) is left out, since two identical
    legend rows explain nothing; the label of the window is added to
    `taken`.

    Args:
        plot: the plot, its x axis carrying the times of the protocol.
        dosing: the protocol, `None` for a curve without one.

    Keyword Args:
        y_limits: the ends of the y axis, the height of a window.
        style: colors and markers (`dose_color`, `alpha`).
        min_doses: fewest doses a protocol needs for the dose lines.
        time_unit: unit of the times of the protocol, for the label of the
            window.
        taken: the labels the plot carries, extended by the label of the
            window.
    """
    if dosing is None:
        return
    taken = set() if taken is None else taken
    durations = dosing.durations
    labelled = False
    low, high = y_limits
    for i, dose_time in enumerate(dosing.times):
        duration = float(durations[i]) if durations is not None else np.nan
        if np.isfinite(duration) and duration > 0.0:
            label: str | None = "infusion"
            if time_unit is not None:
                label = f"infusion ({duration:g} {time_unit})"
            if labelled or label in taken:
                label = None
            labelled = True
            if label is not None:
                taken.add(label)
            start = float(dose_time)
            plot.area(
                [start, start + duration],
                [low, low],
                [high, high],
                color=style.dose_color,
                fill_opacity=style.alpha,
                label=label,
                legend=label,
            )
        elif dosing.n_doses >= min_doses:
            plot.rule(
                float(dose_time),
                orientation="vertical",
                color=style.dose_color,
                line="dotted",
                width=1.0,
            )


def has_infusion(dosing: Dosing | None) -> bool:
    """Whether a protocol has a dose of a duration, which `dose_markers` shades.

    Args:
        dosing: the protocol, `None` for none.

    Returns:
        `True` for a dose of a positive duration.
    """
    if dosing is None or dosing.durations is None:
        return False
    durations = np.asarray(dosing.durations, dtype=float)
    return bool(np.any(np.isfinite(durations) & (durations > 0.0)))


def unit_label(unit: str) -> str:
    """A unit as the short symbols a figure carries, `mg/l` for `milligram / liter`.

    The same rule as `pkpdutils.units.short_unit`, which the tables use: the
    canonical long form of pint is written in its short symbols, a string the
    user spelled is kept, a dimensionless quantity has no label at all, and a
    string which is not a unit of the registry is passed through unchanged.

    Args:
        unit: the unit string of a variable.

    Returns:
        The label, empty for a dimensionless or empty unit.
    """
    return short_unit(unit)


def axis_label(name: str, unit: str) -> str:
    """The label of an axis, `name [unit]`, or the name alone without a unit.

    Args:
        name: name of the variable on the axis.
        unit: its unit, as `unit_label` writes it or `""` for none.

    Returns:
        The label.
    """
    return f"{name} [{unit}]" if unit else name


def group_order(ds: xr.Dataset, sample_dims: Sequence[str], by: str) -> list[str]:
    """The distinct labels of a grouping coordinate, in the order they appear.

    The labels are the ones `sample_labels(ds, sample_dims, by=by)` gives the
    samples, so that a figure can key its colors and its legend by them.

    Args:
        ds: the dataset of the batch or of the result.
        sample_dims: the sample dimensions, in the order the samples are
            enumerated.
        by: name of the grouping coordinate.

    Returns:
        One entry per group.
    """
    order: list[str] = []
    for label in sample_labels(ds, list(sample_dims), by=by):
        if label not in order:
            order.append(label)
    return order


def sample_colors(n: int, cmap: str) -> list[str]:
    """One color per sample from a colormap, evenly spaced across `n` samples.

    Args:
        n: number of samples; colors are evenly spaced from the start to the
            end of the colormap (its start for `n <= 1`).
        cmap: name of the colormap, one of `sbmlsim_core.COLOR_SCALES`.

    Returns:
        One color `#rrggbb` per sample, `n` entries.
    """
    if n <= 0:
        return []
    positions = [i / max(n - 1, 1) for i in range(n)]
    return list(sbmlsim_core.color_scale(cmap, 0.0, 1.0, positions))


#: the last position of the colormap a group color is taken from; the very end
#: of a sequential colormap (the yellow of `viridis`) has too little contrast
#: against the white background for a line a reader has to follow
GROUP_COLOR_END = 0.85


def group_colors(n: int, cmap: str) -> list[str]:
    """One color per group of a grouped figure, evenly spaced across `n` groups.

    Like `sample_colors`, but the colors stop at `GROUP_COLOR_END` of the
    colormap: the groups of `plot_mean_timecourse`, `plot_timecourse(by=...)`
    and `plot_troughs` are few and each of them is a line the reader follows
    across the figure, which the pale end of a sequential colormap does not
    carry.

    Args:
        n: number of groups.
        cmap: name of the colormap, one of `sbmlsim_core.COLOR_SCALES`.

    Returns:
        One color `#rrggbb` per group, `n` entries.
    """
    if n <= 0:
        return []
    positions = [GROUP_COLOR_END * i / max(n - 1, 1) for i in range(n)]
    return list(sbmlsim_core.color_scale(cmap, 0.0, 1.0, positions))
