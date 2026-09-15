"""Helpers shared by the figures: axes and figure creation, ticks, labels and colors."""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LogFormatter

from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.timecourse import Dosing

logger = logging.getLogger(__name__)


def figure_of(
    ax: Axes | None, figsize: tuple[float, float] = (6, 4)
) -> tuple[Figure, Axes]:
    """The axes to draw on and its figure, a new figure without `ax`.

    A function drawing into a caller-supplied `ax` never touches its
    figure's layout engine or size; only a figure created here gets the
    constrained layout engine, so a caller-supplied `ax` keeps its figure's
    own layout, meaning long tick labels can clip unless the caller sets one
    (`fig.set_layout_engine("constrained")`).

    Args:
        ax: axes to draw on, `None` for a new figure.
        figsize: size of a new figure; ignored when `ax` is given.

    Returns:
        The figure and the axes to draw on.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_layout_engine("constrained")
        return fig, ax
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    return fig, ax


def axes_of(
    axes: Any | None, nrows: int, ncols: int, figsize: tuple[float, float]
) -> tuple[Figure, np.ndarray]:
    """The grid of axes to draw on and its figure, a new figure without `axes`.

    Mirrors `figure_of` for a multi-panel figure: a function drawing into
    caller-supplied axes never touches their figure's layout engine.

    Args:
        axes: existing axes to draw the grid into, `nrows * ncols` of them
            (any shape, flattened in C order), `None` for a new figure.
        nrows: number of rows of the grid.
        ncols: number of columns of the grid.
        figsize: size of a new figure; ignored when `axes` is given.

    Returns:
        The figure and the axes, always as an `(nrows, ncols)` array.

    Raises:
        ValueError: if `axes` does not have `nrows * ncols` entries.
    """
    if axes is None:
        fig, new_axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False
        )
        fig.set_layout_engine("constrained")
        return fig, new_axes
    flat = np.asarray(axes).reshape(-1)
    if flat.size != nrows * ncols:
        raise ValueError(
            f"{nrows * ncols} axes are needed for a {nrows}x{ncols} grid, got {flat.size}"
        )
    grid = flat.reshape(nrows, ncols)
    fig = flat[0].get_figure()
    assert isinstance(fig, Figure)
    return fig, grid


class _PlainLogFormatter(LogFormatter):
    """A `LogFormatter` which writes the labels as plain numbers, not as `10^n`.

    Only the rendering of a labelled tick is changed; which ticks carry a
    label stays with `LogFormatter`, which drops the labels between the
    decades once an axis spans more than a couple of them.
    """

    def _num_to_string(self, x: float, vmin: float, vmax: float) -> str:
        """The label of a tick value.

        Args:
            x: the tick value.
            vmin: lower bound of the view interval (unused).
            vmax: upper bound of the view interval (unused).

        Returns:
            The value as a plain number.
        """
        return f"{x:g}"


def plain_log_ticks(axis: Axis) -> None:
    """Format the major and minor ticks of a logarithmic axis as plain numbers.

    Args:
        axis: the axis (`ax.xaxis` or `ax.yaxis`) to format.
    """
    axis.set_major_formatter(_PlainLogFormatter())
    axis.set_minor_formatter(
        _PlainLogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5))
    )


def log_scale(ax: Axes, which: Literal["x", "y"]) -> None:
    """Set a logarithmic scale on one axis of `ax`, with plain tick labels.

    The axis is scaled from the extent of the data already drawn on `ax`
    (`ax.dataLim`), so this must run after the data is plotted. Without a
    positive value the axis stays linear and the fallback is logged at debug
    level, instead of the scale raising `UserWarning: Data has no positive
    values, and therefore cannot be log-scaled` at draw or save time.

    Args:
        ax: axes already carrying the data.
        which: `"x"` or `"y"`, the axis to scale.
    """
    interval = ax.dataLim.intervalx if which == "x" else ax.dataLim.intervaly
    if not np.isfinite(interval).any() or float(np.nanmax(interval)) <= 0.0:
        logger.debug("the %s axis has no positive value, staying linear", which)
        return
    if which == "x":
        ax.set_xscale("log")
        plain_log_ticks(ax.xaxis)
    else:
        ax.set_yscale("log")
        plain_log_ticks(ax.yaxis)


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
            text = format_value(ds.coords[dim].to_numpy()[position])
            unit = str(ds.coords[dim].attrs.get("units", "")) or (
                (units or {}).get(dim) or ""
            )
            if unit and unit != "dimensionless":
                text = f"{text} {unit}"
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


#: the share of the x range added to the right of the data as room for the
#: annotations of a row figure (`plot_ratio`, `plot_forest`)
ANNOTATION_ROOM = 0.45

#: font size of the annotations of a row figure
ANNOTATION_FONTSIZE = "x-small"

#: width of a character of the annotations as a share of the font size; the
#: figures are drawn in the default proportional font, whose digits, spaces
#: and brackets average close to this
_CHARACTER_WIDTH = 0.62


def annotation_room(ax: Axes, texts: Sequence[str]) -> float:
    """The share of the x range a column of annotations needs.

    The width of the column is estimated from the longest text and the font
    size. The width of the axes is measured on the drawn figure, since the
    layout engine gives an axes with long tick labels (the study names of a
    forest plot) much less of the figure than the default position says; the
    figure is drawn once here for that.

    Args:
        ax: the axes the column is written on.
        texts: the annotations.

    Returns:
        The share of the current x range to add to the right, `0` without
        texts, at most `1.5`.
    """
    if not texts:
        return 0.0
    size = FontProperties(size=ANNOTATION_FONTSIZE).get_size_in_points()
    needed = max(len(text) for text in texts) * _CHARACTER_WIDTH * size
    figure = ax.get_figure()
    assert isinstance(figure, Figure)
    figure.canvas.draw()
    available = ax.get_window_extent().width * 72.0 / figure.dpi
    if available <= needed:
        return 1.5
    return min(max(needed / (available - needed), 0.15), 1.5)


def make_room_right(ax: Axes, share: float = ANNOTATION_ROOM) -> float:
    """Widen the x axis to the right, for annotations written beside the data.

    The annotations of a row figure (`estimate [low, high]`) are written in
    one column to the right of the data, where they neither cover the
    intervals nor run out of the axes, in which the layout engine does not
    see them and the figure edge cuts them off. The room is a share of the
    range the data spans, in the units of the axis: a logarithmic axis is
    widened by the same share of its decades. Call it once the scale and the
    ticks of the axis are final, since setting a scale autoscales the view
    again.

    Args:
        ax: the axes, with its data already drawn.
        share: the share of the current range to add.

    Returns:
        The position of the column, as the fraction of the widened axes at
        which the data ended.
    """
    low, high = ax.get_xlim()
    if ax.get_xscale() == "log" and low > 0.0:
        ax.set_xlim(low, high * (high / low) ** share)
    else:
        ax.set_xlim(low, high + share * (high - low))
    return 1.0 / (1.0 + share)


def annotate_column(
    ax: Axes, rows: Sequence[tuple[float, str]], fraction: float
) -> None:
    """Write one annotation per row in a column at `fraction` of the axes.

    Args:
        ax: the axes; its x limits are final (`make_room_right`).
        rows: the position of every row on the y axis, in the units of the
            data, with the text to write there.
        fraction: the x position of the column, as a fraction of the axes.
    """
    for y, text in rows:
        ax.annotate(
            text,
            (fraction, y),
            xycoords=ax.get_yaxis_transform(),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=ANNOTATION_FONTSIZE,
        )


def dose_markers(
    ax: Axes,
    dosing: Dosing | None,
    *,
    style: PlotStyle = DEFAULT_STYLE,
    min_doses: int = 2,
) -> None:
    """Mark the doses of a protocol on the time axis of `ax`.

    An infusion is a window: the span from the dose time to the dose time
    plus the duration of the dose is shaded, so that the figure shows how
    long the drug went in. Every other route is an instant and gets a thin
    dotted line, drawn only for a protocol of at least `min_doses` doses,
    since the single line at the dose time of a single dose curve carries no
    information.

    The first window is labelled `"infusion"`, so that a legend of the axes
    explains the shading; the lines are not labelled.

    Args:
        ax: axes to draw on, its x axis carrying the times of the protocol.
        dosing: the protocol, `None` for a curve without one.

    Keyword Args:
        style: colors and markers (`dose_color`, `alpha`).
        min_doses: fewest doses a protocol needs for the dose lines.
    """
    if dosing is None:
        return
    durations = dosing.durations
    labelled = False
    for i, dose_time in enumerate(dosing.times):
        duration = float(durations[i]) if durations is not None else np.nan
        if np.isfinite(duration) and duration > 0.0:
            ax.axvspan(
                float(dose_time),
                float(dose_time) + duration,
                color=style.dose_color,
                alpha=style.alpha,
                linewidth=0.0,
                # behind every other artist, the shaded areas of the NCA panel
                # included, so that the window never hides the data
                zorder=0.0,
                label=None if labelled else "infusion",
            )
            labelled = True
        elif dosing.n_doses >= min_doses:
            ax.axvline(
                float(dose_time), color=style.dose_color, linestyle=":", linewidth=1
            )


def sample_colors(n: int, cmap: str) -> list[Any]:
    """One color per sample from a colormap, evenly spaced across `n` samples.

    Args:
        n: number of samples; colors are evenly spaced from the start to the
            end of the colormap (`cmap(0.0)` for `n <= 1`).
        cmap: name of the colormap.

    Returns:
        One RGBA color per sample, `n` entries.
    """
    palette = plt.get_cmap(cmap)
    return [palette(i / max(n - 1, 1)) for i in range(n)]


#: the last position of the colormap a group color is taken from; the very end
#: of a sequential colormap (the yellow of `viridis`) has too little contrast
#: against the white background for a line a reader has to follow
GROUP_COLOR_END = 0.85


def group_colors(n: int, cmap: str) -> list[Any]:
    """One color per group of a grouped figure, evenly spaced across `n` groups.

    Like `sample_colors`, but the colors stop at `GROUP_COLOR_END` of the
    colormap: the groups of `plot_mean_timecourse`, `plot_timecourse(by=...)`
    and `plot_troughs` are few and each of them is a line the reader follows
    across the figure, which the pale end of a sequential colormap does not
    carry.

    Args:
        n: number of groups.
        cmap: name of the colormap.

    Returns:
        One RGBA color per group, `n` entries.
    """
    palette = plt.get_cmap(cmap)
    return [palette(GROUP_COLOR_END * i / max(n - 1, 1)) for i in range(n)]
