"""Distribution of a parameter over the individuals, by group."""

import logging
from typing import Any

import numpy as np
from sbmlsim_core.chart import Chart, ChartAxis

from pkpdutils.plot._common import (
    axis_label,
    chart_of,
    data_limits,
    legend_room,
    log_axis,
    marker,
    unit_label,
    value_axis,
)
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.result import ParameterResult
from pkpdutils.stats.sample import ParameterSample, Scale, coerce, summarize

logger = logging.getLogger(__name__)

#: the size of a chart of its own, in inches
_SIZE = (6.0, 4.0)

#: the room of the axes around a panel of a chart of its own in pixels, the
#: labels and ticks left and below and the margins: a panel of the chart is
#: its size less these
_AXES_ROOM = (70.0, 60.0)


def box_stats(values: np.ndarray, whis: float = 1.5) -> dict[str, float]:
    """The statistics of a box plot of values, as matplotlib's `boxplot` computes them.

    The quartiles are the linearly interpolated percentiles, and a whisker
    reaches the most extreme value within `whis` times the interquartile
    range from its quartile (Tukey's rule).

    Args:
        values: the finite values, at least one.
        whis: the reach of the whiskers in interquartile ranges.

    Returns:
        `low_whisker`, `q1`, `median`, `q3` and `high_whisker`.
    """
    q1, median, q3 = (float(v) for v in np.percentile(values, [25, 50, 75]))
    iqr = q3 - q1
    above = values[values <= q3 + whis * iqr]
    below = values[values >= q1 - whis * iqr]
    return {
        "low_whisker": float(below.min()) if below.size else q1,
        "q1": q1,
        "median": median,
        "q3": q3,
        "high_whisker": float(above.max()) if above.size else q3,
    }


def plot_parameters(
    result: ParameterResult,
    name: str,
    dim: str,
    *,
    by: str | None = None,
    log_y: bool = False,
    scale: Scale | str = Scale.LOG,
    ci_level: float = 0.95,
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Chart:
    """Strip and box plot of a parameter over a sample dimension, with the mean and its interval per group.

    Every individual is a jittered point, every group a box plot (matplotlib's
    boxplot without fliers, `box_stats`), and the geometric mean
    (`scale=LOG`) or the arithmetic mean with the t interval of `summarize` at
    `ci_level` a marker with an error bar. The statistic is computed from
    every finite value of the group, whatever the axis: a group with a
    non-positive value has no geometric mean and gets the arithmetic mean
    instead, drawn with a marker of its own (logged at debug level).

    With `log_y=True` the non-positive values are left out of both the strip
    and the box, which are drawn from the same values, since a logarithmic
    axis cannot show them; a group without a positive value keeps its tick
    and nothing else, and a marker whose mean is not positive is left out.
    Without a positive value in any group the axis stays linear and every
    value is drawn.

    The legend names the marker of every statistic once for the figure
    (`geometric mean [95 % CI]`, `mean [...]` with `scale=Scale.LINEAR` or
    for the groups which fall back to it); the groups themselves are the
    ticks of the x axis and stay out of it. It sits in the upper right corner
    in room made above the data (`legend_room`), so it covers no point, box
    or interval.

    Args:
        result: the result the parameter is taken from.
        name: name of the parameter.
        dim: the sample dimension of the individuals.

    Keyword Args:
        by: a coordinate along `dim` which groups the individuals, one
            group named after the parameter without it.
        log_y: logarithmic y axis; without a positive value across every group
            the axis stays linear (logged at debug level).
        scale: scale of the mean and its interval, the member or its string
            (`Scale.LOG`, `"log"`).
        ci_level: level of the interval.
        chart: the chart to draw into, a new one by default.
        row: the row of the panel in `chart`.
        column: the column of the panel in `chart`.
        style: colors and markers.
        **indexers: coordinate label per remaining sample dimension.

    Returns:
        The chart.

    Raises:
        ValueError: if `by` is not a coordinate along `dim`, if `scale` is not
            a `Scale`, or if the cell is not in the grid of `chart`.
    """
    scale = coerce(scale, Scale)
    sample = result.sample(name, dim, **indexers)
    assert sample.values is not None
    if by is None:
        groups: dict[str, ParameterSample] = {name: sample}
    else:
        if by not in sample.coords:
            raise ValueError(
                f"'{by}' is not a coordinate along '{dim}': {sorted(sample.coords)}"
            )
        keys = sample.coords[by]
        groups = {
            str(key): sample.select(keys == key) for key in dict.fromkeys(keys.tolist())
        }
    finite = [group.finite_values for group in groups.values()]
    log = log_axis(log_y, "y", *finite)
    # the values the axis can show: the strip and the box draw the same ones
    shown = [v[v > 0] if log else v for v in finite]
    rng = np.random.default_rng(0)
    level = f"{ci_level * 100:g} % CI"
    # the marker with the whiskers is the statistic, not another data point:
    # every statistic is named once for the figure, and a group falling back
    # to the arithmetic mean gets a marker of its own
    statistic_label = {
        Scale.LOG: f"geometric mean [{level}]",
        Scale.LINEAR: f"mean [{level}]",
    }
    strips: list[tuple[np.ndarray, np.ndarray]] = []
    means: list[tuple[float, float, float, float, Scale]] = []
    for pos, label, group, v, v_plot in zip(
        range(len(groups)), groups, groups.values(), finite, shown, strict=True
    ):
        if v_plot.size < v.size:
            logger.debug(
                "group '%s': %d non-positive values left out of the strip and the box",
                label,
                v.size - v_plot.size,
            )
        strips.append((pos + rng.uniform(-0.15, 0.15, v_plot.size), v_plot))
        if not v.size:
            continue
        effective_scale = scale
        if scale is Scale.LOG and np.any(v <= 0):
            logger.debug(
                "group '%s' has non-positive values, the marker uses the arithmetic mean",
                label,
            )
            effective_scale = Scale.LINEAR
        s = summarize(group, scale=effective_scale, ci_level=ci_level)
        center = s.geomean if effective_scale is Scale.LOG else s.mean
        if log and not center > 0:
            logger.debug(
                "group '%s': the mean %s cannot be shown on a logarithmic axis",
                label,
                center,
            )
            continue
        means.append((pos + 0.3, center, s.ci_low, s.ci_high, effective_scale))
    # a group with nothing to show has no box, rather than one of `NaN` lines
    boxed = [i for i, v in enumerate(shown) if v.size]
    stats = [box_stats(shown[i]) for i in boxed]
    limits = data_limits(
        *shown,
        *(list(s.values()) for s in stats),
        *([m[1], m[2], m[3]] for m in means),
        log=log,
    )
    # only the statistics are named: the groups are the ticks of the x axis
    named = [
        s for s in dict.fromkeys((scale, Scale.LINEAR)) if s in {m[4] for m in means}
    ]
    labels = [statistic_label[s] for s in named]
    target = chart_of(chart, row, column, rows=1, columns=1, size=_SIZE)
    panel = (
        target.width / target.columns - _AXES_ROOM[0],
        target.height / target.rows - _AXES_ROOM[1],
    )
    top, columns = legend_room(labels, limits, panel, log=log)
    names = list(groups)
    plot = target.plot(
        row,
        column,
        x=ChartAxis(
            label=by,
            scale="category",
            categories=names,
            label_visible=by is not None,
        ),
        y=value_axis(
            axis_label(name, unit_label(sample.unit)), log=log, limits=(limits[0], top)
        ),
        legend=bool(labels),
        legend_loc="upper right",
        legend_columns=columns,
        legend_order=labels,
    )
    if boxed:
        plot.box(
            [float(i) for i in boxed],
            stats,
            box_width=0.5,
            median_color="tab:orange",
            width=1.0,
            fill_opacity=0.0,
            color="black",
        )
    for x, v in strips:
        plot.curve(
            x,
            v,
            line="none",
            marker=marker(style.data_marker),
            size=style.markersize,
            color=style.data_color,
            opacity=0.6,
        )
    for x, center, low, high, effective in means:
        err = ([center - low], [high - center]) if np.isfinite(low) else None
        label = statistic_label[effective]
        plot.curve(
            [x],
            [center],
            yerr=err,
            error_cap=3.0,
            line="none",
            marker="diamond" if effective is scale else "square",
            size=6.0,
            color=style.summary_color,
            label=label,
            legend=label,
        )
    return target
