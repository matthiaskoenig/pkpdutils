"""Distribution of a parameter over the individuals, by group."""

import logging
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pkpdutils.plot._common import (
    axis_label,
    figure_of,
    legend_above_data,
    log_scale,
    unit_label,
)
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.result import ParameterResult
from pkpdutils.stats.sample import ParameterSample, Scale, coerce, summarize

logger = logging.getLogger(__name__)


def plot_parameters(
    result: ParameterResult,
    name: str,
    dim: str,
    *,
    by: str | None = None,
    log_y: bool = False,
    scale: Scale | str = Scale.LOG,
    ci_level: float = 0.95,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Figure:
    """Strip and box plot of a parameter over a sample dimension, with the mean and its interval per group.

    Every individual is a jittered point, every group a box plot, and the
    geometric mean (`scale=LOG`) or the arithmetic mean with the t interval
    of `summarize` at `ci_level` a marker with an error bar. The statistic
    is computed from every finite value of the group, whatever the axis: a
    group with a non-positive value has no geometric mean and gets the
    arithmetic mean instead, drawn with a marker of its own (logged at debug
    level).

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
    in room made above the data (`legend_above_data`), so it covers no point,
    box or interval; the room is measured on the figure as laid out when the
    function returns.

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
        ax: axes to draw on, a new figure by default; a caller-supplied `ax`
            keeps its figure's own layout engine, so long tick labels can
            clip unless the caller sets one (`fig.set_layout_engine("constrained")`).
        style: colors and markers.
        **indexers: coordinate label per remaining sample dimension.

    Returns:
        The figure.

    Raises:
        ValueError: if `by` is not a coordinate along `dim` or `scale` is not
            a `Scale`.
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
    log_axis = log_y and any(bool(np.any(v > 0)) for v in finite)
    if log_y and not log_axis:
        logger.debug("'%s' has no positive value, the y axis stays linear", name)
    # the values the axis can show: the strip and the box draw the same ones
    shown = [v[v > 0] if log_axis else v for v in finite]
    fig, ax = figure_of(ax)
    rng = np.random.default_rng(0)
    level = f"{ci_level * 100:g} % CI"
    # the marker with the whiskers is the statistic, not another data point:
    # every statistic is named once for the figure, and a group falling back
    # to the arithmetic mean gets a marker of its own
    statistic_label = {
        Scale.LOG: f"geometric mean [{level}]",
        Scale.LINEAR: f"mean [{level}]",
    }
    handles: dict[Scale, Any] = {}
    positions = np.arange(1, len(groups) + 1)
    # a group with nothing to show has no box, rather than one of `NaN` lines
    boxed = [i for i, v in enumerate(shown) if v.size]
    if boxed:
        ax.boxplot(
            [shown[i] for i in boxed],
            positions=positions[boxed],
            widths=0.5,
            showfliers=False,
            zorder=1,
        )
    for pos, label, group, v, v_plot in zip(
        positions, groups, groups.values(), finite, shown, strict=True
    ):
        if v_plot.size < v.size:
            logger.debug(
                "group '%s': %d non-positive values left out of the strip and the box",
                label,
                v.size - v_plot.size,
            )
        ax.plot(
            pos + rng.uniform(-0.15, 0.15, v_plot.size),
            v_plot,
            linestyle="none",
            marker=style.data_marker,
            color=style.data_color,
            markersize=style.markersize,
            alpha=0.6,
            zorder=2,
            label=label,
        )
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
        if log_axis and not center > 0:
            logger.debug(
                "group '%s': the mean %s cannot be shown on a logarithmic axis",
                label,
                center,
            )
            continue
        err = (
            np.array([[center - s.ci_low], [s.ci_high - center]])
            if np.isfinite(s.ci_low)
            else None
        )
        container = ax.errorbar(
            [pos + 0.3],
            [center],
            yerr=err,
            marker="D" if effective_scale is scale else "s",
            color=style.summary_color,
            capsize=3,
            linestyle="none",
            zorder=3,
            label=None
            if effective_scale in handles
            else statistic_label[effective_scale],
        )
        handles.setdefault(effective_scale, container)
    ax.set_xticks(positions, list(groups))
    ax.set_ylabel(axis_label(name, unit_label(sample.unit)))
    if by is not None:
        ax.set_xlabel(by)
    if log_axis:
        log_scale(ax, "y")
    # only the statistics are named: the groups are the ticks of the x axis
    legend_above_data(
        ax, [handles[s] for s in dict.fromkeys((scale, Scale.LINEAR)) if s in handles]
    )
    return fig
