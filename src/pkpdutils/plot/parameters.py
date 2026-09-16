"""Distribution of a parameter over the individuals, by group."""

import logging
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pkpdutils.plot._common import axis_label, figure_of, log_scale, unit_label
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
    of `summarize` at `ci_level` a marker with an error bar. A group with a
    non-positive value gets the arithmetic mean marker instead of the
    geometric one, which is logged at debug level. With `log_y=True` the
    non-positive points are left out of the strip, since a logarithmic axis
    cannot show them.

    The legend names the marker of the statistic once for the figure
    (`geometric mean [95 % CI]`, `mean [...]` with `scale=Scale.LINEAR`); the
    groups themselves are the ticks of the x axis and stay out of it.

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
    fig, ax = figure_of(ax)
    rng = np.random.default_rng(0)
    # the marker with the whiskers is the statistic, not another data point:
    # it is named once for the figure, since every group draws the same thing
    summary_label = (
        f"{'geometric mean' if scale is Scale.LOG else 'mean'} "
        f"[{ci_level * 100:g} % CI]"
    )
    summary_handle: Any = None
    positions = np.arange(1, len(groups) + 1)
    values = [g.finite_values for g in groups.values()]
    ax.boxplot(values, positions=positions, widths=0.5, showfliers=False, zorder=1)
    for pos, (label, group) in zip(positions, groups.items(), strict=True):
        v = group.finite_values
        v_plot = v[v > 0] if log_y else v
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
        if v.size:
            effective_scale = scale
            if scale is Scale.LOG and np.any(v <= 0):
                logger.debug(
                    "group '%s' has non-positive values, the marker uses the arithmetic mean",
                    label,
                )
                effective_scale = Scale.LINEAR
            s = summarize(group, scale=effective_scale, ci_level=ci_level)
            center = s.geomean if effective_scale is Scale.LOG else s.mean
            err = (
                np.array([[center - s.ci_low], [s.ci_high - center]])
                if np.isfinite(s.ci_low)
                else None
            )
            container = ax.errorbar(
                [pos + 0.3],
                [center],
                yerr=err,
                marker="D",
                color=style.summary_color,
                capsize=3,
                linestyle="none",
                zorder=3,
                label=summary_label if summary_handle is None else None,
            )
            if summary_handle is None:
                summary_handle = container
    if summary_handle is not None:
        # only the statistic is named: the groups are the ticks of the x axis
        ax.legend(handles=[summary_handle], fontsize="small")
    ax.set_xticks(positions, list(groups))
    ax.set_ylabel(axis_label(name, unit_label(sample.unit)))
    if by is not None:
        ax.set_xlabel(by)
    if log_y:
        log_scale(ax, "y")
    return fig
