"""Distribution of a parameter over the individuals, by group."""

import logging
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.plot.timecourse import _figure_of, _plain_log_ticks
from pkpdutils.result import ParameterResult
from pkpdutils.stats.sample import ParameterSample, Scale, summarize

logger = logging.getLogger(__name__)


def plot_parameters(
    result: ParameterResult,
    name: str,
    dim: str,
    *,
    by: str | None = None,
    log: bool = False,
    scale: Scale = Scale.LOG,
    ci_level: float = 0.95,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    **indexers: Any,
) -> Figure:
    """Strip and box plot of a parameter over a sample dimension, with the mean and its interval per group.

    Every individual is a jittered point, every group a box plot, and the
    geometric mean (`scale=LOG`) or the arithmetic mean with the t interval
    of `summarize` at `ci_level` a marker with an error bar.

    Args:
        result: the result the parameter is taken from.
        name: name of the parameter.
        dim: the sample dimension of the individuals.
        by: a coordinate along `dim` which groups the individuals, one
            group named after the parameter without it.
        log: logarithmic y axis.
        scale: scale of the mean and its interval.
        ci_level: level of the interval.
        ax: axes to draw on, a new figure by default.
        style: colors and markers.
        **indexers: coordinate label per remaining sample dimension.

    Returns:
        The figure.

    Raises:
        ValueError: if `by` is not a coordinate along `dim`.
    """
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
    fig, ax = _figure_of(ax)
    rng = np.random.default_rng(0)
    positions = np.arange(1, len(groups) + 1)
    values = [g.finite_values for g in groups.values()]
    ax.boxplot(values, positions=positions, widths=0.5, showfliers=False, zorder=1)
    for pos, (label, group) in zip(positions, groups.items(), strict=True):
        v = group.finite_values
        v_plot = v[v > 0] if log else v
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
            ax.errorbar(
                [pos + 0.3],
                [center],
                yerr=err,
                marker="D",
                color=style.summary_color,
                capsize=3,
                linestyle="none",
                zorder=3,
            )
    ax.set_xticks(positions, list(groups))
    ax.set_ylabel(f"{name} [{sample.unit}]")
    if by is not None:
        ax.set_xlabel(by)
    if log:
        ax.set_yscale("log")
        _plain_log_ticks(ax.yaxis)
    return fig
