"""Ratios with their intervals against acceptance limits and interaction thresholds."""

from collections.abc import Mapping
from typing import Protocol

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator

from pkpdutils.plot._common import figure_of
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.stats.bioequivalence import BEResult
from pkpdutils.stats.ddi import DDIThresholds


class RatioLike(Protocol):
    """A ratio with an interval: `RatioResult`, `BEParameter`."""

    @property
    def gmr(self) -> float:
        """Geometric mean ratio."""
        ...

    @property
    def ci_low(self) -> float:
        """Lower bound of the interval."""
        ...

    @property
    def ci_high(self) -> float:
        """Upper bound of the interval."""
        ...

    @property
    def ci_level(self) -> float:
        """Level of the interval."""
        ...


def plot_ratio(
    ratios: Mapping[str, RatioLike] | BEResult,
    *,
    limits: tuple[float, float] | None = (0.8, 1.25),
    thresholds: DDIThresholds | None = None,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Point estimates and intervals of ratios on a logarithmic axis, with limits and thresholds.

    Args:
        ratios: name to ratio, or a bioequivalence result (its parameters).
        limits: acceptance limits drawn as dashed lines, `None` for none.
        thresholds: interaction thresholds drawn as dotted lines with the
            class names, `None` for none.
        ax: axes to draw on, a new figure by default.
        style: colors and markers.

    Returns:
        The figure.

    Raises:
        ValueError: if `ratios` (or `ratios.parameters`) is empty.
    """
    entries: Mapping[str, RatioLike] = (
        ratios.parameters if isinstance(ratios, BEResult) else ratios
    )
    if not entries:
        raise ValueError("plot_ratio needs at least one ratio")
    fig, ax = figure_of(ax)
    names = list(entries)
    ys = np.arange(len(names))
    for y, name in zip(ys, names, strict=True):
        r = entries[name]
        ax.errorbar(
            [r.gmr],
            [y],
            xerr=[[r.gmr - r.ci_low], [r.ci_high - r.gmr]],
            marker=style.data_marker,
            color=style.data_color,
            capsize=3,
            linestyle="none",
            markersize=style.markersize,
        )
    tick_values = {1.0}
    ax.axvline(1.0, color="gray", linewidth=1.0)
    if limits is not None:
        for limit in limits:
            ax.axvline(
                limit,
                color=style.limit_color,
                linestyle="--",
                linewidth=style.linewidth,
            )
            tick_values.add(limit)
    top, bottom = -0.6, len(names) - 0.4
    ax.set_xscale("log")
    ax.set_yticks(ys, names)
    ax.set_ylim(top, bottom)
    ax.invert_yaxis()
    if thresholds is not None:
        classes = {
            thresholds.inhibitor_weak: "weak inhibitor",
            thresholds.inhibitor_moderate: "moderate inhibitor",
            thresholds.inhibitor_strong: "strong inhibitor",
            thresholds.inducer_weak: "weak inducer",
            thresholds.inducer_moderate: "moderate inducer",
            thresholds.inducer_strong: "strong inducer",
        }
        for value, label in classes.items():
            ax.axvline(value, color=style.limit_color, linestyle=":", linewidth=1.0)
            tick_values.add(value)
            # anchored just below the top border, hanging down into the plot,
            # so the rotated label stays clear of the x tick labels below the axes
            ax.text(
                value,
                top + 0.05 * (bottom - top),
                label,
                rotation=90,
                fontsize="x-small",
                ha="right",
                va="top",
            )
    # the reference values (unity, limits, thresholds) are the ticks that
    # matter here; the automatic log ticks of a narrow range crowd into each
    # other and are turned off
    sorted_ticks = sorted(tick_values)
    ax.set_xticks(sorted_ticks, [f"{v:g}" for v in sorted_ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("ratio test / reference")
    levels = {round(entries[n].ci_level * 100) for n in names}
    ax.set_title(
        f"geometric mean ratios with {', '.join(str(lv) for lv in sorted(levels))} % intervals",
        fontsize="small",
    )
    return fig
