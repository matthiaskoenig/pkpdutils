"""Ratios with their intervals against acceptance limits and interaction thresholds."""

from collections.abc import Mapping
from typing import Protocol

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator

from pkpdutils.plot._common import (
    annotate_column,
    annotation_room,
    estimate_text,
    figure_of,
    log_scale,
    make_room_right,
)
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
    annotate: bool = True,
    labels: Mapping[str, str] | None = None,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
) -> Figure:
    """Point estimates and intervals of ratios on a logarithmic axis, with limits and thresholds.

    With `annotate` the rows carry their numbers, `gmr [low, high]`, in a
    column to the right of the intervals, as the table of a bioequivalence
    report does; the axis is widened to hold the column.

    Args:
        ratios: name to ratio, or a bioequivalence result (its parameters).

    Keyword Args:
        limits: acceptance limits drawn as dashed lines, `None` for none.
        thresholds: interaction thresholds drawn as dotted lines with the
            class names, `None` for none.
        annotate: write `gmr [low, high]` beside every row.
        labels: the tick label of a row per name, e.g.
            `{"auc_inf_obs": "AUC(0-inf)"}`; a name without an entry keeps
            its own spelling.
        ax: axes to draw on, a new figure by default; a caller-supplied `ax`
            keeps its figure's own layout engine, so long tick labels can
            clip unless the caller sets one (`fig.set_layout_engine("constrained")`).
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
    annotations: list[tuple[float, str]] = []
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
        if annotate:
            annotations.append((float(y), estimate_text(r.gmr, r.ci_low, r.ci_high)))
    # unity is always drawn as the reference line; it carries a tick of its
    # own only without the interaction thresholds, whose 0.8 and 1.25 sit so
    # close to it that the three labels run into each other
    tick_values = set() if thresholds is not None else {1.0}
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
    log_scale(ax, "x")
    ax.set_yticks(ys, [(labels or {}).get(name, name) for name in names])
    ax.set_ylim(top, bottom)
    ax.invert_yaxis()
    if thresholds is not None:
        tick_values.update(_shade_interaction_classes(ax, entries, thresholds, top))
    # the reference values (unity, limits, thresholds) are the ticks that
    # matter here; the automatic log ticks of a narrow range crowd into each
    # other and are turned off
    sorted_ticks = sorted(tick_values)
    ax.set_xticks(sorted_ticks, [f"{v:g}" for v in sorted_ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    if annotations:
        room = annotation_room(ax, [text for _, text in annotations])
        annotate_column(ax, annotations, make_room_right(ax, room))
    ax.set_xlabel("ratio test / reference")
    levels = {round(entries[n].ci_level * 100) for n in names}
    ax.set_title(
        f"geometric mean ratios with {', '.join(str(lv) for lv in sorted(levels))} % intervals",
        fontsize="small",
    )
    return fig


#: fill colors of the induction classes, weak to strong (a sequential blue)
INDUCTION_COLORS: tuple[str, str, str] = ("#deebf7", "#9ecae1", "#4292c6")

#: fill colors of the inhibition classes, weak to strong (a sequential orange)
INHIBITION_COLORS: tuple[str, str, str] = ("#fee6ce", "#fdae6b", "#e6550d")

#: fill color of the range without an interaction
NO_INTERACTION_COLOR = "#f2f2f2"

#: transparency of the class bands
CLASS_ALPHA = 0.45


def _shade_interaction_classes(
    ax: Axes,
    entries: Mapping[str, RatioLike],
    thresholds: DDIThresholds,
    top: float,
) -> set[float]:
    """Shade the classes of an interaction as bands of the ratio axis.

    The axis is cut at the thresholds into the bands `strong inducer`,
    `moderate inducer`, `weak inducer`, `no interaction`, `weak inhibitor`,
    `moderate inhibitor` and `strong inhibitor`, each filled with its own
    color (blues for induction, oranges for inhibition, darker the stronger
    the class, light gray for none) and named at the top; the axis is widened
    to show every band, at least a factor of 2 beyond the strong thresholds.

    Args:
        ax: the axes of the ratios.
        entries: the ratios drawn, to keep their intervals inside the axis.
        thresholds: the class thresholds.
        top: the y position of the top border, where the names hang from.

    Returns:
        The threshold values, for the ticks of the axis.
    """
    edges = [
        thresholds.inducer_strong,
        thresholds.inducer_moderate,
        thresholds.inducer_weak,
        thresholds.inhibitor_weak,
        thresholds.inhibitor_moderate,
        thresholds.inhibitor_strong,
    ]
    low = min(edges[0] / 2.0, *(r.ci_low for r in entries.values()))
    high = max(edges[-1] * 2.0, *(r.ci_high for r in entries.values()))
    low = low / 1.1
    high = high * 1.1
    bounds = [low, *edges, high]
    names = [
        "strong inducer",
        "moderate inducer",
        "weak inducer",
        "no interaction",
        "weak inhibitor",
        "moderate inhibitor",
        "strong inhibitor",
    ]
    colors = [
        *reversed(INDUCTION_COLORS),
        NO_INTERACTION_COLOR,
        *INHIBITION_COLORS,
    ]
    for left, right, name, color in zip(
        bounds[:-1], bounds[1:], names, colors, strict=True
    ):
        ax.axvspan(left, right, color=color, alpha=CLASS_ALPHA, linewidth=0, zorder=0)
        ax.text(
            float(np.sqrt(left * right)),
            top,
            name,
            rotation=90,
            fontsize="x-small",
            ha="center",
            va="top",
            color="0.25",
        )
    ax.set_xlim(low, high)
    return set(edges)
