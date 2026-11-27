"""Ratios with their intervals against acceptance limits and interaction thresholds."""

from collections.abc import Mapping
from typing import Protocol

import numpy as np
from sbmlsim_core.chart import Chart, ChartAxis

from pkpdutils.plot._common import (
    FONT_SIZES,
    annotate_column,
    annotation_room,
    chart_of,
    data_limits,
    estimate_text,
    make_room_right,
    marker,
    row_axis,
    row_panel_width,
)
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.stats.bioequivalence import BEResult
from pkpdutils.stats.ddi import DDIThresholds

#: the fonts of the charts of the module: the title in matplotlib's `small`
_FONTS: dict[str, float | str] = {"plot_title": FONT_SIZES["small"]}


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
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
) -> Chart:
    """Point estimates and intervals of ratios on a logarithmic axis, with limits and thresholds.

    With `annotate` the rows carry their numbers, `gmr [low, high]`, in a
    column to the right of the intervals, as the table of a bioequivalence
    report does; the axis is widened to hold the column.

    Args:
        ratios: name to ratio, or a bioequivalence result (its parameters).

    Keyword Args:
        limits: acceptance limits drawn as dashed lines, `None` for none.
        thresholds: interaction thresholds drawn as shaded classes with their
            names, `None` for none.
        annotate: write `gmr [low, high]` beside every row.
        labels: the tick label of a row per name, e.g.
            `{"auc_inf_obs": "AUC(0-inf)"}`; a name without an entry keeps
            its own spelling.
        chart: the chart to draw into, a new one by default.
        row: the row of the panel in `chart`.
        column: the column of the panel in `chart`.
        style: colors and markers.

    Returns:
        The chart.

    Raises:
        ValueError: if `ratios` (or `ratios.parameters`) is empty, or if the
            cell is not in the grid of `chart`.
    """
    entries: Mapping[str, RatioLike] = (
        ratios.parameters if isinstance(ratios, BEResult) else ratios
    )
    if not entries:
        raise ValueError("plot_ratio needs at least one ratio")
    names = list(entries)
    tick_labels = [(labels or {}).get(name, name) for name in names]
    annotations = (
        [estimate_text(r.gmr, r.ci_low, r.ci_high) for r in entries.values()]
        if annotate
        else []
    )
    # unity is always drawn as the reference line; it carries a tick of its
    # own only without the interaction thresholds, whose 0.8 and 1.25 sit so
    # close to it that the three labels run into each other
    tick_values: set[float] = set() if thresholds is not None else {1.0}
    references = [1.0, *(limits or ())]
    tick_values.update(limits or ())
    classes: list[tuple[float, float, str, str]] = []
    if thresholds is not None:
        low, high, classes = _interaction_classes(entries, thresholds)
        tick_values.update(bound for bound, *_ in classes[1:])
    else:
        low, high = data_limits(
            [r.ci_low for r in entries.values()],
            [r.ci_high for r in entries.values()],
            references,
            log=True,
        )
    target = chart_of(
        chart, row, column, rows=1, columns=1, size=(6.0, 4.0), fonts=_FONTS
    )
    data_end = high
    if annotations:
        width = row_panel_width(target, tick_labels)
        high = make_room_right(low, high, annotation_room(annotations, width), log=True)
    # the reference values (unity, limits, thresholds) are the ticks that
    # matter here; the automatic log ticks of a narrow range crowd into each
    # other
    ticks = sorted(tick_values)
    levels = {round(entries[n].ci_level * 100) for n in names}
    top = -0.6
    plot = target.plot(
        row,
        column,
        title=(
            "geometric mean ratios with "
            f"{', '.join(str(lv) for lv in sorted(levels))} % intervals"
        ),
        x=ChartAxis(
            label="ratio test / reference",
            scale="log",
            min=low,
            max=high,
            ticks=ticks,
            tick_labels=[f"{v:g}" for v in ticks],
        ),
        y=row_axis(tick_labels),
        legend=False,
    )
    for left, right, _, color in classes:
        plot.area(
            [left, right],
            [top, top],
            [len(names) - 0.4] * 2,
            color=color,
            fill_opacity=CLASS_ALPHA,
        )
    plot.rule(1.0, orientation="vertical", color="gray", width=1.0)
    for limit in limits or ():
        plot.rule(
            limit,
            orientation="vertical",
            color=style.limit_color,
            line="dashed",
            width=style.linewidth,
        )
    for y, name in enumerate(names):
        r = entries[name]
        plot.curve(
            [r.gmr],
            [y],
            xerr=([r.gmr - r.ci_low], [r.ci_high - r.gmr]),
            error_cap=3.0,
            error_width=1.5,
            line="none",
            marker=marker(style.data_marker),
            size=style.markersize,
            color=style.data_color,
        )
    for left, right, name, _ in classes:
        plot.text(
            float(np.sqrt(left * right)),
            top,
            name,
            rotation=90.0,
            size=FONT_SIZES["x-small"],
            anchor="middle",
            baseline="top",
            color="#404040",
        )
    annotate_column(plot, annotations, data_end)
    return target


#: fill colors of the induction classes, weak to strong (a sequential blue)
INDUCTION_COLORS: tuple[str, str, str] = ("#deebf7", "#9ecae1", "#4292c6")

#: fill colors of the inhibition classes, weak to strong (a sequential orange)
INHIBITION_COLORS: tuple[str, str, str] = ("#fee6ce", "#fdae6b", "#e6550d")

#: fill color of the range without an interaction
NO_INTERACTION_COLOR = "#f2f2f2"

#: transparency of the class bands
CLASS_ALPHA = 0.45


def _interaction_classes(
    entries: Mapping[str, RatioLike], thresholds: DDIThresholds
) -> tuple[float, float, list[tuple[float, float, str, str]]]:
    """The classes of an interaction as bands of the ratio axis.

    The axis is cut at the thresholds into the bands `strong inducer`,
    `moderate inducer`, `weak inducer`, `no interaction`, `weak inhibitor`,
    `moderate inhibitor` and `strong inhibitor`, each filled with its own
    color (blues for induction, oranges for inhibition, darker the stronger
    the class, light gray for none) and named at the top; the axis is widened
    to show every band, at least a factor of 2 beyond the strong thresholds.

    Args:
        entries: the ratios drawn, to keep their intervals inside the axis.
        thresholds: the class thresholds.

    Returns:
        The ends of the axis and the classes, each its bounds, its name and
        its color.
    """
    edges = [
        thresholds.inducer_strong,
        thresholds.inducer_moderate,
        thresholds.inducer_weak,
        thresholds.inhibitor_weak,
        thresholds.inhibitor_moderate,
        thresholds.inhibitor_strong,
    ]
    low = min(edges[0] / 2.0, *(r.ci_low for r in entries.values())) / 1.1
    high = max(edges[-1] * 2.0, *(r.ci_high for r in entries.values())) * 1.1
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
    colors = [*reversed(INDUCTION_COLORS), NO_INTERACTION_COLOR, *INHIBITION_COLORS]
    classes = [
        (left, right, name, color)
        for left, right, name, color in zip(
            bounds[:-1], bounds[1:], names, colors, strict=True
        )
    ]
    return low, high, classes
