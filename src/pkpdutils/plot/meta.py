"""Forest plot of a meta-analysis."""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from sbmlsim_core.chart import Chart

from pkpdutils.plot._common import (
    FONT_SIZES,
    annotate_column,
    annotation_room,
    chart_of,
    data_limits,
    estimate_text,
    make_room_right,
    row_axis,
    row_panel_width,
    value_axis,
)
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.stats.meta import EffectKind, MetaResult

#: axis label per kind of effect
_LABELS = {
    EffectKind.HEDGES_G: "Hedges' g",
    EffectKind.MEAN_DIFF: "mean difference",
    EffectKind.LOG_RATIO: "log ratio",
}

#: the fonts of the charts of the module: the title in matplotlib's `small`
_FONTS: dict[str, float | str] = {"plot_title": FONT_SIZES["small"]}


def plot_forest(
    result: MetaResult,
    *,
    exp: bool | None = None,
    annotate: bool = True,
    chart: Chart | None = None,
    row: int = 0,
    column: int = 0,
    style: PlotStyle = DEFAULT_STYLE,
) -> Chart:
    """Forest plot: the effect of every study with its interval, the pooled effects as diamonds.

    The marker area of a study is proportional to its random effects
    weight. A `LOG_RATIO` analysis is shown as ratios on a logarithmic axis
    (`exp=None` or `True`); the other kinds stay on their scale.

    With `annotate` the rows carry their numbers in a column to the right of
    the intervals, as a published forest plot does: `estimate [low, high]`
    and, for a study, the random effects weight in percent; the axis is
    widened to hold the column.

    Args:
        result: the meta-analysis.

    Keyword Args:
        exp: exponentiate the effects; `None` does so for `LOG_RATIO`.
        annotate: write the effect with its interval (and the weight of a
            study) beside every row.
        chart: the chart to draw into, a new one by default.
        row: the row of the panel in `chart`.
        column: the column of the panel in `chart`.
        style: colors and markers.

    Returns:
        The chart.

    Raises:
        ValueError: if the cell is not in the grid of `chart`.
    """
    use_exp = result.kind is EffectKind.LOG_RATIO if exp is None else exp
    transform: Callable[[Any], np.ndarray] = (
        np.exp if use_exp else (lambda v: np.asarray(v, dtype=float))
    )
    n = result.n_studies
    weights = result.random.weights
    studies: list[tuple[float, float, float]] = []
    annotations: list[str] = []
    for i, e in enumerate(result.effects):
        est, low, high = (
            float(transform(v)) for v in (e.estimate, e.ci_low, e.ci_high)
        )
        studies.append((est, low, high))
        if annotate:
            annotations.append(
                f"{estimate_text(est, low, high)}, {100 * weights[i]:.1f} %"
            )
    pooled: list[tuple[float, float, float]] = []
    for effect in (result.fixed, result.random):
        est, low, high = (
            float(transform(v))
            for v in (effect.estimate, effect.ci_low, effect.ci_high)
        )
        pooled.append((est, low, high))
        if annotate:
            annotations.append(estimate_text(est, low, high))
    null = 1.0 if use_exp else 0.0
    labels = [
        *result.labels,
        "fixed effect",
        f"random effects (tau2 = {result.random.tau2:.3g})",
    ]
    x_low, x_high = data_limits(
        [v for row_ in (*studies, *pooled) for v in row_], [null], log=use_exp
    )
    target = chart_of(
        chart, row, column, rows=1, columns=1, size=(6.0, 4.0), fonts=_FONTS
    )
    data_end = x_high
    if annotations:
        # after the scale of the axis is final, which the room is a share of
        room = annotation_room(annotations, row_panel_width(target, labels))
        x_high = make_room_right(x_low, x_high, room, log=use_exp)
    het = result.heterogeneity
    plot = target.plot(
        row,
        column,
        title=(
            f"Q = {het.q:.2f} (df = {het.df}, p = {het.p_value:.2g}), "
            f"I² = {het.i2:.1f} %"
        ),
        x=value_axis(
            "ratio treatment / control" if use_exp else _LABELS[result.kind],
            log=use_exp,
            limits=(x_low, x_high),
        ),
        y=row_axis(labels),
        legend=False,
    )
    plot.rule(null, orientation="vertical", color="gray", width=1.0)
    for i, (est, low, high) in enumerate(studies):
        plot.curve([low, high], [i, i], color=style.data_color, width=style.linewidth)
        # a scatter of matplotlib of the area `40 + 400 w`, whose diameter is
        # the root of it, with an edge of 1 point in its color
        plot.curve(
            [est],
            [i],
            line="none",
            marker="square",
            size=math.sqrt(40 + 400 * weights[i]),
            color=style.data_color,
        )
    for j, (est, low, high) in enumerate(pooled):
        y = n + j
        plot.polygon(
            [[(low, y), (est, y - 0.3), (high, y), (est, y + 0.3)]],
            line="none",
            color=style.pooled_color,
            fill_opacity=1.0,
        )
    annotate_column(plot, annotations, data_end)
    return target
