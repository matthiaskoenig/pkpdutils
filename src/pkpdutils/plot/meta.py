"""Forest plot of a meta-analysis."""

from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pkpdutils.plot._common import figure_of, plain_log_ticks
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.stats.meta import EffectKind, MetaResult

#: axis label per kind of effect
_LABELS = {
    EffectKind.HEDGES_G: "Hedges' g",
    EffectKind.MEAN_DIFF: "mean difference",
    EffectKind.LOG_RATIO: "log ratio",
}


def plot_forest(
    result: MetaResult,
    *,
    ax: Axes | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    exp: bool | None = None,
) -> Figure:
    """Forest plot: the effect of every study with its interval, the pooled effects as diamonds.

    The marker area of a study is proportional to its random effects
    weight. A `LOG_RATIO` analysis is shown as ratios on a logarithmic axis
    (`exp=None` or `True`); the other kinds stay on their scale.

    Args:
        result: the meta-analysis.
        ax: axes to draw on, a new figure by default.
        style: colors and markers.
        exp: exponentiate the effects; `None` does so for `LOG_RATIO`.

    Returns:
        The figure.
    """
    use_exp = result.kind is EffectKind.LOG_RATIO if exp is None else exp
    transform: Callable[[Any], np.ndarray] = (
        np.exp if use_exp else (lambda v: np.asarray(v, dtype=float))
    )
    # a new figure gets the constrained layout engine from `figure_of`, which
    # keeps the long pooled effect labels ("random effects (tau2 = ...)")
    # from being clipped at the left edge; a caller-supplied `ax` keeps
    # whatever layout its figure already has
    fig, ax = figure_of(ax)
    n = result.n_studies
    weights = result.random.weights
    for i, e in enumerate(result.effects):
        est, low, high = (
            float(transform(v)) for v in (e.estimate, e.ci_low, e.ci_high)
        )
        ax.plot([low, high], [i, i], color=style.data_color, linewidth=style.linewidth)
        ax.scatter(
            [est],
            [i],
            s=40 + 400 * weights[i],
            marker="s",
            color=style.data_color,
            zorder=3,
        )
    for j, pooled in enumerate((result.fixed, result.random)):
        y = n + j
        est, low, high = (
            float(transform(v))
            for v in (pooled.estimate, pooled.ci_low, pooled.ci_high)
        )
        ax.fill(
            [low, est, high, est],
            [y, y - 0.3, y, y + 0.3],
            color=style.pooled_color,
            zorder=3,
        )
    null = 1.0 if use_exp else 0.0
    ax.axvline(null, color="gray", linewidth=1.0)
    ax.set_yticks(
        np.arange(n + 2),
        [
            *result.labels,
            "fixed effect",
            f"random effects (tau2 = {result.random.tau2:.3g})",
        ],
    )
    ax.set_ylim(-0.6, n + 1.6)
    ax.invert_yaxis()
    if use_exp:
        ax.set_xscale("log")
        plain_log_ticks(ax.xaxis)
        ax.set_xlabel("ratio treatment / control")
    else:
        ax.set_xlabel(_LABELS[result.kind])
    het = result.heterogeneity
    ax.set_title(
        f"Q = {het.q:.2f} (df = {het.df}, p = {het.p_value:.2g}), I² = {het.i2:.1f} %",
        fontsize="small",
    )
    return fig
