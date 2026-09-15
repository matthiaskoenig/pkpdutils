"""Figures of timecourses and analyses (matplotlib).

Every function returns the `matplotlib.figure.Figure` it drew on and never
shows it; pass `ax` to draw into an existing axes.
"""

from pkpdutils.plot.fit import (
    plot_bland_altman,
    plot_dose_proportionality,
    plot_fit,
    plot_goodness_of_fit,
)
from pkpdutils.plot.meta import plot_forest
from pkpdutils.plot.nca import plot_nca, plot_nca_grid
from pkpdutils.plot.parameters import plot_parameters
from pkpdutils.plot.ratio import plot_ratio
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.plot.timecourse import plot_timecourse

__all__ = [
    "DEFAULT_STYLE",
    "PlotStyle",
    "plot_bland_altman",
    "plot_dose_proportionality",
    "plot_fit",
    "plot_forest",
    "plot_goodness_of_fit",
    "plot_nca",
    "plot_nca_grid",
    "plot_parameters",
    "plot_ratio",
    "plot_timecourse",
]
