"""Figures of timecourses and analyses (matplotlib).

Every function returns the `matplotlib.figure.Figure` it drew on and never
shows it; pass `ax` to draw into an existing axes.
"""

from pkpdutils.plot.nca import plot_nca, plot_nca_grid
from pkpdutils.plot.style import DEFAULT_STYLE, PlotStyle
from pkpdutils.plot.timecourse import plot_timecourse

__all__ = ["DEFAULT_STYLE", "PlotStyle", "plot_nca", "plot_nca_grid", "plot_timecourse"]
