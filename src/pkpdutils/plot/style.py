"""Style of the figures."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotStyle:
    """Colors, markers and sizes shared by the figures of the package.

    Attributes:
        data_color: color of the data points and lines
        data_marker: marker of the data points
        fit_color: color of regression lines and fitted curves
        auc_color: fill color of the area to the last measurable point
        extrapolation_color: fill color of the extrapolated area
        terminal_marker: marker of the points of the terminal regression
        alpha: transparency of filled areas
        linewidth: width of lines
        markersize: size of markers
        cmap: colormap of the samples of a batch
        limit_color: color of acceptance limits and interaction thresholds
        pooled_color: color of pooled effects
        summary_color: color of means and intervals drawn over individual points
    """

    data_color: str = "black"
    data_marker: str = "o"
    fit_color: str = "tab:blue"
    auc_color: str = "tab:green"
    extrapolation_color: str = "tab:red"
    terminal_marker: str = "s"
    alpha: float = 0.2
    linewidth: float = 1.5
    markersize: float = 5.0
    cmap: str = "viridis"
    limit_color: str = "tab:red"
    pooled_color: str = "tab:orange"
    summary_color: str = "tab:blue"


#: the default style
DEFAULT_STYLE = PlotStyle()
