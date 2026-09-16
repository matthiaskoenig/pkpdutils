"""Style of the figures."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotStyle:
    """Colors, markers and sizes shared by the figures of the package.

    The default colors follow the palette of Okabe and Ito, which stays
    distinguishable for the common forms of color blindness: near-black data,
    a sky blue area, an orange extrapolation, a vermilion regression and a
    bluish green peak.

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
        marker_max_points: most points a curve may have and still be drawn
            with a marker per point; a longer curve (a simulation, a dense
            sampling) is drawn as a line alone, since its markers would merge
            into a band and hide the shape of the curve
        cmap: colormap of the samples of a batch
        limit_color: color of acceptance limits and interaction thresholds
        pooled_color: color of pooled effects
        summary_color: color of means and intervals drawn over individual points
        dose_color: color of the dose time markers of a dosing protocol
        peak_color: color of the marker and the guide lines of the peak
            (`cmax`, `tmax`) in the NCA panel
        band_alpha: transparency of a confidence band
        annotation_fontsize: font size of the annotations and the parameter
            box of the NCA panel
    """

    data_color: str = "#1f1f1f"
    data_marker: str = "o"
    fit_color: str = "#d55e00"
    auc_color: str = "#56b4e9"
    extrapolation_color: str = "#e69f00"
    terminal_marker: str = "s"
    alpha: float = 0.2
    linewidth: float = 1.5
    markersize: float = 5.0
    marker_max_points: int = 60
    cmap: str = "viridis"
    limit_color: str = "tab:red"
    pooled_color: str = "tab:orange"
    summary_color: str = "tab:blue"
    dose_color: str = "gray"
    peak_color: str = "#009e73"
    band_alpha: float = 0.15
    annotation_fontsize: str = "x-small"


#: the default style
DEFAULT_STYLE = PlotStyle()
