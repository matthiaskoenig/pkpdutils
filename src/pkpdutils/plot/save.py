"""Saving a figure in the formats a manuscript needs."""

from collections.abc import Sequence
from pathlib import Path

from sbmlsim_core.chart import Chart

#: the formats `save_figure` writes by default: a vector image, whose texts a
#: journal can edit
DEFAULT_FORMATS: tuple[str, ...] = ("svg",)

#: the formats the function knows: a vector image, a raster image for a slide
#: and the spec of the chart (`figure/1`), which the renderer draws again
FORMATS: tuple[str, ...] = ("svg", "png", "json")


def save_figure(
    chart: Chart,
    path: str | Path,
    *,
    formats: Sequence[str] | None = None,
    dpi: float = 300,
) -> list[Path]:
    """Save a figure as SVG (or the formats given) next to each other.

    `path` is the stem of the files (`figures/fig1` writes `figures/fig1.svg`);
    a `path` with one of the known extensions writes that one file alone
    unless `formats` is given. A PNG is written at `dpi`, the text of an SVG
    stays text, so that a journal can edit the labels, and the JSON is the
    spec of the chart. Every file is written atomically and its directory is
    created.

    Args:
        chart: the figure, a chart of a plot function of the package.
        path: the stem of the files, or one file with its extension.

    Keyword Args:
        formats: the formats to write (`svg`, `png`, `json`),
            `DEFAULT_FORMATS` by default.
        dpi: resolution of a PNG.

    Returns:
        The files written, in the order of `formats`.

    Raises:
        ValueError: for a format which is not known, or an empty `formats`.
    """
    target = Path(path)
    suffix = target.suffix.lower().lstrip(".")
    stem = target.with_suffix("") if suffix in FORMATS else target
    if formats is None:
        formats = (suffix,) if suffix in FORMATS else DEFAULT_FORMATS
    if not formats:
        raise ValueError("'formats' must name at least one format")
    unknown = [f for f in formats if f.lower() not in FORMATS]
    if unknown:
        raise ValueError(f"unknown formats {unknown}, known are {list(FORMATS)}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    return [
        chart.save(stem.with_name(f"{stem.name}.{name.lower()}"), dpi=dpi)
        for name in formats
    ]
