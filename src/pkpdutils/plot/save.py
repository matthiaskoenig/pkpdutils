"""Saving a figure in the formats a manuscript needs."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

#: the formats `save_figure` writes by default: a raster image for a
#: manuscript and a slide, a vector image for a journal, a TIFF for a
#: submission system
DEFAULT_FORMATS: tuple[str, ...] = ("png", "svg", "tif")

#: the formats the function knows, with the matplotlib format name of each
FORMATS: dict[str, str] = {
    "png": "png",
    "svg": "svg",
    "pdf": "pdf",
    "tif": "tiff",
    "tiff": "tiff",
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "eps": "eps",
}


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    formats: Sequence[str] | None = None,
    dpi: int = 300,
    transparent: bool = False,
    **kwargs: Any,
) -> list[Path]:
    """Save a figure as PNG, SVG and TIFF (or the formats given) next to each other.

    `path` is the stem of the files (`figures/fig1` writes `figures/fig1.png`,
    `figures/fig1.svg`, `figures/fig1.tif`); a `path` with one of the known
    extensions writes that one file alone unless `formats` is given. A TIFF is
    compressed losslessly (LZW), the raster formats are written at `dpi`, and
    the text of an SVG stays text (not paths), so that a journal can edit the
    labels. The directory of `path` is created.

    Args:
        fig: the figure.
        path: the stem of the files, or one file with its extension.
        formats: the formats to write (`png`, `svg`, `pdf`, `tif`/`tiff`,
            `jpg`/`jpeg`, `eps`), `DEFAULT_FORMATS` by default.
        dpi: resolution of the raster formats.
        transparent: transparent background instead of white.
        **kwargs: passed on to `Figure.savefig` for every format
            (`bbox_inches`, `metadata`, ...).

    Returns:
        The files written, in the order of `formats`.

    Raises:
        ValueError: for a format which is not known, or an empty `formats`.
    """
    target = Path(path)
    suffix = target.suffix.lower().lstrip(".")
    if formats is None:
        formats = (suffix,) if suffix in FORMATS else DEFAULT_FORMATS
        stem = target.with_suffix("") if suffix in FORMATS else target
    else:
        stem = target.with_suffix("") if suffix in FORMATS else target
    if not formats:
        raise ValueError("'formats' must name at least one format")
    unknown = [f for f in formats if f.lower() not in FORMATS]
    if unknown:
        raise ValueError(f"unknown formats {unknown}, known are {sorted(FORMATS)}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with plt.rc_context({"svg.fonttype": "none"}):
        for name in formats:
            extension = name.lower()
            file = stem.with_name(f"{stem.name}.{extension}")
            options: dict[str, Any] = dict(kwargs)
            if FORMATS[extension] == "tiff":
                options.setdefault("pil_kwargs", {"compression": "tiff_lzw"})
            fig.savefig(
                file,
                format=FORMATS[extension],
                dpi=dpi,
                transparent=transparent,
                **options,
            )
            written.append(file)
    return written
