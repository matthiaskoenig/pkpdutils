import matplotlib
import matplotlib.pyplot
import pytest
from PIL import Image

from pkpdutils.plot import save_figure

matplotlib.use("Agg")


def figure():
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot([0, 1], [0, 1], label="line")
    ax.set_xlabel("time [h]")
    return fig


def test_save_figure_writes_png_svg_and_tif(tmp_path) -> None:
    fig = figure()
    files = save_figure(fig, tmp_path / "figures" / "fig1", dpi=72)
    assert [f.name for f in files] == ["fig1.png", "fig1.svg", "fig1.tif"]
    assert all(f.is_file() and f.stat().st_size > 0 for f in files)
    with Image.open(files[0]) as png:
        assert png.format == "PNG"
    with Image.open(files[2]) as tif:
        assert tif.format == "TIFF"
        assert tif.info.get("compression") == "tiff_lzw"
    svg = files[1].read_text(encoding="utf-8")
    assert "<svg" in svg and "time [h]" in svg  # text stays text
    matplotlib.pyplot.close(fig)


def test_save_figure_one_file_by_extension_or_the_formats_given(tmp_path) -> None:
    fig = figure()
    single = save_figure(fig, tmp_path / "one.pdf")
    assert [f.name for f in single] == ["one.pdf"] and single[0].is_file()
    chosen = save_figure(fig, tmp_path / "two.png", formats=("svg", "tiff"))
    assert [f.name for f in chosen] == ["two.svg", "two.tiff"]
    with pytest.raises(ValueError, match="unknown formats"):
        save_figure(fig, tmp_path / "x", formats=("bmp",))
    with pytest.raises(ValueError, match="at least one"):
        save_figure(fig, tmp_path / "x", formats=())
    matplotlib.pyplot.close(fig)
