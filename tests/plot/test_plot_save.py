import json

import pytest
from sbmlsim_core.chart import Chart, ChartAxis

from pkpdutils.plot import save_figure


def figure() -> Chart:
    chart = Chart()
    plot = chart.plot(x=ChartAxis(label="time [h]"))
    plot.curve([0, 1], [0, 1], label="line")
    return chart


def test_save_figure_writes_svg_by_default(tmp_path) -> None:
    files = save_figure(figure(), tmp_path / "figures" / "fig1")
    assert [f.name for f in files] == ["fig1.svg"]
    svg = files[0].read_text(encoding="utf-8")
    assert "<svg" in svg and "time [h]" in svg  # text stays text


def test_save_figure_one_file_by_extension_or_the_formats_given(tmp_path) -> None:
    chart = figure()
    single = save_figure(chart, tmp_path / "one.png", dpi=72)
    assert [f.name for f in single] == ["one.png"]
    assert single[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    chosen = save_figure(chart, tmp_path / "two.png", formats=("svg", "json"))
    assert [f.name for f in chosen] == ["two.svg", "two.json"]
    assert json.loads(chosen[1].read_text())["contract"] == "figure/1"
    with pytest.raises(ValueError, match="unknown formats"):
        save_figure(chart, tmp_path / "x", formats=("tif",))
    with pytest.raises(ValueError, match="at least one"):
        save_figure(chart, tmp_path / "x", formats=())
