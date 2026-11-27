"""The study report: the sections, the HTML and the markdown."""

import base64
import re
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sbmlsim_core.chart import Chart

from pkpdutils import NCAOptions, Report, Route, Timecourses, nca, study_report
from pkpdutils.report import format_cell

N = 6
TIME = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])


def line_chart(y: list[float]) -> Chart:
    """A chart of one line."""
    chart = Chart(300, 200)
    chart.plot().curve([0, 1], y)
    return chart


def batch() -> Timecourses:
    """Six oral single dose curves in two arms."""
    rng = np.random.default_rng(11)
    ke, ka = 0.15, 1.2
    values = np.stack(
        [
            scale
            * 100
            * ka
            / (ka - ke)
            * (np.exp(-ke * TIME) - np.exp(-ka * TIME))
            / 30
            for scale in rng.lognormal(0.0, 0.2, N)
        ]
    )
    return Timecourses.from_arrays(
        TIME,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={
            "individual": [f"s{i + 1}" for i in range(N)],
            "arm": ("individual", ["a"] * 3 + ["b"] * 3),
        },
        dose={"amount": np.full(N, 100.0), "unit": "mg"},
        route=Route.ORAL,
        substance="drug",
    )


def test_a_report_writes_its_sections_as_html(tmp_path: Path) -> None:
    """A text, a table and a figure in one self-contained page."""
    figure = line_chart([0, 1])
    report = Report(title="A study", subtitle="six subjects")
    report.add_text("The first paragraph.", heading="Methods").add_table(
        pd.DataFrame({"parameter": ["cmax"], "value": [1.23456], "ok": [True]}),
        "The table.",
        heading="Results",
    ).add_figure(figure, "The figure.")
    path = report.write_html(tmp_path / "report.html")
    text = path.read_text(encoding="utf-8")

    assert text.startswith("<!DOCTYPE html>")
    assert "<h1>A study</h1>" in text
    assert "<h2>Methods</h2>" in text and "<h2>Results</h2>" in text
    assert "<p>The first paragraph.</p>" in text
    # the numbers are formatted like the cells of `summary_table`
    assert "<td>1.23</td>" in text and "<td>True</td>" in text
    assert text.count("<table>") == 1
    embedded = re.search(r'src="data:image/svg\+xml;base64,([^"]+)"', text)
    assert embedded is not None
    assert base64.b64decode(embedded.group(1)).startswith(b"<svg")


def test_a_report_writes_markdown_with_its_figures_next_to_it(tmp_path: Path) -> None:
    """The markdown carries the tables and references the SVG files it wrote."""
    figure = line_chart([1, 0])
    report = Report(title="A study")
    report.add_table(pd.DataFrame({"a": [1.0], "b": ["x"]}), "A caption")
    report.add_figure(figure, "A figure")
    path = report.write_markdown(tmp_path / "study.md")
    text = path.read_text(encoding="utf-8")

    assert text.startswith("# A study")
    assert "| a | b |" in text
    assert "![A figure](study_1.svg)" in text
    assert (tmp_path / "study_1.svg").read_bytes().startswith(b"<svg")


def test_the_figure_is_rendered_once_and_survives_a_change(tmp_path: Path) -> None:
    """`add_figure` keeps the SVG, not the chart, so the caller may change it."""
    figure = line_chart([0, 1])
    report = Report().add_figure(figure, "once", heading="A figure")
    drawn = base64.b64encode(figure.svg().encode("utf-8")).decode("ascii")
    figure.plot(0, 0).curve([0, 1], [1, 0])
    first = report.write_html(tmp_path / "a.html").read_text(encoding="utf-8")
    second = report.write_html(tmp_path / "b.html").read_text(encoding="utf-8")
    assert first == second.replace("b.html", "a.html")
    assert drawn in first and "<h2>A figure</h2>" in first


def test_the_escaping_keeps_the_markup_of_a_cell_out_of_the_page(
    tmp_path: Path,
) -> None:
    """A parameter name with angle brackets does not become an element."""
    report = Report().add_table(pd.DataFrame({"name": ["<b>auc</b>"]}), "")
    text = report.write_html(tmp_path / "r.html").read_text(encoding="utf-8")
    assert "&lt;b&gt;auc&lt;/b&gt;" in text
    assert "<b>auc</b>" not in text


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.23456, "1.23"),
        (float("nan"), ""),
        (None, ""),
        (True, "True"),
        (np.int64(7), "7"),
        ("cmax", "cmax"),
    ],
)
def test_the_cells_are_formatted_like_the_publication_tables(
    value: object, expected: str
) -> None:
    """The same rounding `pkpdutils.result.format_number` gives every table."""
    assert format_cell(value) == expected


def test_a_heading_level_outside_the_document_is_rejected() -> None:
    """A heading is `h1` to `h6`."""
    with pytest.raises(ValueError, match="between 1 and 6"):
        Report().add_text("x", heading="y", level=9)


def test_the_study_report_assembles_the_m13a_package(tmp_path: Path) -> None:
    """The methods sentence, the three tables and the two figures, in order."""
    data = batch()
    result = nca(data)
    report = study_report(data, result, dim="individual", by="arm")
    kinds = [section.kind for section in report.sections]
    assert kinds == ["text", "table", "table", "table", "figure", "figure"]
    assert "trapezoidal" in report.sections[0].text
    summary = report.sections[1].frame
    assert summary is not None
    # the statistics of ICH M13A, in its order
    assert list(summary.columns)[-8:] == [
        "n",
        "geomean",
        "geocv",
        "median",
        "mean",
        "sd",
        "min",
        "max",
    ]
    acceptability = report.sections[2].frame
    assert acceptability is not None
    assert list(acceptability["individual"]) == [f"s{i + 1}" for i in range(N)]
    individuals = report.sections[3].frame
    assert individuals is not None
    assert len(individuals) == N

    path = report.write_html(tmp_path / "study.html")
    text = path.read_text(encoding="utf-8")
    assert text.count("data:image/svg+xml;base64,") == 2
    assert text.count("<table>") == 3
    assert text.rstrip().endswith("</html>")


def test_the_study_report_describes_the_options_it_is_given() -> None:
    """The methods sentence follows the options of the analysis."""
    data = batch()
    options = NCAOptions(auc_method="linear")
    report = study_report(
        data, nca(data, options=options), dim="individual", options=options
    )
    assert "linear trapezoidal method" in report.sections[0].text


def test_the_study_report_rejects_a_dimension_which_is_none() -> None:
    """The sample dimension has to be one of the result."""
    data = batch()
    with pytest.raises(ValueError, match="is not a sample dimension"):
        study_report(data, nca(data), dim="subject")


def test_report_uses_recorded_analysis_options() -> None:
    data = batch()
    result = nca(data, options=NCAOptions(auc_method="linear"))
    report = study_report(data, result, dim="individual")
    assert "linear trapezoidal method" in report.sections[0].text
    with pytest.raises(ValueError, match="differ"):
        study_report(data, result, dim="individual", options=NCAOptions())


def test_report_requires_options_for_legacy_results() -> None:
    data = batch()
    result = nca(data)
    result.ds.attrs.pop("nca_options")
    with pytest.raises(ValueError, match="no recorded NCA options"):
        study_report(data, result, dim="individual")


def test_markdown_escapes_data_and_raw_html(tmp_path: Path) -> None:
    report = Report(title="<script>title</script>")
    report.add_table(pd.DataFrame({"a|b": ["x | y\n<script>bad</script>"]}))
    report.add_text("**Markdown** and <script>bad</script>")
    text = report.write_markdown(tmp_path / "report.md").read_text()
    assert "a\\|b" in text and "x \\| y" in text
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "**Markdown**" in text


def test_report_distinguishes_unevaluable_extrapolation() -> None:
    data = Timecourses.from_arrays(
        [0, 1, 2], [[0, 1, 2]], time_unit="h", unit="mg/l", dims=("individual",)
    )
    report = study_report(data, nca(data), dim="individual")
    caption = report.sections[2].caption
    assert "cannot be evaluated" in caption
    assert "is acceptable" not in caption


@pytest.mark.parametrize("kind", ["html", "markdown"])
def test_large_report_tables_have_bounded_working_memory(
    tmp_path: Path, kind: str
) -> None:
    frame = pd.DataFrame(
        {"id": np.arange(5_000), "text": ["<data> " + "x" * 1024] * 5_000}
    )
    report = Report().add_table(frame)
    path = tmp_path / f"table.{kind}"
    tracemalloc.start()
    try:
        getattr(report, f"write_{kind}")(path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # The output is over 5 MB. Export should retain only a row and the file
    # buffer, rather than formatted copies of every cell and the whole file.
    assert peak < 1024 * 1024
    text = path.read_text(encoding="utf-8")
    assert text.count("&lt;data&gt;") == 5_000
    assert "4999" in text
