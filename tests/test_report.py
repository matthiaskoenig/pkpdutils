"""The study report: the sections, the HTML and the markdown."""

import base64
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pkpdutils import NCAOptions, Report, Route, Timecourses, nca, study_report
from pkpdutils.report import format_cell

N = 6
TIME = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])


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
    figure = plt.figure()
    figure.add_subplot().plot([0, 1], [0, 1])
    report = Report(title="A study", subtitle="six subjects")
    report.add_text("The first paragraph.", heading="Methods").add_table(
        pd.DataFrame({"parameter": ["cmax"], "value": [1.23456], "ok": [True]}),
        "The table.",
        heading="Results",
    ).add_figure(figure, "The figure.")
    plt.close(figure)
    path = report.write_html(tmp_path / "report.html")
    text = path.read_text(encoding="utf-8")

    assert text.startswith("<!DOCTYPE html>")
    assert "<h1>A study</h1>" in text
    assert "<h2>Methods</h2>" in text and "<h2>Results</h2>" in text
    assert "<p>The first paragraph.</p>" in text
    # the numbers are formatted like the cells of `summary_table`
    assert "<td>1.23</td>" in text and "<td>True</td>" in text
    assert text.count("<table>") == 1
    embedded = re.search(r'src="data:image/png;base64,([^"]+)"', text)
    assert embedded is not None
    assert base64.b64decode(embedded.group(1)).startswith(b"\x89PNG")


def test_a_report_writes_markdown_with_its_figures_next_to_it(tmp_path: Path) -> None:
    """The markdown carries the tables and references the PNG files it wrote."""
    figure = plt.figure()
    figure.add_subplot().plot([0, 1], [1, 0])
    report = Report(title="A study")
    report.add_table(pd.DataFrame({"a": [1.0], "b": ["x"]}), "A caption")
    report.add_figure(figure, "A figure")
    plt.close(figure)
    path = report.write_markdown(tmp_path / "study.md")
    text = path.read_text(encoding="utf-8")

    assert text.startswith("# A study")
    assert "| a | b |" in text
    assert "![A figure](study_1.png)" in text
    assert (tmp_path / "study_1.png").read_bytes().startswith(b"\x89PNG")


def test_the_figure_is_rendered_once_and_survives_closing(tmp_path: Path) -> None:
    """`add_figure` keeps the PNG, not the figure, so the caller may close it."""
    figure = plt.figure()
    figure.add_subplot().plot([0, 1], [0, 1])
    report = Report().add_figure(figure, "once", dpi=72)
    plt.close(figure)
    first = report.write_html(tmp_path / "a.html").read_text(encoding="utf-8")
    second = report.write_html(tmp_path / "b.html").read_text(encoding="utf-8")
    assert first == second.replace("b.html", "a.html")


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
    assert text.count("data:image/png;base64,") == 2
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
