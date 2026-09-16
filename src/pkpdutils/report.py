r"""The study report: the tables and the figures of an analysis in one document.

An analysis of `pkpdutils` ends in data frames and figures; a study report is
those pieces in the order ICH M13A (2024, 2.2.2) asks for, in a file which can
be sent around. `Report` collects the pieces - a paragraph, a table, a figure -
and writes them as a self-contained HTML page (the figures embedded as base64
PNG, so the file travels alone) or as markdown next to its figure files.
`study_report` assembles the package of a bioequivalence or single dose study
from a batch and its result: the sentence describing the methods, the summary
statistics M13A names, the acceptability of the extrapolation, the parameters
of every subject, the mean curves and the individual panels.

Nothing here decides anything: every number comes from the analysis which was
run, and every table is the one the matching function of `pkpdutils` returns,
so a report can be extended with any further frame or figure of the package
before it is written.
"""

import base64
import html
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from pkpdutils.nca.options import NCAOptions
from pkpdutils.nca.report import M13A_STATISTICS, acceptability_table, methods_line
from pkpdutils.nca.result import NCAResult
from pkpdutils.result import format_number, summary_table
from pkpdutils.timecourse import Timecourses

if TYPE_CHECKING:  # pragma: no cover - the annotation of `add_figure`
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

#: the kinds of section a report holds
SectionKind = Literal["text", "table", "figure"]

#: the style sheet of the HTML page, kept small and printable
STYLE: str = """
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
       margin: 2rem auto; max-width: 62rem; padding: 0 1rem; color: #1a1a1a; }
h1 { font-size: 1.6rem; margin-bottom: 0.2rem; }
h2 { font-size: 1.2rem; margin-top: 2rem; border-bottom: 1px solid #ddd;
     padding-bottom: 0.2rem; }
p.subtitle { color: #666; margin-top: 0; }
p.caption { color: #444; font-size: 0.9rem; font-style: italic; }
table { border-collapse: collapse; font-size: 0.85rem; margin: 0.5rem 0; }
th, td { border: 1px solid #ddd; padding: 0.25rem 0.6rem; text-align: right; }
th { background: #f4f4f4; text-align: left; }
td:first-child, th:first-child { text-align: left; }
figure { margin: 1rem 0; }
img { max-width: 100%; height: auto; }
"""


@dataclass(frozen=True)
class Section:
    """One section of a report.

    Attributes:
        kind: `"text"`, `"table"` or `"figure"`
        heading: the heading above the section, empty for none
        text: the paragraph of a text section
        frame: the table of a table section
        caption: the caption below a table or a figure
        image: the rendered PNG of a figure section
        digits: significant digits of the numbers of a table
        level: the level of the heading
    """

    kind: SectionKind
    heading: str = ""
    text: str = ""
    frame: pd.DataFrame | None = None
    caption: str = ""
    image: bytes = b""
    digits: int = 3
    level: int = 2


def format_cell(value: Any, digits: int = 3) -> str:
    """One cell of a report table, formatted as the publication tables are.

    A float is rounded to `digits` significant digits with
    `pkpdutils.result.format_number`, which leaves a missing value empty; a
    boolean and an integer are written as they are and anything else as its
    string.

    Args:
        value: the cell value.
        digits: significant digits of a float.

    Returns:
        The cell as a string.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return format_number(float(value), digits)
    return str(value)


def _cells(frame: pd.DataFrame, digits: int) -> list[list[str]]:
    """Every cell of a frame as a formatted string, row by row.

    Args:
        frame: the table.
        digits: significant digits of the numbers.

    Returns:
        One list of strings per row.
    """
    return [
        [format_cell(value, digits) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]


@dataclass
class Report:
    """The sections of a report, written as HTML or as markdown.

    A report is built by adding sections in the order they are read; every
    `add_*` returns the report itself, so the calls chain. The figures are
    rendered to PNG when they are added, so the report does not keep the
    matplotlib figures alive and writing it twice gives the same bytes.

    Attributes:
        title: the title of the document
        subtitle: the line below the title, empty for none
        digits: significant digits of the numbers of a table which does not
            ask for its own
        sections: the sections, in order
    """

    title: str = "Study report"
    subtitle: str = ""
    digits: int = 3
    sections: list[Section] = field(default_factory=list)

    def add_text(
        self, text: str, *, heading: str | None = None, level: int = 2
    ) -> "Report":
        """Add a paragraph, optionally under a heading.

        Args:
            text: the paragraph; several paragraphs are separated by an empty
                line, as in markdown.

        Keyword Args:
            heading: the heading above it, none by default.
            level: the level of the heading, 2 for a section of the document.

        Returns:
            The report, so that the calls chain.

        Raises:
            ValueError: if `level` is not between 1 and 6.
        """
        if not 1 <= level <= 6:
            raise ValueError(f"'level' must lie between 1 and 6, got {level}")
        self.sections.append(
            Section(
                kind="text",
                heading="" if heading is None else heading,
                text=text,
                level=level,
            )
        )
        return self

    def add_table(
        self,
        frame: pd.DataFrame,
        caption: str = "",
        *,
        heading: str | None = None,
        digits: int | None = None,
    ) -> "Report":
        """Add a table with a caption.

        Args:
            frame: the table, as any function of the package returns it; its
                index is not written, so a frame whose index carries
                information is reset by the caller.
            caption: the caption below the table.

        Keyword Args:
            heading: the heading above it, none by default.
            digits: significant digits of the numbers, the report's own by
                default; a frame of formatted strings is unaffected.

        Returns:
            The report, so that the calls chain.
        """
        self.sections.append(
            Section(
                kind="table",
                heading="" if heading is None else heading,
                frame=frame,
                caption=caption,
                digits=self.digits if digits is None else digits,
            )
        )
        return self

    def add_figure(
        self,
        fig: "Figure",
        caption: str = "",
        dpi: int = 150,
        *,
        heading: str | None = None,
    ) -> "Report":
        """Add a figure with a caption, rendered to PNG right away.

        Args:
            fig: the figure, as every plotting function of the package
                returns it; it is not closed, the caller owns it.
            caption: the caption below the figure.
            dpi: resolution of the rendered image.

        Keyword Args:
            heading: the heading above it, none by default.

        Returns:
            The report, so that the calls chain.
        """
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        self.sections.append(
            Section(
                kind="figure",
                heading="" if heading is None else heading,
                caption=caption,
                image=buffer.getvalue(),
            )
        )
        return self

    def write_html(self, path: str | Path) -> Path:
        """Write the report as one self-contained HTML file.

        The figures are embedded as base64 PNG and the style sheet is part of
        the document, so the file carries everything it needs and can be
        mailed or archived on its own.

        Args:
            path: the file to write; its directory is created.

        Returns:
            The file which was written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        parts = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{html.escape(self.title)}</title>",
            f"<style>{STYLE}</style>",
            "</head>",
            "<body>",
            f"<h1>{html.escape(self.title)}</h1>",
        ]
        if self.subtitle:
            parts.append(f'<p class="subtitle">{html.escape(self.subtitle)}</p>')
        for section in self.sections:
            parts.extend(self._html_section(section))
        parts.extend(["</body>", "</html>", ""])
        target.write_text("\n".join(parts), encoding="utf-8")
        logger.info("report written: %s", target)
        return target

    def _html_section(self, section: Section) -> list[str]:
        """The lines of one section of the HTML document.

        Args:
            section: the section.

        Returns:
            The lines.
        """
        lines: list[str] = []
        if section.heading:
            level = min(max(section.level, 1), 6)
            lines.append(f"<h{level}>{html.escape(section.heading)}</h{level}>")
        if section.kind == "text":
            lines.extend(
                f"<p>{html.escape(paragraph)}</p>"
                for paragraph in section.text.split("\n\n")
                if paragraph.strip()
            )
        elif section.kind == "table":
            assert section.frame is not None
            lines.append("<table>")
            header = "".join(
                f"<th>{html.escape(str(column))}</th>" for column in section.frame
            )
            lines.append(f"<thead><tr>{header}</tr></thead><tbody>")
            for row in _cells(section.frame, section.digits):
                cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
                lines.append(f"<tr>{cells}</tr>")
            lines.append("</tbody></table>")
            if section.caption:
                lines.append(f'<p class="caption">{html.escape(section.caption)}</p>')
        else:
            encoded = base64.b64encode(section.image).decode("ascii")
            lines.append("<figure>")
            lines.append(
                f'<img src="data:image/png;base64,{encoded}" '
                f'alt="{html.escape(section.caption)}">'
            )
            if section.caption:
                lines.append(f"<figcaption>{html.escape(section.caption)}</figcaption>")
            lines.append("</figure>")
        return lines

    def write_markdown(self, path: str | Path) -> Path:
        """Write the report as markdown with its figures as files next to it.

        A figure is written as `<stem>_<number>.png` in the directory of the
        markdown file and referenced by that name, so the document and its
        images move together.

        Args:
            path: the markdown file to write; its directory is created.

        Returns:
            The markdown file which was written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"# {self.title}", ""]
        if self.subtitle:
            lines.extend([self.subtitle, ""])
        number = 0
        for section in self.sections:
            if section.heading:
                level = min(max(section.level, 1), 6)
                lines.extend([f"{'#' * level} {section.heading}", ""])
            if section.kind == "text":
                lines.extend([section.text, ""])
            elif section.kind == "table":
                assert section.frame is not None
                columns = [str(column) for column in section.frame]
                lines.append("| " + " | ".join(columns) + " |")
                lines.append("| " + " | ".join("---" for _ in columns) + " |")
                for row in _cells(section.frame, section.digits):
                    lines.append("| " + " | ".join(row) + " |")
                lines.append("")
                if section.caption:
                    lines.extend([f"*{section.caption}*", ""])
            else:
                number += 1
                name = f"{target.stem}_{number}.png"
                (target.parent / name).write_bytes(section.image)
                lines.extend([f"![{section.caption}]({name})", ""])
                if section.caption:
                    lines.extend([f"*{section.caption}*", ""])
        target.write_text("\n".join(lines), encoding="utf-8")
        logger.info("report written: %s", target)
        return target


def study_report(
    batch: Timecourses,
    result: NCAResult,
    *,
    dim: str,
    by: str | None = None,
    options: NCAOptions | None = None,
    title: str = "Non-compartmental analysis report",
    subtitle: str = "",
) -> Report:
    """The report package of a study: the methods, the tables and the figures.

    The sections are the ones ICH M13A (2024, 2.2.2) names, in its order: the
    sentence describing the non-compartmental methods (`methods_line`), the
    summary statistics of every parameter with the statistics M13A lists
    (`M13A_STATISTICS`), the acceptability of the extrapolation of every
    subject with its verdict (`acceptability_table`, left out when the result
    carries no `auc_inf_obs`), the parameters of every subject
    (`NCAResult.to_dataframe`), the mean curves per group
    (`pkpdutils.plot.plot_mean_timecourse`) and one panel per subject
    (`pkpdutils.plot.plot_nca_grid`). The report is returned, not written, so
    that further sections can be added before `write_html` or
    `write_markdown`.

    Args:
        batch: the timecourses the analysis ran on.
        result: the analysis of that batch.

    Keyword Args:
        dim: the sample dimension of the individuals.
        by: coordinate along `dim` grouping the subjects (the treatment, the
            dose group), one group by default.
        options: the options the analysis was run with, for the methods
            sentence; the defaults are described when none are given.
        title: the title of the document.
        subtitle: the line below the title.

    Returns:
        The report.

    Raises:
        ValueError: if `dim` is not a sample dimension of the result.
    """
    # the figures are the only part which needs matplotlib, and a report of
    # tables alone should not pay for the import
    from matplotlib import pyplot as plt

    from pkpdutils.plot import plot_mean_timecourse, plot_nca_grid

    if dim not in result.sample_dims:
        raise ValueError(f"'{dim}' is not a sample dimension {result.sample_dims}")
    resolved = NCAOptions() if options is None else options
    report = Report(title=title, subtitle=subtitle)
    report.add_text(methods_line(resolved, result), heading="Methods")
    report.add_table(
        summary_table(result, dim, by=by, stats=M13A_STATISTICS),
        "Summary statistics of the pharmacokinetic parameters (ICH M13A).",
        heading="Pharmacokinetic parameters",
    )
    if "auc_inf_obs" in result.ds.data_vars and "auc_last" in result.ds.data_vars:
        table, accepted = acceptability_table(result, dim)
        verdict = (
            "The extrapolation is acceptable."
            if accepted
            else "The ratio is below the threshold in more than the accepted share "
            "of the observations; the study is questioned by this criterion."
        )
        report.add_table(
            table,
            f"AUC(0-t) / AUC(0-inf) of every subject. {verdict}",
            heading="Acceptability of the extrapolation",
        )
    else:
        logger.info(
            "the result carries no 'auc_inf_obs', the acceptability table is skipped"
        )
    report.add_table(
        result.to_dataframe(),
        "The parameters of every subject.",
        heading="Individual parameters",
    )
    mean_figure = plot_mean_timecourse(batch, by=by)
    report.add_figure(
        mean_figure,
        "Mean concentration-time curves with their standard deviation, linear "
        "and semi-logarithmic.",
        heading="Figures",
    )
    plt.close(mean_figure)
    grid = plot_nca_grid(batch, result)
    report.add_figure(grid, "The analysis of every subject.")
    plt.close(grid)
    return report
