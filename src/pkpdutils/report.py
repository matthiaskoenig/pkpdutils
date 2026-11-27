r"""The study report: the tables and the figures of an analysis in one document.

An analysis of `pkpdutils` ends in data frames and figures; a study report is
those pieces in the order ICH M13A (2024, 2.2.2) asks for, in a file which can
be sent around. `Report` collects the pieces - a paragraph, a table, a figure -
and writes them as a self-contained HTML page (the figures embedded as base64
SVG, so the file travels alone) or as markdown next to its figure files.
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
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

import numpy as np
import pandas as pd

from pkpdutils.nca.options import NCAOptions, deserialize_nca_options
from pkpdutils.nca.report import M13A_STATISTICS, acceptability_table, methods_line
from pkpdutils.nca.result import NCAResult
from pkpdutils.result import format_number, summary_table
from pkpdutils.timecourse import Timecourses

if TYPE_CHECKING:  # pragma: no cover - the annotation of `add_figure`
    from sbmlsim_core.chart import Chart

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
        image: the rendered SVG of a figure section
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


def _cells(frame: pd.DataFrame, digits: int) -> Iterator[list[str]]:
    """Every cell of a frame as a formatted string, row by row.

    Args:
        frame: the table.
        digits: significant digits of the numbers.

    Yields:
        One list of strings per row.
    """
    for row in frame.itertuples(index=False, name=None):
        yield [format_cell(value, digits) for value in row]


def _markdown_text(text: str) -> str:
    """Escape a plain text field for a Markdown heading, cell or caption.

    Args:
        text: the plain text field.

    Returns:
        Escaped inline Markdown, with line breaks represented by spaces.
    """
    escaped = html.escape(text, quote=False).replace("\r", " ").replace("\n", " ")
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", escaped)


@dataclass
class Report:
    """The sections of a report, written as HTML or as markdown.

    A report is built by adding sections in the order they are read; every
    `add_*` returns the report itself, so the calls chain. The figures are
    rendered to SVG when they are added, so the report does not keep the
    charts and writing it twice gives the same bytes.

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
        chart: "Chart",
        caption: str = "",
        *,
        heading: str | None = None,
    ) -> "Report":
        """Add a figure with a caption, rendered to SVG right away.

        Args:
            chart: the figure, as every plotting function of the package
                returns it.
            caption: the caption below the figure.

        Keyword Args:
            heading: the heading above it, none by default.

        Returns:
            The report, so that the calls chain.
        """
        self.sections.append(
            Section(
                kind="figure",
                heading="" if heading is None else heading,
                caption=caption,
                image=chart.svg().encode("utf-8"),
            )
        )
        return self

    def write_html(self, path: str | Path) -> Path:
        """Write the report as one self-contained HTML file.

        The figures are embedded as base64 SVG and the style sheet is part of
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
        lines = chain(
            parts,
            chain.from_iterable(
                self._html_section(section) for section in self.sections
            ),
            ["</body>", "</html>"],
        )
        with target.open("w", encoding="utf-8") as output:
            output.writelines(line + "\n" for line in lines)
        logger.info("report written: %s", target)
        return target

    def _html_section(self, section: Section) -> Iterator[str]:
        """The lines of one section of the HTML document.

        Args:
            section: the section.

        Yields:
            The lines.
        """
        if section.heading:
            level = min(max(section.level, 1), 6)
            yield f"<h{level}>{html.escape(section.heading)}</h{level}>"
        if section.kind == "text":
            for paragraph in section.text.split("\n\n"):
                if paragraph.strip():
                    yield f"<p>{html.escape(paragraph)}</p>"
        elif section.kind == "table":
            assert section.frame is not None
            yield "<table>"
            header = "".join(
                f"<th>{html.escape(str(column))}</th>" for column in section.frame
            )
            yield f"<thead><tr>{header}</tr></thead><tbody>"
            for row in _cells(section.frame, section.digits):
                cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
                yield f"<tr>{cells}</tr>"
            yield "</tbody></table>"
            if section.caption:
                yield f'<p class="caption">{html.escape(section.caption)}</p>'
        else:
            encoded = base64.b64encode(section.image).decode("ascii")
            yield "<figure>"
            yield (
                f'<img src="data:image/svg+xml;base64,{encoded}" '
                f'alt="{html.escape(section.caption)}">'
            )
            if section.caption:
                yield f"<figcaption>{html.escape(section.caption)}</figcaption>"
            yield "</figure>"

    def write_markdown(self, path: str | Path) -> Path:
        """Write the report as markdown with its figures as files next to it.

        A figure is written as `<stem>_<number>.svg` in the directory of the
        markdown file and referenced by that name, so the document and its
        images move together.

        Args:
            path: the markdown file to write; its directory is created.

        Returns:
            The markdown file which was written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as output:
            output.writelines(line + "\n" for line in self._markdown_lines(target))
        logger.info("report written: %s", target)
        return target

    def _markdown_lines(self, target: Path) -> Iterator[str]:
        """Yield Markdown rows and write adjacent images without buffering tables.

        Args:
            target: the destination, determining adjacent image filenames.

        Yields:
            Lines of the document.
        """
        yield f"# {_markdown_text(self.title)}"
        yield ""
        if self.subtitle:
            yield _markdown_text(self.subtitle)
            yield ""
        number = 0
        for section in self.sections:
            if section.heading:
                level = min(max(section.level, 1), 6)
                yield f"{'#' * level} {_markdown_text(section.heading)}"
                yield ""
            if section.kind == "text":
                # Prose deliberately supports Markdown, but never raw HTML.
                yield html.escape(section.text, quote=False)
                yield ""
            elif section.kind == "table":
                assert section.frame is not None
                columns = [_markdown_text(str(column)) for column in section.frame]
                yield "| " + " | ".join(columns) + " |"
                yield "| " + " | ".join("---" for _ in columns) + " |"
                for row in _cells(section.frame, section.digits):
                    yield "| " + " | ".join(map(_markdown_text, row)) + " |"
                yield ""
                if section.caption:
                    yield f"*{_markdown_text(section.caption)}*"
                    yield ""
            else:
                number += 1
                name = f"{target.stem}_{number}.svg"
                (target.parent / name).write_bytes(section.image)
                yield f"![{_markdown_text(section.caption)}]({quote(name)})"
                yield ""
                if section.caption:
                    yield f"*{_markdown_text(section.caption)}*"
                    yield ""


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
        options: optional confirmation of the recorded analysis options;
            required for older results without recorded options.
        title: the title of the document.
        subtitle: the line below the title.

    Returns:
        The report.

    Raises:
        ValueError: if `dim` is not a sample dimension of the result.
    """
    # the figures are the only part which needs the chart builder, and a
    # report of tables alone should not pay for the import
    from pkpdutils.plot import plot_mean_timecourse, plot_nca_grid

    if dim not in result.sample_dims:
        raise ValueError(f"'{dim}' is not a sample dimension {result.sample_dims}")
    saved_options = result.ds.attrs.get("nca_options")
    if saved_options is not None:
        resolved = deserialize_nca_options(saved_options)
        if options is not None:
            requested = options.model_copy(
                update={
                    "uncertainty": options.resolve_uncertainty(batch.has_uncertainty)
                }
            )
            if requested != resolved:
                raise ValueError(
                    "report options differ from the recorded analysis options"
                )
    elif options is not None:
        resolved = options
    else:
        raise ValueError("the result has no recorded NCA options; supply 'options'")
    report = Report(title=title, subtitle=subtitle)
    report.add_text(methods_line(resolved, result), heading="Methods")
    report.add_table(
        summary_table(result, dim, by=by, stats=M13A_STATISTICS),
        "Summary statistics of the pharmacokinetic parameters (ICH M13A).",
        heading="Pharmacokinetic parameters",
    )
    if "auc_inf_obs" in result.ds.data_vars and "auc_last" in result.ds.data_vars:
        table, accepted = acceptability_table(result, dim)
        if accepted is None:
            verdict = "The extrapolation cannot be evaluated: no finite ratios."
        elif accepted:
            verdict = "The extrapolation is acceptable."
        else:
            verdict = (
                "The ratio is below the threshold in more than the accepted share "
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
    grid = plot_nca_grid(batch, result)
    report.add_figure(grid, "The analysis of every subject.")
    return report
