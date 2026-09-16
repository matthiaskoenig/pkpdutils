"""Shared rich console.

The console is used for the output of scripts and examples; library code logs
instead of printing, see `pkpdutils.log`.

```python
from pkpdutils.console import console, print_table

console.rule("Section", style="white")
console.print(result)  # a result renders as the table of its samples
print_table(result.summary_table("individual", by="dose"), title="Parameters")
```

`rich_table` turns any data frame of the package (`summary_table`,
`to_dataframe`, `flag_table`, `intervals`, the ratio and interaction tables)
into a rich table, `print_table` prints it, and a `ParameterResult` renders
itself as one through the rich protocol (`__rich__`), so `console.print(result)`
shows the parameters of every sample with their units in the header.

Importing this module has no side effects on the interpreter. To get rich
representations in an interactive session, install them explicitly with
`rich.pretty.install()`.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from rich.theme import Theme

from pkpdutils.result import format_number

custom_theme = Theme(
    {
        "success": "green",
        "info": "blue",
        "warning": "orange3",
        "error": "red",
    }
)

console = Console(theme=custom_theme)


def rich_table(
    frame: pd.DataFrame,
    *,
    title: str | None = None,
    digits: int = 3,
    index: bool = False,
    caption: str | None = None,
) -> Table:
    """A rich table of a data frame, the console rendering of the tables of the package.

    A string cell is written as it is (the cells of `summary_table` are
    formatted strings already), a number is rounded to `digits` significant
    digits with `pkpdutils.result.format_number`, a missing value is an empty
    cell and a boolean is written as `yes`/`no`. Numeric columns are aligned to
    the right, text columns to the left; the header carries the column names
    of the frame.

    Args:
        frame: the table, e.g. of `summary_table`, `to_dataframe`,
            `flag_table` or `intervals`.
        title: the title above the table.
        digits: significant digits of the numeric cells.
        index: whether the index of the frame is the first column.
        caption: a caption below the table.

    Returns:
        The table, to print with `console.print(table)` or to embed in
        another rich renderable.
    """
    table = Table(
        title=title,
        caption=caption,
        show_header=True,
        header_style="bold",
        box=box.SIMPLE_HEAVY,
        title_justify="left",
        caption_justify="left",
    )
    shown = frame.reset_index() if index else frame
    for name in shown.columns:
        numeric = pd.api.types.is_numeric_dtype(
            shown[name]
        ) and not pd.api.types.is_bool_dtype(shown[name])
        table.add_column(str(name), justify="right" if numeric else "left")
    for row in shown.itertuples(index=False):
        table.add_row(*(_cell(value, digits) for value in row))
    return table


def print_table(
    frame: pd.DataFrame,
    *,
    title: str | None = None,
    digits: int = 3,
    index: bool = False,
    caption: str | None = None,
    console: Console | None = None,
) -> None:
    """Print a data frame as a rich table on the console.

    The table of a script or an example: `print_table(result.summary_table(
    "individual", by="dose"), title="Pharmacokinetic parameters")`. The frame
    itself is unchanged; a manuscript takes it with `to_csv`, `to_markdown` or
    `to_latex`.

    Args:
        frame: the table.
        title: the title above the table.
        digits: significant digits of the numeric cells.
        index: whether the index of the frame is the first column.
        caption: a caption below the table.
        console: the console to print on, the shared one by default.
    """
    (console or globals()["console"]).print(
        rich_table(frame, title=title, digits=digits, index=index, caption=caption)
    )


def _cell(value: Any, digits: int) -> str:
    """The text of one cell.

    Args:
        value: the value of the frame.
        digits: significant digits of a number.

    Returns:
        The text.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, bool | np.bool_):
        return "yes" if value else "no"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return format_number(float(value), digits)
    return str(value)
