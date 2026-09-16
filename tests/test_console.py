import io

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from pkpdutils import NCAOptions, Timecourses, nca
from pkpdutils.console import print_table, rich_table


def render(renderable) -> str:
    buffer = io.StringIO()
    Console(file=buffer, width=120, force_terminal=False).print(renderable)
    return buffer.getvalue()


def make_result():
    time = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    values = np.array(
        [[2.0, 3.5, 3.0, 2.0, 1.0, 0.5, 0.125], [4.0, 7.0, 6.0, 4.0, 2.0, 1.0, 0.25]]
    )
    batch = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        coords={"individual": ["a", "b"], "dose": ("individual", [50.0, 100.0])},
        dose={"amount": [50.0, 100.0], "unit": "mg"},
        route="oral",
    )
    return nca(batch, options=NCAOptions())


def test_rich_table_formats_numbers_strings_and_booleans() -> None:
    frame = pd.DataFrame(
        {
            "parameter": ["cmax", "tmax"],
            "value": [3.14159, np.nan],
            "count": [3, 4],
            "flagged": [True, False],
        }
    )
    text = render(rich_table(frame, title="Parameters", digits=3))
    assert "Parameters" in text
    assert "3.14" in text and "3.14159" not in text
    assert "yes" in text and "no" in text
    assert "nan" not in text.lower()
    table = rich_table(frame)
    assert [column.header for column in table.columns] == list(frame.columns)
    assert [column.justify for column in table.columns] == [
        "left",
        "right",
        "right",
        "left",
    ]


def test_rich_table_index_and_caption() -> None:
    frame = pd.DataFrame({"x": [1.0]}, index=pd.Index(["row"], name="label"))
    table = rich_table(frame, index=True, caption="below")
    assert [column.header for column in table.columns] == ["label", "x"]
    assert "below" in render(table)
    assert [column.header for column in rich_table(frame).columns] == ["x"]


def test_print_table_prints_on_the_given_console() -> None:
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)
    frame = pd.DataFrame({"parameter": ["auc"], "geomean": ["12.3"]})
    print_table(frame, title="Table 1", console=console)
    text = buffer.getvalue()
    assert "Table 1" in text and "auc" in text and "12.3" in text


def test_result_renders_as_a_rich_table() -> None:
    result = make_result()
    # two samples: one row per variable, one column per sample
    table = result.__rich__()
    headers = [column.header for column in table.columns]
    assert table.title == "NCAResult: 2 samples (individual)"
    assert headers == ["variable", "unit", "a", "b"]
    text = render(result)
    assert "cmax" in text and "mg/l" in text and "h⋅mg/l" in text
    assert text.rstrip().splitlines()[-1].strip().startswith("flags")
    # one row per sample on request, the units in the header
    wide = result.rich_table(transpose=False, parameters=["cmax", "auc_inf_obs"])
    assert [column.header for column in wide.columns] == [
        "individual",
        "cmax [mg/l]",
        "auc_inf_obs [h⋅mg/l]",
        "flags",
    ]
    assert result.rich_table(title="T").title == "T"
    with pytest.raises(ValueError, match="no variables"):
        result.rich_table(parameters=["nonsense"])
    # the frame of to_dataframe keeps its names and its precision
    assert "cmax" in result.to_dataframe().columns
    # a summary table prints through the same renderer
    summary = render(
        rich_table(
            result.summary_table("individual", parameters=["cmax", "thalf"]),
            title="Parameters",
        )
    )
    assert "cmax" in summary and "geomean" in summary


def test_many_samples_render_one_row_per_sample() -> None:
    time = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0])
    base = np.array([2.0, 3.5, 3.0, 2.0, 1.0, 0.5, 0.125])
    values = np.outer(np.arange(1, 11), base)
    batch = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        coords={"individual": [f"s{i}" for i in range(10)]},
        dose={"amount": 100.0, "unit": "mg"},
        route="oral",
    )
    table = nca(batch).__rich__()
    headers = [column.header for column in table.columns]
    assert headers[0] == "individual" and headers[-1] == "flags"
    assert "cmax [mg/l]" in headers and len(table.rows) == 10
    # the headline parameters of the NCA, not all of them
    assert "thalf [h]" in headers and "cl_f [l/h]" in headers
    assert len(headers) < 20 and "lambda_z_r2" not in headers
