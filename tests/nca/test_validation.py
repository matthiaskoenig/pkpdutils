"""Validate the analysis against the published results of PKNCA and WinNonlin.

The datasets and the reference values live in `tests/data/validation/`, the
comparison itself in `scripts/validation.py`, which the documentation page
`docs/validation.md` and its table are generated from. Importing the script
here keeps the test and the committed table on the same code.
"""

from typing import Any

import pytest

from scripts.validation import (
    PAGE_PATH,
    TABLE_END,
    TABLE_PATH,
    TABLE_START,
    analyse,
    case,
    comparison,
    deviation,
    entries,
    known_difference,
    markdown_table,
    reference,
)
from scripts.validation import value_of as analysed_value

CASE_IDS = list(reference()["cases"]) + list(reference()["summary_cases"])


def _cases() -> list[Any]:
    parameters = []
    for case_id in CASE_IDS:
        for subject, parameter, expected in entries(case_id):
            reason = known_difference(case_id, subject, parameter)
            marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
            parameters.append(
                pytest.param(
                    case_id,
                    subject,
                    parameter,
                    expected,
                    id=f"{case_id}-{subject}-{parameter}",
                    marks=marks,
                )
            )
    return parameters


@pytest.mark.parametrize(("case_id", "subject", "parameter", "expected"), _cases())
def test_reference_value(
    case_id: str, subject: str, parameter: str, expected: dict[str, Any]
) -> None:
    value = analysed_value(case_id, subject, parameter)
    relative = deviation(expected["value"], value)
    assert relative <= expected["tolerance"], (
        f"{case_id}, subject {subject}, {parameter}: {value} against "
        f"{expected['value']} {expected['unit']} of {expected['source']}, "
        f"relative deviation {relative:.3g} above {expected['tolerance']:.0e}"
    )


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_units_of_the_references(case_id: str) -> None:
    """Every reference carries the unit the analysis reports for the variable."""
    result = analyse(case_id)
    for _, parameter, expected in entries(case_id):
        name = parameter
        for suffix in ("_geomean", "_geocv", "_sd", "_median", "_min", "_max"):
            name = name.removesuffix(suffix)
        if expected["unit"] == "dimensionless" and name != parameter:
            continue
        assert result.units(name) == expected["unit"], (
            f"{case_id}, {parameter}: the reference says '{expected['unit']}', "
            f"the analysis reports '{result.units(name)}'"
        )


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_sources_are_named(case_id: str) -> None:
    sources = reference()["sources"]
    entry = case(case_id)
    assert entry["source"] in sources
    for _, _, expected in entries(case_id):
        assert expected["source"] in sources
    for source in sources.values():
        assert source["url"].startswith("https://")


def test_known_differences_are_reachable() -> None:
    """Every known difference names a subject and a parameter which exist."""
    for case_id in CASE_IDS:
        references = case(case_id)["references"]
        for difference in case(case_id).get("known_differences", []):
            assert difference["reason"]
            for subject in difference["subjects"]:
                assert difference["parameter"] in references[subject]


def test_table_is_current() -> None:
    """The committed table and page hold the table the script writes today."""
    table = markdown_table(comparison())
    assert table in TABLE_PATH.read_text(encoding="utf-8"), (
        f"{TABLE_PATH} is stale, run 'uv run python scripts/validation.py'"
    )
    page = PAGE_PATH.read_text(encoding="utf-8")
    start = page.index(TABLE_START) + len(TABLE_START)
    assert page[start : page.index(TABLE_END)].strip() == table, (
        f"{PAGE_PATH} is stale, run 'uv run python scripts/validation.py'"
    )
