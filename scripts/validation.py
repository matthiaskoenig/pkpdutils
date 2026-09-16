"""Compare the non-compartmental analysis against PKNCA and Phoenix WinNonlin.

The datasets, the reference values and their sources live in
`tests/data/validation/`. The script runs every case of
`tests/data/validation/reference.json` through `pkpdutils.nca` with the options
the case names, compares every parameter of every subject against the published
number and writes the comparison table to `docs/validation_table.md`. The same
table is spliced into `docs/validation.md` between the markers
`<!-- validation-table:start -->` and `<!-- validation-table:end -->`, so the
page and the table never drift apart. Both files are committed; run the script
after the analysis or the reference values changed:

```bash
uv run python scripts/validation.py
```

The reference file holds three kinds of case: `cases`, one value per subject
and parameter of a whole analysis; `partial_cases`, the area of one time window
of `partial_auc`; and `summary_cases`, the statistics of the whole batch over
its sample dimension.

`tests/nca/test_validation.py` runs the same comparison from the same functions,
so the test and the committed table can never disagree. The script exits
non-zero when a deviation which is not a known difference exceeds its
tolerance.
"""

import copy
import json
import math
import sys
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

from pkpdutils import (
    AUCMethod,
    NCAOptions,
    Route,
    TerminalMethod,
    TerminalPhase,
    Timecourses,
    nca,
)
from pkpdutils.nca.nca import partial_auc
from pkpdutils.nca.result import NCAResult

REPO_DIR: Path = Path(__file__).parent.parent
DATA_DIR: Path = REPO_DIR / "tests" / "data" / "validation"
REFERENCE_PATH: Path = DATA_DIR / "reference.json"
TABLE_PATH: Path = REPO_DIR / "docs" / "validation_table.md"
PAGE_PATH: Path = REPO_DIR / "docs" / "validation.md"

TABLE_START = "<!-- validation-table:start -->"
TABLE_END = "<!-- validation-table:end -->"

#: the sections of the reference file, in the order the table lists them
SECTIONS: tuple[str, ...] = ("cases", "partial_cases", "summary_cases")

#: name of the column of the subject in the comparison frames
SUBJECT = "subject"

#: name of the variable a `partial_cases` case compares
PARTIAL = "auc_partial"


@cache
def _document() -> dict[str, Any]:
    """The parsed reference file, loaded once and shared; never mutated."""
    with REFERENCE_PATH.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def reference() -> dict[str, Any]:
    """A copy of the reference file, safe to change without affecting a caller."""
    return copy.deepcopy(_document())


def case_ids() -> list[str]:
    """The identifiers of every case, in the order of the sections and the file."""
    document = _document()
    return [case_id for section in SECTIONS for case_id in document[section]]


def case(case_id: str) -> dict[str, Any]:
    """One case of the reference file.

    The returned mapping is the cached document itself, not a copy: read it,
    do not change it. `reference()` hands out a copy for that.
    """
    document = _document()
    for section in SECTIONS:
        if case_id in document[section]:
            entry: dict[str, Any] = document[section][case_id]
            return entry
    raise KeyError(f"no case '{case_id}' in {REFERENCE_PATH}")


def batch_of(dataset_id: str, *, dose: float) -> Timecourses:
    dataset = _document()["datasets"][dataset_id]
    frame = pd.read_csv(DATA_DIR / dataset["file"])
    frame["dose_amount"] = dose
    return Timecourses.from_dataframe(
        frame,
        sample=[dataset["subject_column"]],
        time_unit=dataset["time_unit"],
        unit=dataset["unit"],
        time=dataset["time_column"],
        value=dataset["value_column"],
        dose_amount="dose_amount",
        dose_unit=dataset["dose_unit"],
        route=Route(dataset["route"]),
        substance=dataset["substance"],
    )


def options_of(spec: dict[str, Any]) -> NCAOptions:
    terminal = spec["terminal"]
    return NCAOptions(
        auc_method=AUCMethod(spec["auc_method"]),
        terminal=TerminalPhase(
            method=TerminalMethod(terminal["method"]),
            min_points=terminal["min_points"],
            exclude_cmax=terminal["exclude_cmax"],
        ),
    )


@cache
def batch_and_options(case_id: str) -> tuple[Timecourses, NCAOptions]:
    entry = case(case_id)
    dataset = _document()["datasets"][entry["dataset"]]
    return batch_of(entry["dataset"], dose=dataset["dose"]), options_of(
        entry["options"]
    )


@cache
def analyse(case_id: str) -> NCAResult:
    batch, options = batch_and_options(case_id)
    return nca(batch, options=options)


@cache
def run_case(case_id: str) -> pd.DataFrame:
    """One row per subject, one column per variable, the subject label a string."""
    entry = case(case_id)
    dataset = _document()["datasets"][entry["dataset"]]
    if "window" in entry:
        batch, options = batch_and_options(case_id)
        start, end = entry["window"]
        area = partial_auc(batch, float(start), float(end), options=options)
        frame = area.to_dataframe(name=PARTIAL).reset_index()
    else:
        frame = analyse(case_id).to_dataframe()
    frame[SUBJECT] = frame[dataset["subject_column"]].astype(str)
    return frame.set_index(SUBJECT)


@cache
def summarize_case(case_id: str) -> pd.Series:
    """The statistics of a summary case, one entry per `<parameter>_<statistic>`."""
    entry = case(case_id)
    summary = analyse(case_id).summarize(entry["dim"])
    return pd.Series(
        {name: float(summary.ds[name].values) for name in summary.ds.data_vars}
    )


@cache
def units_of(case_id: str) -> dict[str, str]:
    """The unit the analysis reports for every variable of a case."""
    entry = case(case_id)
    if "window" in entry:
        batch, options = batch_and_options(case_id)
        start, end = entry["window"]
        area = partial_auc(batch, float(start), float(end), options=options)
        return {PARTIAL: str(area.attrs["units"])}
    result = analyse(case_id)
    return {str(name): result.units(str(name)) for name in result.ds.data_vars}


def value_of(case_id: str, subject: str, parameter: str) -> float:
    if "dim" in case(case_id):
        return float(summarize_case(case_id)[parameter])
    return float(run_case(case_id).loc[subject, parameter])


def deviation(expected: float, value: float) -> float:
    """Relative deviation from the reference, absolute where the reference is 0.

    A `NaN` on one side only is an infinite deviation: the analysis either
    reports a parameter the comparator does not or misses one it reports.
    """
    if math.isnan(expected) and math.isnan(value):
        return 0.0
    if math.isnan(expected) or math.isnan(value):
        return math.inf
    if expected == 0.0:
        return abs(value)
    return abs(value - expected) / abs(expected)


def known_difference(case_id: str, subject: str, parameter: str) -> str:
    """The reason a difference is known and accepted, `""` when there is none."""
    for difference in case(case_id).get("known_differences", []):
        if difference["parameter"] == parameter and subject in difference["subjects"]:
            return str(difference["reason"])
    return ""


def entries(case_id: str) -> list[tuple[str, str, dict[str, Any]]]:
    """The `(subject, parameter, reference)` triples of a case, in file order."""
    entry = case(case_id)
    if "dim" in entry:
        return [("all", name, ref) for name, ref in entry["references"].items()]
    return [
        (subject, name, ref)
        for subject, parameters in entry["references"].items()
        for name, ref in parameters.items()
    ]


def comparison() -> pd.DataFrame:
    """One row per case and parameter with the largest deviation over the subjects."""
    document = _document()
    rows: list[dict[str, Any]] = []
    for case_id in case_ids():
        entry = case(case_id)
        dataset = document["datasets"][entry["dataset"]]
        worst: dict[str, dict[str, Any]] = {}
        for subject, parameter, expected in entries(case_id):
            row = worst.setdefault(
                parameter,
                {
                    "dataset": entry["dataset"],
                    "n_subjects": dataset["n_subjects"],
                    "comparator": entry["comparator"],
                    "rule": entry["options"]["auc_method"],
                    "parameter": parameter,
                    "n": 0,
                    "n_known": 0,
                    "max_deviation": 0.0,
                    "tolerance": 0.0,
                    "source": entry["source"],
                    "case": case_id,
                },
            )
            row["tolerance"] = max(row["tolerance"], expected["tolerance"])
            if known_difference(case_id, subject, parameter):
                row["n_known"] += 1
                continue
            row["n"] += 1
            row["max_deviation"] = max(
                row["max_deviation"],
                deviation(expected["value"], value_of(case_id, subject, parameter)),
            )
        rows.extend(worst.values())
    return pd.DataFrame(rows)


def format_deviation(row: pd.Series) -> str:
    if row["n"] == 0:
        return "not compared"
    value = float(row["max_deviation"])
    if value == 0.0:
        return "0"
    if math.isinf(value):
        return "infinite"
    return f"{value:.1e}"


def markdown_table(frame: pd.DataFrame) -> str:
    header = (
        "| dataset | comparator | rule | parameter | n subjects | "
        "max relative deviation | tolerance | source |"
    )
    lines = [header, "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, row in frame.iterrows():
        subjects = str(row["n"])
        if row["n_known"]:
            plural = "" if row["n_known"] == 1 else "s"
            subjects = f"{row['n']} ({row['n_known']} known difference{plural})"
        lines.append(
            f"| {row['dataset']} | {row['comparator']} | {row['rule']} | "
            f"`{row['parameter']}` | {subjects} | "
            f"{format_deviation(row)} | "
            f"{row['tolerance']:.0e} | {row['source']} |"
        )
    return "\n".join(lines)


def table_document(table: str) -> str:
    """The whole content of `docs/validation_table.md` for a rendered table."""
    return (
        "# Validation table\n\n"
        "The comparison of `pkpdutils.nca` against the published reference values of "
        "`tests/data/validation/reference.json`, written by `scripts/validation.py`. "
        "The table is part of [Validation](validation.md), which explains the "
        "datasets, the option mapping and the known differences.\n\n"
        f"{table}\n"
    )


def write_table(table: str) -> None:
    TABLE_PATH.write_text(table_document(table), encoding="utf-8")
    page = PAGE_PATH.read_text(encoding="utf-8")
    start = page.index(TABLE_START) + len(TABLE_START)
    end = page.index(TABLE_END)
    PAGE_PATH.write_text(
        page[:start] + f"\n\n{table}\n\n" + page[end:], encoding="utf-8"
    )


def main() -> int:
    frame = comparison()
    table = markdown_table(frame)
    write_table(table)
    print(table)
    failed = frame[(frame["n"] > 0) & (frame["max_deviation"] > frame["tolerance"])]
    print(f"\n{len(frame)} comparisons written to {TABLE_PATH}")
    if len(failed):
        print("\ndeviations beyond the tolerance:", file=sys.stderr)
        print(failed.to_string(index=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
