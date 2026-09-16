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
    case,
    case_ids,
    comparison,
    deviation,
    entries,
    known_difference,
    markdown_table,
    reference,
    table_document,
    units_of,
)
from scripts.validation import value_of as analysed_value

CASE_IDS = case_ids()

#: the parameters every WinNonlin case carries for every subject
WINNONLIN_PARAMETERS: tuple[str, ...] = (
    "cmax",
    "tmax",
    "tlast",
    "clast",
    "auc_last",
    "auc_all",
    "auc_inf_obs",
    "auc_inf_pred",
    "auc_extrap_fraction",
    "aumc_last",
    "aumc_inf",
    "mrt",
    "lambda_z",
    "lambda_z_r2",
    "lambda_z_r2_adj",
    "lambda_z_n_points",
    "lambda_z_t_first",
    "lambda_z_t_last",
    "thalf",
    "cmax_dn",
    "auc_inf_dn",
)
THEOPH_PARAMETERS: tuple[str, ...] = (*WINNONLIN_PARAMETERS, "tlag", "cl_f", "vz_f")
INDOMETH_PARAMETERS: tuple[str, ...] = (
    *WINNONLIN_PARAMETERS,
    "c0",
    "auc_back_extrap_fraction",
    "cl",
    "vz",
    "vss",
)
PKNCA_SUMMARY_PARAMETERS: tuple[str, ...] = (
    "cmax_geomean",
    "cmax_geocv",
    "auc_last_geomean",
    "auc_last_geocv",
    "auc_inf_obs_geomean",
    "auc_inf_obs_geocv",
    "thalf",
    "thalf_sd",
    "tmax_median",
    "tmax_min",
    "tmax_max",
)

#: the parameters the specification names for the validation, as alternatives
#: where the route decides the name of the clearance and the volume
SPEC_PARAMETERS: tuple[tuple[str, ...], ...] = (
    ("auc_last",),
    ("auc_inf_obs",),
    ("cmax",),
    ("tmax",),
    ("thalf",),
    ("lambda_z",),
    ("cl", "cl_f"),
    ("vz", "vz_f"),
    ("mrt",),
)

#: the reference file as it has to be: every case with every subject it covers
#: and the parameters of that subject. A pruned or emptied `reference.json`
#: silently turns the parametrized comparison into no test at all, so the
#: coverage is asserted here rather than read from the file.
EXPECTED: dict[str, dict[str, tuple[str, ...]]] = {
    "theoph-winnonlin-linear": {
        str(subject): THEOPH_PARAMETERS for subject in range(1, 13)
    },
    "theoph-winnonlin-linear-log": {
        str(subject): THEOPH_PARAMETERS for subject in range(1, 13)
    },
    "indometh-winnonlin-linear": {
        str(subject): INDOMETH_PARAMETERS for subject in range(1, 7)
    },
    "indometh-winnonlin-linear-log": {
        str(subject): INDOMETH_PARAMETERS for subject in range(1, 7)
    },
    # the PKNCA vignette prints the per-subject results of two subjects only
    "theoph-pknca": {
        "1": ("cmax", "clast", "lambda_z", "tlast", "tmax"),
        "6": ("auc_last", "cmax", "clast", "lambda_z", "tlast", "tmax"),
    },
    "theoph-pknca-partial": {"1": ("auc_partial",)},
    "theoph-pknca-summary": {"all": PKNCA_SUMMARY_PARAMETERS},
}

#: every reference of the file, the number of comparisons this test is worth
TOTAL_REFERENCES = 911

#: the cases which cover every subject of their dataset
FULL_CASES: tuple[str, ...] = (
    "theoph-winnonlin-linear",
    "theoph-winnonlin-linear-log",
    "indometh-winnonlin-linear",
    "indometh-winnonlin-linear-log",
)


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


def test_reference_is_complete() -> None:
    """The reference file covers every case, subject and parameter it has to.

    Without this the parametrized comparison above degrades silently: an
    emptied or pruned `reference.json` produces an empty parameter set, which
    pytest skips rather than fails.
    """
    assert set(CASE_IDS) == set(EXPECTED), "a case was added to or lost from the file"
    total = 0
    for case_id, subjects in EXPECTED.items():
        references = case(case_id)["references"]
        # a summary case has no subjects, its references are the statistics of
        # the whole batch and `entries` reports them under the subject "all"
        by_subject = (
            {"all": references} if set(subjects) == {"all"} else dict(references)
        )
        assert set(by_subject) == set(subjects), f"{case_id}: the subjects changed"
        for subject, parameters in subjects.items():
            found = by_subject[subject]
            assert set(found) == set(parameters), (
                f"{case_id}, subject {subject}: the parameters changed"
            )
            total += len(parameters)
    assert total == TOTAL_REFERENCES
    compared = [triple for case_id in CASE_IDS for triple in entries(case_id)]
    assert len(compared) == TOTAL_REFERENCES


@pytest.mark.parametrize("case_id", FULL_CASES)
def test_full_cases_cover_every_subject(case_id: str) -> None:
    """A WinNonlin case carries every subject of its dataset and the same parameters."""
    entry = case(case_id)
    dataset = reference()["datasets"][entry["dataset"]]
    references = entry["references"]
    assert len(references) == dataset["n_subjects"]
    parameter_sets = {frozenset(parameters) for parameters in references.values()}
    assert len(parameter_sets) == 1, "the subjects carry different parameters"
    parameters = next(iter(parameter_sets))
    for alternatives in SPEC_PARAMETERS:
        assert parameters & set(alternatives), (
            f"{case_id} carries none of {alternatives}, which the specification names"
        )


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_units_of_the_references(case_id: str) -> None:
    """Every reference carries the unit the analysis reports for the variable."""
    units = units_of(case_id)
    for _, parameter, expected in entries(case_id):
        name = parameter
        for suffix in ("_geomean", "_geocv", "_sd", "_median", "_min", "_max"):
            name = name.removesuffix(suffix)
        if expected["unit"] == "dimensionless" and name != parameter:
            continue
        assert units[name] == expected["unit"], (
            f"{case_id}, {parameter}: the reference says '{expected['unit']}', "
            f"the analysis reports '{units[name]}'"
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


def test_reference_is_not_shared() -> None:
    """`reference()` hands out a copy, so a caller cannot corrupt the cache."""
    first = reference()
    first["cases"].clear()
    assert reference()["cases"], "the cached document was changed by a caller"


def test_table_is_current() -> None:
    """The committed table and page hold the table the script writes today."""
    table = markdown_table(comparison())
    assert TABLE_PATH.read_text(encoding="utf-8") == table_document(table), (
        f"{TABLE_PATH} is stale, run 'uv run python scripts/validation.py'"
    )
    page = PAGE_PATH.read_text(encoding="utf-8")
    start = page.index(TABLE_START) + len(TABLE_START)
    assert page[start : page.index(TABLE_END)] == f"\n\n{table}\n\n", (
        f"{PAGE_PATH} is stale, run 'uv run python scripts/validation.py'"
    )
