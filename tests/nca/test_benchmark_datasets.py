"""The analysis reproduces Phoenix WinNonlin, PKNCA and NonCompart on the benchmark datasets.

Every scenario of `docs/data/benchmarks/scenarios.csv` is analysed and every
parameter of every subject is compared against the result table of each tool,
read with the readers of `pkpdutils.crosswalk`. The published WinNonlin tables
carry 8 to 15 significant digits, the PKNCA and NonCompart tables the full
precision of R. `docs/benchmark_datasets.md` shows the same comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pkpdutils import (
    AUCMethod,
    NCAOptions,
    TerminalMethod,
    TerminalPhase,
    Timecourses,
    nca,
)
from pkpdutils.crosswalk import (
    NONCOMPART_UNMAPPED,
    WINNONLIN_UNMAPPED,
    read_noncompart,
    read_pknca_results,
    read_winnonlin,
)

DATA_DIR = Path(__file__).parent.parent.parent / "docs" / "data" / "benchmarks"
SCENARIOS = pd.read_csv(DATA_DIR / "scenarios.csv", index_col="scenario")
TOOLS = ("winnonlin", "pknca", "noncompart")

#: the relative tolerance of the comparison, above the rounding of the
#: published WinNonlin tables (at most 6e-9)
TOLERANCE = 1e-8


def analyse(scenario: str) -> pd.DataFrame:
    s = SCENARIOS.loc[scenario]
    data = pd.read_csv(DATA_DIR / s["dataset"])
    data["dose"] = s["dose"]
    data["duration"] = s["duration"]
    batch = Timecourses.from_dataframe(
        data,
        sample=[s["subject"]],
        time=s["time"],
        value=s["conc"],
        time_unit=s["time_unit"],
        unit=s["unit"],
        dose_amount="dose",
        dose_unit=s["dose_unit"],
        dose_duration="duration" if s["route"] == "iv_infusion" else None,
        route=s["route"],
    )
    options = NCAOptions(
        auc_method=AUCMethod(s["auc_method"]),
        terminal=TerminalPhase(
            method=TerminalMethod.BEST_FIT,
            min_points=3,
            exclude_cmax=bool(s["exclude_cmax"]),
        ),
    )
    return nca(batch, options=options).to_dataframe().set_index(s["subject"])


def reference(scenario: str, tool: str) -> pd.DataFrame:
    s = SCENARIOS.loc[scenario]
    if tool == "winnonlin":
        return read_winnonlin(DATA_DIR / "winnonlin" / s["winnonlin"])
    if tool == "pknca":
        table = read_pknca_results(DATA_DIR / "pknca" / f"{scenario}.csv")
        return table.droplevel(["start", "end"])
    return read_noncompart(DATA_DIR / "noncompart" / f"{scenario}.csv")


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("scenario", SCENARIOS.index)
def test_benchmark(scenario: str, tool: str) -> None:
    ours = analyse(scenario)
    expected = reference(scenario, tool).dropna(axis="columns", how="all")
    assert sorted(expected.index) == sorted(ours.index)
    # the only reported parameter pkpdutils leaves out is NonCompart's lag time
    # of an infusion, which is 0 there and not defined for an intravenous dose
    missing = set(expected.columns) - set(ours.columns)
    assert missing <= {"tlag"}, missing
    assert len(expected.columns) >= 30
    for name in sorted(set(expected.columns) & set(ours.columns)):
        e = expected[name].to_numpy(dtype=np.float64)
        v = ours.loc[expected.index, name].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(v, e, rtol=TOLERANCE, atol=1e-12, err_msg=name)


def test_every_reference_column_is_named() -> None:
    # every column of the three tools has a pkpdutils name, except the
    # correlation of the regression and NonCompart's intercept
    for scenario in SCENARIOS.index:
        s = SCENARIOS.loc[scenario]
        winnonlin = pd.read_csv(DATA_DIR / "winnonlin" / s["winnonlin"])
        unnamed = {"Subject"} | (WINNONLIN_UNMAPPED & set(winnonlin.columns))
        assert len(read_winnonlin(winnonlin).columns) == len(winnonlin.columns) - len(
            unnamed
        )
        noncompart = pd.read_csv(DATA_DIR / "noncompart" / f"{scenario}.csv")
        unnamed = {"Subject"} | (NONCOMPART_UNMAPPED & set(noncompart.columns))
        assert len(read_noncompart(noncompart).columns) == len(
            noncompart.columns
        ) - len(unnamed)
        pknca = pd.read_csv(DATA_DIR / "pknca" / f"{scenario}.csv")
        assert len(read_pknca_results(pknca).columns) == pknca["PPTESTCD"].nunique()
