"""The tables and the methods sentence of a regulatory report (`pkpdutils.nca.report`)."""

import numpy as np
import pytest

from pkpdutils import (
    AUCMethod,
    NCAOptions,
    Route,
    TerminalMethod,
    TerminalPhase,
    Timecourses,
    nca,
    summary_table,
)
from pkpdutils.nca import M13A_STATISTICS, acceptability_table, methods_line
from pkpdutils.result import TABLE_STATISTICS

TIME = np.arange(0.0, 24.1, 1.0)
K = 0.2
OPTIONS = NCAOptions(auc_method=AUCMethod.LOG)


def study(n_low: int, n_total: int = 12) -> Timecourses:
    r"""A batch of bolus curves, `n_low` of them sampled to 6 hours only.

    With the logarithmic trapezoid rule the area of a mono-exponential is
    exact, so `auc_last / auc_inf_obs = 1 - exp(-k tlast)`: a profile which
    ends at 6 hours has a ratio of 0.70 and one which ends at 24 hours of 0.99.
    """
    values = np.tile(10.0 * np.exp(-K * TIME), (n_total, 1))
    values[:n_low, TIME > 6.0] = np.nan
    return Timecourses.from_arrays(
        TIME,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": [f"s{i + 1}" for i in range(n_total)]},
        dose={"amount": np.full(n_total, 100.0), "unit": "mg"},
        route=Route.IV_BOLUS,
        substance="drug",
    )


def test_the_acceptability_table_reports_the_ratio_of_every_subject() -> None:
    result = nca(study(3), options=OPTIONS)
    table, acceptable = acceptability_table(result, "individual")
    assert list(table.columns) == [
        "individual",
        "auc_last",
        "auc_inf_obs",
        "ratio",
        "below",
    ]
    assert len(table) == 12
    assert table["ratio"].to_numpy()[:3] == pytest.approx(1.0 - np.exp(-K * 6.0))
    assert table["ratio"].to_numpy()[3:] == pytest.approx(1.0 - np.exp(-K * 24.0))
    assert table["below"].tolist() == [True] * 3 + [False] * 9
    # 3 of 12 is 25 %, more than the 20 % of ICH M13A
    assert acceptable is False


def test_two_of_twelve_subjects_below_the_threshold_are_acceptable() -> None:
    result = nca(study(2), options=OPTIONS)
    table, acceptable = acceptability_table(result, "individual")
    assert table["below"].sum() == 2
    assert acceptable is True


def test_the_acceptability_table_skips_an_excluded_subject() -> None:
    result = nca(study(3), options=OPTIONS).exclude(individual="s1")
    table, acceptable = acceptability_table(result, "individual")
    assert len(table) == 11
    assert table["below"].sum() == 2
    # 2 of 11 is below the 20 % share
    assert acceptable is True
    full, verdict = acceptability_table(result, "individual", include_excluded=True)
    assert len(full) == 12
    assert verdict is False


def test_the_m13a_statistics_are_the_ones_the_guidance_names() -> None:
    assert M13A_STATISTICS == (
        "n",
        "geomean",
        "geocv",
        "median",
        "mean",
        "sd",
        "min",
        "max",
    )
    assert set(M13A_STATISTICS) <= set(TABLE_STATISTICS)
    result = nca(study(0), options=OPTIONS)
    table = summary_table(
        result, "individual", parameters=["cmax"], stats=M13A_STATISTICS
    )
    assert list(table.columns) == ["parameter", "unit", *M13A_STATISTICS]
    assert table["n"].tolist() == ["12"]


def test_the_methods_line_names_the_rules_and_the_number_of_points() -> None:
    result = nca(study(0), options=OPTIONS)
    counts = result["lambda_z_n_points"].to_numpy()
    line = methods_line(OPTIONS, result)
    assert line == (
        "The areas were computed with the logarithmic trapezoidal method. The "
        "terminal log-linear phase was selected as the points of the largest "
        "adjusted coefficient of determination and estimated by log-linear "
        f"regression using {int(counts.min())} data points."
    )

    other = NCAOptions(
        auc_method=AUCMethod.LINEAR,
        terminal=TerminalPhase(method=TerminalMethod.ALL_AFTER_TMAX),
    )
    mixed = nca(study(3), options=other)
    low, high = (
        int(np.nanmin(mixed["lambda_z_n_points"].to_numpy())),
        int(np.nanmax(mixed["lambda_z_n_points"].to_numpy())),
    )
    assert low < high
    assert methods_line(other, mixed) == (
        "The areas were computed with the linear trapezoidal method. The "
        "terminal log-linear phase was selected as every point after the "
        f"maximum and estimated by log-linear regression using {low} to "
        f"{high} data points."
    )


def test_the_methods_line_says_when_windows_were_set_by_hand() -> None:
    options = NCAOptions(
        auc_method=AUCMethod.LINEAR_LOG,
        terminal=TerminalPhase(windows={"s1": (6.0, 24.0)}),
    )
    result = nca(study(0), options=options)
    line = methods_line(options, result)
    assert "linear up / logarithmic down trapezoidal method" in line
    assert line.endswith("The terminal window of single profiles was set by hand.")


def test_the_acceptability_table_needs_a_sample_dimension_and_the_areas() -> None:
    result = nca(study(0), options=OPTIONS)
    with pytest.raises(ValueError, match="is not a sample dimension"):
        acceptability_table(result, "subject")
