import numpy as np
import xarray as xr

from pkpdutils.fit import fit_table, proportionality_table, proportionality_test
from pkpdutils.fit.models import Power

DOSES = np.array([25.0, 50.0, 100.0, 200.0])
INDIVIDUALS = ["a", "b", "c"]


def exposure() -> xr.Dataset:
    rng = np.random.default_rng(1)
    auc = 2.0 * DOSES[:, None] ** 1.05 * rng.lognormal(0, 0.05, (4, 3))
    return xr.Dataset(
        {
            "auc_inf_obs": (
                ("dose", "individual"),
                auc,
                {"units": "hour * milligram / liter"},
            )
        },
        coords={
            "dose": ("dose", DOSES, {"units": "milligram"}),
            "individual": INDIVIDUALS,
        },
    )


def test_proportionality_table_of_one_escalation() -> None:
    ds = exposure().mean("individual")
    result = fit_table(Power(), ds, "dose", "auc_inf_obs", dim="dose")
    verdict = proportionality_test(result, dose_range=(25.0, 200.0))
    df = proportionality_table(verdict)
    assert list(df.columns) == [
        "slope",
        "ci_low",
        "ci_high",
        "bound_low",
        "bound_high",
        "dose_low",
        "dose_high",
        "verdict",
    ]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["slope"] == f"{float(verdict.slope):.3g}"
    assert row["bound_low"] == f"{verdict.bounds[0]:.3g}"
    assert row["bound_high"] == f"{verdict.bounds[1]:.3g}"
    assert row["dose_low"] == "25.0" and row["dose_high"] == "200"
    assert row["verdict"] in ("proportional", "inconclusive", "not proportional")
    expected = (
        "proportional"
        if bool(verdict.proportional)
        else "inconclusive"
        if bool(verdict.inconclusive)
        else "not proportional"
    )
    assert row["verdict"] == expected


def test_proportionality_table_of_a_batch() -> None:
    result = fit_table(Power(), exposure(), "dose", "auc_inf_obs", dim="dose")
    verdict = proportionality_test(result, dose_range=(25.0, 200.0))
    df = proportionality_table(verdict, digits=4)
    assert df["individual"].tolist() == INDIVIDUALS
    assert next(iter(df.columns)) == "individual"
    assert df.iloc[1]["slope"] == f"{float(verdict.slope.sel(individual='b')):.4g}"
    assert df["bound_low"].nunique() == 1  # the criterion is the same for every sample
