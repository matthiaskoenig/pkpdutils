"""Clearance against body weight: allometric scaling with a free and a fixed exponent.

Run from the root of the repository with `python -m examples.covariate`.
Writes `covariate.png` into the working directory.
"""

import numpy as np
import xarray as xr

from pkpdutils.console import console
from pkpdutils.fit import compare_models, fit_table
from pkpdutils.fit.models import Allometric, Linear
from pkpdutils.plot import plot_fit

# clearances of nine subjects, generated with the exponent 0.72 and 8 % noise
weights = np.array([45.0, 52, 60, 68, 75, 82, 90, 105, 120])
rng = np.random.default_rng(5)
clearance = 0.9 * weights**0.72 * rng.lognormal(0, 0.08, weights.size)
ds = xr.Dataset(
    {"cl": (("individual",), clearance, {"units": "liter / hour"})},
    coords={
        "weight": ("individual", weights, {"units": "kg"}),
        "individual": [f"s{i}" for i in range(weights.size)],
    },
)

if __name__ == "__main__":
    console.rule("Free exponent")
    free = fit_table(Allometric(), ds, "weight", "cl", dim="individual")
    console.print(
        free.to_dataframe().T.loc[["a", "b", "b_se", "b_ci_low", "b_ci_high", "r2"]]
    )

    console.rule("Fixed exponent 0.75 and a linear model")
    fixed = fit_table(Allometric(exponent=0.75), ds, "weight", "cl", dim="individual")
    console.print(fixed.to_dataframe().T.loc[["a", "a_se", "r2"]])
    comparison = compare_models(
        [Allometric(), Allometric(exponent=0.75), Linear()],
        weights,
        clearance,
        x_unit="kg",
        y_unit="liter/hour",
    )
    console.print(comparison.table)

    # the allometric fit on log-log axes, where it is a straight line
    plot_fit(free, log_x=True, log_y=True, title="allometric scaling").savefig(
        "covariate.png", dpi=120
    )
    console.print("written: covariate.png")
