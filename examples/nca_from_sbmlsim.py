"""Non-compartmental analysis of a simulation result.

The result of an sbmlsim parameter scan is an `XResult`, an xarray dataset with
the `_time` dimension and one dimension per scan dimension. `Timecourses.from_xresult`
turns it into a batch, `nca` analyses every simulated curve at once. sbmlsim is
not a dependency of pkpdutils: without it the example builds a dataset of the
same shape and analyses that.

Run from the root of the repository with `python -m examples.nca_from_sbmlsim`.
"""

import numpy as np
import xarray as xr

from pkpdutils import Dose, NCAOptions, Route, Timecourses, nca
from pkpdutils.console import console
from pkpdutils.nca import AUCMethod


def simulated_dataset() -> xr.Dataset:
    """A dataset shaped like an sbmlsim scan over the dose."""
    time = np.linspace(0, 24, 241)
    doses = np.array([25.0, 50.0, 100.0, 200.0])
    values = (
        doses[None, :]
        / 20
        * np.exp(-0.2 * time[:, None])
        * (1 - np.exp(-1.5 * time[:, None]))
    )
    return xr.Dataset(
        {"[Cve]": (("_time", "dim_dose"), values)},
        coords={"_time": time, "dim_dose": doses},
    )


if __name__ == "__main__":
    try:
        from sbmlsim.result import XResult  # ty: ignore[unresolved-import]

        console.print("sbmlsim is installed:", XResult.__name__)
    except ImportError:
        console.print("sbmlsim is not installed, using a dataset of the same shape")

    ds = simulated_dataset()
    batch = Timecourses.from_dataset(
        ds,
        "[Cve]",
        unit="mmol/l",
        time_unit="hr",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        substance="drug",
    )
    result = nca(batch, NCAOptions(auc_method=AUCMethod.LINEAR_LOG))
    console.rule("NCA over the scan dimension")
    console.print(
        result.to_dataframe()[
            ["dim_dose", "auc_inf_obs", "cmax", "tmax", "thalf", "flags"]
        ]
    )
