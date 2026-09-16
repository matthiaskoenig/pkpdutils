"""Non-compartmental analysis of a simulation result.

The result of an sbmlsim parameter scan is an `XResult`, an xarray dataset with
the `_time` dimension and one dimension per scan dimension; `Timecourses.from_xresult`
turns such a result into a batch and `nca` analyses every simulated curve at once.
sbmlsim is not a dependency of pkpdutils, so this example builds a dataset of the
same shape and reads it with `Timecourses.from_dataset`; with a real `XResult` the
call is `Timecourses.from_xresult(xresult, "[Cve]", dose=..., substance=...)`, which
takes the units from the result instead of the `unit` and `time_unit` arguments.

Run from the root of the repository with `python -m examples.nca_from_sbmlsim`.
Writes `nca_from_sbmlsim.png` into the working directory.
"""

import numpy as np
import xarray as xr

from pkpdutils import NCAOptions, Route, Timecourses, nca
from pkpdutils.console import console
from pkpdutils.nca import AUCMethod
from pkpdutils.plot import plot_timecourse

#: the doses of the scan, the scan dimension of the simulated dataset
DOSES = np.array([25.0, 50.0, 100.0, 200.0])


def simulated_dataset() -> xr.Dataset:
    """A dataset shaped like an sbmlsim scan over the dose."""
    time = np.linspace(0, 24, 241)
    values = (
        DOSES[None, :]
        / 20
        * np.exp(-0.2 * time[:, None])
        * (1 - np.exp(-1.5 * time[:, None]))
    )
    return xr.Dataset(
        {"[Cve]": (("_time", "dose"), values)},
        coords={"_time": time, "dose": ("dose", DOSES, {"units": "mg"})},
    )


if __name__ == "__main__":
    try:
        from sbmlsim.result import XResult  # ty: ignore[unresolved-import]

        console.print("sbmlsim is installed:", XResult.__name__)
    except ImportError:
        console.print("sbmlsim is not installed, using a dataset of the same shape")

    ds = simulated_dataset()
    # every simulated curve carries the dose of its scan point, so that the
    # dose dependent parameters (cl_f, vz_f, auc_inf_dn) are the ones of the
    # dose the curve was simulated with
    batch = Timecourses.from_dataset(
        ds,
        "[Cve]",
        unit="mmol/l",
        time_unit="hr",
        dose={"amount": DOSES, "unit": "mg"},
        route=Route.ORAL,
        substance="drug",
    )
    result = nca(batch, options=NCAOptions(auc_method=AUCMethod.LINEAR_LOG))
    console.rule("NCA over the scan dimension")
    console.print(
        result.to_dataframe()[["dose", "auc_inf_obs", "cmax", "tmax", "thalf", "flags"]]
    )

    # the simulated curves of the scan, one color per scanned dose
    plot_timecourse(batch, by="dose").savefig("nca_from_sbmlsim.png", dpi=120)
    console.print("written: nca_from_sbmlsim.png")
