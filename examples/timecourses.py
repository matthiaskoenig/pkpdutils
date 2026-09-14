"""Creating timecourses.

Run from the root of the repository with `python -m examples.timecourses`.
The example prints the objects and writes `timecourses.tsv` into the working
directory.
"""

import numpy as np
import pandas as pd
import xarray as xr

from pkpdutils import Dose, Route, Timecourse, Timecourses
from pkpdutils.console import console


def single_timecourse() -> Timecourse:
    """One oral caffeine curve with a dose and group uncertainty."""
    return Timecourse(
        time=[0.5, 1, 2, 4, 8, 12, 24],
        value=[1.2, 2.5, 2.1, 1.3, 0.5, 0.2, 0.03],
        sd=[0.3, 0.5, 0.4, 0.3, 0.1, 0.05, 0.01],
        n=12,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        substance="caffeine",
        label="healthy",
        tissue="plasma",
    )


def batch_from_arrays() -> Timecourses:
    """Three individuals on a shared sampling grid."""
    time = np.array([0.5, 1, 2, 4, 8, 12, 24])
    kel = np.array([0.1, 0.15, 0.2])
    values = (
        3.0 * np.exp(-kel[:, None] * time[None, :]) * (1 - np.exp(-2 * time[None, :]))
    )
    return Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["s1", "s2", "s3"]},
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        substance="caffeine",
    )


def batch_from_dataframe() -> Timecourses:
    """Two subjects with different sampling times from a long table."""
    df = pd.DataFrame(
        {
            "subject": ["a", "a", "a", "a", "b", "b", "b"],
            "time": [0.5, 1, 2, 4, 1, 4, 8],
            "value": [1.0, 2.0, 1.5, 0.8, 1.8, 1.0, 0.4],
            "dose": [50, 50, 50, 50, 100, 100, 100],
        }
    )
    return Timecourses.from_dataframe(
        df,
        sample=["subject"],
        time_unit="hr",
        unit="mg/l",
        dose_amount="dose",
        dose_unit="mg",
        route=Route.ORAL,
        substance="caffeine",
    )


def batch_from_simulation() -> Timecourses:
    """A dataset shaped like an sbmlsim scan result: `_time` plus a scan dimension."""
    time = np.linspace(0, 24, 49)
    doses = np.array([25.0, 50.0, 100.0])
    values = doses[None, :] / 40 * np.exp(-0.15 * time[:, None])
    ds = xr.Dataset(
        {"[Cve]": (("_time", "dim_dose"), values)},
        coords={"_time": time, "dim_dose": doses},
    )
    return Timecourses.from_dataset(
        ds, "[Cve]", unit="mg/l", time_unit="hr", substance="caffeine"
    )


if __name__ == "__main__":
    tc = single_timecourse()
    console.rule("Timecourse")
    console.print(tc)
    console.print(tc.to_dataframe())

    console.rule("Timecourses from arrays")
    tcs = batch_from_arrays()
    console.print(tcs.ds)
    for item in tcs:
        console.print(item.label, item.value.round(3))

    console.rule("Timecourses from a data frame (ragged)")
    ragged = batch_from_dataframe()
    console.print(ragged.ds)
    console.print(ragged.sel(subject="b"))

    console.rule("Timecourses from a simulation dataset")
    scan = batch_from_simulation()
    console.print(scan.sample_dims, scan.sample_shape)

    tcs.to_dataframe().to_csv("timecourses.tsv", sep="\t", index=False)
    console.print("written: timecourses.tsv")
