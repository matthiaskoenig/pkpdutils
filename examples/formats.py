"""Exchange formats: event records, and a CDISC ADaM ADNCA extract.

Builds a BID batch, writes it as event records with `to_events`, reads the
table back with `from_events` and runs the multiple dosing NCA on it; then
reads a small ADNCA fixture with `from_adnca` and prints the recovered dosing
protocols.

Run from the root of the repository with `python -m examples.formats`.
Writes `events.csv` and `formats.png` into the working directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pkpdutils import Dose, Dosing, NCAOptions, Route, Timecourses, nca
from pkpdutils.console import console
from pkpdutils.nca import AUCMethod
from pkpdutils.plot import plot_intervals

#: fixture of the design spec, `tests/data/formats/adnca.csv` relative to the
#: repository root (this file lives in `examples/`, one level below it)
ADNCA_PATH = Path(__file__).parents[1] / "tests" / "data" / "formats" / "adnca.csv"

individuals = ["s1", "s2", "s3", "s4"]
dose = Dose(amount=100, unit="mg", time=0, route=Route.ORAL)
protocol = Dosing.regimen(dose, interval=12, n_doses=4)  # BID, 2 days

# samples every dosing interval: predose, several timepoints, the trough (the
# predose sample of the next dose, or the extra draw closing the last one)
offsets = np.array([0.5, 1, 2, 4, 8, 12])
time = np.concatenate([dose_time + offsets for dose_time in protocol.times])

rng = np.random.default_rng(3)
ke = rng.uniform(0.12, 0.18, size=len(individuals))
ka = rng.uniform(0.8, 1.2, size=len(individuals))
values = np.empty((len(individuals), time.size))
for j in range(len(individuals)):
    concentration = np.zeros_like(time)
    for dose_time in protocol.times:
        elapsed = time - dose_time
        single = (
            protocol.amounts[0]
            / 20.0
            * ka[j]
            / (ka[j] - ke[j])
            * (np.exp(-ke[j] * elapsed) - np.exp(-ka[j] * elapsed))
        )
        concentration = concentration + np.where(elapsed >= 0, single, 0.0)
    values[j] = concentration * rng.lognormal(0, 0.03, size=time.size)

batch = Timecourses.from_arrays(
    time,
    values,
    time_unit="hr",
    unit="mg/l",
    dims=("individual",),
    coords={"individual": individuals},
    dose=protocol,
    route=Route.ORAL,
    substance="drug",
)

if __name__ == "__main__":
    events = batch.to_events()
    events.to_csv("events.csv", index=False)
    console.rule("Event records written to events.csv")
    console.print(events.head())

    read_back = Timecourses.from_events(
        pd.read_csv("events.csv"),
        time_unit="hr",
        unit="mg/l",
        dose_unit="mg",
        route=Route.ORAL,
    )
    result = nca(read_back, NCAOptions(auc_method=AUCMethod.LOG))

    console.rule("Parameters of the dosing intervals")
    console.print(
        result.intervals()[
            [
                "individual",
                "interval",
                "interval_start",
                "interval_auc",
                "interval_ctrough",
            ]
        ]
    )

    console.rule("Steady state parameters of the last interval")
    console.print(
        result.to_dataframe()[
            [
                "individual",
                "n_doses",
                "tau",
                "auc_tau",
                "cmax_ss",
                "ctrough",
                "accumulation_ratio_obs",
                "flags",
            ]
        ]
    )

    plot_intervals(result, "interval_ctrough").savefig("formats.png", dpi=120)
    console.print("written: events.csv, formats.png")

    console.rule("CDISC ADaM ADNCA: recovered dosing protocols")
    adnca_batch = Timecourses.from_adnca(pd.read_csv(ADNCA_PATH), analyte="XAN")
    for label in adnca_batch.ds["individual"].to_numpy():
        console.print(str(label), adnca_batch.dosing_of(individual=str(label)))
