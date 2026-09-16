"""Exchange formats: event records, and a CDISC ADaM ADNCA extract.

Builds a BID batch, writes it as event records with `to_events`, reads the
table back with `from_events` and runs the multiple dosing NCA on it; then
reads a small ADNCA extract with `from_adnca` and prints the recovered dosing
protocols.

Run from the root of the repository with `python -m examples.formats`.
Writes `events.csv` and `formats.png` into the working directory.
"""

import numpy as np
import pandas as pd

from pkpdutils import AUCMethod, Dose, Dosing, NCAOptions, Route, Timecourses, nca
from pkpdutils.console import console
from pkpdutils.plot import plot_intervals

#: a small CDISC ADaM ADNCA (ADPC) extract: one row per concentration record,
#: the time since the first dose (`AFRLT`) and since the reference dose
#: (`ARRLT`), the pre-dose record of the second interval duplicated into the
#: first one (`DTYPE == "COPY"`)
ADNCA = pd.DataFrame(
    {
        "USUBJID": ["S1", "S1", "S1", "S2", "S2", "S2", "S2", "S2"],
        "PARAMCD": ["XAN"] * 8,
        "AVAL": [0.05, 4.2, 1.0, 4.0, 1.1, 1.1, 5.5, 2.0],
        "AVALU": ["ng/mL"] * 8,
        "AFRLT": [0.5, 1.0, 12.0, 1.0, 12.0, 12.0, 13.0, 24.0],
        "ARRLT": [0.5, 1.0, 12.0, 1.0, 12.0, 0.0, 1.0, 12.0],
        "DOSEA": [100.0] * 8,
        "DOSEU": ["mg"] * 8,
        "ROUTE": ["ORAL"] * 8,
        "DTYPE": [None, None, None, None, None, "COPY", None, None],
        "ALLOQ": [0.1] * 8,
    }
)

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
    result = nca(read_back, options=NCAOptions(auc_method=AUCMethod.LOG))

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
    adnca_batch = Timecourses.from_adnca(ADNCA, analyte="XAN")
    for label in adnca_batch.ds["individual"].to_numpy():
        console.print(str(label), adnca_batch.dosing_of(individual=str(label)))
