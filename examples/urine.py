"""Urinary excretion: the rate curve, the amount recovered and the renal clearance.

A simulated intravenous bolus of 100 mg of which 60 % is excreted unchanged in
the urine, collected over eight intervals of a day, together with the plasma
curve of the same subject. The analysis reports the parameters of the excretion
rate curve, how much was recovered and the renal clearance over the collection
span.

Run from the root of the repository with `python -m examples.urine`.
Writes `urine.png` into the working directory.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pkpdutils import Dose, Excretion, Route, Timecourse, nca_urine
from pkpdutils.console import console, print_table
from pkpdutils.plot import plot_excretion

#: dose of the bolus, in mg
DOSE = 100.0

#: fraction of the dose excreted unchanged in the urine
FE = 0.6

#: elimination rate constant, in 1/hr
K = 0.15

#: volume of distribution, in liter
VD = 25.0

#: the ends of the collection intervals, in hr
EDGES = np.array([0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 36.0])


def excretion() -> Excretion:
    """The amounts and volumes of the collections of the simulated subject."""
    start, end = EDGES[:-1], EDGES[1:]
    amount = FE * DOSE * (np.exp(-K * start) - np.exp(-K * end))
    volume = np.array([180.0, 150.0, 260.0, 240.0, 210.0, 190.0, 220.0, 360.0])
    return Excretion(
        start=start,
        end=end,
        amount=amount,
        volume=volume,
        unit="mg",
        time_unit="hr",
        volume_unit="ml",
        dose=Dose(amount=DOSE, unit="mg", route=Route.IV_BOLUS),
        substance="drug",
        label="S1",
    )


def plasma() -> Timecourse:
    """The plasma curve of the same subject, a mono-exponential bolus."""
    time = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 36.0])
    return Timecourse(
        time=time,
        value=DOSE / VD * np.exp(-K * time),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=DOSE, unit="mg", route=Route.IV_BOLUS),
        substance="drug",
        tissue="plasma",
        label="S1",
    )


if __name__ == "__main__":
    urine = excretion()
    result = nca_urine(urine, plasma=plasma())

    console.rule("The collections and their excretion rate")
    print_table(
        pd.DataFrame(
            {
                "start [hr]": urine.start,
                "end [hr]": urine.end,
                "midpoint [hr]": urine.midpoint,
                "volume [ml]": urine.volume,
                "amount [mg]": urine.amount,
                "rate [mg/hr]": urine.rate,
                "cumulative [mg]": urine.cumulative,
            }
        ),
        title="urine collections",
    )

    console.rule("Parameters of the excretion rate curve")
    frame = result.to_dataframe().T
    frame.columns = ["value"]
    frame["unit"] = [result.units(name) for name in frame.index]
    print_table(frame.reset_index(names="parameter"), title="urine parameters")

    console.rule("Recovery and renal clearance")
    quantities = result.to_quantities()
    for name in ("amount_recovered", "percent_recovered", "vol_ur", "clr", "thalf"):
        console.print(f"{name:<18} {quantities[name]:~P}")
    console.print(
        f"analytic: {FE * DOSE * (1 - np.exp(-K * EDGES[-1])):.3f} mg recovered, "
        f"CLr = {FE * K * VD:.3f} l/hr"
    )

    fig = plot_excretion(result, urine)
    fig.set_size_inches(7.0, 4.5)
    fig.savefig("urine.png", dpi=120)
    plt.close(fig)
    console.print("written: urine.png")
