"""Sparse sampling: the Bailer standard error of the area of a mean curve.

A simulated toxicokinetic study of 24 mice at six nominal times. In the serial
design every mouse is sacrificed for its single sample, four per time point; in
the batch design the same 24 animals are split into two batches of twelve, each
sampled at three of the six times. Both designs give the same estimator, and the
batch design pays for its extra samples with the covariance between the times an
animal is shared by.

Run from the root of the repository with `python -m examples.sparse`.
Writes `sparse.png` into the working directory.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pkpdutils import Dose, Route, nca_sparse, sparse_mean
from pkpdutils.console import console, print_table
from pkpdutils.plot import plot_sparse

#: the nominal sampling times of the study, in hr
TIMES = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])

#: the dose of every animal
DOSE = Dose(amount=5.0, unit="mg/kg", route=Route.ORAL)

#: the mean concentration of the underlying one compartment model, in ng/ml
MEAN = 120.0 * (np.exp(-0.18 * TIMES) - np.exp(-1.4 * TIMES))

#: between-animal variability of the concentration, as a geometric CV
CV = 0.30


def serial_design(seed: int = 7, per_time: int = 4) -> np.ndarray:
    """One sample per animal, `per_time` animals at every nominal time."""
    rng = np.random.default_rng(seed)
    values = np.full((per_time * TIMES.size, TIMES.size), np.nan)
    for j, mean in enumerate(MEAN):
        draws = mean * rng.lognormal(0.0, CV, size=per_time)
        values[j * per_time : (j + 1) * per_time, j] = draws
    return values


def batch_design(seed: int = 7, per_batch: int = 12) -> np.ndarray:
    """Two batches of animals, each sampled at three of the six nominal times."""
    rng = np.random.default_rng(seed)
    schedule = [(0, 2, 4), (1, 3, 5)]
    values = np.full((2 * per_batch, TIMES.size), np.nan)
    for b, columns in enumerate(schedule):
        # one animal is one row: its samples share its own level, which is
        # what makes the means of two times of a batch correlated
        level = rng.lognormal(0.0, 0.25, size=per_batch)
        rows = slice(b * per_batch, (b + 1) * per_batch)
        for column in columns:
            residual = rng.lognormal(0.0, 0.15, size=per_batch)
            values[rows, column] = MEAN[column] * level * residual
    return values


if __name__ == "__main__":
    serial = serial_design()
    curve = sparse_mean(
        TIMES, serial, time_unit="hr", unit="ng/ml", dose=DOSE, substance="drug"
    )
    result = nca_sparse(
        TIMES, serial, design="serial", time_unit="hr", unit="ng/ml", dose=DOSE
    )

    console.rule("The mean curve of the serial design")
    print_table(
        pd.DataFrame(
            {
                "time [hr]": TIMES,
                "n": result["n_points"].to_numpy(),
                "mean [ng/ml]": curve.values.reshape(-1),
                "sd [ng/ml]": curve.ds["sd"].to_numpy().reshape(-1),
                "se [ng/ml]": curve.ds["se"].to_numpy().reshape(-1),
            }
        ),
        title="mean curve",
    )

    console.rule("Bailer estimator of the area, serial design")
    quantities = result.to_quantities()
    for name in (
        "auc_last",
        "auc_last_se",
        "auc_last_df",
        "auc_all",
        "cmax",
        "cmax_se",
    ):
        console.print(f"{name:<14} {quantities[name]:~P}")

    console.rule("The same animals as two batches")
    batch = nca_sparse(
        TIMES, batch_design(), design="batch", time_unit="hr", unit="ng/ml", dose=DOSE
    )
    console.print(
        f"auc_last {float(batch['auc_last']):.1f} ± {float(batch['auc_last_se']):.1f} "
        f"{batch.units('auc_last')}, df {float(batch['auc_last_df']):.1f}"
    )

    fig = plot_sparse(curve, result)
    fig.set_size_inches(7.0, 4.5)
    fig.savefig("sparse.png", dpi=120)
    plt.close(fig)
    console.print("written: sparse.png")
