"""Non-compartmental analysis of a batch: individuals of a dose escalation.

Run from the root of the repository with `python -m examples.nca_batch`.
Writes `nca_batch.png` and `nca_batch.tsv` into the working directory.
"""

import numpy as np

from pkpdutils import NCAOptions, Route, Timecourses, nca
from pkpdutils.console import console
from pkpdutils.plot import plot_nca_grid, plot_timecourse

rng = np.random.default_rng(1)
time = np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 24])
doses = np.array([50.0, 100.0, 200.0])
individuals = ["s1", "s2", "s3", "s4"]
ke = rng.uniform(0.15, 0.3, size=len(individuals))
ka = rng.uniform(1.0, 3.0, size=len(individuals))
values = np.empty((len(doses), len(individuals), time.size))
for i, dose in enumerate(doses):
    for j in range(len(individuals)):
        curve = (
            dose
            / 40
            * ka[j]
            / (ka[j] - ke[j])
            * (np.exp(-ke[j] * time) - np.exp(-ka[j] * time))
        )
        values[i, j] = curve * rng.lognormal(0, 0.05, size=time.size)

batch = Timecourses.from_arrays(
    time,
    values,
    time_unit="hr",
    unit="mg/l",
    dims=("dose", "individual"),
    coords={"dose": doses, "individual": individuals},
    dose={"amount": np.broadcast_to(doses[:, None], (3, 4)), "unit": "mg"},
    route=Route.ORAL,
    substance="drug",
)

if __name__ == "__main__":
    result = nca(batch, options=NCAOptions())
    console.rule("Parameters over (dose, individual)")
    console.print(result.ds)
    df = result.to_dataframe()
    console.print(
        df[["dose", "individual", "auc_inf_obs", "cmax", "thalf", "cl_f", "flags"]]
    )
    df.to_csv("nca_batch.tsv", sep="\t", index=False)

    console.rule("Dose proportionality at a glance: AUC / dose")
    console.print(result["auc_inf_dn"].mean(dim="individual").values)

    plot_timecourse(batch, log_y=True, by="individual").savefig(
        "nca_batch_curves.png", dpi=120
    )
    plot_nca_grid(batch, result, ncols=4).savefig("nca_batch.png", dpi=100)
    console.print("written: nca_batch.tsv, nca_batch_curves.png, nca_batch.png")
