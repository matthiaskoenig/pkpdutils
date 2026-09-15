"""Fitting exponential models to a concentration timecourse.

Run from the root of the repository with `python -m examples.fitting_exponential`.
Writes `fitting_exponential.png` and `fitting_gof.png` into the working directory.
"""

import numpy as np

from pkpdutils import (
    Dose,
    FitOptions,
    Route,
    Timecourse,
    compare_models,
    fit_timecourse,
)
from pkpdutils.console import console
from pkpdutils.fit import Weighting
from pkpdutils.fit.models import Bateman, BiExp, MonoExp
from pkpdutils.plot import plot_fit, plot_goodness_of_fit

# an oral curve: a Bateman curve with a 5 % log-normal error per point
t = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 24])
rng = np.random.default_rng(0)
true = Bateman().predict(t, np.array([12.0, 1.8, 0.25]))
value = true * rng.lognormal(0, 0.05, t.size)
tc = Timecourse(
    time=t,
    value=value,
    sd=0.05 * value,
    time_unit="hr",
    unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    substance="drug",
    label="oral",
)

if __name__ == "__main__":
    console.rule("Bateman fit with 1/sd weighting and a residual bootstrap")
    result = fit_timecourse(
        Bateman(),
        tc,
        options=FitOptions(
            weighting=Weighting.INV_SD, n_starts=5, bootstrap=200, seed=1
        ),
    )
    console.print(result.to_dataframe().T)  # the last row holds the decoded flags
    plot_fit(result, log_y=True).savefig("fitting_exponential.png", dpi=120)

    console.rule("Which model? AICc of mono-, bi-exponential and Bateman")
    comparison = compare_models(
        [MonoExp(), BiExp(), Bateman()],
        t,
        value,
        x_unit="hr",
        y_unit="mg/l",
        options=FitOptions(n_starts=5, seed=2),
    )
    console.print(comparison.table)
    console.print("best:", str(comparison.best.values))
    plot_goodness_of_fit(comparison.results["bateman"]).savefig(
        "fitting_gof.png", dpi=120
    )
    console.print("written: fitting_exponential.png, fitting_gof.png")
