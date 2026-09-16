"""Concentration-effect relationship with the Emax family.

Run from the root of the repository with `python -m examples.emax`.
Writes `emax.png` into the working directory.
"""

import numpy as np

from pkpdutils import FitOptions, compare_models, fit
from pkpdutils.console import console
from pkpdutils.fit.models import Emax, Linear, SigmoidEmax
from pkpdutils.plot import plot_fit

# blood pressure change against the concentration, a sigmoid Emax curve with noise
concentration = np.array([0.1, 0.3, 1, 3, 10, 30, 100, 300])
rng = np.random.default_rng(3)
effect = SigmoidEmax().predict(
    concentration, np.array([5.0, 40.0, 8.0, 1.8])
) + rng.normal(0, 1.0, concentration.size)

if __name__ == "__main__":
    console.rule("Sigmoid Emax fit")
    result = fit(
        SigmoidEmax(),
        concentration,
        effect,
        x_unit="ng/ml",
        y_unit="mmHg",
        x_name="concentration",
        y_name="effect",
        options=FitOptions(n_starts=10, seed=0),
    )
    q = result.to_quantities()
    for name in ("e0", "emax", "ec50", "hill", "ec90"):
        console.print(f"{name:<6} {q[name]:.4g~P}  se {q[name + '_se']:.3g~P}")
    console.print("flags:", result.flags())
    plot_fit(result, log_x=True).savefig("emax.png", dpi=120)

    console.rule("Emax versus sigmoid Emax versus linear")
    comparison = compare_models(
        [Emax(), SigmoidEmax(), Linear()],
        concentration,
        effect,
        x_unit="ng/ml",
        y_unit="mmHg",
        options=FitOptions(n_starts=10, seed=0),
    )
    console.print(comparison.table)
    console.print("written: emax.png")
