"""Concentration-effect models (Emax family).

The Emax model `E = E0 + Emax C / (EC50 + C)` and its sigmoid form with the
Hill coefficient `n` describe a saturable response (Gabrielsson & Weiner 2016,
ch. 4; Bonate 2011), the exposure-response relationship regulatory guidance
addresses (FDA 2003); `Imax` is the inhibitory form
`E = E0 (1 - Imax C / (IC50 + C))`. `EC90 = 9^{1/n} EC50` is the concentration
of 90 % of the maximal effect. The same models describe a pharmacokinetic
parameter against an inhibitor dose.
"""

from typing import ClassVar

import numpy as np

from pkpdutils.fit.model import Model, ModelParameter


def _crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """First `x` at which the sorted curve crosses `level` (linear interpolation), else the median of `x`."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    above = ys >= level if ys[-1] >= ys[0] else ys <= level
    idx = np.argmax(above) if above.any() else 0
    if idx == 0 or not above.any():
        return float(np.median(xs[xs > 0])) if (xs > 0).any() else float(np.median(xs))
    x1, x2, y1, y2 = xs[idx - 1], xs[idx], ys[idx - 1], ys[idx]
    if y2 == y1:
        return float(x2)
    return float(x1 + (level - y1) / (y2 - y1) * (x2 - x1))


class Emax(Model):
    """`E = e0 + emax x / (ec50 + x)`."""

    name = "emax"
    parameters = (
        ModelParameter("e0", "[y]", positive=False, description="baseline effect"),
        ModelParameter(
            "emax", "[y]", positive=False, description="maximal effect above baseline"
        ),
        ModelParameter(
            "ec50", "[x]", description="concentration of half-maximal effect"
        ),
    )
    derived_units: ClassVar[dict[str, str]] = {"ec90": "[x]"}

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve."""
        return p[0] + p[1] * x / (p[2] + x)

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """`ec90 = 9 ec50`."""
        return {"ec90": 9.0 * float(p[2])}

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Baseline at the smallest `x`, plateau at the largest, `ec50` at the half-way crossing."""
        ok = np.isfinite(x) & np.isfinite(y)
        xs, ys = x[ok], y[ok]
        order = np.argsort(xs)
        e0, top = float(ys[order[0]]), float(ys[order[-1]])
        emax = top - e0 if top != e0 else float(ys.max() - e0) or 1.0
        ec50 = max(_crossing(xs, ys, e0 + emax / 2.0), 1e-12)
        return np.array([e0, emax, ec50])


class SigmoidEmax(Emax):
    """`E = e0 + emax x^n / (ec50^n + x^n)` with the Hill coefficient `n`."""

    name = "sigmoid_emax"
    parameters = (
        *Emax.parameters,
        ModelParameter(
            "hill",
            "dimensionless",
            lower=0.1,
            upper=20.0,
            description="Hill coefficient",
        ),
    )

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve."""
        xn = np.power(np.maximum(x, 0.0), p[3])
        return p[0] + p[1] * xn / (p[2] ** p[3] + xn)

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """`ec90 = 9^(1/n) ec50`."""
        return {"ec90": float(9.0 ** (1.0 / p[3]) * p[2])}

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The Emax guess with `hill = 1`."""
        return np.append(super().initial_guess(x, y), 1.0)


class Imax(Model):
    """`E = e0 (1 - imax x / (ic50 + x))`."""

    name = "imax"
    parameters = (
        ModelParameter("e0", "[y]", description="baseline effect"),
        ModelParameter(
            "imax",
            "dimensionless",
            lower=0.0,
            upper=1.0,
            positive=False,
            description="maximal fractional inhibition",
        ),
        ModelParameter(
            "ic50", "[x]", description="concentration of half-maximal inhibition"
        ),
    )
    derived_units: ClassVar[dict[str, str]] = {"ic90": "[x]"}

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve."""
        return p[0] * (1.0 - p[1] * x / (p[2] + x))

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """`ic90 = 9 ic50`."""
        return {"ic90": 9.0 * float(p[2])}

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Baseline at the smallest `x`, inhibition from the minimum, `ic50` at the half-way crossing."""
        ok = np.isfinite(x) & np.isfinite(y)
        xs, ys = x[ok], y[ok]
        e0 = max(float(ys[np.argmin(xs)]), 1e-12)
        imax = float(np.clip(1.0 - ys.min() / e0, 0.05, 0.95))
        ic50 = max(_crossing(xs, ys, e0 * (1.0 - imax / 2.0)), 1e-12)
        return np.array([e0, imax, ic50])


class SigmoidImax(Imax):
    """`E = e0 (1 - imax x^n / (ic50^n + x^n))`."""

    name = "sigmoid_imax"
    parameters = (
        *Imax.parameters,
        ModelParameter(
            "hill",
            "dimensionless",
            lower=0.1,
            upper=20.0,
            description="Hill coefficient",
        ),
    )

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve."""
        xn = np.power(np.maximum(x, 0.0), p[3])
        return p[0] * (1.0 - p[1] * xn / (p[2] ** p[3] + xn))

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """`ic90 = 9^(1/n) ic50`."""
        return {"ic90": float(9.0 ** (1.0 / p[3]) * p[2])}

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The Imax guess with `hill = 1`."""
        return np.append(super().initial_guess(x, y), 1.0)
