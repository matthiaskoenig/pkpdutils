"""Linear, log-linear, power and allometric models.

`Power` (`y = a x^b`) is the model of dose proportionality (Smith et al.
2000): `b = 1` is proportional; `proportionality_test` in
`pkpdutils.fit.proportionality` applies the confidence interval criterion.
`Allometric` is the same model for a parameter against body weight with the
exponent free or fixed (0.75 for clearances, 1 for volumes; Rowland & Tozer
2011, ch. 12). `Linear` and `LogLinear` describe an effect or a parameter
against a concentration or covariate.
"""

import warnings

import numpy as np

from pkpdutils.fit.model import Model, ModelParameter


def _finite(
    x: np.ndarray, y: np.ndarray, positive_x: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """The finite points, optionally with positive `x`."""
    ok = np.isfinite(x) & np.isfinite(y)
    if positive_x:
        ok &= x > 0
    return x[ok], y[ok]


def _polyfit(x: np.ndarray, y: np.ndarray, deg: int) -> np.ndarray:
    """`np.polyfit` with the rank-deficiency warning silenced (too few or collinear points)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=np.exceptions.RankWarning)
        return np.polyfit(x, y, deg)


class Linear(Model):
    """`y = intercept + slope x`."""

    name = "linear"
    parameters = (
        ModelParameter(
            "intercept", "[y]", positive=False, description="value at x = 0"
        ),
        ModelParameter("slope", "[y]/[x]", positive=False, description="slope"),
    )

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The line."""
        return p[0] + p[1] * x

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ordinary least squares."""
        xs, ys = _finite(x, y)
        slope, intercept = (
            _polyfit(xs, ys, 1) if xs.size >= 2 else (0.0, float(ys.mean()))
        )
        return np.array([intercept, slope], dtype=np.float64)


class LogLinear(Model):
    """`y = intercept + slope ln x` for `x > 0`."""

    name = "loglinear"
    parameters = (
        ModelParameter(
            "intercept", "[y]", positive=False, description="value at x = 1"
        ),
        ModelParameter(
            "slope", "[y]", positive=False, description="change per e-fold of x"
        ),
    )

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve (`NaN` for `x <= 0`)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return p[0] + p[1] * np.log(x)

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Least squares on `ln x`."""
        xs, ys = _finite(x, y, positive_x=True)
        slope, intercept = (
            _polyfit(np.log(xs), ys, 1) if xs.size >= 2 else (0.0, float(ys.mean()))
        )
        return np.array([intercept, slope], dtype=np.float64)


class Power(Model):
    """`y = a x^b` for `x > 0`."""

    name = "power"
    parameters = (
        ModelParameter("a", "[y]", description="value at x = 1"),
        ModelParameter("b", "dimensionless", positive=False, description="exponent"),
    )

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return p[0] * np.power(x, p[1])

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Least squares on the log-log data."""
        xs, ys = _finite(x, y, positive_x=True)
        ok = ys > 0
        if ok.sum() >= 2:
            b, log_a = _polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)
            return np.array([np.exp(log_a), b], dtype=np.float64)
        return np.array([max(float(ys.mean()), 1e-12), 1.0])


class Allometric(Model):
    """`y = a x^b` against body weight with the exponent free or fixed.

    Args:
        exponent: fixed exponent (`0.75` for clearances, `1` for volumes), `None` to fit it
    """

    name = "allometric"

    def __init__(self, exponent: float | None = None) -> None:
        """Create the model with a free or a fixed exponent.

        Args:
            exponent: fixed exponent, `None` to fit it.
        """
        self.exponent = exponent
        self.name = "allometric" if exponent is None else f"allometric_{exponent:g}"
        a = ModelParameter("a", "[y]", description="value at unit weight")
        if exponent is None:
            self.parameters = (
                a,
                ModelParameter(
                    "b",
                    "dimensionless",
                    positive=False,
                    description="allometric exponent",
                ),
            )
        else:
            self.parameters = (a,)

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve."""
        b = p[1] if self.exponent is None else self.exponent
        with np.errstate(divide="ignore", invalid="ignore"):
            return p[0] * np.power(x, b)

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Log-log least squares (`a` only when the exponent is fixed)."""
        xs, ys = _finite(x, y, positive_x=True)
        ok = ys > 0
        if self.exponent is None:
            if ok.sum() >= 2:
                b, log_a = _polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)
                return np.array([np.exp(log_a), b], dtype=np.float64)
            return np.array([max(float(ys.mean()), 1e-12), 0.75])
        if ok.any():
            log_a = np.mean(np.log(ys[ok]) - self.exponent * np.log(xs[ok]))
            return np.array([np.exp(log_a)], dtype=np.float64)
        return np.array([1.0])
