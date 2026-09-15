"""Exponential models of concentration timecourses.

Sums of exponentials describe the decline of a concentration after an
intravenous dose and, with an absorption term, after an extravascular dose
(Gibaldi & Perrier 1982, ch. 1-2; Gabrielsson & Weiner 2016, ch. 3). The
models here are descriptive: the coefficients `a_i` and rate constants `k_i`
carry no compartmental interpretation; `lambda_z` is the smallest rate
constant, `t½ = ln 2 / k`, and the area is `sum a_i / k_i`.

- `MonoExp`: `y = a e^{-k x}`
- `BiExp`: `y = a1 e^{-k1 x} + a2 e^{-k2 x}` with `k1 > k2`
- `TriExp`: three terms with `k1 > k2 > k3`
- `Bateman`: `y = a ka / (ka - ke) (e^{-ke t} - e^{-ka t})`, the one
  compartment curve with first order absorption, optionally with a lag time;
  `ka < ke` is a flip-flop (the terminal phase reflects absorption)

The initial guesses use the method of residuals (curve stripping): the
terminal phase is regressed on the last points, its contribution is
subtracted and the residuals give the faster phase.
"""

import math
import warnings
from typing import ClassVar

import numpy as np

from pkpdutils.fit.model import Model, ModelParameter

LN2 = math.log(2.0)


def log_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of `ln y` on `x` over the finite positive points.

    Args:
        x: independent variable.
        y: dependent variable.

    Returns:
        `(slope, intercept)`, `(nan, nan)` with fewer than two usable points.
    """
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if ok.sum() < 2:
        return math.nan, math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=np.exceptions.RankWarning)
        slope, intercept = np.polyfit(x[ok], np.log(y[ok]), 1)
    return float(slope), float(intercept)


def terminal_guess(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """`a` and `k` of the terminal phase from the last half of the points after the maximum.

    Args:
        x: independent variable.
        y: dependent variable.

    Returns:
        `(a, k)`, falling back to the maximum and `ln 2 / (range / 3)` when no
        regression is possible.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return math.nan, math.nan
    imax = int(np.argmax(y))
    tail = np.arange(x.size) >= max(imax, imax + (x.size - imax) // 2)
    if tail.sum() < 3:
        tail = np.arange(x.size) >= imax
    slope, intercept = log_linear_regression(x[tail], y[tail])
    if not np.isfinite(slope) or slope >= 0:
        k = LN2 / max((x.max() - x.min()) / 3.0, 1e-12)
        return float(y.max()), float(k)
    return float(math.exp(intercept)), float(-slope)


def _strip(x: np.ndarray, y: np.ndarray, phases: int) -> np.ndarray:
    """Curve stripping into `phases` exponential phases, slowest last.

    The terminal phase is regressed on the last half of the points after the
    maximum; its contribution is subtracted from the earlier points and the
    residuals give the next faster phase, repeated until `phases` phases are
    found (Gibaldi & Perrier 1982, ch. 2).

    Args:
        x: independent variable.
        y: dependent variable.
        phases: number of exponential phases to strip.

    Returns:
        `[a_fast, k_fast, ..., a_slow, k_slow]`.
    """
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[ok], y[ok]
    coefficients: list[tuple[float, float]] = []
    residual = y.copy()
    exhausted = False
    for _ in range(phases - 1):
        if exhausted:
            coefficients.append((math.nan, math.nan))
            continue
        a, k = terminal_guess(x, residual)
        coefficients.append((a, k))
        residual = residual - a * np.exp(-k * x)
        keep = residual > 0
        if keep.sum() < 2:
            # too few points left for another phase: fill the rest from the
            # slowest phase below, never truncate the phase count
            exhausted = True
            continue
        # the remaining phase lives in the early points
        x, residual = x[keep], residual[keep]
    if exhausted or residual.size < 2:
        a, k = math.nan, math.nan
    else:
        a, k = terminal_guess(x, residual)
    coefficients.append((a, k))
    # fill failed phases from the slowest one
    slow_a, slow_k = coefficients[0]
    fixed: list[tuple[float, float]] = []
    for i, (a, k) in enumerate(reversed(coefficients)):
        if not (np.isfinite(a) and np.isfinite(k)) or k <= 0:
            factor = 5.0 ** (len(coefficients) - 1 - i)
            a, k = slow_a, slow_k * factor
        fixed.append((a, k))
    # fixed is fast..slow; enforce strict ordering of the rates
    out = []
    for a, k in fixed:
        out.extend([a, k])
    p = np.array(out, dtype=np.float64)
    return _sort_phases(p)


def _sort_phases(p: np.ndarray) -> np.ndarray:
    """Order `[a1, k1, a2, k2, ...]` by decreasing rate.

    Args:
        p: phases as `[a1, k1, a2, k2, ...]`, any order.

    Returns:
        The phases ordered by decreasing rate constant; equal rates are
        spread apart by 1 % so the start vector is strictly ordered.
    """
    pairs = p.reshape(-1, 2)
    order = np.argsort(-pairs[:, 1], kind="stable")
    pairs = pairs[order]
    for i in range(1, pairs.shape[0]):
        if pairs[i, 1] >= pairs[i - 1, 1]:
            pairs[i, 1] = pairs[i - 1, 1] * 0.99
    return pairs.reshape(-1)


class MonoExp(Model):
    """`y = a exp(-k x)`."""

    name = "monoexp"
    parameters = (
        ModelParameter("a", "[y]", description="value at x = 0"),
        ModelParameter("k", "1/[x]", description="rate constant"),
    )
    derived_units: ClassVar[dict[str, str]] = {"thalf": "[x]", "auc": "[y]*[x]"}

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve."""
        return p[0] * np.exp(-p[1] * x)

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """Half-life and area."""
        return {"thalf": LN2 / p[1], "auc": p[0] / p[1]}

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """From the log-linear regression of the positive points."""
        slope, intercept = log_linear_regression(x, y)
        if np.isfinite(slope) and slope < 0:
            return np.array([math.exp(intercept), -slope])
        a, k = terminal_guess(x, y)
        return np.array([a, k])


class _SumOfExponentials(Model):
    """Shared code of `BiExp` and `TriExp`."""

    #: number of phases
    phases: ClassVar[int] = 2

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The sum of the phases."""
        pairs = p.reshape(-1, 2)
        return np.sum(
            pairs[:, 0][:, None] * np.exp(-pairs[:, 1][:, None] * x[None, :]), axis=0
        )

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """`lambda_z`, the half-lives of the phases and the area.

        `lambda_z` is the smallest of the rate constants, not the rate of the
        last phase: the phases are ordered by decreasing rate after a fit, but
        not when the engine keeps the user's labelling (a fixed or bounded
        phase parameter), and the terminal rate is the slowest one either way.
        """
        pairs = p.reshape(-1, 2)
        out: dict[str, float] = {"lambda_z": float(np.min(pairs[:, 1]))}
        for i, (_, k) in enumerate(pairs, start=1):
            out[f"thalf_{i}"] = LN2 / k
        out["auc"] = float(np.sum(pairs[:, 0] / pairs[:, 1]))
        return out

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Curve stripping."""
        return _strip(x, y, self.phases)

    def parameter_order(self, p: np.ndarray) -> np.ndarray:
        """Permutation of the parameter vector that orders the phases by decreasing rate.

        The engine permutes the parameters, the bounds, the scales, the
        covariance and the columns of the Jacobian with it after a fit. It
        skips the permutation when `FitOptions.fixed` or `FitOptions.bounds`
        names one of the phase parameters: the labels are then the user's and
        the phases are reported as labelled, even if they are not ordered by
        decreasing rate.

        Args:
            p: phases as `[a1, k1, a2, k2, ...]`, any order.

        Returns:
            The indices of `p` which order the phases by decreasing rate
            constant.
        """
        pairs = p.reshape(-1, 2)
        order = np.argsort(-pairs[:, 1], kind="stable")
        return np.concatenate([[2 * i, 2 * i + 1] for i in order]).astype(np.intp)


class BiExp(_SumOfExponentials):
    """`y = a1 exp(-k1 x) + a2 exp(-k2 x)` with `k1 > k2`."""

    name = "biexp"
    phases = 2
    parameters = (
        ModelParameter("a1", "[y]", description="coefficient of the fast phase"),
        ModelParameter("k1", "1/[x]", description="rate constant of the fast phase"),
        ModelParameter("a2", "[y]", description="coefficient of the slow phase"),
        ModelParameter("k2", "1/[x]", description="rate constant of the slow phase"),
    )
    derived_units: ClassVar[dict[str, str]] = {
        "lambda_z": "1/[x]",
        "thalf_1": "[x]",
        "thalf_2": "[x]",
        "auc": "[y]*[x]",
    }


class TriExp(_SumOfExponentials):
    """`y = a1 exp(-k1 x) + a2 exp(-k2 x) + a3 exp(-k3 x)` with `k1 > k2 > k3`."""

    name = "triexp"
    phases = 3
    parameters = (
        ModelParameter("a1", "[y]", description="coefficient of the fastest phase"),
        ModelParameter("k1", "1/[x]", description="rate constant of the fastest phase"),
        ModelParameter("a2", "[y]", description="coefficient of the middle phase"),
        ModelParameter("k2", "1/[x]", description="rate constant of the middle phase"),
        ModelParameter("a3", "[y]", description="coefficient of the slowest phase"),
        ModelParameter("k3", "1/[x]", description="rate constant of the slowest phase"),
    )
    derived_units: ClassVar[dict[str, str]] = {
        "lambda_z": "1/[x]",
        "thalf_1": "[x]",
        "thalf_2": "[x]",
        "thalf_3": "[x]",
        "auc": "[y]*[x]",
    }


class Bateman(Model):
    """One compartment with first order absorption: `y = a ka/(ka - ke) (e^{-ke t} - e^{-ka t})`.

    `t = max(x - tlag, 0)` with the optional lag time. `a` is the dose over the
    apparent volume (`D F / V`), so `auc = a / ke`.

    Args:
        lag: whether a lag time `tlag` is fitted
    """

    name = "bateman"
    derived_units: ClassVar[dict[str, str]] = {
        "tmax": "[x]",
        "cmax": "[y]",
        "thalf": "[x]",
        "auc": "[y]*[x]",
        "flip_flop": "dimensionless",
    }

    def __init__(self, lag: bool = False) -> None:
        """Create the model with or without a lag time.

        Args:
            lag: whether a lag time `tlag` is fitted.
        """
        self.lag = lag
        base = (
            ModelParameter("a", "[y]", description="dose over the apparent volume"),
            ModelParameter("ka", "1/[x]", description="absorption rate constant"),
            ModelParameter("ke", "1/[x]", description="elimination rate constant"),
        )
        self.parameters = (
            (
                *base,
                ModelParameter(
                    "tlag", "[x]", lower=0.0, positive=False, description="lag time"
                ),
            )
            if lag
            else base
        )

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve, with the `ka == ke` limit `a ka t exp(-ke t)`."""
        a, ka, ke = p[0], p[1], p[2]
        t = np.maximum(x - (p[3] if self.lag else 0.0), 0.0)
        if abs(ka - ke) < 1e-9:
            return a * ka * t * np.exp(-ke * t)
        return a * ka / (ka - ke) * (np.exp(-ke * t) - np.exp(-ka * t))

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """Time and value of the maximum, half-life, area and the flip-flop indicator."""
        a, ka, ke = float(p[0]), float(p[1]), float(p[2])
        tlag = float(p[3]) if self.lag else 0.0
        tmax = (
            math.log(ka / ke) / (ka - ke) if abs(ka - ke) >= 1e-9 else 1.0 / ke
        ) + tlag
        cmax = float(self.predict(np.array([tmax]), p)[0])
        return {
            "tmax": tmax,
            "cmax": cmax,
            "thalf": LN2 / ke,
            "auc": a / ke,
            "flip_flop": 1.0 if ka < ke else 0.0,
        }

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """`ke` from the terminal phase, `ka` from the rise (or 5 ke), `a` from the maximum."""
        ok = np.isfinite(x) & np.isfinite(y)
        xs, ys = x[ok], y[ok]
        _, ke = terminal_guess(xs, ys)
        imax = int(np.argmax(ys))
        ka = 5.0 * ke
        if imax >= 2:
            rise = ys[: imax + 1]
            slope, _ = log_linear_regression(
                xs[: imax + 1], np.maximum(rise.max() - rise, 1e-12)
            )
            if np.isfinite(slope) and slope < 0 and -slope > ke:
                ka = -slope
        tmax = xs[imax]
        denominator = math.exp(-ke * tmax) - math.exp(-ka * tmax)
        a = ys[imax] * (ka - ke) / ka / denominator if denominator > 0 else ys[imax]
        guess = [a, ka, ke]
        if self.lag:
            guess.append(0.0)
        return np.array(guess, dtype=np.float64)
