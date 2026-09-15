"""Tests of the linear, log-linear, power and allometric models."""

import numpy as np
import pytest

from pkpdutils.fit import FitFlag, fit
from pkpdutils.fit.models import Allometric, Linear, LogLinear, Power

X = np.array([1.0, 2.0, 4.0, 8.0])
NEGATIVE = np.array([-1.0, -2.0, -4.0, -8.0])


def test_linear_guess() -> None:
    guess = Linear().initial_guess(X, 3.0 + 2.0 * X)
    np.testing.assert_allclose(guess, [3.0, 2.0], atol=1e-9)


def test_loglinear_and_power_guesses() -> None:
    np.testing.assert_allclose(
        LogLinear().initial_guess(X, 3.0 + 2.0 * np.log(X)), [3.0, 2.0], atol=1e-9
    )
    np.testing.assert_allclose(
        Power().initial_guess(X, 3.0 * X**2.0), [3.0, 2.0], rtol=1e-9
    )
    np.testing.assert_allclose(
        Allometric().initial_guess(X, 3.0 * X**0.75), [3.0, 0.75], rtol=1e-9
    )
    np.testing.assert_allclose(
        Allometric(exponent=0.75).initial_guess(X, 3.0 * X**0.75), [3.0], rtol=1e-9
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (LogLinear(), [0.0, 0.0]),
        (Power(), [1.0, 1.0]),
        (Allometric(), [1.0, 0.75]),
        (Allometric(exponent=0.75), [1.0]),
    ],
)
def test_initial_guess_without_a_positive_x(
    model: LogLinear | Power | Allometric, expected: list[float]
) -> None:
    """A model on `ln x` has no usable point when every `x` is negative, and must not warn.

    `numpy.mean` of the empty selection would raise a `RuntimeWarning` ("mean
    of empty slice") that aborts the whole batch under a strict warning
    filter, so the guess falls back to a neutral vector instead.
    """
    guess = model.initial_guess(NEGATIVE, np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(guess, expected)


def test_fit_of_negative_x_returns_a_result() -> None:
    """A power fit of negative `x` has no usable guess and fails cleanly, without a warning."""
    result = fit(Power(), NEGATIVE, np.array([1.0, 2.0, 3.0, 4.0]))
    assert result.flags() == [str(FitFlag.NOT_CONVERGED.name)]
    assert np.isnan(float(result["a"].values))
    assert int(result["n_points"].values) == 4
