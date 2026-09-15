import numpy as np
import pytest

from pkpdutils.fit.models import Bateman, BiExp, MonoExp, TriExp
from pkpdutils.fit.models_exponential import log_linear_regression, terminal_guess

T = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12, 24])


def test_log_linear_regression() -> None:
    slope, intercept = log_linear_regression(T, 5.0 * np.exp(-0.3 * T))
    assert slope == pytest.approx(-0.3)
    assert intercept == pytest.approx(np.log(5.0))
    s, i = log_linear_regression(np.array([1.0]), np.array([2.0]))
    assert np.isnan(s) and np.isnan(i)
    s, _ = log_linear_regression(T, np.where(T > 4, 0.0, 5.0 * np.exp(-0.3 * T)))
    assert s == pytest.approx(-0.3)


def test_monoexp() -> None:
    m = MonoExp()
    assert m.name == "monoexp" and m.parameter_names == ("a", "k")
    p = np.array([5.0, 0.3])
    np.testing.assert_allclose(m.predict(T, p), 5.0 * np.exp(-0.3 * T))
    d = m.derived(p)
    assert d["thalf"] == pytest.approx(np.log(2) / 0.3) and d["auc"] == pytest.approx(
        5.0 / 0.3
    )
    assert m.derived_units == {"thalf": "[x]", "auc": "[y]*[x]"}
    guess = m.initial_guess(T, 5.0 * np.exp(-0.3 * T))
    np.testing.assert_allclose(guess, p, rtol=1e-6)


def test_biexp_predict_strip_and_sort() -> None:
    m = BiExp()
    assert m.parameter_names == ("a1", "k1", "a2", "k2")
    p = np.array([8.0, 2.0, 2.0, 0.2])
    y = m.predict(T, p)
    np.testing.assert_allclose(y, 8 * np.exp(-2 * T) + 2 * np.exp(-0.2 * T))
    guess = m.initial_guess(T, y)
    assert guess[3] == pytest.approx(0.2, rel=0.3)
    assert guess[1] > guess[3]
    d = m.derived(p)
    assert d["lambda_z"] == pytest.approx(0.2) and d["thalf_2"] == pytest.approx(
        np.log(2) / 0.2
    )
    assert d["auc"] == pytest.approx(8 / 2 + 2 / 0.2)
    swapped = m.sort_parameters(np.array([2.0, 0.2, 8.0, 2.0]))
    np.testing.assert_allclose(swapped, p)


def test_triexp() -> None:
    m = TriExp()
    assert m.parameter_names == ("a1", "k1", "a2", "k2", "a3", "k3")
    p = np.array([10.0, 5.0, 4.0, 1.0, 1.0, 0.1])
    y = m.predict(T, p)
    guess = m.initial_guess(T, y)
    assert (
        guess.shape == (6,)
        and np.all(np.isfinite(guess))
        and guess[5] < guess[3] < guess[1]
    )
    assert m.derived(p)["lambda_z"] == pytest.approx(0.1)
    np.testing.assert_allclose(
        m.sort_parameters(np.array([1.0, 0.1, 10.0, 5.0, 4.0, 1.0])), p
    )


def test_bateman_with_and_without_lag() -> None:
    m = Bateman()
    assert m.parameter_names == ("a", "ka", "ke")
    p = np.array([10.0, 2.0, 0.3])
    y = m.predict(T, p)
    expected = 10 * 2 / (2 - 0.3) * (np.exp(-0.3 * T) - np.exp(-2 * T))
    np.testing.assert_allclose(y, expected)
    d = m.derived(p)
    tmax = np.log(2 / 0.3) / (2 - 0.3)
    assert d["tmax"] == pytest.approx(tmax)
    assert d["cmax"] == pytest.approx(
        10 * 2 / 1.7 * (np.exp(-0.3 * tmax) - np.exp(-2 * tmax))
    )
    assert d["thalf"] == pytest.approx(np.log(2) / 0.3) and d["auc"] == pytest.approx(
        10 / 0.3
    )
    assert d["flip_flop"] == 0.0
    assert m.derived(np.array([10.0, 0.2, 0.5]))["flip_flop"] == 1.0
    guess = m.initial_guess(T, y)
    assert guess[2] == pytest.approx(0.3, rel=0.2)
    assert guess[1] > guess[2]
    lag = Bateman(lag=True)
    assert lag.parameter_names == ("a", "ka", "ke", "tlag")
    assert not lag.parameter("tlag").positive and lag.parameter("tlag").lower == 0.0
    y_lag = lag.predict(T, np.array([10.0, 2.0, 0.3, 1.0]))
    assert y_lag[T <= 1.0].max() == 0.0
    np.testing.assert_allclose(
        y_lag[T > 1.0],
        10 * 2 / 1.7 * (np.exp(-0.3 * (T[T > 1] - 1)) - np.exp(-2 * (T[T > 1] - 1))),
    )
    assert lag.derived(np.array([10.0, 2.0, 0.3, 1.0]))["tmax"] == pytest.approx(
        tmax + 1.0
    )


def test_bateman_ka_equals_ke_limit() -> None:
    m = Bateman()
    y = m.predict(np.array([1.0, 2.0]), np.array([10.0, 0.5, 0.5]))
    np.testing.assert_allclose(
        y, 10 * 0.5 * np.array([1.0, 2.0]) * np.exp(-0.5 * np.array([1.0, 2.0]))
    )


def test_terminal_guess() -> None:
    a, k = terminal_guess(T, 5.0 * np.exp(-0.3 * T))
    assert k == pytest.approx(0.3) and a == pytest.approx(5.0)
