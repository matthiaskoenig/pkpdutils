import numpy as np
import pytest

from pkpdutils.fit.models import (
    Allometric,
    Emax,
    Imax,
    Linear,
    LogLinear,
    Power,
    SigmoidEmax,
    SigmoidImax,
)

C = np.array([0.0, 0.5, 1, 2, 5, 10, 20, 50, 100])


def test_emax_family() -> None:
    m = Emax()
    assert m.parameter_names == ("e0", "emax", "ec50")
    assert (
        not m.parameter("e0").positive
        and not m.parameter("emax").positive
        and m.parameter("ec50").positive
    )
    p = np.array([2.0, 10.0, 5.0])
    y = m.predict(C, p)
    np.testing.assert_allclose(y, 2 + 10 * C / (5 + C))
    assert m.derived(p) == {"ec90": 45.0}
    guess = m.initial_guess(C, y)
    assert guess[0] == pytest.approx(2.0, abs=0.5)
    assert guess[1] == pytest.approx(10.0, rel=0.3)
    assert guess[2] == pytest.approx(5.0, rel=0.5)

    s = SigmoidEmax()
    assert s.parameter_names == ("e0", "emax", "ec50", "hill")
    p = np.array([0.0, 1.0, 10.0, 2.0])
    np.testing.assert_allclose(s.predict(C, p), C**2 / (100 + C**2))
    assert s.derived(p)["ec90"] == pytest.approx(3.0 * 10.0)
    assert s.initial_guess(C, s.predict(C, p))[3] == 1.0


def test_imax_family() -> None:
    m = Imax()
    assert m.parameter_names == ("e0", "imax", "ic50")
    p = np.array([10.0, 0.8, 4.0])
    y = m.predict(C, p)
    np.testing.assert_allclose(y, 10 * (1 - 0.8 * C / (4 + C)))
    assert m.derived(p) == {"ic90": 36.0}
    guess = m.initial_guess(C, y)
    assert guess[0] == pytest.approx(10.0, rel=0.1)
    assert 0.05 <= guess[1] <= 0.95
    assert guess[2] == pytest.approx(4.0, rel=0.6)
    assert m.parameter("imax").upper == 1.0 and m.parameter("imax").lower == 0.0
    s = SigmoidImax()
    assert s.parameter_names == ("e0", "imax", "ic50", "hill")
    np.testing.assert_allclose(
        s.predict(C, np.array([10.0, 1.0, 4.0, 1.0])), 10 * (1 - C / (4 + C))
    )


def test_linear_models() -> None:
    lin = Linear()
    np.testing.assert_allclose(lin.predict(C, np.array([1.0, 2.0])), 1 + 2 * C)
    np.testing.assert_allclose(lin.initial_guess(C, 1 + 2 * C), [1.0, 2.0])
    log = LogLinear()
    x = C[1:]
    np.testing.assert_allclose(log.predict(x, np.array([1.0, 2.0])), 1 + 2 * np.log(x))
    np.testing.assert_allclose(log.initial_guess(x, 1 + 2 * np.log(x)), [1.0, 2.0])
    assert (
        lin.parameter("slope").unit_expr == "[y]/[x]"
        and log.parameter("slope").unit_expr == "[y]"
    )


def test_power_and_allometric() -> None:
    pw = Power()
    assert pw.parameter_names == ("a", "b") and not pw.parameter("b").positive
    x = C[1:]
    np.testing.assert_allclose(pw.predict(x, np.array([3.0, 0.8])), 3 * x**0.8)
    np.testing.assert_allclose(pw.initial_guess(x, 3 * x**0.8), [3.0, 0.8])
    free = Allometric()
    assert free.parameter_names == ("a", "b")
    assert free.initial_guess(x, 3 * x**0.75)[1] == pytest.approx(0.75)
    fixed = Allometric(exponent=0.75)
    assert fixed.parameter_names == ("a",)
    np.testing.assert_allclose(fixed.predict(x, np.array([3.0])), 3 * x**0.75)
    assert fixed.initial_guess(x, 3 * x**0.75)[0] == pytest.approx(3.0)
    assert fixed.exponent == 0.75
    assert fixed.name == "allometric_0.75" and free.name == "allometric"
