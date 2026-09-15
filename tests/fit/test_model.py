import numpy as np
import pytest

from pkpdutils.fit.model import Model, ModelParameter, parameter_unit_expression


class Line(Model):
    name = "line"
    parameters = (
        ModelParameter("a", "[y]", positive=False, description="intercept"),
        ModelParameter("b", "[y]/[x]", lower=0.0, description="slope"),
    )
    derived_units = {"x_zero": "[x]"}  # noqa: RUF012

    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] + p[1] * x

    def derived(self, p: np.ndarray) -> dict[str, float]:
        return {"x_zero": -p[0] / p[1]}

    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array([y[0], (y[-1] - y[0]) / (x[-1] - x[0])])


def test_model_metadata() -> None:
    m = Line()
    assert m.parameter_names == ("a", "b")
    assert m.n_parameters == 2
    assert m.parameter("b").lower == 0.0
    lower, upper = m.bounds()
    np.testing.assert_array_equal(lower, [-np.inf, 0.0])
    np.testing.assert_array_equal(upper, [np.inf, np.inf])
    with pytest.raises(KeyError):
        m.parameter("c")
    assert "line" in repr(m)


def test_model_predict_and_derived() -> None:
    m = Line()
    p = np.array([1.0, 2.0])
    np.testing.assert_allclose(m.predict(np.array([0.0, 1.0]), p), [1.0, 3.0])
    assert m.derived(p) == {"x_zero": -0.5}
    np.testing.assert_allclose(
        m.initial_guess(np.array([0.0, 2.0]), np.array([1.0, 5.0])), [1.0, 2.0]
    )


def test_parameter_unit_expression() -> None:
    unit, factor = parameter_unit_expression("[y]/[x]", x_unit="hr", y_unit="mg/l")
    assert unit == "milligram / hour / liter"
    assert factor == pytest.approx(1.0)
    unit, factor = parameter_unit_expression("1/[x]", x_unit="min", y_unit="mg/l")
    assert unit == "1 / minute"
    assert (
        parameter_unit_expression("dimensionless", x_unit="hr", y_unit="mg/l")[0]
        == "dimensionless"
    )
    unit, factor = parameter_unit_expression("[y]*[x]", x_unit="hr", y_unit="ng/ml")
    assert unit == "hour * nanogram / milliliter"
