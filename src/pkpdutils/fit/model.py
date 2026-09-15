"""Models of the curve fitting.

A `Model` names its parameters (`ModelParameter`: name, unit expression,
bounds, whether it is positive and therefore fitted on the log scale) and
computes the curve `predict(x, p)`, the derived parameters `derived(p)` and an
`initial_guess(x, y)`. The unit of a parameter is derived from the units of `x`
and `y` with `parameter_unit_expression`, e.g. `"[y]/[x]"` for a slope.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from pkpdutils.units import ureg


@dataclass(frozen=True)
class ModelParameter:
    """A parameter of a model.

    Attributes:
        name: name of the parameter, the variable name in the result
        unit_expr: unit expression with `[x]` and `[y]` for the units of the data,
            e.g. `"[y]"`, `"1/[x]"`, `"[y]/[x]"`, `"dimensionless"`
        lower: lower bound (linear scale)
        upper: upper bound (linear scale)
        positive: whether the parameter is positive and fitted on the log scale
        description: one line for the documentation
    """

    name: str
    unit_expr: str
    lower: float = -math.inf
    upper: float = math.inf
    positive: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        """Check the bounds and clip a negative lower bound of a positive parameter."""
        if self.upper <= self.lower:
            raise ValueError(
                f"'{self.name}': upper bound {self.upper} <= lower bound {self.lower}"
            )
        if self.positive and self.lower < 0:
            object.__setattr__(self, "lower", 0.0)


class Model(ABC):
    """A curve `y = f(x; p)` with named parameters.

    Subclasses set `name`, `parameters` (and optionally `derived_units`) and
    implement `predict`, `initial_guess` and, when they report derived
    parameters, `derived`. Parameters are passed as a 1-D array in the order
    of `parameters`.
    """

    #: name of the model in results and tables
    name: str = "model"
    #: the parameters, in the order of the parameter vector
    parameters: tuple[ModelParameter, ...] = ()
    #: unit expressions of the derived parameters
    derived_units: ClassVar[dict[str, str]] = {}

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the parameters in order."""
        return tuple(p.name for p in self.parameters)

    @property
    def n_parameters(self) -> int:
        """Number of parameters."""
        return len(self.parameters)

    def parameter(self, name: str) -> ModelParameter:
        """The parameter of a name.

        Args:
            name: name of the parameter.

        Returns:
            The `ModelParameter` of that name.

        Raises:
            KeyError: for an unknown name.
        """
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"{self.name} has no parameter '{name}'")

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds as arrays.

        Returns:
            A `(lower, upper)` tuple of arrays in the order of `parameters`.
        """
        return (
            np.array([p.lower for p in self.parameters], dtype=np.float64),
            np.array([p.upper for p in self.parameters], dtype=np.float64),
        )

    @abstractmethod
    def predict(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """The curve at `x` for the parameters `p`.

        Args:
            x: independent variable, vectorized.
            p: parameters in the order of `parameters`.

        Returns:
            The predicted values at `x`.
        """

    def derived(self, p: np.ndarray) -> dict[str, float]:
        """Derived parameters of `p`, empty by default.

        Args:
            p: parameters in the order of `parameters`.

        Returns:
            A mapping of derived parameter name to value.
        """
        return {}

    @abstractmethod
    def initial_guess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """A start vector from the data (finite points only).

        Args:
            x: independent variable.
            y: dependent variable.

        Returns:
            A start vector in the order of `parameters`.
        """

    def __repr__(self) -> str:
        """`Model(name, parameters)`."""
        return f"{type(self).__name__}({self.name}, {', '.join(self.parameter_names)})"


def parameter_unit_expression(unit_expr: str, *, x_unit: str, y_unit: str) -> str:
    """Unit of a parameter from its expression and the units of the data.

    The unit is the raw combination of the units of the data, without any
    normalization: a fit reports its parameters in the units the data was
    given in (`a` of a monoexponential fit of milliliters is in milliliter,
    its `auc` in milliliter hour), so the parameters and the predicted curve
    always live on the same scale.

    Args:
        unit_expr: expression with `[x]` and `[y]`, e.g. `"[y]/[x]"`
        x_unit: unit of the independent variable
        y_unit: unit of the dependent variable

    Returns:
        The unit string of the parameter.
    """
    raw = unit_expr.replace("[x]", f"({x_unit})").replace("[y]", f"({y_unit})")
    return str(ureg.parse_units(raw))
