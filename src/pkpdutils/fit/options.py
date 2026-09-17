"""Options and flags of the curve fitting."""

from enum import IntFlag, StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pkpdutils.fit.model import ModelParameter
from pkpdutils.result import decode_flags as decode_flag_names

#: loss functions of `scipy.optimize.least_squares`
LOSSES: tuple[str, ...] = ("linear", "soft_l1", "huber", "cauchy", "arctan")


class ParameterScale(StrEnum):
    """Space the optimizer searches; bounds, start values and results stay linear."""

    #: `log10(p)` for positive parameters
    LOG10 = "log10"
    #: `ln(p)` for positive parameters
    LOG = "log"
    #: no transformation
    LINEAR = "linear"


class Weighting(StrEnum):
    """Variance model of the residuals; the weighted residual is `(y - f) / sqrt(var)`."""

    #: constant variance
    NONE = "none"
    #: variance proportional to `y`
    INV_Y = "inv_y"
    #: variance proportional to `y²` (constant CV)
    INV_Y2 = "inv_y2"
    #: variance `sd²` from the data
    INV_SD = "inv_sd"


class FitFlag(IntFlag):
    """Conditions reported per sample in the `flags` variable of a fit result."""

    NONE = 0
    #: the optimizer did not report convergence from any start
    NOT_CONVERGED = 1
    #: a parameter ended within `at_bound_tolerance` of a bound
    AT_BOUND = 2
    #: fewer points than parameters + 1
    TOO_FEW_POINTS = 4
    #: the absorption rate is smaller than the elimination rate (Bateman)
    FLIP_FLOP = 8
    #: the Jacobian is singular, no standard errors
    SINGULAR = 16
    #: fewer than two finite points
    NO_DATA = 32
    #: fewer than two bootstrap replicates converged; the reported
    #: uncertainties are the Jacobian ones
    BOOTSTRAP_FALLBACK = 64


def decode_fit_flags(value: int) -> list[str]:
    """Names of the flags set in an integer value, in bit order.

    Args:
        value: integer value of a `FitFlag` combination.

    Returns:
        The names of the flags set in `value`, in bit order.
    """
    return decode_flag_names(FitFlag, value)


class FitOptions(BaseModel):
    """Options of a fit.

    Attributes:
        parameter_scale: space of the search for positive parameters
        weighting: variance model of the residuals
        loss: loss function of `scipy.optimize.least_squares`; the covariance,
            the standard errors and the confidence intervals of a fit are the
            least-squares quantities and are exact only for `"linear"`, for a
            robust loss they are approximations
        n_starts: number of start points (Latin hypercube in the start box)
        seed: seed of the start point sampling and the bootstrap
        n_workers: worker processes of a batch fit, one row per job (never the
            starts of a single row). `None` is automatic: the calling process
            up to 2 000 rows, where a batch does not earn back the start-up of
            the workers, and one worker per core, at most 8, above it; `1` is
            always serial and `n > 1` uses that many workers, which is how a
            smaller batch of expensive rows (several starts, a residual
            bootstrap) asks for the pool. A pooled call must run under an
            `if __name__ == "__main__":` guard, since python's `spawn` and
            `forkserver` process start methods (the default on macOS and
            Windows, and on Linux from python 3.14) re-import the module
            without re-running it; the NCA (`NCAOptions.n_workers`) runs in
            threads and needs no guard
        ci_level: level of the confidence intervals
        bootstrap: number of residual bootstrap replicates, 0 for none
        max_nfev: maximal function evaluations per start, `None` for the scipy default
        ftol: scipy `ftol`
        xtol: scipy `xtol`
        gtol: scipy `gtol`
        fixed: parameters held at a value (not fitted)
        bounds: bounds overriding the model's, per parameter
        initial: start values overriding the model's guess, per parameter
        start_spread: half width of the start box around the initial guess, as a
            factor for log scale parameters and as a multiple of the guess for linear ones
        at_bound_tolerance: distance to a bound that sets `AT_BOUND`. A
            parameter `p` which started at `p0` rests on a bound `b` when
            `|p - b| <= at_bound_tolerance * (|b| + max(|p0|, 1e-300))`, so
            the distance is relative to the bound and to the start value. The
            start value is needed for a lower bound of 0, which is `-inf` in a
            log search space and can never be reached exactly.
    """

    model_config = ConfigDict(frozen=True)

    parameter_scale: ParameterScale = ParameterScale.LOG10
    weighting: Weighting = Weighting.NONE
    loss: str = "linear"
    n_starts: int = Field(default=1, ge=1)
    seed: int | None = None
    n_workers: int | None = Field(default=None, ge=1)
    ci_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    bootstrap: int = Field(default=0, ge=0)
    max_nfev: int | None = Field(default=None, ge=1)
    ftol: float = Field(default=1e-10, gt=0.0)
    xtol: float = Field(default=1e-10, gt=0.0)
    gtol: float = Field(default=1e-10, gt=0.0)
    fixed: dict[str, float] = Field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    initial: dict[str, float] = Field(default_factory=dict)
    start_spread: float = Field(default=100.0, gt=1.0)
    at_bound_tolerance: float = Field(default=1e-3, gt=0.0)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """Check the loss name and the ordering of the bound overrides.

        Returns:
            `self`, unchanged.

        Raises:
            ValueError: for an unknown `loss` or an inverted bound in `bounds`.
        """
        if self.loss not in LOSSES:
            raise ValueError(f"'loss' must be one of {LOSSES}, not '{self.loss}'")
        for name, (lower, upper) in self.bounds.items():
            if upper <= lower:
                raise ValueError(f"bounds of '{name}': upper {upper} <= lower {lower}")
        return self

    def scale_of(self, parameter: ModelParameter) -> ParameterScale:
        """The scale a parameter is searched on: linear unless it is positive.

        Args:
            parameter: the model parameter.

        Returns:
            `ParameterScale.LINEAR` for a non-positive parameter, else `parameter_scale`.
        """
        if not parameter.positive:
            return ParameterScale.LINEAR
        return self.parameter_scale
