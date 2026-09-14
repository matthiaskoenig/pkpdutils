"""Options and flags of the non-compartmental analysis.

`NCAOptions` selects the methods of an analysis: the kind of timecourse, the
trapezoid rule, the terminal phase selection, the handling of values below the
limit of quantification and the dosing regimen of a steady state analysis.
`NCAFlag` names the conditions an analysis reports per sample instead of
raising or warning.
"""

from enum import IntFlag, StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pkpdutils.timecourse import DosingRegimen


class Kind(StrEnum):
    """What a timecourse measures."""

    #: concentration of a substance: exposure, terminal phase and dose parameters
    CONCENTRATION = "concentration"
    #: pharmacodynamic effect: AUEC, observed maximum, baseline
    EFFECT = "effect"


class AUCMethod(StrEnum):
    """Trapezoid rule of the areas, see `docs/nca.md`."""

    #: linear trapezoid on every segment
    LINEAR = "linear"
    #: linear on rising segments, logarithmic on falling segments (Phoenix "linear up/log down")
    LINEAR_LOG = "linear_log"
    #: logarithmic on every segment with two positive, different values
    LOG = "log"


class TerminalMethod(StrEnum):
    """Selection of the points of the terminal log-linear regression."""

    #: largest adjusted R² over all windows ending at tlast (Phoenix best fit)
    BEST_FIT = "best_fit"
    #: the last `n_points` points
    LAST_N = "last_n"
    #: every point after the maximum (the rule of pkdb_analysis 0.3.1)
    ALL_AFTER_TMAX = "all_after_tmax"
    #: the given point indices
    MANUAL = "manual"


class BLQHandling(StrEnum):
    """Handling of values below the lower limit of quantification."""

    #: values below `lloq` are missing
    NAN = "nan"
    #: values below `lloq` are 0 before tmax and missing after
    ZERO_BEFORE_TMAX = "zero_before_tmax"


class C0Method(StrEnum):
    """Estimate of the concentration at time 0 after an intravenous bolus."""

    #: log-linear back extrapolation of the first two positive values
    LOG_BACK_EXTRAPOLATION = "log_back_extrapolation"
    #: the first observed value
    FIRST_VALUE = "first_value"


class NCAFlag(IntFlag):
    """Conditions reported per sample in the `flags` variable of a result."""

    NONE = 0
    #: the terminal regression has a non-negative slope; lambda_z and dependents are NaN
    POSITIVE_SLOPE = 1
    #: fewer than `min_points` points after the maximum; no terminal phase
    TOO_FEW_POINTS = 2
    #: the extrapolated fraction of AUC(0-inf) exceeds `extrapolation_warning`
    EXTRAPOLATION_HIGH = 4
    #: the maximum is the last point of the curve
    NO_MAX = 8
    #: the maximum is the first point of an extravascular curve
    NO_ABSORPTION = 16
    #: values below `lloq` were removed
    BLQ_TRUNCATED = 32
    #: fewer than two valid points; every parameter is NaN
    NO_DATA = 64


def decode_flags(value: int) -> list[str]:
    """Names of the flags set in an integer flag value, in bit order.

    Args:
        value: an integer combination of `NCAFlag` values.

    Returns:
        The names of the set flags, in the declaration order of `NCAFlag`.
    """
    return [flag.name for flag in NCAFlag if flag.value and value & flag.value]


class TerminalPhase(BaseModel):
    """Selection of the points of the terminal log-linear regression.

    Attributes:
        method: the selection rule
        min_points: minimal number of points of a regression (at least 3)
        exclude_cmax: whether the point of the maximum is excluded from every window
        n_points: number of points for `LAST_N`
        points: indices of the points (in the time order of the curve) for `MANUAL`
        min_adj_r2: minimal adjusted R² a regression must reach, `None` for no limit
        tie_tolerance: a window with more points wins over the best adjusted R²
            when its adjusted R² is within this tolerance of the best
    """

    model_config = ConfigDict(frozen=True)

    method: TerminalMethod = TerminalMethod.BEST_FIT
    min_points: int = Field(default=3, ge=3)
    exclude_cmax: bool = True
    n_points: int | None = Field(default=None, ge=3)
    points: tuple[int, ...] | None = None
    min_adj_r2: float | None = Field(default=None, ge=0.0, le=1.0)
    tie_tolerance: float = Field(default=1e-4, ge=0.0)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """Check the fields required by `method` are set consistently.

        Returns:
            This instance, unchanged.

        Raises:
            ValueError: `LAST_N` without `n_points`, or `MANUAL` without
                `points` or with fewer `points` than `min_points`.
        """
        if self.method is TerminalMethod.LAST_N and self.n_points is None:
            raise ValueError("TerminalMethod.LAST_N needs 'n_points'")
        if self.method is TerminalMethod.MANUAL:
            if self.points is None:
                raise ValueError("TerminalMethod.MANUAL needs 'points'")
            if len(self.points) < self.min_points:
                raise ValueError(
                    f"'points' has {len(self.points)} indices, 'min_points' is {self.min_points}"
                )
        return self


class NCAOptions(BaseModel):
    """Options of a non-compartmental analysis.

    Attributes:
        kind: concentration or effect timecourses
        auc_method: trapezoid rule of the areas
        terminal: selection of the terminal phase
        lloq: lower limit of quantification in the unit of the values, `None` for none
        blq: handling of values below `lloq`
        c0_method: estimate of C(0) after an intravenous bolus
        extrapolation_warning: fraction of AUC(0-inf) above which `EXTRAPOLATION_HIGH` is set
        regimen: dosing regimen of a steady state analysis, `None` for single dose
        effect_threshold: threshold of `time_above` for effect timecourses, `None` for none
        n_workers: number of worker processes for large batches, `None` for the calling process
    """

    model_config = ConfigDict(frozen=True)

    kind: Kind = Kind.CONCENTRATION
    auc_method: AUCMethod = AUCMethod.LINEAR_LOG
    terminal: TerminalPhase = TerminalPhase()
    lloq: float | None = Field(default=None, gt=0.0)
    blq: BLQHandling = BLQHandling.NAN
    c0_method: C0Method = C0Method.LOG_BACK_EXTRAPOLATION
    extrapolation_warning: float = Field(default=0.2, gt=0.0, lt=1.0)
    regimen: DosingRegimen | None = None
    effect_threshold: float | None = None
    n_workers: int | None = Field(default=None, ge=1)
