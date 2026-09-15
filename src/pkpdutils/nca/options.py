"""Options and flags of the non-compartmental analysis.

`NCAOptions` selects the methods of an analysis: the kind of timecourse, the
trapezoid rule, the terminal phase selection, the handling of values below the
limit of quantification and the dosing intervals of a multiple dose analysis.
`NCAFlag` names the conditions an analysis reports per sample instead of
raising or warning.
"""

from enum import IntFlag, StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pkpdutils.result import decode_flags as decode_flag_names


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


class UncertaintyMethod(StrEnum):
    """How the uncertainty of group timecourses is propagated to the parameters."""

    #: no uncertainty variables
    NONE = "none"
    #: parametric bootstrap: resample every time point, analyse the replicates
    BOOTSTRAP = "bootstrap"
    #: delta method: numerical Jacobian of every parameter with respect to the values
    DELTA = "delta"


class BootstrapSpread(StrEnum):
    """Which spread the bootstrap resamples every time point with."""

    #: the standard error of the mean: the uncertainty of the group mean curve
    SE = "se"
    #: the standard deviation: the spread of individual curves
    SD = "sd"


class BootstrapDistribution(StrEnum):
    """Distribution the bootstrap draws every time point from."""

    #: normal with the given mean and spread; draws below 0 are set to 0
    NORMAL = "normal"
    #: log-normal with the same mean and spread; positive by construction
    LOGNORMAL = "lognormal"


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
    #: values below `lloq` were replaced
    BLQ_TRUNCATED = 32
    #: fewer than two valid points; every parameter is NaN
    NO_DATA = 64
    #: the delta method skipped points at which the terminal window changed; the
    #: uncertainty of the terminal parameters is incomplete
    DELTA_WINDOW_CHANGE = 128
    #: the last dosing interval is not covered by the data; its parameters and
    #: the steady state parameters are NaN
    INCOMPLETE_INTERVAL = 256
    #: the trough of at least one dosing interval of a bolus was extrapolated
    #: because the sample at the dose time carries the post-dose value
    EXTRAPOLATED_TROUGH = 512


def decode_flags(value: int) -> list[str]:
    """Names of the flags set in an integer flag value, in bit order.

    Args:
        value: an integer combination of `NCAFlag` values.

    Returns:
        The names of the set flags, in the declaration order of `NCAFlag`.
    """
    return decode_flag_names(NCAFlag, value)


class TerminalPhase(BaseModel):
    """Selection of the points of the terminal log-linear regression.

    Attributes:
        method: the selection rule
        min_points: minimal number of points of a regression (at least 3)
        exclude_cmax: whether the windows must start after the point of the
            maximum (`True`) or may start anywhere (`False`). It applies to
            `BEST_FIT` and to `LAST_N`, whose window then holds the points
            after the maximum when `n_points` reaches beyond it (fewer points
            than asked for, `NCAFlag.TOO_FEW_POINTS` below `min_points`); it
            does not apply to `MANUAL`, which regresses the given `points` as
            they are, and `ALL_AFTER_TMAX` starts after the maximum anyway
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
        tau: length of the last dosing interval, `None` to take it from the
            dosing protocol (the distance of the last two doses); it is needed
            for a steady state curve given with its last dose only and it
            overrides the protocol for the last interval
        intervals: whether the per-interval parameters (`interval_*`) are part
            of the result of a multiple dose analysis
        effect_threshold: threshold of `time_above` for effect timecourses, `None` for none
        n_workers: workers of the analysis. `None` is automatic: the calling
            thread up to `pkpdutils.parallel.NCA_WORKER_THRESHOLD` rows and
            one worker per usable core, at most 8, above it; `1` is always
            serial and `n > 1` uses that many workers. The
            core is vectorized numpy and releases the GIL, so its workers are
            threads of the calling process (`pkpdutils.parallel`) and no
            `if __name__ == "__main__":` guard is needed; the fit
            (`FitOptions.n_workers`) uses processes and does need one
        chunk_rows: most rows of a chunk of the vectorized core, which bounds
            its memory: a run holds the temporaries of as many chunks as run
            at once, `min(n_workers, n_chunks) * chunk_rows` rows. The chunks
            are mapped in order; how many there are follows from the rows, the
            workers and this bound (`pkpdutils.parallel.split_rows`), so a
            serial run of a small batch is one chunk whatever `n_workers` says
        uncertainty: propagation of `sd`/`se` to the parameters; `None` selects
            `BOOTSTRAP` when the batch carries an uncertainty and `NONE` otherwise
        n_boot: number of bootstrap replicates
        seed: seed of the bootstrap random generator; the default `None` draws
            from a fresh generator, so a bootstrap is not reproducible
        ci_level: level of the confidence intervals
        bootstrap_spread: whether the replicates are drawn with `se` or `sd`
        bootstrap_distribution: normal or log-normal draws
        delta_step: relative perturbation of a point, in units of its `se`, for the delta method
    """

    model_config = ConfigDict(frozen=True)

    kind: Kind = Kind.CONCENTRATION
    auc_method: AUCMethod = AUCMethod.LINEAR_LOG
    terminal: TerminalPhase = TerminalPhase()
    lloq: float | None = Field(default=None, gt=0.0)
    blq: BLQHandling = BLQHandling.NAN
    c0_method: C0Method = C0Method.LOG_BACK_EXTRAPOLATION
    extrapolation_warning: float = Field(default=0.2, gt=0.0, lt=1.0)
    tau: float | None = Field(default=None, gt=0.0)
    intervals: bool = True
    effect_threshold: float | None = None
    n_workers: int | None = Field(default=None, ge=1)
    chunk_rows: int = Field(default=5000, ge=1)
    uncertainty: UncertaintyMethod | None = None
    n_boot: int = Field(default=1000, ge=2)
    seed: int | None = None
    ci_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    bootstrap_spread: BootstrapSpread = BootstrapSpread.SE
    bootstrap_distribution: BootstrapDistribution = BootstrapDistribution.NORMAL
    delta_step: float = Field(default=0.01, gt=0.0, lt=1.0)

    def resolve_uncertainty(self, has_uncertainty: bool) -> UncertaintyMethod:
        """The uncertainty method of an analysis.

        Args:
            has_uncertainty: whether the batch carries `sd` or `se`

        Returns:
            `uncertainty` when set, else `BOOTSTRAP` for a batch with an
            uncertainty and `NONE` without.
        """
        if self.uncertainty is not None:
            return self.uncertainty
        return (
            UncertaintyMethod.BOOTSTRAP if has_uncertainty else UncertaintyMethod.NONE
        )
