"""Options and flags of the non-compartmental analysis.

`NCAOptions` selects the methods of an analysis: the kind of timecourse, the
trapezoid rule, the terminal phase selection, the handling of values below the
limit of quantification (`BLQRules`, one rule per position of the curve) and
the dosing intervals of a multiple dose analysis. `NCAFlag` names the
conditions an analysis reports per sample instead of raising or warning.
"""

from enum import IntFlag, StrEnum
from typing import Any, Self

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


class BLQAction(StrEnum):
    """What happens to a value below the lower limit of quantification.

    The action of a position of the curve (`BLQRules`); a `float` in place of
    a member imputes that number. `DROP` and `KEEP` leave no imputed value
    behind, every other action writes one, which enters the areas and, unless
    `BLQRules.terminal_regression`, stays out of the terminal regression.
    """

    #: the value is missing, as if it had not been measured
    DROP = "drop"
    #: the measured value below the limit is kept as it is
    KEEP = "keep"
    #: the value is 0
    ZERO = "zero"
    #: the value is the limit of quantification
    LLOQ = "lloq"
    #: the value is half the limit of quantification
    HALF_LLOQ = "half_lloq"


class C0Method(StrEnum):
    """Estimate of the concentration at time 0 after an intravenous bolus."""

    #: log-linear back extrapolation of the first two positive values
    LOG_BACK_EXTRAPOLATION = "log_back_extrapolation"
    #: the first observed value
    FIRST_VALUE = "first_value"
    #: no estimate: `c0` is `NaN` and the areas start at the first sample
    NONE = "none"


#: `c0_method` of a row whose `C0` was not estimated (`C0Method.NONE`, no
#: bolus, no data)
C0_NONE: int = 0
#: `c0_method` of a row whose `C0` is the log-linear back extrapolation
C0_BACK_EXTRAPOLATION: int = 1
#: `c0_method` of a row whose `C0` is the first observed value
C0_FIRST_VALUE: int = 2


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
    #: values below `lloq` were dropped or imputed (`BLQRules`)
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
    #: the terminal phase spans fewer than two half-lives
    #: (`lambda_z_span < 2`); lambda_z and its half-life are poorly determined
    SPAN_LOW = 1024
    #: a threshold of `Acceptance` is not met; the sample is not accepted
    NOT_ACCEPTED = 2048
    #: a named partial area (`NCAOptions.partial_aucs`) reaches beyond the last
    #: measurable value and was completed with the terminal regression
    PARTIAL_EXTRAPOLATED = 4096


def decode_flags(value: int) -> list[str]:
    """Names of the flags set in an integer flag value, in bit order.

    Args:
        value: an integer combination of `NCAFlag` values.

    Returns:
        The names of the set flags, in the declaration order of `NCAFlag`.
    """
    return decode_flag_names(NCAFlag, value)


class BLQRules(BaseModel):
    r"""Rules for the values below the lower limit of quantification, by position.

    The tools slice a profile on two incompatible axes and a rule set is
    expressed on one of them, never on both (the model raises for a mixture):

    - the **positional** axis `first`, `middle`, `last`: the values before the
      first measurable value, between two measurable values and after the last
      measurable value (PKNCA `conc.blq` with `"first"`/`"middle"`/`"last"`,
      Pumas `Dict(:first => :keep, :middle => :drop, :last => :keep)`);
    - the **tmax** axis `before_tmax`, `after_tmax`, split at the maximum of
      the measurable values (PKNCA `"before.tmax"`/`"after.tmax"`, PKanalix,
      which imputes 0 before and `LLOQ/2` after the maximum).

    A rule is a `BLQAction` or a number, which is imputed as it is; a position
    without a rule drops its values. A row whose values are all below the limit
    has no measurable value: every value of it counts as `first` on the
    positional axis and as `after_tmax` on the tmax axis.

    An imputed value enters the areas (`auc_all` reports what the imputation
    added to the tail) and stays out of the terminal regression unless
    `terminal_regression` is set; a value which `BLQAction.KEEP` keeps is
    treated the same way, since a value below the limit of quantification is
    not a quantified value. ICH M13A (2024) asks for exactly that: values below
    the limit are "treated as zero in PK parameter calculations" and "omitted
    from the calculation of kel and t1/2" (`BLQRules.ich_m13a`).

    Attributes:
        first: rule for the values before the first measurable value
        middle: rule for the values between two measurable values
        last: rule for the values after the last measurable value
        before_tmax: rule for the values before the maximum
        after_tmax: rule for the values at or after the maximum
        terminal_regression: whether an imputed or kept value below the limit
            may enter the terminal regression
    """

    model_config = ConfigDict(frozen=True)

    first: BLQAction | float | None = None
    middle: BLQAction | float | None = None
    last: BLQAction | float | None = None
    before_tmax: BLQAction | float | None = None
    after_tmax: BLQAction | float | None = None
    terminal_regression: bool = False

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """Check that only one of the two axes carries rules.

        Returns:
            This instance, unchanged.

        Raises:
            ValueError: if a positional rule and a tmax rule are both given.
        """
        positional = [
            name
            for name in ("first", "middle", "last")
            if getattr(self, name) is not None
        ]
        by_tmax = [
            name
            for name in ("before_tmax", "after_tmax")
            if getattr(self, name) is not None
        ]
        if positional and by_tmax:
            raise ValueError(
                f"the positional rules {positional} and the tmax rules {by_tmax} "
                "are two axes of the same values, give one of them"
            )
        return self

    @property
    def by_tmax(self) -> bool:
        """Whether the rules split the curve at the maximum instead of by position."""
        return self.before_tmax is not None or self.after_tmax is not None

    @classmethod
    def from_handling(cls, handling: BLQHandling) -> "BLQRules":
        """The rules of one of the two classic `BLQHandling` values.

        Args:
            handling: `BLQHandling.NAN` or `BLQHandling.ZERO_BEFORE_TMAX`.

        Returns:
            `first=middle=last=DROP` for `NAN` and `before_tmax=ZERO`,
            `after_tmax=DROP` for `ZERO_BEFORE_TMAX`.
        """
        if handling is BLQHandling.ZERO_BEFORE_TMAX:
            return cls(before_tmax=BLQAction.ZERO, after_tmax=BLQAction.DROP)
        return cls(first=BLQAction.DROP, middle=BLQAction.DROP, last=BLQAction.DROP)

    @classmethod
    def ich_m13a(cls) -> "BLQRules":
        """The rule set of ICH M13A (2024): zero at both ends, dropped in between.

        Returns:
            `first=ZERO`, `middle=DROP`, `last=ZERO`, the imputed values out of
            the terminal regression.
        """
        return cls(
            first=BLQAction.ZERO,
            middle=BLQAction.DROP,
            last=BLQAction.ZERO,
            terminal_regression=False,
        )

    @classmethod
    def pkanalix(cls) -> "BLQRules":
        """The default rule set of PKanalix: 0 before the maximum, `LLOQ/2` after it.

        Returns:
            `before_tmax=ZERO`, `after_tmax=HALF_LLOQ`.
        """
        return cls(before_tmax=BLQAction.ZERO, after_tmax=BLQAction.HALF_LLOQ)

    @classmethod
    def pumas(cls) -> "BLQRules":
        """The default rule set of Pumas: the ends kept, the middle dropped.

        Returns:
            `first=KEEP`, `middle=DROP`, `last=KEEP`.
        """
        return cls(first=BLQAction.KEEP, middle=BLQAction.DROP, last=BLQAction.KEEP)


class Acceptance(BaseModel):
    r"""Thresholds a sample has to meet for its terminal phase to be accepted.

    A regulatory analysis does not report every terminal regression it can
    compute: the adjusted \(R^2\) of the regression, the extrapolated share of
    \(\mathrm{AUC}_{0\text{-}\infty}\), the number of half-lives the window
    covers and the number of points of the regression are checked against
    thresholds, and the samples which fail them are reported separately or left
    out of the summary statistics. PKanalix ships the four thresholds of
    `Acceptance.pkanalix` as its defaults and restricts its summary statistics
    to the individuals which meet them; Phoenix WinNonlin has the same three
    continuous criteria with an `Accepted`/`Not_Accepted` flag and ships no
    thresholds; PKNCA spells them as the exclusion rules
    `exclude_nca_min.hl.adj.r.squared()`, `exclude_nca_max.aucinf.pext()`,
    `exclude_nca_span_ratio()` and `exclude_nca_count_conc_measured()`.

    Every threshold is `None` by default, so the default analysis accepts every
    sample, and a threshold which is set is checked only where the sample
    carries the value (a sample without a terminal phase has no adjusted
    \(R^2\), so it fails the criterion).

    Attributes:
        r2_adj_min: smallest adjusted \(R^2\) of the terminal regression
            (`lambda_z_r2_adj`)
        extrapolation_max: largest extrapolated fraction
            \((\mathrm{AUC}_{0\text{-}\infty,\mathrm{pred}} -
            \mathrm{AUC}_{0\text{-}t_\mathrm{last}}) /
            \mathrm{AUC}_{0\text{-}\infty,\mathrm{pred}}\), the predicted
            variant PKanalix and Phoenix check
        span_min: smallest number of half-lives the terminal window covers
            (`lambda_z_span`)
        n_points_min: smallest number of points of the terminal regression
            (`lambda_z_n_points`)
        exclude: whether a sample which is not accepted is also marked
            `excluded`, which keeps it out of the summary statistics and of the
            statistics of `pkpdutils.stats`
    """

    model_config = ConfigDict(frozen=True)

    r2_adj_min: float | None = Field(default=None, ge=0.0, le=1.0)
    extrapolation_max: float | None = Field(default=None, gt=0.0, le=1.0)
    span_min: float | None = Field(default=None, gt=0.0)
    n_points_min: int | None = Field(default=None, ge=2)
    exclude: bool = False

    @property
    def any_threshold(self) -> bool:
        """Whether a threshold is set at all."""
        return any(
            value is not None
            for value in (
                self.r2_adj_min,
                self.extrapolation_max,
                self.span_min,
                self.n_points_min,
            )
        )

    @classmethod
    def pkanalix(cls, *, exclude: bool = False) -> "Acceptance":
        r"""The default thresholds of PKanalix.

        Adjusted \(R^2\) of at least 0.98, at most 20 % extrapolated area, a
        span of at least 3 half-lives and at least 3 points of the regression.

        Args:
            exclude: whether a sample which fails a threshold is also excluded.

        Returns:
            The thresholds.
        """
        return cls(
            r2_adj_min=0.98,
            extrapolation_max=0.20,
            span_min=3.0,
            n_points_min=3,
            exclude=exclude,
        )


#: the key of `TerminalPhase.windows` which names every sample the mapping
#: does not name itself
WINDOW_DEFAULT_KEY: str = "*"


class TerminalPhase(BaseModel):
    """Selection of the points of the terminal log-linear regression.

    After an intravenous infusion the samples taken at or before the end of the
    infusion (`t <= t_dose + dose_duration`) are no candidates of any window,
    whatever `method` says: the concentration still rises while the drug is
    given, so the first point a window may start at is the first sample
    strictly after the infusion (Phoenix WinNonlin). It is the only rule of the
    selection which the route decides.

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
        windows: the terminal window `(t_first, t_last)` of single samples,
            keyed by the sample label (the label of a batch with one sample
            dimension, the tuple of labels of a batch with several, and the
            string `"*"` for every sample which the mapping does not name). A
            sample with a window regresses the points inside it, in the times
            of the analysis (relative to its reference dose), as
            `TerminalMethod.MANUAL` does with indices; every other sample
            follows `method`. This is the per-profile window of the interactive
            tools (Phoenix `Lambda_z_lower`/`Lambda_z_upper`, the "Check
            lambda_z" tab of PKanalix), and
            `pkpdutils.nca.NCAResult.terminal_windows` writes the windows of a
            result back in this form, so that a reviewed analysis is re-run
            unchanged
    """

    model_config = ConfigDict(frozen=True)

    method: TerminalMethod = TerminalMethod.BEST_FIT
    min_points: int = Field(default=3, ge=3)
    exclude_cmax: bool = True
    n_points: int | None = Field(default=None, ge=3)
    points: tuple[int, ...] | None = None
    min_adj_r2: float | None = Field(default=None, ge=0.0, le=1.0)
    tie_tolerance: float = Field(default=1e-4, ge=0.0)
    windows: dict[Any, tuple[float, float]] | None = None

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
        for key, window in (self.windows or {}).items():
            if not window[1] > window[0]:
                raise ValueError(
                    f"the terminal window {window} of '{key}' must be "
                    "(t_first, t_last) with t_last > t_first"
                )
        return self


class NCAOptions(BaseModel):
    """Options of a non-compartmental analysis.

    Attributes:
        kind: concentration or effect timecourses
        auc_method: trapezoid rule of the areas
        terminal: selection of the terminal phase
        lloq: lower limit of quantification in the unit of the values, `None`
            to take the per-sample `lloq` of the batch (the coordinate the
            readers of `pkpdutils.io` write), and no limit without one
        blq: handling of values below `lloq`, one of the two classic
            `BLQHandling` values or a `BLQRules` rule set by position
        c0_method: estimate of C(0) after an intravenous bolus
        extrapolation_warning: fraction of AUC(0-inf) above which `EXTRAPOLATION_HIGH` is set
        acceptance: thresholds of the terminal phase every sample is checked
            against (`Acceptance`); the result carries `accepted` and, where
            `Acceptance.exclude` is set, `excluded`
        partial_aucs: named partial areas, name to `(t_start, t_end)` in the
            time unit of the batch, relative to the first dose of the protocol.
            Every one of them becomes a variable of the result with the unit of
            `auc_last`; an interval which reaches beyond the last measurable
            value is completed with the terminal regression and the sample is
            flagged `NCAFlag.PARTIAL_EXTRAPOLATED`. `AUC(0-72)` of a drug with
            a long half-life is `{"auc_0_72": (0.0, 72.0)}` (ICH M13A 2024)
        tau: length of the last dosing interval, `None` to take it from the
            dosing protocol (the distance of the last two doses); it is needed
            for a steady state curve given with its last dose only and it
            overrides the protocol for the last interval
        tau_tolerance: how far the last sample of the analysed dosing interval
            may fall short of its end, as a fraction of `tau`, before the
            interval is given up as incomplete. Within the tolerance the
            exposure of the interval is completed with the terminal regression,
            `auc_tau_extrap_fraction` reports the share which was extrapolated
            and the sample is not flagged; beyond it every steady state
            parameter is `NaN` and the sample carries
            `NCAFlag.INCOMPLETE_INTERVAL`. The default 0.1 covers the sample
            which was taken a few minutes before or after the nominal end of
            the interval, the case EMA and Phoenix WinNonlin both describe; 0
            switches the completion off
        intervals: whether the per-interval parameters (`interval_*`) are part
            of the result of a multiple dose analysis
        effect_threshold: threshold of `time_above` for effect timecourses, `None` for none
        n_workers: workers of the analysis. `None` is automatic: the calling
            thread up to `pkpdutils.parallel.NCA_WORKER_THRESHOLD` rows and
            one worker per usable core, at most 8, above it; `1` is always
            serial and `n > 1` uses that many workers. The core is vectorized
            numpy and releases the GIL, so its workers are threads of the
            calling process (`pkpdutils.parallel`) and no
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
    blq: BLQHandling | BLQRules = BLQHandling.NAN
    c0_method: C0Method = C0Method.LOG_BACK_EXTRAPOLATION
    extrapolation_warning: float = Field(default=0.2, gt=0.0, lt=1.0)
    acceptance: Acceptance = Acceptance()
    partial_aucs: dict[str, tuple[float, float]] = Field(default_factory=dict)
    tau: float | None = Field(default=None, gt=0.0)
    tau_tolerance: float = Field(default=0.1, ge=0.0, lt=1.0)
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

    @model_validator(mode="after")
    def _validate_partial_aucs(self) -> Self:
        """Check that every named partial area is a proper interval.

        Returns:
            This instance, unchanged.

        Raises:
            ValueError: for an empty name or an interval which does not end
                after it starts.
        """
        for name, (t_start, t_end) in self.partial_aucs.items():
            if not name:
                raise ValueError("a partial area needs a name")
            if not t_end > t_start:
                raise ValueError(
                    f"the partial area '{name}' is ({t_start}, {t_end}), "
                    "it must be (t_start, t_end) with t_end > t_start"
                )
        return self

    @property
    def blq_rules(self) -> BLQRules:
        """The rule set of `blq`, the two classic `BLQHandling` values included.

        Returns:
            `blq` itself when it is a `BLQRules`, else the rules of
            `BLQRules.from_handling`.
        """
        if isinstance(self.blq, BLQRules):
            return self.blq
        return BLQRules.from_handling(self.blq)

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
