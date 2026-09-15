"""Classification of drug-drug interactions by the change of the exposure.

The FDA guidance (FDA 2020) classifies a perpetrator by the ratio of the
AUC of a sensitive substrate with and without it: a strong, moderate or
weak inhibitor raises the AUC at least 5-fold, 2- to 5-fold or 1.25- to
2-fold; a strong, moderate or weak inducer lowers it by at least 80 %,
50-80 % or 20-50 %. A substrate is sensitive when a strong inhibitor raises
its AUC at least 5-fold and moderately sensitive at 2- to 5-fold. The EMA
guideline (EMA 2012) uses the same thresholds.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pkpdutils.stats.ratio import RatioResult


class DDIKind(StrEnum):
    """Direction of an interaction."""

    INHIBITOR = "inhibitor"
    INDUCER = "inducer"
    NONE = "none"


class DDIStrength(StrEnum):
    """Strength of an interaction."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


class Sensitivity(StrEnum):
    """Sensitivity of a substrate to a strong inhibitor."""

    SENSITIVE = "sensitive"
    MODERATELY_SENSITIVE = "moderately_sensitive"
    NONE = "none"


@dataclass(frozen=True)
class DDIThresholds:
    """Thresholds of the classification, as AUC ratios with / without the perpetrator.

    An inhibitor threshold is the smallest ratio of its class, an inducer
    threshold the largest (`inducer_strong = 0.2` is an 80 % decrease).

    Attributes:
        inhibitor_weak: weak inhibitor at or above this ratio
        inhibitor_moderate: moderate inhibitor at or above this ratio
        inhibitor_strong: strong inhibitor at or above this ratio
        inducer_weak: weak inducer at or below this ratio
        inducer_moderate: moderate inducer at or below this ratio
        inducer_strong: strong inducer at or below this ratio
        sensitive: sensitive substrate at or above this ratio
        moderately_sensitive: moderately sensitive substrate at or above this ratio
        source: the guidance the thresholds come from
    """

    inhibitor_weak: float = 1.25
    inhibitor_moderate: float = 2.0
    inhibitor_strong: float = 5.0
    inducer_weak: float = 0.8
    inducer_moderate: float = 0.5
    inducer_strong: float = 0.2
    sensitive: float = 5.0
    moderately_sensitive: float = 2.0
    source: str = "FDA 2020"

    @classmethod
    def fda(cls) -> "DDIThresholds":
        """The thresholds of the FDA clinical drug interaction guidance (FDA 2020).

        Returns:
            The thresholds.
        """
        return cls()

    @classmethod
    def ema(cls) -> "DDIThresholds":
        """The thresholds of the EMA guideline on the investigation of drug interactions (EMA 2012).

        These equal the FDA ones.

        Returns:
            The thresholds.
        """
        return cls(source="EMA 2012")

    def classify(self, auc_ratio: float) -> tuple[DDIKind, DDIStrength]:
        """Kind and strength of an interaction from an AUC ratio.

        Args:
            auc_ratio: AUC with / without the perpetrator, positive.

        Returns:
            The kind and the strength.

        Raises:
            ValueError: if the ratio is not positive.
        """
        if not auc_ratio > 0:
            raise ValueError(f"The AUC ratio must be positive, got {auc_ratio}")
        if auc_ratio >= self.inhibitor_strong:
            return DDIKind.INHIBITOR, DDIStrength.STRONG
        if auc_ratio >= self.inhibitor_moderate:
            return DDIKind.INHIBITOR, DDIStrength.MODERATE
        if auc_ratio >= self.inhibitor_weak:
            return DDIKind.INHIBITOR, DDIStrength.WEAK
        if auc_ratio <= self.inducer_strong:
            return DDIKind.INDUCER, DDIStrength.STRONG
        if auc_ratio <= self.inducer_moderate:
            return DDIKind.INDUCER, DDIStrength.MODERATE
        if auc_ratio <= self.inducer_weak:
            return DDIKind.INDUCER, DDIStrength.WEAK
        return DDIKind.NONE, DDIStrength.NONE


@dataclass(frozen=True)
class DDIResult:
    """Classification of an interaction.

    Attributes:
        kind: the classification (from the bound of the interval closer to 1 when an interval is given)
        strength: the strength
        auc_ratio: the AUC ratio
        cmax_ratio: the Cmax ratio, reported only
        ci_low: lower bound of the interval of the AUC ratio, `NaN` without one
        ci_high: upper bound of the interval
        uncertain: whether the interval spans a boundary of the classes
        kind_low: classification of `ci_low`
        strength_low: strength of `ci_low`
        kind_high: classification of `ci_high`
        strength_high: strength of `ci_high`
        thresholds: the thresholds used
    """

    kind: DDIKind
    strength: DDIStrength
    auc_ratio: float
    cmax_ratio: float | None
    ci_low: float
    ci_high: float
    uncertain: bool
    kind_low: DDIKind
    strength_low: DDIStrength
    kind_high: DDIKind
    strength_high: DDIStrength
    thresholds: DDIThresholds

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dictionary with the enumerations as strings.

        Returns:
            Field name to value.
        """
        return {
            "kind": str(self.kind),
            "strength": str(self.strength),
            "auc_ratio": self.auc_ratio,
            "cmax_ratio": self.cmax_ratio,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "uncertain": self.uncertain,
            "kind_low": str(self.kind_low),
            "strength_low": str(self.strength_low),
            "kind_high": str(self.kind_high),
            "strength_high": str(self.strength_high),
            "source": self.thresholds.source,
        }


def _ratio_value(value: float | RatioResult) -> float:
    """The point estimate of a ratio given as a number or as a `RatioResult`.

    Args:
        value: the ratio.

    Returns:
        The number.
    """
    return value.gmr if isinstance(value, RatioResult) else float(value)


def ddi_classification(
    auc_ratio: float | RatioResult,
    *,
    cmax_ratio: float | RatioResult | None = None,
    ci: tuple[float, float] | None = None,
    thresholds: DDIThresholds | None = None,
) -> DDIResult:
    """Classify a perpetrator by the AUC ratio of a substrate with and without it.

    With an interval (given as `ci` or carried by a `RatioResult`) the
    classification is conservative: it uses the bound closer to 1 (the lower
    bound of an increase, the upper bound of a decrease), an interval which
    contains 1 gives no interaction, and `uncertain` is set when the two
    bounds fall into different classes.

    Args:
        auc_ratio: AUC ratio with / without the perpetrator, or the `ratio` result.
        cmax_ratio: Cmax ratio, reported next to the classification.
        ci: interval of the AUC ratio; overrides the interval of a `RatioResult`.
        thresholds: the thresholds, FDA 2020 by default.

    Returns:
        The classification.

    Raises:
        ValueError: if the interval is reversed or a ratio is not positive.
    """
    if thresholds is None:
        thresholds = DDIThresholds.fda()
    value = _ratio_value(auc_ratio)
    if ci is None and isinstance(auc_ratio, RatioResult):
        ci = (auc_ratio.ci_low, auc_ratio.ci_high)
    kind, strength = thresholds.classify(value)
    if ci is None:
        return DDIResult(
            kind=kind,
            strength=strength,
            auc_ratio=value,
            cmax_ratio=None if cmax_ratio is None else _ratio_value(cmax_ratio),
            ci_low=float("nan"),
            ci_high=float("nan"),
            uncertain=False,
            kind_low=kind,
            strength_low=strength,
            kind_high=kind,
            strength_high=strength,
            thresholds=thresholds,
        )
    low, high = float(ci[0]), float(ci[1])
    if not low <= high:
        raise ValueError(f"'ci' must be (low, high), got {ci}")
    kind_low, strength_low = thresholds.classify(low)
    kind_high, strength_high = thresholds.classify(high)
    closer = low if low > 1.0 else high if high < 1.0 else 1.0
    kind, strength = thresholds.classify(closer)
    return DDIResult(
        kind=kind,
        strength=strength,
        auc_ratio=value,
        cmax_ratio=None if cmax_ratio is None else _ratio_value(cmax_ratio),
        ci_low=low,
        ci_high=high,
        uncertain=(kind_low, strength_low) != (kind_high, strength_high),
        kind_low=kind_low,
        strength_low=strength_low,
        kind_high=kind_high,
        strength_high=strength_high,
        thresholds=thresholds,
    )


def substrate_sensitivity(
    auc_ratio: float | RatioResult, thresholds: DDIThresholds | None = None
) -> Sensitivity:
    """Sensitivity of a substrate from its AUC ratio with a strong inhibitor.

    Args:
        auc_ratio: AUC ratio with / without the strong inhibitor.
        thresholds: the thresholds, FDA 2020 by default.

    Returns:
        `SENSITIVE` at or above `thresholds.sensitive`, `MODERATELY_SENSITIVE`
        at or above `thresholds.moderately_sensitive`, else `NONE`.

    Raises:
        ValueError: if the ratio is not positive.
    """
    if thresholds is None:
        thresholds = DDIThresholds.fda()
    value = _ratio_value(auc_ratio)
    if not value > 0:
        raise ValueError(f"The AUC ratio must be positive, got {value}")
    if value >= thresholds.sensitive:
        return Sensitivity.SENSITIVE
    if value >= thresholds.moderately_sensitive:
        return Sensitivity.MODERATELY_SENSITIVE
    return Sensitivity.NONE
