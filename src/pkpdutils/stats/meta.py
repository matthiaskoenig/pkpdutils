r"""Meta-analysis of a parameter over studies: effect sizes, fixed effect and random effects pooling.

An effect size per study (Hedges' g, the mean difference or the log ratio
of the geometric means, the effect native to pharmacokinetics) is pooled
with inverse variance weights: the fixed effect model assumes one true
effect, the random effects model of DerSimonian & Laird (1986) adds the
between-study variance \(\tau^2\) to every weight. The heterogeneity
statistics \(Q\), \(I^2\) and \(H^2\) follow Higgins & Thompson (2002).
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import chi2, norm

from pkpdutils.stats.sample import (
    ParameterSample,
    Scale,
    coerce,
    cohen_d,
    hedges_correction,
)

logger = logging.getLogger(__name__)


class EffectKind(StrEnum):
    """Effect size of a study."""

    #: standardized mean difference with the small sample correction (Hedges 1981)
    HEDGES_G = "hedges_g"
    #: difference of the arithmetic means, treatment minus control
    MEAN_DIFF = "mean_diff"
    #: log of the ratio of the geometric means, treatment over control
    LOG_RATIO = "log_ratio"


@dataclass(frozen=True)
class EffectSize:
    """Effect size of one study.

    Attributes:
        estimate: the effect
        variance: its variance
        se: its standard error
        ci_low: lower bound of the normal interval
        ci_high: upper bound of the normal interval
        ci_level: level of the interval
        kind: the kind of effect
        n_control: size of the control group
        n_treatment: size of the treatment group
        label: label of the study
    """

    estimate: float
    variance: float
    se: float
    ci_low: float
    ci_high: float
    ci_level: float
    kind: EffectKind
    n_control: int
    n_treatment: int
    label: str

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dictionary with the kind as a string.

        Returns:
            Field name to value.
        """
        return {
            "label": self.label,
            "estimate": self.estimate,
            "variance": self.variance,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "kind": str(self.kind),
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
        }


@dataclass(frozen=True)
class Heterogeneity:
    r"""Heterogeneity of the effects of the studies.

    Attributes:
        q: Cochran's \(Q = \sum w_i (\theta_i - \hat\theta_F)^2\)
        df: \(k - 1\)
        p_value: p value of \(Q\) under \(\chi^2_{k-1}\), `NaN` for one study
        i2: \(I^2 = \max(0, (Q - df) / Q)\) in percent
        h2: \(H^2 = Q / df\), `NaN` for one study
        tau2: between-study variance \(\tau^2 = \max(0, (Q - df) / C)\), \(C = \sum w_i - \sum w_i^2 / \sum w_i\)
    """

    q: float
    df: int
    p_value: float
    i2: float
    h2: float
    tau2: float

    def to_dict(self) -> dict[str, Any]:
        """The fields as a dictionary.

        Returns:
            Field name to value.
        """
        return {
            "q": self.q,
            "df": self.df,
            "p_value": self.p_value,
            "i2": self.i2,
            "h2": self.h2,
            "tau2": self.tau2,
        }


@dataclass(frozen=True, eq=False)
class PooledEffect:
    r"""Pooled effect of a meta-analysis.

    Attributes:
        estimate: the pooled effect \(\sum w_i \theta_i / \sum w_i\)
        se: its standard error \(1 / \sqrt{\sum w_i}\)
        ci_low: lower bound of the normal interval
        ci_high: upper bound
        ci_level: level of the interval
        z: \(\hat\theta / \mathrm{se}\)
        p_value: two-sided p value of `z`
        weights: the weights of the studies, normalized to 1
        model: `"fixed"` or `"random"`
        tau2: between-study variance used in the weights (0 for the fixed effect)
    """

    estimate: float
    se: float
    ci_low: float
    ci_high: float
    ci_level: float
    z: float
    p_value: float
    weights: np.ndarray
    model: str
    tau2: float

    def __eq__(self, other: object) -> bool:
        """Whether two pooled effects have the same fields.

        The generated equality of a dataclass compares the `weights` arrays
        with `==`, whose truth value is ambiguous; they are compared with
        `np.array_equal` instead.

        Args:
            other: the object to compare with.

        Returns:
            Whether `other` is a pooled effect with the same fields;
            `NotImplemented` for any other type, so that python falls back
            to the identity comparison.
        """
        if not isinstance(other, PooledEffect):
            return NotImplemented
        return (
            self.estimate,
            self.se,
            self.ci_low,
            self.ci_high,
            self.ci_level,
            self.z,
            self.p_value,
            self.model,
            self.tau2,
        ) == (
            other.estimate,
            other.se,
            other.ci_low,
            other.ci_high,
            other.ci_level,
            other.z,
            other.p_value,
            other.model,
            other.tau2,
        ) and np.array_equal(self.weights, other.weights)

    def to_dict(self) -> dict[str, Any]:
        """The scalar fields as a dictionary.

        Returns:
            Field name to value.
        """
        return {
            "model": self.model,
            "estimate": self.estimate,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "z": self.z,
            "p_value": self.p_value,
            "tau2": self.tau2,
        }


@dataclass(frozen=True)
class Study:
    """A study with a control and a treatment sample of one parameter.

    Attributes:
        label: label of the study
        control: the control sample
        treatment: the treatment sample
        category: category of the study for `meta_analysis_by`
    """

    label: str
    control: ParameterSample
    treatment: ParameterSample
    category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The label, the category and the sizes of the two samples.

        Returns:
            Field name to value.
        """
        return {
            "label": self.label,
            "category": self.category,
            "n_control": self.control.size,
            "n_treatment": self.treatment.size,
        }


@dataclass(frozen=True)
class MetaResult:
    """Result of a meta-analysis.

    Attributes:
        kind: the kind of effect
        effects: the effect per study
        fixed: the fixed effect pooling
        random: the random effects pooling
        heterogeneity: the heterogeneity statistics
        ci_level: level of the intervals
    """

    kind: EffectKind
    effects: tuple[EffectSize, ...]
    fixed: PooledEffect
    random: PooledEffect
    heterogeneity: Heterogeneity
    ci_level: float

    @property
    def labels(self) -> tuple[str, ...]:
        """The labels of the studies."""
        return tuple(e.label for e in self.effects)

    @property
    def n_studies(self) -> int:
        """Number of studies."""
        return len(self.effects)

    def to_dict(self) -> dict[str, Any]:
        """The kind, the labels and the pooled results as nested dictionaries.

        The per study effects are `to_dataframe`.

        Returns:
            `kind`, `n_studies`, `labels`, `ci_level` and the dictionaries
            of `fixed`, `random` and `heterogeneity`.
        """
        return {
            "kind": str(self.kind),
            "n_studies": self.n_studies,
            "labels": self.labels,
            "ci_level": self.ci_level,
            "fixed": self.fixed.to_dict(),
            "random": self.random.to_dict(),
            "heterogeneity": self.heterogeneity.to_dict(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """One row per study with its effect, interval, sizes and weights.

        Returns:
            The dataframe.
        """
        rows = []
        for i, e in enumerate(self.effects):
            rows.append(
                {
                    "label": e.label,
                    "estimate": e.estimate,
                    "se": e.se,
                    "ci_low": e.ci_low,
                    "ci_high": e.ci_high,
                    "n_control": e.n_control,
                    "n_treatment": e.n_treatment,
                    "weight_fixed": float(self.fixed.weights[i]),
                    "weight_random": float(self.random.weights[i]),
                }
            )
        return pd.DataFrame(rows)


def _z(ci_level: float) -> float:
    r"""The normal quantile of a two-sided interval.

    Args:
        ci_level: level of the interval.

    Returns:
        \(z_{1 - \alpha/2}\).
    """
    return float(norm.ppf(1.0 - (1.0 - ci_level) / 2.0))


def _effect(
    estimate: float,
    variance: float,
    kind: EffectKind,
    n_control: int,
    n_treatment: int,
    label: str,
    ci_level: float,
) -> EffectSize:
    """Build an effect size with its normal interval.

    Args:
        estimate: the effect.
        variance: its variance.
        kind: the kind of effect.
        n_control: size of the control group.
        n_treatment: size of the treatment group.
        label: label of the study.
        ci_level: level of the interval.

    Returns:
        The effect size.
    """
    se = float(np.sqrt(variance))
    z = _z(ci_level)
    return EffectSize(
        estimate=float(estimate),
        variance=float(variance),
        se=se,
        ci_low=float(estimate - z * se),
        ci_high=float(estimate + z * se),
        ci_level=ci_level,
        kind=kind,
        n_control=int(n_control),
        n_treatment=int(n_treatment),
        label=label,
    )


def effect_size(
    control: ParameterSample,
    treatment: ParameterSample,
    kind: EffectKind | str = EffectKind.HEDGES_G,
    *,
    ci_level: float = 0.95,
    label: str = "",
) -> EffectSize:
    r"""Effect size of a treatment against a control.

    Hedges' g: \(d = (\bar x_T - \bar x_C) / s_p\) with the pooled standard deviation,
    \(\mathrm{var}(d) = N / (n_C n_T) + d^2 / (2N)\), \(g = J d\),
    \(\mathrm{var}(g) = J^2 \mathrm{var}(d)\) (Hedges 1981). Mean difference:
    \(\bar x_T - \bar x_C\) with \(s_T^2 / n_T + s_C^2 / n_C\). Log ratio:
    \(\mu_T - \mu_C\) of the log moments with \(\sigma_T^2 / n_T + \sigma_C^2 / n_C\).

    A group of a single value and two groups without variance leave the
    standardized difference undefined: `estimate` and `variance` are `NaN`,
    and pooling such an effect raises instead of dropping it silently.

    Args:
        control: the control sample.
        treatment: the treatment sample.
        kind: the kind of effect, as the member or as its string.
        ci_level: level of the interval.
        label: label of the study.

    Returns:
        The effect size.

    Raises:
        ValueError: if `kind` is not an `EffectKind`.
    """
    kind = coerce(kind, EffectKind)
    scale = Scale.LOG if kind is EffectKind.LOG_RATIO else Scale.LINEAR
    m_c, s_c, n_c = control.moments(scale)
    m_t, s_t, n_t = treatment.moments(scale)
    if kind is EffectKind.HEDGES_G:
        d, g = cohen_d(m_t, s_t, n_t, m_c, s_c, n_c)
        if not np.isfinite(d):
            logger.debug(
                "study '%s' has no pooled standard deviation, its effect is NaN", label
            )
            nan = float("nan")
            return _effect(nan, nan, kind, n_c, n_t, label, ci_level)
        total = n_c + n_t
        var_d = total / (n_c * n_t) + d**2 / (2.0 * total)
        j = hedges_correction(total)
        return _effect(g, var_d * j**2, kind, n_c, n_t, label, ci_level)
    return _effect(
        m_t - m_c, s_t**2 / n_t + s_c**2 / n_c, kind, n_c, n_t, label, ci_level
    )


def effects_from_arrays(
    estimates: ArrayLike,
    variances: ArrayLike,
    labels: Sequence[str] | None = None,
    kind: EffectKind | str = EffectKind.HEDGES_G,
    *,
    ci_level: float = 0.95,
) -> list[EffectSize]:
    """Effect sizes from estimates and variances computed elsewhere.

    Args:
        estimates: the effects.
        variances: their variances.
        labels: labels of the studies, the positions by default.
        kind: the kind of effect, as the member or as its string.
        ci_level: level of the intervals.

    Returns:
        The effect sizes (`n_control` and `n_treatment` are 0).

    Raises:
        ValueError: if `kind` is not an `EffectKind` or the lengths differ.
    """
    kind = coerce(kind, EffectKind)
    est = np.asarray(estimates, dtype=np.float64).ravel()
    var = np.asarray(variances, dtype=np.float64).ravel()
    if est.size != var.size:
        raise ValueError(f"{est.size} estimates but {var.size} variances")
    names = [str(i) for i in range(est.size)] if labels is None else list(labels)
    if len(names) != est.size:
        raise ValueError(f"{est.size} estimates but {len(names)} labels")
    return [
        _effect(e, v, kind, 0, 0, name, ci_level)
        for e, v, name in zip(est, var, names, strict=True)
    ]


def _arrays(effects: Sequence[EffectSize]) -> tuple[np.ndarray, np.ndarray]:
    """The estimates and the variances of the effects, validated.

    The inverse variance weights need a positive variance of every study; a
    study without one (a single subject, or a group without variance) would
    otherwise get an infinite weight, or a weight of zero which drops it
    from the pooling without a word.

    Args:
        effects: the effect sizes.

    Returns:
        The estimates and the variances.

    Raises:
        ValueError: without effects, or if a study has a variance which is
            not positive and finite, naming the study.
    """
    if not effects:
        raise ValueError("A meta-analysis needs at least one study")
    for e in effects:
        if not (np.isfinite(e.variance) and e.variance > 0):
            raise ValueError(
                f"Study '{e.label}' has the variance {e.variance}, "
                "the inverse variance pooling needs a positive variance of every study"
            )
    return (
        np.array([e.estimate for e in effects], dtype=np.float64),
        np.array([e.variance for e in effects], dtype=np.float64),
    )


def _pool(
    theta: np.ndarray, w: np.ndarray, model: str, tau2: float, ci_level: float
) -> PooledEffect:
    """Inverse variance pooling.

    Args:
        theta: the effects.
        w: the weights.
        model: `"fixed"` or `"random"`.
        tau2: the between-study variance of the weights.
        ci_level: level of the interval.

    Returns:
        The pooled effect.
    """
    estimate = float((w * theta).sum() / w.sum())
    se = float(np.sqrt(1.0 / w.sum()))
    z = estimate / se
    zq = _z(ci_level)
    return PooledEffect(
        estimate=estimate,
        se=se,
        ci_low=estimate - zq * se,
        ci_high=estimate + zq * se,
        ci_level=ci_level,
        z=float(z),
        p_value=float(2.0 * norm.sf(abs(z))),
        weights=w / w.sum(),
        model=model,
        tau2=tau2,
    )


def fixed_effect(
    effects: Sequence[EffectSize], *, ci_level: float = 0.95
) -> PooledEffect:
    r"""Fixed effect pooling with the weights \(w_i = 1 / v_i\).

    Args:
        effects: the effect sizes.
        ci_level: level of the interval.

    Returns:
        The pooled effect.

    Raises:
        ValueError: as `_arrays`, without effects or for a study without a
            positive variance.
    """
    theta, v = _arrays(effects)
    return _pool(theta, 1.0 / v, "fixed", 0.0, ci_level)


def heterogeneity(effects: Sequence[EffectSize]) -> Heterogeneity:
    r"""Heterogeneity statistics of the effects.

    \(Q = \sum w_i (\theta_i - \hat\theta_F)^2\) with \(w_i = 1/v_i\),
    \(C = \sum w_i - \sum w_i^2 / \sum w_i\), \(\tau^2 = \max(0, (Q - (k-1)) / C)\)
    (DerSimonian & Laird 1986), \(I^2 = \max(0, (Q - (k-1)) / Q)\), \(H^2 = Q / (k-1)\)
    (Higgins & Thompson 2002).

    Args:
        effects: the effect sizes.

    Returns:
        The statistics.

    Raises:
        ValueError: as `_arrays`, without effects or for a study without a
            positive variance.
    """
    theta, v = _arrays(effects)
    return _heterogeneity(theta, v)


def _heterogeneity(theta: np.ndarray, v: np.ndarray) -> Heterogeneity:
    """Heterogeneity statistics of validated estimates and variances.

    Args:
        theta: the effects.
        v: their variances, positive.

    Returns:
        The statistics.
    """
    w = 1.0 / v
    k = theta.size
    theta_f = (w * theta).sum() / w.sum()
    q = float((w * (theta - theta_f) ** 2).sum())
    df = k - 1
    c = float(w.sum() - (w**2).sum() / w.sum())
    tau2 = max(0.0, (q - df) / c) if df > 0 and c > 0 else 0.0
    i2 = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0
    return Heterogeneity(
        q=q,
        df=df,
        p_value=float(chi2.sf(q, df)) if df > 0 else float("nan"),
        i2=i2,
        h2=q / df if df > 0 else float("nan"),
        tau2=tau2,
    )


def random_effects(
    effects: Sequence[EffectSize], *, ci_level: float = 0.95
) -> PooledEffect:
    r"""Random effects pooling of DerSimonian & Laird with the weights \(w_i^* = 1 / (v_i + \tau^2)\).

    Args:
        effects: the effect sizes.
        ci_level: level of the interval.

    Returns:
        The pooled effect.

    Raises:
        ValueError: as `_arrays`, without effects or for a study without a
            positive variance.
    """
    theta, v = _arrays(effects)
    tau2 = _heterogeneity(theta, v).tau2
    return _pool(theta, 1.0 / (v + tau2), "random", tau2, ci_level)


def meta_analysis(
    studies: Sequence[Study],
    kind: EffectKind | str = EffectKind.HEDGES_G,
    *,
    ci_level: float = 0.95,
) -> MetaResult:
    """Meta-analysis of a parameter over studies.

    Args:
        studies: the studies.
        kind: the kind of effect, as the member or as its string.
        ci_level: level of the intervals.

    Returns:
        The per study effects, the fixed effect and random effects pooling and the heterogeneity.

    Raises:
        ValueError: without studies, for an unknown `kind`, or for a study
            without a positive variance of its effect.
    """
    if not studies:
        raise ValueError("A meta-analysis needs at least one study")
    kind = coerce(kind, EffectKind)
    effects = tuple(
        effect_size(s.control, s.treatment, kind, ci_level=ci_level, label=s.label)
        for s in studies
    )
    theta, v = _arrays(effects)
    het = _heterogeneity(theta, v)
    return MetaResult(
        kind=kind,
        effects=effects,
        fixed=_pool(theta, 1.0 / v, "fixed", 0.0, ci_level),
        random=_pool(theta, 1.0 / (v + het.tau2), "random", het.tau2, ci_level),
        heterogeneity=het,
        ci_level=ci_level,
    )


def meta_analysis_by(
    studies: Sequence[Study],
    kind: EffectKind | str = EffectKind.HEDGES_G,
    *,
    ci_level: float = 0.95,
) -> dict[str, MetaResult]:
    """One meta-analysis per category of the studies.

    Args:
        studies: the studies; a study without a category is grouped under `""`.
        kind: the kind of effect, as the member or as its string.
        ci_level: level of the intervals.

    Returns:
        Category to result, in the order of first appearance.

    Raises:
        ValueError: as `meta_analysis`.
    """
    kind = coerce(kind, EffectKind)
    groups: dict[str, list[Study]] = {}
    for study in studies:
        groups.setdefault(study.category or "", []).append(study)
    return {
        key: meta_analysis(group, kind, ci_level=ci_level)
        for key, group in groups.items()
    }
