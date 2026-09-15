import numpy as np
import pytest

from pkpdutils.stats import ParameterSample
from pkpdutils.stats.ddi import (
    DDIKind,
    DDIResult,
    DDIStrength,
    DDIThresholds,
    Sensitivity,
    ddi_classification,
    substrate_sensitivity,
)
from pkpdutils.stats.ratio import ratio


@pytest.mark.parametrize(
    ("auc_ratio", "kind", "strength"),
    [
        (5.0, DDIKind.INHIBITOR, DDIStrength.STRONG),
        (7.3, DDIKind.INHIBITOR, DDIStrength.STRONG),
        (4.99, DDIKind.INHIBITOR, DDIStrength.MODERATE),
        (2.0, DDIKind.INHIBITOR, DDIStrength.MODERATE),
        (1.99, DDIKind.INHIBITOR, DDIStrength.WEAK),
        (1.25, DDIKind.INHIBITOR, DDIStrength.WEAK),
        (1.24, DDIKind.NONE, DDIStrength.NONE),
        (1.0, DDIKind.NONE, DDIStrength.NONE),
        (0.81, DDIKind.NONE, DDIStrength.NONE),
        (0.8, DDIKind.INDUCER, DDIStrength.WEAK),
        (0.51, DDIKind.INDUCER, DDIStrength.WEAK),
        (0.5, DDIKind.INDUCER, DDIStrength.MODERATE),
        (0.21, DDIKind.INDUCER, DDIStrength.MODERATE),
        (0.2, DDIKind.INDUCER, DDIStrength.STRONG),
        (0.05, DDIKind.INDUCER, DDIStrength.STRONG),
    ],
)
def test_thresholds_at_the_boundaries(
    auc_ratio: float, kind: DDIKind, strength: DDIStrength
) -> None:
    assert DDIThresholds.fda().classify(auc_ratio) == (kind, strength)
    res = ddi_classification(auc_ratio)
    assert isinstance(res, DDIResult)
    assert (res.kind, res.strength) == (kind, strength) and not res.uncertain
    assert res.auc_ratio == auc_ratio and res.cmax_ratio is None


def test_interval_uses_the_bound_closer_to_one() -> None:
    res = ddi_classification(2.5, ci=(1.6, 3.9))
    assert (res.kind, res.strength) == (DDIKind.INHIBITOR, DDIStrength.WEAK)
    assert res.uncertain and (res.kind_low, res.strength_low) == (
        DDIKind.INHIBITOR,
        DDIStrength.WEAK,
    )
    assert (res.kind_high, res.strength_high) == (
        DDIKind.INHIBITOR,
        DDIStrength.MODERATE,
    )
    certain = ddi_classification(2.5, ci=(2.1, 3.0))
    assert (certain.kind, certain.strength) == (
        DDIKind.INHIBITOR,
        DDIStrength.MODERATE,
    ) and not certain.uncertain
    inducer = ddi_classification(0.4, ci=(0.3, 0.55))
    assert (inducer.kind, inducer.strength) == (
        DDIKind.INDUCER,
        DDIStrength.WEAK,
    ) and inducer.uncertain
    spanning = ddi_classification(1.3, ci=(0.9, 1.9))
    assert (spanning.kind, spanning.strength) == (
        DDIKind.NONE,
        DDIStrength.NONE,
    ) and spanning.uncertain
    with pytest.raises(ValueError, match="ci"):
        ddi_classification(2.0, ci=(3.0, 1.0))


def test_ratio_result_input() -> None:
    inhibited = ParameterSample(values=np.array([300.0, 320.0, 280.0, 310.0, 290.0]))
    control = ParameterSample(values=np.array([100.0, 105.0, 95.0, 110.0, 90.0]))
    r = ratio(inhibited, control)
    res = ddi_classification(r, cmax_ratio=ratio(control, control))
    assert res.auc_ratio == r.gmr and (res.ci_low, res.ci_high) == (r.ci_low, r.ci_high)
    assert res.cmax_ratio == 1.0
    assert (res.kind, res.strength) == (DDIKind.INHIBITOR, DDIStrength.MODERATE)
    assert res.to_dict()["kind"] == "inhibitor"


def test_ema_and_sensitivity() -> None:
    ema = DDIThresholds.ema()
    assert ema.source.startswith("EMA") and ema.inhibitor_strong == 5.0
    assert ddi_classification(3.0, thresholds=ema).thresholds is ema
    assert substrate_sensitivity(5.0) is Sensitivity.SENSITIVE
    assert substrate_sensitivity(2.0) is Sensitivity.MODERATELY_SENSITIVE
    assert substrate_sensitivity(1.9) is Sensitivity.NONE
    assert substrate_sensitivity(5.0, thresholds=ema) is Sensitivity.SENSITIVE
    # the options of every stats function are keyword-only
    with pytest.raises(TypeError, match="positional"):
        substrate_sensitivity(5.0, ema)  # ty: ignore[too-many-positional-arguments]
    with pytest.raises(ValueError, match="positive"):
        ddi_classification(0.0)
