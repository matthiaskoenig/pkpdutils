import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2, norm

from pkpdutils.stats import ParameterSample
from pkpdutils.stats.meta import (
    EffectKind,
    EffectSize,
    Heterogeneity,
    MetaResult,
    PooledEffect,
    Study,
    effect_size,
    effects_from_arrays,
    fixed_effect,
    heterogeneity,
    meta_analysis,
    meta_analysis_by,
    random_effects,
)
from pkpdutils.stats.tests import hedges_correction

BCG = json.loads(
    (Path(__file__).parent.parent / "data" / "reference" / "meta_bcg.json").read_text()
)


def bcg_effects() -> list[EffectSize]:
    trials = np.array([row[1:] for row in BCG["trials"]], dtype=float)
    a, b, c, d = trials.T
    yi = np.log((a / (a + b)) / (c / (c + d)))
    vi = 1 / a - 1 / (a + b) + 1 / c - 1 / (c + d)
    return effects_from_arrays(
        yi, vi, labels=[row[0] for row in BCG["trials"]], kind=EffectKind.LOG_RATIO
    )


def test_fixed_effect_against_metafor() -> None:
    effects = bcg_effects()
    fixed = fixed_effect(effects)
    assert (
        isinstance(fixed, PooledEffect) and fixed.model == "fixed" and fixed.tau2 == 0.0
    )
    assert fixed.estimate == pytest.approx(BCG["fixed"]["estimate"], abs=1e-6)
    assert fixed.se == pytest.approx(BCG["fixed"]["se"], abs=1e-6)
    assert fixed.weights.sum() == pytest.approx(1.0)
    assert fixed.z == pytest.approx(fixed.estimate / fixed.se)
    assert fixed.p_value == pytest.approx(2 * norm.sf(abs(fixed.z)))
    assert (fixed.ci_low, fixed.ci_high) == pytest.approx(
        (fixed.estimate - 1.959964 * fixed.se, fixed.estimate + 1.959964 * fixed.se),
        abs=1e-6,
    )


def test_random_effects_and_heterogeneity_against_metafor() -> None:
    effects = bcg_effects()
    het = heterogeneity(effects)
    assert isinstance(het, Heterogeneity)
    assert het.q == pytest.approx(BCG["heterogeneity"]["q"], abs=1e-5)
    assert het.df == 12
    assert het.p_value == pytest.approx(chi2.sf(het.q, 12))
    assert het.i2 == pytest.approx(BCG["heterogeneity"]["i2"], abs=1e-3)
    assert het.h2 == pytest.approx(BCG["heterogeneity"]["h2"], abs=1e-3)
    assert het.tau2 == pytest.approx(BCG["random_dl"]["tau2"], abs=1e-6)
    random = random_effects(effects)
    assert random.model == "random" and random.tau2 == het.tau2
    assert random.estimate == pytest.approx(BCG["random_dl"]["estimate"], abs=1e-6)
    assert random.se == pytest.approx(BCG["random_dl"]["se"], abs=1e-6)
    # random effects weights are more even than fixed effect weights
    assert random.weights.max() < fixed_effect(effects).weights.max()


def test_tau2_is_clipped_at_zero() -> None:
    effects = effects_from_arrays([0.1, 0.12, 0.09], [0.04, 0.05, 0.04])
    het = heterogeneity(effects)
    assert het.tau2 == 0.0 and het.i2 == 0.0
    random = random_effects(effects)
    fixed = fixed_effect(effects)
    assert random.estimate == pytest.approx(
        fixed.estimate
    ) and random.se == pytest.approx(fixed.se)
    single = heterogeneity(effects[:1])
    assert (
        single.q == 0.0
        and single.df == 0
        and np.isnan(single.p_value)
        and single.i2 == 0.0
    )
    with pytest.raises(ValueError, match="at least one"):
        fixed_effect([])


def test_hedges_g_matches_the_esc_port() -> None:
    control = ParameterSample(mean=10.0, sd=1.5, n=50)
    treatment = ParameterSample(mean=12.0, sd=2.5, n=60)
    es = effect_size(control, treatment, EffectKind.HEDGES_G)
    # esc_mean_sd(grp1m=10, grp1sd=1.5, grp1n=50, grp2m=12, grp2sd=2.5, grp2n=60): d = 0.94967, var(d) = 0.040766
    d, var_d = 0.9496730565857971, 0.04076611627759853
    j = hedges_correction(110)
    assert es.estimate == pytest.approx(d * j)
    assert es.variance == pytest.approx(var_d * j**2)
    assert es.se == pytest.approx(np.sqrt(var_d) * j)
    assert (es.n_control, es.n_treatment) == (50, 60) and es.kind is EffectKind.HEDGES_G
    assert (es.ci_low, es.ci_high) == pytest.approx(
        (es.estimate - 1.959964 * es.se, es.estimate + 1.959964 * es.se), abs=1e-6
    )
    individual = effect_size(
        ParameterSample(values=np.array([9.0, 10.0, 11.0, 10.0])),
        ParameterSample(values=np.array([11.0, 12.0, 13.0, 12.0])),
    )
    assert individual.estimate == pytest.approx(
        (12 - 10) / np.sqrt(2 / 3) * hedges_correction(8)
    )


def test_mean_difference_and_log_ratio() -> None:
    control = ParameterSample(mean=100.0, sd=30.0, n=12, name="cl", unit="l/hr")
    treatment = ParameterSample(mean=130.0, sd=35.0, n=10)
    md = effect_size(control, treatment, EffectKind.MEAN_DIFF)
    assert md.estimate == 30.0 and md.variance == pytest.approx(
        30.0**2 / 12 + 35.0**2 / 10
    )
    lr = effect_size(control, treatment, EffectKind.LOG_RATIO)
    mu_c, s_c = control.log_moments()
    mu_t, s_t = treatment.log_moments()
    assert lr.estimate == pytest.approx(mu_t - mu_c)
    assert lr.variance == pytest.approx(s_t**2 / 10 + s_c**2 / 12)
    assert lr.to_dict()["kind"] == "log_ratio"


def test_meta_analysis_result() -> None:
    rng = np.random.default_rng(5)
    studies = [
        Study(
            label=f"study {i}",
            control=ParameterSample(values=np.exp(rng.normal(np.log(100), 0.3, 10))),
            treatment=ParameterSample(values=np.exp(rng.normal(np.log(150), 0.3, 12))),
            category="smokers" if i % 2 else "contraceptives",
        )
        for i in range(6)
    ]
    result = meta_analysis(studies, EffectKind.LOG_RATIO)
    assert isinstance(result, MetaResult) and result.n_studies == 6
    assert result.labels == tuple(f"study {i}" for i in range(6))
    assert result.fixed.estimate == pytest.approx(np.log(1.5), abs=0.15)
    assert result.random.ci_low < result.random.estimate < result.random.ci_high
    df = result.to_dataframe()
    assert list(df.columns) == [
        "label",
        "estimate",
        "se",
        "ci_low",
        "ci_high",
        "n_control",
        "n_treatment",
        "weight_fixed",
        "weight_random",
    ]
    assert df["weight_fixed"].sum() == pytest.approx(1.0)
    by = meta_analysis_by(studies, EffectKind.LOG_RATIO)
    assert set(by) == {"smokers", "contraceptives"} and by["smokers"].n_studies == 3
    with pytest.raises(ValueError, match="at least one"):
        meta_analysis([])
