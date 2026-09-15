import numpy as np
import pytest
from scipy import stats

from pkpdutils.stats import ParameterSample, Scale
from pkpdutils.stats.tests import (
    AdjustMethod,
    Alternative,
    TestMethod,
    TestResult,
    compare,
    hedges_correction,
    multiple_comparison,
)

RNG = np.random.default_rng(7)
A = np.exp(RNG.normal(np.log(100), 0.3, 12))
B = np.exp(RNG.normal(np.log(130), 0.3, 10))
A_PAIRED = np.exp(RNG.normal(np.log(100), 0.3, 12))
B_PAIRED = A_PAIRED * np.exp(RNG.normal(0.2, 0.1, 12))


def sample(values: np.ndarray, labels: list[str] | None = None) -> ParameterSample:
    return ParameterSample(
        values=values,
        labels=None if labels is None else np.array(labels),
        name="auc",
        unit="mg*hr/l",
    )


def test_auto_is_welch_on_log_scale() -> None:
    r = compare(sample(A), sample(B))
    assert isinstance(r, TestResult) and r.test is TestMethod.WELCH_T
    ref = stats.ttest_ind(np.log(A), np.log(B), equal_var=False)
    assert r.statistic == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)
    assert r.df == pytest.approx(ref.df)
    assert r.effect == pytest.approx(np.exp(np.log(A).mean() - np.log(B).mean()))
    low, high = ref.confidence_interval(0.95)
    assert (r.ci_low, r.ci_high) == pytest.approx((np.exp(low), np.exp(high)))
    assert r.scale is Scale.LOG and r.n_a == 12 and r.n_b == 10 and r.name == "auc"


def test_student_t_linear() -> None:
    r = compare(sample(A), sample(B), test=TestMethod.STUDENT_T, scale=Scale.LINEAR)
    ref = stats.ttest_ind(A, B, equal_var=True)
    assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(
        ref.pvalue
    )
    assert r.df == 20 and r.effect == pytest.approx(A.mean() - B.mean())
    low, high = ref.confidence_interval(0.95)
    assert (r.ci_low, r.ci_high) == pytest.approx((low, high))


def test_paired_t() -> None:
    r = compare(sample(A_PAIRED), sample(B_PAIRED), paired=True)
    assert r.test is TestMethod.PAIRED_T and r.paired
    ref = stats.ttest_rel(np.log(A_PAIRED), np.log(B_PAIRED))
    assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(
        ref.pvalue
    )
    assert r.df == 11
    d = np.log(A_PAIRED) - np.log(B_PAIRED)
    assert r.effect == pytest.approx(np.exp(d.mean()))


def test_paired_needs_equal_size() -> None:
    with pytest.raises(ValueError, match="paired"):
        compare(sample(A), sample(B), paired=True)


def test_paired_matches_the_labels_not_the_positions() -> None:
    labels = [f"s{i}" for i in range(12)]
    a = sample(A_PAIRED, labels)
    perm = np.random.default_rng(5).permutation(12)
    b_permuted = sample(B_PAIRED[perm], [labels[i] for i in perm])
    r = compare(a, b_permuted, paired=True)
    direct = compare(a, sample(B_PAIRED, labels), paired=True)
    assert r.p_value == pytest.approx(direct.p_value)
    assert r.statistic == pytest.approx(direct.statistic)
    ref = stats.ttest_rel(np.log(A_PAIRED), np.log(B_PAIRED))
    assert r.statistic == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)
    assert r.n_a == 12 and r.n_b == 12


def test_paired_drops_only_the_pair_with_a_missing_value() -> None:
    labels = ["s0", "s1", "s2", "s3"]
    a = sample(np.array([100.0, np.nan, 300.0, 400.0]), labels)
    b = sample(np.array([np.nan, 200.0, 330.0, 450.0]), labels)
    r = compare(a, b, paired=True)
    assert r.n_a == 2 and r.n_b == 2 and r.df == 1
    ref = stats.ttest_rel(np.log([300.0, 400.0]), np.log([330.0, 450.0]))
    assert r.statistic == pytest.approx(ref.statistic)
    assert r.p_value == pytest.approx(ref.pvalue)


def test_single_value_and_zero_variance_give_nan_statistics() -> None:
    nan_fields = ("statistic", "p_value", "df", "ci_low", "ci_high")
    one_value = compare(ParameterSample(values=np.array([1.0])), sample(B))
    one_summary = compare(ParameterSample(mean=1.0, sd=0.1, n=1), sample(B))
    constant = compare(
        sample(np.full(5, 7.0)), sample(np.full(4, 7.0)), scale=Scale.LINEAR
    )
    for r in (one_value, one_summary, constant):
        for field in nan_fields:
            assert np.isnan(getattr(r, field)), f"{r.test} {field}"
        assert np.isnan(r.cohen_d) and np.isnan(r.hedges_g)
        assert np.isfinite(r.effect)
    assert one_value.n_a == 1 and constant.effect == 0.0


def test_one_sided_alternative() -> None:
    r = compare(sample(A), sample(B), alternative=Alternative.LESS)
    ref = stats.ttest_ind(np.log(A), np.log(B), equal_var=False, alternative="less")
    assert r.p_value == pytest.approx(ref.pvalue)
    assert r.ci_low == 0.0 and np.isfinite(r.ci_high)
    high = ref.confidence_interval(0.95).high
    assert r.ci_high == pytest.approx(np.exp(high))
    g = compare(
        sample(A), sample(B), scale=Scale.LINEAR, alternative=Alternative.GREATER
    )
    assert g.ci_high == np.inf and np.isfinite(g.ci_low)


def test_rank_tests() -> None:
    mw = compare(sample(A), sample(B), test=TestMethod.MANN_WHITNEY)
    ref = stats.mannwhitneyu(np.log(A), np.log(B))
    assert mw.statistic == pytest.approx(ref.statistic) and mw.p_value == pytest.approx(
        ref.pvalue
    )
    assert mw.effect == pytest.approx(np.median(A) / np.median(B))
    assert np.isnan(mw.ci_low) and np.isnan(mw.df)
    w = compare(
        sample(A_PAIRED), sample(B_PAIRED), test=TestMethod.WILCOXON, paired=True
    )
    ref_w = stats.wilcoxon(np.log(A_PAIRED), np.log(B_PAIRED))
    assert w.statistic == pytest.approx(ref_w.statistic) and w.p_value == pytest.approx(
        ref_w.pvalue
    )
    with pytest.raises(ValueError, match="paired"):
        compare(sample(A_PAIRED), sample(B_PAIRED), test=TestMethod.WILCOXON)


def test_permutation_reproducible() -> None:
    r1 = compare(sample(A), sample(B), test=TestMethod.PERMUTATION, n_perm=999, seed=3)
    r2 = compare(sample(A), sample(B), test=TestMethod.PERMUTATION, n_perm=999, seed=3)
    assert r1.p_value == r2.p_value
    # A and B (seed 7) are not far apart relative to their spread: the true
    # permutation p value is close to the parametric (equal variance) p value.
    ref = stats.ttest_ind(np.log(A), np.log(B), equal_var=True)
    assert r1.p_value == pytest.approx(ref.pvalue, abs=0.05)
    assert r1.statistic == pytest.approx(np.log(A).mean() - np.log(B).mean())
    assert r1.effect == pytest.approx(np.exp(r1.statistic))
    paired = compare(
        sample(A_PAIRED),
        sample(B_PAIRED),
        test=TestMethod.PERMUTATION,
        paired=True,
        n_perm=499,
        seed=1,
    )
    assert paired.p_value < 0.01


def test_summary_data_welch() -> None:
    a = ParameterSample(mean=100.0, sd=30.0, n=12)
    b = ParameterSample(mean=130.0, sd=35.0, n=10)
    r = compare(a, b, scale=Scale.LINEAR)
    ref = stats.ttest_ind_from_stats(100.0, 30.0, 12, 130.0, 35.0, 10, equal_var=False)
    assert r.test is TestMethod.WELCH_T
    assert r.statistic == pytest.approx(ref.statistic) and r.p_value == pytest.approx(
        ref.pvalue
    )
    assert r.effect == -30.0
    log = compare(a, b)
    mu_a, s_a = a.log_moments()
    mu_b, s_b = b.log_moments()
    ref_log = stats.ttest_ind_from_stats(mu_a, s_a, 12, mu_b, s_b, 10, equal_var=False)
    assert log.p_value == pytest.approx(ref_log.pvalue)
    assert log.effect == pytest.approx(np.exp(mu_a - mu_b))
    with pytest.raises(ValueError, match="individual"):
        compare(a, b, test=TestMethod.MANN_WHITNEY)
    with pytest.raises(ValueError, match="individual"):
        compare(a, sample(B), paired=True)


def test_effect_sizes() -> None:
    r = compare(sample(A), sample(B), scale=Scale.LINEAR)
    sp = np.sqrt((11 * A.var(ddof=1) + 9 * B.var(ddof=1)) / 20)
    d = (A.mean() - B.mean()) / sp
    assert r.cohen_d == pytest.approx(d)
    assert r.hedges_g == pytest.approx(d * hedges_correction(22))
    assert hedges_correction(110) == pytest.approx(1 - 3 / (4 * 110 - 9))
    assert r.to_dict()["test"] == "welch_t"


def test_log_scale_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        compare(sample(np.array([1.0, -2.0, 3.0])), sample(B))


def test_multiple_comparison() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.2])
    bonf = multiple_comparison(p, AdjustMethod.BONFERRONI)
    assert bonf.tolist() == pytest.approx([0.04, 0.16, 0.12, 0.8])
    holm = multiple_comparison(p, AdjustMethod.HOLM)
    # sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08 -> 0.09 (monotone), 0.2*1 -> 0.2
    assert holm.tolist() == pytest.approx([0.04, 0.09, 0.09, 0.2])
    bh = multiple_comparison(p, AdjustMethod.BH)
    assert bh.tolist() == pytest.approx(
        stats.false_discovery_control(p, method="bh").tolist()
    )
    assert multiple_comparison([0.5], AdjustMethod.HOLM).tolist() == [0.5]
    assert multiple_comparison([0.9, 0.9], AdjustMethod.BONFERRONI).max() == 1.0
