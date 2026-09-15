import numpy as np
import pytest
from scipy import stats
from scipy.stats import t as student_t

from pkpdutils.stats import ParameterSample
from pkpdutils.stats.ratio import RatioResult, ratio

RNG = np.random.default_rng(11)
REF = np.exp(RNG.normal(np.log(100), 0.25, 12))
TEST = REF * np.exp(RNG.normal(np.log(0.95), 0.12, 12))
LABELS = np.array([f"s{i}" for i in range(12)])


def test_paired_by_label_order_independent() -> None:
    t = ParameterSample(values=TEST, labels=LABELS, name="auc_inf_obs", unit="mg*hr/l")
    perm = RNG.permutation(12)
    r = ParameterSample(values=REF[perm], labels=LABELS[perm])
    res = ratio(t, r)
    assert isinstance(res, RatioResult) and res.paired
    d = np.log(TEST) - np.log(REF)
    se = d.std(ddof=1) / np.sqrt(12)
    tq = student_t.ppf(0.95, 11)
    assert res.gmr == pytest.approx(np.exp(d.mean()))
    assert res.log_ratio == pytest.approx(d.mean())
    assert (res.ci_low, res.ci_high) == pytest.approx(
        (np.exp(d.mean() - tq * se), np.exp(d.mean() + tq * se))
    )
    assert res.se_log == pytest.approx(se) and res.df == 11
    assert res.n_test == 12 and res.n_reference == 12 and res.ci_level == 0.90
    assert res.name == "auc_inf_obs" and res.unit == "mg*hr/l"


def test_unpaired_is_welch_on_logs() -> None:
    res = ratio(ParameterSample(values=TEST), ParameterSample(values=REF[:10]))
    assert not res.paired
    ref = stats.ttest_ind(np.log(TEST), np.log(REF[:10]), equal_var=False)
    low, high = ref.confidence_interval(0.90)
    assert (res.ci_low, res.ci_high) == pytest.approx((np.exp(low), np.exp(high)))
    assert res.df == pytest.approx(ref.df)
    assert res.gmr == pytest.approx(
        np.exp(np.log(TEST).mean() - np.log(REF[:10]).mean())
    )


def test_paired_rules() -> None:
    with pytest.raises(ValueError, match="equal sizes"):
        ratio(
            ParameterSample(values=TEST), ParameterSample(values=REF[:10]), paired=True
        )
    by_position = ratio(
        ParameterSample(values=TEST), ParameterSample(values=REF), paired=True
    )
    assert by_position.paired and by_position.df == 11
    forced_unpaired = ratio(
        ParameterSample(values=TEST, labels=LABELS),
        ParameterSample(values=REF, labels=LABELS),
        paired=False,
    )
    assert not forced_unpaired.paired
    other_labels = ParameterSample(
        values=REF, labels=np.array([f"x{i}" for i in range(12)])
    )
    assert not ratio(ParameterSample(values=TEST, labels=LABELS), other_labels).paired
    with pytest.raises(ValueError, match="labels"):
        ratio(ParameterSample(values=TEST, labels=LABELS), other_labels, paired=True)


def test_summary_data() -> None:
    t = ParameterSample(geomean=95.0, geocv=0.3, n=12)
    r = ParameterSample(mean=100.0, sd=25.0, n=12)
    res = ratio(t, r)
    mu_t, s_t = t.log_moments()
    mu_r, s_r = r.log_moments()
    # scipy's ttest_ind_from_stats does not expose df, so the Welch-Satterthwaite
    # degrees of freedom are computed here directly, independent of the implementation
    var_t, var_r = s_t**2 / 12, s_r**2 / 12
    expected_df = (var_t + var_r) ** 2 / (var_t**2 / 11 + var_r**2 / 11)
    assert res.gmr == pytest.approx(np.exp(mu_t - mu_r))
    assert res.df == pytest.approx(expected_df)
    assert res.to_dict()["gmr"] == res.gmr
    with pytest.raises(ValueError, match="individual"):
        ratio(t, ParameterSample(values=REF), paired=True)


def test_paired_drops_only_the_pair_with_a_missing_value() -> None:
    labels = np.array(["s0", "s1", "s2"])
    t = ParameterSample(values=np.array([100.0, np.nan, 300.0]), labels=labels)
    r = ParameterSample(values=np.array([np.nan, 200.0, 300.0]), labels=labels)
    res = ratio(t, r)
    assert res.paired
    assert res.gmr == pytest.approx(1.0) and res.log_ratio == pytest.approx(0.0)
    assert res.n_test == 1 and res.n_reference == 1
    assert np.isnan(res.se_log) and np.isnan(res.df)
    assert np.isnan(res.ci_low) and np.isnan(res.ci_high)


def test_unpaired_single_value_gives_nan_statistics() -> None:
    res = ratio(ParameterSample(values=np.array([100.0])), ParameterSample(values=REF))
    assert not res.paired and res.n_test == 1
    assert res.gmr == pytest.approx(100.0 / np.exp(np.log(REF).mean()))
    assert np.isnan(res.se_log) and np.isnan(res.df)
    assert np.isnan(res.ci_low) and np.isnan(res.ci_high)
    summary = ratio(
        ParameterSample(mean=100.0, sd=20.0, n=1), ParameterSample(values=REF)
    )
    assert np.isnan(summary.df) and np.isnan(summary.ci_low)
    assert np.isfinite(summary.gmr)


def test_paired_needs_finite_pairs() -> None:
    t = ParameterSample(values=np.array([np.nan, np.nan]), labels=np.array(["a", "b"]))
    r = ParameterSample(values=np.array([np.nan, np.nan]), labels=np.array(["a", "b"]))
    with pytest.raises(ValueError, match="finite"):
        ratio(t, r)
    unlabelled_t = ParameterSample(values=np.array([np.nan, np.nan]))
    unlabelled_r = ParameterSample(values=np.array([np.nan, np.nan]))
    with pytest.raises(ValueError, match="finite"):
        ratio(unlabelled_t, unlabelled_r, paired=True)
