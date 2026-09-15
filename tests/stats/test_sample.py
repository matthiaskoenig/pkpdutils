import numpy as np
import pytest
from scipy.stats import t as student_t

from pkpdutils.stats.sample import (
    ParameterSample,
    Scale,
    Summary,
    lognormal_from_geometric,
    lognormal_from_moments,
    moments_from_lognormal,
    paired_indices,
    paired_values,
    summarize,
)

VALUES = np.array([80.0, 95.0, 110.0, 120.0, 150.0, np.nan])


def test_lognormal_relations_round_trip() -> None:
    mu, sigma = lognormal_from_moments(100.0, 30.0)
    assert sigma == pytest.approx(np.sqrt(np.log1p(0.09)))
    assert mu == pytest.approx(np.log(100.0) - sigma**2 / 2)
    mean, sd = moments_from_lognormal(mu, sigma)
    assert (mean, sd) == pytest.approx((100.0, 30.0))
    mu2, sigma2 = lognormal_from_geometric(95.78262852211522, 0.3)
    assert (mu2, sigma2) == pytest.approx((mu, sigma))


def test_individual_sample() -> None:
    s = ParameterSample(values=VALUES, name="auc_inf_obs", unit="mg*hr/l")
    assert s.is_individual and s.size == 5
    assert s.finite_values.tolist() == VALUES[:5].tolist()
    mu, sigma = s.log_moments()
    logs = np.log(VALUES[:5])
    assert (mu, sigma) == pytest.approx((logs.mean(), logs.std(ddof=1)))
    mean, sd = s.linear_moments()
    assert (mean, sd) == pytest.approx((VALUES[:5].mean(), VALUES[:5].std(ddof=1)))
    assert s.moments(Scale.LINEAR) == pytest.approx((mean, sd, 5))
    assert s.moments(Scale.LOG) == pytest.approx((mu, sigma, 5))


def test_labels_and_coords_follow_the_values() -> None:
    s = ParameterSample(
        values=np.array([1.0, 2.0, np.nan]),
        labels=np.array(["a", "b", "c"]),
        coords={"period": np.array([1, 2, 1])},
    )
    assert s.finite_labels is not None and s.finite_labels.tolist() == ["a", "b"]
    sub = s.select(np.array([True, False, True]))
    assert sub.labels is not None and sub.labels.tolist() == ["a", "c"]
    assert sub.coords["period"].tolist() == [1, 1]
    with pytest.raises(ValueError, match="length"):
        ParameterSample(values=np.array([1.0, 2.0]), labels=np.array(["a"]))
    with pytest.raises(ValueError, match="length"):
        ParameterSample(values=np.array([1.0, 2.0]), coords={"p": np.array([1])})


def test_summary_sample_moments() -> None:
    s = ParameterSample(mean=100.0, sd=30.0, n=12)
    assert not s.is_individual and s.size == 12
    assert s.finite_values.size == 0
    assert s.log_moments() == pytest.approx(lognormal_from_moments(100.0, 30.0))
    assert s.linear_moments() == (100.0, 30.0)
    g = ParameterSample(mean=100.0, sd=30.0, n=12, geomean=96.0, geocv=0.28)
    assert g.log_moments() == pytest.approx(lognormal_from_geometric(96.0, 0.28))
    only_geo = ParameterSample(geomean=96.0, geocv=0.28, n=12)
    mu, sigma = only_geo.log_moments()
    assert only_geo.linear_moments() == pytest.approx(moments_from_lognormal(mu, sigma))


def test_sample_validation() -> None:
    with pytest.raises(ValueError, match="'values' or"):
        ParameterSample()
    with pytest.raises(ValueError, match="n"):
        ParameterSample(mean=1.0, sd=0.1)
    with pytest.raises(ValueError, match="1-D"):
        ParameterSample(values=np.ones((2, 2)))
    with pytest.raises(ValueError, match="positive"):
        _ = ParameterSample(values=np.array([1.0, -1.0])).log_values
    with pytest.raises(ValueError, match="individual"):
        ParameterSample(mean=1.0, sd=0.1, n=3).select(np.array([True]))


def test_summarize_individual_log_scale() -> None:
    summary = summarize(VALUES, scale=Scale.LOG, name="cl", unit="l/hr")
    assert isinstance(summary, Summary)
    v = VALUES[:5]
    logs = np.log(v)
    assert summary.n == 5
    assert summary.mean == pytest.approx(v.mean())
    assert summary.sd == pytest.approx(v.std(ddof=1))
    assert summary.se == pytest.approx(v.std(ddof=1) / np.sqrt(5))
    assert summary.cv == pytest.approx(v.std(ddof=1) / v.mean())
    assert summary.geomean == pytest.approx(np.exp(logs.mean()))
    assert summary.geocv == pytest.approx(np.sqrt(np.expm1(logs.var(ddof=1))))
    assert summary.median == pytest.approx(np.median(v))
    assert (summary.q25, summary.q75) == pytest.approx(
        tuple(np.percentile(v, [25, 75]))
    )
    assert (summary.min, summary.max) == (80.0, 150.0)
    tq = student_t.ppf(0.975, 4)
    assert summary.ci_low == pytest.approx(
        np.exp(logs.mean() - tq * logs.std(ddof=1) / np.sqrt(5))
    )
    assert summary.ci_high == pytest.approx(
        np.exp(logs.mean() + tq * logs.std(ddof=1) / np.sqrt(5))
    )
    assert (
        summary.scale is Scale.LOG and summary.name == "cl" and summary.unit == "l/hr"
    )
    assert summary.to_dict()["geomean"] == summary.geomean


def test_summarize_linear_scale_interval() -> None:
    summary = summarize(VALUES, scale=Scale.LINEAR)
    v = VALUES[:5]
    tq = student_t.ppf(0.975, 4)
    se = v.std(ddof=1) / np.sqrt(5)
    assert (summary.ci_low, summary.ci_high) == pytest.approx(
        (v.mean() - tq * se, v.mean() + tq * se)
    )


def test_summarize_summary_data() -> None:
    summary = summarize(ParameterSample(mean=100.0, sd=30.0, n=12), scale=Scale.LOG)
    mu, sigma = lognormal_from_moments(100.0, 30.0)
    assert summary.n == 12 and summary.mean == 100.0 and summary.sd == 30.0
    assert summary.geomean == pytest.approx(np.exp(mu))
    assert summary.geocv == pytest.approx(np.sqrt(np.expm1(sigma**2)))
    assert np.isnan(summary.median) and np.isnan(summary.min)
    tq = student_t.ppf(0.975, 11)
    assert summary.ci_low == pytest.approx(np.exp(mu - tq * sigma / np.sqrt(12)))


def test_summarize_single_value() -> None:
    summary = summarize([3.0])
    assert summary.n == 1 and summary.mean == 3.0 and np.isnan(summary.sd)
    assert np.isnan(summary.ci_low) and np.isnan(summary.geocv)


def test_paired_values_by_label_ignores_the_order() -> None:
    a = ParameterSample(
        values=np.array([1.0, 2.0, 3.0]), labels=np.array(["s0", "s1", "s2"])
    )
    b = ParameterSample(
        values=np.array([30.0, 10.0, 20.0]), labels=np.array(["s2", "s0", "s1"])
    )
    x, y = paired_values(a, b)
    assert x.tolist() == [1.0, 2.0, 3.0]
    assert y.tolist() == [10.0, 20.0, 30.0]
    index_a, index_b = paired_indices(a, b)
    assert index_a.tolist() == [0, 1, 2] and index_b.tolist() == [1, 2, 0]


def test_paired_values_by_position_without_labels() -> None:
    a = ParameterSample(values=np.array([1.0, 2.0]))
    b = ParameterSample(values=np.array([10.0, 20.0]))
    x, y = paired_values(a, b)
    assert x.tolist() == [1.0, 2.0] and y.tolist() == [10.0, 20.0]
    # only one sample is labelled: the pairing is by position as well
    labelled = ParameterSample(values=np.array([1.0, 2.0]), labels=np.array(["b", "a"]))
    x2, y2 = paired_values(labelled, b)
    assert x2.tolist() == [1.0, 2.0] and y2.tolist() == [10.0, 20.0]


def test_paired_values_drops_a_pair_with_a_missing_value() -> None:
    labels = np.array(["s0", "s1", "s2"])
    a = ParameterSample(values=np.array([100.0, np.nan, 300.0]), labels=labels)
    b = ParameterSample(values=np.array([np.nan, 200.0, 300.0]), labels=labels)
    x, y = paired_values(a, b)
    assert x.tolist() == [300.0] and y.tolist() == [300.0]
    index_a, index_b = paired_indices(a, b)
    assert index_a.tolist() == [2] and index_b.tolist() == [2]
    # without labels the same pairs are dropped, by position
    x2, y2 = paired_values(
        ParameterSample(values=a.values), ParameterSample(values=b.values)
    )
    assert x2.tolist() == [300.0] and y2.tolist() == [300.0]


def test_paired_values_keeps_only_the_shared_labels() -> None:
    a = ParameterSample(
        values=np.array([1.0, 2.0, 3.0]), labels=np.array(["s0", "s1", "s2"])
    )
    b = ParameterSample(values=np.array([20.0, 30.0]), labels=np.array(["s1", "s2"]))
    x, y = paired_values(a, b)
    assert x.tolist() == [2.0, 3.0] and y.tolist() == [20.0, 30.0]


def test_paired_values_logs_the_dropped_pairs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    labels = np.array(["s0", "s1"])
    a = ParameterSample(values=np.array([1.0, np.nan]), labels=labels, name="auc")
    b = ParameterSample(values=np.array([10.0, 20.0]), labels=labels, name="auc")
    with caplog.at_level("DEBUG", logger="pkpdutils.stats.sample"):
        paired_values(a, b)
    assert "1 of 2 pairs of 'auc' and 'auc' dropped: missing values" in caplog.text


def test_paired_values_errors() -> None:
    a = ParameterSample(values=np.array([1.0, 2.0]), labels=np.array(["s0", "s1"]))
    disjoint = ParameterSample(values=np.array([1.0, 2.0]), labels=np.array(["x", "y"]))
    with pytest.raises(ValueError, match="labels"):
        paired_values(a, disjoint)
    duplicate = ParameterSample(
        values=np.array([1.0, 2.0]), labels=np.array(["s0", "s0"]), name="dup"
    )
    with pytest.raises(ValueError, match="duplicate labels"):
        paired_values(a, duplicate)
    with pytest.raises(ValueError, match="duplicate labels"):
        paired_values(duplicate, a)
    with pytest.raises(ValueError, match="equal sizes"):
        paired_values(
            ParameterSample(values=np.array([1.0, 2.0])),
            ParameterSample(values=np.array([1.0])),
        )
    with pytest.raises(ValueError, match="no pair of finite values"):
        paired_values(
            ParameterSample(values=np.array([1.0, np.nan])),
            ParameterSample(values=np.array([np.nan, 2.0])),
        )
    with pytest.raises(ValueError, match="needs individual data"):
        paired_values(ParameterSample(mean=1.0, sd=0.1, n=3), a)
    with pytest.raises(ValueError, match="needs individual data"):
        paired_values(a, ParameterSample(mean=1.0, sd=0.1, n=3))


def test_summarize_non_positive_value_linear_vs_log() -> None:
    non_positive = np.array([-1.0, 2.0, 3.0])
    summary = summarize(non_positive, scale=Scale.LINEAR)
    assert summary.mean == pytest.approx(4.0 / 3.0)
    assert np.isnan(summary.geomean) and np.isnan(summary.geocv)
    assert np.isfinite(summary.ci_low) and np.isfinite(summary.ci_high)
    with pytest.raises(ValueError, match="positive"):
        summarize(non_positive)
