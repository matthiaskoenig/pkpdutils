"""The Hodges-Lehmann estimator and its distribution free interval."""

import itertools

import numpy as np
import pytest
from scipy import stats

from pkpdutils.stats import ParameterSample, hodges_lehmann
from pkpdutils.stats.tests import (
    TestMethod,
    _mann_whitney_counts,
    _signed_rank_counts,
)

#: four paired subjects with the differences 1, 2, 4 and 8
LABELS = np.array(["s1", "s2", "s3", "s4"])
TEST = np.array([2.0, 3.0, 5.0, 9.0])
REFERENCE = np.array([1.0, 1.0, 1.0, 1.0])


def sample(values: np.ndarray, *, labels: np.ndarray | None = None) -> ParameterSample:
    """A sample of `tmax` values, labelled or not."""
    return ParameterSample(values=values, labels=labels, name="tmax", unit="hour")


def walsh(d: np.ndarray) -> np.ndarray:
    """The Walsh averages of a sample, by hand."""
    return np.sort(
        np.array([(d[i] + d[j]) / 2.0 for i in range(d.size) for j in range(i, d.size)])
    )


def test_the_estimate_is_the_median_of_the_hand_computed_walsh_averages() -> None:
    """Four pairs: ten Walsh averages, the estimate their median."""
    averages = walsh(TEST - REFERENCE)
    assert averages.tolist() == [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 6.0, 8.0]
    result = hodges_lehmann(
        sample(TEST, labels=LABELS), sample(REFERENCE, labels=LABELS)
    )
    assert result.paired is True
    assert result.test is TestMethod.WILCOXON
    assert result.effect == pytest.approx(3.5)
    # with four pairs no 90 % interval exists, the extreme averages are reported
    assert (result.ci_low, result.ci_high) == (1.0, 8.0)
    assert result.n_a == 4 and result.n_b == 4
    assert np.isnan(result.df)


def test_the_paired_p_value_is_the_wilcoxon_test_of_scipy() -> None:
    """The p value is the matching rank test, not one of its own."""
    result = hodges_lehmann(
        sample(TEST, labels=LABELS), sample(REFERENCE, labels=LABELS)
    )
    expected = stats.wilcoxon(TEST, REFERENCE)
    assert result.statistic == pytest.approx(float(expected.statistic))
    assert result.p_value == pytest.approx(float(expected.pvalue))


def test_the_unpaired_estimate_is_the_median_of_the_pairwise_differences() -> None:
    """Two independent samples: the median of every difference `a - b`."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 3.5, 4.5, 6.0, 7.5])
    result = hodges_lehmann(sample(a), sample(b))
    differences = np.sort((a[:, None] - b[None, :]).ravel())
    assert result.paired is False
    assert result.test is TestMethod.MANN_WHITNEY
    assert result.effect == pytest.approx(float(np.median(differences)))
    expected = stats.mannwhitneyu(a, b)
    assert result.p_value == pytest.approx(float(expected.pvalue))
    assert result.ci_low in differences and result.ci_high in differences
    assert result.ci_low <= result.effect <= result.ci_high


def test_the_interval_is_the_one_r_reports_for_a_paired_sample() -> None:
    """Twelve pairs, the order statistics R's `wilcox.test(conf.int=TRUE)` takes.

    With `n = 12` and `alpha = 0.05`, `qsignrank(0.05, 12) = 18`, so the
    interval runs from the 18th smallest to the 18th largest of the 78 Walsh
    averages.
    """
    differences = np.array(
        [0.5, -0.25, 1.0, 0.75, -0.5, 1.25, 0.25, 2.0, 1.5, -0.75, 0.0, 1.75]
    )
    base = np.full(12, 4.0)
    labels = np.array([f"s{i:02d}" for i in range(12)])
    result = hodges_lehmann(
        sample(base + differences, labels=labels), sample(base, labels=labels)
    )
    counts = _signed_rank_counts(12)
    rank = int(np.searchsorted(np.cumsum(counts) / counts.sum(), 0.05))
    assert rank == 18
    averages = walsh(differences)
    assert result.ci_low == pytest.approx(averages[rank - 1])
    assert result.ci_high == pytest.approx(averages[averages.size - rank])


@pytest.mark.parametrize("n", [3, 5, 8])
def test_the_signed_rank_counts_are_the_enumerated_subsets(n: int) -> None:
    """The null distribution of the signed rank statistic, against brute force."""
    expected = np.zeros(n * (n + 1) // 2 + 1)
    for size in range(n + 1):
        for subset in itertools.combinations(range(1, n + 1), size):
            expected[sum(subset)] += 1
    assert _signed_rank_counts(n) == pytest.approx(expected)


@pytest.mark.parametrize(("m", "n"), [(3, 4), (5, 5), (6, 4)])
def test_the_mann_whitney_counts_are_the_enumerated_rank_sums(m: int, n: int) -> None:
    """The null distribution of `U`, against the enumeration of every assignment."""
    expected = np.zeros(m * n + 1)
    for ranks in itertools.combinations(range(m + n), m):
        expected[sum(ranks) - m * (m - 1) // 2] += 1
    assert _mann_whitney_counts(m, n) == pytest.approx(expected)


def test_a_shift_moves_the_estimate_and_the_interval_by_the_same_amount() -> None:
    """The estimator is equivariant under a shift of one sample."""
    labels = np.array([f"s{i}" for i in range(9)])
    a = np.array([1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 4.0, 6.0])
    b = np.array([1.0, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 3.0, 4.0])
    first = hodges_lehmann(sample(a, labels=labels), sample(b, labels=labels))
    second = hodges_lehmann(sample(a + 2.5, labels=labels), sample(b, labels=labels))
    assert second.effect == pytest.approx(first.effect + 2.5)
    assert second.ci_low == pytest.approx(first.ci_low + 2.5)
    assert second.ci_high == pytest.approx(first.ci_high + 2.5)


def test_identical_paired_samples_have_a_zero_effect_and_no_evidence() -> None:
    """Every difference zero: the rank test has nothing to rank."""
    labels = np.array(["s1", "s2", "s3", "s4", "s5"])
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = hodges_lehmann(
        sample(values, labels=labels), sample(values, labels=labels)
    )
    assert result.effect == 0.0
    assert result.p_value == 1.0
    assert result.statistic == 0.0


def test_the_pairing_is_detected_from_the_labels() -> None:
    """Shared labels make the analysis paired, different ones do not."""
    paired = hodges_lehmann(
        sample(TEST, labels=LABELS), sample(REFERENCE, labels=LABELS)
    )
    unpaired = hodges_lehmann(sample(TEST), sample(REFERENCE))
    assert paired.paired is True
    assert unpaired.paired is False
    forced = hodges_lehmann(
        sample(TEST, labels=LABELS), sample(REFERENCE, labels=LABELS), paired=False
    )
    assert forced.paired is False


def test_a_tied_tmax_sample_is_analysed_without_a_warning() -> None:
    """A parameter read from a sampling grid has ties; scipy's fallback is logged."""
    labels = np.array([f"s{i:02d}" for i in range(10)])
    test = np.array([2.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0])
    reference = np.array([1.0, 2.0, 2.0, 1.0, 2.0, 3.0, 2.0, 2.0, 1.0, 3.0])
    result = hodges_lehmann(
        sample(test, labels=labels), sample(reference, labels=labels)
    )
    # the differences are 1, 0, 1, 1, 1, 1, 1, 0, 2, 1, their Walsh median 1
    assert result.effect == pytest.approx(1.0)
    assert np.isfinite(result.ci_low) and np.isfinite(result.ci_high)
    assert 0.0 <= result.p_value <= 1.0


def test_the_normal_approximation_takes_over_above_the_exact_limit() -> None:
    """A large sample uses the approximation and stays close to the exact answer."""
    rng = np.random.default_rng(4)
    labels = np.array([f"s{i:03d}" for i in range(60)])
    reference = rng.uniform(1.0, 5.0, 60)
    test = reference + 0.4
    result = hodges_lehmann(
        sample(test, labels=labels), sample(reference, labels=labels)
    )
    assert result.effect == pytest.approx(0.4)
    assert result.ci_low == pytest.approx(0.4)
    assert result.ci_high == pytest.approx(0.4)


def test_summary_data_and_a_bad_level_are_rejected() -> None:
    """The estimator needs the individual values and a level which is one."""
    summary = ParameterSample(mean=3.0, sd=1.0, n=10, name="tmax", unit="hour")
    with pytest.raises(ValueError, match="individual data"):
        hodges_lehmann(summary, sample(TEST))
    with pytest.raises(ValueError, match="must lie in"):
        hodges_lehmann(sample(TEST), sample(REFERENCE), ci_level=1.5)


def test_a_sample_without_a_finite_value_gives_a_nan_result() -> None:
    """An empty sample is reported as `NaN`, not as an error."""
    result = hodges_lehmann(sample(np.array([np.nan, np.nan])), sample(REFERENCE))
    assert np.isnan(result.effect)
    assert np.isnan(result.p_value)
    assert result.n_a == 0
