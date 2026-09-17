"""`NCAResult.exclude` and the analyses which skip an excluded sample."""

import numpy as np
import pytest

from pkpdutils import (
    AUCMethod,
    NCAOptions,
    NCAResult,
    Route,
    Timecourses,
    nca,
    summary_table,
)
from pkpdutils.stats import ddi_table, ratio, ratio_table

TIME = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0])
LABELS = ["s1", "s2", "s3", "s4"]
OPTIONS = NCAOptions(auc_method=AUCMethod.LOG)


def batch(scale: float = 1.0) -> Timecourses:
    """Four bolus curves of different clearance, the same sampling grid."""
    k = np.array([0.2, 0.25, 0.3, 0.35])
    values = scale * 10.0 * np.exp(-k[:, None] * TIME[None, :])
    return Timecourses.from_arrays(
        TIME,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": LABELS},
        dose={"amount": np.full(4, 100.0), "unit": "mg"},
        route=Route.IV_BOLUS,
        substance="drug",
    )


def result(scale: float = 1.0) -> NCAResult:
    return nca(batch(scale), options=OPTIONS)


def test_exclude_by_mask_and_by_indexer_mark_the_same_sample() -> None:
    plain = result()
    by_mask = plain.exclude(np.array([False, False, True, False]), reason="outlier")
    by_label = plain.exclude(individual="s3", reason="outlier")
    assert by_mask["excluded"].to_numpy().tolist() == [False, False, True, False]
    assert (
        by_mask["excluded"].to_numpy().tolist()
        == by_label["excluded"].to_numpy().tolist()
    )
    assert by_label["excluded_reason"].to_numpy().tolist() == ["", "", "outlier", ""]
    # the original result is untouched
    assert plain["excluded"].to_numpy().tolist() == [False] * 4
    # an excluded sample keeps its reason when a second call marks more
    more = by_label.exclude(individual="s1", reason="protocol deviation")
    assert more["excluded_reason"].to_numpy().tolist() == [
        "protocol deviation",
        "",
        "outlier",
        "",
    ]


def test_exclude_needs_a_mask_or_indexers() -> None:
    plain = result()
    with pytest.raises(ValueError, match="either 'mask'"):
        plain.exclude()
    with pytest.raises(ValueError, match="either 'mask'"):
        plain.exclude(np.zeros(4, dtype=bool), individual="s1")
    with pytest.raises(ValueError, match="has the shape"):
        plain.exclude(np.zeros(3, dtype=bool))


def crossover() -> NCAResult:
    """The curves of `batch` under a reference and a test treatment."""
    reference = batch()
    return nca(
        Timecourses.from_arrays(
            TIME,
            np.stack([reference.values, 1.4 * reference.values]),
            time_unit="hr",
            unit="mg/l",
            dims=("treatment", "individual"),
            coords={"treatment": ["R", "T"], "individual": LABELS},
            dose={"amount": np.full((2, 4), 100.0), "unit": "mg"},
            route=Route.IV_BOLUS,
            substance="drug",
        ),
        options=OPTIONS,
    )


def test_exclude_and_sample_by_the_labels_of_two_sample_dimensions() -> None:
    marked = crossover().exclude(treatment="T", individual="s3")
    assert marked["excluded"].to_numpy().tolist() == [
        [False, False, False, False],
        [False, False, True, False],
    ]
    test = marked.sample("cmax", "individual", treatment="T")
    reference = marked.sample("cmax", "individual", treatment="R")
    assert test.labels is not None and test.labels.tolist() == ["s1", "s2", "s4"]
    assert reference.labels is not None and reference.labels.tolist() == LABELS
    # a dimension without an indexer is excluded as a whole, a list of labels
    # excludes each of them
    both = crossover().exclude(individual=["s1", "s3"])
    assert both["excluded"].to_numpy().tolist() == [[True, False, True, False]] * 2
    plain = crossover()
    by_coordinate = plain.exclude(individual=plain["cmax"]["individual"][[0, 2]])
    assert (
        by_coordinate["excluded"].to_numpy().tolist()
        == [[True, False, True, False]] * 2
    )


def test_exclude_and_sample_reject_an_unknown_indexer_or_label() -> None:
    """A misspelled dimension or label is a `ValueError` naming it (#70)."""
    plain = crossover()
    with pytest.raises(ValueError, match="'subject' is not a sample dimension"):
        plain.exclude(subject="s1")
    with pytest.raises(
        ValueError, match="'s9' is not a label of the sample dimension 'individual'"
    ):
        plain.exclude(individual="s9")
    with pytest.raises(ValueError, match="'s9' is not a label"):
        plain.exclude(individual=["s1", "s9"])
    with pytest.raises(ValueError, match="one label or a list of labels"):
        plain.exclude(individual=slice("s1", "s2"))
    with pytest.raises(ValueError, match="'foo' is not a sample dimension"):
        plain.sample("auc_inf_obs", "individual", treatment="T", foo=1)
    with pytest.raises(ValueError, match="takes no indexer"):
        plain.sample("auc_inf_obs", "individual", treatment="T", individual="s1")
    with pytest.raises(
        ValueError, match="'X' is not a label of the sample dimension 'treatment'"
    ):
        plain.sample("auc_inf_obs", "individual", treatment="X")


def test_to_dataframe_keeps_every_row_and_carries_the_excluded_column() -> None:
    marked = result().exclude(individual="s3", reason="outlier")
    frame = marked.to_dataframe()
    assert len(frame) == 4
    assert frame["excluded"].tolist() == [False, False, True, False]
    assert frame["excluded_reason"].tolist() == ["", "", "outlier", ""]
    # the status columns sit between the parameters and the flags
    columns = list(frame.columns)
    assert columns.index("cmax") < columns.index("excluded") < columns.index("flags")


def test_summarize_skips_the_excluded_sample() -> None:
    plain = result()
    marked = plain.exclude(individual="s3")
    kept = [label for label in LABELS if label != "s3"]
    values = np.array(
        [float(plain["auc_inf_obs"].sel(individual=label)) for label in kept]
    )
    summary = marked.summarize("individual")
    assert float(summary["n"]) == 3.0
    assert float(summary["auc_inf_obs_n"]) == 3.0
    assert float(summary["auc_inf_obs"]) == pytest.approx(values.mean())
    assert float(summary["auc_inf_obs_sd"]) == pytest.approx(values.std(ddof=1))
    # with `include_excluded` every sample is summarized again
    everything = marked.summarize("individual", include_excluded=True)
    assert float(everything["n"]) == 4.0
    assert float(everything["auc_inf_obs"]) == pytest.approx(
        float(plain.summarize("individual")["auc_inf_obs"])
    )


def test_summary_table_skips_the_excluded_sample() -> None:
    marked = result().exclude(individual="s3")
    table = summary_table(marked, "individual", parameters=["cmax"], stats=("n",))
    assert table["n"].tolist() == ["3"]
    full = marked.summary_table(
        "individual", parameters=["cmax"], stats=("n",), include_excluded=True
    )
    assert full["n"].tolist() == ["4"]


def test_sample_ratio_table_and_ddi_table_skip_the_excluded_sample() -> None:
    test = result(scale=1.4).exclude(individual="s3")
    reference = result()
    sample = test.sample("auc_inf_obs", "individual")
    assert sample.labels is not None
    assert sample.labels.tolist() == ["s1", "s2", "s4"]
    everything_labels = test.sample(
        "auc_inf_obs", "individual", include_excluded=True
    ).labels
    assert everything_labels is not None
    assert everything_labels.tolist() == LABELS

    # the ratio pairs by label, so the excluded subject drops out of the pair
    ratios = {
        name: ratio(
            test.sample(name, "individual"), reference.sample(name, "individual")
        )
        for name in ("auc_inf_obs", "cmax")
    }
    table = ratio_table(ratios)
    assert table["n_test"].tolist() == ["3", "3"]
    assert table["n_reference"].tolist() == ["3", "3"]
    # the ratio itself is the scale of the test curves
    assert float(ratios["cmax"].gmr) == pytest.approx(1.4)

    interaction = ddi_table(test, reference, ["auc_inf_obs"], dim="individual")
    assert interaction["n_test"].tolist() == ["3"]
    everything = ddi_table(
        test, reference, ["auc_inf_obs"], dim="individual", include_excluded=True
    )
    assert everything["n_test"].tolist() == ["4"]
