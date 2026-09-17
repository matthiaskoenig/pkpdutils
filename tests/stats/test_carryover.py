"""The pre-dose carryover check of a bioequivalence study."""

import numpy as np
import pytest

from pkpdutils import (
    NCAOptions,
    Route,
    Timecourses,
    bioequivalence,
    carryover_table,
    nca,
)

TIME = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])
LABELS = ["s1", "s2", "s3", "s4", "s5", "s6"]
#: the subject which carries drug from the previous period
CARRIER = "s3"
OPTIONS = NCAOptions()


def profile(scale: float, ka: float = 1.2, ke: float = 0.2) -> np.ndarray:
    """One extravascular profile on `TIME`."""
    return scale * 10.0 * (np.exp(-ke * TIME) - np.exp(-ka * TIME))


def period(*, scale: float, predose_fraction: float = 0.06) -> Timecourses:
    """Six subjects, the pre-dose value of `CARRIER` a share of its own maximum."""
    values = np.stack([profile(scale * (1.0 + 0.05 * i)) for i in range(6)])
    row = LABELS.index(CARRIER)
    values[row, 0] = predose_fraction * values[row].max()
    return Timecourses.from_arrays(
        TIME,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": LABELS},
        dose={"amount": np.full(6, 100.0), "unit": "mg"},
        route=Route.ORAL,
        substance="drug",
    )


def test_the_table_reports_the_predose_value_against_the_maximum() -> None:
    batch = period(scale=1.0)
    result = nca(batch, options=OPTIONS)
    table = carryover_table(batch, result)
    assert list(table.columns) == [
        "individual",
        "predose",
        "cmax",
        "fraction",
        "flagged",
    ]
    assert table["individual"].tolist() == LABELS
    row = table.set_index("individual").loc[CARRIER]
    assert float(row["fraction"]) == pytest.approx(0.06)
    assert bool(row["flagged"]) is True
    assert table["flagged"].sum() == 1
    # every other subject has a pre-dose value of 0
    assert table.loc[table["individual"] != CARRIER, "fraction"].to_numpy() == (
        pytest.approx(0.0)
    )
    # 6 % is above the 5 % of ICH M13A but below a threshold of 10 %
    assert carryover_table(batch, result, threshold=0.10)["flagged"].sum() == 0


def test_the_value_at_the_dose_of_a_bolus_is_not_a_predose_value() -> None:
    """After a bolus the sample at the dose time carries the post-dose value."""
    values = np.stack([10.0 * np.exp(-0.2 * TIME) for _ in range(3)])
    bolus = Timecourses.from_arrays(
        TIME,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b", "c"]},
        dose={"amount": np.full(3, 100.0), "unit": "mg"},
        route=Route.IV_BOLUS,
        substance="drug",
    )
    table = carryover_table(bolus, nca(bolus, options=OPTIONS))
    # the value at t = 0 is Cmax itself, reading it would flag every subject
    assert np.isnan(table["predose"].to_numpy()).all()
    assert table["flagged"].tolist() == [False, False, False]


def test_a_sample_strictly_before_the_dose_is_the_predose_value() -> None:
    """A bolus schedule with a sample before the dose reports that value."""
    time = np.concatenate([[-0.5], TIME])
    values = np.stack(
        [
            np.concatenate([[carry], 10.0 * np.exp(-0.2 * TIME)])
            for carry in (0.0, 0.8, 0.2)
        ]
    )
    bolus = Timecourses.from_arrays(
        time,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b", "c"]},
        dose={"amount": np.full(3, 100.0), "unit": "mg"},
        route=Route.IV_BOLUS,
        substance="drug",
    )
    table = carryover_table(bolus, nca(bolus, options=OPTIONS))
    assert table["predose"].tolist() == [0.0, 0.8, 0.2]
    # Cmax is 10, so 0.8 is 8 % and 0.2 is 2 % of it
    assert table["fraction"].to_numpy() == pytest.approx([0.0, 0.08, 0.02])
    assert table["flagged"].tolist() == [False, True, False]


def test_a_batch_without_a_predose_sample_flags_nothing() -> None:
    batch = Timecourses.from_arrays(
        TIME[1:],
        np.stack([profile(1.0)[1:] for _ in range(2)]),
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b"]},
        dose={"amount": np.full(2, 100.0), "unit": "mg"},
        route=Route.ORAL,
        substance="drug",
    )
    table = carryover_table(batch, nca(batch, options=OPTIONS))
    assert np.isnan(table["predose"].to_numpy()).all()
    assert table["flagged"].tolist() == [False, False]


def test_bioequivalence_flags_and_excludes_the_carryover_subject() -> None:
    test_batch = period(scale=0.95)
    reference_batch = period(scale=1.0)
    test = nca(test_batch, options=OPTIONS)
    reference = nca(reference_batch, options=OPTIONS)

    ignored = bioequivalence(test, reference, ["auc_inf_obs", "cmax"])
    assert ignored["cmax"].n_test == 6
    assert ignored["cmax"].carryover == ()

    flagged = bioequivalence(
        test,
        reference,
        ["auc_inf_obs", "cmax"],
        carryover="flag",
        test_batch=test_batch,
        reference_batch=reference_batch,
    )
    assert flagged["cmax"].carryover == (CARRIER,)
    assert flagged["cmax"].n_test == 6
    assert flagged["cmax"].gmr == pytest.approx(ignored["cmax"].gmr)

    dropped = bioequivalence(
        test,
        reference,
        ["auc_inf_obs", "cmax"],
        carryover="exclude",
        test_batch=test_batch,
        reference_batch=reference_batch,
    )
    assert dropped["cmax"].carryover == (CARRIER,)
    assert dropped["cmax"].n_test == 5
    assert dropped["cmax"].n_reference == 5
    assert "carryover" in dropped.to_dataframe().columns
    # the analysis of the five remaining subjects is the one of a study
    # which never enrolled the sixth
    without = bioequivalence(
        test.exclude(individual=CARRIER),
        reference.exclude(individual=CARRIER),
        ["auc_inf_obs", "cmax"],
    )
    assert dropped["cmax"].gmr == pytest.approx(without["cmax"].gmr)
    assert dropped["cmax"].ci_low == pytest.approx(without["cmax"].ci_low)


def test_a_carryover_check_without_the_batches_raises() -> None:
    batch = period(scale=1.0)
    result = nca(batch, options=OPTIONS)
    with pytest.raises(ValueError, match="needs the timecourses"):
        bioequivalence(result, result, ["cmax"], carryover="exclude")
    with pytest.raises(ValueError, match="'carryover' must be"):
        bioequivalence(result, result, ["cmax"], carryover="drop")  # ty: ignore[invalid-argument-type]


def test_a_batch_of_two_routes_decides_the_predose_rule_per_sample() -> None:
    import numpy as np

    from pkpdutils import Dose, Route, Timecourse, Timecourses, nca

    time = [0.0, 0.5, 1, 2, 4, 8, 12, 24]
    # the oral subject has a genuine pre-dose sample at the dose time (6 % of
    # its peak), the bolus subject's sample at the dose time is the post-dose value
    oral = Timecourse(
        time=time,
        value=[0.18, 1.5, 2.6, 3.0, 2.4, 1.4, 0.8, 0.2],
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        label="s_oral",
    )
    bolus = Timecourse(
        time=time,
        value=list(10 * np.exp(-0.2 * np.asarray(time))),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
        label="s_iv",
    )
    batch = Timecourses.from_timecourses([oral, bolus])
    assert batch.routes is not None
    from pkpdutils.stats.bioequivalence import carryover_table

    table = carryover_table(batch, nca(batch)).set_index("individual")
    assert bool(table.loc["s_oral", "flagged"]) is True
    assert np.isnan(table.loc["s_iv", "predose"])
    assert bool(table.loc["s_iv", "flagged"]) is False
