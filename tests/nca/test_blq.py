"""The rules for the values below the limit of quantification and the per-sample limit."""

import numpy as np
import pandas as pd
import pytest

from pkpdutils import (
    AUCMethod,
    BLQAction,
    BLQHandling,
    BLQRules,
    Dose,
    NCAOptions,
    Route,
    TerminalMethod,
    TerminalPhase,
    Timecourse,
    Timecourses,
    nca,
    nca_single,
)
from pkpdutils.nca.nca import apply_blq

#: a curve with one BLQ value before the first measurable one and two after the
#: last: the tail is what every rule set treats differently
TIME = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0])
VALUE = np.array([0.02, 2.0, 4.0, 3.0, 1.5, 0.75, 0.05, 0.03])
LLOQ = 0.1


def curve(lloq: float | None = None, label: str | None = None) -> Timecourse:
    return Timecourse(
        time=TIME,
        value=VALUE,
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        substance="drug",
        lloq=lloq,
        label=label,
    )


def options(blq: BLQHandling | BLQRules) -> NCAOptions:
    return NCAOptions(
        lloq=LLOQ,
        blq=blq,
        auc_method=AUCMethod.LINEAR,
        terminal=TerminalPhase(method=TerminalMethod.ALL_AFTER_TMAX),
    )


def limits_of(batch: Timecourses) -> list[float]:
    limits = batch.lloq
    assert limits is not None
    return [float(v) for v in limits]


def test_rules_are_one_axis() -> None:
    with pytest.raises(ValueError, match="two axes"):
        BLQRules(first=BLQAction.ZERO, before_tmax=BLQAction.ZERO)
    assert not BLQRules(first=BLQAction.ZERO).by_tmax
    assert BLQRules(after_tmax=BLQAction.LLOQ).by_tmax


def test_presets() -> None:
    assert BLQRules.ich_m13a() == BLQRules(
        first=BLQAction.ZERO, middle=BLQAction.DROP, last=BLQAction.ZERO
    )
    assert BLQRules.pkanalix() == BLQRules(
        before_tmax=BLQAction.ZERO, after_tmax=BLQAction.HALF_LLOQ
    )
    assert BLQRules.pumas() == BLQRules(
        first=BLQAction.KEEP, middle=BLQAction.DROP, last=BLQAction.KEEP
    )
    assert not BLQRules.ich_m13a().terminal_regression


def test_classic_handling_is_a_rule_set() -> None:
    assert NCAOptions().blq_rules == BLQRules(
        first=BLQAction.DROP, middle=BLQAction.DROP, last=BLQAction.DROP
    )
    assert NCAOptions(blq="zero_before_tmax").blq_rules == BLQRules(
        before_tmax=BLQAction.ZERO, after_tmax=BLQAction.DROP
    )


def test_apply_blq_positions() -> None:
    c = np.array([[0.02, 2.0, 0.05, 1.0, 0.03, 0.01]])
    lloq = np.array([0.1])
    values, truncated, in_curve = apply_blq(
        c,
        lloq,
        BLQRules(first=BLQAction.ZERO, middle=5.0, last=BLQAction.LLOQ),
    )
    assert values[0].tolist() == [0.0, 2.0, 5.0, 1.0, 0.1, 0.1]
    assert truncated.tolist() == [True]
    # every value below the limit is still in the curve and out of the regression
    assert in_curve[0].tolist() == [True, False, True, False, True, True]


def test_apply_blq_by_tmax() -> None:
    c = np.array([[0.02, 2.0, 4.0, 0.05, 0.03]])
    values, _, in_curve = apply_blq(c, np.array([0.1]), BLQRules.pkanalix())
    assert values[0].tolist() == [0.0, 2.0, 4.0, 0.05, 0.05]
    assert in_curve[0].tolist() == [True, False, False, True, True]


def test_apply_blq_without_measurable_value() -> None:
    c = np.array([[0.02, 0.03]])
    values, _, _ = apply_blq(c, np.array([0.1]), BLQRules.pumas())
    # no measurable value: every value counts as `first`, which `pumas` keeps
    assert values[0].tolist() == [0.02, 0.03]
    dropped, truncated, _ = apply_blq(c, np.array([0.1]), BLQRules())
    assert np.isnan(dropped).all()
    assert truncated.tolist() == [True]


def test_no_limit_leaves_the_values() -> None:
    values, truncated, in_curve = apply_blq(VALUE[None, :], None, BLQRules.ich_m13a())
    assert values.tolist() == VALUE[None, :].tolist()
    assert not truncated.any()
    assert not in_curve.any()


def test_tail_of_every_preset() -> None:
    dropped = nca_single(curve(), options=options(BLQHandling.NAN)).to_quantities()
    m13a = nca_single(curve(), options=options(BLQRules.ich_m13a())).to_quantities()
    pkanalix = nca_single(curve(), options=options(BLQRules.pkanalix())).to_quantities()
    pumas = nca_single(curve(), options=options(BLQRules.pumas())).to_quantities()

    # the last measurable point is the same for every rule set: a value below
    # the limit is no measurable value, whether it was kept or imputed
    for q in (dropped, m13a, pkanalix, pumas):
        assert q["tlast"].magnitude == 8.0
        assert q["clast"].magnitude == pytest.approx(0.75)

    # the leading value at t = 0: dropped, imputed as 0 (M13A, PKanalix) or
    # kept (Pumas), which is the only difference in the area to `tlast`
    tail = dropped["auc_last"].magnitude
    triangle = 0.5 * 2.0 * 0.5
    assert m13a["auc_last"].magnitude == pytest.approx(tail + triangle)
    assert pkanalix["auc_last"].magnitude == pytest.approx(tail + triangle)
    assert pumas["auc_last"].magnitude == pytest.approx(tail + 0.5 * (0.02 + 2.0) * 0.5)
    # `nan` drops both trailing values, so the area to the last observation is
    # the area to the last measurable one
    assert dropped["auc_all"].magnitude == pytest.approx(tail)
    # M13A writes a zero at both trailing times: the trapezoid down to 0 and
    # nothing after it
    assert m13a["auc_all"].magnitude == pytest.approx(
        m13a["auc_last"].magnitude + 0.5 * 0.75 * 4.0
    )
    # PKanalix writes LLOQ / 2 after the maximum
    half = LLOQ / 2
    assert pkanalix["auc_all"].magnitude == pytest.approx(
        pkanalix["auc_last"].magnitude
        + 0.5 * (0.75 + half) * 4.0
        + 0.5 * (half + half) * 4.0
    )
    # Pumas keeps the measured values below the limit
    assert pumas["auc_all"].magnitude == pytest.approx(
        pumas["auc_last"].magnitude
        + 0.5 * (0.75 + 0.05) * 4.0
        + 0.5 * (0.05 + 0.03) * 4.0
    )


def test_imputed_points_stay_out_of_the_regression() -> None:
    kept = nca_single(curve(), options=options(BLQRules.pkanalix())).to_quantities()
    dropped = nca_single(curve(), options=options(BLQHandling.NAN)).to_quantities()
    assert kept["lambda_z_n_points"].magnitude == dropped["lambda_z_n_points"].magnitude
    assert kept["lambda_z"].magnitude == pytest.approx(dropped["lambda_z"].magnitude)
    assert kept["lambda_z_t_last"].magnitude == 8.0

    # with `terminal_regression` the two imputed points are regressed as well
    rules = BLQRules(
        before_tmax=BLQAction.ZERO,
        after_tmax=BLQAction.HALF_LLOQ,
        terminal_regression=True,
    )
    with_imputed = nca_single(curve(), options=options(rules)).to_quantities()
    assert (
        with_imputed["lambda_z_n_points"].magnitude
        == dropped["lambda_z_n_points"].magnitude + 2
    )
    assert with_imputed["lambda_z_t_last"].magnitude == 16.0


def test_kept_values_stay_out_of_the_regression() -> None:
    # `KEEP` is a measured value below the limit: it enters the areas but is no
    # quantified value, so the regression leaves it out
    pumas = nca_single(curve(), options=options(BLQRules.pumas())).to_quantities()
    dropped = nca_single(curve(), options=options(BLQHandling.NAN)).to_quantities()
    assert (
        pumas["lambda_z_n_points"].magnitude == dropped["lambda_z_n_points"].magnitude
    )
    assert pumas["tlast"].magnitude == 8.0


def test_keep_only_sets_no_flag() -> None:
    rules = BLQRules(first=BLQAction.KEEP, middle=BLQAction.KEEP, last=BLQAction.KEEP)
    result = nca_single(curve(), options=options(rules))
    # nothing was dropped and nothing imputed, so the row is not truncated
    assert "BLQ_TRUNCATED" not in result.flags()
    assert (
        "BLQ_TRUNCATED" in nca_single(curve(), options=options(BLQHandling.NAN)).flags()
    )


def test_per_sample_lloq_travels_through_the_batch() -> None:
    low = curve(lloq=0.01, label="low")
    high = curve(lloq=1.0, label="high")
    batch = Timecourses.from_timecourses([low, high], dim="individual")
    assert limits_of(batch) == [0.01, 1.0]
    assert batch.sel(individual="high").lloq == 1.0

    result = nca(batch, options=NCAOptions(blq=BLQRules.ich_m13a()))
    flagged = result.flag_table().set_index("individual")["BLQ_TRUNCATED"]
    # the limit of the first sample is below every value, the second one is above
    assert not bool(flagged["low"])
    assert bool(flagged["high"])
    # the limit of 1.0 leaves only the three values above it measurable
    assert float(result["tlast"].sel(individual="high")) == 4.0
    assert float(result["tlast"].sel(individual="low")) == 16.0


def test_options_lloq_wins_over_the_batch() -> None:
    batch = Timecourses.from_timecourses(
        [curve(lloq=10.0, label="a")], dim="individual"
    )
    result = nca(batch, options=NCAOptions(lloq=LLOQ, blq="nan"))
    assert float(result["tlast"].sel(individual="a")) == 8.0


def test_lloq_of_a_reader_is_used() -> None:
    rows = []
    for subject, limit in (("s1", 0.1), ("s2", 2.0)):
        for time, value in zip(TIME, VALUE, strict=True):
            rows.append(
                {
                    "USUBJID": subject,
                    "PARAMCD": "DRUG",
                    "AVAL": value,
                    "AVALU": "mg/l",
                    "AFRLT": time,
                    "ARRLT": time,
                    "ARRLTU": "hr",
                    "DOSEA": 100.0,
                    "DOSEU": "mg",
                    "ALLOQ": limit,
                }
            )
    batch = Timecourses.from_adnca(pd.DataFrame(rows), route=Route.ORAL)
    assert limits_of(batch) == [0.1, 2.0]
    result = nca(batch, options=NCAOptions(blq=BLQRules.ich_m13a()))
    # the second subject only has values below its limit up to the maximum
    assert float(result["tlast"].sel(individual="s1")) == 8.0
    assert float(result["clast"].sel(individual="s2")) == pytest.approx(3.0)
