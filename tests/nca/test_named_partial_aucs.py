"""Named partial areas of `NCAOptions.partial_aucs`."""

import numpy as np
import pytest

from pkpdutils import (
    AUCMethod,
    Dose,
    Dosing,
    NCAOptions,
    Route,
    Timecourse,
    Timecourses,
    nca,
    nca_single,
    partial_auc,
)

TIME = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0])
DOSE = Dose(amount=100.0, unit="mg", route=Route.IV_BOLUS)


def curve(k: float = 0.15) -> Timecourse:
    return Timecourse(
        time=TIME,
        value=10.0 * np.exp(-k * TIME),
        time_unit="hr",
        unit="mg/l",
        dose=DOSE,
        substance="drug",
        label="s1",
    )


def test_a_named_area_equals_partial_auc_on_the_same_interval() -> None:
    batch = curve().to_batch(dim="individual")
    options = NCAOptions(
        auc_method=AUCMethod.LOG, partial_aucs={"auc_0_12": (0.0, 12.0)}
    )
    result = nca(batch, options=options)
    reference = partial_auc(batch, 0.0, 12.0, options=options)
    assert float(result["auc_0_12"].isel(individual=0)) == pytest.approx(
        float(reference.isel(individual=0)), rel=1e-12
    )
    assert result.units("auc_0_12") == result.units("auc_last")
    assert "auc_0_12" in result.parameters
    assert "PARTIAL_EXTRAPOLATED" not in result.flags(individual="s1")


def test_a_named_area_beyond_the_last_sample_is_completed_by_the_regression() -> None:
    options = NCAOptions(
        auc_method=AUCMethod.LOG, partial_aucs={"auc_0_48": (0.0, 48.0)}
    )
    result = nca_single(curve(), options=options)
    q = result.to_quantities()
    lambda_z = q["lambda_z"].magnitude
    clast_pred = q["clast_pred"].magnitude
    tlast = q["tlast"].magnitude
    tail = clast_pred / lambda_z * (1.0 - np.exp(-lambda_z * (48.0 - tlast)))
    assert float(result["auc_0_48"]) == pytest.approx(
        q["auc_last"].magnitude + tail, rel=1e-10
    )
    assert "PARTIAL_EXTRAPOLATED" in result.flags()


def test_several_areas_of_a_batch_carry_the_unit_and_are_summarized() -> None:
    batch = Timecourses.from_timecourses(
        [curve(0.15), curve(0.25)], dim="individual", labels=["s1", "s2"]
    )
    options = NCAOptions(
        auc_method=AUCMethod.LOG,
        partial_aucs={"auc_0_2": (0.0, 2.0), "auc_2_8": (2.0, 8.0)},
    )
    result = nca(batch, options=options)
    assert result["auc_0_2"].attrs["units"] == "hour * milligram / liter"
    # the two windows add up to the area over the whole interval
    total = partial_auc(batch, 0.0, 8.0, options=options).to_numpy()
    assert result["auc_0_2"].to_numpy() + result["auc_2_8"].to_numpy() == pytest.approx(
        total
    )
    summary = result.summarize("individual")
    assert float(summary["auc_0_2_n"]) == 2.0


def test_the_area_of_a_multiple_dose_row_is_relative_to_the_first_dose() -> None:
    time = np.arange(0.0, 36.1, 1.0)
    protocol = Dosing.regimen(DOSE, interval=12.0, n_doses=3)
    value = np.zeros_like(time)
    for dose_time in protocol.times:
        after = time >= dose_time
        value[after] += 10.0 * np.exp(-0.15 * (time[after] - dose_time))
    multiple = Timecourse(
        time=time,
        value=value,
        time_unit="hr",
        unit="mg/l",
        dosing=protocol,
        substance="drug",
        label="s1",
    )
    options = NCAOptions(
        auc_method=AUCMethod.LOG,
        partial_aucs={"auc_0_12": (0.0, 12.0), "auc_24_36": (24.0, 36.0)},
    )
    batch = multiple.to_batch(dim="individual")
    result = nca(batch, options=options)
    # the intervals are counted from the first dose, while the point
    # parameters of the row are computed from the last dose at 24 hours
    assert float(result["tlast"].isel(individual=0)) == pytest.approx(12.0)
    for name, (t_start, t_end) in options.partial_aucs.items():
        reference = float(
            partial_auc(batch, t_start, t_end, options=options).isel(individual=0)
        )
        assert float(result[name].isel(individual=0)) == pytest.approx(
            reference, rel=1e-12
        )
    assert float(result["auc_0_12"].isel(individual=0)) < float(
        result["auc_24_36"].isel(individual=0)
    )


def test_a_name_which_collides_with_a_result_variable_raises() -> None:
    batch = curve().to_batch(dim="individual")
    options = NCAOptions(partial_aucs={"auc_last": (0.0, 12.0)})
    with pytest.raises(ValueError, match="carry the name of a variable"):
        nca(batch, options=options)


def test_a_reversed_interval_is_rejected_by_the_options() -> None:
    with pytest.raises(ValueError, match="t_end > t_start"):
        NCAOptions(partial_aucs={"auc_0_0": (12.0, 12.0)})
