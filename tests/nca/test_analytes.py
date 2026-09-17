"""Several analytes in one batch and the metabolite to parent ratio."""

import numpy as np
import pytest

from pkpdutils import Dose, Route, Timecourse, Timecourses, nca
from pkpdutils.nca import metabolite_ratio

TIME = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
#: a parent curve with an absorption phase, scaled per subject
PARENT = 10.0 * (np.exp(-0.2 * TIME) - np.exp(-1.5 * TIME))
SUBJECTS = ["S1", "S2", "S3"]
#: the metabolite is this fraction of the parent, exactly, in every subject
FRACTION = 0.4


def two_analyte_batch(fraction: float = FRACTION) -> Timecourses:
    """A batch over `(analyte, individual)` of a parent and its metabolite."""
    subject_factor = 1.0 + 0.1 * np.arange(len(SUBJECTS))
    parent = subject_factor[:, None] * PARENT[None, :]
    values = np.stack([parent, fraction * parent])
    return Timecourses.from_arrays(
        TIME,
        values,
        time_unit="hr",
        unit="ng/ml",
        dims=("analyte", "individual"),
        coords={
            "analyte": ["parent", "metabolite"],
            "substance": ("analyte", np.array(["parent", "metabolite"], dtype=object)),
            "individual": SUBJECTS,
        },
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
    )


def test_a_batch_of_two_analytes_carries_the_substance_coordinate() -> None:
    """The substances travel as a coordinate along the analyte dimension."""
    batch = two_analyte_batch()
    assert batch.sample_dims == ("analyte", "individual")
    substances = batch.substances
    assert substances is not None
    assert substances.shape == (2, 3)
    assert list(substances[:, 0]) == ["parent", "metabolite"]
    with pytest.raises(ValueError, match="carries the substances"):
        _ = batch.substance
    assert batch.sel(analyte="metabolite", individual="S2").substance == "metabolite"


def test_one_analysis_of_both_analytes_and_the_metabolite_ratio() -> None:
    """The ratio of the exposures is the factor the metabolite was built with."""
    result = nca(two_analyte_batch())
    assert result.ds["cmax"].dims == ("analyte", "individual")
    frame = metabolite_ratio(result, parent="parent", metabolite="metabolite")
    assert list(frame.columns) == ["individual", "auc_inf_obs", "cmax"]
    assert list(frame["individual"]) == SUBJECTS
    np.testing.assert_allclose(frame["auc_inf_obs"], FRACTION)
    np.testing.assert_allclose(frame["cmax"], FRACTION)
    assert frame.attrs["parent"] == "parent"
    assert frame.attrs["metabolite"] == "metabolite"


def test_the_molar_masses_correct_the_ratio() -> None:
    """A metabolite of half the molar mass doubles the molar ratio."""
    result = nca(two_analyte_batch())
    frame = metabolite_ratio(
        result,
        parent="parent",
        metabolite="metabolite",
        parameters=("cmax",),
        molar={"parent": 300.0, "metabolite": 150.0},
    )
    np.testing.assert_allclose(frame["cmax"], FRACTION * 300.0 / 150.0)
    assert frame.attrs["molar_factor"] == pytest.approx(2.0)


def test_the_ratio_needs_the_two_analytes_and_the_molar_masses() -> None:
    """A substance which is not in the result and a missing molar mass raise."""
    result = nca(two_analyte_batch())
    with pytest.raises(ValueError, match="is 0 times among the substances"):
        metabolite_ratio(result, parent="parent", metabolite="other")
    with pytest.raises(ValueError, match="no molar mass"):
        metabolite_ratio(
            result,
            parent="parent",
            metabolite="metabolite",
            molar={"parent": 300.0},
        )
    with pytest.raises(ValueError, match="not a variable"):
        metabolite_ratio(
            result, parent="parent", metabolite="metabolite", parameters=("nope",)
        )


def test_a_result_without_analytes_raises() -> None:
    """The ratio needs a batch which carries the substances."""
    curve = Timecourse(
        time=TIME,
        value=PARENT,
        time_unit="hr",
        unit="ng/ml",
        substance="parent",
        dose=Dose(amount=100, unit="mg", route=Route.ORAL),
        label="S1",
    )
    result = nca(Timecourses.from_timecourses([curve]))
    with pytest.raises(ValueError, match="no 'substance' coordinate"):
        metabolite_ratio(result, parent="parent", metabolite="metabolite")
