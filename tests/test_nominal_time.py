import numpy as np
import pandas as pd
import pytest

from pkpdutils import Route, Timecourses


def batch(nominal: object = None) -> Timecourses:
    """Two subjects sampled beside the nominal schedule 0, 1, 2, 4 hours."""
    actual = np.array([[0.0, 1.1, 2.0, 4.2], [0.0, 0.9, 2.1, 3.8]])
    values = np.array([[0.0, 2.0, 1.5, 0.5], [0.0, 2.2, 1.6, 0.6]])
    return Timecourses.from_arrays(
        actual,
        values,
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b"]},
        nominal_time=nominal,
        dose={"amount": [10.0, 10.0], "unit": "mg"},
        route=Route.ORAL,
    )


def nominal_of(b: Timecourses) -> np.ndarray:
    """The nominal times of a batch which carries them."""
    nominal = b.nominal_times
    assert nominal is not None
    return nominal


def test_a_batch_without_nominal_times_has_none() -> None:
    assert batch().nominal_times is None
    assert "nominal_time" not in batch().ds


def test_a_shared_nominal_grid_is_broadcast_to_every_sample() -> None:
    nominal = [0.0, 1.0, 2.0, 4.0]
    b = batch(nominal)
    assert b.ds["nominal_time"].dims == ("individual", "time")
    assert b.ds["nominal_time"].attrs["units"] == "hr"
    np.testing.assert_allclose(nominal_of(b), np.broadcast_to(nominal, (2, 4)))
    # the actual times stay what the analyses read
    assert b.times[0][1] == 1.1


def test_a_nominal_time_per_sample_and_point_is_kept() -> None:
    nominal = np.array([[0.0, 1.0, 2.0, 4.0], [0.0, 1.0, 2.0, 4.0]])
    np.testing.assert_allclose(nominal_of(batch(nominal)), nominal)


@pytest.mark.parametrize("bad", [[0.0, 1.0], np.zeros((3, 4))])
def test_a_nominal_time_of_the_wrong_shape_raises(bad: object) -> None:
    with pytest.raises(ValueError, match="nominal_time"):
        batch(bad)


def test_the_nominal_times_shift_with_the_dose() -> None:
    nominal = np.array([0.0, 1.0, 2.0, 4.0])
    b = Timecourses.from_arrays(
        np.array([[0.5, 1.5, 2.5, 4.5], [0.5, 1.5, 2.5, 4.5]]),
        np.array([[0.0, 2.0, 1.5, 0.5], [0.0, 2.2, 1.6, 0.6]]),
        time_unit="hr",
        unit="mg/l",
        dims=("individual",),
        coords={"individual": ["a", "b"]},
        nominal_time=nominal + 0.5,
        dose={"amount": [10.0, 10.0], "unit": "mg", "time": [0.5, 0.5]},
        route=Route.ORAL,
    )
    shifted = b.relative_to_dose()
    np.testing.assert_allclose(shifted.times[0], [0.0, 1.0, 2.0, 4.0])
    np.testing.assert_allclose(nominal_of(shifted)[0], nominal)


def test_from_dataframe_reads_a_nominal_time_column() -> None:
    df = pd.DataFrame(
        {
            "individual": ["a"] * 4 + ["b"] * 4,
            "time": [0.0, 1.1, 2.0, 4.2, 0.0, 0.9, 2.1, 3.8],
            "nominal": [0.0, 1.0, 2.0, 4.0] * 2,
            "value": [0.0, 2.0, 1.5, 0.5, 0.0, 2.2, 1.6, 0.6],
        }
    )
    b = Timecourses.from_dataframe(
        df,
        sample=["individual"],
        time_unit="hr",
        unit="mg/l",
        nominal_time="nominal",
    )
    np.testing.assert_allclose(
        nominal_of(b), np.broadcast_to([0.0, 1.0, 2.0, 4.0], (2, 4))
    )
    np.testing.assert_allclose(b.times[0], [0.0, 1.1, 2.0, 4.2])
