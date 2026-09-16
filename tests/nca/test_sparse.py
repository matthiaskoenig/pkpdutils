"""Sparse and destructive sampling designs against the formulas by hand.

The serial case is checked against Bailer's `sum w_j^2 s_j^2 / n_j` and the
Satterthwaite degrees of freedom of Nedelman, Gibiansky and Lau written out for
three time points; the batch case against the covariance form of Nedelman and
Jia as Holder writes it, which the per-animal estimator of `bailer_variance`
has to reproduce.
"""

import warnings

import numpy as np
import pytest
import xarray as xr

from pkpdutils import AUCMethod, Dose, NCAFlag, NCAOptions, NCAResult, Route
from pkpdutils.nca.sparse import (
    area_window,
    bailer_variance,
    nca_sparse,
    point_statistics,
    sparse_mean,
    trapezoid_weights,
)
from pkpdutils.nca.uncertainty import DISCRETE_PARAMETERS

#: the nominal sampling times of the three point design, in hr
TIMES = np.array([1.0, 2.0, 4.0])

#: three animals per time point, one sample each
SERIAL = {
    0: [1.0, 1.2, 1.4],
    1: [2.0, 2.4, 2.2],
    2: [0.5, 0.4, 0.6],
}


def serial_values() -> np.ndarray:
    """The serial design as `(n_animals, n_time)` with one sample per animal."""
    values = np.full((9, 3), np.nan)
    for j, column in SERIAL.items():
        for i, value in enumerate(column):
            values[3 * j + i, j] = value
    return values


def test_trapezoid_weights() -> None:
    """The weights of the linear trapezoid rule are the half distances of the neighbours."""
    np.testing.assert_allclose(trapezoid_weights(TIMES), [0.5, 1.5, 1.0])
    np.testing.assert_allclose(trapezoid_weights(np.array([0.0, 2.0])), [1.0, 1.0])
    # the weighted sum is the area of the polygon
    values = np.array([1.0, 3.0, 2.0])
    np.testing.assert_allclose(
        float(trapezoid_weights(TIMES) @ values), np.trapezoid(values, TIMES)
    )


def test_point_statistics() -> None:
    """The count, the mean and the standard deviation of every time point."""
    n, mean, sd = point_statistics(serial_values())
    np.testing.assert_array_equal(n, [3, 3, 3])
    np.testing.assert_allclose(mean, [1.2, 2.2, 0.5])
    np.testing.assert_allclose(sd, [0.2, 0.2, 0.1], rtol=1e-12)


def test_bailer_estimator_of_a_serial_design() -> None:
    """The area is the trapezoid of the means and its variance `sum w^2 s^2 / n`."""
    result = nca_sparse(TIMES, serial_values(), time_unit="hr", unit="mg/l")
    weights = trapezoid_weights(TIMES)
    mean = np.array([np.mean(column) for column in SERIAL.values()])
    sd = np.array([np.std(column, ddof=1) for column in SERIAL.values()])
    n = np.array([len(column) for column in SERIAL.values()])

    assert float(result["auc_last"]) == pytest.approx(float(weights @ mean))
    variance = float(np.sum(weights**2 * sd**2 / n))
    assert float(result["auc_last_se"]) == pytest.approx(np.sqrt(variance))
    # Nedelman, Gibiansky and Lau: Satterthwaite over the per time point terms
    terms = weights**2 * sd**2 / n
    df = terms.sum() ** 2 / np.sum(terms**2 / (n - 1))
    assert float(result["auc_last_df"]) == pytest.approx(df)
    assert result.units("auc_last") == "hour * milligram / liter"
    assert result.units("auc_last_se") == "hour * milligram / liter"
    assert result.units("auc_last_df") == "dimensionless"


def test_the_peak_and_its_standard_error() -> None:
    """`cmax` is the largest mean, `cmax_se` the standard error of that mean."""
    result = nca_sparse(TIMES, serial_values(), time_unit="hr", unit="mg/l")
    assert float(result["cmax"]) == pytest.approx(2.2)
    assert float(result["tmax"]) == pytest.approx(2.0)
    assert float(result["cmax_se"]) == pytest.approx(
        np.std(SERIAL[1], ddof=1) / np.sqrt(3)
    )
    np.testing.assert_array_equal(result["n_animals"].to_numpy(), [3, 3, 3])
    assert result.point_variables == ["n_animals"]


def test_auc_all_covers_the_trailing_zeros() -> None:
    """`auc_last` ends at the last positive mean, `auc_all` at the last sample."""
    values = np.full((12, 4), np.nan)
    for j, column in SERIAL.items():
        for i, value in enumerate(column):
            values[3 * j + i, j] = value
    values[9:, 3] = 0.0
    times = np.array([1.0, 2.0, 4.0, 6.0])
    result = nca_sparse(times, values, time_unit="hr", unit="mg/l")
    mean = np.array([1.2, 2.2, 0.5, 0.0])
    assert float(result["auc_last"]) == pytest.approx(
        float(trapezoid_weights(TIMES) @ mean[:3])
    )
    assert float(result["auc_all"]) == pytest.approx(
        float(trapezoid_weights(times) @ mean)
    )


def test_the_batch_design_reproduces_the_covariance_form_of_holder() -> None:
    """Two batches: the per animal variance equals the covariance form by hand."""
    rng = np.random.default_rng(3)
    times = np.array([1.0, 2.0, 4.0])
    # batch A is sampled at 1 and 4 hr, batch B at 2 hr alone
    batch_a = rng.lognormal(0.0, 0.2, size=(4, 2)) * np.array([1.2, 0.5])
    batch_b = rng.lognormal(0.0, 0.2, size=(3, 1)) * 2.2
    values = np.full((7, 3), np.nan)
    values[:4, 0] = batch_a[:, 0]
    values[:4, 2] = batch_a[:, 1]
    values[4:, 1] = batch_b[:, 0]

    result = nca_sparse(times, values, design="batch", time_unit="hr", unit="mg/l")
    weights = trapezoid_weights(times)
    n = np.array([4, 3, 4])
    mean = np.array([batch_a[:, 0].mean(), batch_b[:, 0].mean(), batch_a[:, 1].mean()])
    assert float(result["auc_last"]) == pytest.approx(float(weights @ mean))

    # Holder: the independent terms plus the covariance of the two times of
    # batch A, whose 4 animals are sampled at both
    sd = np.array(
        [
            batch_a[:, 0].std(ddof=1),
            batch_b[:, 0].std(ddof=1),
            batch_a[:, 1].std(ddof=1),
        ]
    )
    independent = float(np.sum(weights**2 * sd**2 / n))
    covariance = float(np.cov(batch_a[:, 0], batch_a[:, 1], ddof=1)[0, 1])
    paired = 2.0 * weights[0] * weights[2] * (4.0 / (n[0] * n[2])) * covariance
    assert float(result["auc_last_se"]) == pytest.approx(np.sqrt(independent + paired))
    # two batches, so the Satterthwaite sum has two terms
    variance, df, n_batches = bailer_variance(weights, values)
    assert n_batches == 2
    assert df == pytest.approx(
        variance**2
        / sum(
            term**2 / (size - 1)
            for term, size in (
                (
                    4.0
                    * np.var(
                        weights[0] / 4 * batch_a[:, 0] + weights[2] / 4 * batch_a[:, 1],
                        ddof=1,
                    ),
                    4,
                ),
                (3.0 * np.var(weights[1] / 3 * batch_b[:, 0], ddof=1), 3),
            )
        )
    )


def test_a_serial_design_with_several_samples_per_animal_is_rejected() -> None:
    """`design="serial"` is the promise that every animal carries one sample."""
    values = np.full((3, 3), np.nan)
    values[0, 0] = 1.0
    values[0, 1] = 2.0
    values[1, 1] = 2.2
    values[2, 2] = 0.5
    with pytest.raises(ValueError, match="not a serial design"):
        nca_sparse(TIMES, values, time_unit="hr", unit="mg/l")
    # the same data is a batch design
    result = nca_sparse(TIMES, values, design="batch", time_unit="hr", unit="mg/l")
    assert np.isnan(float(result["auc_last_se"]))


def test_unknown_design_and_broken_shapes_are_rejected() -> None:
    """The design, the shape of the values and the order of the times are checked."""
    values = serial_values()
    with pytest.raises(ValueError, match="unknown design"):
        nca_sparse(TIMES, values, design="complete", time_unit="hr", unit="mg/l")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="time points"):
        nca_sparse(np.array([1.0, 2.0]), values, time_unit="hr", unit="mg/l")
    with pytest.raises(ValueError, match="strictly increasing"):
        nca_sparse(np.array([1.0, 4.0, 2.0]), values, time_unit="hr", unit="mg/l")
    with pytest.raises(ValueError, match="expected"):
        nca_sparse(TIMES, np.array([1.0, 2.0, 3.0]), time_unit="hr", unit="mg/l")


def test_sparse_mean_is_the_mean_curve_with_its_spread() -> None:
    """`sparse_mean` returns a batch of one sample with `sd`, `se` and `n` per time point."""
    batch = sparse_mean(
        TIMES,
        serial_values(),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=10.0, unit="mg", route=Route.ORAL),
        substance="drug",
    )
    assert batch.n_samples == 1
    np.testing.assert_allclose(batch.values.reshape(-1), [1.2, 2.2, 0.5])
    np.testing.assert_allclose(
        batch.ds["sd"].to_numpy().reshape(-1), [0.2, 0.2, 0.1], rtol=1e-12
    )
    np.testing.assert_allclose(
        batch.ds["se"].to_numpy().reshape(-1),
        np.array([0.2, 0.2, 0.1]) / np.sqrt(3.0),
        rtol=1e-12,
    )
    np.testing.assert_allclose(batch.ds["n"].to_numpy().reshape(-1), [3.0, 3.0, 3.0])
    assert batch.dose_unit == "mg"


def test_a_single_time_point_has_no_area() -> None:
    """A design with one usable time point is flagged and carries no area."""
    values = np.full((3, 3), np.nan)
    values[:, 1] = [2.0, 2.2, 2.4]
    result = nca_sparse(TIMES, values, time_unit="hr", unit="mg/l")
    assert NCAFlag.NO_DATA.name in result.flags()
    assert np.isnan(float(result["auc_last"]))


def test_the_dose_travels_into_the_result() -> None:
    """The dose amount is the coordinate a dose normalized parameter divides by."""
    result = nca_sparse(
        TIMES,
        serial_values(),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=10.0, unit="mg", route=Route.ORAL),
    )
    assert result.dose is not None
    assert float(result.dose) == pytest.approx(10.0)
    normalized = result.dose_normalized(["auc_last"])
    assert float(normalized["auc_last_dn"]) == pytest.approx(
        float(result["auc_last"]) / 10.0
    )


def test_area_window_is_the_window_the_estimator_weights() -> None:
    """The observed points up to the last positive mean, empty without one."""
    observed = np.array([True, True, True, True, False])
    mean = np.array([1.0, 2.0, 0.5, 0.0, np.nan])
    np.testing.assert_array_equal(
        area_window(observed, mean), [True, True, True, False, False]
    )
    np.testing.assert_array_equal(area_window(observed, np.zeros(5)), [False] * 5)


def test_a_non_linear_trapezoid_rule_warns_only_when_it_was_asked_for() -> None:
    """The estimator is linear in the means; an explicit `auc_method` is refused loudly."""
    values = serial_values()
    with pytest.warns(UserWarning, match="linear trapezoid rule is used"):
        explicit = nca_sparse(
            TIMES,
            values,
            time_unit="hr",
            unit="mg/l",
            options=NCAOptions(auc_method=AUCMethod.LOG),
        )
    # the default LINEAR_LOG is not an explicit request and only logs
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        default = nca_sparse(TIMES, values, time_unit="hr", unit="mg/l")
    assert float(explicit["auc_last"]) == pytest.approx(float(default["auc_last"]))


def test_the_degrees_of_freedom_are_a_discrete_parameter() -> None:
    """`summarize` gives `auc_last_df` no standard error, interval or CV."""
    assert "auc_last_df" in DISCRETE_PARAMETERS
    values = serial_values()
    first = nca_sparse(TIMES, values, time_unit="hr", unit="mg/l")
    second = nca_sparse(TIMES, 1.1 * values, time_unit="hr", unit="mg/l")
    batch = NCAResult(
        xr.concat([first.ds, second.ds], dim="dose").assign_coords(dose=[10.0, 20.0])
    )
    summary = batch.summarize("dose")
    for suffix in ("_sd", "_se", "_cv", "_ci_low", "_ci_high"):
        assert f"auc_last_df{suffix}" not in summary.ds.data_vars
    assert "auc_last_sd" in summary.ds.data_vars
    assert "auc_last_df_median" in summary.ds.data_vars
