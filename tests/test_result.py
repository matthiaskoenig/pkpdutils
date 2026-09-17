import warnings
from enum import IntFlag

import numpy as np
import pytest
import xarray as xr

from pkpdutils.nca import NCAResult
from pkpdutils.result import ParameterResult, decode_flags, nan_percentile


class MyFlag(IntFlag):
    NONE = 0
    BAD = 1
    WORSE = 2


class MyResult(ParameterResult):
    flag_type = MyFlag
    lognormal_parameters = frozenset({"a"})
    discrete_parameters = frozenset({"k"})
    statistic_variables = frozenset({"cost"})


def make() -> MyResult:
    ds = xr.Dataset(
        {
            "a": (("s",), np.array([1.0, 2.0, 4.0]), {"units": "mg"}),
            "a_se": (("s",), np.array([0.1, 0.2, 0.4]), {"units": "mg"}),
            "k": (("s",), np.array([3.0, 3.0, 4.0]), {"units": "dimensionless"}),
            "cost": (("s",), np.array([1.0, 2.0, 3.0]), {"units": "dimensionless"}),
            "y_pred": (("s", "point"), np.ones((3, 2)), {"units": "mg"}),
            "n": (("s",), np.array([5.0, 5.0, 5.0]), {"units": "dimensionless"}),
            "flags": (("s",), np.array([0, 1, 3]), {"units": "dimensionless"}),
        },
        coords={"s": ["x", "y", "z"]},
    )
    return MyResult(ds)


def test_generic_result_classification() -> None:
    r = make()
    assert r.sample_dims == ("s",)
    assert r.parameters == ["a", "k"]
    assert r.statistics == ["cost"]
    assert r.derived_variables == ["a_se"]
    assert r.point_variables == ["y_pred"]
    assert r.has_uncertainty
    assert r.flags(s="z") == ["BAD", "WORSE"]
    assert set(r.to_quantities(s="x")) == {"a", "a_se", "k", "cost", "n"}
    assert list(r.to_dataframe().columns) == [
        "s",
        "a",
        "a_se",
        "k",
        "cost",
        "n",
        "flags",
    ]
    assert r.flag_table()["WORSE"].tolist() == [False, False, True]


def test_generic_summarize_uses_class_sets() -> None:
    s = make().summarize("s")
    q = s.to_quantities()
    assert q["a"].magnitude == pytest.approx(7.0 / 3.0)
    assert q["a_geomean"].magnitude == pytest.approx(2.0)
    assert "k_geomean" not in s
    assert "y_pred" not in s
    assert "cost" not in s and "cost_sd" not in s  # a statistic is not summarized
    assert s.flags() == ["BAD", "WORSE"]


def test_nca_result_is_parameter_result() -> None:
    assert issubclass(NCAResult, ParameterResult)
    assert NCAResult.flag_type.__name__ == "NCAFlag"


def test_sample_coordinates_keeps_non_dimension_coords() -> None:
    from pkpdutils.result import sample_coordinates

    ds = xr.Dataset(
        {"value": (("s", "time"), np.ones((3, 2)))},
        coords={
            "s": ["x", "y", "z"],
            "period": ("s", [1, 2, 1]),
            "time": [0.0, 1.0],
            "grid": ("time", [0, 1]),
        },
    )
    coords = sample_coordinates(ds, ("s",))
    assert set(coords) == {"s", "period"}
    assert coords["period"].to_numpy().tolist() == [1, 2, 1]


def test_sample_individual_values() -> None:
    r = make()
    ds = r.ds.assign_coords(period=("s", [1, 2, 1]))
    r = MyResult(ds)
    sample = r.sample("a", dim="s")
    assert sample.is_individual and sample.name == "a" and sample.unit == "mg"
    assert sample.values is not None and sample.values.tolist() == [1.0, 2.0, 4.0]
    assert sample.labels is not None and sample.labels.tolist() == ["x", "y", "z"]
    assert sample.coords["period"].tolist() == [1, 2, 1]
    with pytest.raises(ValueError, match="sample dimension"):
        r.sample("a", dim="point")
    with pytest.raises(ValueError, match="not a variable"):
        r.sample("b", dim="s")


def test_sample_two_dims_needs_indexers() -> None:
    ds = xr.Dataset(
        {
            "a": (("g", "s"), np.array([[1.0, 2.0], [3.0, 4.0]]), {"units": "mg"}),
            "flags": (
                ("g", "s"),
                np.zeros((2, 2), dtype=int),
                {"units": "dimensionless"},
            ),
        },
        coords={"g": ["c", "t"], "s": [1, 2]},
    )
    r = MyResult(ds)
    sample = r.sample("a", dim="s", g="t")
    assert sample.values is not None and sample.values.tolist() == [3.0, 4.0]
    with pytest.raises(ValueError, match="remaining"):
        r.sample("a", dim="s")


def two_dims() -> MyResult:
    """A result over a group and a subject dimension, the period along the subjects."""
    ds = xr.Dataset(
        {
            "a": (
                ("g", "s"),
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                {"units": "mg"},
            ),
            "flags": (
                ("g", "s"),
                np.array([[0, 0, 0], [0, 1, 0]]),
                {"units": "dimensionless"},
            ),
        },
        coords={"g": ["c", "t"], "s": ["x", "y", "z"], "period": ("s", [1, 2, 1])},
    )
    return MyResult(ds)


def test_sample_rejects_an_indexer_which_is_not_a_sample_dimension() -> None:
    """An unknown name, a coordinate and `dim` itself are no indexers (#70)."""
    r = two_dims()
    with pytest.raises(
        ValueError, match=r"'foo' is not a sample dimension \('g', 's'\)"
    ):
        r.sample("a", "s", g="t", foo=1)
    with pytest.raises(
        ValueError,
        match="'period' is a coordinate of the result, not a sample dimension",
    ):
        r.sample("a", "s", g="t", period=1)
    with pytest.raises(
        ValueError, match="'s' is the dimension of the values and takes no indexer"
    ):
        r.sample("a", "s", g="t", s="x")
    with pytest.raises(ValueError, match="takes no indexer"):
        r.sample("a", "s", g="t", s=["x", "y"])
    # the dimension of the values is checked before the indexers
    with pytest.raises(ValueError, match="'q' is not a sample dimension"):
        r.sample("a", "q", foo=1)


def test_sample_rejects_a_label_which_is_not_on_its_dimension() -> None:
    """An indexer is one label of its dimension, the error names the labels (#70)."""
    r = two_dims()
    sample = r.sample("a", "s", g="t")
    assert sample.values is not None and sample.values.tolist() == [4.0, 5.0, 6.0]
    with pytest.raises(
        ValueError, match=r"'q' is not a label of the sample dimension 'g' \['c', 't'\]"
    ):
        r.sample("a", "s", g="q")
    with pytest.raises(ValueError, match=r"'g' is one label, got \['t'\]"):
        r.sample("a", "s", g=["t"])
    with pytest.raises(ValueError, match="'g' is one label, got slice"):
        r.sample("a", "s", g=slice("c", "t"))
    with pytest.raises(
        ValueError, match="1 is not a label of the sample dimension 'g'"
    ):
        r.sample("a", "s", g=1)
    # a dimension without labels is indexed by the position
    unlabelled = MyResult(r.ds.drop_vars("g"))
    sample = unlabelled.sample("a", "s", g=1)
    assert sample.values is not None and sample.values.tolist() == [4.0, 5.0, 6.0]
    with pytest.raises(
        ValueError, match=r"2 is not a label of the sample dimension 'g' \[0, 1\]"
    ):
        unlabelled.sample("a", "s", g=2)
    with pytest.raises(ValueError, match="'t' is not a label"):
        unlabelled.sample("a", "s", g="t")
    # a long dimension names its first labels and how many there are
    many = xr.Dataset(
        {
            "a": (("g", "s"), np.ones((30, 2)), {"units": "mg"}),
            "flags": (("g", "s"), np.zeros((30, 2), dtype=int), {"units": "-"}),
        },
        coords={"g": [f"g{i}" for i in range(30)]},
    )
    with pytest.raises(
        ValueError, match=r"\['g0', 'g1', .*'g9', \.\.\.\] \(30 labels\)"
    ):
        MyResult(many).sample("a", "s", g="q")
    # a label which several samples carry does not pick one of them
    duplicated = MyResult(r.ds.assign_coords(g=["c", "c"]))
    with pytest.raises(
        ValueError, match="'c' labels 2 samples of the sample dimension 'g'"
    ):
        duplicated.sample("a", "s", g="c")


def test_one_sample_rejects_an_unknown_indexer_or_label() -> None:
    """`to_quantities` and `flags` select one sample the same way (#70)."""
    r = two_dims()
    assert r.flags(g="t", s="y") == ["BAD"]
    assert r.to_quantities(s="y", g="c")["a"].magnitude == 2.0
    with pytest.raises(ValueError, match="'foo' is not a sample dimension"):
        r.to_quantities(g="t", s="y", foo=1)
    with pytest.raises(
        ValueError, match="'q' is not a label of the sample dimension 's'"
    ):
        r.flags(g="t", s="q")
    with pytest.raises(ValueError, match=r"missing \['s'\]"):
        r.to_quantities(g="t")


def test_sample_takes_the_labels_as_iterating_a_coordinate_yields_them() -> None:
    """A 0-d array of a coordinate and a numpy scalar are labels as well."""
    r = two_dims()
    values = {
        str(group.item()): r.sample("a", "s", g=group).values for group in r.ds["g"]
    }
    assert {key: v.tolist() for key, v in values.items() if v is not None} == {
        "c": [1.0, 2.0, 3.0],
        "t": [4.0, 5.0, 6.0],
    }
    assert r.flags(g=np.str_("t"), s=np.array("y")) == ["BAD"]
    days = MyResult(
        r.ds.assign_coords(g=np.array(["2026-01-01", "2026-01-02"], "M8[ns]"))
    )
    sample = days.sample("a", "s", g=days.ds["g"][1])
    assert sample.values is not None and sample.values.tolist() == [4.0, 5.0, 6.0]


def test_sample_summary_data() -> None:
    ds = xr.Dataset(
        {
            "a": ((), 10.0, {"units": "mg"}),
            "a_sd": ((), 2.0, {"units": "mg"}),
            "a_geomean": ((), 9.8, {"units": "mg"}),
            "a_geocv": ((), 0.2, {"units": "dimensionless"}),
            "n": ((), 12.0, {"units": "dimensionless"}),
            "flags": ((), 0, {"units": "dimensionless"}),
        }
    )
    sample = MyResult(ds).sample("a")
    assert not sample.is_individual
    assert (sample.mean, sample.sd, sample.n) == (10.0, 2.0, 12)
    assert (sample.geomean, sample.geocv) == (9.8, 0.2)
    ds_se = ds.drop_vars(["a_sd", "a_geomean", "a_geocv"]).assign(
        a_se=((), 0.5, {"units": "mg"}), a_n=((), 10.0, {"units": "dimensionless"})
    )
    sample_se = MyResult(ds_se).sample("a")
    assert sample_se.n == 10 and sample_se.sd == pytest.approx(0.5 * np.sqrt(10))
    with pytest.raises(ValueError, match="group data"):
        MyResult(ds.drop_vars(["a_sd", "a_geomean", "a_geocv"])).sample("a")
    with pytest.raises(ValueError, match="remaining"):
        make().sample("a")


def test_check_coordinate_collision() -> None:
    from pkpdutils.result import check_coordinate_collision

    check_coordinate_collision({"period": [1, 2]}, {"a", "flags"})
    with pytest.raises(ValueError, match="collides"):
        check_coordinate_collision({"n": [1, 2]}, {"a", "n", "flags"})


def test_decode_flags_of_a_flag_type() -> None:
    assert decode_flags(MyFlag, 3) == ["BAD", "WORSE"]
    assert decode_flags(MyFlag, 1) == ["BAD"]
    assert decode_flags(MyFlag, 0) == []


def test_summarize_skips_the_uncertainty_of_discrete_parameters() -> None:
    s = make().summarize("s")
    names = set(s.ds.data_vars)
    assert "k" in names and "k_median" in names and "k_n" in names
    for suffix in ("_sd", "_se", "_ci_low", "_ci_high"):
        assert f"k{suffix}" not in names
        assert f"a{suffix}" in names


def _reference(values: np.ndarray, q: float | list[float], axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nanpercentile(values, q, axis=axis))


@pytest.mark.parametrize("q", [[2.5, 97.5], [25.0, 75.0], [0.0, 50.0, 100.0], 10.0])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_nan_percentile_equals_numpy(q: float | list[float], axis: int) -> None:
    # B1 (F7): `np.nanpercentile` falls back to a python loop over the slices as
    # soon as the array holds a NaN; the sorted version must agree bit by bit
    rng = np.random.default_rng(20260915)
    for _ in range(50):
        values = rng.normal(size=(rng.integers(1, 8), rng.integers(1, 11)))
        values[rng.random(values.shape) < 0.35] = np.nan
        np.testing.assert_array_equal(
            nan_percentile(values, q, axis=axis), _reference(values, q, axis)
        )


def test_nan_percentile_all_nan_and_single_value_slices() -> None:
    values = np.array(
        [
            [np.nan, np.nan, np.nan],  # no value at all
            [1.0, np.nan, np.nan],  # a single value
            [1.0, 2.0, np.nan],
            [-np.inf, 2.0, np.inf],  # infinities are ordinary values
        ]
    )
    result = nan_percentile(values, [25.0, 75.0])
    np.testing.assert_array_equal(result, _reference(values, [25.0, 75.0], -1))
    assert np.isnan(result[:, 0]).all()
    assert (result[:, 1] == 1.0).all()


def test_nan_percentile_shapes() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=(2, 3, 4))
    values[0, 1, :] = np.nan
    assert nan_percentile(values, 50.0).shape == (2, 3)
    assert nan_percentile(values, [10.0, 90.0], axis=1).shape == (2, 2, 4)
    np.testing.assert_array_equal(
        nan_percentile(values, [10.0, 90.0], axis=1),
        _reference(values, [10.0, 90.0], 1),
    )
    single = nan_percentile(np.array([3.0, 1.0, np.nan, 2.0]), 50.0)
    assert single.shape == ()
    assert float(single) == 2.0


def test_nan_percentile_of_an_empty_axis_is_nan() -> None:
    assert np.isnan(nan_percentile(np.empty((3, 0)), [25.0, 75.0])).all()
    assert np.isnan(nan_percentile(np.empty((3, 0)), 50.0)).all()


def test_summarize_adds_cv_min_and_max() -> None:
    # a = [1, 2, 4] mg: mean 7/3, sd = sqrt(7/3), cv = sd / mean
    s = make().summarize("s")
    q = s.to_quantities()
    mean, sd = 7.0 / 3.0, float(np.std([1.0, 2.0, 4.0], ddof=1))
    assert q["a_cv"].magnitude == pytest.approx(sd / mean)
    assert str(q["a_cv"].units) == "dimensionless"
    assert q["a_min"].magnitude == pytest.approx(1.0)
    assert q["a_max"].magnitude == pytest.approx(4.0)
    assert str(q["a_min"].units) == "milligram"
    # a discrete parameter has the order statistics but no coefficient of variation
    assert "k_min" in s and "k_max" in s and "k_cv" not in s
    assert s.parameters == ["a", "k"]


def test_summarize_reduces_the_declared_point_variables() -> None:
    class WithPoints(MyResult):
        summarized_point_variables = frozenset({"per_interval"})

    ds = (
        make()
        .ds.assign(
            per_interval=(
                ("s", "interval"),
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                {"units": "mg"},
            )
        )
        .assign_coords(interval=[0, 1])
    )
    s = WithPoints(ds).summarize("s")
    assert s.ds["per_interval"].dims == ("interval",)
    np.testing.assert_allclose(s.ds["per_interval"].to_numpy(), [3.0, 4.0])
    np.testing.assert_allclose(
        s.ds["per_interval_sd"].to_numpy(), np.std([1.0, 3.0, 5.0], ddof=1)
    )
    np.testing.assert_allclose(s.ds["per_interval_n"].to_numpy(), [3.0, 3.0])
    assert s.ds["interval"].to_numpy().tolist() == [0, 1]
    # a point variable which is not declared is still dropped
    assert "y_pred" not in s


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [
        (12.8432, 3, "12.8"),
        (1.71234, 3, "1.71"),
        (123456.0, 3, "123000"),
        (0.000123456, 3, "0.000123"),
        (1.2345e-7, 3, "1.23e-07"),
        (9.87654321e9, 4, "9.877e+09"),
        (0.0, 3, "0"),
        (-2.5, 2, "-2.5"),
        (float("nan"), 3, ""),
        (float("inf"), 3, "inf"),
    ],
)
def test_format_number(value: float, digits: int, expected: str) -> None:
    from pkpdutils.result import format_number

    assert format_number(value, digits) == expected


def test_format_number_rejects_zero_digits() -> None:
    from pkpdutils.result import format_number

    with pytest.raises(ValueError, match="digits"):
        format_number(1.0, 0)


def test_summary_table_cells() -> None:
    from pkpdutils.result import summary_table

    # a = [1, 2, 4] mg: mean 2.333, sd 1.528, cv 65.5 %, geomean 2, geoCV 78.5 %
    df = summary_table(
        make(),
        "s",
        parameters=["a", "k"],
        stats=("n", "mean", "sd", "cv", "geomean", "geocv", "median", "range"),
    )
    assert list(df.columns) == [
        "parameter",
        "unit",
        "n",
        "mean",
        "sd",
        "cv",
        "geomean",
        "geocv",
        "median",
        "range",
    ]
    row = df.iloc[0]
    assert row["parameter"] == "a" and row["unit"] == "mg"
    assert row["n"] == "3"
    assert row["mean"] == "2.33"
    assert row["sd"] == "1.53"
    assert row["cv"] == "65.5 %"
    assert row["geomean"] == "2.00"
    assert row["geocv"] == "78.5 %"
    assert row["range"] == "1.00 - 4.00"
    # a discrete parameter carries no sd, cv or geometric statistics
    discrete = df.iloc[1]
    assert discrete["parameter"] == "k"
    assert discrete["sd"] == "" and discrete["cv"] == "" and discrete["geocv"] == ""
    assert discrete["median"] == "3.00" and discrete["range"] == "3.00 - 4.00"


def test_summary_table_groups_layouts_and_units() -> None:
    from pkpdutils.result import summary_table

    r = MyResult(make().ds.assign_coords(arm=("s", ["a", "b", "a"])))
    grouped = summary_table(r, "s", by="arm", parameters=["a"], stats=("n", "mean"))
    assert grouped["arm"].tolist() == ["a", "b"]
    assert grouped["n"].tolist() == ["2", "1"]
    # the groups keep the order of their first appearance, "a" before "b"
    assert grouped.iloc[0]["mean"] == "2.50"

    header = summary_table(r, "s", parameters=["a"], stats=("mean",), units="header")
    assert header.iloc[0]["parameter"] == "a [mg]"
    assert "unit" not in header.columns

    columns = summary_table(
        r, "s", parameters=["a", "k"], stats=("n", "mean"), layout="parameters_columns"
    )
    assert list(columns.columns) == ["statistic", "a", "k"]
    assert columns["statistic"].tolist() == ["unit", "n", "mean"]
    assert columns.iloc[0]["a"] == "mg"

    long = summary_table(
        r, "s", by="arm", parameters=["a"], stats=("n", "mean"), layout="long"
    )
    assert list(long.columns) == ["parameter", "unit", "arm", "statistic", "value"]
    assert long["statistic"].tolist() == ["n", "mean", "n", "mean"]
    assert long["arm"].tolist() == ["a", "a", "b", "b"]


def test_summary_table_method_and_errors() -> None:
    r = make()
    assert (
        r.summary_table("s", parameters=["a"], stats=("mean",)).iloc[0]["mean"]
        == "2.33"
    )
    with pytest.raises(ValueError, match="not a sample dimension"):
        r.summary_table("nope")
    with pytest.raises(ValueError, match="unknown statistics"):
        r.summary_table("s", stats=("mean", "bogus"))
    with pytest.raises(ValueError, match="no variables of the result"):
        r.summary_table("s", parameters=["nope"])
    with pytest.raises(ValueError, match="'units'"):
        r.summary_table("s", units="both")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="'layout'"):
        r.summary_table("s", layout="wide")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="coordinate along"):
        r.summary_table("s", by="missing")


def test_summary_table_by_another_sample_dimension_explains_itself() -> None:
    # the `(dose, individual)` shape of a dose escalation: `dose` is a column
    # of the table already, `by` groups the samples of `dim` by a coordinate
    ds = xr.Dataset(
        {
            "a": (
                ("dose", "individual"),
                np.arange(6.0).reshape(2, 3),
                {"units": "mg"},
            ),
            "flags": (
                ("dose", "individual"),
                np.zeros((2, 3), dtype=np.int64),
                {"units": "dimensionless"},
            ),
        },
        coords={
            "dose": [50, 100],
            "individual": ["s1", "s2", "s3"],
            "sex": ("individual", ["F", "M", "F"]),
        },
    )
    r = MyResult(ds)
    with pytest.raises(ValueError) as error:
        r.summary_table("individual", by="dose")
    message = str(error.value)
    assert "'dose' is a sample dimension of the result" in message
    assert "one row per 'dose'" in message and "column of its own" in message
    assert "coordinates ['sex']" in message
    # an unknown name names the coordinates which do group
    with pytest.raises(ValueError, match=r"coordinate along 'individual': \['sex'\]"):
        r.summary_table("individual", by="nope")
    # and the coordinate itself groups the table
    table = r.summary_table("individual", by="sex", parameters=["a"], stats=("mean",))
    assert list(table["sex"]) == ["F", "F", "M", "M"]
    assert list(table["dose"]) == [50, 100, 50, 100]


def test_summary_table_rejects_a_point_variable() -> None:
    with pytest.raises(ValueError, match="beyond the sample dimensions"):
        make().summary_table("s", parameters=["y_pred"])


def test_summary_table_digits_per_parameter() -> None:
    from pkpdutils.result import summary_table

    df = summary_table(
        make(),
        "s",
        parameters=["a", "k"],
        stats=("mean",),
        digits={"k": 1},
    )
    # the mapping names the parameters which differ, the rest keeps three digits
    assert df.set_index("parameter")["mean"].to_dict() == {"a": "2.33", "k": "3"}


def test_summary_table_short_units() -> None:
    from pkpdutils.result import summary_table

    result = make()
    result.ds["a"].attrs["units"] = "milligram / liter"
    long = summary_table(result, "s", parameters=["a"])
    short = summary_table(result, "s", parameters=["a"], unit_style="short")
    assert long["unit"].iloc[0] == "milligram / liter"
    assert short["unit"].iloc[0] == "mg/l"
    header = summary_table(
        result, "s", parameters=["a"], units="header", unit_style="short"
    )
    assert header["parameter"].iloc[0] == "a [mg/l]"
    with pytest.raises(ValueError, match="unit_style"):
        summary_table(result, "s", unit_style="tiny")  # ty: ignore[invalid-argument-type]


def test_to_units_converts_a_parameter_and_its_companions() -> None:
    """The named variables and their `_se`/`_ci_*` companions follow the unit."""
    from pkpdutils import Dose, Route, Timecourse, Timecourses, nca

    time = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
    value = 10.0 * (np.exp(-0.2 * time) - np.exp(-1.5 * time))
    curves = [
        Timecourse(
            time=time,
            value=(1.0 + 0.1 * i) * value,
            sd=0.1 * value,
            n=6,
            time_unit="hr",
            unit="ng/ml",
            dose=Dose(amount=100, unit="mg", route=Route.ORAL),
            label=f"s{i}",
        )
        for i in range(3)
    ]
    result = nca(Timecourses.from_timecourses(curves)).dose_normalized()
    converted = result.to_units({"auc_inf_obs": "min*ug/L", "cl_f": "mL/min"})

    # 1 h * ng/ml is 60 min * ug/l, 1 l/h is 1000/60 ml/min
    assert converted.units("auc_inf_obs") == "microgram * minute / liter"
    assert converted.units("cl_f") == "milliliter / minute"
    np.testing.assert_allclose(
        converted.ds["auc_inf_obs"], result.ds["auc_inf_obs"] * 60.0
    )
    np.testing.assert_allclose(converted.ds["cl_f"], result.ds["cl_f"] * 1000.0 / 60.0)
    for suffix in ("_se", "_sd", "_ci_low", "_ci_high"):
        name = f"auc_inf_obs{suffix}"
        assert converted.units(name) == "microgram * minute / liter"
        np.testing.assert_allclose(converted.ds[name], result.ds[name] * 60.0)
    # the dose normalized companion keeps its dose and follows the numerator
    assert converted.units("auc_inf_dn") == "microgram * minute / liter / milligram"
    np.testing.assert_allclose(
        converted.ds["auc_inf_dn"], result.ds["auc_inf_dn"] * 60.0
    )
    # the dimensionless companions and the other parameters are untouched
    assert converted.units("auc_inf_obs_geocv") == "dimensionless"
    np.testing.assert_allclose(
        converted.ds["auc_inf_obs_geocv"], result.ds["auc_inf_obs_geocv"]
    )
    assert converted.units("cmax") == result.units("cmax")
    np.testing.assert_allclose(converted.ds["cmax"], result.ds["cmax"])
    # the result it was called on is unchanged
    assert result.units("auc_inf_obs") == "hour * nanogram / milliliter"


def test_to_units_rejects_an_unknown_variable_and_an_incompatible_unit() -> None:
    """A name which is not a variable and a unit of another dimensionality raise."""
    result = make()
    result.ds["a"].attrs["units"] = "milligram / liter"
    with pytest.raises(KeyError, match="not a variable"):
        result.to_units({"nope": "mg/l"})
    with pytest.raises(ValueError, match="cannot be converted"):
        result.to_units({"a": "hour"})
    with pytest.raises(ValueError, match="not a unit"):
        result.to_units({"a": "bananas"})


def test_to_units_converts_the_companions_of_the_dose_normalized_variable() -> None:
    """`auc_inf_dn_se` follows `auc_inf_dn`, the dimensionless ones do not."""
    from pkpdutils import Dose, Route, Timecourse, Timecourses, nca

    time = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
    value = 10.0 * (np.exp(-0.2 * time) - np.exp(-1.5 * time))
    curves = [
        Timecourse(
            time=time,
            value=(1.0 + 0.1 * i) * value,
            sd=0.1 * value,
            n=6,
            time_unit="hr",
            unit="ng/ml",
            dose=Dose(amount=100, unit="mg", route=Route.ORAL),
            label=f"s{i}",
        )
        for i in range(3)
    ]
    result = nca(Timecourses.from_timecourses(curves)).dose_normalized()
    converted = result.to_units({"auc_inf_obs": "min*ug/L"})
    per_dose = "microgram * minute / liter / milligram"
    for name in ("auc_inf_dn", "auc_inf_dn_se", "auc_inf_dn_ci_low"):
        assert converted.units(name) == per_dose
        np.testing.assert_allclose(converted.ds[name], result.ds[name] * 60.0)
    assert converted.units("auc_inf_dn_geocv") == "dimensionless"
    np.testing.assert_allclose(
        converted.ds["auc_inf_dn_geocv"], result.ds["auc_inf_dn_geocv"]
    )
