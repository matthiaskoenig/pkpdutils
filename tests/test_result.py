from enum import IntFlag

import numpy as np
import pytest
import xarray as xr

from pkpdutils.nca import NCAResult
from pkpdutils.result import ParameterResult, decode_flags


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
