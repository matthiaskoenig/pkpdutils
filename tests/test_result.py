from enum import IntFlag

import numpy as np
import pytest
import xarray as xr

from pkpdutils.nca import NCAResult
from pkpdutils.result import ParameterResult


class MyFlag(IntFlag):
    NONE = 0
    BAD = 1
    WORSE = 2


class MyResult(ParameterResult):
    flag_type = MyFlag
    lognormal_parameters = frozenset({"a"})
    discrete_parameters = frozenset({"k"})


def make() -> MyResult:
    ds = xr.Dataset(
        {
            "a": (("s",), np.array([1.0, 2.0, 4.0]), {"units": "mg"}),
            "a_se": (("s",), np.array([0.1, 0.2, 0.4]), {"units": "mg"}),
            "k": (("s",), np.array([3.0, 3.0, 4.0]), {"units": "dimensionless"}),
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
    assert r.derived_variables == ["a_se"]
    assert r.point_variables == ["y_pred"]
    assert r.has_uncertainty
    assert r.flags(s="z") == ["BAD", "WORSE"]
    assert set(r.to_quantities(s="x")) == {"a", "a_se", "k", "n"}
    assert list(r.to_dataframe().columns) == ["s", "a", "a_se", "k", "n", "flags"]
    assert r.flag_table()["WORSE"].tolist() == [False, False, True]


def test_generic_summarize_uses_class_sets() -> None:
    s = make().summarize("s")
    q = s.to_quantities()
    assert q["a"].magnitude == pytest.approx(7.0 / 3.0)
    assert q["a_geomean"].magnitude == pytest.approx(2.0)
    assert "k_geomean" not in s
    assert "y_pred" not in s
    assert s.flags() == ["BAD", "WORSE"]


def test_nca_result_is_parameter_result() -> None:
    assert issubclass(NCAResult, ParameterResult)
    assert NCAResult.flag_type.__name__ == "NCAFlag"
