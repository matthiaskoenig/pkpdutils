import numpy as np
import pytest
import xarray as xr

from pkpdutils.nca.options import NCAFlag
from pkpdutils.nca.result import NCAResult, parameter_unit


def test_parameter_unit_auc() -> None:
    unit, factor = parameter_unit(
        "({unit}) * ({time})", unit="ng/ml", time_unit="hr", dose_unit=None
    )
    assert unit == "hour * nanogram / milliliter"
    assert factor == pytest.approx(1.0)


def test_parameter_unit_clearance_normalized() -> None:
    unit, factor = parameter_unit(
        "({dose}) / (({unit}) * ({time}))", unit="mg/l", time_unit="min", dose_unit="mg"
    )
    assert unit == "liter / hour"
    assert factor == pytest.approx(60.0)


def test_parameter_unit_volume_per_bodyweight() -> None:
    unit, factor = parameter_unit(
        "({dose}) / ({unit})", unit="ng/ml", time_unit="hr", dose_unit="mg/kg"
    )
    assert unit == "liter / kilogram"
    assert factor == pytest.approx(1e3)


def test_parameter_unit_dimensionless_and_rate() -> None:
    assert (
        parameter_unit("dimensionless", unit="ng/ml", time_unit="hr", dose_unit=None)[0]
        == "dimensionless"
    )
    assert (
        parameter_unit("1 / ({time})", unit="ng/ml", time_unit="hr", dose_unit=None)[0]
        == "1 / hour"
    )


def make_result() -> NCAResult:
    ds = xr.Dataset(
        {
            "auc_last": (
                ("individual",),
                np.array([10.0, 20.0]),
                {"units": "hour * nanogram / milliliter"},
            ),
            "cmax": (
                ("individual",),
                np.array([2.0, np.nan]),
                {"units": "nanogram / milliliter"},
            ),
            "flags": (
                ("individual",),
                np.array([0, int(NCAFlag.NO_DATA | NCAFlag.TOO_FEW_POINTS)]),
                {"units": "dimensionless"},
            ),
        },
        coords={"individual": ["a", "b"]},
    )
    return NCAResult(ds)


def test_result_access() -> None:
    result = make_result()
    assert result.sample_dims == ("individual",)
    assert result.parameters == ["auc_last", "cmax"]
    assert result.units("auc_last") == "hour * nanogram / milliliter"
    assert "cmax" in result and "vz" not in result
    np.testing.assert_allclose(result["auc_last"].values, [10.0, 20.0])


def test_result_to_quantities_and_flags() -> None:
    result = make_result()
    q = result.to_quantities(individual="a")
    assert q["auc_last"].magnitude == pytest.approx(10.0)
    assert str(q["auc_last"].units) == "hour * nanogram / milliliter"
    assert result.flags(individual="a") == []
    assert result.flags(individual="b") == ["TOO_FEW_POINTS", "NO_DATA"]
    with pytest.raises(ValueError, match="individual"):
        result.to_quantities()


def test_result_to_dataframe_and_flag_table() -> None:
    result = make_result()
    df = result.to_dataframe()
    assert list(df.columns) == ["individual", "auc_last", "cmax", "flags"]
    assert df["flags"].tolist() == ["", "TOO_FEW_POINTS|NO_DATA"]
    table = result.flag_table()
    assert table["NO_DATA"].tolist() == [False, True]
    assert table["POSITIVE_SLOPE"].tolist() == [False, False]


def test_result_requires_flags_and_units() -> None:
    ds = xr.Dataset(
        {"auc_last": (("individual",), np.array([1.0]), {"units": "hr*ng/ml"})},
        coords={"individual": ["a"]},
    )
    with pytest.raises(ValueError, match="flags"):
        NCAResult(ds)
    ds2 = xr.Dataset(
        {
            "auc_last": (("individual",), np.array([1.0])),
            "flags": (("individual",), np.array([0])),
        },
        coords={"individual": ["a"]},
    )
    with pytest.raises(ValueError, match="units"):
        NCAResult(ds2)


def test_result_parameters_exclude_derived_variables() -> None:
    ds = xr.Dataset(
        {
            "auc_last": (("i",), np.array([1.0]), {"units": "hr*mg/l"}),
            "auc_last_se": (("i",), np.array([0.1]), {"units": "hr*mg/l"}),
            "auc_last_geocv": (("i",), np.array([0.1]), {"units": "dimensionless"}),
            "n": (("i",), np.array([5.0]), {"units": "dimensionless"}),
            "flags": (("i",), np.array([0]), {"units": "dimensionless"}),
        },
        coords={"i": ["a"]},
    )
    result = NCAResult(ds)
    assert result.parameters == ["auc_last"]
    assert result.derived_variables == ["auc_last_se", "auc_last_geocv"]
    assert result.has_uncertainty
    assert list(result.to_dataframe().columns) == [
        "i",
        "auc_last",
        "auc_last_se",
        "auc_last_geocv",
        "n",
        "flags",
    ]
