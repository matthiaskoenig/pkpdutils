import pytest

from pkpdutils.fit.model import ModelParameter
from pkpdutils.fit.options import (
    FitFlag,
    FitOptions,
    ParameterScale,
    Weighting,
    decode_fit_flags,
)


def test_defaults() -> None:
    o = FitOptions()
    assert o.parameter_scale is ParameterScale.LOG10
    assert o.weighting is Weighting.NONE
    assert o.loss == "linear"
    assert o.n_starts == 1 and o.seed is None and o.n_workers is None
    assert o.ci_level == pytest.approx(0.95) and o.bootstrap == 0
    assert o.fixed == {} and o.bounds == {} and o.initial == {}
    assert o.start_spread == pytest.approx(100.0)


def test_validation() -> None:
    with pytest.raises(ValueError):
        FitOptions(n_starts=0)
    with pytest.raises(ValueError):
        FitOptions(bootstrap=-1)
    with pytest.raises(ValueError):
        FitOptions(start_spread=1.0)
    with pytest.raises(ValueError):
        FitOptions(loss="quadratic")
    with pytest.raises(ValueError):
        FitOptions(bounds={"a": (2.0, 1.0)})


def test_scale_of() -> None:
    pos = ModelParameter("k", "1/[x]")
    lin = ModelParameter("e0", "[y]", positive=False)
    assert FitOptions().scale_of(pos) is ParameterScale.LOG10
    assert FitOptions().scale_of(lin) is ParameterScale.LINEAR
    assert (
        FitOptions(parameter_scale=ParameterScale.LINEAR).scale_of(pos)
        is ParameterScale.LINEAR
    )
    assert (
        FitOptions(parameter_scale=ParameterScale.LOG).scale_of(pos)
        is ParameterScale.LOG
    )


def test_flags() -> None:
    assert decode_fit_flags(int(FitFlag.NOT_CONVERGED | FitFlag.FLIP_FLOP)) == [
        "NOT_CONVERGED",
        "FLIP_FLOP",
    ]
    assert FitFlag.NO_DATA == 32
