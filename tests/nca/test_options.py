import pytest

from pkpdutils.nca.options import (
    AUCMethod,
    BLQHandling,
    C0Method,
    Kind,
    NCAFlag,
    NCAOptions,
    TerminalMethod,
    TerminalPhase,
    decode_flags,
)
from pkpdutils.timecourse import Dose, DosingRegimen


def test_defaults() -> None:
    options = NCAOptions()
    assert options.kind is Kind.CONCENTRATION
    assert options.auc_method is AUCMethod.LINEAR_LOG
    assert options.terminal.method is TerminalMethod.BEST_FIT
    assert options.terminal.min_points == 3
    assert options.terminal.exclude_cmax
    assert options.lloq is None
    assert options.blq is BLQHandling.NAN
    assert options.c0_method is C0Method.LOG_BACK_EXTRAPOLATION
    assert options.extrapolation_warning == pytest.approx(0.2)
    assert options.regimen is None
    assert options.effect_threshold is None
    assert options.n_workers is None


def test_terminal_last_n_requires_n_points() -> None:
    with pytest.raises(ValueError, match="n_points"):
        TerminalPhase(method=TerminalMethod.LAST_N)
    assert TerminalPhase(method=TerminalMethod.LAST_N, n_points=4).n_points == 4


def test_terminal_manual_requires_points() -> None:
    with pytest.raises(ValueError, match="points"):
        TerminalPhase(method=TerminalMethod.MANUAL)
    with pytest.raises(ValueError, match="min_points"):
        TerminalPhase(method=TerminalMethod.MANUAL, points=(5, 6))
    assert TerminalPhase(method=TerminalMethod.MANUAL, points=(4, 5, 6)).points == (
        4,
        5,
        6,
    )


def test_min_points_at_least_three() -> None:
    with pytest.raises(ValueError):
        TerminalPhase(min_points=2)


def test_options_validation() -> None:
    with pytest.raises(ValueError):
        NCAOptions(lloq=0)
    with pytest.raises(ValueError):
        NCAOptions(extrapolation_warning=1.5)
    with pytest.raises(ValueError):
        NCAOptions(n_workers=0)
    regimen = DosingRegimen(dose=Dose(amount=100, unit="mg"), interval=12)
    assert NCAOptions(regimen=regimen).regimen is regimen


def test_options_frozen() -> None:
    options = NCAOptions()
    with pytest.raises(ValueError):
        options.lloq = 1.0  # ty: ignore[invalid-assignment]


def test_flags() -> None:
    value = int(NCAFlag.POSITIVE_SLOPE | NCAFlag.EXTRAPOLATION_HIGH)
    assert value == 5
    assert decode_flags(value) == ["POSITIVE_SLOPE", "EXTRAPOLATION_HIGH"]
    assert decode_flags(0) == []
