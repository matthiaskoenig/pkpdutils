import numpy as np

import pkpdutils


def test_version() -> None:
    assert pkpdutils.__version__ == "1.1.0"


def test_exports() -> None:
    from pkpdutils import Q_, Dose, DosingRegimen, Route, Timecourse, Timecourses, ureg

    assert Route.ORAL.value == "oral"
    assert Dose and DosingRegimen and Timecourse and Timecourses and Q_ and ureg


def test_io_module_is_reachable() -> None:
    # `pkpdutils.io` is a module, not a set of top level names
    assert callable(pkpdutils.io.read_events)
    assert callable(pkpdutils.io.write_events)
    assert callable(pkpdutils.io.read_pknca) and callable(pkpdutils.io.read_adnca)
    assert "read_events" not in pkpdutils.__all__


def test_nca_exports() -> None:
    from pkpdutils import NCAOptions, NCAResult, TerminalPhase, nca, nca_single

    assert callable(nca) and callable(nca_single)
    assert NCAOptions().terminal == TerminalPhase()
    assert NCAResult is not None


def test_fit_exports() -> None:
    from pkpdutils import (
        FitOptions,
        FitResult,
        compare_models,
        fit,
        fit_table,
        fit_timecourse,
        fit_timecourses,
        proportionality_test,
    )

    assert callable(fit) and callable(fit_timecourses) and callable(fit_table)
    assert callable(fit_timecourse)
    assert callable(compare_models) and callable(proportionality_test)
    assert FitOptions().n_starts == 1 and FitResult is not None


def test_stats_exports() -> None:
    from pkpdutils import (
        ParameterSample,
        bioequivalence,
        compare,
        ddi_classification,
        meta_analysis,
        ratio,
    )

    assert callable(compare) and callable(ratio) and callable(bioequivalence)
    assert callable(ddi_classification) and callable(meta_analysis)
    assert ParameterSample(values=np.array([1.0, 2.0])).size == 2
