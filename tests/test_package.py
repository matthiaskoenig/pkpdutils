import pkpdutils


def test_version() -> None:
    assert pkpdutils.__version__ == "1.0.0.dev0"


def test_exports() -> None:
    from pkpdutils import Q_, Dose, DosingRegimen, Route, Timecourse, Timecourses, ureg

    assert Route.ORAL.value == "oral"
    assert Dose and DosingRegimen and Timecourse and Timecourses and Q_ and ureg
