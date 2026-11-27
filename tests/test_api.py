"""The conventions of the public interface: signatures, keywords and exports."""

import importlib
import inspect
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
import xarray as xr
from sbmlsim_core.chart import Chart, ChartPlot

import pkpdutils
from pkpdutils import (
    Dose,
    NCAOptions,
    Route,
    Timecourse,
    Timecourses,
    nca,
    nca_single,
    partial_auc,
)
from pkpdutils.nca import superposition
from pkpdutils.plot import draw_nca_panel, new_chart

#: one dose of a decaying curve, the data of the signature tests
T = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])


def curve() -> Timecourse:
    return Timecourse(
        time=T,
        value=10.0 * np.exp(-0.25 * T),
        time_unit="hr",
        unit="mg/l",
        dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS),
        substance="drug",
        label="s1",
    )


def test_options_is_keyword_only() -> None:
    """`nca`, `nca_single`, `partial_auc` and `superposition` take `options` by keyword."""
    tc = curve()
    batch = tc.to_batch()
    options = NCAOptions()
    with pytest.raises(TypeError):
        nca(batch, options)  # ty: ignore[too-many-positional-arguments]
    with pytest.raises(TypeError):
        nca_single(tc, options)  # ty: ignore[too-many-positional-arguments]
    with pytest.raises(TypeError):
        partial_auc(batch, 1.0, 6.0, options)  # ty: ignore[too-many-positional-arguments]
    regimen = {"dose": Dose(amount=100, unit="mg", route=Route.IV_BOLUS)}
    with pytest.raises(TypeError):
        superposition(
            tc,
            pkpdutils.DosingRegimen(**regimen, interval=12.0, n_doses=2).dosing(),
            options,  # ty: ignore[too-many-positional-arguments]
        )
    # the keyword form runs
    assert np.isfinite(nca(batch, options=options)["auc_last"].to_numpy()).all()
    assert np.isfinite(float(nca_single(tc, options=options)["auc_last"]))
    assert np.isfinite(partial_auc(batch, 1.0, 6.0, options=options).to_numpy()).all()


def test_the_namespace_carries_the_options_enumerations_and_the_models() -> None:
    """A script configures the analyses and fits a model from `pkpdutils` alone."""
    expected = {
        "AUCMethod",
        "BLQHandling",
        "BootstrapDistribution",
        "BootstrapSpread",
        "C0Method",
        "FitFlag",
        "Kind",
        "NCAFlag",
        "ParameterScale",
        "TerminalMethod",
        "UncertaintyMethod",
        "Weighting",
        "Allometric",
        "Bateman",
        "BiExp",
        "Emax",
        "Imax",
        "Linear",
        "LogLinear",
        "MonoExp",
        "Power",
        "ProportionalityResult",
        "SigmoidEmax",
        "SigmoidImax",
        "TriExp",
    }
    assert expected <= set(pkpdutils.__all__)
    for name in expected:
        assert getattr(pkpdutils, name) is not None
    options = NCAOptions(
        auc_method=pkpdutils.AUCMethod.LINEAR_LOG, kind=pkpdutils.Kind.CONCENTRATION
    )
    assert options.auc_method is pkpdutils.AUCMethod.LINEAR_LOG


def test_io_and_plot_are_reachable_after_importing_the_package() -> None:
    fresh = importlib.import_module("pkpdutils")
    assert fresh.io.read_events is not None
    assert fresh.plot.plot_timecourse is not None
    assert {"io", "plot"} <= set(pkpdutils.__all__)
    missing = "nothing"
    with pytest.raises(AttributeError, match="no attribute 'nothing'"):
        getattr(fresh, missing)


def test_importing_the_package_does_not_import_the_chart_builder() -> None:
    """`plot` is imported on first use, so a script which draws nothing skips the charts."""
    script = (
        "import sys, pkpdutils;"
        "print('pkpdutils.plot' in sys.modules);"
        "print('sbmlsim_core.chart' in sys.modules);"
        "print(pkpdutils.plot.plot_timecourse is not None);"
        "print('sbmlsim_core.chart' in sys.modules);"
        "print('matplotlib' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["False", "False", "True", "True", "False"]


def test_the_stats_helpers_are_exported() -> None:
    import pkpdutils.stats as stats

    helpers = {
        "coerce",
        "cohen_d",
        "exp_t_interval",
        "hedges_correction",
        "labels_match",
        "log_positive",
        "lognormal_from_geometric",
        "lognormal_from_moments",
        "moments_from_lognormal",
        "pooled_sd",
        "welch_df",
        "welch_se",
    }
    assert helpers <= set(stats.__all__)
    for name in helpers:
        assert getattr(stats, name) is not None


def test_the_report_and_the_power_are_exported() -> None:
    """The study report and the sample size reach the namespace of a script."""
    expected = {
        "Report",
        "study_report",
        "power_tost",
        "sample_size_tost",
        "hodges_lehmann",
    }
    assert expected <= set(pkpdutils.__all__)
    for name in expected:
        assert getattr(pkpdutils, name) is not None
    import pkpdutils.stats as stats

    assert {
        "abel_limits",
        "hodges_lehmann",
        "power_tost",
        "rsabe_criterion",
        "sample_size_tost",
    } <= set(stats.__all__)


def test_the_readers_take_column_keywords_and_covariates() -> None:
    """Every column name of a reader is a `*_col` keyword and takes `covariates`."""
    from pkpdutils import io

    for reader in (
        io.read_events,
        io.read_pknca,
        io.read_adnca,
        io.write_events,
        io.write_pknca,
        io.write_adnca,
    ):
        parameters = inspect.signature(reader).parameters
        columns = [
            name
            for name, p in parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
            and isinstance(p.default, str)
            and name
            not in {
                "time_unit",
                "unit",
                "dose_unit",
                "dim",
                "analyte_dim",
                "substance",
                "analyte",
            }
        ]
        assert columns, reader.__name__
        assert all(name.endswith("_col") for name in columns), (
            reader.__name__,
            columns,
        )
    for reader in (io.read_events, io.read_pknca, io.read_adnca):
        assert "covariates" in inspect.signature(reader).parameters


#: the exported figures and the data arguments they take before the keywords
PLOT_DATA_ARGUMENTS = {
    "plot_timecourse": 1,
    "plot_nca": 2,
    "plot_nca_grid": 2,
    "plot_intervals": 2,
    "plot_parameters": 3,
    "plot_fit": 1,
    "plot_goodness_of_fit": 1,
    "plot_dose_proportionality": 1,
    "plot_bland_altman": 1,
    "plot_ratio": 1,
    "plot_forest": 1,
    "draw_nca_panel": 3,
}


@pytest.mark.parametrize("name", sorted(PLOT_DATA_ARGUMENTS))
def test_every_figure_has_the_same_signature(name: str) -> None:
    """`f(data, *, <options>, chart=None, row=0, column=0, style=DEFAULT_STYLE)`."""
    import pkpdutils.plot as plot

    function: Any = getattr(plot, name)
    assert name in plot.__all__
    parameters = list(inspect.signature(function).parameters.values())
    positional = [
        p
        for p in parameters
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and p.name not in {"self"}
    ]
    assert len(positional) <= PLOT_DATA_ARGUMENTS[name]
    keywords = {
        p.name: p for p in parameters if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert "style" in keywords
    assert keywords["chart"].default is None
    assert keywords["row"].default == 0 and keywords["column"].default == 0
    assert "log" not in keywords
    assert all(
        not key.startswith("log") or key in {"log_x", "log_y", "log_ratio"}
        for key in keywords
    )
    # `style` is the last keyword of every signature, the cell of the chart
    # before it
    named = [p.name for p in parameters if p.kind is inspect.Parameter.KEYWORD_ONLY]
    assert named[-4:] == ["chart", "row", "column", "style"]
    assert not {"ax", "axes", "fig"} & set(named)


def test_draw_nca_panel_returns_the_panel() -> None:
    tc = curve()
    result = nca_single(tc)
    values = {name: float(result[name]) for name in result.parameters}
    panel = draw_nca_panel(tc, values, result.flags())
    assert isinstance(panel, ChartPlot)
    assert isinstance(panel.chart, Chart)
    chart = new_chart(12, 4, columns=2)
    second = draw_nca_panel(tc, values, [], chart=chart, column=1)
    assert second.chart is chart and second.column == 1


def test_the_batch_and_the_curve_convert_into_each_other() -> None:
    tc = curve()
    batch = tc.to_batch(dim="subject", label="x")
    assert isinstance(batch, Timecourses)
    assert batch.sel(subject="x") == tc.model_copy(update={"label": "x"})


def test_every_figure_draws_into_the_chart_it_is_given() -> None:
    """`chart`, `row` and `column` place the panels of a figure into a given chart."""
    from pkpdutils import (
        ParameterSample,
        Power,
        fit_table,
        meta_analysis,
        proportionality_test,
        ratio,
    )
    from pkpdutils.plot import (
        plot_bland_altman,
        plot_dose_proportionality,
        plot_fit,
        plot_forest,
        plot_goodness_of_fit,
        plot_intervals,
        plot_nca,
        plot_nca_grid,
        plot_parameters,
        plot_ratio,
        plot_timecourse,
    )
    from pkpdutils.stats import Study

    tc = curve()
    batch = tc.to_batch()
    result = nca(batch)
    doses = np.array([25.0, 50.0, 100.0, 200.0])
    table = xr.Dataset(
        {"auc": (("dose",), 2.0 * doses**1.05)},
        coords={"dose": doses},
    )
    power = fit_table(Power(), table, "dose", "auc", dim="dose")
    grid = new_chart(10, 8, rows=2, columns=2)
    assert plot_timecourse(tc, chart=grid) is grid
    assert plot_goodness_of_fit(power, chart=grid, column=1) is grid
    assert plot_bland_altman(power, chart=grid, row=1) is grid
    assert (
        plot_dose_proportionality(
            power,
            test=proportionality_test(power, dose_range=(25.0, 200.0)),
            chart=grid,
            row=1,
            column=1,
        )
        is grid
    )
    spec = grid.spec()
    assert [(p["row"], p["column"]) for p in spec["plots"]] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert spec["plots"][0]["x"]["label"].startswith("time")
    panels = new_chart(11, 9, rows=2, columns=2)
    assert plot_nca(tc, result, annotate=False, chart=panels, individual="s1") is panels
    assert plot_nca_grid(batch, result, ncols=1, chart=panels, row=1) is panels
    assert plot_fit(power, chart=panels, column=1) is panels
    assert len(panels.spec()["plots"]) == 5
    # a figure which does not fit into the grid of the chart is refused
    with pytest.raises(ValueError, match="need a grid"):
        plot_nca(tc, result, chart=panels, column=1, individual="s1")
    # `plot_intervals` draws into a chart as well
    single = new_chart(6, 4)
    protocol = pkpdutils.Dosing.regimen(
        Dose(amount=100, unit="mg", route=Route.IV_BOLUS), interval=6.0, n_doses=3
    )
    times = np.array([0.5, 1.0, 3.0, 6.5, 7.0, 9.0, 12.5, 13.0, 15.0, 17.0])
    multi = Timecourse(
        time=times,
        value=10.0 * np.exp(-0.25 * (times % 6.0)),
        time_unit="hr",
        unit="mg/l",
        dosing=protocol,
    )
    intervals = nca_single(multi)
    assert plot_intervals(intervals, "interval_auc", chart=single) is single
    # the figures of the parameters and of the statistics
    curves = [
        tc.model_copy(
            update={"value": tc.value * scale, "label": label},
        )
        for scale, label in ((1.0, "a"), (1.2, "b"), (0.8, "c"))
    ]
    group = Timecourses.from_timecourses(curves)
    group.ds.coords["sex"] = ("individual", ["m", "f", "m"])
    parameters = nca(group)
    labels = np.array(["a", "b", "c"])
    test_sample = ParameterSample(values=np.array([12.0, 14.0, 11.0]), labels=labels)
    reference_sample = ParameterSample(
        values=np.array([10.0, 12.0, 10.5]), labels=labels
    )
    meta = meta_analysis(
        [
            Study(
                label=f"study {i}",
                control=ParameterSample(mean=1.2, sd=0.4, n=10),
                treatment=ParameterSample(mean=1.6 + 0.1 * i, sd=0.5, n=10),
            )
            for i in range(3)
        ]
    )
    stats = new_chart(15, 4, columns=3)
    assert (
        plot_parameters(parameters, "auc_inf_obs", "individual", by="sex", chart=stats)
        is stats
    )
    assert (
        plot_ratio({"auc": ratio(test_sample, reference_sample)}, chart=stats, column=1)
        is stats
    )
    assert plot_forest(meta, chart=stats, column=2) is stats
    assert [p["column"] for p in stats.spec()["plots"]] == [0, 1, 2]
    stats.svg()
