"""A batch of several routes: the route of every sample drives its parameters."""

import numpy as np
import pytest

from pkpdutils import Dose, NCAOptions, Route, Timecourse, Timecourses, nca

TIME = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
K = 0.2
#: an intravenous bolus, `C(t) = C0 exp(-k t)`, and an oral curve with absorption
IV = 10.0 * np.exp(-K * TIME)
ORAL = 10.0 * (np.exp(-K * TIME) - np.exp(-1.5 * TIME))


def curve(label: str, values: np.ndarray, route: Route) -> Timecourse:
    return Timecourse(
        time=TIME,
        value=values,
        time_unit="hr",
        unit="mg/l",
        substance="drug",
        dose=Dose(amount=100, unit="mg", route=route),
        label=label,
    )


def crossover() -> Timecourses:
    """Two intravenous and two oral subjects in one batch."""
    return Timecourses.from_timecourses(
        [
            curve("iv1", IV, Route.IV_BOLUS),
            curve("po1", ORAL, Route.ORAL),
            curve("iv2", 1.2 * IV, Route.IV_BOLUS),
            curve("po2", 1.2 * ORAL, Route.ORAL),
        ]
    )


def test_from_timecourses_builds_the_route_coordinate() -> None:
    """The routes become a coordinate and `route` raises for the batch."""
    batch = crossover()
    assert list(batch.ds["route"].to_numpy()) == [
        "iv_bolus",
        "oral",
        "iv_bolus",
        "oral",
    ]
    assert "route" not in batch.ds.attrs
    routes = batch.routes
    assert routes is not None
    assert list(routes) == [Route.IV_BOLUS, Route.ORAL, Route.IV_BOLUS, Route.ORAL]
    with pytest.raises(ValueError, match="carries the routes"):
        _ = batch.route
    dose = batch.sel(individual="po1").dose
    assert dose is not None and dose.route is Route.ORAL


def test_the_analysis_follows_the_route_of_every_row() -> None:
    """`cl` for the intravenous rows, `cl_f` for the oral ones, `c0` for the bolus."""
    result = nca(crossover())
    frame = result.to_dataframe().set_index("individual")
    for label in ("iv1", "iv2"):
        assert np.isfinite(frame.loc[label, "cl"])
        assert np.isnan(frame.loc[label, "cl_f"])
        assert np.isfinite(frame.loc[label, "c0"])
        assert np.isfinite(frame.loc[label, "vss"])
        assert np.isnan(frame.loc[label, "tlag"])
    for label in ("po1", "po2"):
        assert np.isnan(frame.loc[label, "cl"])
        assert np.isfinite(frame.loc[label, "cl_f"])
        assert np.isnan(frame.loc[label, "c0"])
        assert np.isnan(frame.loc[label, "vss"])
        assert frame.loc[label, "tlag"] == 0.0
    # the bolus extrapolates back to the dose, C0 = 10 mg/l
    np.testing.assert_allclose(frame.loc["iv1", "c0"], 10.0, rtol=1e-6)
    np.testing.assert_allclose(frame.loc["iv2", "c0"], 12.0, rtol=1e-6)
    # CL = dose / AUC(0-inf) of a one compartment bolus is dose * k / C0
    np.testing.assert_allclose(frame.loc["iv1", "cl"], 100.0 * K / 10.0, rtol=1e-3)


def test_the_rows_of_one_route_are_the_rows_of_their_own_batch() -> None:
    """Grouping by route changes nothing for the rows of one route."""
    mixed = nca(crossover())
    oral = nca(
        Timecourses.from_timecourses(
            [curve("po1", ORAL, Route.ORAL), curve("po2", 1.2 * ORAL, Route.ORAL)]
        )
    )
    intravenous = nca(
        Timecourses.from_timecourses(
            [curve("iv1", IV, Route.IV_BOLUS), curve("iv2", 1.2 * IV, Route.IV_BOLUS)]
        )
    )
    frame = mixed.to_dataframe().set_index("individual")
    for single, labels in ((oral, ["po1", "po2"]), (intravenous, ["iv1", "iv2"])):
        alone = single.to_dataframe().set_index("individual")
        for name in ("cmax", "auc_last", "auc_inf_obs", "lambda_z", "thalf", "mrt"):
            np.testing.assert_allclose(
                frame.loc[labels, name].to_numpy(),
                alone.loc[labels, name].to_numpy(),
                rtol=1e-12,
            )


def test_the_partial_areas_follow_the_route_as_well() -> None:
    """The value at the dose of a partial area is `c0` for a bolus and 0 orally."""
    options = NCAOptions(partial_aucs={"auc_0_4": (0.0, 4.0)})
    mixed = nca(crossover(), options=options)
    alone = nca(
        Timecourses.from_timecourses([curve("iv1", IV, Route.IV_BOLUS)]),
        options=options,
    )
    frame = mixed.to_dataframe().set_index("individual")
    np.testing.assert_allclose(
        frame.loc["iv1", "auc_0_4"],
        float(alone.ds["auc_0_4"].to_numpy().ravel()[0]),
        rtol=1e-12,
    )


def test_nca_options_units_convert_the_result() -> None:
    """`NCAOptions.units` reports the parameters in the units of the submission."""
    batch = Timecourses.from_timecourses([curve("iv1", IV, Route.IV_BOLUS)])
    plain = nca(batch)
    converted = nca(batch, options=NCAOptions(units={"auc_inf_obs": "min*mg/L"}))
    assert plain.units("auc_inf_obs") == "hour * milligram / liter"
    assert converted.units("auc_inf_obs") == "milligram * minute / liter"
    np.testing.assert_allclose(
        converted.ds["auc_inf_obs"], plain.ds["auc_inf_obs"] * 60.0
    )
    assert converted.units("cmax") == plain.units("cmax")


def test_the_two_arms_of_a_crossover_give_the_absolute_bioavailability() -> None:
    """`select` cuts the batch by the route coordinate and the result names it."""
    from pkpdutils.nca import bioavailability

    batch = Timecourses.from_timecourses(
        [
            curve("s1", IV, Route.IV_BOLUS),
            curve("s2", 1.1 * IV, Route.IV_BOLUS),
            curve("s1", 8.0 * (np.exp(-K * TIME) - np.exp(-1.5 * TIME)), Route.ORAL),
            curve(
                "s2", 1.1 * 8.0 * (np.exp(-K * TIME) - np.exp(-1.5 * TIME)), Route.ORAL
            ),
        ],
        labels=["s1", "s2", "s1", "s2"],
    )
    intravenous = batch.select(route="iv_bolus")
    oral = batch.select(route="oral")
    assert intravenous.route is Route.IV_BOLUS
    assert oral.route is Route.ORAL
    estimate = bioavailability(nca(oral), nca(intravenous), dim="individual")
    # the reference is intravenous, which the result of the arm names itself
    assert estimate.name == "f_abs"
    # F = (8 (1/k - 1/1.5)) / (10 / k) with the trapezoid areas of the grid
    assert estimate.gmr == pytest.approx(0.68, abs=0.02)
