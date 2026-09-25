"""The parameter tables of other NCA tools: Phoenix WinNonlin, PKNCA and NonCompart.

Every tool names the same parameter differently (`AUCINF_obs` in Phoenix
WinNonlin, `aucinf.obs` in PKNCA, `AUCIFO` in NonCompart, `auc_inf_obs` here),
reports a fraction as a percentage and lays its results out in its own table.
This module holds the crosswalk from the variables of an `NCAResult` to the
names of each tool and writes and reads the result table of each:

- Phoenix WinNonlin, the "Final Parameters Pivoted" table: one row per
  profile, one column per parameter (`to_winnonlin`, `write_winnonlin`,
  `read_winnonlin`, `WINNONLIN_NAMES`);
- PKNCA, the long table of `as.data.frame(pk.nca(...))`: one row per profile,
  interval and parameter with `PPTESTCD` and `PPORRES` (`to_pknca_results`,
  `write_pknca_results`, `read_pknca_results`, `PKNCA_NAMES`);
- NonCompart, the wide table of `tblNCA`: one row per profile, the CDISC
  `PPTESTCD` code of every parameter as its column (`to_noncompart`,
  `write_noncompart`, `read_noncompart`, `NONCOMPART_NAMES`).

A writer exports what the result carries: a parameter which the tool reports
and the result does not is left out, and so is a variable which the tool has
no name for. A reader returns one row per profile with the variables of
`pkpdutils` as columns and the percentages as fractions, so that a published
result table of another tool compares against `NCAResult.to_dataframe`
column by column; a column without a `pkpdutils` name is dropped and logged.

The names are taken from the tools themselves: the WinNonlin columns from the
Phoenix output of the NonCompart validation report (Han 2018), the PKNCA
parameters from `get.interval.cols()` of PKNCA 0.12.1 and the NonCompart
columns from `sNCA` of NonCompart 0.8.4. The page "Benchmark datasets" of the
documentation compares every name against the output of the three tools.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pkpdutils.result import ParameterResult
from pkpdutils.timecourse import Route

logger = logging.getLogger(__name__)

#: variables of `pkpdutils` to the columns of the Phoenix WinNonlin "Final
#: Parameters Pivoted" table, in the order Phoenix writes them. A clearance and
#: a volume carry the route in their `pkpdutils` name already (`cl` against
#: `cl_f`), which is where WinNonlin puts it as well (`Cl_obs` against
#: `Cl_F_obs`), so no name depends on the route.
WINNONLIN_NAMES: dict[str, str] = {
    "lambda_z_r2": "Rsq",
    "lambda_z_r2_adj": "Rsq_adjusted",
    "lambda_z_n_points": "No_points_lambda_z",
    "lambda_z": "Lambda_z",
    "lambda_z_t_first": "Lambda_z_lower",
    "lambda_z_t_last": "Lambda_z_upper",
    "thalf": "HL_Lambda_z",
    "tlag": "Tlag",
    "tmax": "Tmax",
    "cmax": "Cmax",
    "cmax_dn": "Cmax_D",
    "tlast": "Tlast",
    "clast": "Clast",
    "clast_pred": "Clast_pred",
    "c0": "C0",
    "auc_last": "AUClast",
    "auc_all": "AUCall",
    "auc_inf_obs": "AUCINF_obs",
    "auc_inf_dn": "AUCINF_D_obs",
    "auc_extrap_fraction": "AUC_%Extrap_obs",
    "auc_back_extrap_fraction": "AUC_%Back_Ext_obs",
    "vz_f": "Vz_F_obs",
    "cl_f": "Cl_F_obs",
    "vz": "Vz_obs",
    "cl": "Cl_obs",
    "auc_inf_pred": "AUCINF_pred",
    "auc_inf_pred_dn": "AUCINF_D_pred",
    "auc_extrap_fraction_pred": "AUC_%Extrap_pred",
    "auc_back_extrap_fraction_pred": "AUC_%Back_Ext_pred",
    "vz_f_pred": "Vz_F_pred",
    "cl_f_pred": "Cl_F_pred",
    "vz_pred": "Vz_pred",
    "cl_pred": "Cl_pred",
    "aumc_last": "AUMClast",
    "aumc_inf": "AUMCINF_obs",
    "aumc_extrap_fraction": "AUMC_%Extrap_obs",
    "aumc_inf_pred": "AUMCINF_pred",
    "aumc_extrap_fraction_pred": "AUMC_%Extrap_pred",
    "mrt_last": "MRTlast",
    "mrt": "MRTINF_obs",
    "mrt_pred": "MRTINF_pred",
    "vss": "Vss_obs",
    "vss_pred": "Vss_pred",
}

#: the WinNonlin columns which carry a percentage where `pkpdutils` reports a fraction
WINNONLIN_PERCENT: frozenset[str] = frozenset(
    name for name in WINNONLIN_NAMES.values() if "%" in name
)

#: WinNonlin columns which have no `pkpdutils` variable: `Corr_XY`, the
#: correlation of the terminal regression, is \(-\sqrt{R^2}\) and not reported
WINNONLIN_UNMAPPED: frozenset[str] = frozenset({"Corr_XY"})

#: variables of `pkpdutils` to the `PPTESTCD` of PKNCA which do not depend on the
#: route. PKNCA does not tell a clearance from a clearance over the
#: bioavailability: `cl` and `cl_f` both are `cl.obs`.
PKNCA_NAMES: dict[str, str] = {
    "cmax": "cmax",
    "tmax": "tmax",
    "cmin": "cmin",
    "tlast": "tlast",
    "clast": "clast.obs",
    "clast_pred": "clast.pred",
    "tlag": "tlag",
    "c0": "c0",
    "auc_last": "auclast",
    "auc_all": "aucall",
    "auc_inf_obs": "aucinf.obs",
    "auc_inf_pred": "aucinf.pred",
    "aumc_last": "aumclast",
    "aumc_all": "aumcall",
    "aumc_inf": "aumcinf.obs",
    "aumc_inf_pred": "aumcinf.pred",
    "auc_extrap_fraction": "aucpext.obs",
    "auc_extrap_fraction_pred": "aucpext.pred",
    "thalf": "half.life",
    "lambda_z": "lambda.z",
    "lambda_z_r2": "r.squared",
    "lambda_z_r2_adj": "adj.r.squared",
    "lambda_z_t_first": "lambda.z.time.first",
    "lambda_z_t_last": "lambda.z.time.last",
    "lambda_z_n_points": "lambda.z.n.points",
    "lambda_z_span": "span.ratio",
    "cl": "cl.obs",
    "cl_f": "cl.obs",
    "cl_pred": "cl.pred",
    "cl_f_pred": "cl.pred",
    "vz": "vz.obs",
    "vz_f": "vz.obs",
    "vz_pred": "vz.pred",
    "vz_f_pred": "vz.pred",
    "cmax_dn": "cmax.dn",
    "auc_last_dn": "auclast.dn",
    "auc_all_dn": "aucall.dn",
    "auc_inf_dn": "aucinf.obs.dn",
    "auc_inf_pred_dn": "aucinf.pred.dn",
    "ctrough": "ctrough",
    "cavg": "cav",
    "ptr": "ptr",
}

#: variables of `pkpdutils` whose PKNCA parameter depends on the route: the
#: mean residence time of an intravenous dose is corrected by half the
#: duration of an infusion (`mrt.iv.obs`), and the steady state volume is only
#: defined for it (`vss.iv.obs`)
PKNCA_NAMES_BY_ROUTE: dict[str, dict[Route, str]] = {
    "mrt": {
        Route.ORAL: "mrt.obs",
        Route.IV_BOLUS: "mrt.iv.obs",
        Route.IV_INFUSION: "mrt.iv.obs",
    },
    "mrt_last": {
        Route.ORAL: "mrt.last",
        Route.IV_BOLUS: "mrt.iv.last",
        Route.IV_INFUSION: "mrt.iv.last",
    },
    "mrt_pred": {
        Route.ORAL: "mrt.pred",
        Route.IV_BOLUS: "mrt.iv.pred",
        Route.IV_INFUSION: "mrt.iv.pred",
    },
    "thalf_eff": {
        Route.ORAL: "thalf.eff.obs",
        Route.IV_BOLUS: "thalf.eff.iv.obs",
        Route.IV_INFUSION: "thalf.eff.iv.obs",
    },
    "vss": {Route.IV_BOLUS: "vss.iv.obs", Route.IV_INFUSION: "vss.iv.obs"},
    "vss_pred": {Route.IV_BOLUS: "vss.iv.pred", Route.IV_INFUSION: "vss.iv.pred"},
}

#: the PKNCA parameters which carry a percentage where `pkpdutils` reports a fraction
PKNCA_PERCENT: frozenset[str] = frozenset({"aucpext.obs", "aucpext.pred"})

#: variables of `pkpdutils` to the columns of NonCompart which do not depend on
#: the route; they are the `PPTESTCD` codes of CDISC
NONCOMPART_NAMES: dict[str, str] = {
    "cmax": "CMAX",
    "cmax_dn": "CMAXD",
    "tmax": "TMAX",
    "tlag": "TLAG",
    "clast": "CLST",
    "clast_pred": "CLSTP",
    "tlast": "TLST",
    "thalf": "LAMZHL",
    "lambda_z": "LAMZ",
    "lambda_z_t_first": "LAMZLL",
    "lambda_z_t_last": "LAMZUL",
    "lambda_z_n_points": "LAMZNPT",
    "lambda_z_r2": "R2",
    "lambda_z_r2_adj": "R2ADJ",
    "auc_last": "AUCLST",
    "auc_all": "AUCALL",
    "auc_inf_obs": "AUCIFO",
    "auc_inf_dn": "AUCIFOD",
    "auc_inf_pred": "AUCIFP",
    "auc_inf_pred_dn": "AUCIFPD",
    "auc_extrap_fraction": "AUCPEO",
    "auc_extrap_fraction_pred": "AUCPEP",
    "aumc_last": "AUMCLST",
    "aumc_inf": "AUMCIFO",
    "aumc_inf_pred": "AUMCIFP",
    "aumc_extrap_fraction": "AUMCPEO",
    "aumc_extrap_fraction_pred": "AUMCPEP",
    "c0": "C0",
    "auc_back_extrap_fraction": "AUCPBEO",
    "auc_back_extrap_fraction_pred": "AUCPBEP",
    "vz": "VZO",
    "vz_pred": "VZP",
    "cl": "CLO",
    "cl_pred": "CLP",
    "vss": "VSSO",
    "vss_pred": "VSSP",
    "vz_f": "VZFO",
    "vz_f_pred": "VZFP",
    "cl_f": "CLFO",
    "cl_f_pred": "CLFP",
}

#: variables of `pkpdutils` whose NonCompart column depends on the route: the
#: mean residence time is `MRTEV*` after an extravascular dose and `MRTIV*`
#: after a bolus and an infusion alike (where CDISC has `MRTIB*` and `MRTIC*`)
NONCOMPART_NAMES_BY_ROUTE: dict[str, dict[Route, str]] = {
    "mrt_last": {
        Route.ORAL: "MRTEVLST",
        Route.IV_BOLUS: "MRTIVLST",
        Route.IV_INFUSION: "MRTIVLST",
    },
    "mrt": {
        Route.ORAL: "MRTEVIFO",
        Route.IV_BOLUS: "MRTIVIFO",
        Route.IV_INFUSION: "MRTIVIFO",
    },
    "mrt_pred": {
        Route.ORAL: "MRTEVIFP",
        Route.IV_BOLUS: "MRTIVIFP",
        Route.IV_INFUSION: "MRTIVIFP",
    },
}

#: the NonCompart columns which carry a percentage where `pkpdutils` reports a fraction
NONCOMPART_PERCENT: frozenset[str] = frozenset(
    {"AUCPEO", "AUCPEP", "AUMCPEO", "AUMCPEP", "AUCPBEO", "AUCPBEP"}
)

#: NonCompart columns which have no `pkpdutils` variable: the intercept `b0`
#: of the regression is \(\ln\) of `lambda_z_intercept` only for a positive
#: curve, and `CORRXY` is \(-\sqrt{R^2}\)
NONCOMPART_UNMAPPED: frozenset[str] = frozenset({"b0", "CORRXY"})

#: the clearances and volumes of an intravenous dose, which PKNCA names like
#: those of an extravascular dose (`_EXTRAVASCULAR_ONLY`)
_INTRAVENOUS_ONLY: frozenset[str] = frozenset({"cl", "cl_pred", "vz", "vz_pred"})

#: the clearances and volumes over the bioavailability of an extravascular dose
_EXTRAVASCULAR_ONLY: frozenset[str] = frozenset(
    {"cl_f", "cl_f_pred", "vz_f", "vz_f_pred"}
)

#: the columns of the long PKNCA table, `as.data.frame(pk.nca(...))`, after the
#: grouping columns
PKNCA_COLUMNS: tuple[str, ...] = ("start", "end", "PPTESTCD", "PPORRES", "exclude")


def _result_route(result: ParameterResult) -> Route | None:
    """The route every sample of a result shares, `None` when it names none.

    Raises:
        ValueError: if the samples of the result were given different routes.
    """
    ds = result.ds
    if "route" in ds.coords:
        values = {str(value) for value in np.asarray(ds["route"].values).ravel()}
        if len(values) > 1:
            raise ValueError(
                f"the result mixes the routes {sorted(values)}; export every "
                "route on its own (`NCAResult.select`)"
            )
        return Route(values.pop())
    route = ds.attrs.get("route")
    return None if route is None else Route(route)


def _names_for(
    names: Mapping[str, str],
    by_route: Mapping[str, Mapping[Route, str]],
    route: Route | None,
) -> dict[str, str]:
    """The name of every variable for one route, route independent names first."""
    resolved = dict(names)
    if route is not None:
        for variable, codes in by_route.items():
            if route in codes:
                resolved[variable] = codes[route]
    return resolved


def _wide(
    result: ParameterResult, names: Mapping[str, str], percent: frozenset[str]
) -> pd.DataFrame:
    """One row per sample: the sample dimensions, then the named parameters.

    The columns follow the order of `names`, a percentage is the fraction
    times 100 and a parameter which is `NaN` for every sample is left out (a
    parameter of the other route of a batch of several analytes).
    """
    frame = result.to_dataframe()
    columns = {
        variable: name
        for variable, name in names.items()
        if variable in result.parameters and frame[variable].notna().any()
    }
    out = frame[[*result.sample_dims, *columns]].rename(columns=columns)
    for name in columns.values():
        if name in percent:
            out[name] = out[name] * 100.0
    return out.reset_index(drop=True)


def _read_wide(
    frame: pd.DataFrame,
    names: Mapping[str, str],
    by_route: Mapping[str, Mapping[Route, str]],
    percent: frozenset[str],
    unmapped: frozenset[str],
    tool: str,
) -> pd.DataFrame:
    """The variables of `pkpdutils` from a wide table of a tool.

    The columns before the first parameter column are the grouping columns
    of the profiles and become the index.
    """
    reverse = {name: variable for variable, name in names.items()}
    for variable, codes in by_route.items():
        reverse.update(dict.fromkeys(codes.values(), variable))
    known = set(reverse) | set(unmapped)
    groups: list[str] = []
    for column in frame.columns:
        if column in known:
            break
        groups.append(str(column))
    dropped = [str(c) for c in frame.columns if c not in groups and c not in reverse]
    if dropped:
        logger.info("%s columns without a pkpdutils name dropped: %s", tool, dropped)
    out = frame[[c for c in frame.columns if c in reverse]].astype(np.float64)
    for column in out.columns:
        if column in percent:
            out[column] = out[column] / 100.0
    out = out.rename(columns=reverse)
    if groups:
        out.index = (
            pd.Index(frame[groups[0]])
            if len(groups) == 1
            else pd.MultiIndex.from_frame(frame[groups])
        )
    return out


def _frame(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """A table given as a data frame or as the path of a csv file."""
    if isinstance(source, pd.DataFrame):
        return source
    return pd.read_csv(source)


def to_winnonlin(result: ParameterResult) -> pd.DataFrame:
    """Lay a result out as the "Final Parameters Pivoted" table of Phoenix WinNonlin.

    One row per sample with the sample dimensions of the result as the sort
    columns, then every parameter the result carries under its WinNonlin name
    (`WINNONLIN_NAMES`) in the order Phoenix writes them. The extrapolated
    fractions are percentages (`AUC_%Extrap_obs`), as WinNonlin reports them.
    The table carries no units: Phoenix writes them into a row of their own,
    which the export of the pivoted table leaves out.

    Args:
        result: the result, e.g. of `pkpdutils.nca.nca`

    Returns:
        The table, one row per sample.
    """
    return _wide(result, WINNONLIN_NAMES, WINNONLIN_PERCENT)


def write_winnonlin(result: ParameterResult, path: str | Path) -> pd.DataFrame:
    """Write the "Final Parameters Pivoted" table of a result as a csv file.

    Args:
        result: the result
        path: the file to write

    Returns:
        The table that was written, `to_winnonlin`.
    """
    table = to_winnonlin(result)
    table.to_csv(path, index=False)
    return table


def read_winnonlin(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read a "Final Parameters Pivoted" table of Phoenix WinNonlin.

    The sort columns before the first parameter (`Subject`) become the index,
    every parameter column with a `pkpdutils` name (`WINNONLIN_NAMES`) becomes
    that variable and the percentages become fractions. `Corr_XY` and any
    column without a name are dropped.

    Args:
        source: the table or the path of its csv file

    Returns:
        One row per profile, one column per variable of `pkpdutils`.
    """
    return _read_wide(
        _frame(source),
        WINNONLIN_NAMES,
        {},
        WINNONLIN_PERCENT,
        WINNONLIN_UNMAPPED,
        "WinNonlin",
    )


def to_noncompart(result: ParameterResult) -> pd.DataFrame:
    """Lay a result out as the table of `NonCompart::tblNCA`.

    One row per sample with the sample dimensions of the result, then every
    parameter the result carries under its NonCompart column
    (`NONCOMPART_NAMES`; the mean residence time after
    `NONCOMPART_NAMES_BY_ROUTE`), the extrapolated fractions as percentages.

    Args:
        result: the result, e.g. of `pkpdutils.nca.nca`

    Returns:
        The table, one row per sample.

    Raises:
        ValueError: if the samples of the result were given different routes,
            which name the mean residence time differently.
    """
    names = _names_for(
        NONCOMPART_NAMES, NONCOMPART_NAMES_BY_ROUTE, _result_route(result)
    )
    return _wide(result, names, NONCOMPART_PERCENT)


def write_noncompart(result: ParameterResult, path: str | Path) -> pd.DataFrame:
    """Write the NonCompart table of a result as a csv file.

    Args:
        result: the result
        path: the file to write

    Returns:
        The table that was written, `to_noncompart`.
    """
    table = to_noncompart(result)
    table.to_csv(path, index=False)
    return table


def read_noncompart(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read a table of `NonCompart::tblNCA`.

    The grouping columns before the first parameter become the index, every
    column with a `pkpdutils` name becomes that variable and the percentages
    become fractions. `b0`, `CORRXY` and any column without a name are
    dropped.

    Args:
        source: the table or the path of its csv file

    Returns:
        One row per profile, one column per variable of `pkpdutils`.
    """
    return _read_wide(
        _frame(source),
        NONCOMPART_NAMES,
        NONCOMPART_NAMES_BY_ROUTE,
        NONCOMPART_PERCENT,
        NONCOMPART_UNMAPPED,
        "NonCompart",
    )


def to_pknca_results(result: ParameterResult) -> pd.DataFrame:
    """Lay a result out as the long table of PKNCA, `as.data.frame(pk.nca(...))`.

    One row per sample and parameter: the sample dimensions of the result,
    then `start` and `end` of the interval the parameter belongs to,
    `PPTESTCD` the name of the parameter in PKNCA (`PKNCA_NAMES`,
    `PKNCA_NAMES_BY_ROUTE`), `PPORRES` its value and `exclude`, which is empty
    as PKNCA leaves it for a parameter it computed. The interval is `0` to
    `inf` for a single dose, `0` to `tau` for a sample analysed over its
    dosing intervals, in the times of the analysis (relative to the dose).
    A parameter which is `NaN` gets no row, and the extrapolated fractions are
    percentages.

    A clearance over the bioavailability is `cl.obs` like a clearance: PKNCA
    does not tell them apart.

    Args:
        result: the result, e.g. of `pkpdutils.nca.nca`

    Returns:
        The long table.
    """
    ds = result.ds
    dims = result.sample_dims
    shape = tuple(int(ds.sizes[d]) for d in dims)
    labels = {
        dim: (ds[dim].to_numpy() if dim in ds.coords else np.arange(ds.sizes[dim]))
        for dim in dims
    }
    route_values = (
        np.asarray(ds["route"].transpose(*dims).to_numpy()) if "route" in ds else None
    )
    attrs_route = ds.attrs.get("route")
    tau = (
        np.asarray(ds["tau"].transpose(*dims).to_numpy(), dtype=np.float64)
        if "tau" in ds
        else None
    )
    values = {
        name: np.asarray(ds[name].transpose(*dims).to_numpy(), dtype=np.float64)
        for name in result.parameters
    }
    rows: list[dict[str, Any]] = []
    for index in np.ndindex(*shape) if shape else [()]:
        raw_route = route_values[index] if route_values is not None else attrs_route
        route = None if raw_route is None else Route(str(raw_route))
        names = _names_for(PKNCA_NAMES, PKNCA_NAMES_BY_ROUTE, route)
        end = (
            float(tau[index]) if tau is not None and np.isfinite(tau[index]) else np.inf
        )
        group = {dim: labels[dim][index[i]] for i, dim in enumerate(dims)}
        for variable, code in names.items():
            if variable not in values:
                continue
            value = float(values[variable][index])
            if np.isnan(value):
                continue
            if code in PKNCA_PERCENT:
                value *= 100.0
            rows.append(
                {
                    **group,
                    "start": 0.0,
                    "end": end,
                    "PPTESTCD": code,
                    "PPORRES": value,
                    "exclude": pd.NA,
                }
            )
    return pd.DataFrame(rows, columns=[*dims, *PKNCA_COLUMNS])


def write_pknca_results(result: ParameterResult, path: str | Path) -> pd.DataFrame:
    """Write the long PKNCA table of a result as a csv file.

    Args:
        result: the result
        path: the file to write

    Returns:
        The table that was written, `to_pknca_results`.
    """
    table = to_pknca_results(result)
    table.to_csv(path, index=False)
    return table


def _pknca_route(codes: set[str]) -> Route | None:
    """The route a set of PKNCA parameters implies, `None` when it implies none."""
    if codes & {"c0"}:
        return Route.IV_BOLUS
    if any(code.startswith(("mrt.iv.", "vss.iv.", "thalf.eff.iv.")) for code in codes):
        return Route.IV_INFUSION
    if codes & {"tlag", "mrt.obs", "mrt.last", "mrt.pred", "thalf.eff.obs"}:
        return Route.ORAL
    return None


def read_pknca_results(
    source: str | Path | pd.DataFrame,
    *,
    route: Route | str | None = None,
) -> pd.DataFrame:
    """Read the long table of PKNCA into one row per profile and interval.

    The grouping columns before `start` and `end` and the two bounds of the
    interval become the index, every `PPTESTCD` with a `pkpdutils` name
    becomes that variable and the percentages become fractions. A row PKNCA
    excluded (a non-empty `exclude`) is dropped, and so is a parameter
    without a name. The same PKNCA parameter is a clearance or a clearance
    over the bioavailability depending on the route (`cl.obs` is `cl` or
    `cl_f`), so the route decides the variable; without `route` it is
    recognized from the parameters PKNCA reports for it (`c0` after a bolus,
    `mrt.iv.obs` after an intravenous dose, `tlag` and `mrt.obs` after an
    extravascular one).

    Args:
        source: the table or the path of its csv file

    Keyword Args:
        route: route of the dose, recognized from the parameters by default

    Returns:
        One row per profile and interval, one column per variable.

    Raises:
        ValueError: if the table lacks the columns of PKNCA or the route can
            not be recognized from it.
    """
    frame = _frame(source)
    missing = [c for c in ("start", "end", "PPTESTCD", "PPORRES") if c not in frame]
    if missing:
        raise ValueError(f"not a PKNCA result table, the columns {missing} are missing")
    if "exclude" in frame:
        kept = frame["exclude"].isna() | (
            frame["exclude"].astype(str).str.strip() == ""
        )
        frame = frame[kept]
    codes = set(frame["PPTESTCD"].astype(str))
    resolved = Route(route) if route is not None else _pknca_route(codes)
    if resolved is None:
        raise ValueError(
            "the route can not be recognized from the PKNCA parameters, give `route`"
        )
    # `cl.obs` is `cl` after an intravenous dose and `cl_f` otherwise
    other_route = _EXTRAVASCULAR_ONLY if resolved.is_iv else _INTRAVENOUS_ONLY
    reverse = {
        code: variable
        for variable, code in _names_for(
            PKNCA_NAMES, PKNCA_NAMES_BY_ROUTE, resolved
        ).items()
        if variable not in other_route
    }
    groups = [str(c) for c in frame.columns[: list(frame.columns).index("start")]]
    unnamed = sorted(codes - set(reverse))
    if unnamed:
        logger.info("PKNCA parameters without a pkpdutils name dropped: %s", unnamed)
    named = frame[frame["PPTESTCD"].isin(list(reverse))].copy()
    named["value"] = named["PPORRES"].astype(np.float64)
    percent = named["PPTESTCD"].isin(list(PKNCA_PERCENT))
    named.loc[percent, "value"] = named.loc[percent, "value"] / 100.0
    named["variable"] = named["PPTESTCD"].map(reverse)
    wide = named.pivot_table(
        index=[*groups, "start", "end"],
        columns="variable",
        values="value",
        aggfunc="first",
        sort=False,
    )
    wide.columns.name = None
    return wide
