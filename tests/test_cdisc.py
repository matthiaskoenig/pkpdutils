"""The CDISC map: `PKPARMCD` codes, `PKUNIT` spellings and the `PP` domain."""

import logging

import numpy as np
import pandas as pd
import pytest

from pkpdutils import Dose, Dosing, NCAOptions, Route, Timecourse, Timecourses, nca
from pkpdutils.cdisc import (
    PKPARMCD,
    PKPARMCD_BY_ROUTE,
    PKPARMCD_NAMES,
    PKUNIT,
    pkparmcd,
    pkunit,
    to_pp,
    write_pp,
)

TIME = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
ORAL = 10.0 * (np.exp(-0.2 * TIME) - np.exp(-1.5 * TIME))
SUBJECTS = ["S1", "S2"]


def oral_batch() -> Timecourses:
    """Two oral subjects of the same curve, scaled."""
    curves = [
        Timecourse(
            time=TIME,
            value=(1.0 + 0.1 * index) * ORAL,
            time_unit="hr",
            unit="ng/ml",
            substance="drug",
            dose=Dose(amount=100, unit="mg", route=Route.ORAL),
            label=subject,
        )
        for index, subject in enumerate(SUBJECTS)
    ]
    return Timecourses.from_timecourses(curves)


def steady_state_batch() -> Timecourses:
    """One subject of three doses twelve hours apart."""
    times = np.concatenate([TIME, 12.0 + TIME, 24.0 + TIME])
    values = np.tile(ORAL, 3) + 0.5
    curve = Timecourse(
        time=times,
        value=values,
        time_unit="hr",
        unit="ng/ml",
        substance="drug",
        dosing=Dosing.regimen(
            Dose(amount=100, unit="mg", route=Route.ORAL), interval=12.0, n_doses=3
        ),
        label="S1",
    )
    return Timecourses.from_timecourses([curve])


def test_the_codes_come_from_the_terminology_file() -> None:
    """Every code of the crosswalk carries the CDISC name of the code."""
    assert PKPARMCD["auc_inf_obs"] == "AUCIFO"
    assert PKPARMCD["auc_last"] == "AUCLST"
    assert PKPARMCD["lambda_z_span"] == "LAMZSPN"
    assert PKPARMCD["cl_ss"] == "CLTAU"
    assert PKPARMCD["accumulation_ratio"] == "AILAMZ"
    assert PKPARMCD["accumulation_ratio_obs"] == "ARAUC"
    assert PKPARMCD["cmax_ss"] == "CMAX"
    assert PKPARMCD["cmin_ss"] == "CMIN"
    assert PKPARMCD_NAMES["AUCIFO"] == "AUC Infinity Obs"
    assert PKPARMCD_NAMES["LAMZSPN"] == "Lambda z Span"
    # the codes the survey found not to exist are not in the map
    assert not {"CLSTP", "AEAMT", "FE", "ACCIND", "PTROUGH", "CMAXSS", "CMINSS"} & set(
        PKPARMCD.values()
    )
    assert "clast_pred" not in PKPARMCD
    assert pkparmcd("clast_pred") is None


def test_the_mean_residence_time_follows_the_route() -> None:
    """`mrt` has one code per route and none without one."""
    assert PKPARMCD_BY_ROUTE["mrt"] == {
        Route.ORAL: "MRTEVIFO",
        Route.IV_BOLUS: "MRTIBIFO",
        Route.IV_INFUSION: "MRTICIFO",
    }
    assert pkparmcd("mrt", Route.ORAL) == "MRTEVIFO"
    assert pkparmcd("mrt", "iv_bolus") == "MRTIBIFO"
    assert pkparmcd("mrt") is None
    assert "mrt" not in PKPARMCD


@pytest.mark.parametrize(
    ("unit", "spelled"),
    [
        ("hour * nanogram / milliliter", "h*ng/mL"),
        ("h*ng/mL", "h*ng/mL"),
        ("nanogram / milliliter", "ng/mL"),
        ("liter / hour", "L/h"),
        ("milliliter / minute", "mL/min"),
        ("1 / hour", "/h"),
        ("hour", "h"),
        ("hour ** 2 * nanogram / milliliter", "h2*ng/mL"),
        ("hour * nanogram / milligram / milliliter", "h*ng/mL/mg"),
        ("dimensionless", ""),
        ("percent", "%"),
        ("liter", "L"),
        # a unit the table does not know, written in the same symbols
        ("milligram / deciliter", "mg/dL"),
    ],
)
def test_the_pkunit_spellings(unit: str, spelled: str) -> None:
    """The units of a result are written as CDISC spells them."""
    assert pkunit(unit) == spelled


def test_to_pp_writes_one_row_per_subject_and_parameter() -> None:
    """The layout of the `PP` domain, with the codes, the names and the units."""
    result = nca(oral_batch())
    pp = to_pp(result, subject_dim="individual")
    assert list(pp.columns) == [
        "DOMAIN",
        "USUBJID",
        "PPSEQ",
        "PPTESTCD",
        "PPTEST",
        "PPCAT",
        "PPSCAT",
        "PPORRES",
        "PPORRESU",
        "PPSTRESN",
        "PPSTRESU",
        "PPSPEC",
        "PPRFTDTC",
    ]
    assert set(pp["USUBJID"]) == set(SUBJECTS)
    assert set(pp["DOMAIN"]) == {"PP"}
    assert set(pp["PPCAT"]) == {"drug"}
    assert set(pp["PPSPEC"]) == {"PLASMA"}
    assert set(pp["PPRFTDTC"]) == {""}
    first = pp.loc[pp["USUBJID"] == "S1"]
    assert list(first["PPSEQ"]) == list(range(1, len(first) + 1))
    auc = first.loc[first["PPTESTCD"] == "AUCIFO"].iloc[0]
    assert auc["PPTEST"] == "AUC Infinity Obs"
    assert auc["PPORRESU"] == "h*ng/mL"
    assert auc["PPSTRESU"] == "h*ng/mL"
    assert auc["PPSCAT"] == "SINGLE DOSE"
    assert auc["PPSTRESN"] == pytest.approx(float(result.ds["auc_inf_obs"][0]))
    # the mean residence time took the code of the route of the batch
    assert "MRTEVIFO" in set(first["PPTESTCD"])
    # the fraction extrapolated is a percentage in CDISC
    extrapolated = first.loc[first["PPTESTCD"] == "AUCPEO"].iloc[0]
    assert extrapolated["PPORRESU"] == "%"
    assert extrapolated["PPSTRESN"] == pytest.approx(
        100.0 * float(result.ds["auc_extrap_fraction"][0])
    )


def test_to_pp_marks_the_steady_state_parameters() -> None:
    """Every parameter of a multiple dose sample carries `PPSCAT = STEADY STATE`."""
    result = nca(steady_state_batch())
    pp = to_pp(result, subject_dim="individual")
    rows = {(row["PPTESTCD"], row["PPSCAT"]): row for _, row in pp.iterrows()}
    assert ("CMAX", "STEADY STATE") in rows
    assert ("AUCTAU", "STEADY STATE") in rows
    assert ("CTROUGH", "STEADY STATE") in rows
    assert rows[("AUCTAU", "STEADY STATE")]["PPTEST"] == "AUC Over Dosing Interval"
    # the point parameters of such a sample are computed from the last dose on
    assert ("TMAX", "STEADY STATE") in rows
    assert set(pp["PPSCAT"]) == {"STEADY STATE"}
    # a single dose sample keeps SINGLE DOSE, the steady state variables apart
    single = to_pp(nca(oral_batch()), subject_dim="individual")
    assert set(single["PPSCAT"]) == {"SINGLE DOSE"}


def test_to_pp_marks_the_steady_state_samples_of_a_mixed_batch() -> None:
    """`PPSCAT` follows the sample: a batch of a single and a multiple dose subject."""
    from pkpdutils import Timecourses as Batch

    batch = Batch.from_timecourses(
        [
            steady_state_batch().sel(individual="S1"),
            oral_batch().sel(individual="S1"),
        ],
        labels=["multiple", "single"],
    )
    pp = to_pp(nca(batch), subject_dim="individual")
    scat = {subject: set(rows["PPSCAT"]) for subject, rows in pp.groupby("USUBJID")}
    assert scat["multiple"] == {"STEADY STATE"}
    assert scat["single"] == {"SINGLE DOSE"}


def test_to_pp_warns_about_a_variable_without_a_code(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A parameter the terminology has no code for is left out and named."""
    result = nca(oral_batch())
    with caplog.at_level(logging.WARNING, logger="pkpdutils.cdisc"):
        pp = to_pp(result, subject_dim="individual")
    assert "clast_pred" in caplog.text
    assert "no PKPARMCD code" in caplog.text
    assert "CLSTP" not in set(pp["PPTESTCD"])


def test_to_pp_takes_the_study_the_identifiers_and_the_adam_spec() -> None:
    """`STUDYID`, `USUBJID` and the analysis variables of an `ADPP` dataset."""
    result = nca(oral_batch())
    pp = to_pp(
        result,
        subject_dim="individual",
        usubjid={"S1": "STUDY-001", "S2": "STUDY-002"},
        spec="ADaM",
        studyid="STUDY",
    )
    assert set(pp["USUBJID"]) == {"STUDY-001", "STUDY-002"}
    assert set(pp["STUDYID"]) == {"STUDY"}
    assert "DOMAIN" not in pp.columns
    assert list(pp["PARAMCD"]) == list(pp["PPTESTCD"])
    assert list(pp["PARAM"]) == list(pp["PPTEST"])
    np.testing.assert_allclose(pp["AVAL"], pp["PPSTRESN"])
    assert list(pp["AVALU"]) == list(pp["PPORRESU"])


def test_to_pp_names_the_analyte_of_every_row() -> None:
    """A batch of several analytes writes the substance of every row in `PPCAT`."""
    from tests.nca.test_analytes import two_analyte_batch

    result = nca(two_analyte_batch())
    pp = to_pp(result, subject_dim="individual")
    assert set(pp["PPCAT"]) == {"parent", "metabolite"}
    assert set(pp.loc[pp["PPCAT"] == "parent", "USUBJID"]) == {"S1", "S2", "S3"}


def test_to_pp_names_a_partial_area_interval() -> None:
    """A named partial area is `AUCINT` with its interval in `PPTEST`."""
    options = NCAOptions(partial_aucs={"auc_0_8": (0.0, 8.0)})
    result = nca(oral_batch(), options=options)
    pp = to_pp(result, subject_dim="individual")
    row = pp.loc[pp["PPTESTCD"] == "AUCINT"].iloc[0]
    assert row["PPTEST"] == "AUC from 0 to 8 h"
    assert row["PPSTRESN"] == pytest.approx(float(result.ds["auc_0_8"][0]))


def test_to_pp_needs_the_subject_dimension() -> None:
    """A dimension which is not a sample dimension of the result raises."""
    result = nca(oral_batch())
    with pytest.raises(ValueError, match="not a sample dimension"):
        to_pp(result, subject_dim="subject")


def test_write_pp_writes_the_domain(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The csv holds the rows of `to_pp`."""
    result = nca(oral_batch())
    path = tmp_path / "pp.csv"
    frame = write_pp(result, path, subject_dim="individual")
    written = pd.read_csv(path)
    assert len(written) == len(frame)
    assert list(written.columns) == list(frame.columns)
    assert set(written["PPTESTCD"]) == set(frame["PPTESTCD"])


def test_every_pkunit_spelling_is_a_submission_value_of_the_codelist() -> None:
    """The spellings of `PKUNIT` are values of `C85494`, not inventions."""
    import csv
    from pathlib import Path

    path = Path(__file__).parent / "data" / "cdisc" / "pkunit_c85494.csv"
    rows = csv.DictReader(
        line for line in path.read_text().splitlines() if not line.startswith("#")
    )
    values = {row["value"] for row in rows}
    assert len(values) == 608
    assert not set(PKUNIT.values()) - values


def test_a_unit_the_terminology_does_not_spell_is_named_in_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`to_pp` says which units it wrote itself and points at `to_units`."""
    # a concentration per litre: CDISC spells mass concentrations per millilitre
    result = nca(oral_batch()).to_units({"cmax": "mg/L"})
    with caplog.at_level(logging.WARNING, logger="pkpdutils.cdisc"):
        pp = to_pp(result, subject_dim="individual")
    assert "milligram / liter" in caplog.text
    assert "to_units" in caplog.text
    assert pp.loc[pp["PPTESTCD"] == "CMAX", "PPORRESU"].iloc[0] == "mg/L"
    # the units of the table say nothing
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pkpdutils.cdisc"):
        to_pp(nca(oral_batch()), subject_dim="individual")
    assert "no PKUNIT value" not in caplog.text


def test_to_pp_reads_a_transposed_variable_of_a_two_dimensional_result() -> None:
    """A variable stored in another dimension order still lands on its subject."""
    from tests.nca.test_analytes import two_analyte_batch

    result = nca(two_analyte_batch())
    transposed = type(result)(result.ds.transpose("individual", "analyte", ...))
    straight = to_pp(result, subject_dim="individual").sort_values(
        ["USUBJID", "PPCAT", "PPTESTCD"], ignore_index=True
    )
    other = to_pp(transposed, subject_dim="individual").sort_values(
        ["USUBJID", "PPCAT", "PPTESTCD"], ignore_index=True
    )
    np.testing.assert_allclose(straight["PPSTRESN"], other["PPSTRESN"])
    assert list(straight["USUBJID"]) == list(other["USUBJID"])
    assert list(straight["PPCAT"]) == list(other["PPCAT"])
    # the values differ between the subjects, so a wrong index would show
    assert straight["PPSTRESN"].nunique() > 5
