# Data formats

Pharmacokinetic data is exchanged as tables, not as `Timecourse` objects, and the field uses a handful of table layouts: the event record of NONMEM and Monolix, the two tables of PKNCA, and the CDISC ADaM ADNCA dataset. `pkpdutils.io` reads every one of them into a `Timecourses` batch and writes the event record format back, so a study read from any of the three formats runs through the same non-compartmental analysis.

## Concepts

Three layouts, three readers, one batch:

```mermaid
flowchart LR
  EV["NONMEM / Monolix<br/>event records<br/>ID, TIME, DV, AMT, EVID, MDV"] --> RE["read_events<br/>ADDL/II expansion<br/>SS history<br/>RATE/TINF duration"]
  PK["PKNCA<br/>concentration table<br/>+ dose table"] --> RP["read_pknca<br/>joined on the subject"]
  AD["CDISC ADaM ADNCA<br/>USUBJID, AVAL, AFRLT, ARRLT"] --> RA["read_adnca<br/>dose times = AFRLT - ARRLT<br/>DTYPE == COPY dropped"]
  RE --> BB["one sample dimension, one route,<br/>a Dosing per subject,<br/>constant columns -> coordinates"]
  RP --> BB
  RA --> BB
  BB --> T["Timecourses"]
  T -->|"write_events / to_events"| EV
  T --> N["nca"]
```

**Event records.** NONMEM and Monolix exchange data as one row per event of one subject: a row is a dose or an observation, and the columns describe what happened at that time - `AMT`/`DV` the amount or the value, `EVID`/`MDV` which kind of row it is. Repeated dosing does not need one row per dose: `ADDL`/`II` expand one dose record into several at a fixed interval, and `SS` marks a dose as already at steady state, standing for a dosing history rather than a single administration. With an `EVID` column the two rules are read independently, so a row may be a dose (`EVID 1`) and carry an observed value; without one a row with `AMT > 0` is a dose and nothing else, as NM-TRAN reads such a table, and a `DV` on it is ignored with a warning. A table which records a dose and a sample in one row therefore needs an `EVID` column.

**The two table layout.** PKNCA keeps the concentrations and the doses in separate tables, joined by the subject (and, for a multi-analyte or multi-period study, further grouping columns). This is closer to how data usually arrives from a bioanalytical lab and a dosing log than the single event table, and `pkpdutils.io.read_pknca` reads both without merging them first.

**The ADaM layout.** The CDISC ADaM ADNCA (ADPC) dataset is one row per concentration record of one analyte, already reshaped for a non-compartmental analysis: the time is given twice, once since the first dose of the subject (`AFRLT`) and once since the reference (most recent) dose (`ARRLT`), so the dose times of a subject are recovered as the distinct values of `AFRLT - ARRLT`. A predose sample can appear twice, once for the current interval and once, duplicated, for the previous one (`DTYPE == "COPY"`); `pkpdutils` drops the duplicate.

**The column keywords.** Every column name a reader takes is a keyword ending in `_col` (`id_col`, `time_col`, `dv_col`, `amt_col`, `subject_col`, `conc_col`, `dose_col`, ...), the default being the name the format uses; the columns kept as coordinates are named by `covariates` on all three readers.

**What every reader does.** A reader returns a `Timecourses` batch with one sample dimension (`individual` by default), the observation times exactly as given (readers never shift the time axis to the dose), one route for the whole batch, and the dosing protocol of every subject as a `Dosing`. A subject which is not a curve is an error naming the subject, a `ValueError` and never a pydantic dump: fewer than two observations, a time which is not a number, duplicate sampling times, or dose records which are not a protocol (an infusion without a duration, a dose time which is not a number). Columns are looked up case-insensitively, so `TIME` and `time` are the same column; a column that is not in the table is treated as absent rather than as an error, except the columns a reader cannot do without. Extra columns which are constant within every subject - a covariate such as body weight, sex, or a dose group - become coordinates along the sample dimension and travel with every later result.

## NONMEM / Monolix event records

`read_events`/`Timecourses.from_events` and the inverse `write_events`/`to_events`, the one row per event format[^bauer].

| column | role | notes |
| --- | --- | --- |
| `ID` | subject | one batch is one route: a table of several routes is filtered by the caller and read into separate batches |
| `TIME` | time of the row | in `time_unit` |
| `DV` (Monolix `OBSERVATION`) | observed value | in `unit`; a dose row leaves it empty |
| `AMT` (Monolix `AMOUNT`) | dose amount | in `dose_unit`; `0` or empty for an observation |
| `EVID` | event kind | `1` dose, `0` observation, `2`/`3` dropped (logged), `4` (reset and dose) raises: the reader would merge the periods it separates into one protocol; without this column `AMT > 0` is a dose only and a value in `DV` on such a row is ignored (logged) |
| `MDV` | missing dependent value | `1` marks the value of the row as missing even with a value in `DV`; the row keeps its sampling time and is read as `NaN` (`keep_missing=False` drops it instead) |
| `SD`/`SE`/`N` | uncertainty of a group curve | written by `write_events` when the batch carries them and read back into `sd`, `se` and `n`; `N` is constant within a subject |
| `RATE` | infusion rate | duration `= AMT / RATE` for `RATE > 0`; `RATE -1`/`-2` (a modelled rate) is not data and raises |
| `TINF` (Monolix `INFUSION DURATION`) | infusion duration | wins over `RATE` when positive |
| `ADDL` (Monolix `ADDITIONAL DOSES`) | additional doses | expands into `ADDL` further doses at `II` |
| `II` (Monolix `INTERDOSE INTERVAL`) | interdose interval | required with a positive `ADDL` or `SS == 1` |
| `SS` (Monolix `STEADY STATE`) | steady state dose | `1` stands for `ss_doses` (default 5) preceding doses at `II` and sets `attrs["steady_state_marker"]` on the batch; `0` is a plain dose, any other value raises |
| `CMT`/`ADM` | compartment | not interpreted, not a route |

```python
import io

import pandas as pd

from pkpdutils import Route, Timecourses

# a small event table: two subjects, one dose record each which stands for two
# doses twelve hours apart (ADDL/II), and the body weight as a covariate
table = """ID,TIME,DV,AMT,EVID,MDV,ADDL,II,WT
1,0,.,100,1,1,1,12,70
1,1,5.1,.,0,0,.,.,70
1,4,3.9,.,0,0,.,.,70
1,12,1.2,.,0,0,.,.,70
1,13,5.4,.,0,0,.,.,70
1,24,1.4,.,0,0,.,.,70
2,0,.,100,1,1,1,12,85
2,1,4.4,.,0,0,.,.,85
2,4,3.4,.,0,0,.,.,85
2,12,1.0,.,0,0,.,.,85
2,13,4.7,.,0,0,.,.,85
2,24,1.1,.,0,0,.,.,85
"""
events = pd.read_csv(io.StringIO(table))  # a file: pd.read_csv("study_events.csv")
batch = Timecourses.from_events(
    events,
    time_unit="hr",
    unit="ng/ml",
    dose_unit="mg",
    route=Route.ORAL,
    substance="drug",
    covariates=["WT"],
)
print(batch.sample_dims, batch.sample_shape, batch.ds["WT"].values)
print(batch.dosing_of(individual=1))
print(batch.to_events().head(4).to_string(index=False))  # the inverse
```

```text
('individual',) (2,) [70 85]
amounts=array([100., 100.]) times=array([ 0., 12.]) durations=None unit='mg' route=<Route.ORAL: 'oral'>
 ID  TIME  DV   AMT  EVID  MDV  RATE  WT
  1   0.0 NaN 100.0     1    1   0.0  70
  1   1.0 5.1   0.0     0    0   0.0  70
  1   4.0 3.9   0.0     0    0   0.0  70
  1  12.0 NaN 100.0     1    1   0.0  70
```

The one dose record of the table became a protocol of two doses, the weight became a coordinate of the batch, and `to_events` writes the expanded protocol back as one row per dose. The batch is the input of `nca`, which analyses both dosing intervals, see [Non-compartmental analysis](nca.md).

`examples/formats.py` writes a twice daily batch as event records, reads it back and analyses every dosing interval of the round trip:

![The trough concentration of every dosing interval of four subjects](images/formats.png)

## PKNCA tables

`read_pknca`/`Timecourses.from_pknca`: the concentration table and the dose table of the R package `PKNCA`[^pknca], joined on the subject.

| keyword | default column | table | role | notes |
| --- | --- | --- | --- | --- |
| `subject_col` | `subject` | both | subject | joins the two tables |
| `time_col` | `time` | concentrations | observation time | in `time_unit` |
| `conc_col` | `conc` | concentrations | observed value | `0` codes below the limit of quantification, `NA` codes missing, both kept as given |
| `dose_time_col` | `time` | doses | dose time | in `time_unit`, `0` when the column is absent |
| `dose_col` | `dose` | doses | dose amount | in `dose_unit` |
| `duration_col` | none | doses | infusion duration | optional, `None` without infusions |
| `covariates` | none | either | covariate columns | constant per subject, become coordinates along the sample dimension |

```python
import io

import pandas as pd

from pkpdutils import Route, Timecourses

conc = pd.read_csv(
    io.StringIO(
        """subject,treatment,time,conc
1,A,0,0
1,A,1,4.2
1,A,4,3.0
1,A,12,1.0
2,B,0,0
2,B,1,4.0
2,B,13,5.5
2,B,24,2.0
"""
    )
)
doses = pd.read_csv(
    io.StringIO(
        """subject,treatment,time,dose
1,A,0,100
2,B,0,100
2,B,12,100
"""
    )
)
batch = Timecourses.from_pknca(
    conc,
    doses,
    time_unit="hr",
    unit="ng/ml",
    dose_unit="mg",
    route=Route.ORAL,
    covariates=["treatment"],
)
print(batch.ds["treatment"].values, batch.n_doses)  # ['A' 'B'] [1 2]
```

The two tables are the fixtures `tests/data/formats/pknca_conc.csv` and `pknca_dose.csv` of the repository, row for row; the second subject has two dose records and therefore a protocol of two doses, the first one a single dose.

## CDISC ADaM ADNCA

`read_adnca`/`Timecourses.from_adnca`: the analysis dataset of a non-compartmental analysis[^cdisc-adnca].

| keyword | default column | role | notes |
| --- | --- | --- | --- |
| `subject_col` | `USUBJID` | subject | |
| `param_col` | `PARAMCD` | analyte code | rows are filtered to `analyte`, the single analyte of the dataset by default |
| `value_col`, `value_unit_col` | `AVAL`, `AVALU` | value, its unit | `unit` given by the caller wins over `AVALU` |
| `time_first_col` | `AFRLT` | time since the first dose | the observation time |
| `time_ref_col` | `ARRLT` | time since the reference dose | `AFRLT - ARRLT` gives the dose times of a subject, one per distinct value |
| `dose_col`, `dose_unit_col` | `DOSEA`, `DOSEU` | dose amount, its unit | the amount of the dose at the recovered time; disagreeing amounts at the same dose time raise |
| `route_col` | `ROUTE` | route | `ORAL`/`PO`, `IV`/`INTRAVENOUS`/`IV BOLUS`, `IV INFUSION`; the caller's `route=` wins |
| none | none | infusion duration | not in the dataset: an infusion protocol cannot be read and `Route.IV_INFUSION` raises, such a study is read from the event records or the PKNCA tables |
| `dtype_col` | `DTYPE` | derivation type | `COPY` rows (the predose record duplicated into the previous interval) are dropped |
| `lloq_col` | `ALLOQ` | lower limit of quantification | kept as the coordinate `lloq` along the sample dimension, informational: the analysis reads the scalar `NCAOptions.lloq` and never this coordinate |
| `covariates` | none | covariate columns | constant per subject, become coordinates along the sample dimension |

```python
import io

import pandas as pd

from pkpdutils import Timecourses

# one row per concentration record; the pre-dose sample of the second interval
# is duplicated into the first one (DTYPE == COPY) and is dropped
adnca = pd.read_csv(
    io.StringIO(
        """USUBJID,PARAMCD,AVAL,AVALU,AFRLT,ARRLT,DOSEA,DOSEU,ROUTE,DTYPE,ALLOQ
S1,XAN,0.05,ng/mL,0.5,0.5,100,mg,ORAL,,0.1
S1,XAN,4.2,ng/mL,1,1,100,mg,ORAL,,0.1
S1,XAN,1.0,ng/mL,12,12,100,mg,ORAL,,0.1
S2,XAN,4.0,ng/mL,1,1,100,mg,ORAL,,0.1
S2,XAN,1.1,ng/mL,12,12,100,mg,ORAL,,0.1
S2,XAN,1.1,ng/mL,12,0,100,mg,ORAL,COPY,0.1
S2,XAN,5.5,ng/mL,13,1,100,mg,ORAL,,0.1
S2,XAN,2.0,ng/mL,24,12,100,mg,ORAL,,0.1
"""
    )
)
batch = Timecourses.from_adnca(adnca, analyte="XAN")
for label in batch.ds["individual"].to_numpy():
    print(label, batch.dosing_of(individual=str(label)).times)
```

```text
S1 [0.]
S2 [ 0. 12.]
```

The dose times were recovered from `AFRLT - ARRLT`: the first subject was dosed once, the second one twice. The same extract is the fixture `tests/data/formats/adnca.csv` of the repository and `examples/formats.py` reads it.

## What is not read

Only the event format round trips: `write_events`/`Timecourses.to_events` writes it back, and there is no `write_pknca` and no `write_adnca`, so a batch read from the two PKNCA tables or from an ADNCA dataset is written as event records (or as the long frame of `to_dataframe`). An `lloq` coordinate read from a table is informational as well: the analysis reads the scalar `NCAOptions.lloq` and never the coordinate.

Compartment columns (`CMT`, `ADM`) are not interpreted: a study with several compartments or several routes is filtered by the caller before reading, since a batch has one route. A record which resets the subject and doses (`EVID 4`) and a steady state code other than `SS 0`/`SS 1` describe a dosing history the protocol of a subject cannot hold, and raise rather than being read as an ordinary dose; the caller splits the periods into separate tables. Infusion protocols are read from the event records (`TINF`/`RATE`) and from the PKNCA dose table (`duration_col`), not from an ADNCA dataset, which carries no duration. Modelled rates (`RATE -1`, `RATE -2`) are not data and raise: the infusion duration is given directly (`TINF`/duration column) or as a positive rate. The PP/ADPP parameter output domains, bioequivalence period and sequence (given as coordinates by the caller), and reading SAS/XPT files directly (the caller uses `pandas`/`pyreadstat` and passes the resulting `DataFrame`) are out of scope of the readers.

## References

[^bauer]: Bauer RJ. NONMEM Tutorial Part I: Description of Commands and Options, with Simple Examples of Population Analysis. *CPT: Pharmacometrics & Systems Pharmacology.* 2019;8(8):525-537. See [References](references.md#data-formats).
[^pknca]: Denney W, Duvvuri S, Buckeridge C. Simple, automatic noncompartmental analysis: the PKNCA R package. *Journal of Pharmacokinetics and Pharmacodynamics.* 2015;42:S65. See [References](references.md#data-formats).
[^cdisc-adnca]: CDISC. ADaM Implementation Guide for Non-compartmental Analysis Input Data (ADNCA). 2021. See [References](references.md#data-formats).

The example is `examples/formats.py`, the reference of the module is in [API: io](api/io.md).
