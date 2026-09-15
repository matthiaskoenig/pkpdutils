# Dosing protocols, multiple dosing analysis and exchange formats

Date: 2026-09-15. Status: design of a first implementation for review by the maintainer (decisions below were taken by the agent on the maintainer's request "do a first implementation without asking questions").

## Summary

A timecourse is accompanied by its dosing protocol: the vector of the doses given and the times they were given, not one dose. With the protocol the non-compartmental analysis computes the parameters of every dosing interval of a multiple dose curve (partial parameters), the steady state parameters of the last interval and the accumulation between the first and the last interval, for concentration and for effect timecourses. Timecourses of many subjects are read from the exchange formats of the field: the event record format of NONMEM and Monolix (one row per observation or dose), the two table format of PKNCA (concentrations and doses) and the CDISC ADaM ADNCA/ADPC dataset. Dosing stays optional: a timecourse without a protocol keeps working as today and the dose dependent parameters are `NaN`.

## Decisions

| topic | decision |
| --- | --- |
| dosing optional | `Timecourse.dosing` may be `None` (maintainer's choice); the NCA reports the dose dependent parameters as `NaN` without it |
| protocol model | one `Dosing` object per timecourse: arrays `amounts`, `times`, `durations`, one `unit`, one `route`; a `Dose` stays the single administration and `Dosing.single(dose)` wraps it |
| routes | one route per protocol and per batch (mixed routes are separate timecourses / batches), as today |
| `DosingRegimen` | kept as a constructor of a regular protocol (`DosingRegimen(...).dosing()`); no longer an `NCAOptions` field |
| batch layout | `Timecourses` carries `dose_amount`, `dose_time`, `dose_duration` over `(*sample_dims, "dose_index")`, padded with `NaN` for samples with fewer doses (corrected from `"dose"` during implementation: `dose` stays free as a sample dimension, e.g. the dose groups of a dose proportionality study) |
| analysis of multiple doses | the NCA derives the dosing intervals from the protocol; the result gains an `interval` dimension with the per-interval parameters, and the steady state parameters of the last complete interval without that dimension |
| reference dose of the point parameters | with more than one dose the terminal phase, `cmax`, `tmax`, `clast`, `auc_last`, the extrapolated areas, `cl`, `vz`, `mrt` are computed from the last dose on (times relative to the last dose): the steady state analysis; with one dose everything is as today |
| exchange formats | readers `Timecourses.from_events` (NONMEM/Monolix), `from_pknca` (PKNCA tables), `from_adnca` (CDISC ADaM); writer `to_events`; the existing `from_dataframe` long format stays |
| backwards compatibility | `Timecourse(dose=Dose(...))` keeps working (converted to a protocol of one dose); `Timecourse.dose` returns the first dose; `NCAOptions.regimen` is removed, `NCAOptions.tau` replaces it for a last interval whose length is not given by the protocol; `superposition(timecourse, dosing)` takes a `Dosing` or a `DosingRegimen` |

## 1. Dosing protocol (`pkpdutils.timecourse`)

`Dosing` is a frozen pydantic model:

- `amounts: np.ndarray` (1-D, non-negative), `times: np.ndarray` (1-D, same length, strictly increasing; unsorted input is sorted with the amounts and durations), `durations: np.ndarray | None` (1-D, `NaN` where not an infusion; required with positive values for `Route.IV_INFUSION`, not allowed otherwise), `unit: str` (checked with `check_dose_unit`), `route: Route`
- constructors: `Dosing.single(dose: Dose)`, `Dosing.from_doses(doses: Sequence[Dose])` (same unit and route required), `Dosing.regimen(dose: Dose, interval: float, n_doses: int)` (`dose.time + k * interval`); `DosingRegimen.dosing()` delegates to it and needs `n_doses`
- properties: `n_doses`, `doses -> list[Dose]`, `first`, `last` (`Dose`), `intervals` (`np.diff(times)`), `tau` (the common interval when all intervals are equal within `1e-9` relative, else `None`; `None` for one dose), `is_regular`, `total_amount`, `quantity` (total), `per_bodyweight`
- `shifted(offset)` returns a copy with `times - offset`

`Timecourse`:

- field `dosing: Dosing | None = None`; the constructor also accepts `dose: Dose | None` (a `model_validator(mode="before")` moves it into `dosing`; giving both raises); `dose` is a read-only property returning `dosing.first` or `None`, so `tc.dose.amount`, `tc.dose.route`, `tc.dose.time` keep working
- `relative_to_dose(which="first")` shifts by the first (`"first"`) or the last (`"last"`) dose time and shifts the protocol along
- `to_dataframe`/`from_dataframe` of one curve unchanged (a single dose); a protocol with several doses is given as `dosing=`

`Timecourses`:

- data variables `dose_amount`, `dose_time`, `dose_duration` over `(*sample_dims, "dose_index")`, `NaN` padded; `attrs["route"]` as today (corrected during implementation: the dimension is `dose_index`, not `dose`, so that `dose` stays free as a sample dimension)
- properties `dose_amount`, `dose_time`, `dose_duration` return the 2-D arrays (`(*sample_shape, n_dose)`); `n_doses` the finite count per sample; `has_dose` unchanged; `dosing_of(**indexers) -> Dosing | None`
- `from_arrays(..., dose=...)` accepts a `Dose`, a `Dosing` (same protocol for all samples) or a mapping with `amount`/`time`/`duration` arrays of shape `sample_shape` (one dose) or `(*sample_shape, n_dose)`; `from_timecourses` pads the protocols; `from_dataframe` (long format) keeps the single dose columns and gains `dose_time` rows: with several rows per sample carrying different dose times the doses form the protocol (a row per dose time; the amount may differ per row)
- `_timecourse` (used by `sel`, `isel`, `__iter__`) rebuilds the `Dosing` from the finite entries

## 2. Multiple dosing analysis (`pkpdutils.nca`)

`NCAOptions`:

- `regimen` removed. New `tau: float | None = None`: length of the last dosing interval when the protocol has one dose (a steady state curve given with its last dose only) or to override the interval of the last dose; `None` uses the protocol (`times[-1] - times[-2]` for the last interval, or the common `tau`); a single dose protocol without `tau` is a single dose analysis
- `intervals: bool = True`: compute the per-interval parameters when the protocol has more than one dose

Analysis of a row with the protocol `t_1 < ... < t_K` (times in the frame of the curve) and `tau_K` (see `tau`):

- **intervals** `[t_k, t_{k+1}]` for `k < K` and `[t_K, t_K + tau_K]` for the last; the last interval is "complete" when `t_K + tau_K` lies within the observed range (`ctrough` interpolable), else its interval parameters are `NaN` and `NCAFlag.INCOMPLETE_INTERVAL` is set
- **per-interval parameters** over the dimension `interval` (coordinate `interval` = `1..K`, coordinates `interval_start`, `interval_end`, `interval_dose` amount): for concentrations `auc_tau_k`, `cmax_k`, `tmax_k` (relative to the interval start), `cmin_k`, `ctrough_k` (interpolated at the end), `cavg_k`, `fluctuation_k`, `swing_k`, `c_pre_k` (value interpolated at the start), `n_points_k`; for effects `auec_tau_k`, `emax_k`, `temax_k`, `emin_k`, `eavg_k`, `time_above_k`. Values inside an interval are those with `t_k <= t <= t_{k+1}` plus the interpolated boundary values (the existing `insert_point`/`interpolate_at` of `auc.py`); an interval without a point inside is `NaN`. The per-interval variables live over `(*sample_dims, "interval")` and are prefixed `interval_` so that they do not clash with the single dose and steady state names: `interval_auc`, `interval_cmax`, `interval_tmax`, `interval_cmin`, `interval_ctrough`, `interval_cavg`, `interval_fluctuation`, `interval_swing`, `interval_c_pre`, `interval_n_points` (effects: `interval_auec`, `interval_emax`, `interval_temax`, `interval_emin`, `interval_eavg`, `interval_time_above`). They are point variables of the `ParameterResult` (extra dimension), excluded from `to_dataframe`; `NCAResult.intervals()` returns them as a dataframe with one row per sample and interval, with the coordinates
- **steady state parameters** of the last complete interval, without the `interval` dimension, keep today's names: `auc_tau`, `cmin_ss`, `cmax_ss`, `ctrough`, `cavg`, `fluctuation`, `swing`, `cl_ss` (`dose_K / auc_tau`), `accumulation_ratio` (predicted `1 / (1 - exp(-lambda_z tau))`), plus new `accumulation_ratio_obs` (`auc_tau` of the last over the first interval, `NaN` with one dose or an incomplete first interval; the first interval must also be complete), `n_doses`, `tau`; for effects `auec_tau`, `emin_ss`, `emax_ss`, `eavg`, `time_above_tau`
- **point parameters** (`cmax`, `tmax`, `clast`, `tlast`, `auc_last`, `auc_inf_*`, `aumc_*`, `mrt`, `lambda_z*`, `thalf`, `cl`, `vz`, `vss`, `c0`, `cmax_half`, `tmax_half`, `auc_inf_dn`, `cmax_dn`; `e0`, `emax_obs`, `temax`, `auec_last`, `auec_baseline`, `emax_baseline`, `time_above`) are computed on the values from the last dose on, with times relative to the last dose and the amount of the last dose; with one dose this is the whole curve as today (`c0` back-extrapolation for a bolus uses the last dose)
- flags: `NCAFlag.INCOMPLETE_INTERVAL` (last interval not covered by the data), `NCAFlag.NO_STEADY_STATE` is not introduced (steady state is asserted by the user's data); the existing flags apply to the post-last-dose analysis

`superposition(timecourse, dosing | regimen, options, t_end)`: superposes the single dose curve at every dose of the protocol, scaling by `amount_k / amount_single`; the returned `Timecourse` carries the protocol.

`accumulation_ratio(steady_state, single_dose)` stays (observed ratio of two results).

Chunking and the worker pool: the interval computation runs inside `compute_parameters`' chunk (one function `compute_intervals(t, c, dose_amount, dose_time, options) -> dict[str, np.ndarray]` with `(N, K)` outputs), vectorized over the rows, looping over the `K` intervals (small).

## 3. Exchange formats (`pkpdutils.io`, new module `src/pkpdutils/io.py`)

Every reader returns a `Timecourses` with one sample dimension (default `"individual"`), the times as given (the reader does not shift), one route, and the protocol per subject; the values are the observations. Every reader takes a pandas `DataFrame` (the caller reads the csv/sas file). Columns are looked up case-insensitively.

- `read_events(df, *, time_unit, unit, dose_unit, id="ID", time="TIME", dv="DV", amt="AMT", evid="EVID", mdv="MDV", rate="RATE", tinf=None, addl="ADDL", ii="II", ss="SS", cmt=None, route: Route | None = None, substance="substance") -> Timecourses` (also `Timecourses.from_events`): NONMEM/Monolix event records. A row is a dose when `EVID in (1, 4)` or, without `EVID`, when `AMT > 0`; an observation when `EVID == 0` (or `MDV == 0`) with `DV` not missing; `EVID 2, 3` rows are ignored (a warning counts them). `RATE > 0` gives `duration = AMT / RATE`, `RATE == -1/-2` and `TINF` (Monolix) are accepted (`TINF` as the duration; `RATE -1/-2` raise: modelled rates are not data). `ADDL`/`II` expand into `ADDL` extra doses at `II`; `SS == 1` marks the dose as given at steady state: the reader adds `options.ss_doses` (default 5, as Monolix) preceding doses at `II` so the protocol represents the history, and sets the attribute `attrs["steady_state_marker"] = True` on the batch. Monolix column aliases: `AMOUNT`, `OBSERVATION`, `ADM`, `TINF`, `ADDITIONAL DOSES`, `INTERDOSE INTERVAL`, `STEADY STATE`. The route is given by the caller (`route=`) since the event format has no route column (`CMT` is not a route); a batch has one route: rows of several routes raise and the caller filters the table first (`ADM`/`CMT` are not interpreted). Extra columns which are constant per subject (covariates such as `WT`, `SEX`, `DOSE` group) become coordinates along the sample dimension.
- `write_events(timecourses, *, id="ID", ...) -> DataFrame` (`Timecourses.to_events`): the inverse, `EVID`, `MDV`, `AMT`, `RATE` (amount / duration for infusions), `CMT` absent, one row per dose and per observation, sorted by subject and time, doses before observations at the same time.
- `read_pknca(conc: DataFrame, dose: DataFrame, *, time_unit, unit, dose_unit, conc_col="conc", time_col="time", dose_col="dose", dose_time_col="time", subject="subject", groups: Sequence[str] = (), route: Route | None = None, duration_col=None) -> Timecourses` (`Timecourses.from_pknca`): the two tables of PKNCA; the grouping columns besides the subject become coordinates; PKNCA codes BLQ as `0` and missing as `NA`, both are kept as given (the `lloq`/`blq` options of the NCA handle them).
- `read_adnca(df, *, time_unit="hr", unit=None, dose_unit=None, subject="USUBJID", analyte: str | None = None, param="PARAMCD", value="AVAL", value_unit="AVALU", time_first="AFRLT", time_ref="ARRLT", dose="DOSEA", dose_unit_col="DOSEU", route_col="ROUTE", dtype="DTYPE", lloq="ALLOQ") -> Timecourses` (`Timecourses.from_adnca`): the concentration records of one analyte (`PARAMCD == analyte`, or the single analyte present); the time of a record is `AFRLT`; the dose times are the distinct values of `AFRLT - ARRLT` per subject with the amount `DOSEA` of the record; records with `DTYPE == "COPY"` are dropped (duplicated pre-dose records); `ALLOQ` is stored as the coordinate `lloq` along the sample dimension for the user (the NCA option `lloq` is one number for the batch). The route comes from `ROUTE` (case-insensitive: `ORAL`/`PO` -> `ORAL`, `INTRAVENOUS BOLUS`/`IV BOLUS`/`IV` -> `IV_BOLUS`, `INTRAVENOUS INFUSION`/`IV INFUSION` -> `IV_INFUSION`) or from the `route=` argument; the units from `AVALU`/`DOSEU` of the first record or the arguments.

`docs/formats.md` documents the three formats with the column tables and the examples; `examples/formats.py` writes an event table with `to_events`, reads it back with `from_events` and runs the multiple dosing NCA. Test fixtures: `tests/data/formats/theoph_events.csv` (the Theophylline data of R `datasets::Theoph` in NONMEM layout: 12 subjects, one oral dose, weight covariate; the values are typed into the fixture from the R dataset), `tests/data/formats/multiple_dose_events.csv` (a synthetic 3 subject, 4 dose BID protocol with `ADDL`/`II`), `tests/data/formats/pknca_conc.csv` and `pknca_dose.csv` (the same Theophylline data in the PKNCA layout), `tests/data/formats/adnca.csv` (a small hand-made ADNCA extract with `USUBJID`, `PARAMCD`, `AVAL`, `AVALU`, `AFRLT`, `ARRLT`, `NFRLT`, `NRRLT`, `DOSEA`, `DOSEU`, `ROUTE`, `DTYPE`, `ALLOQ`).

## 4. Plotting

`plot_timecourse` draws the dose times as thin vertical lines (`style.dose_color`, default `"gray"`, `linestyle=":"`) when the curve has a protocol with more than one dose; `plot_nca` shows the analysed (last) interval shaded when the result has steady state parameters. `plot_intervals(result, name, **indexers)` (new, `plot/nca.py`): the per-interval parameter `name` against the interval number for one sample or all samples (one line per sample).

## 5. Documentation and examples

`docs/timecourses.md` gains the section "Dosing protocols"; `docs/nca.md` the section "Multiple dosing" (intervals, steady state, accumulation, the reference dose rule) replacing the current steady state paragraph; `docs/pd.md` the effect parameters per interval; `docs/formats.md` new (nav after Timecourses); `docs/glossary.md` the new variables; `examples/steady_state.py` rewritten on the protocol (no manual slicing), `examples/formats.py` new, `examples/nca_batch.py` unchanged.

## 6. Tests

- `tests/test_timecourse.py`, `tests/test_timecourses.py`: `Dosing` validation and constructors, `dose=` compatibility, batch padding, `from_arrays` with protocols, round trip through `sel`/`__iter__`, `relative_to_dose(which=)`
- `tests/nca/test_intervals.py`: analytic multiple dose curve (superposition of a mono-exponential): every interval's `auc` equals the closed form, the last interval's parameters equal the current steady state values, `accumulation_ratio_obs` equals `auc_tau(last)/auc_tau(first)`, the point parameters come from the last dose (`cmax`, `tmax` relative to the last dose), effect intervals on a synthetic effect curve, incomplete last interval flagged, batch with different numbers of doses per sample, chunking/workers equal serial
- `tests/nca/test_steady_state.py`: adapted to the protocol (`tau` option and the protocol path give the same numbers as the frozen expectations of the current tests)
- `tests/nca/test_reference.py` unchanged (single dose)
- `tests/test_io.py`: the fixtures round trip (`from_events` -> `to_events` -> `from_events` equal), `ADDL/II/SS` expansion, Monolix aliases, `from_pknca` equals `from_events` for the Theophylline data, `from_adnca` recovers the dose times from `AFRLT - ARRLT`, `DTYPE == "COPY"` dropped, covariates as coordinates, error cases (mixed routes, missing columns, `RATE -1`)
- `tests/plot/`: the new figure elements

## Out of scope

Compartment assignment (`CMT`), modelled rates (`RATE -1/-2`), reading SAS/XPT files (the caller uses pandas / `pyreadstat`), the PP/ADPP output domains, bioequivalence period/sequence from ADaM (given as coordinates by the caller).
