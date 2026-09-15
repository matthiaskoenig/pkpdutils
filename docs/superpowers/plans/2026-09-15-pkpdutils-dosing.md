# pkpdutils Dosing Protocols, Multiple Dosing and Exchange Formats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single `Dose` of a timecourse by a dosing protocol (`Dosing`: vectors of amounts, times, durations), carry it through the batch layout, compute the per-interval and steady state parameters of multiple dose curves (PK and PD) from the protocol, read and write the exchange formats of the field (NONMEM/Monolix event records, PKNCA tables, CDISC ADNCA), document it, and open a pull request for the maintainer's review.

**Architecture:** `Dosing` (pydantic, frozen) in `timecourse.py` holds the protocol; `Timecourse.dosing` replaces `dose` (kept as a compatibility keyword and a read-only property of the first dose). `Timecourses` stores `dose_amount`/`dose_time`/`dose_duration` over `(*sample_dims, "dose")`, NaN padded. The NCA core gets `compute_intervals` (vectorized over rows, looping over the K intervals) producing `(N, K)` arrays for the per-interval parameters; the steady state parameters are the last complete interval; the point parameters are computed from the last dose on. `NCAOptions.regimen` goes, `NCAOptions.tau` and `intervals` come. `io.py` holds the readers/writers, exposed as `Timecourses.from_events`, `to_events`, `from_pknca`, `from_adnca`. Figures: dose lines, interval shading, `plot_intervals`. Docs: `timecourses.md`, `nca.md`, `pd.md`, new `formats.md`, glossary.

**Tech Stack:** numpy, pandas, xarray, pydantic, matplotlib, pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-15-dosing-protocols-and-exchange-formats-design.md` (binding; the decisions table and sections 1-6).

## Global Constraints

- python 3.13 and 3.14; runtime dependencies unchanged
- ty `error-on-warning = true`: zero diagnostics; rule-specific `# ty: ignore[rule]` only; narrowing asserts in tests
- every module, class and function of `src/pkpdutils` has full type annotations and a google style docstring (formula + citation key where a formula exists; formula docstrings are `r"""` with single backslashes); `examples/`, `tests/`, `scripts/` exempt from `D`
- library code logs with `logging.getLogger(__name__)` and lazy `%s`, never prints, never calls `plt.show()`
- `uv run pytest -q -W error` passes with pristine output; `uv run ruff check`, `uv run ruff format --check`, `uvx ty check`, `uv run zensical build --clean --strict` clean; `uv run tox run-parallel` green before the PR
- results are `xarray.Dataset` objects with `attrs["units"]` on every variable; per-interval variables live over `(*sample_dims, "interval")` with the prefix `interval_`
- `tests/nca/test_reference.py` (the frozen pkdb_analysis 0.3.1 numbers) must keep passing unchanged: a single dose protocol analyses exactly as before
- one route per protocol and per batch; mixed routes raise `ValueError`
- text rules: never the em dash character (U+2014); commit messages carry NO `Co-Authored-By` line and end with the single line `Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z`; markdown has no hard line wraps
- work happens on the branch `dosing` off `develop`; the final pull request is opened WITHOUT auto-merge: the maintainer reviews the prototype
- `release-notes/1.0.1.md` is not touched: this is a feature for the next minor release; the version stays `1.0.1.dev0` for now (the maintainer decides the version at the release)

## Interfaces this plan consumes

- `pkpdutils.timecourse`: `Route` (`IV_BOLUS`, `IV_INFUSION`, `ORAL`; `is_intravenous`), `Dose(amount, unit, route, time, duration)` with `quantity`, `per_bodyweight`; `DosingRegimen(dose, interval, n_doses)` with `dose_times()`; `Timecourse` (pydantic frozen, `arbitrary_types_allowed`; fields `time`, `value`, `time_unit`, `unit`, `sd`, `se`, `n`, `dose`, `substance`, `label`, `tissue`; `relative_to_dose()`, `to_dataframe()`, `from_dataframe(...)`); `Timecourses` (wraps `ds`; `TIME_DIM = "time"`, `TIMES_VAR`; properties `times`, `values`, `sd`, `se`, `n`, `dose_amount`, `dose_time`, `dose_duration`, `dose_unit`, `route`, `has_dose`, `sample_dims`, `sample_shape`, `n_samples`, `n_time`, `time_unit`, `unit`, `substance`; constructors `from_timecourses`, `from_dataframe`, `from_arrays`, `from_dataset`; `_timecourse(sample, label)`, `sel`, `isel`, `__iter__`, `to_dataframe`); `_dose_of_group` helper of `from_dataframe`
- `pkpdutils.units`: `check_dose_unit`, `is_per_bodyweight`, `parse_unit`, `Q_`
- `pkpdutils.nca.options`: `NCAOptions` (fields incl. `kind`, `auc_method`, `terminal`, `lloq`, `blq`, `regimen`, `effect_threshold`, `n_workers`, `chunk_rows`), `Kind`, `AUCMethod`, `TerminalMethod`, `TerminalPhase`, `NCAFlag` (IntFlag), `decode_flags`
- `pkpdutils.nca.nca`: `PARAMETER_UNITS`/`unit_expression`, `_effect_parameters(tp, cp, n_valid, options)`, `compute_parameters(t, c, *, dose_amount, dose_time, dose_duration, route, options) -> dict[str, np.ndarray]` (rows `(N, n)`, times shifted by `dose_time` inside), `_compute_chunk(args)` (dispatches to `compute_steady_state` when `options.regimen`), `run_rows(...)` (chunks + pool), `nca(timecourses, options)`, `nca_single`, `_to_result(values, timecourses, shape)` (units via `parameter_unit`, `sample_coordinates`, `check_coordinate_collision`), `partial_auc`
- `pkpdutils.nca.auc`: `pack_valid(t, c) -> (tp, cp, n_valid)`, `insert_point(tp, cp, n_valid, t_new, c_new)`, `interpolate_at(tp, cp, n_valid, t_at, method)`, `auc_aumc(tp, cp, n_valid, method, t_start=, t_end=)`
- `pkpdutils.nca.steady_state`: `compute_steady_state(...)`, `accumulation_ratio(ss, sd)`, `superposition(timecourse, regimen, options, t_end)`
- `pkpdutils.nca.result`: `NCAResult`, `parameter_unit`; `pkpdutils.result.ParameterResult` (`point_variables` = variables with an extra dimension, excluded from `to_dataframe`)
- `pkpdutils.nca.uncertainty`: `LOGNORMAL_PARAMETERS`, `DISCRETE_PARAMETERS`, `TERMINAL_INDEPENDENT_PARAMETERS` (add the new names where they belong)
- `pkpdutils.plot`: `PlotStyle`, `_figure_of`, `plot_timecourse`, `plot_nca`, `draw_nca_panel`, `plot_nca_grid`
- `pkpdutils.fit.frontends.fit_timecourses` reads `timecourses.dose_time` to shift the times (adapt to the first dose)

## File structure

| file | responsibility |
| --- | --- |
| `src/pkpdutils/timecourse.py` | `Dosing`, `Timecourse.dosing`, batch dose layout, `dosing_of`, `n_doses`, the wrappers `from_events`/`to_events`/`from_pknca`/`from_adnca` |
| `src/pkpdutils/io.py` | `read_events`, `write_events`, `read_pknca`, `read_adnca`, the column aliases, the route mapping |
| `src/pkpdutils/nca/options.py` | `NCAOptions.tau`, `intervals`; `NCAFlag.INCOMPLETE_INTERVAL` |
| `src/pkpdutils/nca/intervals.py` | `compute_intervals` (per-interval parameters, PK and PD), `INTERVAL_UNITS` |
| `src/pkpdutils/nca/steady_state.py` | steady state of the last interval from the protocol, `superposition` on a protocol, `accumulation_ratio` |
| `src/pkpdutils/nca/nca.py` | the reference dose rule (last dose), the `interval` dimension in `_to_result`, `PARAMETER_UNITS` additions |
| `src/pkpdutils/nca/result.py` | `NCAResult.intervals()` |
| `src/pkpdutils/plot/timecourse.py`, `plot/nca.py`, `plot/style.py` | dose lines, interval shading, `plot_intervals`, `dose_color` |
| `docs/timecourses.md`, `docs/nca.md`, `docs/pd.md`, `docs/formats.md`, `docs/glossary.md`, `docs/api/*.md`, `zensical.toml` | documentation |
| `examples/steady_state.py`, `examples/formats.py`, `examples/README.md`, `tests/examples/test_examples.py` | examples |
| `tests/test_timecourse.py`, `tests/test_timecourses.py`, `tests/test_io.py`, `tests/nca/test_intervals.py`, `tests/nca/test_steady_state.py`, `tests/plot/test_plot_nca.py`, `tests/data/formats/*.csv` | tests and fixtures |

---

### Task 1: `Dosing`, `Timecourse.dosing`, `relative_to_dose(which)`

**Files:**
- Modify: `src/pkpdutils/timecourse.py` (after `DosingRegimen`; `Timecourse` fields and methods), `src/pkpdutils/__init__.py` (export `Dosing`)
- Test: `tests/test_timecourse.py`

**Interfaces (produces):**
- `class Dosing(BaseModel)` frozen, `arbitrary_types_allowed`: fields `amounts: np.ndarray`, `times: np.ndarray`, `durations: np.ndarray | None = None`, `unit: str`, `route: Route = Route.ORAL`. Validator (`mode="after"`): arrays to 1-D `float64` (via the existing `_as_float_array`), equal lengths, at least one dose, amounts `>= 0`, `check_dose_unit(unit)`; sort by `times` (stable) when unsorted (amounts and durations follow; `logger.warning("The dose times of the protocol were not sorted")`), duplicate times raise `ValueError("Duplicate dose times")`; `IV_INFUSION` needs `durations` with every value `> 0`, other routes need `durations is None` or all `NaN` (then stored as `None`).
- constructors: `Dosing.single(dose: Dose) -> Dosing`, `Dosing.from_doses(doses: Sequence[Dose]) -> Dosing` (`ValueError` on differing `unit` or `route`, empty sequence), `Dosing.regimen(dose: Dose, interval: float, n_doses: int) -> Dosing` (`interval > 0`, `n_doses >= 1`, times `dose.time + k * interval`, amounts and durations repeated).
- properties: `n_doses: int`, `doses: list[Dose]`, `first: Dose`, `last: Dose`, `intervals: np.ndarray` (`np.diff(times)`), `tau: float | None` (common interval when `n_doses >= 2` and `np.allclose(intervals, intervals[0], rtol=1e-9, atol=0)`, else `None`), `is_regular: bool` (`tau is not None`), `total_amount: float`, `quantity: Quantity` (total amount with unit), `per_bodyweight: bool`; method `shifted(offset: float) -> Dosing` (times minus offset); `__len__` = `n_doses`; `__eq__` by arrays (`np.array_equal` with NaN equality for durations) and scalar fields (pydantic's default compares arrays elementwise and raises: implement `__eq__` explicitly, as `Timecourse` does; look at its implementation).
- `DosingRegimen.dosing(self) -> Dosing`: `Dosing.regimen(self.dose, self.interval, self.n_doses)`; raises `ValueError` without `n_doses`.
- `Timecourse`: field `dosing: Dosing | None = None`; `model_validator(mode="before")` `_dose_to_dosing(cls, data)`: when `data` is a dict with a `dose` key: `dose is None` -> drop the key; a `Dose` -> `data["dosing"] = Dosing.single(dose)` (raise `ValueError("Give 'dose' or 'dosing', not both")` when `dosing` is also given and not `None`); a `Dosing` under `dose` is accepted too (moved). Property `dose -> Dose | None`: `self.dosing.first` when the protocol exists. Every internal use of `self.dose` keeps working through the property; `model_copy(update={"dose": ...})` does not run validators: change the internal uses (`relative_to_dose`, `from_dataframe`, `_timecourse`) to `dosing`.
- `relative_to_dose(self, which: Literal["first", "last"] = "first") -> Timecourse`: shift by `dosing.first.time` or `dosing.last.time`; the returned curve carries `dosing.shifted(offset)`; unchanged object when no protocol or the offset is 0.
- `Timecourse.__eq__` compares `dosing` (it compared `dose`).
- `to_dataframe()` unchanged; `from_dataframe(...)` unchanged signature (`dose=` argument stays, goes through the validator).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_timecourse.py` (keep the existing tests; they must pass unchanged, which proves the compatibility):
```python
def test_dosing_validation_and_properties() -> None:
    d = Dosing(amounts=[100, 100, 50], times=[0, 12, 24], unit="mg", route=Route.ORAL)
    assert d.n_doses == 3 and len(d) == 3
    assert d.times.tolist() == [0.0, 12.0, 24.0]
    assert d.intervals.tolist() == [12.0, 12.0]
    assert d.tau == 12.0 and d.is_regular
    assert d.first.amount == 100.0 and d.last.amount == 50.0 and d.last.time == 24.0
    assert d.total_amount == 250.0 and d.quantity.units == ureg.mg
    assert not d.per_bodyweight
    assert d.durations is None
    irregular = Dosing(amounts=[1, 1, 1], times=[0, 8, 24], unit="mg")
    assert irregular.tau is None and not irregular.is_regular
    single = Dosing.single(Dose(amount=5, unit="mg/kg", route=Route.IV_BOLUS, time=2.0))
    assert single.n_doses == 1 and single.tau is None and single.per_bodyweight and single.route is Route.IV_BOLUS
    assert single.shifted(2.0).times.tolist() == [0.0]


def test_dosing_sorts_and_rejects_duplicates_and_mixed_routes() -> None:
    d = Dosing(amounts=[1, 2], times=[12, 0], unit="mg")
    assert d.times.tolist() == [0.0, 12.0] and d.amounts.tolist() == [2.0, 1.0]
    with pytest.raises(ValueError, match="Duplicate"):
        Dosing(amounts=[1, 1], times=[0, 0], unit="mg")
    with pytest.raises(ValueError, match="length"):
        Dosing(amounts=[1, 1], times=[0], unit="mg")
    with pytest.raises(ValueError, match="at least one"):
        Dosing(amounts=[], times=[], unit="mg")
    with pytest.raises(ValueError, match="duration"):
        Dosing(amounts=[1], times=[0], unit="mg", route=Route.IV_INFUSION)
    with pytest.raises(ValueError, match="duration"):
        Dosing(amounts=[1], times=[0], durations=[0.5], unit="mg", route=Route.ORAL)
    inf = Dosing(amounts=[1, 1], times=[0, 12], durations=[0.5, 0.5], unit="mg", route=Route.IV_INFUSION)
    assert inf.doses[1].duration == 0.5
    with pytest.raises(ValueError, match="route"):
        Dosing.from_doses([Dose(amount=1, unit="mg"), Dose(amount=1, unit="mg", route=Route.IV_BOLUS, time=1)])
    with pytest.raises(ValueError, match="unit"):
        Dosing.from_doses([Dose(amount=1, unit="mg"), Dose(amount=1, unit="mmol", time=1)])


def test_dosing_regimen_constructors() -> None:
    dose = Dose(amount=100, unit="mg", route=Route.ORAL, time=1.0)
    d = Dosing.regimen(dose, interval=12, n_doses=4)
    assert d.times.tolist() == [1.0, 13.0, 25.0, 37.0] and d.amounts.tolist() == [100.0] * 4
    assert DosingRegimen(dose=dose, interval=12, n_doses=4).dosing() == d
    with pytest.raises(ValueError, match="n_doses"):
        DosingRegimen(dose=dose, interval=12).dosing()
    assert Dosing.from_doses(d.doses) == d


def test_timecourse_dose_keyword_and_property() -> None:
    dose = Dose(amount=100, unit="mg", route=Route.ORAL, time=0.5)
    tc = Timecourse(time=[1, 2, 4], value=[1, 2, 1], time_unit="hr", unit="mg/l", dose=dose)
    assert tc.dosing is not None and tc.dosing.n_doses == 1
    assert tc.dose == dose
    protocol = Dosing.regimen(dose, interval=12, n_doses=3)
    tc2 = Timecourse(time=[1, 2, 4, 13, 25, 30], value=[1, 2, 1, 3, 3, 2], time_unit="hr", unit="mg/l", dosing=protocol)
    assert tc2.dose == dose and tc2.dosing == protocol
    with pytest.raises(ValueError, match="not both"):
        Timecourse(time=[1], value=[1], time_unit="hr", unit="mg/l", dose=dose, dosing=protocol)
    assert Timecourse(time=[1], value=[1], time_unit="hr", unit="mg/l").dose is None


def test_relative_to_dose_first_and_last() -> None:
    protocol = Dosing(amounts=[1, 1], times=[2, 14], unit="mg")
    tc = Timecourse(time=[3, 8, 15, 20], value=[1, 2, 3, 4], time_unit="hr", unit="mg/l", dosing=protocol)
    first = tc.relative_to_dose()
    assert first.time.tolist() == [1, 6, 13, 18] and first.dosing is not None and first.dosing.times.tolist() == [0.0, 12.0]
    last = tc.relative_to_dose(which="last")
    assert last.time.tolist() == [-11, -6, 1, 6] and last.dosing is not None and last.dosing.times.tolist() == [-12.0, 0.0]
    assert Timecourse(time=[1], value=[1], time_unit="hr", unit="mg/l").relative_to_dose() is not None
```
Add `Dosing`, `DosingRegimen`, `ureg` to the imports of the test file as needed.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_timecourse.py -q -k "dosing or dose_keyword or relative_to_dose_first"`
Expected: FAIL with `ImportError: cannot import name 'Dosing'`

- [ ] **Step 3: Implement**

Add `Dosing` after `DosingRegimen` in `src/pkpdutils/timecourse.py` (it uses `Dose`, `Route`, `_as_float_array`, `check_dose_unit`, `is_per_bodyweight`, `Q_`; note `DosingRegimen.dosing` references `Dosing` defined later: fine inside a method). Module docstring: mention the protocol. `Timecourse`: replace the field `dose: Dose | None = None` by `dosing: Dosing | None = None`, add the before-validator and the `dose` property, adapt `__eq__`, `relative_to_dose`, `from_dataframe` (it builds `Dose(...)`: keep, the validator converts), `_timecourse` of `Timecourses` (Task 2 changes it fully; for now it must keep working: it builds `data["dose"] = Dose(...)`, which the validator accepts). Export `Dosing` in `src/pkpdutils/__init__.py` (`__all__` sorted). The `Timecourses` batch code is unchanged in this task (still one dose per sample via `dose.first`... it reads `tc.dose` which still works).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_timecourse.py tests/test_timecourses.py tests/nca -q -W error` then the full suite `uv run pytest -q -W error`, `uv run ruff check src tests && uv run ruff format src tests && uvx ty check`.
Expected: PASS everywhere (the existing tests exercise `dose=` and `tc.dose`).

- [ ] **Step 5: Commit**

```bash
git add src/pkpdutils/timecourse.py src/pkpdutils/__init__.py tests/test_timecourse.py
git commit -q -m "Add Dosing, the dosing protocol of a timecourse

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 2: The protocol in the batch layout (`Timecourses`)

**Files:**
- Modify: `src/pkpdutils/timecourse.py` (`Timecourses`), `src/pkpdutils/nca/nca.py` (`nca`, `partial_auc`: read the first dose of the 2-D arrays), `src/pkpdutils/nca/steady_state.py` (`compute_steady_state` receives the 2-D arrays: use the last dose column for now), `src/pkpdutils/fit/frontends.py` (`fit_timecourses`: shift by the first dose)
- Test: `tests/test_timecourses.py`, existing NCA and fit tests unchanged

**Interfaces (produces):**
- `Timecourses` dataset: `dose_amount`, `dose_time`, `dose_duration` over `(*sample_dims, "dose")`, coordinate `dose = np.arange(n_dose)`, NaN padded; `attrs["units"]` as before; `attrs["route"]`.
- properties: `dose_amount`, `dose_time`, `dose_duration` -> `np.ndarray | None` of shape `(*sample_shape, n_dose)`; new `n_doses -> np.ndarray | None` (finite count per sample, shape `sample_shape`), `n_dose -> int` (size of the `dose` dimension, 0 without doses), `first_dose_time`/`last_dose_time -> np.ndarray | None` (shape `sample_shape`; the last finite time per sample), `first_dose_amount`/`last_dose_amount`; `dosing_of(**indexers) -> Dosing | None` (protocol of one sample from the finite entries); `has_dose` unchanged.
- `from_arrays(..., dose=...)`: `Dose` (one dose for all), `Dosing` (same protocol for all samples), or a mapping `{"amount": array, "unit": str, "time": array | None, "duration": array | None}` where the arrays have shape `sample_shape` (one dose each) or `(*sample_shape, n_dose)` (protocols, NaN padded; `time` required then); `route` required with a mapping. `from_timecourses`: pads the protocols to the longest; requires one route. `from_dataframe` (long format): as today for one dose per sample; when the `dose_time` column carries several distinct values within a sample, every distinct `(dose_time, dose_amount)` pair is one dose of the protocol (rows with `NaN` dose columns are observations only); `_dose_of_group` returns a `Dosing`. `from_dataset` (simulation datasets): unchanged unless it builds doses (check; it takes `dose=` like `from_arrays`).
- `_timecourse(sample, label)`: builds `dosing=Dosing(amounts=finite, times=finite, durations=finite or None, unit=..., route=...)`.
- `to_dataframe()` of a batch: one row per sample and time as today; the dose columns become `dose_amount` of the first dose? No: the batch `to_dataframe` drops the dose variables (they have another dimension) and documents that `to_events` (Task 4) writes the protocol.
- NCA and fit callers: `nca.py` flattens `dose_amount`/`dose_time`/`dose_duration` to `(N, n_dose)`; in this task `compute_parameters` receives the FIRST dose column (`[:, 0]`) so that every existing test passes unchanged, and `compute_steady_state` the LAST finite dose per row; `partial_auc` shifts by the first dose time; `fit_timecourses` shifts by the first dose time. Task 3 replaces this with the reference dose rule.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_timecourses.py`:
```python
def test_batch_carries_protocols_padded() -> None:
    t = np.array([1.0, 2.0, 4.0, 13.0, 25.0])
    a = Timecourse(time=t, value=[1, 2, 1, 3, 3], time_unit="hr", unit="mg/l", dosing=Dosing(amounts=[100, 100], times=[0, 12], unit="mg"))
    b = Timecourse(time=t, value=[1, 2, 1, 3, 3], time_unit="hr", unit="mg/l", dose=Dose(amount=50, unit="mg", time=0.0))
    batch = Timecourses.from_timecourses([a, b], labels=["a", "b"])
    assert batch.n_dose == 2 and batch.has_dose
    assert batch.dose_amount is not None and batch.dose_amount.shape == (2, 2)
    assert batch.dose_amount[0].tolist() == [100.0, 100.0]
    assert batch.dose_amount[1][0] == 50.0 and np.isnan(batch.dose_amount[1][1])
    assert batch.dose_time is not None and batch.dose_time[0].tolist() == [0.0, 12.0]
    assert batch.n_doses is not None and batch.n_doses.tolist() == [2, 1]
    assert batch.last_dose_time is not None and batch.last_dose_time.tolist() == [12.0, 0.0]
    assert batch.first_dose_amount is not None and batch.first_dose_amount.tolist() == [100.0, 50.0]
    assert list(batch.ds["dose_amount"].dims) == ["individual", "dose"]
    assert batch.dosing_of(individual="a") == a.dosing
    assert batch.dosing_of(individual="b") == b.dosing
    assert batch.sel(individual="a") == a
    assert [tc.dosing for tc in batch] == [a.dosing, b.dosing]


def test_from_arrays_with_a_protocol_and_with_padded_mapping() -> None:
    time = np.array([1.0, 2.0, 4.0, 13.0])
    values = np.ones((3, 4))
    protocol = Dosing(amounts=[10, 10], times=[0, 12], unit="mg", route=Route.IV_BOLUS)
    batch = Timecourses.from_arrays(time, values, time_unit="hr", unit="mg/l", dose=protocol, route=None)
    assert batch.dose_amount is not None and batch.dose_amount.shape == (3, 2) and batch.route is Route.IV_BOLUS
    mapping = {
        "amount": np.array([[10, 10], [20, np.nan], [10, 5]]),
        "time": np.array([[0, 12], [0, np.nan], [0, 24]]),
        "unit": "mg",
    }
    batch2 = Timecourses.from_arrays(time, values, time_unit="hr", unit="mg/l", dose=mapping, route=Route.ORAL)
    assert batch2.n_doses is not None and batch2.n_doses.tolist() == [2, 1, 2]
    assert batch2.dosing_of(individual=2) == Dosing(amounts=[10, 5], times=[0, 24], unit="mg", route=Route.ORAL)
    with pytest.raises(ValueError, match="time"):
        Timecourses.from_arrays(time, values, time_unit="hr", unit="mg/l", dose={"amount": mapping["amount"], "unit": "mg"}, route=Route.ORAL)


def test_from_dataframe_builds_protocols_from_dose_rows() -> None:
    rows = []
    for subject, doses in (("s1", [(0.0, 100.0), (12.0, 100.0)]), ("s2", [(0.0, 50.0)])):
        for i, t in enumerate([1.0, 2.0, 13.0]):
            dt, da = doses[min(i, len(doses) - 1)]
            rows.append({"subject": subject, "time": t, "value": 1.0 + i, "dose_amount": da, "dose_time": dt})
    df = pd.DataFrame(rows)
    batch = Timecourses.from_dataframe(df, sample=["subject"], time_unit="hr", unit="mg/l", dose_amount="dose_amount", dose_unit="mg", dose_time="dose_time", route=Route.ORAL)
    assert batch.dosing_of(subject="s1") == Dosing(amounts=[100, 100], times=[0, 12], unit="mg", route=Route.ORAL)
    assert batch.dosing_of(subject="s2") == Dosing(amounts=[50], times=[0], unit="mg", route=Route.ORAL)


def test_single_dose_batch_layout_is_one_column() -> None:
    time = np.array([1.0, 2.0])
    batch = Timecourses.from_arrays(time, np.ones((2, 2)), time_unit="hr", unit="mg/l", dose=Dose(amount=1, unit="mg"), route=None)
    assert batch.n_dose == 1 and batch.dose_time is not None and batch.dose_time.shape == (2, 1)
    assert batch.dosing_of(individual=0) == Dosing.single(Dose(amount=1, unit="mg"))
```
(`route=None` with a `Dose`/`Dosing` means the route of the object; adapt if `from_arrays` rejects `route=None` there, the existing tests show the convention.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_timecourses.py -q -k "protocol or padded or dose_rows or one_column"`
Expected: FAIL (`n_dose` missing, shapes `(2,)`).

- [ ] **Step 3: Implement**

In `Timecourses`: a module constant `DOSE_DIM = "dose"`; a helper `_dose_variables(amounts, times, durations, dims, dose_unit, time_unit) -> dict` building the three `(dims + (DOSE_DIM,))` variables; `_pad_protocols(protocols: list[Dosing | None], n_dose) -> tuple[amounts, times, durations]` (NaN padded 2-D); the constructors use them. `dosing_of` selects with `self.ds.sel(indexers)` (labels) and rebuilds. `_timecourse` likewise. Properties as in the interfaces. Adapt `nca.py` (`flat` of the 2-D arrays: reshape to `(N, n_dose)`; `compute_parameters` gets `dose_amount[:, 0]`, `dose_time[:, 0]`, `dose_duration[:, 0]`; `compute_steady_state` gets the last finite column per row: write `_last_finite(a: np.ndarray) -> np.ndarray` in `nca.py`), `partial_auc` (`dose_time` first column), `fit_timecourses` (first column). Check `Timecourses.from_dataset` and `to_dataframe` for dose handling and adapt. Every docstring that says "one dose per sample" is updated.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -W error` (all; the NCA reference test, steady state tests, fit tests and examples must pass unchanged), `uv run ruff check src tests && uv run ruff format src tests && uvx ty check`.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -q -m "Carry the dosing protocol in the batch layout

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 3: Multiple dosing analysis in the NCA

**Files:**
- Create: `src/pkpdutils/nca/intervals.py`
- Modify: `src/pkpdutils/nca/options.py` (`tau`, `intervals`, remove `regimen`, `NCAFlag.INCOMPLETE_INTERVAL`), `src/pkpdutils/nca/nca.py` (reference dose rule, interval variables in `_to_result`, units), `src/pkpdutils/nca/steady_state.py` (protocol based steady state, `superposition` on a protocol), `src/pkpdutils/nca/result.py` (`NCAResult.intervals()`), `src/pkpdutils/nca/uncertainty.py` (parameter sets), `src/pkpdutils/nca/__init__.py`, `examples/steady_state.py`
- Test: `tests/nca/test_intervals.py` (new), `tests/nca/test_steady_state.py` (adapted), `tests/nca/test_options.py`

**Interfaces (produces):**
- `NCAOptions`: `regimen` removed; `tau: float | None = Field(default=None, gt=0)`; `intervals: bool = True`. `NCAFlag.INCOMPLETE_INTERVAL` (next free bit; the last dosing interval is not covered by the data).
- Reference dose rule in `nca.py`: `compute_parameters` is called with the LAST dose of every row (`_last_finite` of amount, time, duration), so the point parameters are computed from the last dose on; the rows with one dose are unchanged. A row without any dose: as today.
- `intervals.py`: `compute_intervals(t, c, *, dose_amount, dose_time, tau, options) -> dict[str, np.ndarray]` with `t`, `c` `(N, n)` in the frame of the curve, `dose_amount`/`dose_time` `(N, K)` NaN padded, `tau` `(N,)` the length of the last interval per row (NaN when unknown); returns arrays `(N, K)`: `interval_start`, `interval_end`, `interval_dose`, `interval_n_points`, and for `Kind.CONCENTRATION` `interval_auc`, `interval_cmax`, `interval_tmax` (relative to the interval start), `interval_cmin`, `interval_ctrough`, `interval_c_start`, `interval_cavg`, `interval_fluctuation`, `interval_swing`; for `Kind.EFFECT` `interval_auec`, `interval_emax`, `interval_temax`, `interval_emin`, `interval_eavg`, `interval_time_above` (NaN without `effect_threshold`). Per interval `k` of a row: start `s = dose_time[:, k]`, end `e = dose_time[:, k+1]` (or `s + tau` for the last); `pack_valid` the row once, then for the interval interpolate `c(s)` and `c(e)` with `interpolate_at` (NaN when outside the observed range), insert both points (`insert_point`), and evaluate the area with `auc_aumc(..., t_start=s, t_end=e)` (the area between the boundaries: the packed arrays already exclude points outside when the masks are applied as `compute_steady_state` does), max/min/argmax over the points with `s <= t <= e` plus the boundary values, `cavg = auc / (e - s)`, `fluctuation = (cmax - cmin) / cavg`, `swing = (cmax - cmin) / cmin`. An interval with `e` beyond the last observation (`c(e)` NaN) is incomplete: every value NaN, `n_points` the count inside. `INTERVAL_UNITS: dict[str, str]` maps every interval variable to a unit expression as `PARAMETER_UNITS` does (`interval_start`, `interval_end`, `interval_tmax`, `interval_temax`, `interval_time_above` -> `{time}`; `interval_dose` -> `{dose}`; areas `({unit}) * ({time})`; concentrations `{unit}`; `interval_cavg`, `interval_eavg` `{unit}`; `interval_fluctuation`, `interval_swing`, `interval_n_points` dimensionless).
- `steady_state.py`: `compute_steady_state(t, c, *, dose_amount, dose_time, dose_duration, route, options)` now takes the 2-D dose arrays; it computes the last interval `[t_K, t_K + tau_K]` where `tau_K = options.tau` when given, else `t_K - t_{K-1}` for rows with `K >= 2`, else NaN (no steady state for such rows: NaN values, no flag). Its outputs keep the names `auc_tau`, `cmin_ss`, `cmax_ss`, `ctrough`, `cavg`, `fluctuation`, `swing`, `cl_ss`, `accumulation_ratio` (predicted from `lambda_z` of the post-last-dose analysis), plus `accumulation_ratio_obs` (`interval_auc[:, -1] / interval_auc[:, 0]`, NaN when either is NaN or `K == 1`), `n_doses`, `tau`; for effects `auec_tau`, `emin_ss`, `emax_ss`, `eavg`, `time_above_tau`. Implementation: call `compute_intervals` for the last interval (and all intervals when `options.intervals`), take its last column. `INCOMPLETE_INTERVAL` set in `flags` when `K >= 1`, `tau_K` finite and the last interval is incomplete. `_compute_chunk` in `nca.py` dispatches to `compute_steady_state` when any row has `n_doses >= 2` or `options.tau` is set; otherwise `compute_parameters` as before (single dose batches unchanged, the reference test untouched).
- `nca.py` `_to_result`: variables with `(N, K)` values get the dims `(*sample_dims, "interval")` with the coordinate `interval = np.arange(1, K + 1)`; `PARAMETER_UNITS` gains the steady state additions (`accumulation_ratio_obs`, `n_doses`, `tau`, `auec_tau`, `emin_ss`, `emax_ss`, `eavg`, `time_above_tau`) and merges `INTERVAL_UNITS`. `unit_expression` covers them.
- `NCAResult.intervals() -> pd.DataFrame`: one row per sample and interval with the sample coordinates, `interval`, and every `interval_*` variable (empty frame without the dimension). `NCAResult.has_intervals -> bool`.
- `uncertainty.py`: `interval_*` variables are point variables (extra dim) and are left alone by the bootstrap/delta code (check that `bootstrap`/`delta` skip variables with the interval dim; if they iterate over `result.parameters`, nothing to do); `DISCRETE_PARAMETERS` gains `n_doses`, `tau`, `interval_n_points`; `LOGNORMAL_PARAMETERS` gains `accumulation_ratio_obs`, `auec_tau`, `eavg`.
- `superposition(timecourse, dosing: Dosing | DosingRegimen, options=None, t_end=None) -> Timecourse`: superposes at every dose time with the factor `amount_k / amount_single` (the single dose amount is `timecourse.dose.amount`); the result carries `dosing` (the `Dosing`; a `DosingRegimen` is converted with `.dosing()`); the times of the result start at the first dose time.
- `nca/__init__.py` exports `compute_intervals` is not needed; export nothing new besides what exists (`superposition`, `accumulation_ratio`).

- [ ] **Step 1: Write the failing tests**

`tests/nca/test_intervals.py`:
```python
import numpy as np
import pytest

from pkpdutils import Dose, Dosing, NCAOptions, Route, Timecourse, Timecourses, nca, nca_single
from pkpdutils.nca import AUCMethod, superposition
from pkpdutils.nca.options import Kind, NCAFlag

K, C0, TAU, N_DOSES = 0.2, 10.0, 12.0, 5


def multiple_dose_curve(times: np.ndarray) -> np.ndarray:
    """Superposition of a mono-exponential bolus curve given every TAU hours."""
    c = np.zeros_like(times)
    for k in range(N_DOSES):
        shifted = times - k * TAU
        c += np.where(shifted >= 0, C0 * np.exp(-K * shifted), 0.0)
    return c


def interval_auc_closed_form(k: int) -> float:
    """Area of interval k (0-based) of the superposed curve, the sum over the doses given so far."""
    return sum(C0 / K * (np.exp(-K * (k - j) * TAU) - np.exp(-K * (k + 1 - j) * TAU)) for j in range(k + 1))


TIMES = np.sort(np.concatenate([np.arange(0, N_DOSES * TAU + 0.01, 0.5), [N_DOSES * TAU + 24]]))
PROTOCOL = Dosing.regimen(Dose(amount=100, unit="mg", route=Route.IV_BOLUS), interval=TAU, n_doses=N_DOSES)
TC = Timecourse(time=TIMES, value=multiple_dose_curve(TIMES), time_unit="hr", unit="mg/l", dosing=PROTOCOL)


def test_interval_parameters_match_the_closed_form() -> None:
    result = nca_single(TC, NCAOptions(auc_method=AUCMethod.LOG))
    assert result.has_intervals
    auc = result["interval_auc"].to_numpy()
    assert auc.shape == (N_DOSES,)
    for k in range(N_DOSES):
        assert auc[k] == pytest.approx(interval_auc_closed_form(k), rel=1e-3)
    assert result["interval_start"].to_numpy().tolist() == [0, 12, 24, 36, 48]
    assert result["interval_end"].to_numpy().tolist() == [12, 24, 36, 48, 60]
    assert result["interval_dose"].to_numpy().tolist() == [100.0] * 5
    cmax = result["interval_cmax"].to_numpy()
    assert cmax[0] == pytest.approx(C0) and cmax[4] == pytest.approx(multiple_dose_curve(np.array([48.0]))[0])
    assert result["interval_tmax"].to_numpy().tolist() == [0.0] * 5
    ctrough = result["interval_ctrough"].to_numpy()
    assert ctrough[0] == pytest.approx(C0 * np.exp(-K * TAU))
    assert np.all(np.diff(ctrough) > 0)  # accumulation
    assert result["interval_cavg"].to_numpy()[2] == pytest.approx(auc[2] / TAU)
    df = result.intervals()
    assert list(df["interval"]) == [1, 2, 3, 4, 5] and "interval_auc" in df.columns


def test_steady_state_is_the_last_interval_and_point_parameters_follow_the_last_dose() -> None:
    result = nca_single(TC, NCAOptions(auc_method=AUCMethod.LOG))
    q = result.to_quantities()
    assert float(q["auc_tau"].magnitude) == pytest.approx(interval_auc_closed_form(N_DOSES - 1), rel=1e-3)
    assert float(q["n_doses"].magnitude) == N_DOSES and float(q["tau"].magnitude) == TAU
    assert float(q["ctrough"].magnitude) == pytest.approx(result["interval_ctrough"].to_numpy()[-1])
    assert float(q["accumulation_ratio_obs"].magnitude) == pytest.approx(interval_auc_closed_form(4) / interval_auc_closed_form(0), rel=1e-3)
    assert float(q["accumulation_ratio"].magnitude) == pytest.approx(1 / (1 - np.exp(-K * TAU)), rel=1e-2)
    # the point parameters are computed from the last dose on
    assert float(q["tmax"].magnitude) == 0.0
    assert float(q["cmax"].magnitude) == pytest.approx(multiple_dose_curve(np.array([48.0]))[0])
    assert float(q["lambda_z"].magnitude) == pytest.approx(K, rel=1e-3)
    assert float(q["auc_last"].magnitude) == pytest.approx(result["interval_auc"].to_numpy()[-1] + (multiple_dose_curve(np.array([60.0]))[0] - multiple_dose_curve(np.array([84.0]))[0]) / K, rel=1e-2)
    assert "INCOMPLETE_INTERVAL" not in result.flags()


def test_single_dose_analysis_is_unchanged() -> None:
    t = np.array([0.5, 1, 2, 4, 8, 12, 24])
    tc = Timecourse(time=t, value=C0 * np.exp(-K * t), time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS))
    result = nca_single(tc)
    assert not result.has_intervals and "auc_tau" not in result
    assert result.intervals().empty


def test_tau_option_makes_a_single_dose_curve_a_steady_state_interval() -> None:
    t = np.arange(0, 12.5, 0.5)
    tc = Timecourse(time=t, value=multiple_dose_curve(t + 48.0), time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS))
    result = nca_single(tc, NCAOptions(tau=TAU, auc_method=AUCMethod.LOG))
    assert result.has_intervals and result["interval_auc"].to_numpy().shape == (1,)
    assert float(result["auc_tau"].to_numpy()) == pytest.approx(interval_auc_closed_form(4), rel=1e-3)
    assert np.isnan(float(result["accumulation_ratio_obs"].to_numpy()))


def test_incomplete_last_interval_is_flagged() -> None:
    t = TIMES[TIMES <= 54]
    tc = Timecourse(time=t, value=multiple_dose_curve(t), time_unit="hr", unit="mg/l", dosing=PROTOCOL)
    result = nca_single(tc)
    assert "INCOMPLETE_INTERVAL" in result.flags()
    assert np.isnan(result["interval_auc"].to_numpy()[-1]) and np.isnan(float(result["auc_tau"].to_numpy()))
    assert result["interval_n_points"].to_numpy()[-1] > 0
    assert np.isfinite(result["interval_auc"].to_numpy()[:-1]).all()


def test_batch_with_different_numbers_of_doses_and_workers() -> None:
    three = Dosing.regimen(Dose(amount=100, unit="mg", route=Route.IV_BOLUS), interval=TAU, n_doses=3)
    t3 = TIMES[TIMES <= 36]
    c3 = np.zeros_like(t3)
    for k in range(3):
        c3 += np.where(t3 - k * TAU >= 0, C0 * np.exp(-K * (t3 - k * TAU)), 0.0)
    a = TC
    b = Timecourse(time=t3, value=c3, time_unit="hr", unit="mg/l", dosing=three)
    batch = Timecourses.from_timecourses([a, b], labels=["five", "three"])
    serial = nca(batch, NCAOptions(auc_method=AUCMethod.LOG))
    assert serial.ds.sizes["interval"] == 5
    auc = serial["interval_auc"]
    assert np.isfinite(auc.sel(individual="three").to_numpy()[:3]).all() and np.isnan(auc.sel(individual="three").to_numpy()[3:]).all()
    assert serial["n_doses"].to_numpy().tolist() == [5, 3]
    parallel = nca(batch, NCAOptions(auc_method=AUCMethod.LOG, n_workers=2, chunk_rows=1))
    for name in serial.ds.data_vars:
        np.testing.assert_allclose(serial[name].to_numpy(), parallel[name].to_numpy(), equal_nan=True)


def test_effect_intervals() -> None:
    t = np.arange(0, 36.5, 0.5)
    effect = 5.0 + 3.0 * np.sin(np.pi * (t % 12) / 12)  # rises and falls in every 12 h interval
    tc = Timecourse(time=t, value=effect, time_unit="hr", unit="mmHg", dosing=Dosing.regimen(Dose(amount=1, unit="mg"), interval=12, n_doses=3))
    result = nca_single(tc, NCAOptions(kind=Kind.EFFECT, effect_threshold=6.0))
    assert result["interval_emax"].to_numpy() == pytest.approx([8.0, 8.0, 8.0])
    assert result["interval_temax"].to_numpy() == pytest.approx([6.0, 6.0, 6.0])
    assert result["interval_emin"].to_numpy() == pytest.approx([5.0, 5.0, 5.0])
    auec = result["interval_auec"].to_numpy()
    assert auec == pytest.approx([5 * 12 + 3 * 24 / np.pi] * 3, rel=2e-3)
    assert result["interval_time_above"].to_numpy() == pytest.approx([8.0, 8.0, 8.0], rel=2e-2)
    q = result.to_quantities()
    assert float(q["auec_tau"].magnitude) == pytest.approx(auec[-1])
    assert float(q["emax_ss"].magnitude) == pytest.approx(8.0) and float(q["eavg"].magnitude) == pytest.approx(auec[-1] / 12)


def test_superposition_on_a_protocol_with_different_amounts() -> None:
    t = np.array([0.5, 1, 2, 4, 8, 12, 24, 36, 48])
    single = Timecourse(time=t, value=C0 * np.exp(-K * t), time_unit="hr", unit="mg/l", dose=Dose(amount=100, unit="mg", route=Route.IV_BOLUS))
    protocol = Dosing(amounts=[100, 200, 100], times=[0, 12, 24], unit="mg", route=Route.IV_BOLUS)
    predicted = superposition(single, protocol, NCAOptions(auc_method=AUCMethod.LOG))
    assert predicted.dosing == protocol
    expected = C0 * np.exp(-K * 24.5) + 2 * C0 * np.exp(-K * 12.5) + C0 * np.exp(-K * 0.5)
    at = predicted.value[np.isclose(predicted.time, 24.5)]
    assert at.size == 1 and at[0] == pytest.approx(expected, rel=1e-3)
```

Adapt `tests/nca/test_steady_state.py`: every `NCAOptions(regimen=DosingRegimen(dose=..., interval=tau))` on a curve that carries only the last dose becomes `NCAOptions(tau=tau)` with the curve's `dose`; the expected numbers stay; `test_superposition_needs_n_doses_and_terminal_phase` passes a `DosingRegimen` without `n_doses` (still raises) and a `Dosing`. `tests/nca/test_options.py`: `regimen` gone, `tau` and `intervals` present (`NCAOptions(tau=0)` raises).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/nca/test_intervals.py -q`
Expected: FAIL (`NCAOptions` has no `tau`, no `interval_auc`).

- [ ] **Step 3: Implement**

`intervals.py` first (pure numpy, test it through `nca_single`), then `steady_state.py`, `options.py`, `nca.py`, `result.py`, `uncertainty.py`. Keep `compute_parameters` untouched except its call site. Rewrite `examples/steady_state.py`: build the multiple dose curve by `superposition(single, Dosing.regimen(dose, 12, 10))`, run `nca_single(predicted, NCAOptions(auc_method=AUCMethod.LOG))` directly, print `result.intervals()` and the steady state parameters, plot the curve (the figure keeps its name `steady_state.png`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -W error` (all, incl. `tests/nca/test_reference.py` unchanged and the examples), `uv run ruff check src tests examples && uv run ruff format src tests examples && uvx ty check`.

- [ ] **Step 5: Commit**

```bash
git add src tests examples
git commit -q -m "Analyse every dosing interval of a multiple dose curve and the steady state from the protocol

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 4: Exchange formats (`io.py`)

**Files:**
- Create: `src/pkpdutils/io.py`, `tests/test_io.py`, `tests/data/formats/multiple_dose_events.csv`, `tests/data/formats/monolix_events.csv`, `tests/data/formats/pknca_conc.csv`, `tests/data/formats/pknca_dose.csv`, `tests/data/formats/adnca.csv`
- Modify: `src/pkpdutils/timecourse.py` (`Timecourses.from_events`, `to_events`, `from_pknca`, `from_adnca` delegating to `io`), `src/pkpdutils/__init__.py` (no new top-level names; `pkpdutils.io` is imported as a module)

**Interfaces (produces), exactly as spec section 3:**
- `read_events(df, *, time_unit, unit, dose_unit, route, id="ID", time="TIME", dv="DV", amt="AMT", evid="EVID", mdv="MDV", rate="RATE", tinf="TINF", addl="ADDL", ii="II", ss="SS", ss_doses=5, dim="individual", substance="substance", covariates: Sequence[str] | None = None) -> Timecourses`. Column lookup case-insensitive with the Monolix aliases (`AMOUNT`->amt, `OBSERVATION`->dv, `ADDITIONAL DOSES`->addl, `INTERDOSE INTERVAL`->ii, `STEADY STATE`->ss, `INFUSION DURATION`->tinf, `INFUSION RATE`->rate); missing optional columns are treated as absent. Dose rows: `EVID in (1, 4)` when `evid` exists, else `AMT > 0`. Observation rows: `EVID == 0` (or no `evid`) and (`MDV == 0` or no `mdv`) and `DV` not NaN; `EVID 2/3` rows dropped with a `logger.warning` count. Duration: `TINF` when present and `> 0`, else `AMT / RATE` when `RATE > 0`; `RATE < 0` raises `ValueError("Modelled rates (RATE -1/-2) are not data")`. `ADDL`/`II`: `ADDL` extra doses at `+II` each. `SS == 1`: `ss_doses` preceding doses at `-II` (only when `II > 0`), `attrs["steady_state_marker"] = True`. Covariates: `covariates` names, or when `None` every extra column which is constant within each subject and not one of the known columns, become coordinates along `dim`. Times are kept as given. Mixed routes cannot occur (one `route` argument). Subjects sorted by first appearance; the `dim` coordinate carries the `ID` values.
- `write_events(timecourses, *, id="ID", time="TIME", dv="DV", amt="AMT", evid="EVID", mdv="MDV", rate="RATE") -> pd.DataFrame`: per sample the dose rows (`EVID 1`, `MDV 1`, `DV NaN`, `AMT`, `RATE = AMT / duration` for infusions else 0) and the observation rows (`EVID 0`, `MDV 0`, `AMT 0`, `RATE 0`), sorted by subject then time, dose rows before observations at equal times; covariate coordinates along `dim` become columns.
- `read_pknca(conc, dose, *, time_unit, unit, dose_unit, route, conc_col="conc", time_col="time", dose_col="dose", dose_time_col="time", subject="subject", groups: Sequence[str] = (), duration_col: str | None = None, dim="individual", substance="substance") -> Timecourses`: subjects from `conc[subject]`; `groups` columns must be constant per subject and become coordinates; every dose row of a subject is a dose of the protocol; a subject without dose rows gets no protocol; `NA` values kept as NaN, `0` kept as 0.
- `read_adnca(df, *, time_unit="hr", unit=None, dose_unit=None, route=None, subject="USUBJID", analyte=None, param="PARAMCD", value="AVAL", value_unit="AVALU", time_first="AFRLT", time_ref="ARRLT", dose="DOSEA", dose_unit_col="DOSEU", route_col="ROUTE", dtype="DTYPE", lloq="ALLOQ", dim="individual", substance=None) -> Timecourses`: rows with `PARAMCD == analyte` (when `analyte` is `None` the single distinct value, else `ValueError` listing them); `DTYPE == "COPY"` rows dropped; time = `AFRLT`; dose times per subject = distinct `round(AFRLT - ARRLT, 6)` with amount `DOSEA` of the rows (a subject whose rows disagree on the amount for one dose time raises); units from the arguments, else from the first row's `AVALU`/`DOSEU` (pint parses `ng/mL`, `mg`; a unit string that pint rejects raises with the string); route from `route`, else from `ROUTE` mapped case-insensitively (`ORAL`, `PO` -> `ORAL`; `INTRAVENOUS`, `IV`, `INTRAVENOUS BOLUS`, `IV BOLUS` -> `IV_BOLUS`; `INTRAVENOUS INFUSION`, `IV INFUSION` -> `IV_INFUSION`; unknown raises); `ALLOQ` becomes the coordinate `lloq` along `dim` when present; `substance` defaults to the analyte.
- `Timecourses.from_events(df, **kwargs)`, `to_events(**kwargs)`, `from_pknca(conc, dose, **kwargs)`, `from_adnca(df, **kwargs)`: thin delegations (import `pkpdutils.io` inside the methods to avoid the cycle, `io` imports `timecourse`).

- [ ] **Step 1: Fixtures**

`tests/data/formats/multiple_dose_events.csv` (NONMEM layout; 3 subjects, oral, 100 mg BID via `ADDL`/`II` for subject 1, explicit rows for subject 2, one dose with `SS=1` for subject 3; weight covariate):
```csv
ID,TIME,DV,AMT,EVID,MDV,RATE,ADDL,II,SS,WT
1,0,.,100,1,1,0,3,12,0,70
1,1,5.1,0,0,0,0,0,0,0,70
1,4,3.9,0,0,0,0,0,0,0,70
1,12,1.2,0,0,0,0,0,0,0,70
1,37,4.4,0,0,0,0,0,0,0,70
1,48,1.5,0,0,0,0,0,0,0,70
2,0,.,100,1,1,0,0,0,0,82
2,12,.,100,1,1,0,0,0,0,82
2,1,4.8,0,0,0,0,0,0,0,82
2,13,6.0,0,0,0,0,0,0,0,82
2,24,1.9,0,0,0,0,0,0,0,82
3,0,.,100,1,1,0,0,12,1,65
3,0.5,3.0,0,0,0,0,0,0,0,65
3,2,5.5,0,0,0,0,0,0,0,65
3,12,1.8,0,0,0,0,0,0,0,65
```
`tests/data/formats/monolix_events.csv` (Monolix names, iv infusion via `TINF`):
```csv
ID,TIME,OBSERVATION,AMOUNT,TINF,ADM
1,0,.,50,1,1
1,0.5,2.1,.,.,.
1,1,4.0,.,.,.
1,2,3.1,.,.,.
1,6,1.0,.,.,.
```
`tests/data/formats/pknca_conc.csv` and `pknca_dose.csv` (2 subjects, `treatment` group, subject 2 with two doses):
```csv
subject,treatment,time,conc
1,A,0,0
1,A,1,4.2
1,A,4,3.0
1,A,12,1.0
2,B,0,0
2,B,1,4.0
2,B,13,5.5
2,B,24,2.0
```
```csv
subject,treatment,time,dose
1,A,0,100
2,B,0,100
2,B,12,100
```
`tests/data/formats/adnca.csv` (1 analyte, 2 subjects, subject 2 with two doses, one `COPY` row, BLQ row):
```csv
USUBJID,PARAMCD,AVAL,AVALU,AFRLT,ARRLT,NFRLT,NRRLT,DOSEA,DOSEU,ROUTE,DTYPE,ALLOQ
S1,XAN,0.05,ng/mL,0.5,0.5,0.5,0.5,100,mg,ORAL,,0.1
S1,XAN,4.2,ng/mL,1,1,1,1,100,mg,ORAL,,0.1
S1,XAN,1.0,ng/mL,12,12,12,12,100,mg,ORAL,,0.1
S2,XAN,4.0,ng/mL,1,1,1,1,100,mg,ORAL,,0.1
S2,XAN,1.1,ng/mL,12,12,12,12,100,mg,ORAL,,0.1
S2,XAN,1.1,ng/mL,12,0,12,0,100,mg,ORAL,COPY,0.1
S2,XAN,5.5,ng/mL,13,1,13,1,100,mg,ORAL,,0.1
S2,XAN,2.0,ng/mL,24,12,24,12,100,mg,ORAL,,0.1
```

- [ ] **Step 2: Write the failing tests**

`tests/test_io.py`:
```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pkpdutils import Dosing, Route, Timecourses
from pkpdutils.io import read_adnca, read_events, read_pknca, write_events

DATA = Path(__file__).parent / "data" / "formats"


def events() -> pd.DataFrame:
    return pd.read_csv(DATA / "multiple_dose_events.csv", na_values=".")


def test_read_events_expands_addl_and_ss_and_keeps_covariates() -> None:
    batch = read_events(events(), time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL)
    assert batch.sample_dims == ("individual",) and batch.ds["individual"].to_numpy().tolist() == [1, 2, 3]
    assert batch.dosing_of(individual=1) == Dosing(amounts=[100] * 4, times=[0, 12, 24, 36], unit="mg", route=Route.ORAL)
    assert batch.dosing_of(individual=2) == Dosing(amounts=[100, 100], times=[0, 12], unit="mg", route=Route.ORAL)
    ss = batch.dosing_of(individual=3)
    assert ss is not None and ss.times.tolist() == [-60, -48, -36, -24, -12, 0] and batch.ds.attrs["steady_state_marker"] is True
    assert batch.ds["WT"].to_numpy().tolist() == [70, 82, 65]
    tc = batch.sel(individual=2)
    assert tc.time.tolist() == [1, 13, 24] and tc.value.tolist() == [4.8, 6.0, 1.9]
    assert Timecourses.from_events(events(), time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL) == batch


def test_read_events_monolix_aliases_and_infusion() -> None:
    df = pd.read_csv(DATA / "monolix_events.csv", na_values=".")
    batch = read_events(df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.IV_INFUSION)
    d = batch.dosing_of(individual=1)
    assert d is not None and d.route is Route.IV_INFUSION and d.durations is not None and d.durations.tolist() == [1.0]
    assert batch.sel(individual=1).value.tolist() == [2.1, 4.0, 3.1, 1.0]


def test_read_events_rate_and_errors() -> None:
    df = pd.DataFrame({"ID": [1, 1], "TIME": [0, 1], "DV": [np.nan, 2.0], "AMT": [30, 0], "EVID": [1, 0], "RATE": [15, 0]})
    d = read_events(df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.IV_INFUSION).dosing_of(individual=1)
    assert d is not None and d.durations is not None and d.durations.tolist() == [2.0]
    df["RATE"] = [-1, 0]
    with pytest.raises(ValueError, match="Modelled rates"):
        read_events(df, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.IV_INFUSION)
    with pytest.raises(ValueError, match="column"):
        read_events(df.drop(columns=["TIME"]), time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL)


def test_events_round_trip() -> None:
    batch = read_events(events(), time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL)
    table = write_events(batch)
    assert list(table.columns)[:7] == ["ID", "TIME", "DV", "AMT", "EVID", "MDV", "RATE"] and "WT" in table.columns
    again = read_events(table, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL)
    for i in (1, 2, 3):
        assert again.dosing_of(individual=i) == batch.dosing_of(individual=i)
        assert again.sel(individual=i) == batch.sel(individual=i)
    assert batch.to_events().equals(table)


def test_read_pknca() -> None:
    conc = pd.read_csv(DATA / "pknca_conc.csv")
    dose = pd.read_csv(DATA / "pknca_dose.csv")
    batch = read_pknca(conc, dose, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL, groups=["treatment"])
    assert batch.ds["treatment"].to_numpy().tolist() == ["A", "B"]
    assert batch.dosing_of(individual=2) == Dosing(amounts=[100, 100], times=[0, 12], unit="mg", route=Route.ORAL)
    assert batch.sel(individual=1).value.tolist() == [0.0, 4.2, 3.0, 1.0]
    assert Timecourses.from_pknca(conc, dose, time_unit="hr", unit="mg/l", dose_unit="mg", route=Route.ORAL, groups=["treatment"]) == batch


def test_read_adnca() -> None:
    df = pd.read_csv(DATA / "adnca.csv")
    batch = read_adnca(df)
    assert batch.unit == "ng/mL" or batch.unit == "nanogram / milliliter"
    assert batch.route is Route.ORAL and batch.substance == "XAN"
    assert batch.dosing_of(individual="S2") == Dosing(amounts=[100, 100], times=[0, 12], unit="mg", route=Route.ORAL)
    assert batch.sel(individual="S2").time.tolist() == [1, 12, 13, 24]  # the COPY row is dropped
    assert batch.ds["lloq"].to_numpy().tolist() == [0.1, 0.1]
    with pytest.raises(ValueError, match="analyte"):
        read_adnca(pd.concat([df, df.assign(PARAMCD="OTHER")]))
    assert Timecourses.from_adnca(df) == batch
```
(`batch.unit` is whatever `parse_unit` canonicalizes `ng/mL` to; adjust the assertion to the actual convention of `Timecourses.unit` after reading `parse_unit`.)

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_io.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pkpdutils.io'`

- [ ] **Step 4: Implement `io.py` and the wrappers**

Module docstring with the three formats and the references (NONMEM: Bauer 2019 CPT:PSP tutorial; Monolix data format documentation; PKNCA: Denney et al.; CDISC ADaM ADNCA IG) as citation keys; add the four entries to `docs/references.md` under a new heading "Data formats" (Task 6 does the docs, but the references are needed for the docstrings: add them in this task).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_io.py tests/test_timecourses.py -q -W error`, then the full suite, ruff, ty.

- [ ] **Step 6: Commit**

```bash
git add src tests docs/references.md
git commit -q -m "Read and write the event, PKNCA and ADNCA exchange formats

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 5: Figures

**Files:**
- Modify: `src/pkpdutils/plot/style.py` (`dose_color: str = "gray"`), `src/pkpdutils/plot/timecourse.py` (dose lines), `src/pkpdutils/plot/nca.py` (interval shading in `draw_nca_panel`, `plot_intervals`), `src/pkpdutils/plot/__init__.py`
- Test: `tests/plot/test_plot_nca.py`, `tests/plot/test_plot_timecourse.py`

**Interfaces:**
- `plot_timecourse(...)`: for a single `Timecourse` with a protocol of `n_doses >= 2`, one `ax.axvline` per dose time (`color=style.dose_color`, `linestyle=":"`, `linewidth=1`); for a batch no lines (protocols differ).
- `draw_nca_panel(...)`: when the result has `auc_tau` (steady state), the analysed interval `[t_last_dose, t_last_dose + tau]` is shaded with `style.auc_color` at `alpha` (the AUC(0-tlast) shading keeps its role for single dose; for a multiple dose result the shading is the last interval and the legend entry reads `AUC(0-tau)`).
- `plot_intervals(result: NCAResult, name: str = "interval_auc", *, ax=None, style=DEFAULT_STYLE, **indexers) -> Figure`: the per-interval variable against the interval number; without indexers one line per sample (label = sample label), with indexers one sample; y label `name [unit]`, x label `interval`; `ValueError` when the result has no intervals or `name` is not an interval variable.

- [ ] **Step 1: Tests** (`tests/plot/`): a protocol curve draws `n_doses` dotted vertical lines; `plot_intervals` on the batch of `test_batch_with_different_numbers_of_doses_and_workers` (rebuild it in the test) returns a figure with two lines and the y label `interval_auc [...]`; `ValueError` on a single dose result. Run to see them fail.

- [ ] **Step 2: Implement, run `uv run pytest tests/plot -q -W error`, ruff, ty, view the figures once (save to the scratchpad, Read them), commit:**

```bash
git add src/pkpdutils/plot tests/plot
git commit -q -m "Draw the dose times, the analysed interval and the per-interval parameters

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 6: Documentation, examples, exports

**Files:**
- Create: `docs/formats.md`, `docs/api/io.md`, `docs/api/nca.intervals.md`, `examples/formats.py`
- Modify: `docs/timecourses.md` (section "Dosing protocols"), `docs/nca.md` (section "Multiple dosing" replacing the steady state paragraphs of Concepts/Math/Parameters/API), `docs/pd.md` (effect intervals), `docs/glossary.md` (new variables), `docs/plotting.md` (`plot_intervals`, dose lines), `docs/api/index.md`, `docs/api/timecourse.md` (unchanged directive, check it renders `Dosing`), `zensical.toml` (nav: "Data formats" after Timecourses; api pages), `examples/README.md`, `tests/examples/test_examples.py`, `CLAUDE.md` (architecture: `Dosing`, batch dose dimension, `intervals.py`, `io.py`, the reference dose rule; commands: `python -m examples.formats`), `README.md`/`docs/index.md` feature bullets (timecourses "with dosing protocols", a bullet "Data formats")

`examples/formats.py`: builds a 4 subject BID batch with `Timecourses.from_arrays` and a protocol, writes `events.csv` with `to_events`, reads it back with `from_events`, runs `nca` and prints `result.intervals()` and the steady state table, plots `plot_intervals(result, "interval_ctrough")` to `formats.png`; also reads `tests/data/formats/adnca.csv` (path relative to the repository root via `Path(__file__).parents[1]`) with `from_adnca` and prints the protocols.

`docs/formats.md` structure: Concepts (what an event record is, one row per event; the two-table and the ADaM layouts), a table per format with the columns and how they map, API snippets, "What is not read" (CMT, modelled rates), References.

`docs/nca.md` "Multiple dosing": the interval definition, the per-interval parameters table (`interval_*`), the steady state parameters of the last interval, the reference dose rule for the point parameters, `tau`, `accumulation_ratio` predicted vs observed, the `INCOMPLETE_INTERVAL` flag, an API snippet with `nca_single(tc, NCAOptions(auc_method=AUCMethod.LOG))` on a protocol curve and `result.intervals()`.

- [ ] **Steps:** write the pages and the example, register the example, `uv run zensical build --clean --strict` clean, `uv run python scripts/llms_txt.py`, `uv run pytest tests/examples -q -W error`, view `formats.png`, commit:

```bash
git add docs zensical.toml examples tests/examples CLAUDE.md README.md
git commit -q -m "Document the dosing protocols, the multiple dosing analysis and the data formats

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 7: Gate and pull request for review

- [ ] `uv run ruff check && uv run ruff format --check && uvx ty check && uv run pytest -q -W error && uv run tox run-parallel && uv run zensical build --clean --strict`
- [ ] Push and open the pull request WITHOUT auto-merge:

```bash
git push -u origin dosing
gh pr create --base develop --title "Dosing protocols, multiple dosing analysis and exchange formats" --body-file - <<'EOF'
## Summary

Prototype for review (design: `docs/superpowers/specs/2026-09-15-dosing-protocols-and-exchange-formats-design.md`): a `Timecourse` carries a dosing protocol (`Dosing`: amounts, times, durations, one unit and route; `dose=` keeps working), the batch stores it over a `dose` dimension, the NCA computes the parameters of every dosing interval (`interval_*` over an `interval` dimension, `result.intervals()`), the steady state parameters of the last interval and the observed accumulation from the protocol (`NCAOptions.tau` replaces `regimen`), for concentration and effect curves; the point parameters of a multiple dose curve are computed from the last dose on. Readers for the NONMEM/Monolix event format (`from_events`, `ADDL`/`II`/`SS`, `RATE`/`TINF`), the PKNCA tables (`from_pknca`) and CDISC ADNCA (`from_adnca`), a writer `to_events`; figures with dose lines, the analysed interval and `plot_intervals`; documentation (`formats.md`, multiple dosing in `nca.md`/`pd.md`) and two examples.

Decisions taken without asking are listed in the decisions table of the design document; the maintainer's review decides what stays.

## Checklist

- [x] the pull request targets `develop`
- [x] tests were added or updated for the change
- [x] `tox run-parallel` passes locally (tests and `ty`)
- [x] `ruff check` and `ruff format` are clean, e.g., via `pre-commit run --all-files`
- [x] public functions and classes have type annotations and a docstring
- [x] user visible changes are in `docs/` (release notes are written at the release)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z
EOF
```
No `gh pr merge`; report the PR URL to the maintainer.

## Deviations from the spec

- none intended; the spec was written for this plan. Rulings during execution go into the ledger and the final report.
