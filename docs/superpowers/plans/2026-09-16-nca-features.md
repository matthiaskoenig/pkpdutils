# NCA feature round implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the 37 suggestions of the tool survey as eight pull requests into `develop`, in the order and with the decisions of the spec.

**Architecture:** New modules `nca/report.py`, `nca/tss.py`, `nca/bioavailability.py`, `nca/urine.py`, `nca/sparse.py`, `cdisc.py`, `stats/power.py`, `report.py`; the rest extends `nca/options.py`, `nca/nca.py`, `nca/steady_state.py`, `nca/terminal.py`, `nca/result.py`, `result.py`, `stats/bioequivalence.py`, `stats/tests.py`, `io.py`, `timecourse.py`, `plot/nca.py`, `plot/timecourse.py`.

**Tech Stack:** unchanged (numpy, scipy, pandas, xarray, pint, pydantic, matplotlib, rich).

**Spec:** `docs/superpowers/specs/2026-09-16-nca-features-design.md` (the survey `docs/superpowers/specs/2026-09-16-nca-tools-feature-survey.md` is its argument).

## Global Constraints

- python 3.13 and 3.14; runtime dependencies unchanged
- default results of an existing analysis do not change; new rules are off by default; variables are added, never renamed
- every new variable: `PARAMETER_UNITS` entry, glossary row, docstring with formula (`r"""`, single backslashes) and citation key of `docs/references.md`
- ty `error-on-warning = true`: zero diagnostics; rule-specific ignores only; google docstrings on everything in `src`
- library code logs, never prints, never `plt.show()`
- `uv run ruff check && uv run ruff format --check` (no path arguments), `uvx ty check`, `uv run pytest -q -W error`, `uv run zensical build --clean --strict`, `uv run tox run-parallel` green before every PR; `uv run python scripts/render_examples.py` leaves `docs/images` reproducible
- `tests/nca/test_reference.py` unchanged and passing; every feature has a closed-form or reference test; every docs snippet runs (`tests/docs`), pasted outputs from real runs
- markdown without hard line wraps; never the em dash; commit messages carry NO `Co-Authored-By` and end with `Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z`
- one branch per PR (`features-1` to `features-8`), each off the merged `develop` of the previous; PRs squash-merged with auto-merge; the spec's "Breaking changes" list is appended when a task breaks something

---

### Task F1: parameters and BLQ rules

**Files:** `src/pkpdutils/nca/options.py`, `nca/nca.py`, `nca/auc.py`, `nca/result.py`, `nca/uncertainty.py` (discrete parameters), `timecourse.py` (`lloq`), `io.py` (per-sample lloq into the batch), `docs/nca.md`, `docs/glossary.md`, `tests/nca/*`
**Produces:** spec section F1: `tlag`, `auc_all`, `aumc_all`, `clast_pred`, `auc_back_extrap_fraction`, `aumc_back_extrap_fraction`, the `_dn` variables and `NCAResult.dose_normalized`, `BLQRules`/`BLQAction` with the presets, `Timecourse.lloq` and the per-sample LLOQ, `C0Method.NONE`, the `c0_method` variable and the documented fallback chain. Commit: `Add the missing NCA parameters, the BLQ rules and the per-sample LLOQ`.

### Task F2: acceptance, exclusions, partial AUCs, windows, M13A tables, carryover

**Files:** `nca/options.py`, `nca/nca.py`, `nca/terminal.py`, `nca/result.py`, `result.py`, new `nca/report.py`, `stats/bioequivalence.py`, `stats/ratio.py`, `stats/ddi.py`, `stats/sample.py`, `docs/nca.md`, `docs/bioequivalence.md`, `docs/glossary.md`, tests
**Produces:** spec section F2. Commit: `Add the acceptance criteria, the exclusions, the named partial areas and the tables of a regulatory report`.

### Task F3: validation

**Files:** `tests/data/validation/*`, `tests/nca/test_validation.py`, `scripts/validation.py`, `docs/validation.md`, `zensical.toml`
**Produces:** spec section F3. Commit: `Validate the analysis against PKNCA on the theophylline and indomethacin datasets`.

### Task F4: steady state completion and bioavailability

**Files:** `nca/steady_state.py`, `nca/nca.py`, new `nca/tss.py`, new `nca/bioavailability.py`, `nca/__init__.py`, `__init__.py`, `docs/nca.md`, `docs/glossary.md`, tests
**Produces:** spec section F4. Commit: `Complete the steady state parameters, the time to steady state and the bioavailability`.

### Task F5: urine and sparse sampling

**Files:** new `nca/urine.py`, new `nca/sparse.py`, `nca/nca.py` (`PARAMETER_UNITS` placeholders), `plot/nca.py`, `plot/__init__.py`, new `examples/urine.py`, `examples/sparse.py`, `tests/examples/test_examples.py`, `docs/urine.md`, `docs/sparse.md`, `docs/gallery.md`, `zensical.toml`, `CLAUDE.md`, tests
**Produces:** spec section F5. Commit: `Analyse urinary excretion and sparse sampling designs`.

### Task F6: analytes, CDISC map, unit conversion, writers, routes

**Files:** `timecourse.py`, `io.py`, new `cdisc.py`, `src/pkpdutils/data/pkparmcd.csv`, `result.py`, `nca/options.py`, `nca/nca.py`, `units.py`, `docs/formats.md`, `docs/units.md`, `docs/glossary.md`, `pyproject.toml` (package data), tests
**Produces:** spec section F6. Commit: `Carry analytes and routes per sample, map the parameters to CDISC and write every format`.

### Task F7: bioequivalence completion, tmax, power, report

**Files:** `stats/tests.py`, `stats/bioequivalence.py`, new `stats/power.py`, new `report.py`, new `examples/report.py`, `docs/bioequivalence.md`, `docs/reporting.md`, `docs/statistics.md`, `zensical.toml`, tests
**Produces:** spec section F7. Commit: `Add the replicate designs, the reference scaled limits, the tmax test, the power and the study report`.

### Task F8: figures

**Files:** `plot/nca.py`, `plot/timecourse.py`, `plot/style.py`, `plot/__init__.py`, `nca/terminal.py`, `nca/options.py`, `docs/plotting.md`, `docs/gallery.md`, examples, tests
**Produces:** spec section F8. Commit: `Draw the terminal window diagnostic, the partial areas and the figure pair of a study`.

Every task ends with the gate of the Global Constraints, the PR (`gh pr create --base develop`, body listing the features and the breaking changes, `gh pr merge --squash --auto --delete-branch`) and waits for the merge before the next branch.
