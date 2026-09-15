# pkpdutils Release 1.0.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release `pkpdutils` 1.0.0 from `develop` (phase 7 of the spec): release notes with the breaking changes, the features, the known gaps and a migration note from `pkdb_analysis`, a warning-free strict documentation build, the version bump, the pull request, the one-time repair of `main`, the tag that publishes to PyPI and creates the GitHub release, the installation test, the Zenodo citation update and the start of the next development cycle.

**Architecture:** No library code changes. The release follows `docs/development.md` ("Release"): a branch `release/1.0.0`, `release-notes/1.0.0.md`, `uvx bump-my-version bump dev` (`1.0.0.dev0` -> `1.0.0`, commits without a tag), pull request into `develop`, tag `1.0.0` on the merged commit, which runs the `release` and `sync-main` jobs of `.github/workflows/ci-cd.yml`. Two repository facts found while planning shape the tasks: the PyPI project `pkpdutils` does not exist yet (`https://pypi.org/simple/pkpdutils/` is 404), so a pending trusted publisher must be created on PyPI before the tag; and `origin/main` (56af208, a merge commit of the old PR #56) is not an ancestor of `develop`, so the `sync-main` fast-forward would fail: `main` is reset once, by the maintainer, with the `main` ruleset temporarily disabled. The 39 "page does not exist" warnings of the docs build come from broken nested code fences in the plan files under `docs/superpowers/`; they are fixed and the build becomes strict.

**Tech Stack:** uv, bump-my-version, gh CLI (authenticated as the maintainer, admin on the repository), zensical, GitHub Actions (`ci-cd.yml`: test matrix, `release` with trusted publishing to the environment `pypi`, `sync-main`), PyPI trusted publishing, Zenodo GitHub integration (the release webhook is active on the repository).

**Spec:** `docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md`, section "6. Repository, tooling, documentation, tests, release" ("Release") and phase 7 of "Implementation phases". The release note items come from the "Deviations from the spec" section of `docs/superpowers/plans/2026-09-15-pkpdutils-statistics.md` and from the rulings recorded in the project memory for the NCA, uncertainty and fitting plans (listed verbatim in Task 3).

## Global Constraints

- no change of library behaviour in this plan; `src/` is touched only by the version bump
- `develop` accepts only pull requests with the checks `tests`, `ruff`, `ty`, `docs`; work happens on `release/1.0.0` (and `release/1.0.0-citation` for the post-release step)
- the tag is created on `develop` after the merge, never on the release branch; a tag cannot be deleted or moved (`tags.json` ruleset), so every check runs before the tag
- outward actions that need the maintainer's explicit go before they run: creating the pending publisher on PyPI (web interface, maintainer only), closing the old PK-DB issues, the one-time force push to `main` with the ruleset disabled, pushing the tag. The executor stops and asks at those steps; everything else runs without asking
- text rules: never the em dash character (U+2014), use "-"; commit messages carry NO `Co-Authored-By` line and end with the single line `Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z` (the version bump commit is made by bump-my-version with its own message, that is fine); markdown has no hard line wraps
- `uv run pytest -q -W error`, `uv run ruff check`, `uv run ruff format --check`, `uvx ty check`, `uv run tox run-parallel` and `uv run zensical build --clean --strict` must pass before the pull request
- commands run from the repository root with `uv run`

## Facts this plan relies on (verified 2026-09-15)

- `src/pkpdutils/__init__.py` has `__version__ = "1.0.0.dev0"`, `.bumpversion.toml` `current_version = "1.0.0.dev0"`, `uvx bump-my-version bump dev --dry-run` reports `Bump version: 1.0.0.dev0 → 1.0.0` and updates `src/pkpdutils/__init__.py`, `CITATION.cff`, `.bumpversion.toml`, no tag
- `.github/workflows/ci-cd.yml`: on a tag the `release` job (environment `pypi`, `pypa/gh-action-pypi-publish`) publishes and `softprops/action-gh-release` creates the GitHub release with `body_path: release-notes/<tag>.md`; `sync-main` runs `git push origin HEAD:main` without `--force`
- `.github/rulesets/main.json`: rules `deletion` and `non_fast_forward`, no bypass actors; `.github/rulesets/apply.sh` re-applies the rulesets idempotently
- `git rev-list --count origin/develop..origin/main` is 1 (the merge commit 56af208 with the parents fdb22c7 and c4584e8), `origin/main..origin/develop` is 67
- repository webhooks: Zenodo (`release` events, active), codecov (active, unused since the rewrite), Travis (inactive)
- open GitHub issues are all from the PK-DB era: #54 mypy, #50 robust PK calculation, #49 substance information in PKData, #41 environment variables, #39 multi-dimensional pk calculation, #34 documentation notebooks, #28 interactive plot reports, #27 excel table headers, #24 google spreadsheet API
- `uv run zensical build --clean` prints 39 `Warning: page does not exist`, all located in `superpowers/plans/2026-09-14-pkpdutils-fitting.md`; `zensical build --strict` exists ("abort the build on warnings")
- the old API being replaced: `pkdb_analysis.pk.pharmacokinetics.TimecoursePK(time, concentration, dose, ureg, intervention_time, substance)` with `.pk` (`PKParameters`: `compound, auc, aucinf, tmax, cmax, tmaxhalf, cmaxhalf, kel, thalf, slope, intercept, r_value, p_value, std_err, max_idx, dose, vd, vdss, cl`), `.info()`, `.figure()`; `tests/nca/test_reference.py` reproduces it with `AUCMethod.LINEAR` and `TerminalMethod.ALL_AFTER_TMAX` (check the exact option names there before writing the migration note)

## File structure

| file | change |
| --- | --- |
| `docs/superpowers/plans/2026-09-14-pkpdutils-fitting.md`, `2026-09-14-pkpdutils-nca.md`, `2026-09-14-pkpdutils-scaffold-and-data-model.md`, `2026-09-14-pkpdutils-uncertainty.md`, `2026-09-15-pkpdutils-statistics.md` | outer fences around embedded markdown become four backticks so nested fences do not break the page |
| `.github/workflows/docs.yml`, `docs/development.md`, `CLAUDE.md` | the docs build runs with `--strict` |
| `docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md` | wording of the exponential ordering corrected |
| `.zenodo.json` | affiliation aligned with `CITATION.cff` |
| `release-notes/1.0.0.md` | the release notes |
| `src/pkpdutils/__init__.py`, `CITATION.cff`, `.bumpversion.toml` | version bump (by bump-my-version) |
| `docs/development.md` | the one-time note on the reset of `main` |
| after the release: `CITATION.cff`, `README.md`, `docs/index.md` | date and version DOI of 1.0.0; next development version |

---

### Task 1: Branch, nested fences in the plan files, strict documentation build

**Files:**
- Modify: the five plan files under `docs/superpowers/plans/`, `.github/workflows/docs.yml:42`, `docs/development.md` (Documentation section), `CLAUDE.md` (docs commands)

- [ ] **Step 1: Branch**

```bash
git switch develop && git pull -q && git switch -c release/1.0.0
uv run zensical build --clean 2>&1 | grep -c "page does not exist"
```
Expected: `39`.

- [ ] **Step 2: Find the broken fences**

A plan embeds documentation pages as `` ```markdown `` blocks which themselves contain `` ```python `` or `` ```bash `` blocks; the inner opening fence closes the outer one and the rest of the embedded page is rendered as markdown, with relative links that do not resolve. List every outer fence:

```bash
grep -n '^```markdown' docs/superpowers/plans/*.md
```

For each hit, look at the block it opens: if the block contains another line starting with three backticks before its closing fence, the outer fence pair must become four backticks (`` ````markdown `` ... `` ```` ``). A block without inner fences stays as it is. Do the edit by hand per block (the closing fence is the next line that is exactly three backticks after the last inner block); do not use a global replace.

- [ ] **Step 3: Build strict**

```bash
uv run zensical build --clean --strict 2>&1 | tail -3
```
Expected: the build finishes, no `Warning:` line. If a warning remains, it names the file and line: fix that fence (or the link, if it is a real broken link in a page under `docs/` outside `superpowers/`) and rebuild until clean.

- [ ] **Step 4: Make the build strict everywhere**

In `.github/workflows/docs.yml` change the build line to `run: uv run zensical build --clean --strict`. In `docs/development.md`, the Documentation section: the two `uv run zensical build --clean` commands become `uv run zensical build --clean --strict` and one sentence follows the first: "The build is strict: a broken link or a missing page fails it, in continuous integration as well." In `CLAUDE.md` the docs command becomes `uv run zensical build --clean --strict`. Run `uv run python scripts/llms_txt.py` once to see it still works on the strict build's output.

- [ ] **Step 5: Commit**

```bash
git add docs .github/workflows/docs.yml CLAUDE.md
git commit -q -m "Fix the nested code fences of the plans and make the documentation build strict

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 2: Spec wording, Zenodo metadata

**Files:**
- Modify: `docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md:175`, `.zenodo.json`

- [ ] **Step 1: The exponential ordering in the spec**

Line 175 of the spec says `λ sorted ascending`. The implementation orders the phases by decreasing rate (`k1 > k2 > k3`, the fastest phase first, `lambda_z = min k_i`), see the `BiExp`/`TriExp` docstrings in `src/pkpdutils/fit/models_exponential.py` (confirm by reading them). Replace `λ sorted ascending` by `λ sorted descending (the fastest phase first, as implemented in 1.0.0; the spec originally said ascending)`.

- [ ] **Step 2: Zenodo affiliation**

`.zenodo.json` gives the affiliation `Humboldt-University Berlin, Institute for Theoretical Biology, Berlin`; `CITATION.cff` gives `Humboldt-University Berlin, Faculty of Life Science, Institute for Biology, Berlin; University Lübeck; University Hospital Schleswig-Holstein, Campus Lübeck, First Department of Medicine, Germany`. Set the `.zenodo.json` affiliation to the `CITATION.cff` string. Validate: `uv run python -c "import json; json.load(open('.zenodo.json'))"`.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs .zenodo.json
git commit -q -m "Correct the exponential ordering in the spec and align the Zenodo affiliation

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 3: Release notes

**Files:**
- Create: `release-notes/1.0.0.md`

- [ ] **Step 1: Check the names used in the migration table**

```bash
grep -n "AUCMethod\|TerminalMethod\|TerminalPhase(" tests/nca/test_reference.py | head -5
grep -n "^class AUCMethod\|^class TerminalMethod" -A 8 src/pkpdutils/nca/options.py
uv run python -c "from pkpdutils import nca_single, Timecourse, Dose, Route; from pkpdutils.nca import NCAOptions; print(sorted(nca_single(Timecourse(time=[1,2,4,8,12,24], value=[8,6.5,4.2,1.8,0.8,0.15], time_unit='hr', unit='mg/l', dose=Dose(amount=100, unit='mg', route=Route.IV_BOLUS))).parameters))"
```
Use the printed enumeration members and parameter names in the table below; correct the table where it differs.

- [ ] **Step 2: Write `release-notes/1.0.0.md`**

````markdown
# Release notes for pkpdutils 1.0.0

`pkpdutils` 1.0.0 is a rewrite of `pkdb-analysis` as a library for the pharmacokinetic and pharmacodynamic analysis of timecourses and parameters, without any dependency on PK-DB. The package, the repository and the documentation carry the new name: [pypi.org/project/pkpdutils](https://pypi.org/project/pkpdutils/), [github.com/matthiaskoenig/pkpdutils](https://github.com/matthiaskoenig/pkpdutils), [matthiaskoenig.github.io/pkpdutils](https://matthiaskoenig.github.io/pkpdutils). `pkdb-analysis` stays available on PyPI at 0.3.1 and receives no further releases.

## Breaking changes

- The package is `pkpdutils`; `pkdb_analysis` is gone. Everything PK-DB specific was removed: the client (`PKData`, `PKFilter`, queries, the environment variables), the reports (LaTeX, Jekyll, interactive plots, tables, spreadsheets), the Gaussian process kernels, the circos plots and the caffeine utilities. PK-DB access is the job of the PK-DB tooling, not of this library.
- The pharmacokinetic analysis moved from `pkdb_analysis.pk.pharmacokinetics.TimecoursePK` to `pkpdutils.nca` on a new data model (`Timecourse`, `Timecourses`, `Dose`, `Route`, `DosingRegimen`), with new parameter names and new default methods; see the migration note below.
- Python 3.13 or newer is required (3.13 and 3.14 are tested on Linux, macOS and Windows). The dependencies are `numpy`, `scipy`, `pandas`, `xarray`, `pint`, `pydantic`, `matplotlib` and `rich`; `requests`, `depinfo`, `coloredlogs`, `openpyxl`, `XlsxWriter`, `pyyaml`, `gspread-pandas`, `altair`, `seaborn`, `scikit-learn` and `IPython` are no longer needed.
- The library does not configure logging and does not print; `pkpdutils.log.enable_rich_logging()` is the opt-in for scripts.

## Features

- **Data model** (`pkpdutils.timecourse`): `Timecourse` for one curve with units, uncertainties (`sd`, `se`, `n`), a `Dose` with a `Route` and metadata; `Timecourses` as an `xarray.Dataset` over a `time` dimension and any sample dimensions, built from timecourses, data frames, arrays or a simulation dataset; `DosingRegimen` for steady state.
- **Units** (`pkpdutils.units`): one pint registry, quantities at the boundaries, result units derived from the units of the input.
- **Non-compartmental analysis** (`pkpdutils.nca`): exposure, peak, terminal phase, clearance and volume parameters of concentration curves and the descriptive parameters of effect curves, for iv bolus, iv infusion and extravascular dosing, single dose and steady state (with superposition), vectorized over a batch with an optional process pool; the terminal phase by best adjusted R², last n points, all points after tmax or a manual window; linear, linear-up/log-down and log trapezoids; integer flags per sample; results as `NCAResult`, an `xarray.Dataset` with `attrs["units"]` per variable.
- **Uncertainty** (`pkpdutils.nca.uncertainty`): parametric bootstrap and delta method for group timecourses (mean with SD or SE and n), `summarize` over the individuals with the same variables, partial areas.
- **Curve fitting** (`pkpdutils.fit`): mono-, bi- and tri-exponential, Bateman, Emax, sigmoid Emax, Imax, linear, log-linear, power and allometric models; least squares in a scaled parameter space from Latin hypercube starts, standard errors from the Jacobian, t intervals, delta method for derived parameters, residual bootstrap, AIC/AICc/BIC and Akaike weights (`compare_models`), the confidence interval criterion of dose proportionality (`proportionality_test`); `fit_timecourse`, `fit_timecourses` and `fit_table`.
- **Statistics on parameters** (`pkpdutils.stats`): `ParameterSample` from a result or from published summary statistics; `compare` (Student, Welch and paired t, Mann-Whitney, Wilcoxon, permutation, effect sizes) and `multiple_comparison` (Holm, Bonferroni, Benjamini-Hochberg); the geometric mean ratio (`ratio`); average bioequivalence by the two one-sided tests for parallel, paired and 2x2 crossover designs (`bioequivalence`, `tost`); the FDA/EMA classification of drug-drug interactions (`ddi_classification`, `substrate_sensitivity`); fixed effect and DerSimonian-Laird random effects meta-analysis with heterogeneity statistics (`meta_analysis`).
- **Figures** (`pkpdutils.plot`): timecourses, the NCA diagnostic figure and grid, fits with residuals, goodness of fit, Bland-Altman, dose proportionality, parameter distributions, ratio plots against bioequivalence limits or interaction thresholds, forest plots. Functions return the `Figure` and never show it.
- **Documentation**: a user guide per topic with concepts, math, parameter tables, API examples and references, the API reference from the docstrings, runnable examples in `examples/`, and the `llms.txt` files for agents.
- **Tooling**: uv, hatchling, ruff, ty, tox, pre-commit, bump-my-version, Zensical; pull requests into `develop` with the checks `tests`, `ruff`, `ty` and `docs`; releases by tag with trusted publishing to PyPI.

## Known gaps and decisions

- The parameter aliases `kel` and `auc_inf` of the design document are not provided; the names are `lambda_z` and `auc_inf_obs` / `auc_inf_pred`.
- `cmax_half` and `tmax_half` are computed for extravascular dosing only; `exclude_cmax=False` leaves the start of the terminal window unrestricted; `cmax_ss` (the maximum within one dosing interval) drives fluctuation and swing; a bolus with pre-dose samples interpolates the value at the dose time for `auc_tau`.
- A `Timecourses` batch has one route; batches with mixed routes are analysed separately.
- Fitted parameters are reported in the raw units of the data (no normalization of volumes and clearances); the information criteria count the residual variance as an estimated parameter (`K = k + 1`); the exponential phases are ordered by decreasing rate; the residual bootstrap centers and inflates the residuals and falls back to the Jacobian intervals with `FitFlag.BOOTSTRAP_FALLBACK` when fewer than two replicates converge; `plot_fit` draws no bootstrap band.
- `compare` reports `ci_low`/`ci_high` and the effect sizes with every test; the rank tests report the difference or ratio of the medians without an interval; samples of one value or without variance give NaN statistics rather than an error.
- `bioequivalence` takes two results and a sample dimension; the 2x2 crossover is the period-difference analysis of Chow and Liu, which equals the ANOVA with sequence, period and subject effects, and is verified against an ordinary least squares fit with subject effects rather than against a textbook data set.
- `ddi_classification` classifies on the AUC ratio and reports the Cmax ratio; the EMA thresholds equal the FDA ones.
- The meta-analysis pools `Study` objects (`meta_analysis_by` groups them by category); the variance of Hedges' g is `J² var(d)`; the regression reference is the BCG example of the R package metafor.
- Compartmental and ODE models, population modelling, covariate model building and report generation are out of scope.

## Migration from pkdb-analysis

`TimecoursePK` becomes a `Timecourse` and a call of `nca_single`; a batch of curves is a `Timecourses` and a call of `nca`:

```python
from pkpdutils import Dose, Route, Timecourse, nca_single
from pkpdutils.plot import plot_nca

tc = Timecourse(
    time=time, value=concentration, time_unit="hr", unit="mg/l",
    dose=Dose(amount=100, unit="mg", route=Route.ORAL, time=0.0),
    substance="caffeine",
)
result = nca_single(tc)
result.to_quantities()["auc_inf_obs"]  # pint quantity
result.to_dataframe()                  # one row with every parameter
plot_nca(tc, result)                   # the diagnostic figure of TimecoursePK.figure()
```

| `TimecoursePK.pk` (0.3.1) | `NCAResult` (1.0.0) |
| --- | --- |
| `auc` | `auc_last` |
| `aucinf` | `auc_inf_obs` (observed last value) and `auc_inf_pred` (predicted) |
| `tmax`, `cmax` | `tmax`, `cmax` |
| `tmaxhalf`, `cmaxhalf` | `tmax_half`, `cmax_half` (extravascular dosing) |
| `kel` | `lambda_z` |
| `thalf` | `thalf` |
| `slope`, `intercept`, `r_value`, `std_err` | `lambda_z`, `lambda_z_intercept`, `lambda_z_r2`, `lambda_z_stderr` (`lambda_z_r2_adj`, `lambda_z_n_points`, `lambda_z_t_first` are new) |
| `p_value`, `max_idx` | dropped |
| `vd` | `vz` (iv) or `vz_f` (extravascular) |
| `vdss` | `vss` |
| `cl` | `cl` (iv) or `cl_f` (extravascular) |
| `dose` | `dose_amount` of the batch; `auc_inf_dn`, `cmax_dn` are the dose normalized values |
| `info()` | `to_dataframe()`, `to_quantities()`, `flags()` |
| `figure()` | `plot_nca(timecourse, result)` |

The defaults changed: the terminal phase is chosen by the best adjusted R² of at least three points after `cmax`, and the area uses the linear-up/log-down trapezoid. The numbers of 0.3.1 are reproduced with

```python
from pkpdutils.nca import NCAOptions, TerminalPhase
from pkpdutils.nca.options import AUCMethod, TerminalMethod

options = NCAOptions(auc_method=AUCMethod.LINEAR, terminal=TerminalPhase(method=TerminalMethod.ALL_AFTER_TMAX))
result = nca_single(tc, options)
```

which is what `tests/nca/test_reference.py` checks against the frozen results of 0.3.1.

The effect analysis of `pkdb_analysis.reports.effect_analysis` (`OutputPair`, `fixed_effect`, `random_effects`) is `pkpdutils.stats.meta` (`effect_size`, `fixed_effect`, `random_effects`, `meta_analysis`); the pairs are `Study(label, control, treatment)` objects built from `ParameterSample` values or summary statistics.

Your pkpdutils team
````

Correct the option and parameter names against Step 1 (the import path of `AUCMethod`/`TerminalMethod`, the exact `TerminalPhase` keyword, whether `lambda_z_t_first` and `dose_amount` exist on an `NCAResult`; drop a row whose name does not exist). Check the "Known gaps" bullets against the memory rulings quoted in the spec section of this plan header: every ruling is represented once.

- [ ] **Step 3: Check the notes**

```bash
grep -cP '\x{2014}' release-notes/1.0.0.md        # 0, no em dash
uv run python - <<'EOF'
import re, pathlib
text = pathlib.Path("release-notes/1.0.0.md").read_text()
names = set(re.findall(r"`([a-z_]+)`", text))
import pkpdutils, pkpdutils.stats, pkpdutils.plot, pkpdutils.nca, pkpdutils.fit
public = set(dir(pkpdutils)) | set(dir(pkpdutils.stats)) | set(dir(pkpdutils.plot)) | set(dir(pkpdutils.nca)) | set(dir(pkpdutils.fit))
print(sorted(n for n in names if n in public))
EOF
```
The second command lists the backticked names which are public API; read the printed list against the text once (a function named in the notes and not in the list is either a parameter name, which is fine, or a typo, which is not).

- [ ] **Step 4: Commit**

```bash
git add release-notes/1.0.0.md
git commit -q -m "Add the release notes of 1.0.0

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

---

### Task 4: Version bump and the full verification

**Files:**
- Modify (by bump-my-version): `src/pkpdutils/__init__.py`, `CITATION.cff`, `.bumpversion.toml`

- [ ] **Step 1: Bump**

```bash
uvx bump-my-version bump dev --dry-run -vv 2>&1 | grep -E "Would commit|Would not tag"
uvx bump-my-version bump dev
git log --oneline -1                      # Bump version: 1.0.0.dev0 → 1.0.0
grep -n "__version__" src/pkpdutils/__init__.py; grep -n "^version" CITATION.cff; grep -n "^current_version" .bumpversion.toml
```
Expected: all three say `1.0.0`. `tests/test_package.py::test_version` asserts the old string: update it to `"1.0.0"` and commit that change separately:

```bash
sed -i 's/"1.0.0.dev0"/"1.0.0"/' tests/test_package.py
git add tests/test_package.py
git commit -q -m "Expect the release version in the package test

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
```

- [ ] **Step 2: Every check**

```bash
uv run ruff check && uv run ruff format --check
uvx ty check
uv run pytest -q -W error
uv run tox run-parallel
uv run zensical build --clean --strict && uv run python scripts/llms_txt.py
```
Expected: all green, `py3.13`, `py3.14`, `ty` OK; the pytest count matches the tox count.

- [ ] **Step 3: Build and inspect the distribution**

```bash
rm -rf dist && uv build
ls dist                                            # pkpdutils-1.0.0.tar.gz  pkpdutils-1.0.0-py3-none-any.whl
uvx twine check dist/*                             # PASSED for both (README renders on PyPI)
uv run --isolated --no-project --with dist/pkpdutils-1.0.0-py3-none-any.whl python -c "import pkpdutils; print(pkpdutils.__version__); from pkpdutils import nca_single, Timecourse, Dose, Route; r = nca_single(Timecourse(time=[1,2,4,8,12,24], value=[8,6.5,4.2,1.8,0.8,0.15], time_unit='hr', unit='mg/l', dose=Dose(amount=100, unit='mg', route=Route.IV_BOLUS))); print(r.to_quantities()['auc_inf_obs'])"
unzip -l dist/pkpdutils-1.0.0-py3-none-any.whl | grep -c "pkpdutils/" ; unzip -l dist/pkpdutils-1.0.0-py3-none-any.whl | grep -E "examples|tests|superpowers" || echo "no examples/tests/plans in the wheel"
```
Expected: `1.0.0`, an AUC quantity in `milligram * hour / liter`, and no examples, tests or plans inside the wheel. `dist/` is gitignored (check `git status --short` is clean apart from nothing).

---

### Task 5: Pull request

- [ ] **Step 1: Push and open**

```bash
git push -u origin release/1.0.0
gh pr create --base develop --title "Release 1.0.0" --body-file - <<'EOF'
## Summary

Release of pkpdutils 1.0.0 (phase 7 of `docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md`): the release notes `release-notes/1.0.0.md` with the breaking changes, the features, the known gaps and the migration note from `pkdb-analysis`, the version bump to 1.0.0, the strict documentation build (the nested code fences of the plan files are fixed), the corrected exponential ordering in the spec and the Zenodo affiliation. The tag `1.0.0` is created on `develop` after the merge.

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
gh pr merge --squash --auto --delete-branch
```

- [ ] **Step 2: Wait and sync**

```bash
until [ "$(gh pr view --json state -q .state)" = "MERGED" ] || [ "$(gh pr checks --json bucket -q '[.[] | select(.bucket=="fail")] | length')" != "0" ]; do sleep 30; done; gh pr checks; git switch develop && git pull
grep -n "__version__" src/pkpdutils/__init__.py    # 1.0.0 on develop
```

---

### Task 6: Follow-up issues and the old PK-DB issues

Outward actions on GitHub. Creating the follow-up issues runs without asking (they document known gaps of the released code); closing the nine old issues needs the maintainer's go: show the list and the comment text and wait for "yes".

- [ ] **Step 1: Follow-up issues**

Create six issues with `gh issue create --title ... --body ...` (label `enhancement` where it exists, `gh label list` shows the labels):

1. Title `compare(paired=True, test=WILCOXON) reports the effect from all finite values`. Body: "`compare` pairs the samples by label before dropping missing values (`paired_values`), but `_median_effect` in `src/pkpdutils/stats/tests.py` takes the medians of `a.finite_values` and `b.finite_values`, so with a missing value on one side the effect mixes different subject sets. The effect of a paired rank test should come from the surviving pairs."
2. Title `ParameterResult.sample raises a raw KeyError for an unknown indexer`. Body: "`ParameterResult.sample(name, dim, **indexers)` passes the indexers to `xarray.Dataset.sel`; an unknown key, or `dim` given as an indexer as well, surfaces as a `KeyError` from xarray instead of the `ValueError` the method raises for its other misuse. Validate the indexer keys against the sample dimensions first."
3. Title `plot_parameters: box plot uses the unmasked values under log=True`. Body: "With `log=True` the strip points are masked to positive values and a group with a non-positive value gets the arithmetic mean marker, but `ax.boxplot` still receives every finite value. The box should use the same masked values, and the regression test should also assert that an all-positive group keeps its geometric mean marker."
4. Title `Coordinate collision check of the fit does not cover the parameter and point dimensions`. Body: "`check_coordinate_collision` in `build_result` (`src/pkpdutils/fit/engine.py`) compares the batch coordinates with the data variables only; a batch coordinate named `parameter`, `parameter_` or `point` would collide with the dimensions of the fit result. Extend the check to the dimension names the result introduces."
5. Title `Fit engine warnings escape on data spanning many orders of magnitude`. Body: "Data spanning 1e-300 to 1e300 lets numpy warnings escape from the covariance matmul, the weighted residual sum, the scipy trust region and the terminal guess (`terminal_guess`) of the fit engine. The engine should either guard those steps with `np.errstate` and flag the sample, or reject such data with a clear error."
6. Title `Polish of the statistics module`. Body: "Collected small items from the review of `pkpdutils.stats`: `ParameterSample` keeps summary fields given together with `values` without using them (validate or document); `effects_from_arrays` types its inputs `Any` instead of `ArrayLike` and takes `kind`/`ci_level` positionally unlike the rest of the module; `meta_analysis` computes `heterogeneity` twice; `plot_ratio` on an empty mapping draws a degenerate axis; the glossary row `gmr, log_ratio, se_log` lists two units for three names."

Record the issue numbers in the report.

- [ ] **Step 2: Close the old PK-DB issues (ask first)**

Show the maintainer this list and wait for the go: #50 and #39 are closed as completed (comment: "Done in 1.0.0: the non-compartmental analysis is vectorized over any sample dimensions (`pkpdutils.nca.nca`) and the terminal phase is chosen by the adjusted R² over every candidate window, see https://matthiaskoenig.github.io/pkpdutils/nca/."); #54, #49, #41, #34, #28, #27, #24 are closed as not planned (comment: "Closed with the 1.0.0 rewrite: `pkpdutils` is a library for the analysis of timecourses and parameters without PK-DB access, reports or spreadsheets, see the design document `docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md`. The type checking moved from mypy to ty and is part of every pull request."). Then:

```bash
DONE="Done in 1.0.0: the non-compartmental analysis is vectorized over any sample dimensions (\`pkpdutils.nca.nca\`) and the terminal phase is chosen by the adjusted R² over every candidate window, see https://matthiaskoenig.github.io/pkpdutils/nca/."
GONE="Closed with the 1.0.0 rewrite: \`pkpdutils\` is a library for the analysis of timecourses and parameters without PK-DB access, reports or spreadsheets, see the design document \`docs/superpowers/specs/2026-09-14-pkpdutils-redesign-design.md\`. The type checking moved from mypy to ty and is part of every pull request."
for n in 50 39; do gh issue close $n --reason completed --comment "$DONE"; done
for n in 54 49 41 34 28 27 24; do gh issue close $n --reason "not planned" --comment "$GONE"; done
gh issue list --state open
```
Expected: only the six new issues are open.

---

### Task 7: The gates before the tag

Three things must be true before the tag is pushed, because a tag cannot be moved and the `release` and `sync-main` jobs run on it. The executor checks what it can, asks the maintainer for the rest, and does not proceed to Task 8 without the three confirmations.

- [ ] **Step 1: PyPI pending trusted publisher (maintainer, web interface)**

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://pypi.org/simple/pkpdutils/
```
`404` means the project does not exist yet, which is expected: trusted publishing creates it on the first upload if a *pending publisher* exists. Ask the maintainer to confirm that on https://pypi.org/manage/account/publishing/ a pending publisher exists with exactly: PyPI project name `pkpdutils`, owner `matthiaskoenig`, repository `pkpdutils`, workflow name `ci-cd.yml`, environment name `pypi`. Also that the GitHub environment `pypi` exists (`gh api repos/matthiaskoenig/pkpdutils/environments --jq '.environments[].name'` prints `pypi`). Without the pending publisher the `release` job fails at the upload and the tag is burnt.

- [ ] **Step 2: The one-time reset of `main` (maintainer's go, then the executor runs it)**

`sync-main` pushes the tagged commit to `main` without `--force`; `origin/main` (56af208) is not an ancestor of `develop`, so the push would be rejected and `main` would stay at 0.3.1. `main` carries no content that is not in `develop` (`git rev-list --count origin/develop..origin/main` is 1, the merge commit itself), so it is reset once to the release commit. The `main` ruleset forbids the non-fast-forward push and has no bypass actor; it is disabled for the push and re-applied afterwards. Show the maintainer these commands and run them on "yes":

```bash
git fetch origin
git merge-base --is-ancestor origin/main origin/develop && echo "already an ancestor, nothing to do" && exit 0
ID=$(gh api repos/matthiaskoenig/pkpdutils/rulesets --jq '.[] | select(.name=="main") | .id')
gh api -X PUT repos/matthiaskoenig/pkpdutils/rulesets/$ID --input - <<EOF
$(jq '.enforcement = "disabled"' .github/rulesets/main.json)
EOF
git push --force origin origin/develop:main
.github/rulesets/apply.sh                          # re-applies main.json with enforcement active
gh api repos/matthiaskoenig/pkpdutils/rulesets --jq '.[] | select(.name=="main") | .enforcement'   # active
git fetch origin && git merge-base --is-ancestor origin/main origin/develop && echo "main is now an ancestor of develop"
```
If `apply.sh` needs arguments or a different invocation, read it first (`cat .github/rulesets/apply.sh`) and follow it. Add to `docs/development.md`, at the end of the "Branch model" section, the sentence: "`main` was reset once to the 1.0.0 release commit, because the history of the `pkdb-analysis` releases was not an ancestor of the rewritten `develop`; since then every release fast-forwards it." (this edit goes into the post-release pull request of Task 9).

- [ ] **Step 3: Zenodo**

```bash
gh api repos/matthiaskoenig/pkpdutils/hooks --jq '.[] | select(.config.url | contains("zenodo")) | {active, events}'
```
Expected: `active: true`, events `["release"]`. Zenodo archives the GitHub release the `release` job creates and mints the version DOI under the concept DOI 10.5281/zenodo.3997539. Nothing to do unless the hook is inactive, in which case the maintainer switches the repository on at https://zenodo.org/account/settings/github/ before the tag.

---

### Task 8: Tag, release, verification

- [ ] **Step 1: Tag (ask first)**

The three gates of Task 7 are confirmed. Show the maintainer the commit and wait for the go:

```bash
git switch develop && git pull -q && git log --oneline -1 && grep -n "__version__" src/pkpdutils/__init__.py && test -f release-notes/1.0.0.md && echo ready
git tag 1.0.0
git push origin 1.0.0
```

- [ ] **Step 2: Watch the workflow**

```bash
sleep 60; RUN=$(gh run list --workflow ci-cd.yml --event push --branch 1.0.0 --limit 1 --json databaseId -q '.[0].databaseId'); echo $RUN
gh run watch $RUN --exit-status && gh run view $RUN --json jobs -q '.jobs[] | "\(.name): \(.conclusion)"'
```
Expected: every test job, `tests`, `release`, `sync-main` with `success`. If `release` fails at the PyPI upload, the tag stays and the fix is on the PyPI side (the pending publisher); a re-run of the workflow (`gh run rerun $RUN --failed`) publishes once it is fixed, no new tag is needed. If `sync-main` fails, Task 7 Step 2 was skipped: run it and re-run the failed job.

- [ ] **Step 3: Verify PyPI, the GitHub release and `main`**

```bash
curl -s https://pypi.org/pypi/pkpdutils/1.0.0/json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['info']['version'], [f['filename'] for f in d['urls']])"
cd /tmp && rm -rf pkpdutils-install && mkdir pkpdutils-install && cd pkpdutils-install && uv venv --python 3.14 -q && uv pip install -q pkpdutils==1.0.0 && uv run --no-project python -c "import pkpdutils; print(pkpdutils.__version__); from pkpdutils import Timecourse, Dose, Route, nca_single; print(nca_single(Timecourse(time=[1,2,4,8,12,24], value=[8,6.5,4.2,1.8,0.8,0.15], time_unit='hr', unit='mg/l', dose=Dose(amount=100, unit='mg', route=Route.IV_BOLUS))).to_dataframe().T)"
cd /home/mkoenig/git/pkdb_analysis
gh release view 1.0.0 --json name,tagName,isDraft,isPrerelease -q '"\(.name) \(.tagName) draft=\(.isDraft) pre=\(.isPrerelease)"'
git fetch origin && git rev-parse origin/main origin/develop | uniq -c | wc -l     # 1: main == develop
```
Expected: `1.0.0` with the wheel and the sdist, the installation prints `1.0.0` and a parameter table, the release exists and is not a draft or prerelease, `main` points at the released commit.

---

### Task 9: After the release: citation, next development version

**Files:**
- Modify: `CITATION.cff` (`date-released`, `version` by bump), `README.md` and `docs/index.md` (citation line), `docs/development.md` (the `main` note of Task 7), `src/pkpdutils/__init__.py`, `.bumpversion.toml`, `tests/test_package.py`

- [ ] **Step 1: The version DOI**

Zenodo archives the release within minutes. Poll:

```bash
until DOI=$(curl -s "https://zenodo.org/api/records?q=conceptdoi:%2210.5281/zenodo.3997539%22&sort=mostrecent&size=1" | python3 -c "import json,sys; h=json.load(sys.stdin)['hits']['hits']; print(h[0]['doi'] if h and h[0]['metadata'].get('version')=='1.0.0' else '')") && [ -n "$DOI" ]; do sleep 60; done; echo $DOI
```
If nothing appears after 30 minutes, ask the maintainer to check https://zenodo.org/account/settings/github/ and continue with the rest of this task; the DOI line is then filled in a later pull request.

- [ ] **Step 2: Citation and the next cycle**

```bash
git switch develop && git pull -q && git switch -c release/1.0.0-citation
```
In `CITATION.cff` set `date-released` to today (`date +%F`) and `doi` to the version DOI. In `README.md` and `docs/index.md` change the citation line to `> König, M. & Grzegorzewski, J. (2026). *pkpdutils: pharmacokinetic and pharmacodynamic analysis of timecourses and parameters* (Version 1.0.0) \[Computer software\]. Zenodo. https://doi.org/<version DOI>` and keep the concept DOI badge above it. Add the `main` sentence of Task 7 Step 2 to `docs/development.md`. Then open the next development cycle:

```bash
uvx bump-my-version bump minor --dry-run -vv 2>&1 | grep "Would commit"     # 1.0.0 → 1.1.0.dev0
uvx bump-my-version bump minor
sed -i 's/"1.0.0"/"1.1.0.dev0"/' tests/test_package.py
uv run pytest tests/test_package.py -q && uv run zensical build --clean --strict >/dev/null && echo ok
git add CITATION.cff README.md docs/index.md docs/development.md tests/test_package.py
git commit -q -m "Cite the 1.0.0 release and start the 1.1.0 development cycle

Claude-Session: https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
git push -u origin release/1.0.0-citation
gh pr create --base develop --title "Cite the 1.0.0 release and start the 1.1.0 development cycle" --body "Version DOI and date of the 1.0.0 release in CITATION.cff, README and the documentation index, the note on the reset of main, and the version bump to 1.1.0.dev0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017h5AZEctvrGTruqpqg4E2z"
gh pr merge --squash --auto --delete-branch
until [ "$(gh pr view --json state -q .state)" = "MERGED" ] || [ "$(gh pr checks --json bucket -q '[.[] | select(.bucket=="fail")] | length')" != "0" ]; do sleep 30; done; gh pr checks; git switch develop && git pull
```

Note: `bump minor` on `1.0.0` gives `1.1.0.dev0` only if the `dev` part serializes from `final`; the dry run shows the target. If it prints `1.1.0` instead, run `uvx bump-my-version bump --new-version 1.1.0.dev0` and use that string in the sed.

- [ ] **Step 3: Memory**

Update `/home/mkoenig/.claude/projects/-home-mkoenig-git-pkdb-analysis/memory/pkpdutils-redesign-status.md`: 1.0.0 released (date, version DOI, PR numbers), `main` reset done, follow-up issue numbers, next cycle 1.1.0.dev0; the redesign is complete.

## Deviations from the spec

- The spec's step "check the Zenodo GitHub integration follows the rename" is verified by the webhook query of Task 7; the citation update needs the version DOI and therefore a second pull request after the tag (Task 9), as `docs/development.md` step 9 says.
- The one-time reset of `main` is not in the spec; it is forced by the history of the repository and documented in `docs/development.md`.
- The docs build becomes strict (not in the spec); the plan files stay under `docs/` and are fixed rather than excluded, since Zensical has no exclude option.
