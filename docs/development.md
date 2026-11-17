# Development

Contributions are welcome. The repository is [matthiaskoenig/pkpdutils](https://github.com/matthiaskoenig/pkpdutils); development happens against the `develop` branch via pull requests.

## Branch model

Two branches are permanent:

- **`develop`** is the default branch and the branch everything is integrated into. The documentation on [matthiaskoenig.github.io/pkpdutils](https://matthiaskoenig.github.io/pkpdutils) is published from it.
- **`main`** tracks the latest published release. It is fast-forwarded to the released commit by the `sync-main` job of the `CI-CD` workflow after the package went to pypi, so `main` and the newest version on pypi always agree. Nothing is developed on `main` and nothing is merged into it by hand.

Work happens on short lived branches off `develop`, which GitHub deletes after the merge. Releases are tagged on `develop`, see [Release](#release).

`main` was reset once to the 1.0.0 release commit, because the history of the `pkdb-analysis` releases was not an ancestor of the rewritten `develop`, and the classic branch protection of `main` from that time (which required the codecov checks) was removed in favour of the rulesets; since then every release fast-forwards it.

## Pull requests

Neither branch accepts a direct push, every change goes through a pull request against `develop`. This includes the maintainer, there is no bypass.

A pull request can only be merged once the four required checks are green:

| check   | workflow      | content                                                              |
| ------- | ------------- | -------------------------------------------------------------------- |
| `tests` | `ci-cd.yml`   | the test matrix, python 3.14 on linux, macos and windows              |
| `ruff`  | `ruff.yml`    | `ruff check` and `ruff format --check`                                |
| `ty`    | `ty.yml`      | `tox r -e ty`                                                         |
| `docs`  | `docs.yml`    | the zensical build including the api reference and the agent files    |

`tests` aggregates the test matrix into a single job, so the name of the required check stays the same when the matrix changes.

Continuous integration is kept small: every workflow cancels its running build when a newer commit of the same branch or pull request arrives, uv caches the packages and the interpreters between runs, dependabot proposes its updates once a month, and the matrix tests only python 3.14. The other python versions are tested locally, see [Testing](#testing), before a pull request is opened.

Further rules of a pull request:

- conversations have to be resolved before the merge
- an approval is dismissed when new commits are pushed
- the history stays linear, i.e., a pull request is merged with squash or rebase; merge commits are disabled
- the maintainer is the code owner of the repository (`.github/CODEOWNERS`) and is requested for review on every pull request. A pull request of a contributor is therefore reviewed and merged by the maintainer, who has the only write access. The rulesets themselves do not require an approval: on a personal repository a ruleset cannot ask for an approval only from somebody else, and requiring one would block the pull requests of the maintainer, who cannot approve their own. Once a second person has write access, a ruleset requiring an approving review of a code owner can be added

[Auto-merge](https://docs.github.com/pull-requests/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/automatically-merging-a-pull-request) is enabled for the repository, so a pull request can be queued and is merged as soon as the checks pass and the required approval is there.

### Repository policies { #repository-policies }

The protection is implemented with [repository rulesets](https://docs.github.com/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets). They are part of the repository in `.github/rulesets/` instead of only living in the web interface, so a change to a policy is reviewed like any other change:

| ruleset                 | applies to | rules                                                                                                                                       |
| ----------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `develop.json`          | `develop`  | pull request required, the four checks above, resolved conversations, linear history, no force push, no deletion. **No bypass, for anybody.** |
| `main.json`             | `main`     | no force push, no deletion, no bypass. The fast-forward of the release workflow needs none, only a force push would be rejected. `main` mirrors `develop`, whose history carries merge commits from before merge commits were disabled, so `main` cannot require a linear history |
| `tags.json`             | all tags   | a tag cannot be deleted or moved, so a release tag keeps pointing at what was released                                                       |

Changing a policy means changing the json and applying it:

```bash
.github/rulesets/apply.sh
```

The script is idempotent: it updates the rulesets which exist and creates the missing ones. It also sets the merge settings of the repository, i.e., auto-merge, delete branch on merge, and squash and rebase as the only merge methods. It needs the [github cli](https://cli.github.com) authenticated as a user with admin permission on the repository.

## Setup development environment

Development needs [uv](https://docs.astral.sh/uv/) and a checkout of the repository:

```bash
git clone https://github.com/matthiaskoenig/pkpdutils.git
cd pkpdutils
```

A single sync creates the virtual environment in `.venv`, installs `pkpdutils` into it in editable mode and adds the complete tooling:

```bash
uv sync --extra dev
```

The environment is resolved from `uv.lock`, which is committed, so local development, including `uv run ty check`, and the documentation build use the same versions. The tox environments, i.e. the test matrix and the `ty` check of continuous integration, resolve from `pyproject.toml` and do not use the lock; the lower bounds in `pyproject.toml` are what a user of the library installs against. After changing a dependency in `pyproject.toml` run `uv lock`: the `documentation` workflow syncs with `uv sync --locked`, which fails when `pyproject.toml` and the lock disagree. `uv lock --upgrade` moves the lock to the newest releases. Dependabot bumps the locked dependencies as one grouped monthly pull request, with `versioning-strategy: lockfile-only`, so the lower bounds in `pyproject.toml` are never raised by it and only ever changed by hand.

The `dev` extra contains everything used below, i.e., pytest, ruff, ty, tox, pre-commit, zensical and bump-my-version, so nothing has to be installed separately. The python version is taken from `.python-version` (currently 3.14); to work against the oldest supported version instead use `uv sync --extra dev --python 3.13`, which replaces the environment.

The tools are then run either with `uv run <command>`, which uses the environment without activating it, or from the activated environment:

```bash
source .venv/bin/activate        # Linux and macOS
.venv\Scripts\activate           # Windows
```

The commands in this document are written without the `uv run` prefix; prepend it if the environment is not activated.

The last step installs the git hook:

```bash
uv run pre-commit install          # install the hook, once per checkout
uv run pre-commit run --all-files  # check the current state of the repository
```

From now on every commit is checked with ruff (lint and format) and ty, i.e., the same checks that run in continuous integration. On a commit only the changed files are looked at, `--all-files` checks the whole repository and is what a newly added hook should be tried with.

## Testing

The tests are written with pytest, tox runs them against every supported python version.

The tox environments are named after the interpreter (`py3.13` to `py3.15`, see `envlist` in `tox.ini`), a single one is run with

```bash
tox r -e py3.14
```

and the complete matrix, including the `ty` environment, in parallel with

```bash
tox run-parallel
```

This needs the interpreters to be available, which uv installs with `uv python install 3.13 3.14 3.15`. It is the complete test and is run before a pull request is opened: continuous integration runs only `py3.14`, on linux, macos and windows, as `uvx --with tox-uv tox -e py3.14 -- --durations=15`, which lists the slowest tests in the log. The snippets of the documentation and the example scripts (the markers `docs` and `examples`) start an interpreter each and dominate the time of the suite, so continuous integration runs them on linux only and deselects them on macos and windows with `-m "not docs and not examples"`; locally they always run.

To run the tests directly against the development environment use

```bash
pytest                                          # the full suite
pytest -n 0                                     # in one process, e.g. for --pdb
pytest tests/test_units.py                  # a single module
pytest tests/test_units.py::test_parse_unit  # a single test
```

The tests run in parallel, `addopts = "-n auto"` in `pyproject.toml` gives pytest-xdist one worker per core; `-n 0` on the command line runs everything in one process, which the debugger needs.

The `conftest.py` at the root of the repository selects the non-interactive matplotlib backend for the session and puts the repository on `sys.path`, so that the tests can import the examples.

`tests/examples/test_examples.py` runs the examples as `python -m examples.<module>` in a temporary working directory, so a broken example fails the test suite.

A number pinned in a test has to be reproducible, so the generator of a reference is committed next to it. `scripts/reference_t_isf.py` prints the Student t inverse survival function at 60 digits with mpmath, which is where the far-tail values of `tests/test_quantiles.py` and of the `stats.rs` tests of `pkpdutils-core` come from; mpmath is not part of the development environment and is asked for on the command line:

```bash
uv run --with mpmath python scripts/reference_t_isf.py 1e-300 3 10 2.5
```

The twin generator of the percentile tables lives in the kernel repository, `scripts/reference_percentile.py` of `pkpdutils-core`, which prints the rows of `crates/core/src/percentile.rs` from `numpy.nanpercentile`, including the rows a seeded search found where a naive interpolation differs from numpy in the last bit. Both exist because `nan_percentile` and the t interval of a summary are bit-exact requirements, and a reference generated by hand once cannot be checked again.

## Linting and formatting

Linting and formatting use [ruff](https://docs.astral.sh/ruff/):

```bash
ruff check     # lint
ruff format    # format
```

The `ruff` workflow installs the ruff version locked in `uv.lock`, so continuous integration and `uv run ruff` agree.

The docstring rules (`D`) are enforced for the package, not for `examples/` and `tests/`, which are scripts and fixtures, see `.ruff.toml`.

## Type checking

Type checking is performed with [ty](https://docs.astral.sh/ty/):

```bash
tox r -e ty
```

Or directly in the working tree:

```bash
uvx ty check
```

The configuration lives in `[tool.ty]` in `pyproject.toml`. Warnings are treated as errors, so the codebase is kept free of diagnostics. Suppress an unavoidable diagnostic with a rule specific `# ty: ignore[rule-name]` rather than a blanket comment.

## Benchmarks

`scripts/benchmark.py` times the hot paths of the package: the analysis of a small, a large and a multiple dose batch, the bootstrap and the delta method, a batch fit, the construction of timecourses and the iteration over a batch.

```bash
uv run python scripts/benchmark.py all                        # every case, about 35 s
uv run python scripts/benchmark.py nca-large bootstrap --repeat 5
```

The hot path cases are `nca-small`, `nca-large`, `nca-multiple`, `bootstrap`, `delta`, `fit`, `constructors`, `iterate`, `mean-ragged`, `nested-samples`, `delta-long` and `worker-churn`; `--repeat` (3 by default) is the number of timed runs after one warm-up run. The additional cases exercise distinct actual sampling times, sparse identifier combinations, long uncertainty curves, and changing worker counts. The script prints a markdown table with the size, first-call time, median warmed wall time and peak resident set size. Every case runs in a fresh interpreter, so the memory and the caches (`pkpdutils.units`) of one case do not carry into the next. The first-call time excludes imports and fixture construction; RSS measures the parent process and does not include live fit workers.

The `engine-*` cases, collectively `engines`, are the second table: one single dose analysis through both engines of the analysis, the rust kernel of `pkpdutils-core` and the python core of `pkpdutils.nca.nca`, alternating run after run in one process so that both halves are measured in the same window, over ten shapes from one curve to 10 000 curves of 200 points. The python half is reached through `pkpdutils.nca.engine._force_python`, the private switch the package documents for its tests and for this benchmark. `check_dispatch` runs before anything is timed and fails the case rather than publishing a plausible ratio for it: the dispatch has to send the batch to the kernel, the forced python path has to be forced, the differential gate must not be armed (`PKPDUTILS_COMPARE_ENGINES`, which would run the python core inside the kernel half and halve the ratio, and which the `engines` job of continuous integration sets) and the kernel itself has to accept the batch, since a refusal of `run_kernel` is answered with a warning and the python path. `all` runs the hot paths and the engine cases.

```bash
uv run python scripts/benchmark.py engines                    # the comparison of the two engines
uv run python scripts/benchmark.py --write --repeat 5         # rewrites docs/benchmark.md
uv run python scripts/benchmark.py --check                    # exit 1 if that page is stale
```

`--write` runs every engine case and pastes the table, the machine it ran on, the day and the sentences of `summary_sentences` into [Performance](benchmark.md), together with the table of the analyses the kernel does not carry, whose reasons are read from the dispatch and never written by hand. Those sentences carry the figures the prose of the page leans on (the largest and the smallest ratio, the two shapes the crossover lies between, what the kernel path costs below it and the first clear win above it), because a figure copied into a paragraph is the one staleness the check below cannot see; the prose around them is written to survive a rerun on another machine, and a `--write` which moves the table far enough to make one of its claims untrue is a reason to rewrite the paragraphs. `--check` compares the page against the cases of the script, not against the timings, which differ between machines and between runs: the cases, their size labels, the columns and that every measured cell is a number, plus the refusal table word for word. `tests/docs/test_benchmark.py` runs that check, so a case added, removed, renamed or reshaped fails the test suite until the page is rewritten; the timings themselves are only ever written by `--write`.

The numbers are machine specific, they depend on the cores, the memory and the load of the machine they were measured on: use them to compare a change against the same table taken before it on the same machine, never as an absolute performance claim.

## Parallelism

`src/pkpdutils/parallel.py` holds the worker pools of the package. `executor(kind, n_workers)` returns one lazily created executor per kind and size, shared by every call of the process and closed by an `atexit` handler, so that the start-up of a process pool - about 0.7 s, since every worker imports `pkpdutils`, numpy, scipy, xarray and pint - is paid once and not once per analysis. The process pool starts its workers with `PROCESS_START_METHOD` on every python version, the default of python 3.14: `forkserver` on Linux, `spawn` on macOS and Windows. It never forks, whatever `multiprocessing.set_start_method` says: the `fork` default of python 3.13 on Linux copies a parent whose NCA threads may hold a lock into a child which can then block forever, which python 3.13 warns about with `DeprecationWarning: This process is multi-threaded, use of fork() may lead to deadlocks in the child`. `resolve_workers(n_workers, n_rows, threshold=..., max_workers=8)` turns the option into a worker count (`None` automatic and serial below the threshold, `1` serial, anything else taken as given) and `split_rows(n_rows, n_workers, min_rows=1000, max_rows=None)` cuts the rows into about one contiguous slice per worker, never shorter than `min_rows` while there is more than one and never longer than `max_rows`.

The two analyses use different workers, because their rows cost different things:

Analyses acquire pools through `worker_pool(kind, n_workers)`. A lease protects a pool until its work completes. At most two managed pools of each kind remain cached; simultaneous active leases may temporarily exceed that bound. Idle pools are retired in order of last acquisition when a lease ends. The low-level `executor` function retains explicit lifecycle management through `evict` and `shutdown_executors`.

| analysis | workers | automatic from | why |
|---|---|---|---|
| `nca` (`run_rows`) | threads | 20 000 rows (`NCA_WORKER_THRESHOLD`) | the core is vectorized numpy and releases the GIL; no pickling and no copy of the batch, and the pool starts in half a millisecond |
| `fit` (`fit_rows`) | processes | 2 000 rows (`FIT_WORKER_THRESHOLD`) | a row is a python-heavy `scipy.optimize.least_squares` search, which only a process escapes the GIL for; the threshold is the measured break-even of the first pooled call, whose workers import the package, against the serial run |

A pooled fit needs the `if __name__ == "__main__":` guard of `multiprocessing`, and a model the workers can import rather than one defined in an interactive session, since both start methods re-import the main module in a fresh interpreter; the threads of the NCA need neither. Neither pool is used when the caller asks for `n_workers=1`. A pool that broke - a worker process killed by the operating system - is dropped and replaced by the next `executor` call, and `fit_rows` retries the batch once in the fresh pool; the pools are not re-entrant, so work running in a worker must never submit to the pool it runs in.

## Examples

The examples are runnable scripts in `examples/`, they are not part of the package. They are run as modules from the root of the repository:

```bash
python -m examples.timecourses
```

An example writes what it creates into the current working directory and never opens a window: a plotting example saves its figure to a file. `tests/examples/test_examples.py` runs the example scripts in a temporary directory, so a broken example fails the test suite. See `examples/README.md` and the [Gallery](gallery.md), which shows the figure and the core snippet of every example.

## Documentation

The documentation is built with [Zensical](https://zensical.org/), the static site generator of the Material for MkDocs authors. The sources are markdown files in `docs/`, the site is configured in `zensical.toml` in the repository root. Nothing rendered is committed: the site is built by the `documentation` workflow on every push and published to [matthiaskoenig.github.io/pkpdutils](https://matthiaskoenig.github.io/pkpdutils) from the `develop` branch.

Build the site into `site/`:

```bash
uv run zensical build --clean --strict
```

The build is strict: a broken link or a missing page fails it, in continuous integration as well.

For writing, the preview rebuilds on save:

```bash
uv run zensical serve
```

The API reference is rendered from the docstrings by [mkdocstrings](https://mkdocstrings.github.io/); a page in `docs/api/` only contains the module directive:

```markdown
# timecourse

::: pkpdutils.timecourse
```

Docstrings are therefore the place to document functions and classes, the markdown files provide the narrative around them. Adding a module to the reference means adding such a page and an entry to `nav` in `zensical.toml`.

### Rendering the example figures

The figures of the documentation are the figures of the examples, and they are committed to `docs/images/`, so that the build of the site stays a plain `zensical build` and does not run any analysis. `scripts/render_examples.py` refreshes them: it runs every example of `tests/examples/test_examples.py` as a module in a temporary directory, with the `Agg` backend and warnings as errors, and copies every PNG the example wrote into `docs/images/` under its own name. It prints the files it wrote and fails when an example fails or writes no figure at all.

```bash
uv run python scripts/render_examples.py                  # every example
uv run python scripts/render_examples.py nca_single emax  # a selection
```

Run it after an example changed, after a plot function changed, and commit the images it wrote with that change; a page shows a figure with `![description](images/<example>.png)`.

A page only embeds a figure an example writes, so the committed images and the pages cannot drift apart: a snippet of the documentation which draws the same figure as an example builds the same data, and a figure nothing produces is described in a sentence instead.

### Snippets of the documentation

Every ` ```python ` block of the pages of the core functionality, of [Data formats](formats.md), of [Workflows](workflows.md) and of the validation follows two rules:

- **It runs.** The first block of the usage section of a page is self-contained (its imports, its data, the call and the output it prints) and runs from the root of the repository with warnings as errors:

    ```bash
    uv run python -W error snippet.py
    ```

    A later block of the same page may be a fragment, but then it names in a comment or in the sentence before it where every object it uses comes from ("the `batch` of the snippet above"). The output a snippet prints is shown below it, as a `text` block or as a markdown table, and is pasted from a run, never written by hand.

    `tests/docs/test_snippets.py` keeps this honest: it runs the blocks of every page in the order they appear and in one namespace per page, in a temporary working directory with the files of `docs/data/` next to them, in a subprocess with `-W error`. A fragment which names objects the page cannot build (the result of another page, a simulation, a study a reader brings) carries the comment `# not executed` as its first line and is skipped; every other block has to run.

- **It is formatted.** `ruff format` formats the code blocks of the markdown files as well, so `ruff format --check` covers the documentation and a snippet is written the way ruff would write it:

    ```bash
    uv run ruff format docs/
    ```

`docs/data/` holds the files a snippet may read, the event table of the Quickstart and the directory `benchmarks/` of [Benchmark datasets](benchmark_datasets.md): the theophylline and indomethacin datasets, `scenarios.csv` with the settings of every scenario and the result tables of Phoenix WinNonlin, PKNCA and NonCompart. That page pastes no output by hand at all: every snippet is followed by the markers `<!-- output:start -->` and `<!-- output:end -->`, `scripts/benchmark_datasets.py` runs the snippets of the page and writes what each one prints between them, and `tests/docs/test_benchmark_datasets.py` fails when the page is stale. The PKNCA and NonCompart tables are written by `scripts/benchmark_datasets.R`, which needs R with both packages:

```bash
Rscript scripts/benchmark_datasets.R          # docs/data/benchmarks/{pknca,noncompart}/, versions.csv
uv run python scripts/benchmark_datasets.py   # rewrites the output of docs/benchmark_datasets.md
```

The walk-throughs of [Workflows](workflows.md) are the longest of these snippets: they simulate their study in the first lines so that a reader can paste them anywhere, and the figures they save are the figures of the examples of the same data.

### Files for agents { #files-for-agents }

Agents and language models read markdown, not rendered html. `scripts/llms_txt.py` writes the files of the [llms.txt convention](https://llmstxt.org/) into the built site, i.e., [llms.txt](https://matthiaskoenig.github.io/pkpdutils/llms.txt) as an annotated index of all pages, [llms-full.txt](https://matthiaskoenig.github.io/pkpdutils/llms-full.txt) with the complete documentation in a single file, and the markdown of every page next to its html (`/nca.md` for `/nca/`). The markdown of the API reference is generated from the docstrings with `inspect`, since the pages themselves only contain the mkdocstrings directive.

```bash
uv run zensical build --clean --strict
uv run python scripts/llms_txt.py
```

The `documentation` workflow runs both steps, so the files are regenerated with every push. `docs/robots.txt` points crawlers at the sitemap and at these files. Zensical will provide agent context files itself at some point, then this script can go.

## Release

A release is made from `develop`. Since `develop` only accepts pull requests, the release is prepared on a branch and tagged once that pull request is merged:

1. branch off `develop`: `git switch -c release/x.y.z develop`
2. write the release notes for the version in `release-notes/x.y.z.md`
3. make sure everything passes: `tox run-parallel`, `ruff check`, `tox r -e ty`
4. check the version bump: `uvx bump-my-version bump [dev|major|minor|patch] --dry-run -vv`. A development version (`x.y.z.devN`) is finalized with `uvx bump-my-version bump dev`, which drops the `.devN` suffix (`1.0.0.dev0` becomes `1.0.0`); `major`, `minor` and `patch` start the next development cycle instead (`1.0.0.dev0` becomes `2.0.0.dev0`, `1.1.0.dev0`, `1.0.1.dev0`)
5. bump the version: `uvx bump-my-version bump [dev|major|minor|patch]`, which updates `src/pkpdutils/__init__.py`, `tests/test_package.py` and the `version` of `CITATION.cff` (not its `cff-version`, the version of the citation file format) and commits. Use `dev` to release the current development version and `major|minor|patch` to open the next one. It does not create the tag; a squash or rebase merge would rewrite the commit and leave the tag behind on a commit which is not part of `develop`
6. push the branch, open the pull request against `develop` and merge it once the checks are green
7. tag the merged commit on `develop` and push the tag:

    ```bash
    git switch develop
    git pull
    git tag x.y.z
    git push origin x.y.z
    ```

    This starts the `CI-CD` workflow, which runs the test matrix, publishes to [pypi](https://pypi.org/project/pkpdutils/), creates the GitHub release from `release-notes/x.y.z.md` and fast-forwards `main` to the tagged commit. Check the version before pushing, a tag cannot be moved or deleted afterwards.

8. test the installation from pypi in a fresh environment:

    ```bash
    uv venv --python 3.14
    uv pip install pkpdutils
    ```

9. once Zenodo has archived the release, update the citation information, i.e., `date-released` in `CITATION.cff` and the version, date and version DOI of the release in the citation of `README.md` and `docs/index.md`. `bump-my-version` only updates the version, not the date and the DOI, which are only known after the release. These changes go in through a pull request like everything else

### The floor of the rust kernel { #kernel-floor }

The analyses of an eligible batch run in `pkpdutils-core`, a hard dependency whose two contracts are closed (`pkpdutils.nca.engine`): `options_dict` sends every field of `NCAOptions` and `batch_dict` every field of the batch, and a build which does not know one of them refuses the whole call. That refusal is no `UnsupportedError` and no fallback to the python core, so it raises for every batch the kernel would have analysed.

A field added to `NCAOptions`, a new member of one of its enumerations and every other change of the two contracts therefore goes out together with the `pkpdutils-core` release which reads it: the same pull request raises the floor `pkpdutils-core>=x.y.z` in `pyproject.toml` to that release, runs `uv lock` and adds the python side, as every one of our libraries is used at its latest release.

The dependency is capped as well, `pkpdutils-core>=0.4.1,<0.5`, and that direction is the user's resolver rather than our pull requests: `uv add pkpdutils` would otherwise take any 0.x of the kernel, and a release which reports one parameter more than the host knows an expression for makes every eligible `nca()` call raise, since the computing path resolves the unit of every column the kernel reports (`pkpdutils.nca.engine._as_computed`, where `host_units` of the differential gate skips an unknown name instead). The cap is lifted by the pull request which raises the floor to that release, which is the convention for a cap on one of our own libraries. The floor is 0.4.1 because 0.4.0 carries the quantiles, the summary statistics of `summarize` and the two time to steady state estimators, all three of which the host now sends to the kernel, and 0.4.1 is the latest release of it, which is the convention for one of our own libraries even when the newer release changes no code. Two earlier releases are worth remembering for why a floor is raised rather than widened: 0.2.0 reported a unit its registry cannot read (`week`, `U/l`, an implicit product such as `mg l`) as an error of the batch and a numeric sample label which met a window key of another numeric type (`1.0` against `1`) as a missing sample, so `nca()` raised where the python path answers, and 0.2.1 refuses the first as unsupported, which the fallback catches, and matches the second as python does.

### The differential gate of the two engines { #differential-gate }

The gate covers the analysis. `ParameterResult.summarize` and `time_to_steady_state` dispatch to the kernel as well and are not in it: they are held to their python implementations by the test suite, where the fixtures `summary_engine` and `tss_engine` run every test of each of them through both engines and assert at teardown that the engine the test names is the one which really ran, and where one test compares the two answers of a whole summary variable by variable. Adding them to the gate would mean comparing every call of a user's own run, which is what the `--require-engine-gate` session does for the analysis.

The two engines of the analysis compute the same numbers, and the gate is what holds them to it. With `PKPDUTILS_COMPARE_ENGINES=1` set, every batch the kernel analyses is analysed by the python core as well and the two answers are compared (`pkpdutils.nca.engine.compare_engines`): the names of the variables in both directions, their order, the shape and the dtype of every column, the positions of `NaN` exactly, the numbers within `rel=1e-6, abs=1e-12`, the unit of every variable as a string and a counted variable, a flag word among them, exactly rather than within a tolerance. A disagreement raises `EngineMismatch` naming every one of them with both values. Without the variable an analysis pays one lookup of the environment, so the gate is a test run and a bisection and no production setting:

```bash
PKPDUTILS_COMPARE_ENGINES=1 pytest -m "not docs and not examples"
PKPDUTILS_COMPARE_ENGINES=1 pytest --require-engine-gate    # and refuse to run unarmed
```

That is the `engines` job of `.github/workflows/ci-cd.yml`, which the required `tests` check fails with, on linux and python 3.14 alone, since the two engines compute the same numbers on every platform.

A gate believed to be running is worse than no gate, so the arming is required rather than assumed. The suite takes `--require-engine-gate`, which fails the session before it collects a test unless the gate is really armed, and the `engines` job runs the suite with it: `tox -e py3.14 -- -m "not docs and not examples" --require-engine-gate`. The requirement arrives on the command line and the variable through the `passenv` of `tox.ini`, which is the asymmetry the check needs - a `passenv` which stops forwarding the variable, or a job which stops setting it, leaves the flag in place and the session refuses to run, where a suite which only read the environment would pass as a second copy of `test`. A misspelled flag is an unrecognized argument and fails the job just as loudly, and the job still names the variable once, at job level.

It was a separate step until the continuous integration of the pull request measured what that step did: `tox exec -e py3.14 -- python -c "... engine.comparing() ..."` exits 1 for a disarmed gate and exits 1 for `ModuleNotFoundError: No module named 'pkpdutils'` just as well, because `tox exec` installs no package (`package = wheel` applies to `tox r`), and it had only ever passed on a machine where an earlier `tox r` had left a wheel in `.tox/py3.14`. A check whose failure is indistinguishable from its catch is no check, which is why the question is asked inside the session now.

Beside the flag the session carries its own sentinel, which fails a run that passed every test when the name the job sets and the name the library reads disagree about whether the gate is armed, in either direction, and when they agree that it is armed, batches were built and not one of them was compared (`tests/conftest.py`, `pkpdutils.nca.engine.comparisons`); a distributed run from which no worker reported its count at all fails there too, since the sentinel would otherwise be reading zeros of its own. Every test passes in such a run and nothing else would say a word.

The gate compares the answer of a batch which reached the kernel, so it cannot see a batch which did not: a predicate of the host which quietly stops accepting some shape sends it down the python path, where the numbers are still right. `tests/conftest.py` is the other half and needs no variable: it asks both predicates about every batch of the suite which reaches `nca` (`engine.unsupported_reason` against `pkpdutils_core.is_supported`: the host forwards every refusal of the kernel verbatim and adds only its own, which the kernel cannot see, so the two agree word for word on everything else) and fails any test in which the kernel refused a batch the host let through. That half is the secondary one: a refusal after the host accepted a batch already means the two predicates disagreed, which the first half asserts before the kernel is called at all. It is watched on two channels which fail independently, the count of `UnsupportedError` raised by `engine.run_kernel` and the fallback warning itself, so that neither a logging level set by a test nor a test which replaces the kernel call can leave it quiet. A test which makes a refusal happen on purpose carries the marker `engine_fallback`, which is checked in both directions.
