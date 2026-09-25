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
| `tests` | `ci-cd.yml`   | the test matrix, linux with python 3.13, 3.14 and 3.15, macos and windows with 3.15 |
| `ruff`  | `ruff.yml`    | `ruff check` and `ruff format --check`                                |
| `ty`    | `ty.yml`      | `tox r -e ty`                                                         |
| `docs`  | `docs.yml`    | the zensical build including the api reference and the agent files    |

`tests` aggregates the test matrix into a single job, so the name of the required check stays the same when the matrix changes.

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

This needs the interpreters to be available, which uv installs with `uv python install 3.13 3.14 3.15`. Continuous integration runs the same environments as `uvx --with tox-uv tox -e py3.14`.

To run the tests directly against the development environment use

```bash
pytest                                          # the full suite
pytest -n 0                                     # in one process, e.g. for --pdb
pytest tests/test_units.py                  # a single module
pytest tests/test_units.py::test_parse_unit  # a single test
```

The tests run in parallel, `addopts = "-n auto"` in `pyproject.toml` gives pytest-xdist one worker per core; `-n 0` on the command line runs everything in one process, which the debugger needs.

The `conftest.py` at the root of the repository selects the non-interactive matplotlib backend for the session and puts the repository on `sys.path`, so that the tests can import the examples.

`tests/examples/test_example_scripts.py` runs the examples as `python -m examples.<module>` in a temporary working directory, so a broken example fails the test suite.

## Linting and formatting

Linting and formatting use [ruff](https://docs.astral.sh/ruff/):

```bash
ruff check     # lint
ruff format    # format
```

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
uv run python scripts/benchmark.py all                        # every case, about 15 s
uv run python scripts/benchmark.py nca-large bootstrap --repeat 5
```

The cases are `nca-small`, `nca-large`, `nca-multiple`, `bootstrap`, `delta`, `fit`, `constructors`, `iterate` and `all`; `--repeat` (3 by default) is the number of timed runs after one warm-up run. The script prints a markdown table with the size of the case, the median wall time and the peak resident set size. Every case runs in a fresh interpreter, so the memory and the caches (`pkpdutils.units`) of one case do not carry into the next.

The numbers are machine specific, they depend on the cores, the memory and the load of the machine they were measured on: use them to compare a change against the same table taken before it on the same machine, never as an absolute performance claim.

## Parallelism

`src/pkpdutils/parallel.py` holds the worker pools of the package. `executor(kind, n_workers)` returns one lazily created executor per kind and size, shared by every call of the process and closed by an `atexit` handler, so that the start-up of a process pool - about 0.7 s, since every worker imports `pkpdutils`, numpy, scipy, xarray and pint - is paid once and not once per analysis. The process pool starts its workers with `PROCESS_START_METHOD` on every python version, the default of python 3.14: `forkserver` on Linux, `spawn` on macOS and Windows. It never forks, whatever `multiprocessing.set_start_method` says: the `fork` default of python 3.13 on Linux copies a parent whose NCA threads may hold a lock into a child which can then block forever, which python 3.13 warns about with `DeprecationWarning: This process is multi-threaded, use of fork() may lead to deadlocks in the child`. `resolve_workers(n_workers, n_rows, threshold=..., max_workers=8)` turns the option into a worker count (`None` automatic and serial below the threshold, `1` serial, anything else taken as given) and `split_rows(n_rows, n_workers, min_rows=1000, max_rows=None)` cuts the rows into about one contiguous slice per worker, never shorter than `min_rows` while there is more than one and never longer than `max_rows`.

The two analyses use different workers, because their rows cost different things:

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

Every ` ```python ` block of the user guide and of [Workflows](workflows.md) follows two rules:

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
5. bump the version: `uvx bump-my-version bump [dev|major|minor|patch]`, which updates `src/pkpdutils/__init__.py` and `CITATION.cff` and commits. Use `dev` to release the current development version and `major|minor|patch` to open the next one. It does not create the tag; a squash or rebase merge would rewrite the commit and leave the tag behind on a commit which is not part of `develop`
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
