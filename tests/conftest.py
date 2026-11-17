"""The cross-check of the two engines every test of the suite runs under.

The analysis runs in the rust kernel of `pkpdutils.nca.engine` where that
build carries it and in the python core otherwise, and the decision is taken
by a predicate of the host which forwards to the kernel's own
(`engine.unsupported_reason` and `pkpdutils_core.is_supported`) and answers
itself only the three conditions about the python objects. The forwarding is
the authority and the host's own conditions are the part which can drift
silently and permanently: a condition which stops accepting some shape sends
it down the python path, the numbers are still right, and nothing ever
complains.

That is measured rather than feared. A predicate mutated to refuse every
batch whose unit is not `mg/l` leaves `tests/nca/test_engine.py` green at 57
passed, including the one regression test of a unit defect which had put
`vz_f` off by a factor of 1e9 - it passes while its batch never reaches the
kernel at all. The same holds for a predicate refusing every batch which
carries a limit of quantification. Many batches of the suite reach no kernel
at all - a forced python path, a batch of no samples, an analysis this build
does not carry - and which ones do changes with every release of
`pkpdutils-core`, so naming them one by one is a list somebody has to
maintain. The counts of a run are read off the hook itself (`CrossCheck`), not
written down here where they would go stale.

So the check is not a list but a hook of two halves, a sentinel over the
session and a flag the session can be held to:

- **every batch which reaches `nca` is asked of both predicates**, and their
  answers must be equal, word for word. A predicate which refuses what the
  kernel carries and one which accepts what it refuses are both caught, in
  whichever test builds such a batch.
- **a test in which the kernel refuses a batch the host let through fails**,
  since that is a defect of the predicate which the right numbers of the
  python core would otherwise hide. The one test which makes it happen on
  purpose carries the marker `engine_fallback`, and the marker is checked in
  both directions: a test which carries it and refuses nothing fails too, so
  it cannot be left behind on a test which stopped exercising the fallback.
- **a session which believes it armed the differential gate and compared
  nothing fails** (`pytest_sessionfinish`), whether because the library reads
  another name for the variable than the job sets, because the dispatch no
  longer runs the gate, or because no worker of a distributed run reported its
  count at all, which would leave the sentinel reading zeros of its own.
  Nothing else in the suite can see that: every test passes in the ordinary way
  and the evidence of two engines agreeing is a job which was never armed.
- **a session which is told it must be armed and is not fails before it
  collects a test** (`pytest_configure`, `REQUIRE_FLAG`). This is the one
  question a test process cannot ask of itself: a process which never sees the
  variable cannot know it was meant to be armed, so the requirement arrives on
  the command line while the variable arrives through the environment (the
  `passenv` of `tox.ini`). A variable which stops being forwarded leaves the
  flag behind and the session says so, instead of passing as a second copy of
  the ordinary test run.

  It is a flag of the suite and no longer a step of the job outside pytest,
  and the reason is measured: that step asked the same question with
  `tox exec -e py3.14 -- python -c "... engine.comparing() ..."`, which exits 1
  for a disarmed gate and exits 1 just as well because `tox exec` installs no
  package, so `import pkpdutils` fails. It passed where a `tox r` had left a
  wheel in the environment and failed on a clean runner, which means it had
  never once answered the question it was written for. A flag cannot fail that
  way: a misspelled one is an unrecognized argument and fails the job loudly,
  where a command which always exits 1 is indistinguishable from a real catch
  and one which always exits 0 is invisible.

**Half 2 is secondary to half 1.** A refusal of the kernel after the host
accepted the batch already means the two predicates disagreed, and half 1
asserts that at the call of the predicate, before `run_kernel` is reached. The
drift half 2 is there for is the one half 1 cannot see: `is_supported`
answering `None` while `analyze_batch` refuses anyway, the arm the kernel's own
comment calls reachable only if its predicate and its kernel have drifted
apart.

Half 2 therefore watches two channels which fail independently, and a refusal
in either is a refusal:

- **the count of `UnsupportedError`** raised by `engine.run_kernel`, taken by
  standing in for that function the way half 1 stands in for the predicate.
  It depends on no logging configuration at all, which the handler below does:
  a `caplog.set_level` on `pkpdutils.nca.nca`, a `logging.disable` or a
  `propagate` of that logger turned off would leave the handler silent, and a
  guard which can fall silent is exactly the kind that gets trusted after it
  has stopped working.
- **the fallback warning itself** (`pkpdutils.nca.nca.FALLBACK_WARNING`),
  which carries the kernel's own sentence for the failure message and which
  also sees a refusal in a test that replaces `engine.run_kernel` with a stub
  of its own, where the count above cannot.

The verdict of half 2 is reached in the call phase of a test rather than in
the teardown of the fixture, so that it is reported as a failure of that test
and not as an error of its teardown.

Two properties the hook is built for, both of them lessons of this plan:

- **it is armed once per test, not once per session.** `importlib.reload` of
  the engine module, which one test of `tests/nca/test_engine.py` does, resets
  every attribute of it and would silently disarm a patch applied once for the
  whole run; the measured cost of that was a count of 5 where the truth was
  14.
- **it is unconditional on the batches it sees.** A guard which only runs when
  the host says a batch is eligible is vacuous under a predicate which refuses
  everything, so the two conditions under which no question is asked of the
  kernel are read from the inputs - the forced python path of
  `engine._force_python` and a batch of no samples - and never from the
  predicate's answer.

The third refusal of the host's own, a batch of labels of several types with a
terminal window given by hand (`engine.MIXED_LABELS_REASON`), is the one
exemption read from the answer rather than from the inputs, because its
condition is a property of two encoders and not a flag: re-deriving it here
would be a second copy of the rule, which drifts. What is asserted instead is
that such a refusal was warranted, by the necessary condition of it - the
batch is labelled by python objects and a window names a sample - so a
predicate which answered that reason for every batch fails here on the first
ordinary one, which is what the rule above is for.
"""

import logging
import os
from collections.abc import Callable, Generator, Iterator

import numpy as np
import pkpdutils_core
import pytest

from pkpdutils.nca import engine
from pkpdutils.nca.nca import FALLBACK_WARNING
from pkpdutils.nca.options import WINDOW_DEFAULT_KEY, NCAOptions
from pkpdutils.timecourse import Timecourses

FALLBACK_MARKER = "engine_fallback"
"""The marker of a test which makes the kernel refuse a batch on purpose."""

NCA_LOGGER = "pkpdutils.nca.nca"
"""The logger the fallback warning is written to."""

COMPARE_VARIABLE = "PKPDUTILS_COMPARE_ENGINES"
"""The environment variable the differential gate is armed with.

Spelled out here rather than imported from `engine.COMPARE_VARIABLE` on
purpose: this is the name the `engines` job of `.github/workflows/ci-cd.yml`
writes, and the sentinel of `pytest_sessionfinish` is the assertion that the
name the job sets and the name the library reads are still the same one. A
rename on one side alone arms nothing and would otherwise turn that job into a
second green copy of the ordinary test run.
"""

REQUIRE_FLAG = "--require-engine-gate"
"""The flag which makes a session fail unless the differential gate is armed.

The one thing the `engines` job of `.github/workflows/ci-cd.yml` says on the
command line, where the variable which arms the gate is the one thing it says
through the environment. The two travel by different roads on purpose: a
`passenv` of `tox.ini` which stops forwarding the variable, or a job which
stops setting it, leaves the flag in place and the session refuses to run,
where a suite which only reads the environment would pass as a second copy of
the ordinary test run. See the module docstring for why this is a flag and no
longer a step of the job.
"""

CHECK_KEY = pytest.StashKey["CrossCheck"]()
"""Where the cross-check of a test is kept for the hooks below to read."""


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the flag which requires the differential gate to be armed.

    Args:
        parser: the parser of the command line.
    """
    parser.addoption(
        REQUIRE_FLAG,
        action="store_true",
        default=False,
        help=(
            "fail the session unless the differential gate of the two engines "
            f"is armed ({COMPARE_VARIABLE}), which is how the `engines` job of "
            "continuous integration runs the suite"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Refuse a session which has to compare the two engines and would not.

    Reached before a test is collected, so a job which lost the variable says
    so in one line instead of reporting the agreement of two engines it ran
    once. The value is read from the library (`engine.comparing`) rather than
    from the environment here, so that the flag asks the question the analysis
    itself asks.

    Args:
        config: the configuration of the session.

    Raises:
        pytest.UsageError: if the flag is given and the gate is not armed.
    """
    if not config.getoption(REQUIRE_FLAG) or engine.comparing():
        return
    raise pytest.UsageError(
        f"{REQUIRE_FLAG} says this session must compare the two engines of "
        f"the analysis, and the gate is not armed: {COMPARE_VARIABLE} is "
        f"{os.environ.get(COMPARE_VARIABLE, '')!r} and the library arms the "
        f"gate with {engine.COMPARE_VARIABLE!r}. Every test of such a run "
        "passes through one engine and reports the agreement of two. Set "
        f"{COMPARE_VARIABLE}=1 to arm it, or drop the flag to run the suite "
        "through whichever engine the dispatch picks."
    )


# the wheel of `pkpdutils-core` carries no stub beside its compiled module, so
# the one lookup of the kernel's own predicate is suppressed here rather than
# at every call of it
is_supported = pkpdutils_core.is_supported  # ty: ignore[missing-attribute]

Predicate = Callable[..., str | None]
"""The signature of `engine.unsupported_reason`, which the hook stands in for."""

Kernel = Callable[..., dict[str, np.ndarray]]
"""The signature of `engine.run_kernel`, which the hook stands in for."""


def _object_labels(timecourses: Timecourses) -> bool:
    """Whether a sample dimension of a batch is labelled by python objects.

    The necessary condition of `engine.MIXED_LABELS_REASON`, read from the
    batch: labels of several types are objects, whatever else they are.

    Args:
        timecourses: the batch.

    Returns:
        Whether one of its sample dimensions carries an object coordinate.
    """
    return any(
        timecourses.ds.coords[dim].dtype.kind == "O"
        for dim in timecourses.sample_dims
        if dim in timecourses.ds.coords
    )


def _routes_differ_in_dosing(timecourses: Timecourses, options: NCAOptions) -> bool:
    """Whether some routes of a batch have a multiple dose row and others have not.

    The necessary condition of `engine.MIXED_ROUTE_DOSING_REASON`, derived from
    the dose arrays on its own and not from the function of the engine which
    answers the reason, so that the two cannot agree by being the same code.

    Args:
        timecourses: the batch.
        options: the options of the analysis.

    Returns:
        Whether the routes of the batch differ in having a protocol of two or
        more doses, `False` when `tau` is given (every row is then a multiple
        dose row).
    """
    routes = timecourses.routes
    if routes is None or options.tau is not None:
        return False
    amounts, times, _ = engine.dose_arrays(timecourses)
    doses = (np.isfinite(amounts) & np.isfinite(times)).sum(axis=1)
    rows = routes.ravel()
    has = {bool((doses[rows == route] >= 2).any()) for route in set(rows.tolist())}
    return len(has) > 1


def _named_window(options: NCAOptions) -> bool:
    """Whether a terminal window is keyed by the label of a sample.

    The other necessary condition of `engine.MIXED_LABELS_REASON`, read from
    the options: the default key names every sample the mapping does not name
    and is no label of the batch.

    Args:
        options: the options of the analysis.

    Returns:
        Whether at least one window is keyed by a label.
    """
    return any(key != WINDOW_DEFAULT_KEY for key in options.terminal.windows or ())


class CrossCheck:
    """The host's eligibility predicate and its kernel call, both watched.

    It stands in for `engine.unsupported_reason` and for `engine.run_kernel`
    for the duration of one test and counts what it saw, so that a test can
    assert the hook really looked at its batches rather than trusting that it
    did.
    """

    def __init__(self, predicate: Predicate, kernel: Kernel) -> None:
        """Wrap the real predicate and the real kernel call.

        Args:
            predicate: `engine.unsupported_reason` as it was before the hook
                replaced it; called rather than the module attribute, which is
                the hook itself.
            kernel: `engine.run_kernel` as it was before the hook replaced it.
        """
        self._predicate = predicate
        self._kernel = kernel
        self.compared: int = 0
        """How many batches both predicates answered."""
        self.forced: int = 0
        """How many batches took the python path because it was forced."""
        self.empty: int = 0
        """How many batches had no sample at all."""
        self.mixed: int = 0
        """How many batches the host refused for the types of their labels."""
        self.dosing: int = 0
        """How many batches the host refused for routes which differ in dosing."""
        self.refusals: list[str] = []
        """The kernel's refusals of a batch the host let through."""
        self.fallbacks: list[logging.LogRecord] = []
        """The fallback warnings logged while this test ran."""

    def unsupported_reason(
        self,
        timecourses: Timecourses,
        options: NCAOptions,
        *,
        prepared: engine.KernelInput | None = None,
    ) -> str | None:
        """Why the kernel would refuse this analysis, checked against the kernel.

        Args:
            timecourses: the batch
            options: the options of the analysis

        Returns:
            Whatever the host's predicate answers, unchanged.
        """
        reason = self._predicate(timecourses, options, prepared=prepared)
        # the two conditions are read from the inputs and never from `reason`,
        # so that a predicate which answers one of these reasons for every
        # batch does not make the check vacuous
        if engine._force_python:
            self.forced += 1
            assert reason == engine.FORCED_REASON, (
                "the python path is forced and the predicate answers "
                f"{reason!r} instead of the forced reason"
            )
            return reason
        if timecourses.n_samples == 0:
            self.empty += 1
            assert reason == engine.EMPTY_REASON, (
                "a batch of no samples is the host's own refusal and the "
                f"predicate answers {reason!r} instead of the empty reason"
            )
            return reason
        if reason == engine.MIXED_ROUTE_DOSING_REASON:
            # the host's fourth own refusal, a statement about how the python
            # core groups a batch which the kernel's predicate does not make.
            # As for the labels, the exemption is read from the answer and its
            # necessary condition from the inputs: a batch of several routes
            # with a protocol of more than one dose or a given `tau`
            self.dosing += 1
            assert _routes_differ_in_dosing(timecourses, options), (
                "the predicate refuses this batch for routes which differ in "
                "their dosing, and the routes of the batch either all have a "
                "multiple dose row or none has one (or `tau` is given): a "
                "predicate which answers that reason for any other batch "
                "sends it down the python path for good."
            )
            return reason
        if reason == engine.MIXED_LABELS_REASON:
            # the host's third own refusal, which the kernel's predicate
            # accepts and its analysis then rejects with a plain `ValueError`,
            # so the two answers are unequal by design. The condition is read
            # from the answer here and not from the inputs, see the module
            # docstring; what is read from the inputs is the necessary
            # condition of it, so the exemption cannot be claimed for a batch
            # it does not apply to
            self.mixed += 1
            assert _object_labels(timecourses) and _named_window(options), (
                "the predicate refuses this batch for the types of its sample "
                f"labels, and the batch is labelled over {timecourses.sample_dims} "
                f"with the terminal windows {options.terminal.windows!r}: that "
                "reason names a batch of labels of several types with a window "
                "given by hand, and a predicate which answers it for any other "
                "batch sends that batch down the python path for good."
            )
            return reason
        kernel = is_supported(
            engine.batch_dict(timecourses), engine.options_dict(options)
        )
        assert reason == kernel, (
            f"the two predicates disagree on a batch of {timecourses.n_samples} "
            f"sample(s) by {timecourses.n_time} time point(s) in "
            f"{timecourses.unit!r}: `engine.unsupported_reason` answers "
            f"{reason!r} and `pkpdutils_core.is_supported` {kernel!r}. A host "
            "predicate which refuses an analysis the kernel carries sends it "
            "down the python path for good and without a sound; one which "
            "accepts an analysis the kernel refuses raises at the call site of "
            "a user. See tests/conftest.py."
        )
        self.compared += 1
        return reason

    def run_kernel(
        self,
        timecourses: Timecourses,
        options: NCAOptions,
        *,
        units: dict[str, str] | None = None,
        prepared: engine.KernelInput | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the kernel, counting a refusal of a batch the host let through.

        The channel of half 2 which does not depend on the logging
        configuration: the refusal is counted where it is raised and the
        exception goes on to the caller untouched.

        Args:
            timecourses: the batch
            options: the options of the analysis

        Keyword Args:
            units: the sink of the kernel's units, passed through

        Returns:
            Whatever the kernel answers, unchanged.

        Raises:
            UnsupportedError: as the kernel raised it.
        """
        try:
            return self._kernel(timecourses, options, units=units, prepared=prepared)
        except engine.UnsupportedError as refusal:
            self.refusals.append(str(refusal))
            raise


class FallbackWatch(logging.Handler):
    """The handler which collects the fallback warning of `nca`.

    The record is recognized by the format string itself
    (`pkpdutils.nca.nca.FALLBACK_WARNING`) rather than by a wording, so the
    watch cannot come loose when the sentence is rewritten.
    """

    def __init__(self, check: CrossCheck) -> None:
        """Collect into the cross-check of one test.

        Args:
            check: the cross-check of the running test.
        """
        super().__init__(level=logging.WARNING)
        self._check = check

    def emit(self, record: logging.LogRecord) -> None:
        """Collect the record if it is the fallback warning.

        Args:
            record: the record.
        """
        if record.msg == FALLBACK_WARNING:
            self._check.fallbacks.append(record)


@pytest.fixture(autouse=True)
def engine_cross_check(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[CrossCheck]:
    """Hold the host's dispatch against the kernel's own answer, for every test.

    Armed per test rather than per session, see the module docstring. A test
    may ask for it by name to assert what it saw.

    Args:
        request: the running test, which the cross-check is stashed on
        monkeypatch: the patch of the predicate and of the kernel call, undone
            after the test

    Yields:
        The cross-check of this test.
    """
    check = CrossCheck(engine.unsupported_reason, engine.run_kernel)
    monkeypatch.setattr(engine, "unsupported_reason", check.unsupported_reason)
    monkeypatch.setattr(engine, "run_kernel", check.run_kernel)
    request.node.stash[CHECK_KEY] = check
    watch = FallbackWatch(check)
    logger = logging.getLogger(NCA_LOGGER)
    logger.addHandler(watch)
    try:
        yield check
    finally:
        logger.removeHandler(watch)
        global _batches_seen
        _batches_seen += (
            check.compared + check.forced + check.empty + check.mixed + check.dosing
        )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None]:
    """Reach the verdict of half 2 in the call phase of a test.

    In the call phase and not in the teardown of the fixture, so that a
    fallback is reported as a failure of the test which caused it. A test whose
    body already failed or was skipped raises out of `yield` and is left alone,
    so this never masks the real failure.

    Args:
        item: the running test.

    Yields:
        Nothing; the test runs inside the `yield`.
    """
    yield
    check = item.stash.get(CHECK_KEY, None)
    if check is None:
        return
    expected = item.get_closest_marker(FALLBACK_MARKER) is not None
    refused = sorted({*check.refusals, *(r.getMessage() for r in check.fallbacks)})
    if refused and not expected:
        pytest.fail(
            "the kernel refused a batch which `engine.unsupported_reason` let "
            "through, so the analysis fell back to the python core: "
            + "; ".join(refused)
            + ". Mark a test which does this on purpose with "
            f"`@pytest.mark.{FALLBACK_MARKER}`."
        )
    if expected and not refused:
        pytest.fail(
            f"the marker `{FALLBACK_MARKER}` says this test makes the kernel "
            "refuse a batch and none was refused; remove the marker, which "
            "would otherwise hide a real fallback."
        )


_batches_seen: int = 0
"""The batches the cross-check of this process was asked about."""

_worker_comparisons: int = 0
"""The comparisons the finished xdist workers reported, see `pytest_testnodedown`."""

_worker_batches: int = 0
"""The batches the finished xdist workers were asked about."""

_workers_reported: int = 0
"""How many finished xdist workers reported a count at all.

The evidence of the sentinel itself: with `-n auto` the controller runs no test
and counts nothing, so every number it reaches its verdict on arrives through
the `workeroutput` of a worker. A break in that plumbing - a hook which is no
longer called, a key renamed on one side - leaves both sums at zero, which
reads exactly like a selection that built no batch, and the sentinel would
return without a word. This counter is what tells the two apart.
"""


def pytest_testnodedown(node: object, error: object) -> None:
    """Collect the gate's comparison count from a finished xdist worker.

    The counter of `engine.comparisons` lives in the process which did the
    comparing, so with `-n auto` every worker reports its own count up to the
    session which started them.

    Args:
        node: the worker which finished
        error: whatever it crashed with, which leaves its count behind.
    """
    global _worker_comparisons, _worker_batches, _workers_reported
    output = getattr(node, "workeroutput", None) or {}
    if "engine_comparisons" in output:
        _workers_reported += 1
    _worker_comparisons += int(output.get("engine_comparisons", 0))
    _worker_batches += int(output.get("engine_batches", 0))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail a session which armed the differential gate and compared nothing.

    The gate is armed by an environment variable, which gives it three ways to
    be believed and absent, and two of them are visible from inside the test
    process:

    - **the two names disagree, in either direction.** The name here is a
      literal of the suite (`COMPARE_VARIABLE`) and the session asks the
      library whether that name armed it (`engine.comparing()`); the job sets
      one name and the library reads the other, or this suite watches a name
      nobody sets while the gate runs armed, and either way a rename on one
      side alone is a failure here rather than a silence. The check is
      symmetric on purpose: the second direction would otherwise disarm every
      question below without a word, since an unset variable reads like an
      ordinary unarmed run.
    - **the gate is no longer reached.** Armed, batches built and zero of them
      compared (`engine.comparisons`) means the dispatch of `nca` no longer
      runs it. A selection which builds no batch at all says nothing about the
      gate and is left alone.
    - **nothing reported.** The counts of a distributed run are the ones the
      workers send up (`pytest_testnodedown`, `_workers_reported`), and a break
      in that plumbing leaves them at zero, which reads like the selection
      above. A run which collected tests under xdist and from which no worker
      reported therefore fails rather than being read as a run which compared
      nothing on purpose.
    - the variable never reaches pytest at all, through a `passenv` of
      `tox.ini` or the `env:` of a job. That one is **not** reached here, since
      a session which never sees the variable cannot know it was meant to be
      armed: `REQUIRE_FLAG` is that half, and such a run is refused in
      `pytest_configure` before a test is collected, so the sentinel never gets
      the chance to be read as an ordinary unarmed run.

    The verdict is only reached on a session which passed everything: a run
    which failed, was interrupted or collected no test says something more
    useful than this already.

    Args:
        session: the finished session, whose exit status is set on a failure
        exitstatus: the status it would otherwise report.
    """
    config = session.config
    if getattr(config, "workerinput", None) is not None:
        # an xdist worker: report this process's count up to the session
        output = getattr(config, "workeroutput", None)
        if output is not None:
            output["engine_comparisons"] = engine.comparisons()
            output["engine_batches"] = _batches_seen
        return
    if exitstatus != pytest.ExitCode.OK:
        # a run which failed, was interrupted or collected nothing already
        # reports something more useful than this
        return
    armed_by_the_job = (
        os.environ.get(COMPARE_VARIABLE, "").strip().lower() not in engine.FALSE_WORDS
    )
    # no monkeypatch of any test is in place any more, so this is the session's
    # own arming as the library reads it
    armed_by_the_library = engine.comparing()
    if armed_by_the_job and not armed_by_the_library:
        _fail_session(
            session,
            "the gate is armed by another name",
            f"the job sets {COMPARE_VARIABLE} and the library arms the gate "
            f"with {engine.COMPARE_VARIABLE!r}, so this run compared the two "
            "engines on nothing while reporting that it did. Rename it in "
            "both places, here and in tests/conftest.py.",
        )
        return
    if armed_by_the_library and not armed_by_the_job:
        _fail_session(
            session,
            "the suite watches another name",
            f"the gate is armed by {engine.COMPARE_VARIABLE!r} and this suite "
            f"watches {COMPARE_VARIABLE} for it, so every check of whether a "
            "run was really armed reads an empty value and passes without "
            "asking anything. Rename it in both places, here and in the "
            "`engines` job of .github/workflows/ci-cd.yml.",
        )
        return
    if not armed_by_the_job:
        return
    if (
        config.pluginmanager.hasplugin("dsession")
        and session.testscollected
        and not _workers_reported
    ):
        # every count of a distributed run arrives through `workeroutput`, so a
        # break in that plumbing leaves the two sums below at zero and the
        # verdict reads like a selection which built no batch
        _fail_session(
            session,
            "no worker reported to the gate's sentinel",
            f"{session.testscollected} test(s) ran in xdist workers and not "
            "one of them reported its count of the differential gate up to "
            "this session, so every question below would be asked of two "
            "zeros. `pytest_testnodedown` of tests/conftest.py no longer sees "
            "the `workeroutput` of a worker, or the worker no longer writes "
            "it.",
        )
        return
    # the count includes the batches of the gate's own armed tests, which arm
    # it with `monkeypatch` and travel the same dispatch; a gate which no
    # longer runs stops counting those too, so the floor stays honest
    if engine.comparisons() + _worker_comparisons:
        return
    if not _batches_seen + _worker_batches:
        # a selection which builds no batch at all, e.g. a single file of
        # another subsystem, says nothing about the gate either way
        return
    _fail_session(
        session,
        "the differential gate compared nothing",
        f"the gate is armed ({COMPARE_VARIABLE} is set) and not one batch was "
        "compared: this run reports the agreement of two engines and took the "
        "suite through one. The gate is no longer reached by "
        "`pkpdutils.nca.nca.nca`, or no batch of this selection reached the "
        "kernel at all.",
    )


def _fail_session(session: pytest.Session, title: str, message: str) -> None:
    """Fail a session which passed every test, and say why.

    Args:
        session: the finished session
        title: the separator line of the report
        message: what went wrong.
    """
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep("=", title, red=True, bold=True)
        reporter.write_line(message)
    session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(params=["kernel", "python"])
def summary_engine(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[str]:
    """Run a test of `ParameterResult.summarize` once through each engine.

    `"python"` forces the scipy path with `engine._force_python`, `"kernel"` leaves
    the dispatch alone. Either way the fixture counts the calls of
    `pkpdutils_core.summarize` and, at teardown, asserts the engine the test
    names is the one which ran: the kernel was called, the python path never
    reached it. Without that a test of "both engines" could pass twice through
    one of them.

    Args:
        request: the parametrization.
        monkeypatch: restores the forced flag and the wrapped entry point.

    The `kernel` parameter skips when the installed `pkpdutils-core` has no
    `summarize` (the wheels below 0.4 which the declared floor admits).

    Yields:
        The name of the engine the test runs under.
    """
    calls: list[int] = []
    if request.param == "kernel" and not hasattr(pkpdutils_core, "summarize"):
        pytest.skip("this build of pkpdutils-core carries no `summarize`")
    if hasattr(pkpdutils_core, "summarize"):
        real = pkpdutils_core.summarize
        monkeypatch.setattr(
            pkpdutils_core, "summarize", lambda *args: (calls.append(1), real(*args))[1]
        )
    monkeypatch.setattr(engine, "_force_python", request.param == "python")
    yield request.param
    if request.param == "kernel":
        assert calls, "the kernel was never asked to summarize"
    else:
        assert not calls, "the forced python path reached the kernel"
