# Performance

The non-compartmental analysis of `pkpdutils` runs in two engines: the rust kernel [`pkpdutils-core`](https://github.com/matthiaskoenig/pkpdutils-core), which is installed with the package, and the python core of `pkpdutils.nca.nca`. `nca()` takes the decision once per batch and the two compute the same numbers, so the only difference a caller sees is the time it takes. This page measures that difference. `ParameterResult.summarize`, `time_to_steady_state`, `nca_urine` and `nca_sparse` dispatch to the kernel in the same way, each with its python implementation as the fallback; all four are small next to an analysis and none is timed here.

The short answer is a crossover and not a headline: the kernel is worth more than an order of magnitude on the largest batch measured here and costs a fraction of a millisecond on a small one, because converting a batch and calling into the kernel costs the same whether the batch holds one curve or a hundred thousand. The figures are the sentences below the table, which the benchmark writes together with it.

That the two compute the same numbers is measured and not assumed: with `PKPDUTILS_COMPARE_ENGINES=1` every batch which reaches the kernel is analysed by the python core as well and the two answers are held against each other variable by variable, down to the positions of `NaN`, the dtype of every column and the unit of every variable. Continuous integration runs the whole test suite that way. The gate is the analysis alone: the summary, the time to steady state, the urine analysis and the sparse analysis are held to their python implementations by tests which run both engines on the same input, which is where their one documented difference is pinned as well (see [NCA](nca.md#the-two-engines)).

## The two engines on one batch

One batch of single dose intravenous bolus curves, analysed through `nca()` with the defaults, is timed through both engines by `scripts/benchmark.py`. Every case runs in a fresh interpreter and alternates the two engines run after run in that one process, kernel, python, kernel, python: the two halves are measured in the same window, so a ratio is not an artefact of whatever else the machine did between two blocks of runs. The python column is the python core of the same installation, reached through the switch `pkpdutils.nca.engine._force_python`. Before anything is timed the benchmark asks four questions a published ratio cannot be read back from, and fails the case rather than answering any of them with a plausible number: whether the dispatch sends this batch to the kernel at all, whether the forced python path is really forced, whether the differential gate is armed (which would run the python core inside the kernel half and halve the ratio) and whether the kernel accepts the batch when it is really called, since a refusal would be answered with a warning and the python path. `points` is the work of the analysis, the curves times the points per curve; `speedup` is the python median over the kernel median, so a number above 1 is the factor the kernel saves and a number below 1 the factor it costs.

<!-- timings:start -->
Measured on 2026-11-17, python 3.14.10 on Linux x86_64 with 20 cores, `pkpdutils` 1.4.0 with `pkpdutils-core` 0.3.0, 9 timed runs per engine and case after one warm-up run each.

| case | size | points | kernel ms | python ms | speedup | peak RSS MB |
|---|---|---|---|---|---|---|
| engine-1x12 | 1 curve x 12 points | 12 | 3.16 | 2.85 | 0.90x | 169 |
| engine-100x12 | 100 curves x 12 points | 1 200 | 3.24 | 3.17 | 0.98x | 169 |
| engine-10x200 | 10 curves x 200 points | 2 000 | 3.22 | 3.25 | 1.01x | 169 |
| engine-250x12 | 250 curves x 12 points | 3 000 | 3.35 | 3.56 | 1.06x | 172 |
| engine-1000x12 | 1 000 curves x 12 points | 12 000 | 3.74 | 5.40 | 1.44x | 185 |
| engine-100x200 | 100 curves x 200 points | 20 000 | 3.56 | 7.47 | 2.10x | 187 |
| engine-10000x12 | 10 000 curves x 12 points | 120 000 | 7.46 | 32.36 | 4.34x | 211 |
| engine-1000x200 | 1 000 curves x 200 points | 200 000 | 5.18 | 52.37 | 10.10x | 235 |
| engine-100000x12 | 100 000 curves x 12 points | 1 200 000 | 68.10 | 123.27 | 1.81x | 572 |
| engine-10000x200 | 10 000 curves x 200 points | 2 000 000 | 22.81 | 558.63 | 24.49x | 510 |

The largest ratio of this run is 24.49x, at 10 000 curves x 200 points, and the smallest 0.90x, at 1 curve x 12 points. The two engines cost the same between 1 200 and 2 000 points: 100 curves x 12 points is 0.98x and 10 curves x 200 points is 1.01x. Below the crossover the kernel path costs at most 0.31 ms more per call, and the ratios there run from 0.90x to 0.98x. Of two shapes of nearly equal work the ratios differ most between 10 000 curves x 12 points at 4.34x and 1 000 curves x 200 points at 10.10x. From 1 000 curves x 12 points on every shape of this run is a clear win, the first of them at 1.44x.
<!-- timings:end -->

## Where the crossover is

Below the crossover named under the table the kernel is the slower of the two, which is what lying below it means, and the sentence there names the range those ratios span and the largest amount the kernel path costs in milliseconds. The loss is bounded rather than a factor: the kernel path pays the conversion of the batch and the call into the kernel whatever the shape, which is a fixed cost and no multiple of the analysis. A ratio below 1 is that fixed cost against an analysis which is itself a few milliseconds, so which of the small shapes carries the smallest ratio of a run is noise and no property of the sweep. The ratio right at the crossover is worth no more than its first digit: it moves by a few percent between runs, and the sentence to read instead is the shape the clear wins start at, which is far enough above 1 to survive a rerun. From there the gain grows with the work, which the `points` and the `speedup` columns of the table show side by side.

What crosses over is the work rather than the shape of a batch, which is why the sweep carries a sparse and a dense case at every size - 10 curves of 200 points next to 100 curves of 12 points - rather than rows alone. How far that holds is a figure of the run and no claim of this paragraph: the sentence under the table names the two shapes of nearly equal work whose ratios differ most, which is the pair it holds least well for. The practical reading is that a bioequivalence study of two arms of 24 subjects pays the fixed cost that sentence bounds in milliseconds and nothing else, and that a simulation sweep or a population of thousands of profiles is the case the kernel was written for.

The sweep is not monotone in its ratio, and that is a property of the defaults rather than of this run: a batch is spread over several threads by **both** engines, by the python core above `NCA_WORKER_THRESHOLD` (20 000 rows, `pkpdutils.parallel`) and by the kernel under its own shard plan, so every shape above that many rows holds two parallel analyses against each other where every shape below it holds two serial ones. The ratio of such a shape need not follow the trend of the smaller ones and can fall below any of them, which is why the sentences under the table are read off the sweep in the order of the work: a run which crosses 1.00x at more than one shape names no single crossover and says so, instead of naming the largest and the smallest shape of the run as if they were neighbours. It is also why the pair of nearly equal work under the table leaves out the pair which straddles that threshold, however far apart its two ratios are: a parallel analysis against a serial one is a figure about the worker pools and none about the shape of a batch. The ratios of the defaults are what a caller gets; `NCAOptions(n_workers=1)` on both sides is a different and larger ratio.

## What the memory column says

`peak RSS MB` is the peak resident set size of the whole case process, which ran both engines, so it is the high-water mark of the larger of the two and not a per-engine figure. Most of it is the data and what the analysis computes over it: the batch of a million points and the 51 variables of its result are some 60 MB of float64, both cores work over whole columns rather than row by row, and the smallest cases of the table show what an interpreter which has imported `pkpdutils` costs before any batch exists. The wheel of `pkpdutils-core` also allocates through mimalloc, which its own benchmark measured at about 22 % less CPU on a batch of 100 000 rows by 12 points for about a third more peak resident set size on that shape, so this column reads a little higher than a glibc build of the same analysis would.

## What the kernel does not carry

An analysis `pkpdutils-core` does not carry takes the python path, with the numbers and the timings it always had, and no table of this page covers it: the two columns would be the same engine twice. The dispatch decides once per batch, and the reason it gives is the kernel's own sentence:

<!-- refusals:start -->
| analysis | the reason the dispatch gives |
|---|---|
| an effect curve | the analysis of an effect curve (Kind::Effect) |
| the acceptance criteria of the terminal phase | the acceptance criteria (NcaOptions::acceptance) |
| a named partial area | the named partial areas (NcaOptions::partial_aucs) |
| the uncertainty of a group curve | the propagation of the uncertainty of a group curve |
| routes which differ in their dosing | a batch of several routes of which only some have a multiple dose row, whose variables the python core reports per route |
| the reporting units of a result | the unit conversion of a result (NcaOptions::units) |
<!-- refusals:end -->

A batch of several routes is analysed by whichever engine the dispatch picked, in one call, and the result is the same: the kernel reports the union of the variables of the routes in its own order and the host sorts it into the order the python core would have reported, so the columns of a table do not depend on which engine ran.

## Reproducing this page

```bash
uv run python scripts/benchmark.py engines          # the table above
uv run python scripts/benchmark.py --write          # the table above, pasted into this page
uv run python scripts/benchmark.py --check          # exit 1 if the page no longer matches the cases
uv run python scripts/benchmark.py all --repeat 5   # the hot paths of the package as well
```

The numbers are of the machine named above the table, and of its load on that day. They are comparable against another run of the same script on the same machine, which is what a change to either engine is measured with, and they are no promise about another machine: the crossover in particular moves with the cost of a python call against the cost of a memory copy. Rerun `--write` after a release of `pkpdutils-core`, after a change to the dispatch or to the python core, and on the machine whose numbers you want to quote.

`--check` compares what the script controls and not what the machine does: the cases, their size labels, the columns of the table and that every measured cell is a number, plus the refusal table word for word, since that one is the dispatch's own text. It cannot tell whether a timing was measured or invented, which is why the timings belong to `--write` alone and never to a hand edit. `tests/docs/test_benchmark.py` runs the check, so a case added, removed, renamed or reshaped fails the test suite until the page is rewritten.

A figure in a paragraph is the one staleness neither of them can see, which is why the sentences under the table are generated with it and the prose of this page leans on them instead of repeating a number. The prose that is left makes claims which survive a rerun on another machine, a crossover between two named shapes and a gain which grows with the work; if a `--write` ever moves the table far enough that one of them stops being true, the paragraphs are what has to be rewritten, and no check will say so.
