# Performance

The non-compartmental analysis (NCA) runs through the Rust kernel [`pkpdutils-core`](https://github.com/matthiaskoenig/pkpdutils-core), installed with the package, or the Python implementation in `pkpdutils.nca.nca`. This benchmark measures the speedup of the public `nca()` call and profiles where both paths spend their time.

The workload is reproducible, seeded, noisy mono-exponential intravenous bolus profiles, with a single dose, automatic terminal-window selection, and no uncertainty propagation. The sweep covers both short and densely sampled profiles. Imports and fixture construction are excluded. These results do not measure bootstrap, delta-method uncertainty, multiple dosing, other routes, or the separate summary, urine, sparse and steady-state APIs. Use the other hot-path cases to investigate those workloads.

Every case checks dispatch eligibility and compares warmed results, including values, missing values, dtypes and unit attributes, before timing. The benchmark refuses to run with `PKPDUTILS_COMPARE_ENGINES=1`: that differential gate runs both engines inside the Rust path and would invalidate the speedup. Continuous integration uses the gate separately to check numerical parity across the test suite; see [NCA](nca.md#the-two-engines).

## The two engines on one batch

`scripts/benchmark.py` runs each shape in a fresh interpreter. Both engines use the same prebuilt batch and options. Each engine warms once, then timed pairs alternate order: Rust/Python, Python/Rust. Garbage collection runs before each measurement, outside its timer. The Python path uses the private benchmark switch `pkpdutils.nca.engine._force_python`.

Times are median wall times. Each adjacent IQR column is the difference between the 75th and 25th percentiles of that engine's samples, not a confidence interval. Speedup is the Python median divided by the Rust median; values above 1 favor Rust. Small differences near 1 need reruns. [Raw samples and machine metadata](data/nca_benchmark.json) retain every run, dependency versions, CPU affinity, worker policy and thread environment.

<!-- timings:start -->
Measured on 2026-11-19, python 3.14.10 on Linux x86_64, 12th Gen Intel(R) Core(TM) i7-12700, with 20 logical CPUs, `pkpdutils` 1.7.0 with `pkpdutils-core` 0.7.0, 15 timed runs per engine and case after one warm-up run each, with automatic workers.

| case | size | points | kernel ms | IQR ms | python ms | IQR ms | speedup | peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| engine-1x12 | 1 curve x 12 points | 12 | 1.29 | 0.05 | 1.57 | 0.06 | 1.22x | 171 |
| engine-100x12 | 100 curves x 12 points | 1 200 | 1.36 | 0.03 | 1.89 | 0.07 | 1.38x | 171 |
| engine-10x200 | 10 curves x 200 points | 2 000 | 1.41 | 0.04 | 1.97 | 0.04 | 1.39x | 172 |
| engine-250x12 | 250 curves x 12 points | 3 000 | 1.47 | 0.05 | 2.29 | 0.07 | 1.56x | 174 |
| engine-1000x12 | 1 000 curves x 12 points | 12 000 | 1.77 | 0.24 | 4.19 | 0.14 | 2.37x | 187 |
| engine-100x200 | 100 curves x 200 points | 20 000 | 1.64 | 0.26 | 6.00 | 0.33 | 3.66x | 189 |
| engine-10000x12 | 10 000 curves x 12 points | 120 000 | 5.27 | 0.22 | 30.95 | 0.49 | 5.88x | 216 |
| engine-1000x200 | 1 000 curves x 200 points | 200 000 | 3.50 | 0.55 | 52.06 | 1.94 | 14.89x | 238 |
| engine-100000x12 | 100 000 curves x 12 points | 1 200 000 | 52.13 | 2.68 | 97.03 | 5.31 | 1.86x | 615 |
| engine-10000x200 | 10 000 curves x 200 points | 2 000 000 | 18.91 | 1.83 | 477.87 | 9.11 | 25.27x | 511 |

The largest ratio of this run is 25.27x, at 10 000 curves x 200 points, and the smallest 1.22x, at 1 curve x 12 points. Every shape of this run is at or above 1.00x; no crossover is observed among the measured shapes. Of two shapes of nearly equal work the ratios differ most between 10 000 curves x 12 points at 5.88x and 1 000 curves x 200 points at 14.89x. From 100 curves x 12 points on every shape of this run is a clear win, the first of them at 1.38x.
<!-- timings:end -->

## Interpreting batch size and workers

Dispatch and dataset construction impose a fixed per-call cost, while array conversion, output copies and numerical work also scale with batch size. Batching profiles amortizes the fixed part. Shape matters too: equal point counts do not imply equal timings, since every profile produces its own output parameters and terminal-window search. The generated sentences report the measured range rather than assuming a crossover or a monotone speedup.

Both columns use automatic workers. Python switches to a shared thread pool at `NCA_WORKER_THRESHOLD` (20 000 profiles); Rust uses its own shard policy. This is the speedup of the installed defaults, including their worker policies, not an equal-thread comparison of languages. Use `--n-workers 1` for a controlled serial comparison, or an explicit count for scaling studies. Profile counts and sampling density should both inform future worker-threshold tuning.

### Serial control

Both engines use `n_workers=1` in these controls, measured in fresh processes after the automatic-worker sweep, with the same repeat count. The comparison includes host and binding costs; it does not isolate language arithmetic. Machine load can differ between cases, so compare dispersion as well as medians. The raw JSON retains these cases under `serial_rows`.

<!-- serial:start -->
| case | size | points | kernel ms | IQR ms | python ms | IQR ms | speedup | peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| engine-100000x12 | 100 000 curves x 12 points | 1 200 000 | 100.07 | 9.05 | 273.66 | 2.35 | 2.73x | 467 |
| engine-10000x200 | 10 000 curves x 200 points | 2 000 000 | 88.37 | 3.32 | 474.16 | 10.16 | 5.37x | 503 |
<!-- serial:end -->

## Profiling the timings

The phase table comes from separate instrumented calls after each case's speed measurements. These are **mean wall times**, not the medians above: phases are exclusive, and their means sum to the mean total before rounding. Instrumentation adds overhead, so use the first table for speedup. The raw JSON includes every phase sample for every shape; this table shows representative shapes.

<!-- profiles:start -->
| case | engine | total ms | input + dispatch ms | binding ms | output adaptation ms | Python compute ms | result ms | other ms |
|---|---|---|---|---|---|---|---|---|
| engine-1x12 | kernel | 1.224 | 0.207 | 0.106 | 0.079 | 0.000 | 0.559 | 0.273 |
| engine-1x12 | python | 1.536 | 0.002 | 0.000 | 0.000 | 0.663 | 0.562 | 0.310 |
| engine-1000x200 | kernel | 3.339 | 0.656 | 1.532 | 0.117 | 0.000 | 0.667 | 0.368 |
| engine-1000x200 | python | 52.066 | 0.001 | 0.000 | 0.000 | 50.872 | 0.661 | 0.532 |
| engine-100000x12 | kernel | 55.603 | 3.218 | 44.559 | 0.335 | 0.000 | 4.244 | 3.248 |
| engine-100000x12 | python | 100.028 | 0.001 | 0.000 | 0.000 | 91.610 | 4.256 | 4.161 |
| engine-10000x200 | kernel | 18.426 | 3.846 | 12.877 | 0.149 | 0.000 | 0.926 | 0.627 |
| engine-10000x200 | python | 474.506 | 0.001 | 0.000 | 0.000 | 471.941 | 0.909 | 1.655 |

For one curve, result assembly is 45.7 % of the profiled Rust path and the binding is 8.6 %. For 10 000 curves x 200 points, the binding is 69.9 % and result assembly is 5.0 %. Even eliminating the entire binding call would cap further speedup on that shape at 3.32x with the remaining costs unchanged. For 100 000 curves x 12 points, output adaptation is 0.6 % and result assembly is 7.6 % of the profiled Rust path. The independent serial Python function profile of the dense batch ranks `auc.py:60(segment_areas)` (191.73 ms self time), `terminal.py:106(window_statistics)` (105.04 ms self time), `_shape_base_impl.py:60(take_along_axis)` (33.35 ms self time) highest.
<!-- profiles:end -->

The boundaries are:

- **Input + dispatch:** preparing the batch and options dictionaries and checking kernel support. The JSON separates input preparation from the rest of dispatch.
- **Binding:** `pkpdutils_core.analyze_batch`, including PyO3 input copies, native computation, native workers and Python output conversion. This is not a measurement of pure Rust computation.
- **Output adaptation:** the rest of `engine.run_kernel`, including parameter mapping, dtype and unit normalization, excluding the binding and any input preparation.
- **Python compute:** `run_analysis_rows`, including NumPy computation and any thread-pool wait. No Python computation runs in the eligible Rust path.
- **Result:** `_to_result`, including parameter scaling, array copies, coordinates, units and the xarray result dataset.
- **Other:** the remaining host orchestration, validation, flags, dose handling and option serialization, plus instrumentation overhead outside the named boundaries.

`--profile` also collects an independent `cProfile` run for each engine and case, using `n_workers=1` in both engines so Python numerical functions stay in the calling thread. These are diagnostic profiles of the serial paths, not timings of the automatic worker policy. Their overhead varies with Python call count; function times are never used in either table. The JSON retains the highest self and cumulative times. `--profile-dir` additionally saves `.prof` and `.txt` files, sorted by cumulative and self time. Python profiling cannot split native Rust work from conversion inside the binding. Use phase timers for parallel wall-time attribution and native sampling with symbols before attributing a Rust cost to a specific algorithm.

## Further improvements to investigate

These are candidates supported by the measured boundaries and code inspection, not measured gains from changes:

1. **Reduce result copies and repeated result construction.** `_to_result` multiplies numeric columns even when the unit factor is 1, and casts integer columns again after the Rust adapter normalizes their dtype. Investigate reusing arrays for identity conversions and `astype(copy=False)`, while preserving result ownership and independence. Reuse validated schema and unit metadata where inputs permit it. The result share in the phase table bounds the benefit.
2. **Avoid decoding the same inputs twice.** The support predicate and the analysis binding each decode the input batch into owned Rust buffers. Consider a validated prepared-input handle, or a support check that reads only metadata, while preserving fallback semantics and safe buffer ownership when the GIL is released. The input/dispatch phase and binding phase contain this work, but the profile does not isolate its share. Rust output vectors already transfer to NumPy without copying; a new output buffer interface alone would not remove these input costs. Returning computation-unit columns could also avoid reversing and reapplying scale factors, but the measured output-adaptation share is small.
3. **Batch small analyses.** Submit many compatible profiles in one `Timecourses` object to amortize dispatch and xarray construction. Measure the actual workload: combining profiles changes output size and worker selection as well as call count.
4. **Tune workers against both dimensions.** Compare serial and automatic runs on the same machine. Benchmark profile counts near the Python threshold, long profiles and concurrent callers before changing shard sizes or thresholds. Avoid inferring thread efficiency from the default speedup alone.
5. **Profile inside Rust before changing its algorithms.** Native sampling or kernel microbenchmarks can distinguish terminal-window fitting, AUC integration, allocations and binding conversion. The serial Python hotspots identify costs the Rust path already avoids; they do not identify Rust hotspots. The optimistic binding-removal bound applies only to changes inside that boundary with all other costs held constant.

## What the memory column says

`peak RSS MB` is the process high-water mark after both uninstrumented engines have run, before profiling. It includes imports, fixture construction, warm-up validation and both engine allocations. It is not a per-engine comparison or an incremental allocation measurement. Allocator retention can keep RSS above the size of live arrays.

## What the kernel does not carry

An analysis `pkpdutils-core` does not carry takes the python path, with the numbers and the timings it always had, and no table of this page covers it: the two columns would be the same engine twice. The dispatch decides once per batch, and the reason it gives is the kernel's own sentence:

<!-- refusals:start -->
| analysis | the reason the dispatch gives |
|---|---|
| an effect curve | the analysis of an effect curve (Kind::Effect) |
| the acceptance criteria of the terminal phase | the acceptance criteria (NcaOptions::acceptance) |
| a named partial area | the named partial areas (NcaOptions::partial_aucs) |
| routes which differ in their dosing | a batch of several routes of which only some have a multiple dose row, whose variables the python core reports per route |
| the reporting units of a result | the unit conversion of a result (NcaOptions::units) |
<!-- refusals:end -->

A batch of several routes is analysed by whichever engine the dispatch picked, in one call, and the result is the same: the kernel reports the union of the variables of the routes in its own order and the host sorts it into the order the python core would have reported, so the columns of a table do not depend on which engine ran.

## Reproducing this page

```bash
uv run python scripts/benchmark.py engines          # the table above
uv run python scripts/benchmark.py --write --repeat 15 # speedups, serial control, phases and JSON
uv run python scripts/benchmark.py --check          # exit 1 if the page no longer matches the cases
uv run python scripts/benchmark.py all --repeat 5   # the hot paths of the package as well

# Save phases, raw samples and independent Python profiles for the same run.
uv run python scripts/benchmark.py --write --repeat 15 --profile-dir /tmp/nca-profiles

# Controlled worker comparison without replacing the automatic-worker page.
uv run python scripts/benchmark.py engine-100000x12 engine-10000x200 \
  --n-workers 1 --repeat 9 --profile --output /tmp/nca-serial.json

# Read the independent function profile interactively.
python -m pstats /tmp/nca-profiles/engine-10000x200-kernel.prof
```

The numbers are of the machine named above the table, and of its load on that day. They are comparable against another run of the same script on the same machine, which is what a change to either engine is measured with, and they are no promise about another machine: the crossover in particular moves with the cost of a python call against the cost of a memory copy. Rerun `--write` after a release of `pkpdutils-core`, after a change to the dispatch or to the python core, and on the machine whose numbers you want to quote.

`--check` compares what the script controls: the cases, size labels, timing and phase columns, numeric cells, and the refusal table. It does not impose machine-specific speed thresholds. `tests/docs/test_benchmark.py` runs this check; profiling tests check accounting, actual engine selection and restoration of instrumentation. Timings and measured summary sentences are written by `--write`.

The JSON preserves provenance and dispersion so a rerun can be assessed instead of comparing rounded headline ratios. The phase percentages and binding-removal bound are generated from that run, alongside the tables.
