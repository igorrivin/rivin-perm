# Complete permanent polynomial comparisons

For the separate exact Schur/circulant CUDA comparison and CPU orbit audit, see
[SCHUR.md](SCHUR.md).

This checkout includes a public, seeded, CPU-only benchmark for complete integer
coefficient vectors. It imports no historical evaluator inputs or answer files.
The default sweep is a bounded pilot, not evidence of a universal speed record.

## Run and resume

Build the project native extensions using the project's normal build procedure;
the benchmark deliberately refuses Python fallbacks. A C++17 compiler is also
required. From the project root:

```sh
python notes/benchmarks/polynomial_compare.py \
  --output outputs/polynomial-comparison-pilot
```

Defaults are sizes 6, 9, 12, weight scales 1 and 100, one CPU thread, one warmup,
and three measured repeats. The budget is 600 seconds, checked **between complete
method calls**; a single exponential call can overrun it. Exit code 2 means a
clean budget stop. Repeat the identical command to resume from the last complete
method call. No background process is launched. Output is written atomically
after each method, and an advisory lock prevents concurrent writers.

The first sweep contains 156 cases: affine and principal contracts each include
nine matrix families; the nonnegative rook contract excludes signed dense input.
"Binary" denotes a random zero/one support multiplied by the selected scale;
scale one exercises the public `rook_polynomial` API, scale 100 its weighted
counterpart. The other families are dense, 20% sparse, bandwidth-one, diagonal,
blocks of at most three rows/columns, two repeated row types, rank-one, and signed
dense. Matrix seeds are derived independently from the full case description;
changing sweep order does not change an input. All matrices are recorded.

After calibrating the pilot, an explicit larger sweep can be run:

```sh
python notes/benchmarks/polynomial_compare.py \
  --output outputs/polynomial-comparison-larger \
  --sizes 15 18 --scales 1 100 --allow-large --max-seconds 1800
```

Orders above 12 require `--allow-large`; the hard limit is 20. Use `--families`
and `--contracts` to bound a calibration further. Matrix definitions, repeat
counts, interpreter, CPU identity/affinity, sources, and binary hashes must match
on resume. Source changes require a new output directory. The time budget may
change on resume. Do not change the source midway through an evidence run.

Export all complete cases to CSV and a Markdown table, preserving wins and losses:

```sh
python notes/benchmarks/summarize_polynomial_comparison.py \
  outputs/polynomial-comparison-pilot outputs/polynomial-comparison-larger \
  --output outputs/polynomial-comparison-tables
```

The exporter rechecks archived sources and replays every stored modular/CRT
vector before reporting. It displays common and public prime counts separately;
its end-to-end ratio adds shared bound/prime planning to each common-prime
alternative before comparing with the public exact API.

## Equal output contracts

Every vector is in ascending degree order and contains exactly `n+1` coefficients,
including trailing zeros:

| Contract | Output |
| --- | --- |
| Affine | Coefficients of `per(C + xL)` for integer matrices C and L |
| Principal | Coefficients of `per(I + xA)`; sums of principal permanents by size |
| Weighted rook | Sums of products over partial row/column matchings, by size |

For the rook contract, affine methods compute `per(J+xA)` and divide coefficient
`k` by `(n-k)!` in each prime field. The prime always exceeds the order. This
conversion is timed and its full output retained.

The following are compared on **identical inputs, primes, exact integer output,
and one CPU thread**:

* **Polynomial-valued Ryser:** independent C++17 Gray-code implementation over
  the polynomial ring. It uses sparse column updates, handles constant/linear
  zero factors, and uses the same Mersenne-31 fast-reduction opportunity as the
  production generic polynomial kernel. Other 31-bit primes use integer modular
  division. This is a compiled implementation of an established algorithm, not
  a Python strawman.
* **Occupied-column weighted-rook DP:** independent C++17 skip-or-match row DP,
  only for the rook contract. Descending mask updates use one `2^n`-entry array;
  sparse rows omit zero transitions. It has exponential memory and can win on
  structured or small instances.
* **Scalar evaluation/interpolation:** the existing compiled general scalar
  `perm_c_ext.permanent_mod_p` dispatcher (or explicit native Ryser for older
  extensions), evaluated at 0 through n, followed by exact Newton interpolation.
  Principal/rook values at zero are known (`per(I)=1`, `per(J)=n!`) and are recorded
  analytically without calling the exponential scalar kernel.
  Python performs matrix preparation and the small interpolation; both are
  separately timed. All scalar samples are retained.
* **Native polynomial API:** the public native affine/principal modular API,
  with factorial unscaling for the rook contract.

All four paths use independent row-product absolute coefficient bounds and the
same descending 31-bit primes. For nonnegative vectors the prime product exceeds
the maximum bound; signed vectors require twice that bound. Every method performs
and records its own full CRT reconstruction. These simple common bounds are
deliberately separate from the production bound/parity/fusion optimizations.

The **public exact API** is also measured: `affine_permanent`,
`principal_permanent_polynomial`, `rook_polynomial`, or
`weighted_rook_polynomial`, always with `backend="native", num_threads=1`.
This is a separate end-to-end comparison. In particular, specialized rook APIs
choose their own smaller pseudo-Mersenne primes, fuse primes, and use parity
lifts. Their prime count and bounds differ from the common-prime experiment.
Temporary Python wrappers capture every production residue row, prime, and fused
call duration, without modifying the implementation. The instrumentation overhead
is included. Do not present that lane as a matched-prime kernel comparison.

## Timing, evidence, and limits

Each round cyclically rotates the method list and reverses it on alternating
rounds; the exact execution order is retained. Three repeats are not a fully
balanced ordering experiment. Warmups are recorded but excluded from medians. Every measured result must match
the full exact vector; a mismatch stops the campaign and archives the failing
result. No degree is checked only by evaluation or hashing.

`manifest.json` records configuration, source and native binary hashes, compiler
command/version, machine/kernel/Python identity, CPU affinity, and thread settings.
The baseline is built with `-O3 -std=c++17 -fPIC -shared`, with no architecture-only
flags or OpenMP. The public extension binary hashes identify the compared builds;
their historical compiler invocation is not inferred from their sources.
Source snapshots and a hashed build record are retained. Compilation is reported
separately from steady-state timings.

Each `cases/*.json` contains the complete matrices, independent bounds, common
prime list, every warmup and measurement, each prime's full coefficient vector,
all interpolation samples, CRT and setup timings, exact output, and the public
API's actual prime/residue transcript. The wrapper SHA-256 detects accidental
changes; it is **not an external digital signature**. Resume also replays recorded
CRT/interpolation and checks agreement of every saved exact vector and public
residue transcript. Checkpoint IO is outside method timings. A method interrupted
before its checkpoint is rerun.

Method `total_seconds` includes modular input preparation, all kernel calls,
postprocessing, and CRT. Shared case generation/bound/prime planning is recorded
separately. The public exact total includes its own planning. The latter's
non-residue time combines planning, CRT, validation, and instrumentation; it is
not labeled as a pure CRT timer. Do not compare an isolated C++ kernel duration
with another method's entire API duration.

Memory has intentionally limited scope: C++ vector payload formulas are recorded
(`4*2^n + 4*n` bytes for rook DP's state/column vectors; `4*(3*n+1)` for Ryser's
row sums/product). They exclude input/output buffers, allocators, and Python.
Linux process peak RSS is cumulative across the process's cases and algorithms;
it is **not a per-method peak** and does not support per-method RAM rankings.
An isolated-process memory campaign is still needed before publishing peak-RAM
comparisons. These one-thread pilot runs may overlap independent GPU verification
on the same host, so clean isolated reruns and thread-scaling experiments remain.

The suite does not yet implement Sage/Butera–Pernici, bandwidth-aware transfer
methods, structure factorization, degree truncation, or GPU polynomial kernels.
It establishes compiled dense baselines and explicitly includes structures where
generic exponential traversals can lose; it is not an exhaustive best-algorithm
comparison for those structures. Reports should preserve losses and crossover
points, together with actual prime counts and weight scale.

## Prior work and validation

The separate [weighted-rook source selector](WEIGHTED_ROOK_DISPATCH.md) evaluates
a conservative choice between the public implementation and the independent
Python/compiled frontier tools. Its frozen retained/fresh comparison is described
in the dispatch results (maintainer archive; not included). It is opt-in and
does not change installed public dispatch.

Polynomial-valued Ryser and factorial reweighting of `per(J+xA)` already appear in
Gao, Jiang, Li, Gershman, and McKay, *Statistical Eigenmode Transmission over
Jointly-Correlated MIMO Channels* (2009), Section III.C:
<https://arxiv.org/abs/0903.1952>. The baseline is an independent implementation
of that established approach; no novelty of the general identity is claimed.

The independent small-input tests enumerate permutations and constant/linear
choices, or partial matchings directly. They exercise signed entries, composites
for methods that allow them, 31-bit modular boundaries, multi-prime exact outputs,
interpolation, captured production residues, failing mismatches, and corrupted or
incompatible resumes:

```sh
python -m pytest -q tests/test_polynomial_benchmark.py
```
