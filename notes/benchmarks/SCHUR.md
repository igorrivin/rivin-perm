# Exact Schur and circulant CUDA comparisons

[schur_compare.py](schur_compare.py) compares fixed-sign Glynn and
rotation/complement necklace traversal using unchanged kernel definitions from
`perm_modp_cuda.cu`. It uses public, seeded inputs and requires exact modular
equality. The companion [wrapper](schur_compare_wrapper.cu) supplies matched
arithmetic and additional template instantiations; it does not change production
dispatch. The [polynomial comparison](README.md) is a separate benchmark.

## Prepare, compile, and run

Run these commands from the repository root. Preparation uses Python's standard
library and does not initialize CUDA. Output directories must be fresh. Prepare
both profiles before measuring to freeze all 55 cases:

```sh
python notes/benchmarks/schur_compare.py prepare \
  --output outputs/schur-benchmark-pilot --profile pilot
python notes/benchmarks/schur_compare.py prepare \
  --output outputs/schur-benchmark-main --profile main
```

Read each output directory's `PROTOCOL.md` and `manifest.json`. The manifest
contains complete base vectors, full-matrix hashes, exact references where
available, source hashes, and the fixed call schedule. Use the archived runner
for later commands so subsequent checkout edits cannot change the experiment.

Compilation requires a CUDA toolkit and an explicit architecture. Compilation
alone does not execute the generated program or initialize the GPU:

```sh
python outputs/schur-benchmark-pilot/frozen/schur_compare.py run \
  --output outputs/schur-benchmark-pilot --nvcc nvcc --arch sm_90 --compile-only
python outputs/schur-benchmark-main/frozen/schur_compare.py run \
  --output outputs/schur-benchmark-main --nvcc nvcc --arch sm_90 --compile-only
```

`sm_90` targets the GH200 used in this project's campaign. Choose the appropriate
explicit target for a different device and use fresh archives for a new hardware
comparison. Compilation uses `-O3`, C++14, and compiler resource reporting; it
does not enable fast math. Compiler version, command, logs, and executable hash
are retained.

The following commands **execute CUDA kernels**. Arrange exclusive access to the
GPU and record competing activity, clocks, power settings, and driver state in
the campaign notes. The public runner does not stop model servers or other jobs.

```sh
python outputs/schur-benchmark-pilot/frozen/schur_compare.py run \
  --output outputs/schur-benchmark-pilot --arch sm_90 --timeout 1800
python outputs/schur-benchmark-pilot/frozen/schur_compare.py verify \
  --output outputs/schur-benchmark-pilot

python outputs/schur-benchmark-main/frozen/schur_compare.py run \
  --output outputs/schur-benchmark-main --arch sm_90 --timeout 1800
python outputs/schur-benchmark-main/frozen/schur_compare.py verify \
  --output outputs/schur-benchmark-main
```

The pilot has dimensions `M=4,6,10,12,16,18` and 30 cases. The main profile has
`M=22,28,30,36,40` and 25 cases. Each dimension has one Schur minor, two dense
circulants, and two period-two circulants. The dense and periodic seeds are
`2026090501` and `2026090502`. Each dimension uses one shared prime: the largest
permitted prime below `2**32` congruent to one modulo `M+1`.

The separately prepared `frontier` profile contains five order-46 minor cases
and requires `run --allow-large`. It is not part of the 55-case pilot/main
campaign and needs its own execution budget. These algorithms take exponential
time; the profile name is not an estimate of runtime on another device.

## Limits and resumption

`--timeout` bounds one case attempt, including its warmup and measured calls;
it is not a separate allowance for every kernel. `--max-cases` bounds the number
of cases attempted in one invocation. `--max-seconds` bounds the run's case
execution interval, excluding compilation and initial archive validation, and
may interrupt a case. A clean limit stop produces a `PARTIAL` report; a timed-out
or failed worker exits nonzero and retains its completed calls and logs.

```sh
python outputs/schur-benchmark-main/frozen/schur_compare.py run \
  --output outputs/schur-benchmark-main --arch sm_90 \
  --resume --max-cases 5 --max-seconds 3600 --timeout 1800
```

Resumption executes only missing call slots. Successful calls are never repeated
or replaced by faster measurements. The manifest, archived sources, executable,
inputs, prior results, and schedules are checked before reuse. Exact mismatches
remain fatal on resumption. An advisory lock rejects concurrent writers to the
same archive. SIGTERM/SIGINT and worker timeouts terminate the owned worker
process group and retain the attempt; abrupt machine or parent-process loss can
leave incomplete attempt metadata, which requires explicit review.

A resumed case can span CUDA contexts without a new warmup, because completed
slots must not be repeated. Reports retain every attempt and set
`timing_run_contiguous=false` for such cases. Inspect `noncompleted_attempts` and
`all_attempts_completed_normally`; numerical completion alone does not mean an
uninterrupted timing run.

## Output and timing contracts

Both methods receive the same circulant:

```text
A[i,j] = base[(j-i) % M]
```

They return `per(A) mod p` using the same unsigned 32-bit pseudo-Mersenne
multiplication. Primality, root order, and the exact two-fold reduction bounds
are checked. For a Schur case, `M=n-1` and multiplying the minor residue by `n`
recovers the full Schur residue. This benchmark measures one modulus per case;
it does not measure full-integer CRT reconstruction.

Every case has one warmup pair and five measured pairs, with method order
alternating between pairs. The primary `wall_ms` covers the complete wrapper
method call: host matrix/frontier construction, allocation, transfers, kernels,
reductions, modular normalization, and cleanup. Parsing, process startup, initial
CUDA context creation, compilation, and archive I/O are outside this interval.

`device_interval_ms` is the CUDA-event interval spanning kernels and reductions,
including intervening host launch/synchronization gaps. It is not a sum of pure
kernel durations. `setup_ms` and `finalize_ms` are separately recorded host wall
intervals; they do not form an exact additive decomposition with the event time.
`explicit_device_bytes` and `host_payload_bytes` account for explicit buffers
and payloads. They are not measured peak RAM or total CUDA memory usage.

The reported primary speedup is the median of the five paired
`fixed_sign.wall_ms / necklace.wall_ms` ratios. Values below one favor fixed-sign
traversal. Retain small and losing cases; a faster device interval does not
necessarily imply a faster complete call. Five repeated measurements describe
the recorded run, not a confidence interval or a universal speedup.

## Exactness and implementation scope

- All `M<=12` cases use an independent Python-integer subset-Ryser reference.
- Schur cases also check the archived public integer table, reduced modulo the
  prime and divided by `n` in the field.
- Period-two cases use an independent two-class counting formula at every size.
  For `M=2q` and base `(a,b,a,b,...)`, the permanent is
  `(q!)**2 * sum(binomial(q,j)**2 * a**(2*j) * b**(2*q-2*j), j=0..q)`.
- Larger dense cases establish exact agreement between the two traversals.
  That agreement is not an independent full-permanent certificate.

The periodic reference is itself an efficient specialized algorithm. These
controls test the traversals and their behavior on repeated structure; they do
not show that either exponential method is the best solver for periodic inputs.

The unchanged production necklace launcher supports only `M=40,46`. Its
fixed-sign pseudo-Mersenne launcher supports even `M=28,...,44`; order 46 takes
a different production fallback. The benchmark instantiates both unchanged
pseudo-Mersenne templates at every registered dimension to avoid confounding
orbit reduction with arithmetic width. These extra instantiations are benchmark
configurations, not new public API support.

Launch settings are fixed before timing: 256 threads per block; the production
fixed-sign output-count policy; four necklace blocks per multiprocessor; and
necklace prefix depth `min(M,24)`, except depth 28 at `M=46`. They are not retuned
for each matrix family. Historical frontier timings retain their separate
provenance and are not replaced by this current-source comparison.

## CPU-only audit and retained files

`verify` reads and checks data without loading a CUDA library or running the
executable. It validates frozen inputs, sources, executable linkage, schedules,
device/configuration fields, exact residues, and repeated timing records.
Use `--allow-partial` only when intentionally auditing an incomplete campaign.

A source/data release may deliberately omit the compiled executable while
retaining `build/build.json` and its recorded hash:

```sh
python notes/benchmarks/schur_compare.py verify \
  --output PATH_TO_RELEASE_BENCHMARK --allow-missing-binary
```

That explicit option still checks attempt-to-build hash linkage and the raw
records, but reports `binary_hash_checked=false` when the executable is absent.
It does not establish the missing executable's bytes or rerun any GPU arithmetic.

The principal retained files are:

| Path within each archive | Content |
| --- | --- |
| `manifest.json`, `PROTOCOL.md` | Frozen cases, schedule, identities, and interpretation |
| `frozen/` | Exact production source, wrapper, runner, and public Schur table |
| `build/build.json`, `build/compile.*` | Compiler/build provenance and logs |
| `cases/ID/attempt-NNN/` | Exact input, remaining schedule, raw stdout/stderr, and attempt metadata |
| `cases/ID/result.json` | Per-case summary, including all samples |
| `report.json` | Recomputed campaign summary after a successful runner invocation |

The benchmark JSON records use `{ "sha256": ..., "payload": ... }`; the digest
covers canonical JSON of the payload. Hashes establish artifact consistency,
not mathematical correctness by themselves. Raw attempted results are retained
even if an exact mismatch prevents a final campaign report.

For an independent, bounded CPU check of the orbit identity, generator, and
modular row updates, use [check_schur_orbits.py](check_schur_orbits.py):

```sh
python notes/benchmarks/check_schur_orbits.py \
  --source perm_modp_cuda.cu --output outputs/schur-orbit-cpu-audit
python -m pytest -q tests/test_schur_benchmark.py
```

The orbit checker requires an ordinary C++17 compiler and a fresh output
directory. It checks the expected production-source hash, extracts selected
production logic into ordinary C++, and compares bounded cases with independent
orbit enumeration and literal permutation sums. It does not validate CUDA
compilation, GPU execution, or the full frontier traversal. Keep Python
assertions enabled when running it.
