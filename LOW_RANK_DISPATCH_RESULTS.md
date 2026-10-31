# Exact low-rank integer dispatch

`matrix_permanent` now selects a bounded exact low-rank calculation on eligible
inputs. The existing generic path remains available explicitly, and is used
when the cost rule or any CRT prime's factorization declines the specialization.
The frozen candidate passed all 36 fresh whole-call comparisons: every exact
answer agreed, and all 15 selected low-rank calls were faster. Generic fallback
calls had small measured losses as well as wins; this is not a uniform speedup.

## Public interface

```python
from perm import matrix_permanent, LowRankLimitExceeded

A = [[1, 2], [3, 4]]
matrix_permanent(A)                      # auto; exact integer
matrix_permanent(A, method="generic")    # previous general computation
matrix_permanent(A, method="low_rank")   # require the bounded specialization
```

The new argument is keyword-only. Existing positional calls, integer and
NumPy input validation, signed reconstruction, and generic parallel settings
are preserved. `verbose=True` reports when the low-rank calculation is used.
Explicit low-rank calls raise `LowRankLimitExceeded` when a reconstruction
prime exceeds rank four or 50,000 homogeneous coefficients per side.

The low-rank kernel currently computes residues serially. Automatic selection
compares its estimated work with the generic path's available OpenMP threads
or CRT worker capacity. Floating-point, binary-specialized, and specialized
Schur entry points retain their own algorithms.

## Exactness and bounded work

For `A=UV` over a prime field, with U of size N by r and V of size r by N,
the coefficient identity is

\[
\operatorname{per}(A)=\sum_{|\alpha|=N}\alpha!
[x^\alpha]\prod_i(U_i x)\,[x^\alpha]\prod_j(V_{:j}x).
\]

The pairing of row and column latent-index assignments gives `alpha!`
permutations with the same multiplicities. The computation multiplies by
factorials rather than dividing by them, so the identity also holds in small
characteristic. Rank is determined by exact modular elimination, never SVD.

Every reconstruction prime is preflighted before coefficient summation. A
small rank modulo the first prime is not treated as evidence of small integer
rank. In auto mode any budget rejection returns the entire calculation to the
existing generic/parallel path; explicit mode raises. Unexpected arithmetic or
backend errors propagate. The row-product bound, prime sequence, CRT procedure,
and centered signed lift are shared with the existing exact API.

The policy frozen before the holdout uses N >= 20, rank at most four, and at
most 50,000 coefficients per side. Write

\[
G=N2^{N-1},\qquad
W_r=2r\binom{N+r-1}{r}+N^2r.
\]

The planner permits a rank only when `G >= 200*q*W_r`, where q is the existing
generic execution path's parallel capacity. It assumes ideal generic scaling
as a conservative comparison; the factor 200 accounts for Python coefficient
updates being costlier than native modular operations. It does not use a
runtime timing probe. The coefficient cap is a state-count limit, not an
isolated byte-memory guarantee.

## Development and fresh holdout

Measurements used an Apple M2 Pro, Python 3.11.17, the existing clang `-O3`
OpenMP generic extension, and `OMP_NUM_THREADS=8`. A requested thread setting
of 1 uses serial CRT; 4 and 0 use the existing generic parallel policy, with
0 resolving to the environment's eight OpenMP threads where that policy
chooses threaded residues. Process-pool startup is included when that is the
existing policy's choice. CPU pinning is unavailable on this Mac.

All times are complete-call medians of three measurements after one warmup,
with method order rotated. They include validation, bounds, rank planning,
coefficient construction, and CRT. The independent generic and low-rank
methods must agree exactly on every call. Timings are descriptive observations
on a shared machine, not significance tests or portable performance guarantees.

The development grid had 18 comparisons at orders 18, 20, and 22 and ranks
2–4, with serial and requested-four-thread generic execution. The rule avoids
the small rank-4 case and the parallel order-22/rank-4 case. Nine selected
comparisons improved by 5.86–320.71x.

The protocol and frozen source hashes (maintainer archive; not included)
were saved before the fresh holdout (maintainer archive; not included).
The holdout contains 12 matrices under three thread settings: signed and
positive low-rank products, rank-five controls, dense signed and binary inputs,
identity and triangular matrices, and a low-rank matrix perturbed by an
identity matrix. No holdout matrix has a zero row, zero column, or zero
permanent. The policy was not retuned after inspecting these results.

| Fresh input | Generic execution setting | Generic, ms | Auto, ms | Speedup |
|---|---|---:|---:|---:|
| Order 20, rank 2 | Serial CRT | 141.729 | 1.779 | 79.68x |
| Order 22, rank 3 | Serial CRT | 781.022 | 17.408 | 44.87x |
| Order 22, rank 3 | Automatic, OMP limit 8 | 174.156 | 17.698 | 9.84x |
| Order 22, rank 4 | Serial CRT | 904.733 | 166.180 | 5.44x |
| Order 24, rank 3 | Serial CRT | 3381.081 | 24.028 | 140.71x |
| Order 24, rank 3 | Automatic, OMP limit 8 | 811.571 | 24.282 | 33.42x |
| Order 24, rank 4 | Requested four threads | 798.190 | 210.450 | 3.79x |

All **15 selected** comparisons improved by **3.79–201.34x**. The other
**21 comparisons retained generic computation**, including three conservative
declines on low-rank inputs. Across those 21 calls, measured auto/generic
speed ratios ranged from **0.942x to 1.126x**. The largest observed slowdown
was **6.21%**, about **1.30 ms**, on an automatic-thread triangular input.
Its source, input, and raw samples remain in the capture.

There were **12 measured losses** among all 36 comparisons; all were fallback
calls. None exceeded the protocol's descriptive material-slowdown band of
both 10% and 100 microseconds. No loss was dropped and no revised thresholds
were fitted to the holdout. The driver also records explicit low-rank timings
on constructed eligible families, even when auto declines them.

## Reproduction and validation

Final pre-commit verification of the accumulated low-rank, portability and
research changes passed **2,205 tests, 70 skipped**, in 50.67 seconds with the
native extensions enabled, including source-distribution boundary checks.
The fresh validation record (maintainer archive; not included)
retains the command, source hashes and scope. This is a fresh correctness and
packaging check; the performance results above remain the earlier retained
measurements.

The packaged source benchmark has a bounded default:

```sh
python benchmark_low_rank.py --output low-rank-quick.json
```

The recorded native comparisons used the following settings; load a built
generic extension before the larger runs:

```sh
OMP_NUM_THREADS=8 python benchmark_low_rank.py \
  --output development.json --phase development --allow-large \
  --threads 1 4 --repeats 3
OMP_NUM_THREADS=8 python benchmark_low_rank.py \
  --output holdout.json --phase holdout --allow-large \
  --threads 1 4 0 --repeats 3
```

Captures retain the actual interpreter, command, platform, compiler, native
library hash, thread environment, matrix seeds/hashes, method choices, answers,
and every timing sample. Existing capture paths are rejected rather than
overwritten. Larger runs require the explicit `--allow-large` opt-in.

Production tests include exhaustive binary matrices through order 3, signed
rank-1/2/3/4 factors through order 7 against permutation enumeration, modular
factor reconstruction in characteristics 2, 3, and 101, NumPy integer limits,
row/column symmetries, an order-40 factorial oracle, cost/state rejection,
generic opt-out, error propagation, and an exceptional-first-prime example
that must fall back. Source-distribution tests also execute these checks from
an unpacked distribution without the research prototype. Final test outcomes
and capture audits are retained in
validation.json (maintainer archive; not included).

The native full suite passed **2,164 tests**, with 70 optional/device skips.
The portable full suite passed **1,913 tests**, with 175 optional/device skips,
including the strengthened source-distribution checks. An isolated import
from a built pure-Python wheel also reproduced an order-40 factorial oracle,
the generic and explicit low-rank methods, and the public budget exception.
