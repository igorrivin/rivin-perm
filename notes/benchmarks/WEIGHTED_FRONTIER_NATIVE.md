# Compiled weighted-matching frontier

This source tool computes the complete weighted partial-matching coefficient
vector for a rectangular matrix of nonnegative integers. It is a serial C++17
frontier recurrence with exact modular arithmetic and Python CRT reconstruction.
It is available in a source checkout or source distribution. It is not installed
by a wheel or selected by `weighted_rook_polynomial`.

## Build and run

Build once in a new directory; compilation never occurs inside evaluation:

```sh
python notes/benchmarks/weighted_matching_frontier_native.py build \
  outputs/weighted-frontier-local-build
```

For a JSON matrix file containing `[[1,2,0],[0,3,4]]`:

```sh
python notes/benchmarks/weighted_matching_frontier_native.py evaluate \
  outputs/weighted-frontier-local-build/weighted_matching_frontier_native.so \
  matrix.json --max-states 100000 --max-workspace-bytes 67108864
```

The coefficient vector is `[1,10,15]`: the empty matching, all one-edge
matchings, then all two-edge matchings. The output includes the complete CRT
transcript, coefficient bounds, row order, frontier widths, native workspace
peaks, build/source hashes and timing components. Use `--row-order 1,0` to
specify an order; the default uses the existing deterministic greedy planner.

From Python in a source tree:

```python
from notes.benchmarks.weighted_matching_frontier_native import FrontierBaseline

solver = FrontierBaseline(
    "outputs/weighted-frontier-local-build/weighted_matching_frontier_native.so"
)
result = solver.evaluate([[1, 2, 0], [0, 3, 4]])
assert result["coefficients"] == (1, 10, 15)
```

Builds retain a source snapshot, compiler command/version, logs and hashes.
The loader checks the recorded library and current source identities. Use a
fresh build directory after source changes; an existing build is never silently
overwritten. GCC's C++17 compiler is required by the build helper; there are no
additional native library dependencies or OpenMP/GPU requirements.

## Exactness

For matrix `A` with shape `m` by `n`, coefficient `r[k]` sums the products of
weights of all matchings of size `k`. Results have length `min(m,n)+1`, including
trailing zeros. Empty or zero-column input returns `(1,)`. Arbitrary-size
integer weights are reduced before conversion to fixed-width C buffers;
negative or nonintegral weights are rejected. Boolean and `__index__` weights
are accepted, matching the Python reference's contract.

Each native state records which currently live columns are occupied and carries
a cardinality polynomial. A row can be skipped or matched to one unused
neighbor. After the row, expired columns are forgotten whether matched or not.
This preserves partial matchings and their lower-degree coefficients.

If `s[i]` are row sums and `t[j]` column sums, the coefficient of degree `k`
in each of `product(1+s[i]*x)` and `product(1+t[j]*x)` bounds `r[k]`: each drops
one endpoint-exclusion condition. Their coefficientwise minimum gives a
rigorous nonnegative bound. Distinct 31-bit primes are generated until their
product strictly exceeds every bound. Reconstruction checks canonical residue
vectors, coverage and all coefficient bounds. No floating-point arithmetic is
used to obtain the coefficients.

The original nonzero support remains present even if an edge weight vanishes
modulo a chosen prime. Zero polynomials retain their structural states too, so
prime-dependent cancellation cannot conceal a state-limit failure. The Python
frontier shares input normalization and row ordering only; its independent
counting recurrence remains available for validation.

## Limits and failures

The default native workspace limit is **256 MiB** and the default state limit
is **1,000,000 states per layer**. Native memory is allocated as needed.

`max_workspace_bytes` caps simultaneous **requested C++ heap allocation bytes**:
hash nodes, hash bucket arrays and coefficient-vector storage. The shared
accounting includes both row layers, old/new buckets during rehash, and growing
polynomial lengths. This is an allocation-payload limit. Allocator metadata,
stack, Python planning/bounds/CRT objects, retained transcripts, and caller-owned
C input/output buffers are outside it. It is not a process RSS or elapsed-time
limit. An underlying allocation failure is also reported as failure.

Reusable 64-bit mask positions permit more than 64 global columns, but the union
of previously live columns and the current row's support must fit **64 slots**.
A closing slot is recycled only after the row completes. Therefore even a
single row with 65 neighbors exceeds this implementation's width limit. A long
diagonal or narrow band can use many more than 64 global columns. Row ordering
can change whether the width/state limits are reached; it does not change the
mathematical answer. Dense inputs remain exponential.

The Python interface raises `FrontierMemoryLimitError`,
`FrontierStateLimitError` or `FrontierWidthLimitError` on those limits. The CLI
emits `status: RESOURCE_LIMIT`, `exact: false` and exits with code 2. No truncated
coefficient vector is returned as an exact answer. Invalid inputs and unexpected
native failures also terminate without an exact result. The C ABI validates
its plan and canonical residues and leaves its output vector untouched on
errors; its caller must provide correctly sized accessible arrays.

Total evaluation time includes normalization, row planning, bounds, prime
selection, buffers, input reduction, native allocation/recurrences and CRT.
Build and loader time are excluded. `call_seconds` includes native allocation
and plan validation, rather than measuring only arithmetic instructions.

## Verification and remaining work

Run the focused suite with a compiler required:

```sh
PERM_REQUIRE_NATIVE_TESTS=1 python -m pytest -q \
  tests/test_weighted_matching_frontier_native.py
```

The tests cover exhaustive small definitions, rectangular and permuted inputs,
large weights, modular-zero support, high mask bits, long narrow graphs, native
ABI failures and exact byte/state boundaries. This implementation phase does
not establish a speedup. A retained multi-family comparison against the Python
reference and other compiled methods is the next phase before public dispatch
integration. The September 7 weighted study continues to describe its original
Python frontier measurements; those evidence files are unchanged.
