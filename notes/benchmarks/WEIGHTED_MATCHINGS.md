# Exact weighted matchings and occupancy

This source study computes the weighted matching-size vector once, then uses
it to obtain exact partition functions and occupancy statistics over an
activity sweep. It compares both computation methods and approximation error.
The inputs are new public synthetic graphs, not private evaluation instances.

The completed September 7 study and its measured tradeoffs are summarized in
WEIGHTED_MATCHINGS_RESULTS.md (maintainer archive; not included), with
raw data, figures and provenance (maintainer archive; not included).

A separate [compiled weighted-frontier source tool](WEIGHTED_FRONTIER_NATIVE.md)
now provides exact modular/CRT evaluation with native allocation and state
limits. It has its own explicit build/evaluation interface; the retained study
and its Python-frontier measurements remain as originally recorded.

Build the optional native extensions, then run a bounded local study:

```sh
python setup.py build_ext --inplace
python notes/benchmarks/weighted_matching_study.py \
  --output outputs/weighted-matching-study-local
python notes/benchmarks/summarize_weighted_matching_study.py \
  --input outputs/weighted-matching-study-local \
  --output outputs/weighted-matching-study-local/report
```

Defaults are orders 8 and 12, six graph families, three public seeds, scales
1 and 100, one warmup and three measurements. Every method runs serially in
a fresh process, pinned to one allowed CPU when the host exposes CPU-affinity
APIs. On macOS the default is explicitly unpinned; requesting `--cpu` on a
host without those APIs fails before creating an output directory or building
baselines. The manifest records the available affinity (or `null`), and each
worker records whether pinning was supported and actually applied.
Compilation, imports, startup and
checkpoint IO are excluded from method timings; normalization, planning,
allocations, reconstruction, and result instrumentation are included.
Process wall times are recorded separately. Sequential isolated workers do not
prove that other users or services on the host were idle.

An explicit larger run is:

```sh
python notes/benchmarks/weighted_matching_study.py \
  --output outputs/weighted-matching-study-larger \
  --sizes 8 12 16 18 --allow-large
```

The hard maximum order is 20. Each worker has a timeout (default 90 seconds;
application workers at least 300 seconds). The frontier has a retained-state
guard (default 8192); it does not silently return a truncated answer. Guard
and timeout outcomes are reported rather than treated as timings of successful
computation. Its polynomial payload and two live layers require additional
memory beyond the mask count. Exponential runs can still take substantial
time. The script starts no detached background job.

Repeat the same command to resume. Changed source, native binaries, compiled
baselines, machine identity, configuration or prechange source require a new
output directory. A directory lock prevents simultaneous writers. Optional
`--previous-source PATH` adds a prechange Python implementation using the same
current native binary, to measure dispatch/planner changes fairly. This is
not a reproduction of historical native builds.

**Inputs and exact outputs.** Dense, 20%-sparse-with-diagonal, tridiagonal,
diagonal, size-three blocks and repeated-row matrices use integer weights 1..9.
Hash-derived streams depend on family, order and seed, but not scale: the
scale-100 matrix is exactly 100 times the scale-1 matrix. Every vector must
therefore satisfy `r_k(100W)=100**k*r_k(W)`. The selected seeds do not support
universal ensemble or record claims.

The native API, compiled occupied-column DP and compiled polynomial-valued
Ryser return the same complete vector of exact integers. The two independent
compiled methods share row-product bounds and 31-bit CRT primes. The native
method retains its own tighter bounds, parity lifting, fused primes and exact
component dispatch, so its time is an end-to-end comparison with a different
planner. Full modular vectors and exact outputs are retained. The independent
frontier implementation uses Python arbitrary integers and keeps a polynomial
per used-column state. It implements the nilpotent-variable/forgetting
approach described by
[Butera and Pernici](https://arxiv.org/abs/1406.5337); it is not a timing of
Sage's implementation. Sage is not installed on this host.

The September 8 dispatcher also evaluates decomposed factors with smaller
side at most three using at most eight arbitrary-integer subset states.
These factors have explicit exact-component transcripts, independently
replayed by enumerating injective assignments. They do not have CRT residues;
their timings are recorded separately from residue calls. Larger factors and
standalone connected full-order inputs retain the original CRT path. Historical
transcripts are still understood by the current arithmetic validator, while
whole-study source checks continue to require their recorded versions.

Fresh-process `ru_maxrss` supplies a separate peak for each method and case.
It is normalized to bytes using the host's units: bytes on macOS, KiB on Linux.
It includes that method's interpreter/imports, warmup and repeats. The startup
high-water mark is also recorded. Subtracting the two is not an allocator-level
peak-memory measurement, and small differences may be hidden by startup.

**Scientific quantities.** For coefficients r_k and nonnegative rational z,

    Z(z) = sum_k r_k z^k
    P(K=k) = r_k z^k / Z(z)
    E[K] = z Z'(z) / Z(z)
    Var(K) = z Z'/Z + z² Z''/Z - (z Z'/Z)².

All are computed as `Fraction` values; log Z is a numerical convenience for
figures. The APIs distinguish exact quantities from numerical approximations.
The current study uses square integer matrices. Rational edge weights W=A/D
can be represented by r_k(W)=r_k(A)/D^k; measured real weights require an
explicit approximation/error model. Matching size is not an individual-edge
posterior. The last coefficient need not represent an attainable matching.

At each of 13 rational activities between 1/1000 and 1000, a second C++ DP
independently computes only Z and the first two moments. For z=p/q its row
transitions multiply skips by q and occupied edges by pW_ij. The three integer
sums F_j=sum_k k^j r_k p^k q^(n-k), with k^0=1, are reconstructed by CRT and
checked against the coefficient-derived rational answers. This is a genuine
per-activity solver, not a deliberately repeated full-vector computation.
Its independent bound drops column exclusion:
`F0 <= B = product_i(q+p*sum_j W_ij)`, `F1 <= nB`, `F2 <= n²B`.

Measured query break-even compares coefficient setup plus cheap queries with
this direct exact solver. Their common outputs are Z, mean and variance; the
coefficient query additionally supplies the full size distribution. A formula
using average query times is a workload estimate, and the report separately
shows the measured cumulative activity-grid totals. It is not a hardware-
independent crossover guarantee.

The substantive numerical comparison is monomer-dimer belief propagation:
cavity messages exclude the recipient edge, and Bethe normalization yields
log Z and mean occupancy. Log-domain evaluation avoids overflow; damping and
an undamped fixed-point residual govern convergence. An iteration cap reports
nonconvergence without supplying accepted estimates. The residual is not a
bound on model error. Compare mean error and partition error separately.
[Lelarge's matching analysis](https://arxiv.org/abs/1401.7923) motivates the
finite-activity recursion; weighted factors enter as zW_ij.

A deliberately simpler independent-edge model is also retained as a diagnostic:
edges are Bernoulli with probability zW/(1+zW), ignoring shared endpoints. Its
partition function is an upper bound, not a matching approximation guarantee.
Its mass above min(rows,columns) counts only one subset of invalid edge
configurations. Total variation comparisons here concern matching-size
distributions, not the distribution over individual matching configurations.

The report preserves all wins, losses, resource limits and BP convergence
outcomes. There is no new asymptotic algorithm claim: polynomial-valued
permanent evaluation and factorial rook reweighting are established prior art,
including [Gao et al., Section III.C](https://arxiv.org/abs/0903.1952).
