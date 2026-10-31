# Binary permanents: algorithms and application cases

The permanent of a zero-one biadjacency matrix is the number of perfect
matchings of the corresponding bipartite graph.  This repository therefore
keeps binary matrices as a separate exact problem class rather than sending
them through the general integer kernel.

```python
from perm_binary import binary_permanent

binary_permanent([[1, 1], [1, 1]])  # 2
```

The default input contract is strict: entries must be integral zeroes or ones.
`strict=False` explicitly replaces an integral matrix by its support matrix.
Rectangular inputs are rejected because this API counts perfect matchings of
both complete bipartition classes.

## Specialized exact algorithm

Before exponential work, the implementation:

1. finds a perfect matching, returning zero immediately if Hall feasibility
   fails;
2. uses alternating-cycle strongly connected components relative to that
   matching to factor the matrix into independent square blocks; and
3. bounds each block with the minimum of `n!`, row/column degree products, and
   rigorously rounded grouped Bregman--Minc bounds.

The native residue kernel uses fixed-sign Glynn, so it visits `2^(n-1)` sign
states.  It fixes the densest column, assigns low-degree columns to frequently
toggled Gray bits, and chooses the cheaper orientation between the matrix and
its transpose.  Signed row sums are small.  A Gray transition touches only
the set bits of one packed column; a precomputed ratio updates the product of
nonzero row sums while a separate counter handles zero factors.

The fast path uses 31-bit Montgomery arithmetic without division in the hot
loop.  A 64-bit Montgomery fallback preserves modular correctness for every
supported odd signed-64-bit prime greater than the matrix order.  Because a
binary permanent is nonnegative, exact CRT reconstruction needs a modulus
product greater than the upper bound, not twice that bound.

The portable fallback is a definition-level subset dynamic program with
arbitrary-size Python integers.  It is intended for correctness, small
problems, and installations without OpenMP; dense large blocks should use the
native backend.

## Live-column frontier algorithm

`binary_permanent_frontier` is an exact alternative for sparse matrices whose
support admits a favorable row order.  After each row, a state records which
currently live columns have already received their matching edge.  Processing
a row selects one adjacent unused column.  When the last row adjacent to a
column has been processed, that column must already be selected; the bit can
then be forgotten.  Different partial matchings with the same live state are
combined using arbitrary-size Python integers.

```python
from perm_binary import (
    FrontierStateLimitExceeded,
    binary_permanent_frontier,
)

K4 = [[0, 1, 1, 1],
      [1, 0, 1, 1],
      [1, 1, 0, 1],
      [1, 1, 1, 0]]

binary_permanent_frontier(K4)  # 9
```

The full signature is

```python
binary_permanent_frontier(
    matrix, *, row_order=None, max_states=500_000, strict=True
)
```

`row_order=None` invokes a deterministic greedy heuristic.  It tries to keep
the live-column frontier small, but it neither computes nor certifies an
optimal path decomposition.  A supplied order must contain every original
row index exactly once; it is interpreted before any internal reordering.
Runtime depends on the distinct reachable states, not on frontier width
alone, so even a plausible width estimate is not a memory guarantee.

`max_states` must be a positive integer.  It bounds the number of distinct
states retained in any one layer.  If the next layer would exceed it, an
explicit call raises `FrontierStateLimitExceeded`; it never changes algorithms
behind the caller's back.  The exception exposes the configured limit, the
minimum observed state count, and the number of processed rows.

The high-level `binary_permanent` dispatcher has deliberately different
semantics.  Only `backend="auto"` may try the frontier route on an eligible
matching-factor block.  If that attempt reaches its internal state cap, the
dispatcher discards it and continues with the exact native backend when
available, or the existing portable subset DP otherwise.  Explicit
`backend="native"` and `backend="python"` preserve their selected algorithms.

### Cubic adjacency matrices and cycle covers

For the adjacency matrix `A` of an undirected graph, `per(A)` counts directed
permutation cycle covers: every vertex has one selected outgoing and one
selected incoming arc.  In particular, a transposition selects both
orientations of one undirected edge and is a permitted directed 2-cycle.  This
is not the same statistic as an oriented spanning 2-factor that requires
cycles of length at least three.

Sparse cubic adjacency matrices are therefore natural frontier candidates,
but cubicity by itself does not promise that the greedy order will be good.
The asymmetric 42-vertex evaluation instance has exact permanent `1,361,137`.
Its independently recorded row order has at most 14 live columns and 2,944
reachable states, allowing an exact integer frontier computation in a tiny
fraction of the time required by generic sign-vector enumeration.  Other
cubic instances can have much broader frontiers; the state cap makes that
instance dependence explicit.  Dense matrices, and sparse matrices with poor
orders, remain better candidates for the native fixed-sign Glynn backend.

## Application 1: bipartite dimers

`grid_biadjacency` and `grid_dimer_count` construct parity-bipartitioned
Cartesian grids with open boundaries and selected periodic axes.

```python
from perm_binary_applications import grid_dimer_count

grid_dimer_count((4, 4))       # 36
grid_dimer_count((2, 2, 4))    # 121
grid_dimer_count((2, 3, 4))    # 1845
grid_dimer_count((4, 4), periodic_axes=(0, 1))  # 272
```

Open two-dimensional grids are planar Kasteleyn oracles, not a claim that the
generic permanent is the best planar-dimer algorithm.  Doubly periodic grids
live on a torus and admit fixed-genus Pfaffian treatments.  The proposed
`4 x 4 x 5` grid with periodic x/y axes and open z boundaries is instead a
genuinely nonplanar 80-vertex benchmark.  Its biadjacency matrix has order 40,
224 ones, and row/column degree multiset `5^16 6^24`.

```python
from perm_binary_applications import grid_biadjacency

matrix = grid_biadjacency((4, 4, 5), periodic_axes=(0, 1))
```

The certified bound is

`1,393,140,695,040,000,000`,

so two descending 31-bit CRT primes suffice.  The GH200 host CPU, using 64
OpenMP threads, computed

`per(matrix) = 2,551,276,235,535,120`.

The two required residues were:

| prime | residue | native kernel wall time |
|---:|---:|---:|
| 2,147,483,647 | 1,238,389,710 | 121.97 s |
| 2,147,483,629 | 1,259,774,250 | 120.42 s |

Their 242.38-second sum (4.04 minutes) excludes the independent unused-prime
holdout.  This is an exact finite-volume benchmark, not a novelty claim:
transfer-matrix and other specialized dimer methods also exist.

The optional source-only CUDA path computed the same two residues in 10.25
and 10.13 seconds, and the complete Python normalization, matching,
two-residue CRT, and validation path took 22.73 seconds.  That is a 10.66x
end-to-end speedup over the specialized 64-thread CPU call on this host.  The
sm_90 kernel uses 56 registers per thread, 1 KiB of shared memory per block,
and no stack or spills; a representative order-24 Compute Sanitizer run
reported zero errors.  It is built and imported separately through
`perm_binary_cuda.py` and does not add a CUDA dependency to the package.

As an algorithmically independent check, the standalone layer-transfer
verifier in `notes/frontier/verify_binary_grid_4x4x5_transfer.py` returned the
same integer in 2.54 seconds.  It is deliberately separate from the permanent
implementation and demonstrates why highly structured lattices should also
be compared with problem-specific transfer methods.

### Anisotropic polynomial

`weighted_grid_biadjacency` assigns an integer weight by edge axis, while
`axis_dimer_profile` computes small exact coefficient oracles.
`axis_dimer_profile_mod` uses the coefficient-valued Glynn backend to compute
the complete profile modulo any odd signed-64-bit integer, avoiding separate
evaluation and interpolation calls.  The corresponding constant and marked
edge matrices may instead be passed to `perm_polynomial.affine_permanent` for
certified exact integer coefficients via CRT.  For the open
`2 x 2 x 2` cube,

`Z(w) = 4 + 4 w^2 + w^4`.

For the `4 x 4 x 5` periodic-x/y target, the interpolation proposal in the
source conversation can be sharpened:

- each of the 16 length-five z-columns contains at most two vertical dimers,
  so `deg Z <= 32`;
- every layer has 16 vertices, forcing the number of inter-layer dimers—and
  hence the total vertical count—to be even;
- therefore `Z(w) = Q(w^2)` with `deg Q <= 16`, so 17 evaluations in
  `u = w^2` suffice rather than 41 generic values;
- at `w=0`, the five toroidal layers decouple, giving the exact holdout
  `Z(0) = 272^5 = 1,488,827,973,632`.

The exact-integer profile DP remains intentionally guarded to small orders.
The modular affine-polynomial backend scales exponentially in the
bipartition order but carries every coefficient through a single sign-vector
enumeration; exact large profiles can subsequently be reconstructed from
enough coprime modular runs when a coefficient bound is available.

## Application 2: restricted permutations

A binary matrix can encode allowed positions of a permutation.  Its permanent
counts the permitted permutations exactly.  In particular, `derangement_matrix`
returns `J-I`, whose permanent is the subfactorial `!n`.

Block diagonal all-ones matrices are also provided as exact Bregman--Minc
extremizers.  A matrix with `k` blocks of order `d` has permanent `(d!)^k`;
these cases test arbitrary-size results and the SCC factorization path.

## Application 3: Latin transversals, correctly qualified

A Latin transversal is a three-dimensional matching and is naturally a
three-dimensional zero-one permanent—not one ordinary matrix permanent.  The
reference `latin_transversal_count` instead uses the exact coefficient-
extraction identity

`T(L) = sum_S (-1)^(n-|S|) per(A_S)`,

where `A_S[i,j]` is one when the symbol `L[i,j]` belongs to `S`.  It makes
`2^n` ordinary binary-permanent calls and is explicitly guarded to small
orders.  Tests include cyclic group tables through order nine and the
order-four Klein group table, which has eight transversals; even cyclic tables
having zero must not be generalized to all even Latin squares.

## Application 4: partial matchings and rook polynomials

`perm_polynomial.rook_polynomial` returns all partial-matching counts of a
square binary matrix.  Entry `k` counts placements of `k` ones with distinct
rows and columns.  It uses the identity

`[x^k] per(J + xA) = (n-k)! r_k`

to obtain every coefficient in one coefficient-valued Glynn traversal per
CRT prime in the portable fallback. The native rook specialization instead
packs all required pseudo-Mersenne primes with the coefficient dimension, so
one sign-vector traversal produces every modular vector. Exact fixed-point
augmented-matrix Bregman--Minc bounds are applied directly to the
partial-matching counts. An independent characteristic-two Cauchy--Binet
determinant supplies the coefficient parities, which halves the CRT range and
selects the unique lift `x` or `x + P` below the certified bound.

On the dense SHA-generated order-33 evaluation board, this reduces four
sequential native CRT traversals (369.26 seconds on the GH200 host CPU) to one
fused traversal and exact reconstruction (44.00 seconds). Increasing the
order from 33 to 37 multiplies the fixed-sign state count by 16 and requires
five packed 27-bit fields rather than four. The fresh order-37 `/1` evaluation
instance completed in 1,212.81 seconds (20.21 minutes) on the same 64-core
host.

## Weighted partial matchings and occupancy

`weighted_rook_polynomial(W)` supports arbitrary nonnegative integer edge
weights. Its coefficient `r[k]` sums the products of the edge weights over
all size-`k` matchings. Disconnected bipartite support components factor
exactly, including diagonal and star components; the returned tuple still
contains `n+1` entries with trailing zeros.

For nonnegative rational activity `z`, `Z(z)=sum(r[k]*z**k)` is the finite
monomer-dimer partition function with unit monomer weights. The matching-size
probability is `r[k]*z**k/Z(z)`. In a source checkout:

```python
from fractions import Fraction
from perm_polynomial import weighted_rook_polynomial
from notes.benchmarks.weighted_matching_statistics import exact_occupancy_statistics

coefficients = weighted_rook_polynomial([[1, 2], [3, 4]])
statistics = exact_occupancy_statistics(coefficients, Fraction(1, 2))
assert statistics["partition"] == Fraction(17, 2)
assert statistics["mean"] == Fraction(20, 17)
assert statistics["variance"] == Fraction(110, 289)
```

One complete coefficient vector supports many activity queries. The exact
statistics helper retains rational probabilities and moments; only its
convenience logarithm is floating point. Its separate belief-propagation
comparison returns numerical approximations and explicit convergence status.
Convergence of the message iteration does not certify approximation accuracy.

The [public application study](notes/benchmarks/WEIGHTED_MATCHINGS.md) compares
native coefficients, compiled subset DP, polynomial-valued Ryser, and an
independent implementation of the Butera-Pernici frontier approach. It also
compares the cost of reusing coefficients with a compiled exact per-activity
partition-and-moment query. The study includes dense and structured cases,
fixed-matrix rescaling, per-method process memory, and retained losses.

## Boundary conventions

- A periodic axis must have even length to preserve the parity bipartition.
- Periodic extent two is rejected: its positive and negative bonds have the
  same endpoints, producing a multigraph that cannot be represented faithfully
  by a zero-one matrix.
- An odd-volume grid has zero close-packed dimer coverings.  The high-level
  count/partition/profile functions return zero (or `(0,)`), while the matrix
  constructors reject its unequal parity classes because the result would not
  be a square biadjacency matrix.
- Merely having three nontrivial dimensions does not prove nonplanarity; for
  example, `2 x 2 x k` grid graphs are planar.

## Primary references

- L. G. Valiant, [*The Complexity of Computing the Permanent*](https://doi.org/10.1016/0304-3975(79)90044-6).
- P. W. Kasteleyn, [*The Statistics of Dimers on a Lattice. I. The Number of Dimer Arrangements on a Quadratic Lattice*](https://doi.org/10.1016/0031-8914(61)90063-5).
- D. Cimasoni and N. Reshetikhin, [*Dimers on Surface Graphs and Spin Structures. I*](https://doi.org/10.1007/s00220-007-0302-7).
- I. Beichl and F. Sullivan, [*Approximating the Permanent via Importance Sampling with Application to the Dimer Covering Problem*](https://doi.org/10.1006/jcph.1998.6149).
- Y. Huh and S. Oh, [*Dense Dimer Packings in the 3D Cubic Lattice*](https://doi.org/10.1088/1402-4896/ae5605).
- L. M. Brègman, [*Some Properties of Nonnegative Matrices and Their Permanents*](https://www.mathnet.ru/eng/dan/v211/i1/p27).
- D. Donovan, K. Johnson, and I. M. Wanless, [*Permanents and Determinants of Latin Squares*](https://doi.org/10.1002/jcd.21418).
- A. A. Taranenko, [*Multidimensional Permanents and an Upper Bound on the Number of Transversals in Latin Squares*](https://doi.org/10.1002/jcd.21413).

These references motivate and validate the application families.  They do not
support the stronger claims that the `4 x 4 x 5` value is previously unknown,
that every three-dimensional dimer instance requires a permanent algorithm,
or that a Hofstadter Hamiltonian permanent is automatically a dimer partition
function.  Those claims are intentionally not made here.
