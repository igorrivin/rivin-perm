# rivin-perm - Fast Exact Permanent Computation

A Python library for computing matrix permanents with Ryser/Glynn algorithms.
Integer and Schur-matrix results are reconstructed exactly with the Chinese
Remainder Theorem; real and complex matrices are also supported in floating
point.

## Features

- **Exact computation** - returns the true integer value, not a floating-point approximation
- **Schur matrix specialization** - optimized for DFT/Schur matrices with roots of unity
- **General integer matrices** - works with any integer matrix
- **Binary matrix specialization** - exact perfect-matching counts with graph preprocessing and a dedicated OpenMP kernel
- **Affine polynomial matrices** - computes every coefficient of `per(C + X*L)` in one Glynn traversal
- **Principal and rook polynomials** - exact principal-minor permanent sums and binary or weighted partial-matching vectors
- **Multiple CPU backends** - optional C/OpenMP extensions, Numba JIT, or pure Python
- **Parallel computation** - uses multiple CPU cores when an accelerated backend is available

The installed library provides portable CPU implementations and optional
OpenMP and Numba accelerators. This public repository also contains supported
CUDA source tools and reproducible benchmarks. Their support and installation
boundaries are described in [REPOSITORY_LAYOUT.md](REPOSITORY_LAYOUT.md).

## Installation

The pip distribution is named **`rivin-perm`**, while Python code continues to
use `import perm` and the existing companion module names. Release candidates
target Python 3.10–3.14. PyPI publication is pending; do not install the
unrelated PyPI project named `perm` to obtain this library.

To install a validated wheel supplied with a release candidate:

```bash
python -m pip install /path/to/rivin_perm-1.0.0-py3-none-any.whl
```

For a source checkout:

```bash
# Clone the repository
git clone https://github.com/igorrivin/rivin-perm.git
cd rivin-perm

# Install the package and attempt to build the optional OpenMP extensions
python -m pip install .

# Or install the optional Numba backend too
python -m pip install ".[numba]"
```

NumPy is installed automatically. Linux source builds attempt to compile the
OpenMP accelerators; if OpenMP is unavailable, the package remains usable with
the Python or Numba backend. Set `PERM_BUILD_OPENMP=0` while installing to skip
the C extensions explicitly. On macOS and Windows they are skipped by default.
`PERM_BUILD_OPENMP=required` makes a source build fail if any accelerator cannot
compile; release checks use this mode for native wheels. NumPy 1.23.5–2.x is
supported, with NumPy 2.x headers used for isolated native builds. Portable
wheels require no compiler. See [the packaging guide](PACKAGING.md) for clean
builds, isolated installation checks, wheel CI, and the publication boundary.

Run `python -m pytest -q` from a source checkout for the full default suite.
For an accelerated build, setting `PERM_REQUIRE_NATIVE_TESTS=1` makes the
polynomial-native test module fail instead of skip if its extension cannot be
imported:

```bash
PERM_REQUIRE_NATIVE_TESTS=1 \
    python -m pytest -q tests/test_polynomial_native.py
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the complete CPU/CUDA test matrix,
long-test gates, exactness requirements, and evaluation privacy rules.

### CUDA support

The CUDA files in this repository are manually built, source-only tools. They
are available in a checkout and retained in source distributions for
reproducibility, but they are not compiled by `pip install`, installed from a
wheel, or selected by the packaged CPU APIs. Using them requires a compatible
NVIDIA GPU, CUDA toolkit, and a separate build with `nvcc`.
For the binary backend, keep the `lib` prefix so the plain C ABI library does
not shadow its Python wrapper:

```bash
nvcc -shared -Xcompiler -fPIC -O3 -std=c++17 -arch=native \
    -o libperm_binary_cuda.so perm_binary_cuda.cu
```

[`CUDA.md`](CUDA.md) inventories the individual contracts, wrappers, build
boundary, opt-in device tests, and validated proofs of concept.

## Usage

### Schur Matrix Permanent

The Schur matrix $S_n$ has entries $s_{jk} = \omega^{jk}$ where $\omega = e^{2\pi i/n}$ is a primitive $n$-th root of unity. The permanent is always an integer.

```python
from perm import permanent

# Compute perm(S_n)
result = permanent(23)  # -31152650265
```

### General Integer Matrix Permanent

```python
from perm import matrix_permanent

A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

result = matrix_permanent(A)  # 450
```

The default `method="auto"` can use an exact low-rank coefficient calculation
for sufficiently large matrices. Its conservative cost rule accounts for the
generic backend's available parallelism. It checks rank separately at every
CRT prime and uses the generic path if any prime exceeds the selected budget.
The CRT bound and exact signed result are the same for both methods.

Use `matrix_permanent(A, method="generic")` to bypass this specialization, or
`method="low_rank"` to require it. An explicit low-rank request raises
`perm.LowRankLimitExceeded` above rank four or 50,000 coefficients per side.
The low-rank kernel currently computes residues serially; `parallel` and
`num_threads` control the generic path. `verbose=True` reports the selected
calculation. Floating-point and specialized Schur APIs retain their own paths.
See the [low-rank dispatch measurements](LOW_RANK_DISPATCH_RESULTS.md) for the
cost rule, fresh comparisons, retained losses, and reproduction commands.

For worked applications, see [low-rank permanent showcases](LOW_RANK_SHOWCASES.md):
polynomial-feature matching, Gaussian intensity moments, and exact profile
likelihoods. Run `python showcase_low_rank.py` from the checkout or unpacked
source archive. The exact showcases are included here. The optional full-rank approximation
research helper is not part of this public repository.

### Binary Matrices and Perfect Matchings

```python
from perm_binary import binary_permanent

# Number of perfect matchings in the corresponding bipartite graph
A = [[1, 1, 0, 0],
     [1, 1, 0, 0],
     [0, 0, 1, 1],
     [0, 0, 1, 1]]

result = binary_permanent(A)  # 4
```

Binary input is strict by default: entries must be integral zeroes or ones.
Passing `strict=False` explicitly computes the permanent of an integral
matrix's support. The portable backend uses exact Python integers; an optional
OpenMP extension is selected automatically when available.

Sparse support can instead be handled explicitly by the live-column frontier
dynamic program:

```python
from perm_binary import binary_permanent_frontier

# Adjacency matrix of K4: nine directed permutation cycle covers.
K4 = [[0, 1, 1, 1],
      [1, 0, 1, 1],
      [1, 1, 0, 1],
      [1, 1, 1, 0]]

binary_permanent_frontier(K4)  # 9
```

With `row_order=None`, the frontier routine uses a deterministic greedy
ordering heuristic; this is not a guarantee of minimum width. An explicit
`row_order` must be a permutation of the original row indices. The default
`max_states=500_000` is a resource guard: an explicit frontier call raises
`FrontierStateLimitExceeded` when a layer would retain more states. In
contrast, `binary_permanent(..., backend="auto")` may try the frontier method
on eligible matching-factor blocks and silently continue with its existing
exact native or Python backend if the cap is reached. Explicit
`backend="native"` and `backend="python"` calls do not dispatch to the
frontier routine.

Concrete generators cover grid dimers, anisotropic dimer profiles,
derangements, Bregman--Minc extremizers, and a correctly qualified
inclusion--exclusion reduction for Latin transversals:

```python
from perm_binary_applications import grid_dimer_count, grid_biadjacency

grid_dimer_count((2, 2, 4))  # 121 open-box dimer coverings

# A nonplanar 40x40 biadjacency benchmark for 80 lattice vertices.
target = grid_biadjacency((4, 4, 5), periodic_axes=(0, 1))
```

After the manual CUDA build above, the separate source-checkout wrapper can
compute a checked modular residue or reconstruct the exact nonnegative count:

```python
from perm_binary_cuda import (
    binary_permanent_cuda,
    binary_permanent_mod_p_cuda,
)

binary_permanent_mod_p_cuda(target, 2_147_483_647)  # 1238389710
binary_permanent_cuda(target)                       # 2551276235535120
```

This does not add `cuda` to the packaged `perm_binary` backend choices.

See the [binary algorithms and applications guide](https://github.com/igorrivin/rivin-perm/blob/main/BINARY_APPLICATIONS.md)
for the specialized method, boundary conventions, benchmark provenance, and
primary references.

### Affine Polynomial Matrices

`perm_polynomial` computes a complete matching-generating polynomial without
separately evaluating and interpolating it.  The exact API uses certified CRT
reconstruction; the modular API performs a single traversal.  Coefficients
are returned in ascending degree order, including trailing zeroes:

```python
from perm_polynomial import affine_permanent, affine_permanent_mod

C = [[1, 0],
     [0, 1]]
L = [[0, 1],
     [1, 0]]

# per(C + X*L) = 1 + X**2, exactly over the integers
affine_permanent(C, L)                            # (1, 0, 1)
affine_permanent_mod(C, L, 2_147_483_647)        # (1, 0, 1)
```

For the identity-constant case, the convenience API returns the sums of all
principal-submatrix permanents by cardinality:

```python
from perm_polynomial import (
    polynomial_backend_capabilities,
    principal_permanent_polynomial,
    principal_permanent_polynomial_mod,
)

K = [[2, -3],
     [5,  7]]

# per(I + X*K) = 1 + 9*X - X**2
principal_permanent_polynomial(K)  # (1, 9, -1)
principal_permanent_polynomial_mod(K, 11)  # (1, 9, 10)

# Stable source-tool query; "python" is always available.
capabilities = polynomial_backend_capabilities()
assert "python" in capabilities["available_backends"]
```

The portable Python path recognizes this case and uses the direct fixed-sign
projection, avoiding the general multiplication of affine row factors. Source
tools that need to select or report an accelerator can use
`polynomial_backend_capabilities()` instead of inspecting a private extension
module.

For a square zero-one matrix, the exact rook polynomial counts partial
matchings of every size in one call.  Entry `k` is the number using exactly
`k` rows and `k` columns.  When the native extension is available, all CRT
residues and all degrees are accumulated in one fused Glynn traversal:

```python
from perm_polynomial import rook_polynomial

A = [[1, 1],
     [1, 0]]

rook_polynomial(A)  # (1, 3, 1), meaning 1 + 3*X + X**2
```

`weighted_rook_polynomial` applies the same construction to arbitrary
nonnegative integer weights.  Entry `k` is now the sum of the products of the
selected entries over all size-`k` partial matchings:

```python
from perm_polynomial import weighted_rook_polynomial

W = [[1, 2],
     [3, 4]]

weighted_rook_polynomial(W)  # (1, 10, 10)
```

Arbitrary-size Python integers are supported. Disconnected bipartite support
components are evaluated separately and their exact coefficient vectors
convolved. Diagonal and star components have direct formulas. Other factors
produced by decomposition use at most eight arbitrary-integer subset states
when their smaller side has at most three vertices, avoiding CRT setup. These are
components of the full allowed-edge graph; perfect-matching-only factorizations
would not preserve partial matchings. Connected inputs use the fused native
multi-prime traversal when safe, with an affine-permanent fallback otherwise.
The planner stops as soon as all later modulus candidates are provably unsafe.

For activity `z`, the partition function is `Z(z) = sum(r[k] * z**k)` and the
probability of `k` matched pairs is `r[k] * z**k / Z(z)`. A source-checkout
[weighted-matching study](notes/benchmarks/WEIGHTED_MATCHINGS.md) provides exact
rational occupancy statistics, independent full-vector and per-query baselines,
and comparisons with belief propagation. Run `python benchmark_weighted_rook.py`
for the separate reproducible dense scaling benchmark.

A source-only [weighted-rook selector](notes/benchmarks/WEIGHTED_ROOK_DISPATCH.md)
offers explicit, experimental selection of the Python or compiled frontier for
bounded sparse inputs. It preserves exact integer results and records resource
fallbacks. See the retained/fresh dispatch evaluation (maintainer archive; not included)
for its measured coverage and regressions before opting in.

Computing all coefficients inside a permanent traversal has prior art:
[Gao et al., Section III.C](https://arxiv.org/pdf/0903.1952) give
polynomial-valued Ryser evaluation and factorial reweighting for sums of
subpermanents. Related implementations include
[Butera–Pernici](https://arxiv.org/abs/1406.5337) and
[Sage's permanental-minor polynomial](https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/matrix_misc.html#sage.matrix.matrix_misc.permanental_minor_polynomial).
The traversal descriptions above specify this implementation; they are not
priority or uniform performance claims. Its memory advantage over subset
dynamic programming does not extend to every polynomial or interpolation
algorithm. Public native principal evaluation currently uses the generic
affine kernel; direct principal and Krawtchouk GPU paths remain prototypes.

The principal output contains `n + 1` cardinality sums, rather than every
individual principal permanent. For a proper complex-Gaussian covariance
matrix these sums encode intensity moments. Weighted-rook coefficients also
support expected-determinant bounds under an independent-entry eigenmode
channel model: the logarithm of the expected determinant bounds expected
log-determinant, and is not exact ergodic capacity. Real-valued applications
need an explicit precision or rational-quantization treatment.

The same backend powers modular anisotropic dimer profiles beyond the small
definition-level subset-DP range:

```python
from perm_binary_applications import axis_dimer_profile_mod

axis_dimer_profile_mod((2, 2, 2), 97)
# (4, 0, 4, 0, 1)
```

### Real/Complex Matrices (Floating-Point)

For real or complex matrices where exact integer results aren't needed:

```python
from perm_float import permanent_float, permanent_schur_float
import numpy as np

# General matrix
A = np.random.randn(15, 15)
result = permanent_float(A)

# Schur matrix (fast)
result = permanent_schur_float(17)  # 25219857.0
```

### Check Backend

```python
from perm import get_backend

print(get_backend())  # 'c', 'numba', or 'python'
```

## Algorithm

### Chinese Remainder Theorem Approach

Computing permanents exactly over the integers is challenging because intermediate values can overflow. We solve this by:

1. **Choose primes** $p_1, p_2, \ldots, p_k$ whose product exceeds twice the bound on $|\text{perm}(A)|$
2. **Compute** $\text{perm}(A) \mod p_i$ with the applicable Ryser/Glynn backend
3. **Reconstruct** the exact integer via CRT

For **Schur matrices**, we need primes $p \equiv 1 \pmod{n}$ so that $\mathbb{F}_p$ contains a primitive $n$-th root of unity: $\omega = g^{(p-1)/n}$ where $g$ is a primitive root mod $p$.

For **general integer matrices**, any sufficiently large primes work.

For **binary matrices**, the answer is nonnegative, so the CRT product only
needs to exceed a certified upper bound once rather than twice. The planner
combines row and column degree products, $n!$, and rigorously integer-rounded
Bregman--Minc bounds.

### Ryser's Algorithm

Ryser's formula computes the permanent in $O(2^n \cdot n)$ time:

$$\text{perm}(A) = (-1)^n \sum_{S \subseteq [n]} (-1)^{|S|} \prod_{i=1}^{n} \sum_{j \in S} a_{ij}$$

Our implementation uses:

- **Gray code ordering** for incremental row-sum updates in serial paths
- **OpenMP Gray-code blocks** that give each worker a contiguous range of states
- Automatic selection among the available C/OpenMP, Numba, and Python paths

For general integer matrices, current C extensions also select the full
fixed-sign Glynn formula when the order is at least four and the modulus is
odd with `1 < p <= 3_037_000_499`. This visits `2**(n-1)` sign states and is
valid for arbitrary integer matrices, including odd composite moduli. It uses
none of the Schur-specific cofactor or necklace reductions. The exact CRT
planner normally uses primes in this range.

Tiny matrices, even moduli, and wider moduli retain Ryser. Wide modular
arithmetic can favor Ryser on structured matrices because its intermediate
zero products avoid expensive reductions; halving the state count alone is
not a guarantee of faster execution. The explicit
`perm_c_ext.ryser_permanent_mod_p` entry point remains available, and the Python
API continues to work with older extensions that expose only that entry point.

The source-checkout benchmark compares general matrix families, modulus
ranges, and thread counts, with a separate complete exact-CRT comparison:

```bash
python benchmark_native_dispatch.py --output dispatch.json
```

### Binary Specialization

The binary API first finds a perfect matching. If none exists, it returns zero
without entering an exponential kernel. Relative to a found matching, strongly
connected alternating-cycle components factor the permanent into independent
square blocks; forced and block-triangular structure can therefore collapse a
large input into cheap exact factors.

For each remaining block, fixed-sign Glynn enumerates $2^{n-1}$ states. The
native kernel fixes the densest column, assigns low-degree columns to the most
frequently toggled Gray bits, and chooses between the matrix and its transpose
by an exact update-cost estimate. A toggle touches only rows containing a one.
It updates the product of nonzero signed row sums by a precomputed modular
ratio while a separate counter handles zero factors, eliminating the usual
full $n$-factor product at every state.

The hot path uses division-free 32-bit Montgomery arithmetic with descending
primes below $2^{31}$; a slower 64-bit Montgomery path preserves the public
signed-64-bit prime contract. Exact reconstruction uses the nonnegative
interval and grouped Bregman--Minc bounds described above. The pure-Python
fallback is an arbitrary-precision subset dynamic program.

### Affine-Polynomial Specialization

For `A(X) = C + X L`, fixed-sign Glynn visits `2**(n-1)` sign vectors once.
Each signed row sum remains affine in `X`; multiplying those `n` affine
factors produces the complete coefficient vector for that state.  Gray-code
updates change only one signed column at a time.  This replaces `n` or more
independent scalar permanent traversals with one coefficient-valued
enumeration.

The optional OpenMP backend fixes a dense column, assigns sparse columns to
the frequently toggled Gray bits, and uses a checked signed-64-bit convolution
when its coefficient-norm bound proves it safe.  It otherwise falls back to
fully modular convolution.  The modulus `2**31 - 1` has a division-free
Mersenne reduction; arbitrary odd signed-64-bit moduli, including composites,
remain supported.  The portable Python implementation follows the same
formula as a correctness fallback.

For exact integer output, four inexpensive coefficientwise bounds are formed:
row and column absolute-sum relaxations, plus `n!` times row and column
absolute-maximum relaxations.  Their coefficientwise minimum bounds
`abs([X**k] per(C + X*L))` rigorously.  Modular vectors are evaluated at
deterministic descending 31-bit primes until their product exceeds twice the
largest bound, then reconstructed in the centered signed interval.  When
both input matrices are nonnegative, every coefficient is nonnegative and
the prime product need only exceed the bound once.

Binary rook polynomials have a stronger native specialization. Fixed-point
Bregman--Minc bounds avoid whole-integer root rounding, while a polynomial-time
characteristic-two Cauchy--Binet determinant supplies every coefficient's
parity. The parity bit distinguishes `x` from `x + P`, so a CRT product only
slightly greater than half the largest bound is sufficient. The native hot
loop packs up to sixteen same-width pseudo-Mersenne prime fields beside the
coefficient degree and accumulates direct rook residues in one sign-vector
enumeration. It dispatches to SVE at runtime on supported AArch64 hosts and
retains a portable scalar implementation elsewhere. Unsafe modulus/degree
combinations are rejected by exact 32-bit overflow and one-fold-reduction
checks.

### Schur Specialization

For odd $n$, equal Schur cofactors reduce the problem to
$\text{perm}(S_n)=n\,\text{perm}(B_{n-1})$. Applying Glynn's formula to the
even $(n-1)\times(n-1)$ minor visits $2^{n-2}$ sign states instead of the
$2^n$ subsets of full-matrix Ryser, a 4× state-count reduction.

For prime $n$, the CPU path additionally writes the minor in circulant form
and enumerates sign necklaces modulo rotation and global complement, using
exact stabilizer/orbit weights. For prime $n\geq5$, the Graham–Lehmer
congruence

$$\text{perm}(S_n)\equiv n!\pmod{n^3}$$

provides a coprime CRT seed. For odd composite $n$ (and the exceptional prime
$n=3$), the corresponding prime-power divisibility theorem supplies the
zero-residue seed

$$D_n=\prod_{p^a\mathbin\Vert n}
p^{(p^a-1)n/((p-1)p^a)}\mid\text{perm}(S_n).$$

These theorem-only moduli reduce the number of separately computed prime
residues needed for exact reconstruction.

The CUDA sources specialize arithmetic for CRT primes
$q=2^{32}-c\equiv1\pmod n$, using pseudo-Mersenne reduction. Packed necklace
kernels cover minor sizes $M=40$ and $M=46$, the $40\times40$ and
$46\times46$ cofactors of prime Schur orders 41 and 47. Other supported sizes
retain Glynn, including the fixed-sign $M=44$ specialization used at order
45. CUDA remains source-only and must be built manually; it is not part of
the installed Python package.

### Bounds

**Schur matrix**: Since $S_n = \sqrt{n} \cdot U$ where $U$ is unitary (the DFT matrix), we have:
$$|\text{perm}(S_n)| \leq n^{n/2}$$

**General matrix**: Using the row-sum bound:
$$|\text{perm}(A)| \leq \prod_{i=1}^{n} \sum_{j=1}^{n} |a_{ij}|$$

## Performance

For the dense order-31 SHA-generated affine matrix in
`notes/evals/sha_bipartite_polynomial_prompt.md`, the original reference route
used 31 scalar Ryser calls to recover the coefficients modulo `2**31 - 1`.
The coefficient-valued backend returns the identical 31 residues directly:

| Order-31 polynomial task | GH200 CPU, 64 threads |
|---|---:|
| 31 scalar Ryser root evaluations | 196.14 s |
| One coefficient-valued Glynn run | 4.09 s |
| Full verifier: two coefficient moduli plus three exact scalar certificates | 11.13 s |
| Exact integer coefficient vector, four-prime CRT | 19.88 s |

The one-modulus improvement is 48.0x in wall time.  The algorithm visits
`2**30` fixed-sign states rather than `31 * 2**31` scalar Ryser states; zero
row tracking and dynamically balanced Gray subranges reduce the realized time
further.  The independent verifier's `--recompute` mode exercises the new
two-modulus path.

The dense SHA-generated order-33 rook task in
`notes/evals/rook33_polynomial_prompt.md` exercises the fused exact path:

| Exact rook polynomial | GH200 CPU, 64 threads |
|---|---:|
| Previous sequential 31-bit CRT traversals | 369.26 s |
| Fused four-prime 27-bit traversal, bound/parity/CRT included | 44.00 s |

All 34 exact coefficients match the independently pinned answer. For the
analogous deterministic order-37 board, the sign space is 16 times larger
and the certified range needs five rather than four 27-bit primes. Accounting
for the longer coefficient vector and fifth packed field, the fresh `/1`
evaluation instance completed in 1,212.81 seconds (20.21 minutes) on the same
host. The full order-37 run is intentionally not part of the default test
suite.

Representative binary one-residue timings on the GH200 host CPU using 64
OpenMP threads compare the specialized kernel with the general integer Ryser
backend on the identical modulus:

| Matrix | General CPU | Binary CPU | Speedup |
|---|---:|---:|---:|
| order 30, cyclic 3-regular | 1.6109 s | 0.07246 s | 22.23x |
| order 30, cyclic 6-regular | 1.6512 s | 0.14562 s | 11.34x |
| order 30, planted 50% dense | 1.7222 s | 0.21698 s | 7.94x |

For the exact 40x40 binary matrix from the periodic-x/y 4x4x5 dimer case,
the specialized CPU path took 242.39 seconds and the manually built GH200
CUDA path took 22.73 seconds end to end, with exact agreement at
`2,551,276,235,535,120`.  An independent, lattice-specific layer-transfer
verifier took 2.54 seconds; this is a cross-check and a reminder that highly
structured applications should also be compared with specialized methods.

For the adjacency matrix of the asymmetric random cubic graph in the
cycle-cover evaluation, two GH200 CUDA residues took 60.78 seconds and
certified the exact permanent `1,361,137`.  A separately implemented sparse
frontier DP found a width-14 ordering and returned the same value in 0.066
seconds.  The production `binary_permanent_frontier` API now exposes this
live-column method, and its default greedy ordering also solves that instance
below the state cap.  The stored evaluation verifier remains an independent
implementation and certificate.  This deliberately records both realities:
the generic exponential benchmark and the graph-specific shortcut a capable
solver may discover.

### Schur performance

Representative measured exact Schur computations:

| Hardware and backend | Order | Current time | Historical project time | Speedup |
|----------------------|-------|--------------|-------------------------|---------|
| 64-core CPU, C/OpenMP | $n=29$ | 0.0518 s | -- | -- |
| 64-core CPU, C/OpenMP | $n=37$ | 18.43 s | 56.8 min | ~185× |
| NVIDIA GH200, manual CUDA 40×40-minor necklace | $n=41$ | 47.39 s | 87.9 min (H100) | ~111× |
| NVIDIA GH200, manual CUDA 44×44-minor Glynn | $n=45$ | 1949.882 s (32.50 min) | -- | -- |
| NVIDIA GH200, manual CUDA 46×46-minor necklace | $n=47$ | 3702.521 s (61.71 min) | -- | -- |

These are machine-specific measurements; the speedups compare with the
project's recorded historical implementations. The order-45 and order-47
times sum the three and four required GPU residue wall times, respectively;
independent holdout and cross-check runs are excluded. The four order-47
residues took 927.630, 924.995, 924.999, and 924.897 seconds. GH200 results use
the source-only CUDA code and do not imply packaged CUDA support.

The verified sequence frontier is now
$\text{perm}(S_{45})=-470483360490107435783203125$ and
$\text{perm}(S_{47})=-52876272639843868105982022705$. See
`schur_permanents.txt` for recorded exact values through $n=47$.

## OEIS

[OEIS A003112](https://oeis.org/A003112) records the odd-size Schur
permanents, with sequence index $k$ corresponding to matrix size $n=2k+1$.

## Files

Start with these repository guides:

- [`REPOSITORY_LAYOUT.md`](REPOSITORY_LAYOUT.md) - Five support tiers, the
  checkout/sdist/wheel matrix, component ownership, and path-stability rules
- [`CUDA.md`](CUDA.md) - Source-only CUDA inventory, manual builds, device
  tests, and large-run discipline
- [`CONTRIBUTING.md`](CONTRIBUTING.md) - Development setup, verification
  requirements, benchmark provenance, and evaluation privacy
- [`CHANGELOG.md`](CHANGELOG.md) - User-visible unreleased changes and version
  history policy
- [`BINARY_APPLICATIONS.md`](BINARY_APPLICATIONS.md) - Binary algorithms,
  applications, boundary conventions, and references

### Installed library and packaged acceleration

- [`perm.py`](perm.py) - Exact Schur and general integer permanents
- [`perm_binary.py`](perm_binary.py) - Exact binary permanents, preprocessing,
  bounds, and CRT
- [`perm_binary_applications.py`](perm_binary_applications.py) - Grid-dimer and
  restricted-permutation generators
- [`perm_polynomial.py`](perm_polynomial.py) - Exact affine/principal
  permanents and binary/weighted rook polynomials
- [`perm_float.py`](perm_float.py) - Floating-point real/complex permanents
- [`perm_mod3.py`](perm_mod3.py) - Specialized characteristic-three
  permanents
- [`perm_numba.py`](perm_numba.py) - Optional Numba JIT backend
- [`perm_c_omp.c`](perm_c_omp.c), [`perm_binary_omp.c`](perm_binary_omp.c),
  [`perm_mod3_omp.c`](perm_mod3_omp.c), and
  [`perm_polynomial_omp.c`](perm_polynomial_omp.c) - Optional OpenMP extension
  sources
- [`setup.py`](setup.py), [`pyproject.toml`](pyproject.toml), and
  [`MANIFEST.in`](MANIFEST.in) - Build and distribution configuration

### Source tools and reproducible benchmarks

- [`perm_binary_cuda.py`](perm_binary_cuda.py) and
  [`perm_binary_cuda.cu`](perm_binary_cuda.cu) - Manually built binary CUDA
  path
- [`benchmark.py`](benchmark.py), [`benchmark_binary.py`](benchmark_binary.py),
  and [`benchmark_weighted_rook.py`](benchmark_weighted_rook.py) - Performance
  benchmarks
- [`experiment_random_cubic.py`](experiment_random_cubic.py) - Reproducible
  random cubic-graph distribution experiment
- [`notes/frontier/verify_binary_grid_4x4x5_transfer.py`](notes/frontier/verify_binary_grid_4x4x5_transfer.py)
  - Independent structured target verifier

## License

[MIT](LICENSE)
