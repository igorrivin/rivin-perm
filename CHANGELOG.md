# Changelog

This file records user-visible library, build, and support changes. Historical
research results remain in their dated notes and session logs.

## Unreleased

### Added

- Clean wheel/sdist installation checks covering the installed public API,
  optional Numba, and strict native-extension availability; distribution CI
  for portable and Linux accelerator wheels.
- `PERM_BUILD_OPENMP=required` to prevent native release builds from silently
  falling back when compilation fails.
- Reproducible low-rank showcases with explanations of weighted assignments,
  proper complex Gaussian intensity moments, and exact unlabeled profile
  probabilities. Include a grouped-column comparator and an optional
  checkout-only full-rank covariance example with certified approximation.
- Exact low-rank integer computation in `matrix_permanent`, selected by a
  conservative rank/order/parallelism cost rule. New keyword-only `method`
  choices allow explicit generic or low-rank computation, with a documented
  rank/state budget and `LowRankLimitExceeded` for oversized explicit requests.
  Every CRT prime is preflighted before using the specialization; its CRT bound
  and signed reconstruction are shared with the existing general path.
- A compiled weighted partial-matching frontier source tool with exact modular
  arithmetic/CRT, reusable column slots, explicit state and native allocation
  limits, independent correctness tests and recorded build provenance. Public
  weighted-rook dispatch is unchanged.
- A public weighted-matching application study with exact rational occupancy
  distributions, mean and variance, damped monomer-dimer belief propagation,
  an independent Butera-Pernici-style frontier baseline, and a compiled exact
  per-activity partition/moment baseline. Source tools retain fixed-input
  weight scaling, isolated process RSS, raw timings, and replayable CRT records.
- A general `perm_c_ext.permanent_mod_p` entry point and independent native
  definition/parallel tests, plus a reproducible benchmark across matrix
  families, modulus ranges, and thread counts.
- Exact `principal_permanent_polynomial` and modular
  `principal_permanent_polynomial_mod` support for the coefficient vector of
  `per(I + xA)`.
- Exact `weighted_rook_polynomial` support for nonnegative integer matrices,
  including a fused native multi-prime path.
- A stable `polynomial_backend_capabilities` query for source tools and
  benchmarks that need to report optional native hooks without importing a
  private extension module.
- Repository support tiers, packaging boundaries, CUDA inventory, and
  contributor guidance.

### Changed

- Name the pip distribution `rivin-perm`, preserving all Python import paths.
  Declare metadata in `pyproject.toml`, target Python 3.10 and later, require
  NumPy 1.23.5–2.x at runtime, and build native modules with NumPy 2.x headers
  targeting the NumPy 1.23 API.
- Build sdists from an explicit public-file allowlist, including reviewed tests;
  omit the three research-only CUDA utilities previously included by name.
- Reduce weighted-rook dispatch overhead with a dense-row connectivity check
  and bitmask component traversal. Decomposed factors with at most three
  vertices on the smaller side use a bounded exact subset calculation;
  standalone connected inputs retain their CRT backend. Research transcripts
  now distinguish exact small factors from modular factors and replay both.
- Factor disconnected bipartite support components in `weighted_rook_polynomial`
  and convolve their exact coefficient vectors. Diagonal and star components
  use direct formulas; rectangular components are padded internally and the
  original output length is preserved.
- General exact integer computations use native Glynn for orders at least
  four with odd moduli in the direct-multiplication range, while retaining
  Ryser for tiny matrices, even moduli, and wider arithmetic. The explicit
  Ryser entry point and compatibility with older extensions are preserved.
- Cache matrix-independent fixed-point factors used by rook-polynomial bounds
  in bounded caches, reducing repeated small-matrix call overhead.
- The single-prime backend benchmark now compares general Ryser kernels on
  the same matrix with one thread for each backend.
- The native rook-polynomial engine can pack up to sixteen same-width
  pseudo-Mersenne prime fields in a fused traversal.
- Documentation now distinguishes installed public APIs, optional packaged
  accelerators, manually built source tools, validated prototypes, and
  private or archival evidence.
- Answer-dependent experiment captures and source pins are classified as
  private evaluator evidence regardless of their directory.
- CUDA sources and selected wrappers are documented as checkout/sdist
  reproducibility material rather than wheel-installed modules.
- The sdist excludes private checkout-only oracle/integrity tests; its
  remaining test tree is self-contained and skips only unavailable optional
  accelerators or explicitly gated device runs.

### Fixed

- Decode the characteristic-three portable result using Python integers before
  bit operations, preserving compatibility with NumPy 1.23.5 as well as 2.x.
- Make weighted-study workers portable to macOS: report unavailable CPU pinning,
  reject unsupported explicit pinning, and normalize peak RSS using host units.
- Preserve Schur queue parent-loss and timeout cleanup on macOS with a POSIX
  pipe supervisor, while retaining Linux's direct PDEATHSIG worker path.
  Process identity probes no longer rely on procfs outside Linux or silently
  interpret inspection errors as worker death.
- Stop the weighted pseudo-Mersenne prime search when its monotone safety
  condition rules out every remaining candidate, preserving selected primes
  and fallback decisions.
- Correct the full-Haar permanent second moment in research analyses;
  replace unequal and same-sample fitted KS comparisons with common held-out
  distance diagnostics and retain sample/provenance archives for new runs.
- Replace numerical cycle-path permanent evaluation with a tested analytic
  product, with explicit phase conventions and exact even-midpoint zeros.
- Revise manuscript novelty claims, statistical interpretations, and
  unsupported finite-field proofs; document withdrawn historical figures
  and remaining empirical verification requirements.
- Convert NumPy integer scalars to Python integers before exact bounds and
  portable modular arithmetic, preventing fixed-width overflow from silently
  corrupting permanent results.
- Preserve depth-first finishing order when splitting binary matching graphs
  into strongly connected components, avoiding unnecessary merging of
  independent blocks.

### Verification

- Start the September 5 independent verification campaign: standalone Schur
  reconstruction audits and resumable finite-field rank enumeration reproduce
  all eight retained prime-field counts through 23 and the F9 count. Polynomial
  audits replay the retained order-36 interpolation and CRT transcript;
  missing projection-residue arrays are explicitly reported. Reconstruction guarantees
  are explicitly conditional on residues; computation records and fresh
  cross-checks have separately stated coverage.
- Complete six fresh scalar GPU checks across principal orders 36/38 and rook
  order 40; all agree with retained vectors. Freeze challenges before execution,
  retain source snapshots, and resume from completed calls after interruption.
- Add a separate full-matrix Schur-45 CUDA Ryser checker with indexed block
  checkpoints and exact input/build manifests. Small GPU and actual-block
  pilots pass; the full independent modular traversal is running. The final
  default suite passes 1,280 tests; future completion must be read from the
  campaign result rather than inferred from a partial run.
- Principal-permanent and Krawtchouk prototype tests compare against
  independent small-order definitions and seeded random cases.
- Retained long computations carry canonical hashes and independent
  certificates; their assessments state which checks are recomputed and which
  executions are preserved evidence.

## Historical version note

`setup.py` currently reports version `1.0.0`. The repository does not have a
matching release tag or a pre-existing changelog from which trustworthy
release dates and earlier version boundaries can be reconstructed. Earlier
changes are therefore left in git history and dated session logs instead of
being assigned invented release entries here.
