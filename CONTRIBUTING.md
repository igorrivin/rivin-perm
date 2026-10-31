# Contributing

Contributions should preserve exactness, portable fallbacks, reproducibility,
and the separation between installed APIs and research evidence. Read the
[`repository support levels`](REPOSITORY_LAYOUT.md) before
changing a backend or moving a file.

## Development setup

From a source checkout:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,numba]"
```

Linux source builds normally attempt the optional OpenMP extensions. To force
a portable installation while working on fallback behavior:

```bash
PERM_BUILD_OPENMP=0 python -m pip install -e .
```

To rebuild the optional extensions in place:

```bash
python setup.py build_ext --inplace
```

Do not assume a clean worktree. Inspect `git status --short` before editing,
preserve unrelated changes, and keep refactors separate from algorithmic
changes.

## Test matrix

The default suite is the required baseline:

```bash
python -m pytest -q
```

It exercises portable code, installed extensions when present, source-wrapper
contracts, independent oracles, and quick retained-value checks. CUDA device
tests remain skipped unless explicitly enabled.
[`tests/README.md`](tests/README.md) maps the suite into portable, native,
source-wrapper, device, prototype, and repository-integrity groups.

The source distribution includes the reviewed public test suite. Both
checkout and unpacked sdist tests must collect and pass independently.
Distribution checks install real wheels/sdists outside the checkout; follow
[PACKAGING.md](PACKAGING.md) before preparing a pip release.

For generic or polynomial native work, require rather than silently skip the
extensions:

```bash
python setup.py build_ext --inplace
PERM_REQUIRE_NATIVE_TESTS=1 \
    python -m pytest -q tests/test_generic_native.py tests/test_polynomial_native.py
```

Run the smallest focused module while iterating, then the full default suite.
Finish documentation and source changes with:

```bash
git diff --check
```

### CUDA tests

CUDA changes require the opt-in device modules on a compatible host:

```bash
PERM_RUN_CUDA_TESTS=1 \
PERM_CUDA_TEST_ARCH=native \
PERM_CUDA_ARCH=native \
    python -m pytest -q \
    tests/test_cuda_device_binary.py \
    tests/test_cuda_device_modular.py \
    tests/test_cuda_device_parallel.py \
    tests/test_cuda_device_schur_glynn.py \
    tests/test_cuda_device_utilities.py
```

The order-41 and order-47 regressions require
`PERM_RUN_CUDA_LONG_TESTS=1` and `PERM_RUN_CUDA_FRONTIER_TESTS=1`,
respectively. They are deliberate long runs, not ordinary CI. See
[`CUDA.md`](CUDA.md#device-tests) for details.

## Exactness requirements

An optimized exact backend needs more than agreement on one large answer.
Include, as applicable:

- definition-level exhaustive tests at small order;
- seeded random comparisons with an independent implementation;
- edge cases for empty, singular, signed, sparse, and oversized inputs;
- a rigorous coefficient or permanent bound;
- CRT range, parity, divisibility, or centered-lift checks;
- deterministic instance and output hashes for retained long runs; and
- a foreign-modulus or otherwise independent holdout for expensive results.

Unsafe arithmetic ranges must be rejected or sent to a proven fallback. A
benchmark speedup is not permission to weaken an input contract.

## Benchmarks and retained results

Benchmark scripts should have a quick safe default, deterministic seeds or
instance labels, machine-readable output, repeat consistency checks, and an
explicit opt-in such as `--allow-large` for expensive exponential work.

When retaining a result, record:

- the exact command and relevant environment variables;
- CPU/GPU identity, compiler, architecture, and thread count;
- source and executable hashes when binaries matter;
- kernel and wall time separately when useful;
- the complete output or a canonical output hash; and
- the independent checks that passed.

Do not commit generated shared objects, compiler output, logs, or LaTeX
intermediates. Curated captures and figures are evidence only when their
provenance is documented.

## Documentation and compatibility

- Update README usage when a Tier-1 API changes.
- Update `REPOSITORY_LAYOUT.md` when support or packaging status changes.
- Update `CUDA.md` when a CUDA contract, build, or test gate changes.
- Add user-visible changes to the Unreleased section of `CHANGELOG.md`.
- Preserve current import paths and documented return conventions unless a
  migration and compatibility plan is explicit.
- Treat native/CUDA file paths as interfaces: build files, tests, verifiers,
  and retained hashes currently refer to them directly.

Historical session logs describe what happened at a date; do not silently
rewrite them to look current. Put current instructions in README, this guide,
or an adjacent assessment instead.
