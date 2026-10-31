# Public test suite

Run `python -m pytest -q` from the repository root. Portable API tests compare
with independent definition-level oracles. Native agreement tests run when the
corresponding extension is present; `PERM_REQUIRE_NATIVE_TESTS=1` makes the
generic and polynomial native modules fail instead of silently skipping.

The suite includes exact low-rank, binary/frontier, polynomial, floating-point,
source-tool, showcase, and distribution-boundary checks. Optional dependency
and CUDA tests skip unless their dependency or explicit opt-in is available.
`test_source_distribution.py` builds a fresh archive and verifies its contents
and independent test collection.

See [CONTRIBUTING.md](../CONTRIBUTING.md#test-matrix),
[CUDA.md](../CUDA.md#device-tests), and [PACKAGING.md](../PACKAGING.md) for the
portable, native, device, and installed-distribution checks.
