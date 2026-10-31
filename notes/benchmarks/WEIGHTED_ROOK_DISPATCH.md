# Opt-in exact weighted-rook dispatch

`weighted_rook_dispatch.py` is a source-checkout API for testing conservative
selection between the existing weighted-rook implementation and two independent
frontier implementations. It is included in the source distribution, alongside
its dependencies, but is not an installed wheel API. It changes no existing
public entry point and does not build or discover native libraries.

```python
from notes.benchmarks.weighted_rook_dispatch import WeightedRookDispatcher

# Pure-Python frontier selection is available without a frontier library.
solver = WeightedRookDispatcher()
matrix = [[2, 3], [5, 7]]
record = solver.evaluate(matrix)
assert record["coefficients"] == (1, 17, 29)

# Optional: explicitly supply the existing library built and verified using
# WEIGHTED_FRONTIER_NATIVE.md. Construction/loading is outside each evaluation.
solver = WeightedRookDispatcher("/absolute/path/weighted_matching_frontier_native.so")
coefficients = solver.weighted_rook_polynomial(matrix)
record = solver.evaluate(matrix)
print(record["decision"], record["resource_fallback"])
```

The default is serial, `backend="auto", parallel=False, num_threads=1`.
The original square nonnegative-integer input contract is preserved, including
arbitrary-size integer weights, booleans, exact trailing zeros, and the order-63
limit. Rectangles remain available directly through the frontier tools. Explicit
backends and effective thread counts other than one follow the existing public
implementation. Auto selection also requires the existing polynomial native
extension; otherwise the public Python backend is used. Passing a bad native
library raises during construction; it never silently changes implementations.

The frozen initial policy considers only connected support spanning both sides,
orders 8–18, at most six nonzeros per row, and weights of at most ten bits.
Existing small, dense, disconnected, large-weight, and uncalibrated cases keep
the public path. Greedy row ordering is computed once for eligible candidates
and passed explicitly to the chosen frontier tool.

- Python frontier: at most three nonzeros per row and at most four columns live
  across any row cut.
- Native frontier: orders 16–18, density at most one quarter, cut widths 6–11,
  and at most twelve columns needed during a row.
- Otherwise: the original validated public calculation, reusing normalization
  and component discovery already done for this call.

Defaults cap Python at 64 states and native at 4,096 states and 16 MiB of its
explicit workspace. These are solver limits, not process-RSS guarantees.
Documented state, width, or workspace-limit exceptions cause an exact public
fallback and are retained in `resource_fallback`. Unexpected exceptions and
malformed/incomplete solver results propagate. No floating point or numerical
tolerance enters the calculation. The selector itself has no wall-time limit.

`evaluate` returns the complete coefficient tuple, the selected method/reason,
any resource fallback, and the frontier's full record when used. Validation and
diagnostics are included in its measured call time. The tuple-only convenience
method calls the same implementation and adds no alternate numerical path.

The thresholds are empirical and cannot guarantee a speedup for arbitrary
matrices, layouts, interpreters, or hardware. Consult
the dispatch results (maintainer archive; not included) before using the
selector for a workload. It remains opt-in unless a separate public integration
is justified; the frozen source and holdout campaign are research evidence.
