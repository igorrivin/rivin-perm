# Mathematical scope and attribution

The library's exact integer results and floating-point estimates have different
contracts. Numerical fits and finite-size benchmarks are not proofs of limiting
laws or uniform speedups. Research manuscripts and historical experiments are
not included in this public library export.

The source helpers and documentation retain the following qualifications:

* For a full Haar unitary of order n, the second moment is
  `1 / binom(2*n - 1, n)`; see [Nezami, equation 97](https://arxiv.org/pdf/2104.06423).
* Finite Gaussian permanents have finite moments. Empirical finite-sample
  distribution fits do not identify an exact or limiting distribution.
* The cycle-geodesic product identities are attributed to
  [Kamenetskii, Theorems 3–4](https://arxiv.org/pdf/0907.2860).
  Literal and principal eigenangle paths use distinct phase conventions.
* Coefficient-valued permanent traversal predates this implementation; see
  [Gao et al., Section III.C](https://arxiv.org/pdf/0903.1952).
* Low-rank timings are structured-workload measurements; the generic fallback
  and explicit method controls remain part of the public interface.

See [LOW_RANK_DISPATCH_RESULTS.md](LOW_RANK_DISPATCH_RESULTS.md) and
[LOW_RANK_SHOWCASES.md](LOW_RANK_SHOWCASES.md) for supported examples and
reproduction commands.
