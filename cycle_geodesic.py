"""Analytic permanents along spectral paths from the identity to an n-cycle.

For C[j, (j + 1) % n] = 1, the ``literal`` path has Fourier eigenangles
2*pi*k/n, k=0,...,n-1. Its permanent is

    prod(r + (n-r)*exp(2*pi*i*t), r=1,...,n) / n**n.

This follows from the Cauchy permanent identity, not a numerical fit; see
https://arxiv.org/abs/0907.2860, Theorems 3--4. ``principal`` uses angles
in (-pi, pi], explicitly choosing +pi at the even-order branch cut by
default. ``pi_branch='negative'`` selects [-pi, pi) instead. These two
even-order paths are equally valid shortest unitary paths.

The literal and principal paths have equal permanent magnitudes. Their
phases differ by exp(-2*pi*i*m*t), where m=(n-1)//2 for the positive
branch and m=n//2 for the negative branch. Odd-order principal permanents
are positive real; even-order principal permanents are generally complex.
"""

import cmath
from fractions import Fraction
import math
import operator

import numpy as np


def _validate_n(n):
    if isinstance(n, (bool, np.bool_)):
        raise ValueError("n must be a positive integer")
    try:
        n = operator.index(n)
    except TypeError as exc:
        raise ValueError("n must be a positive integer") from exc
    if n < 1:
        raise ValueError("n must be a positive integer")
    return n


def _validate_t(t):
    t = float(t)
    if not math.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("t must be finite and lie in [0, 1]")
    return t


def _branch_shift(n, convention, pi_branch):
    if convention not in {"literal", "principal"}:
        raise ValueError("convention must be 'literal' or 'principal'")
    if pi_branch not in {"positive", "negative"}:
        raise ValueError("pi_branch must be 'positive' or 'negative'")
    if convention == "literal":
        return 0
    return (n - 1) // 2 if pi_branch == "positive" else n // 2


def cyclic_shift(n):
    """Return the cycle permutation C[j, (j+1) % n] = 1."""
    n = _validate_n(n)
    matrix = np.zeros((n, n), dtype=np.complex128)
    matrix[np.arange(n), (np.arange(n) + 1) % n] = 1
    return matrix


def cycle_geodesic_matrix(n, t, *, convention="principal", pi_branch="positive"):
    """Construct the selected spectral path for small reference calculations.

    Use :func:`cycle_permanent` to evaluate its permanent without constructing
    a matrix. The literal entries are (1-z**n)/(n*(1-z*omega**(j-l))),
    with removable singularities evaluated here through the Fourier sum.
    """
    n, t = _validate_n(n), _validate_t(t)
    shift = _branch_shift(n, convention, pi_branch)
    if t == 0:
        return np.eye(n, dtype=np.complex128)
    if t == 1:
        return cyclic_shift(n)
    indices = np.arange(n)
    modes = indices - shift
    fourier = np.exp(2j * np.pi * np.outer(indices, modes) / n) / math.sqrt(n)
    eigenvalues = np.exp(2j * np.pi * t * modes / n)
    return (fourier * eigenvalues) @ fourier.conj().T


def cycle_permanent_logabs(n, t):
    """Return log|per(gamma(t))| using O(n) paired positive factors.

    The value is independent of the spectral branch. At the even-order
    midpoint it is exactly -inf. Unlike evaluating the permanent first,
    this remains useful when its magnitude underflows double precision.
    """
    n, t = _validate_n(n), _validate_t(t)
    if n == 1 or t in {0.0, 1.0}:
        return 0.0
    if n % 2 == 0 and t == 0.5:
        return -math.inf
    # Reflection improves accuracy near t=1 and makes symmetry explicit.
    folded = min(t, 1 - t)
    if folded > 0.25:
        angle = math.pi * (0.5 - folded)
        cosine, sine = math.sin(angle), math.cos(angle)
    else:
        angle = math.pi * folded
        cosine, sine = math.cos(angle), math.sin(angle)
    terms = (
        2 * math.log(math.hypot(cosine, (n - 2 * r) * sine / n))
        for r in range(1, (n - 1) // 2 + 1)
    )
    result = math.fsum(terms)
    if n % 2 == 0:
        result += math.log(cosine)
    return result


def cycle_midpoint_exact(n, *, convention="principal", pi_branch="positive"):
    """Return the exact rational permanent at t=1/2 for the chosen path."""
    n = _validate_n(n)
    shift = _branch_shift(n, convention, pi_branch)
    if n % 2 == 0:
        return Fraction(0)
    odd_product = math.prod(range(1, n - 1, 2))
    sign = -1 if ((n - 1) // 2 - shift) % 2 else 1
    return Fraction(sign * odd_product**2, n ** (n - 1))


def cycle_permanent(n, t, *, convention="principal", pi_branch="positive"):
    """Return the analytic complex permanent in O(n) arithmetic operations.

    Extremely small values can underflow; use ``cycle_permanent_logabs``
    when a logarithmic value is needed. Exact midpoint values are available
    separately through ``cycle_midpoint_exact``.
    """
    n, t = _validate_n(n), _validate_t(t)
    shift = _branch_shift(n, convention, pi_branch)
    if n == 1 or t in {0.0, 1.0}:
        return 1 + 0j
    if t == 0.5 and n % 2 == 0:
        return 0j
    magnitude = math.exp(cycle_permanent_logabs(n, t))
    if t == 0.5:
        sign = -1 if ((n - 1) // 2 - shift) % 2 else 1
        return complex(sign * magnitude)
    if n % 2:
        if convention == "principal":
            return complex(magnitude)
        phase_turns = ((n - 1) // 2) * t
        return magnitude * cmath.exp(2j * math.pi * (phase_turns % 1))
    # Pair r and n-r; the remaining factor is exp(i*pi*t)*cos(pi*t).
    sign = 1 if t < 0.5 else -1
    phase = math.pi * ((n - 1 - 2 * shift) * t % 2)
    return sign * magnitude * cmath.exp(1j * phase)


def cycle_scaling_limit(t):
    """Return 1-a*cot(a), a=pi*min(t,1-t), with continuous endpoints.

    This is the limit of -log|per(gamma_n(t))|/n away from t=1/2.
    Its midpoint value 1 applies only to the odd-n subsequence; even-n
    midpoint permanents vanish exactly.
    """
    t = _validate_t(t)
    a = math.pi * min(t, 1 - t)
    if a < 1e-3:
        square = a * a
        return square * (1 / 3 + square * (1 / 45 + square * (2 / 945)))
    if t == 0.5:
        return 1.0
    return 1 - a / math.tan(a)


def write_analytic_figure(output_stem, *, num_points=501):
    """Write a cycle-only PDF, PNG, and JSON with formula provenance.

    This is a fresh analytic illustration, not a reanalysis of saved
    GPU measurements. No old data or figure is read or overwritten unless
    its path is explicitly selected by the caller.
    """
    from datetime import datetime, timezone
    import hashlib
    import json
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if num_points < 3 or num_points % 2 == 0:
        raise ValueError("num_points must be odd and at least 3, to include t=1/2")
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0, 1, num_points)
    limit = np.array([cycle_scaling_limit(t) for t in times])
    orders = ((7, 17, 101), (8, 18, 102))
    colors = ("#0072B2", "#D55E00", "#009E73")
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    for ax, ns, parity in zip(axes, orders, ("Odd orders", "Even orders")):
        for n, color in zip(ns, colors):
            logs = np.array([cycle_permanent_logabs(n, t) for t in times])
            values = -logs / n
            ax.plot(times, np.where(np.isfinite(values), values, np.nan),
                    label=f"$n={n}$", color=color, linewidth=1.5)
            midpoint = cycle_midpoint_exact(n, convention="literal")
            records.append({
                "n": n,
                "log_abs_permanent": [float(x) if np.isfinite(x) else None for x in logs],
                "exact_zero": [bool(n % 2 == 0 and t == 0.5) for t in times],
                "literal_midpoint": {"numerator": str(midpoint.numerator),
                                     "denominator": str(midpoint.denominator)},
            })
        ax.plot(times, limit, "--", color="#222222", linewidth=1.8,
                label=r"$1-a\cot a$")
        ax.set_title(parity, fontsize=12)
        ax.set_xlabel("Path parameter $t$")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.3)
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].set_ylabel(r"$-\log|\mathrm{per}(\gamma_n(t))|/n$")
    axes[1].annotate("Exact zero at $t=1/2$\n(logarithm diverges)",
                     xy=(0.5, 1.27), xytext=(0.69, 1.10), fontsize=9,
                     ha="center", arrowprops={"arrowstyle": "->", "color": "#555555"})
    fig.suptitle("Cycle permanents from the analytic Cauchy product", fontsize=14)
    fig.text(0.5, 0.01,
             "$a=\\pi\\min(t,1-t)$. Equal magnitudes for literal and principal angle conventions.\n"
             "The finite midpoint limit applies only to odd orders.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 0.96))
    for suffix, dpi in ((".pdf", 300), (".png", 180)):
        fig.savefig(str(output_stem) + suffix, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    record = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "analytic Cauchy permanent product; paired factors accumulated in log space",
        "formula": "per_literal = n^(-n) * product(r + (n-r)*exp(2*pi*i*t), r=1,...,n)",
        "source": "https://arxiv.org/abs/0907.2860 (Theorems 3-4); circulant Cauchy specialization",
        "phase_conventions": {
            "literal": "Fourier eigenangles 2*pi*k/n, k=0,...,n-1",
            "principal_positive": "(-pi, pi]; per = exp(-2*pi*i*((n-1)//2)*t) * per_literal",
            "principal_negative": "[-pi, pi); per = exp(-2*pi*i*(n//2)*t) * per_literal",
        },
        "scaling_limit": "1-a*cot(a), a=pi*min(t,1-t); value 1 at midpoint only along odd n",
        "null_log_value": "exact zero permanent, hence log absolute value = -infinity",
        "source_file": Path(__file__).name,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provenance": "Fresh formula evaluations, not GPU measurements or recovered historical samples.",
        "t": times.tolist(),
        "scaling_limit_values": limit.tolist(),
        "curves": records,
    }
    Path(str(output_stem) + ".json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-stem", default="outputs/geodesic_cycle_analytic_2026-09-04")
    parser.add_argument("--num-points", type=int, default=501)
    arguments = parser.parse_args()
    write_analytic_figure(arguments.output_stem, num_points=arguments.num_points)
