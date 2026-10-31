"""Occupancy statistics and labelled approximations for bipartite matchings.

This source-only research helper never computes a permanent or writes files.
Matrix entries, rook coefficients and activity accept only ``int``/``Fraction``
(excluding bool); floats are not silently converted into exact input data.
Exact occupancy results are Fractions. All fields of the two approximate models
are floating-point diagnostics, without a relative-error/confidence guarantee.

Monomer-dimer BP uses cavity probabilities m[u->v] with edge activity a=z*w:
    m[u->v] = 1 / (1 + sum(a[u,w]*m[w->u], w != v))
    p[uv] = a[uv]*m[u->v]*m[v->u] / (1+a[uv]*m[u->v]*m[v->u])
    log Z_B = sum_v log(1+sum_u a[uv]*m[u->v])
              - sum_uv log(1+a[uv]*m[u->v]*m[v->u]).
These are the weighted sum-product equations (one occupied edge per vertex).
For equal edge activities, setting Y[u->v]=z*m[u->v] gives the equations in
Lelarge, arXiv:1401.7923, Sec. 2.3 and Prop. 6 / Eq. (18):
https://arxiv.org/html/1401.7923 . The weighted version follows by inserting the
occupied-edge factor a[uv] in each local sum. The vertex/edge log-normalizer
formula follows from the same Bethe factorization. On forests it is exact at
the fixed point; cycles generally introduce error (ibid., Theorem 3).

We iterate at fixed activity, with damping; this is not the paper's annealing
optimization algorithm. Convergence here means a numerically small *undamped*
fixed-point residual, not a proof of closeness to the true partition function.
"""

from fractions import Fraction
import math


def _rational(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"{name} must be an int or Fraction, not {type(value).__name__}")
    value = Fraction(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _vector(values, name):
    if isinstance(values, (str, bytes, dict)):
        raise TypeError(f"{name} must be an iterable of rational numbers")
    try:
        return tuple(_rational(v, f"{name}[{i}]") for i, v in enumerate(values))
    except TypeError as exc:
        raise TypeError(f"invalid {name}: {exc}") from exc


def _matrix(matrix):
    if isinstance(matrix, (str, bytes, dict)):
        raise TypeError("matrix must be a rectangular iterable of rows")
    try:
        rows = tuple(_vector(row, f"matrix[{i}]") for i, row in enumerate(matrix))
    except TypeError as exc:
        raise TypeError(f"invalid matrix: {exc}") from exc
    columns = len(rows[0]) if rows else 0
    if any(len(row) != columns for row in rows):
        raise ValueError("matrix must be rectangular")
    return rows, len(rows), columns


def log_rational(value):
    """Numerical natural log of a positive exact input, without float overflow.

    The near-one branch avoids subtracting two nearly equal logarithms. The
    result is still a binary64 approximation, not an interval enclosure.
    """
    value = _rational(value, "value")
    if value == 0:
        raise ValueError("log_rational requires a positive value")
    if Fraction(1, 2) <= value <= 2:
        return math.log1p(float(value - 1))
    return math.log(value.numerator) - math.log(value.denominator)


def exact_occupancy_statistics(coefficients, activity):
    """Return rational statistics of P(K=k)=r[k]*z**k / sum(r[j]*z**j).

    Require r[0]=1, the empty matching's weight. A nonnegative coefficient
    vector need not be realizable by a graph; this function does not certify
    graph realizability. Trailing zeros are retained. ``max_size`` denotes the
    last *supplied* index, not necessarily an attainable or perfect matching.
    ``log_partition`` alone is floating point, for plotting large exact Z.
    """
    coefficients = _vector(coefficients, "coefficients")
    if not coefficients or coefficients[0] != 1:
        raise ValueError("rook coefficients must be nonempty with r[0] = 1")
    activity = _rational(activity, "activity")
    terms = []
    power = Fraction(1)
    for coefficient in coefficients:
        terms.append(coefficient * power)
        power *= activity
    partition = sum(terms, Fraction(0))
    probabilities = tuple(term / partition for term in terms)
    mean = sum((k * p for k, p in enumerate(probabilities)), Fraction(0))
    variance = sum(((k - mean) ** 2 * p for k, p in enumerate(probabilities)), Fraction(0))
    return {
        "model": "exact_rook_occupancy",
        "arithmetic": "Fraction; log_partition is binary64",
        "activity": activity,
        "partition": partition,
        "log_partition": log_rational(partition),
        "probabilities": probabilities,
        "mean": mean,
        "variance": variance,
        "empty_probability": probabilities[0],
        "max_size": len(coefficients) - 1,
        "max_size_probability": probabilities[-1],
    }


def independent_edge_statistics(matrix, activity):
    """Diagnostic relaxation ignoring matching exclusion, NOT matching BP.

    Each nonzero edge is independent with p=z*w/(1+z*w). The distribution is
    a numerical Bernoulli convolution, padded through the number of nonzero
    edges (including zero-activity edges). Its partition function prod(1+z*w)
    is an upper bound for the exact matching Z. ``invalid_size_probability``
    counts K>min(rows,columns), only a LOWER bound on the probability of an
    invalid matching: conflicting edges can also have smaller cardinality.
    Underflow/rounding of very small probabilities remains possible.
    """
    matrix, rows, columns = _matrix(matrix)
    activity = _rational(activity, "activity")
    probabilities = [1.0]
    edge_probabilities = [[0.0] * columns for _ in range(rows)]
    means, variances, logs = [], [], []
    nonzero_edges = 0
    for i, row in enumerate(matrix):
        for j, weight in enumerate(row):
            if weight == 0:
                continue
            nonzero_edges += 1
            a = activity * weight
            p_exact = a / (1 + a)
            q_exact = 1 / (1 + a)
            p, q = float(p_exact), float(q_exact)
            edge_probabilities[i][j] = p
            means.append(p)
            variances.append(float(p_exact * q_exact))
            logs.append(math.log1p(float(a)) if a <= 1 else log_rational(1 + a))
            updated = [0.0] * (len(probabilities) + 1)
            for k, mass in enumerate(probabilities):
                updated[k] += mass * q
                updated[k + 1] += mass * p
            probabilities = updated
    mass_before_normalization = math.fsum(probabilities)
    if not math.isfinite(mass_before_normalization) or mass_before_normalization <= 0:
        raise ArithmeticError("independent-edge convolution lost finite positive mass")
    probabilities = tuple(p / mass_before_normalization for p in probabilities)
    return {
        "model": "independent_edges_ignoring_exclusion",
        "arithmetic": "binary64 approximation after exact rational input validation",
        "partition_relation": "prod(1+z*w) is an upper bound, not the matching partition",
        "rows": rows,
        "columns": columns,
        "nonzero_edges": nonzero_edges,
        "probabilities": probabilities,
        "edge_probabilities": edge_probabilities,
        "mean": math.fsum(means),
        "variance": math.fsum(variances),
        "log_partition": math.fsum(logs),
        "empty_probability": probabilities[0],
        "matching_size_limit": min(rows, columns),
        "invalid_size_probability": math.fsum(probabilities[min(rows, columns) + 1:]),
        "mass_before_normalization": mass_before_normalization,
    }


def total_variation_distance(first, second):
    """Numerical TV distance of normalized size distributions, padding zeros.

    This compares distributions of cardinality only, not distributions over
    particular matchings. Float entries are permitted HERE because this is
    the explicit interface between exact probabilities and numeric models.
    No clipping or renormalization is performed on the supplied distributions.
    """
    def checked(values):
        if isinstance(values, (str, bytes, dict)):
            raise TypeError("probabilities must be an iterable")
        result = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, Fraction, float)):
                raise TypeError("probabilities must be int, Fraction or float")
            if value < 0 or value > 1:
                raise ValueError("probabilities must lie in [0, 1]")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("probabilities must be finite")
            result.append(number)
        if not result or not math.isclose(math.fsum(result), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("probabilities must sum to 1 within 1e-12")
        return result

    first, second = checked(first), checked(second)
    size = max(len(first), len(second))
    first.extend([0.0] * (size - len(first)))
    second.extend([0.0] * (size - len(second)))
    return 0.5 * math.fsum(abs(p - q) for p, q in zip(first, second))


def _logaddexp(first, second):
    if first == -math.inf:
        return second
    if second == -math.inf:
        return first
    high, low = max(first, second), min(first, second)
    return high + math.log1p(math.exp(low - high))


def _numeric_control(value, name, *, positive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise TypeError(f"{name} must be a finite numeric control")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return value


def belief_propagation_statistics(matrix, activity, *, damping=0.5,
                                  tolerance=1e-10, max_iterations=10000):
    """Damped sum-product monomer-dimer BP on the matrix's bipartite graph.

    ``damping`` is the weight on a new update, in (0,1]. Log-domain arithmetic
    avoids overflowing a=z*w. Prefix/suffix log sums avoid subtracting a large
    excluded edge from a vertex sum. Iteration cost and storage are O(E+V).

    Residual = max |log(m)-log(F(m))| for the UNDAMPED update map F, at the
    returned iterate. It is not a bound on model error. On nonconvergence,
    log_partition/mean/edge_probabilities are None rather than usable estimates.
    No matching-size distribution or variance is inferred from edge marginals.
    """
    matrix, rows, columns = _matrix(matrix)
    activity = _rational(activity, "activity")
    damping = _numeric_control(damping, "damping")
    tolerance = _numeric_control(tolerance, "tolerance")
    if damping > 1:
        raise ValueError("damping must be in (0, 1]")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")

    # Each incidence stores an outgoing directed message. Opposite = index^1.
    adjacency = [[] for _ in range(rows + columns)]
    edges, log_activities = [], []
    if activity:
        for i, row in enumerate(matrix):
            for j, weight in enumerate(row):
                if weight:
                    e = len(edges)
                    edges.append((i, j))
                    log_activities.append(log_rational(activity * weight))
                    adjacency[i].append(2 * e)
                    adjacency[rows + j].append(2 * e + 1)

    def update(log_messages):
        targets = [0.0] * len(log_messages)
        vertex_logs = []
        for incident in adjacency:
            incoming = [log_activities[d // 2] + log_messages[d ^ 1] for d in incident]
            # Prefix includes the unmatched-vertex contribution 1 (log=0).
            prefix = [0.0]
            for term in incoming:
                prefix.append(_logaddexp(prefix[-1], term))
            suffix = -math.inf
            for k in range(len(incident) - 1, -1, -1):
                targets[incident[k]] = -_logaddexp(prefix[k], suffix)
                suffix = _logaddexp(suffix, incoming[k])
            vertex_logs.append(prefix[-1])
        return targets, vertex_logs

    log_messages = [0.0] * (2 * len(edges))
    targets, vertex_logs = update(log_messages)
    residual = max((abs(a - b) for a, b in zip(log_messages, targets)), default=0.0)
    iterations = 0
    log_new = math.log(damping)
    log_old = math.log1p(-damping) if damping < 1 else -math.inf
    while residual > tolerance and iterations < max_iterations:
        log_messages = [_logaddexp(log_old + old, log_new + new)
                        for old, new in zip(log_messages, targets)]
        targets, vertex_logs = update(log_messages)
        residual = max((abs(a - b) for a, b in zip(log_messages, targets)), default=0.0)
        iterations += 1
    converged = math.isfinite(residual) and residual <= tolerance
    result = {
        "model": "monomer_dimer_belief_propagation",
        "arithmetic": "binary64 log-domain messages; no approximation-error guarantee",
        "status": "CONVERGED" if converged else "NOT_CONVERGED",
        "converged": converged,
        "rows": rows,
        "columns": columns,
        "active_edges": len(edges),
        "iterations": iterations,
        "max_iterations": max_iterations,
        "damping": damping,
        "tolerance": tolerance,
        "residual": residual,
        "residual_definition": "max absolute undamped fixed-point residual in log messages",
        "log_partition": None,
        "mean": None,
        "edge_probabilities": None,
        "vertex_monomer_probabilities": None,
    }
    if not converged:
        return result
    edge_probabilities = [[0.0] * columns for _ in range(rows)]
    means, edge_logs = [], []
    for e, (i, j) in enumerate(edges):
        log_odds = log_activities[e] + log_messages[2 * e] + log_messages[2 * e + 1]
        edge_logs.append(_logaddexp(0.0, log_odds))
        # Separate logistic branches avoid exp overflow and preserve tiny p.
        probability = (1 / (1 + math.exp(-log_odds)) if log_odds >= 0
                       else math.exp(log_odds) / (1 + math.exp(log_odds)))
        edge_probabilities[i][j] = probability
        means.append(probability)
    result.update({
        "log_partition": math.fsum(vertex_logs + [-term for term in edge_logs]),
        "mean": math.fsum(means),
        "edge_probabilities": edge_probabilities,
        # Row vertices first, then column vertices.
        "vertex_monomer_probabilities": [math.exp(-term) for term in vertex_logs],
    })
    return result
