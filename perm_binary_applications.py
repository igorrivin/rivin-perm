"""Concrete applications of exact zero-one permanents.

A binary biadjacency matrix has one permanent term for each perfect matching
of its bipartite graph.  This module supplies small, reproducible application
families rather than embedding graph-library dependencies in the core solver:

* domino/dimer coverings of open or periodically wrapped grid graphs;
* anisotropic grid-dimer generating polynomials, with small exact oracles and
  a scalable modular coefficient backend;
* derangements, viewed as permutations avoiding the diagonal;
* block all-ones matrices attaining the Bregman--Minc bound; and
* Latin transversals reduced by inclusion--exclusion to multiple ordinary
  binary permanents (not misidentified as one ordinary permanent).

Open 2-D grid cases are useful exact oracles (planar dimers admit the classical
Kasteleyn determinant treatment).  The proposed 4-by-4-by-5 grid with two
periodic axes is a genuinely nonplanar permanent target; merely having three
nontrivial dimensions is not by itself sufficient for nonplanarity.  A
periodic grid axis must have even length at least four: odd wraparound breaks
the parity bipartition, while length two creates parallel physical bonds that
a zero-one matrix cannot represent without multiplicities.
"""

from itertools import product
import math
import operator
from typing import Dict, Iterable, List, Sequence, Tuple

from perm import matrix_permanent
from perm_binary import binary_permanent
from perm_polynomial import affine_permanent_mod


Coordinate = Tuple[int, ...]
GridEdge = Tuple[int, int]


def _index(value, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return int(result)


def _execution_options(parallel: bool, num_threads: int) -> Tuple[bool, int]:
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a boolean")
    threads = _index(num_threads, "num_threads")
    if threads < 0:
        raise ValueError("num_threads must be non-negative")
    return parallel, threads


def _normalize_shape(shape: Sequence[int]) -> Tuple[int, ...]:
    try:
        raw_shape = tuple(shape)
    except TypeError as exc:
        raise TypeError("shape must be a sequence of positive integers") from exc
    if not raw_shape:
        raise ValueError("shape must contain at least one dimension")

    normalized = []
    for index, extent in enumerate(raw_shape):
        extent = _index(extent, f"shape[{index}]")
        if extent <= 0:
            raise ValueError("shape dimensions must be positive")
        normalized.append(extent)
    return tuple(normalized)


def _normalize_periodic_axes(
    periodic_axes: Iterable[int], shape: Sequence[int]
) -> Tuple[int, ...]:
    try:
        raw_axes = tuple(periodic_axes)
    except TypeError as exc:
        raise TypeError("periodic_axes must be an iterable of axis indices") from exc

    axes = []
    seen = set()
    for raw_axis in raw_axes:
        axis = _index(raw_axis, "periodic axis")
        if axis < 0 or axis >= len(shape):
            raise ValueError("periodic axis is outside the grid dimension")
        if axis in seen:
            raise ValueError("periodic_axes must not contain duplicates")
        extent = shape[axis]
        if extent % 2:
            raise ValueError(
                "a periodic grid axis must have even length to preserve "
                "the parity bipartition"
            )
        if extent < 4:
            raise ValueError(
                "a periodic binary-grid axis must have length at least 4; "
                "length 2 creates parallel bonds"
            )
        seen.add(axis)
        axes.append(axis)
    return tuple(sorted(axes))


def _grid_edges(
    shape: Sequence[int], periodic_axes: Iterable[int]
) -> Tuple[Tuple[Coordinate, ...], Tuple[Coordinate, ...], Tuple[Tuple[GridEdge, ...], ...]]:
    """Return balanced parity classes and ``(column, axis)`` row edges."""
    normalized_shape = _normalize_shape(shape)
    periodic = set(_normalize_periodic_axes(periodic_axes, normalized_shape))
    coordinates = tuple(
        product(*(range(extent) for extent in normalized_shape))
    )
    left = tuple(point for point in coordinates if sum(point) % 2 == 0)
    right = tuple(point for point in coordinates if sum(point) % 2 == 1)
    if len(left) != len(right):
        raise ValueError(
            "the grid has unequal parity classes and therefore no square "
            "bipartite adjacency matrix"
        )

    right_index = {point: index for index, point in enumerate(right)}
    rows: List[Tuple[GridEdge, ...]] = []
    for point in left:
        row: Dict[int, int] = {}
        for axis, extent in enumerate(normalized_shape):
            for direction in (-1, 1):
                neighbor = list(point)
                neighbor[axis] += direction
                if not 0 <= neighbor[axis] < extent:
                    if axis not in periodic:
                        continue
                    neighbor[axis] %= extent
                neighbor_tuple = tuple(neighbor)
                column = right_index.get(neighbor_tuple)
                if column is None:
                    raise RuntimeError(
                        "grid edge failed to cross the parity bipartition"
                    )
                previous_axis = row.get(column)
                if previous_axis is not None and previous_axis != axis:
                    raise RuntimeError("two grid axes produced the same edge")
                row[column] = axis
        rows.append(tuple(sorted(row.items())))
    return left, right, tuple(rows)


def grid_biadjacency(
    shape: Sequence[int], *, periodic_axes: Iterable[int] = ()
) -> List[List[int]]:
    """Return the zero-one biadjacency matrix of a Cartesian grid.

    Vertices are partitioned by coordinate-sum parity.  Open boundaries are
    used except on the explicitly listed periodic axes.
    """
    left, right, edges = _grid_edges(shape, periodic_axes)
    matrix = [[0] * len(right) for _ in left]
    for row, row_edges in enumerate(edges):
        for column, _axis in row_edges:
            matrix[row][column] = 1
    return matrix


def weighted_grid_biadjacency(
    shape: Sequence[int],
    axis_weights: Sequence[int],
    *,
    periodic_axes: Iterable[int] = (),
) -> List[List[int]]:
    """Return a grid biadjacency matrix weighted by edge direction.

    The permanent is the finite-volume dimer partition function.  Setting one
    axis weight to ``w`` and the others to one weights a matching by
    ``w**k``, where ``k`` is its number of dimers parallel to that axis.
    """
    normalized_shape = _normalize_shape(shape)
    try:
        raw_weights = tuple(axis_weights)
    except TypeError as exc:
        raise TypeError("axis_weights must be a sequence of integers") from exc
    if len(raw_weights) != len(normalized_shape):
        raise ValueError("axis_weights must contain one weight per grid axis")
    weights = tuple(
        _index(weight, f"axis_weights[{axis}]")
        for axis, weight in enumerate(raw_weights)
    )

    left, right, edges = _grid_edges(normalized_shape, periodic_axes)
    matrix = [[0] * len(right) for _ in left]
    for row, row_edges in enumerate(edges):
        for column, axis in row_edges:
            matrix[row][column] = weights[axis]
    return matrix


def grid_dimer_count(
    shape: Sequence[int],
    *,
    periodic_axes: Iterable[int] = (),
    backend: str = "auto",
    parallel: bool = True,
    num_threads: int = 0,
) -> int:
    """Count perfect-matchings (close-packed dimers) of a grid exactly."""
    normalized_shape = _normalize_shape(shape)
    parallel, num_threads = _execution_options(parallel, num_threads)
    # Validate the named backend even if an odd vertex count lets us return
    # before constructing a matrix.
    binary_permanent(
        [], backend=backend, parallel=parallel, num_threads=num_threads
    )
    # Validate periodic geometry even when the odd vertex count makes the
    # matching count trivially zero.
    periodic = _normalize_periodic_axes(periodic_axes, normalized_shape)
    if math.prod(normalized_shape) % 2:
        return 0
    matrix = grid_biadjacency(normalized_shape, periodic_axes=periodic)
    return binary_permanent(
        matrix,
        backend=backend,
        parallel=parallel,
        num_threads=num_threads,
    )


def anisotropic_grid_partition(
    shape: Sequence[int],
    weight: int,
    *,
    axis: int = -1,
    periodic_axes: Iterable[int] = (),
    parallel: bool = True,
    num_threads: int = 0,
) -> int:
    """Evaluate a one-axis anisotropic grid-dimer partition function."""
    normalized_shape = _normalize_shape(shape)
    axis = _index(axis, "axis")
    if axis < 0:
        axis += len(normalized_shape)
    if axis < 0 or axis >= len(normalized_shape):
        raise ValueError("axis is outside the grid dimension")
    weight = _index(weight, "weight")
    parallel, num_threads = _execution_options(parallel, num_threads)
    periodic = _normalize_periodic_axes(periodic_axes, normalized_shape)
    if math.prod(normalized_shape) % 2:
        return 0
    weights = [1] * len(normalized_shape)
    weights[axis] = weight
    matrix = weighted_grid_biadjacency(
        normalized_shape, weights, periodic_axes=periodic
    )
    return matrix_permanent(
        matrix, parallel=parallel, num_threads=num_threads
    )


def axis_dimer_profile(
    shape: Sequence[int],
    *,
    axis: int = -1,
    periodic_axes: Iterable[int] = (),
    max_bipartition_order: int = 20,
) -> Tuple[int, ...]:
    """Return coefficients by the number of dimers parallel to one axis.

    This definition-level subset DP is intended for application oracles and
    small finite volumes.  Larger profiles should use
    :func:`axis_dimer_profile_mod`, which carries the coefficient vector
    directly; the explicit order guard prevents an accidental memory
    explosion.
    """
    normalized_shape = _normalize_shape(shape)
    axis = _index(axis, "axis")
    if axis < 0:
        axis += len(normalized_shape)
    if axis < 0 or axis >= len(normalized_shape):
        raise ValueError("axis is outside the grid dimension")
    max_order = _index(max_bipartition_order, "max_bipartition_order")
    if max_order < 0:
        raise ValueError("max_bipartition_order must be non-negative")

    periodic = _normalize_periodic_axes(periodic_axes, normalized_shape)
    if math.prod(normalized_shape) % 2:
        return (0,)

    left, right, edges = _grid_edges(normalized_shape, periodic)
    order = len(left)
    if order > max_order:
        raise ValueError(
            f"bipartition order {order} exceeds the profile-DP limit "
            f"{max_order}; use axis_dimer_profile_mod instead"
        )

    ordered_edges = sorted(edges, key=len)
    states: Dict[int, Tuple[int, ...]] = {0: (1,)}
    for row_edges in ordered_edges:
        next_states: Dict[int, List[int]] = {}
        for used, coefficients in states.items():
            for column, edge_axis in row_edges:
                bit = 1 << column
                if used & bit:
                    continue
                shift = int(edge_axis == axis)
                target = used | bit
                output = next_states.get(target)
                required = len(coefficients) + shift
                if output is None:
                    output = [0] * required
                    next_states[target] = output
                elif len(output) < required:
                    output.extend([0] * (required - len(output)))
                for degree, coefficient in enumerate(coefficients):
                    output[degree + shift] += coefficient
        if not next_states:
            return (0,) * (order + 1)
        states = {
            used: tuple(coefficients)
            for used, coefficients in next_states.items()
        }

    coefficients = states.get((1 << len(right)) - 1, (0,))
    return coefficients + (0,) * (order + 1 - len(coefficients))


def axis_dimer_profile_mod(
    shape: Sequence[int],
    modulus: int,
    *,
    axis: int = -1,
    periodic_axes: Iterable[int] = (),
    backend: str = "auto",
    parallel: bool = True,
    num_threads: int = 0,
) -> Tuple[int, ...]:
    """Return an anisotropic dimer profile modulo an odd integer.

    Edges parallel to ``axis`` contribute ``X`` and all other grid edges
    contribute one.  Unlike :func:`axis_dimer_profile`, this path carries the
    complete coefficient vector through fixed-sign Glynn enumeration and is
    therefore suitable beyond the small subset-DP oracle range.
    """
    normalized_shape = _normalize_shape(shape)
    axis = _index(axis, "axis")
    if axis < 0:
        axis += len(normalized_shape)
    if axis < 0 or axis >= len(normalized_shape):
        raise ValueError("axis is outside the grid dimension")
    parallel, num_threads = _execution_options(parallel, num_threads)
    periodic = _normalize_periodic_axes(periodic_axes, normalized_shape)

    # Validate the modulus and backend even when the grid has no perfect
    # matching for the trivial parity reason.
    if math.prod(normalized_shape) % 2:
        affine_permanent_mod(
            [],
            [],
            modulus,
            backend=backend,
            parallel=parallel,
            num_threads=num_threads,
        )
        return (0,)

    left, right, edges = _grid_edges(normalized_shape, periodic)
    constant = [[0] * len(right) for _ in left]
    linear = [[0] * len(right) for _ in left]
    for row, row_edges in enumerate(edges):
        for column, edge_axis in row_edges:
            if edge_axis == axis:
                linear[row][column] = 1
            else:
                constant[row][column] = 1
    return affine_permanent_mod(
        constant,
        linear,
        modulus,
        backend=backend,
        parallel=parallel,
        num_threads=num_threads,
    )


def evaluate_polynomial(coefficients: Sequence[int], value: int) -> int:
    """Evaluate coefficients in ascending order with exact Horner arithmetic."""
    value = _index(value, "value")
    result = 0
    for coefficient in reversed(tuple(coefficients)):
        coefficient = _index(coefficient, "coefficient")
        result = result * value + coefficient
    return result


def derangement_matrix(order: int) -> List[List[int]]:
    """Return ``J-I``; its permanent is the subfactorial ``!order``."""
    order = _index(order, "order")
    if order < 0:
        raise ValueError("order must be non-negative")
    return [
        [int(row != column) for column in range(order)]
        for row in range(order)
    ]


def bregman_extremal_matrix(
    block_size: int, block_count: int
) -> List[List[int]]:
    """Return diagonal all-ones blocks with permanent ``(d!)**k``."""
    block_size = _index(block_size, "block_size")
    block_count = _index(block_count, "block_count")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if block_count < 0:
        raise ValueError("block_count must be non-negative")
    order = block_size * block_count
    return [
        [
            int(row // block_size == column // block_size)
            for column in range(order)
        ]
        for row in range(order)
    ]


def cyclic_latin_square(order: int) -> List[List[int]]:
    """Return the addition table ``L[i,j] = i+j (mod order)``."""
    order = _index(order, "order")
    if order < 0:
        raise ValueError("order must be non-negative")
    if order == 0:
        return []
    return [
        [(row + column) % order for column in range(order)]
        for row in range(order)
    ]


def _normalize_latin_square(square) -> Tuple[Tuple[int, ...], ...]:
    try:
        order = len(square)
    except TypeError as exc:
        raise TypeError("Latin square must be a sequence of rows") from exc
    raw_rows = []
    for row_index, row in enumerate(square):
        try:
            raw_row = tuple(row)
        except TypeError as exc:
            raise TypeError("Latin square rows must be sequences") from exc
        if len(raw_row) != order:
            raise ValueError("Latin square must be square")
        normalized_row = tuple(
            _index(entry, f"Latin entry ({row_index}, {column})")
            for column, entry in enumerate(raw_row)
        )
        raw_rows.append(normalized_row)
    if len(raw_rows) != order:
        raise ValueError("Latin square row count changed during validation")
    if order == 0:
        return ()

    first_row = raw_rows[0]
    if len(set(first_row)) != order:
        raise ValueError(
            "each Latin-square row must contain every symbol exactly once"
        )
    symbol_index = {symbol: index for index, symbol in enumerate(first_row)}
    expected = set(range(order))
    rows = []
    for raw_row in raw_rows:
        try:
            normalized_row = tuple(symbol_index[symbol] for symbol in raw_row)
        except KeyError as exc:
            raise ValueError(
                "every Latin-square row must use the same symbol set"
            ) from exc
        if set(normalized_row) != expected:
            raise ValueError(
                "each Latin-square row must contain every symbol exactly once"
            )
        rows.append(normalized_row)
    for column in range(order):
        if {rows[row][column] for row in range(order)} != expected:
            raise ValueError(
                "each Latin-square column must contain every symbol exactly once"
            )
    return tuple(rows)


def latin_transversal_count(
    square,
    *,
    backend: str = "auto",
    parallel: bool = False,
    num_threads: int = 0,
    max_order: int = 10,
) -> int:
    """Count Latin-square transversals via ordinary binary permanents.

    A transversal is fundamentally a 3-dimensional matching, not one ordinary
    matrix permanent.  If ``A_S[i,j]`` records whether ``square[i,j]`` belongs
    to a symbol subset ``S``, coefficient extraction gives the exact reduction

    ``sum_S (-1)**(n-|S|) * permanent(A_S)``.

    Thus this clear reference implementation makes ``2**n`` binary-permanent
    calls and is deliberately guarded to small orders.
    """
    normalized = _normalize_latin_square(square)
    order = len(normalized)
    limit = _index(max_order, "max_order")
    if limit < 0:
        raise ValueError("max_order must be non-negative")
    if order > limit:
        raise ValueError(
            f"Latin-square order {order} exceeds the inclusion-exclusion "
            f"limit {limit}"
        )
    parallel, num_threads = _execution_options(parallel, num_threads)

    total = 0
    for symbols in range(1 << order):
        matrix = [
            [int((symbols >> normalized[row][column]) & 1)
             for column in range(order)]
            for row in range(order)
        ]
        count = binary_permanent(
            matrix,
            backend=backend,
            parallel=parallel,
            num_threads=num_threads,
        )
        if (order - bin(symbols).count("1")) & 1:
            total -= count
        else:
            total += count
    return total


__all__ = [
    "anisotropic_grid_partition",
    "axis_dimer_profile",
    "axis_dimer_profile_mod",
    "bregman_extremal_matrix",
    "cyclic_latin_square",
    "derangement_matrix",
    "evaluate_polynomial",
    "grid_biadjacency",
    "grid_dimer_count",
    "latin_transversal_count",
    "weighted_grid_biadjacency",
]
