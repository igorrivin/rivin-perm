"""Exact permanents of zero-one matrices.

For a zero-one matrix the permanent counts perfect matchings in the
corresponding bipartite graph.  The portable implementations below include a
subset dynamic program and a live-column frontier dynamic program for sparse
supports.  Both avoid the arithmetic and CRT overhead of the general
integer-matrix implementation and naturally return arbitrary-size Python
integers.

The public API deliberately treats rectangular matrices as an error.  Several
different quantities are called a "rectangular permanent" in the literature;
perfect matchings of all vertices, which is the interpretation used here, are
defined only when the two bipartition classes have the same size.
"""

import math
import operator
from typing import List, Optional, Sequence, Tuple

import numpy as np


try:
    import perm_binary_ext as _binary_ext
except ImportError:
    _binary_ext = None


HAS_BINARY_EXT = (
    _binary_ext is not None
    and hasattr(_binary_ext, "binary_permanent_mod_p")
)


_FRONTIER_ORDERING_WORK_BUDGET = 2_000_000
_AUTO_FRONTIER_MIN_ORDER = 20
_AUTO_FRONTIER_MAX_AVERAGE_DEGREE = 4
_AUTO_FRONTIER_MAX_WIDTH = 24
_AUTO_FRONTIER_MAX_STATES = 500_000


class FrontierStateLimitExceeded(RuntimeError):
    """Raised when a frontier layer would exceed its configured state cap."""

    def __init__(
        self,
        max_states: int,
        observed_states: int,
        processed_rows: int,
    ) -> None:
        self.max_states = max_states
        self.observed_states = observed_states
        self.processed_rows = processed_rows
        super().__init__(
            f"frontier state limit {max_states} exceeded while processing "
            f"row {processed_rows}; at least {observed_states} states "
            "would be required"
        )


def _validate_num_threads(num_threads: int) -> int:
    """Validate a thread count, where zero requests backend auto-selection."""
    if isinstance(num_threads, (bool, np.bool_)):
        raise TypeError("num_threads must be an integer")
    try:
        value = operator.index(num_threads)
    except TypeError as exc:
        raise TypeError("num_threads must be an integer") from exc
    if value < 0:
        raise ValueError("num_threads must be non-negative")
    return value


def _validate_frontier_max_states(max_states: int) -> int:
    """Return a positive integral frontier state limit."""
    if isinstance(max_states, (bool, np.bool_)):
        raise TypeError("max_states must be an integer")
    try:
        value = operator.index(max_states)
    except TypeError as exc:
        raise TypeError("max_states must be an integer") from exc
    value = int(value)
    if value <= 0:
        raise ValueError("max_states must be positive")
    return value


def _validate_frontier_row_order(
    row_order, order: int
) -> Optional[Tuple[int, ...]]:
    """Validate an exact non-boolean permutation of the original rows."""
    if row_order is None:
        return None
    try:
        declared_length = len(row_order)
    except TypeError as exc:
        raise TypeError("row_order must be a sequence of row indices") from exc
    if declared_length != order:
        raise ValueError("row_order must contain every row exactly once")

    normalized = []
    for position, raw_row in enumerate(row_order):
        if isinstance(raw_row, (bool, np.bool_)):
            raise TypeError("row_order entries must be non-boolean integers")
        try:
            row = operator.index(raw_row)
        except TypeError as exc:
            raise TypeError(
                "row_order entries must be integers; entry "
                f"{position} is {type(raw_row).__name__}"
            ) from exc
        normalized.append(int(row))
    if len(normalized) != order:
        raise ValueError("row_order length changed during validation")
    if set(normalized) != set(range(order)):
        raise ValueError("row_order must contain every row exactly once")
    return tuple(normalized)


def _validate_modulus(modulus: int) -> int:
    """Return a positive signed-64-bit modulus greater than one."""
    if isinstance(modulus, (bool, np.bool_)):
        raise TypeError("modulus must be an integer")
    try:
        value = operator.index(modulus)
    except TypeError as exc:
        raise TypeError("modulus must be an integer") from exc
    if value <= 1:
        raise ValueError("modulus must be greater than 1")
    if value > (1 << 63) - 1:
        raise ValueError("modulus must fit in a positive signed 64-bit integer")
    return value


def _validate_residue_prime(modulus: int, order: int) -> int:
    """Validate the modulus required by the native ratio-based kernel."""
    modulus = _validate_modulus(modulus)
    if modulus == 2 or modulus <= order or not _is_prime(modulus):
        raise ValueError(
            "modulus must be an odd prime greater than the matrix order"
        )
    return modulus


def _matrix_order(matrix) -> int:
    """Validate the two-dimensional square shape and return its order."""
    shape = getattr(matrix, "shape", None)
    if shape is not None:
        try:
            shape = tuple(shape)
        except TypeError as exc:
            raise ValueError("matrix must have a two-dimensional square shape") from exc
        if len(shape) != 2:
            raise ValueError("matrix must be two-dimensional")
        if shape[0] != shape[1]:
            raise ValueError("matrix must be square")

    try:
        n = len(matrix)
    except TypeError as exc:
        raise TypeError("matrix must be a sequence of rows") from exc

    for row in matrix:
        try:
            width = len(row)
        except TypeError as exc:
            raise TypeError("matrix rows must be sequences") from exc
        if width != n:
            raise ValueError("matrix must be square")
    return n


def _normalize_binary_matrix(matrix, *, strict: bool) -> Tuple[Tuple[int, ...], ...]:
    """Return an immutable normalized matrix after validating its entries.

    Strict mode accepts integral zeroes and ones (including booleans).  The
    explicit non-strict mode constructs the support matrix of an integral
    matrix, mapping every nonzero entry to one.  Floats and objects accepted
    merely because they are truthy are rejected in both modes.
    """
    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean")

    n = _matrix_order(matrix)
    normalized = []
    for row_index, row in enumerate(matrix):
        normalized_row = []
        for column_index, entry in enumerate(row):
            if isinstance(entry, (bool, np.bool_)):
                value = int(entry)
            else:
                try:
                    value = operator.index(entry)
                except TypeError as exc:
                    raise TypeError(
                        "binary matrix entries must be integers; "
                        f"entry ({row_index}, {column_index}) is "
                        f"{type(entry).__name__}"
                    ) from exc
            value = int(value)
            if strict and value not in (0, 1):
                raise ValueError(
                    "binary matrix entries must be 0 or 1 in strict mode; "
                    f"entry ({row_index}, {column_index}) is {value}"
                )
            normalized_row.append(1 if value else 0)
        if len(normalized_row) != n:
            raise ValueError("matrix row width changed during validation")
        normalized.append(tuple(normalized_row))

    # A hostile/custom sequence can report one length and yield another number
    # of rows.  Ordinary list/tuple/ndarray inputs pay only this cheap check.
    if len(normalized) != n:
        raise ValueError("matrix row count changed during validation")
    return tuple(normalized)


def _row_masks(matrix: Sequence[Sequence[int]]) -> Tuple[int, ...]:
    """Pack every binary row into a Python integer bit mask."""
    masks = []
    for row in matrix:
        mask = 0
        for column, entry in enumerate(row):
            if entry:
                mask |= 1 << column
        masks.append(mask)
    return tuple(masks)


def _permanent_dp(
    row_masks: Sequence[int], modulus: Optional[int] = None
) -> int:
    """Subset-DP permanent for packed binary rows.

    Rows are processed in nondecreasing degree.  Row permutation does not
    change the permanent and this order reduces the number of live partial
    matchings for sparse inputs.
    """
    n = len(row_masks)
    if n == 0:
        return 1 if modulus is None else 1 % modulus
    if any(mask == 0 for mask in row_masks):
        return 0

    ordered_masks = sorted(row_masks, key=lambda mask: bin(mask).count("1"))
    states = {0: 1}
    all_columns = (1 << n) - 1

    for allowed in ordered_masks:
        next_states = {}
        for used, count in states.items():
            available = allowed & (all_columns ^ used)
            while available:
                bit = available & -available
                target = used | bit
                value = next_states.get(target, 0) + count
                if modulus is not None:
                    value %= modulus
                next_states[target] = value
                available ^= bit
        if not next_states:
            return 0
        states = next_states

    return states.get(all_columns, 0)


def _transpose_row_masks(row_masks: Sequence[int]) -> Tuple[int, ...]:
    """Return packed rows for the transpose of a square binary matrix."""
    order = len(row_masks)
    transposed = [0] * order
    for row, mask in enumerate(row_masks):
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            transposed[bit.bit_length() - 1] |= 1 << row
            remaining ^= bit
    return tuple(transposed)


def _frontier_greedy_order(
    row_masks: Sequence[int],
) -> Tuple[Tuple[int, ...], int, int]:
    """Choose a deterministic row order with a small live-column frontier.

    A bounded number of starting rows is tried.  Each greedy step minimizes
    the maximum width reached, then the new width, favoring rows that close
    columns.  The work budget keeps ordering cheap for large or dense inputs.
    The returned integers are maximum width and total frontier area.
    """
    order = len(row_masks)
    if order == 0:
        return (), 0, 0

    row_columns = []
    column_degrees = [0] * order
    total_entries = 0
    for mask in row_masks:
        columns = []
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            column = bit.bit_length() - 1
            if column >= order:
                raise ValueError("row mask contains a column outside the matrix")
            columns.append(column)
            column_degrees[column] += 1
            total_entries += 1
            remaining ^= bit
        row_columns.append(tuple(columns))

    average_degree_ceiling = max(1, (total_entries + order - 1) // order)
    work_per_start = max(1, order * order * average_degree_ceiling)
    start_count = min(
        order,
        max(1, _FRONTIER_ORDERING_WORK_BUDGET // work_per_start),
    )
    ranked_starts = sorted(
        range(order), key=lambda row: (len(row_columns[row]), row)
    )
    if start_count == order:
        starts = ranked_starts
    elif start_count == 1:
        starts = [ranked_starts[0]]
    else:
        starts = [
            ranked_starts[index * (order - 1) // (start_count - 1)]
            for index in range(start_count)
        ]

    best = None
    for start in starts:
        selected = [False] * order
        processed_counts = [0] * order
        candidate_order = []
        width = 0
        maximum_width = 0
        frontier_area = 0

        for step in range(order):
            choice = None
            candidates = (start,) if step == 0 else range(order)
            for row in candidates:
                if selected[row]:
                    continue
                opened = 0
                closed = 0
                for column in row_columns[row]:
                    count = processed_counts[column]
                    degree = column_degrees[column]
                    if count == 0 and degree > 1:
                        opened += 1
                    if count > 0 and count == degree - 1:
                        closed += 1
                next_width = width + opened - closed
                score = (
                    max(maximum_width, next_width),
                    next_width,
                    -closed,
                    opened,
                    row,
                )
                if choice is None or score < choice[0]:
                    choice = (score, row, next_width)

            if choice is None:
                raise RuntimeError("frontier ordering failed to select a row")
            _, row, width = choice
            selected[row] = True
            candidate_order.append(row)
            for column in row_columns[row]:
                processed_counts[column] += 1
            maximum_width = max(maximum_width, width)
            frontier_area += width

        candidate = (
            maximum_width,
            frontier_area,
            tuple(candidate_order),
        )
        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise RuntimeError("frontier ordering failed")
    return best[2], best[0], best[1]


def _frontier_permanent_masks(
    row_masks: Sequence[int],
    row_order: Sequence[int],
    max_states: int,
) -> int:
    """Exact live-column frontier DP for validated masks and row order."""
    order = len(row_masks)
    if order == 0:
        return 1

    last_positions = [-1] * order
    for position, row in enumerate(row_order):
        remaining = row_masks[row]
        while remaining:
            bit = remaining & -remaining
            column = bit.bit_length() - 1
            if column >= order:
                raise ValueError("row mask contains a column outside the matrix")
            last_positions[column] = position
            remaining ^= bit
    if any(position < 0 for position in last_positions):
        return 0

    expiring_masks = [0] * order
    for column, position in enumerate(last_positions):
        expiring_masks[position] |= 1 << column

    states = {0: 1}
    for position, row in enumerate(row_order):
        expiring = expiring_masks[position]
        next_states = {}
        for used, count in states.items():
            available = row_masks[row] & ~used
            while available:
                bit = available & -available
                available ^= bit
                target = used | bit
                if target & expiring != expiring:
                    continue
                target &= ~expiring
                previous = next_states.get(target)
                if previous is None:
                    if len(next_states) >= max_states:
                        raise FrontierStateLimitExceeded(
                            max_states, max_states + 1, position + 1
                        )
                    next_states[target] = count
                else:
                    next_states[target] = previous + count
        states = next_states
        if not states:
            return 0
    return states.get(0, 0)


def binary_permanent_frontier(
    matrix,
    *,
    row_order=None,
    max_states: int = 500_000,
    strict: bool = True,
) -> int:
    """Compute an exact binary permanent with a live-column frontier DP.

    ``row_order`` is either ``None`` for deterministic greedy ordering or an
    exact permutation of the original matrix row indices.  At most
    ``max_states`` distinct states may be retained in any layer; exceeding
    the cap raises :class:`FrontierStateLimitExceeded`.  Counts use arbitrary
    size Python integers.  Infeasible matrices return zero before entering
    the capped recurrence.
    """
    max_states = _validate_frontier_max_states(max_states)
    normalized = _normalize_binary_matrix(matrix, strict=strict)
    normalized_order = _validate_frontier_row_order(
        row_order, len(normalized)
    )
    row_masks = _row_masks(normalized)
    if _perfect_matching(row_masks) is None:
        return 0
    if normalized_order is None:
        normalized_order, _, _ = _frontier_greedy_order(row_masks)
    return _frontier_permanent_masks(
        row_masks, normalized_order, max_states
    )


def _auto_frontier_plan(
    row_masks: Sequence[int],
) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Return an orientation and order for a conservative auto fast path."""
    order = len(row_masks)
    if order < _AUTO_FRONTIER_MIN_ORDER:
        return None
    total_entries = sum(bin(mask).count("1") for mask in row_masks)
    if total_entries > _AUTO_FRONTIER_MAX_AVERAGE_DEGREE * order:
        return None

    direct_order, direct_width, direct_area = _frontier_greedy_order(
        row_masks
    )
    transpose = _transpose_row_masks(row_masks)
    if transpose == tuple(row_masks):
        planned_masks = tuple(row_masks)
        planned_order = direct_order
        width = direct_width
    else:
        transpose_order, transpose_width, transpose_area = (
            _frontier_greedy_order(transpose)
        )
        direct_score = (direct_width, direct_area, 0, direct_order)
        transpose_score = (
            transpose_width,
            transpose_area,
            1,
            transpose_order,
        )
        if transpose_score < direct_score:
            planned_masks = transpose
            planned_order = transpose_order
            width = transpose_width
        else:
            planned_masks = tuple(row_masks)
            planned_order = direct_order
            width = direct_width

    if width > _AUTO_FRONTIER_MAX_WIDTH or 2 * width > order:
        return None
    return planned_masks, planned_order


def binary_permanent_bound(matrix, *, strict: bool = True) -> int:
    """Return a rigorous upper bound on a binary matrix's permanent.

    Matching-induced irreducible blocks are bounded independently.  Within
    each block, the minimum of ``n!``, the row- and column-degree products,
    and rigorously integer-rounded Bregman--Minc bounds is used.  Unlike a
    signed integer permanent, the result is nonnegative, so a modular
    reconstruction only needs a modulus *strictly* greater than this bound
    rather than greater than twice the bound.
    """
    normalized = _normalize_binary_matrix(matrix, strict=strict)
    feasible, blocks = _matching_blocks(_row_masks(normalized))
    if not feasible:
        return 0
    if not blocks:
        return 1
    return math.prod(_normalized_bound(_unpack_masks(block)) for block in blocks)


def _normalized_bound(matrix: Sequence[Sequence[int]]) -> int:
    """Bound helper for an already-normalized binary matrix."""
    n = len(matrix)
    if n == 0:
        return 1

    row_product = 1
    for row in matrix:
        row_product *= sum(row)

    column_product = 1
    for column in range(n):
        column_product *= sum(row[column] for row in matrix)

    factorial = math.factorial(n)
    row_degrees = [sum(row) for row in matrix]
    column_degrees = [sum(row[column] for row in matrix) for column in range(n)]
    return min(
        factorial,
        row_product,
        column_product,
        _grouped_bregman_minc_bound(row_degrees),
        _grouped_bregman_minc_bound(column_degrees),
    )


def _ceil_nth_root(value: int, degree: int) -> int:
    """Return ``ceil(value ** (1 / degree))`` using integer arithmetic."""
    if value < 0:
        raise ValueError("value must be non-negative")
    if degree <= 0:
        raise ValueError("degree must be positive")
    if value <= 1 or degree == 1:
        return value

    low = 0
    high = 1 << ((value.bit_length() + degree - 1) // degree)
    while low + 1 < high:
        middle = (low + high) // 2
        if pow(middle, degree) < value:
            low = middle
        else:
            high = middle
    return high


def _grouped_bregman_minc_bound(degrees: Sequence[int]) -> int:
    """Integer upper bound from Bregman--Minc, grouped by equal degrees.

    Bregman--Minc says ``per(A) <= product((r_i!) ** (1/r_i))``.
    For ``count`` rows of degree ``d``, its contribution is computed as an
    exact integer power times one rigorously rounded integer ``d``-th root.
    Grouping equal degrees avoids both floating-point rounding and enormous
    common-denominator powers.
    """
    counts = {}
    for degree in degrees:
        if degree == 0:
            return 0
        counts[degree] = counts.get(degree, 0) + 1

    bound = 1
    for degree, count in counts.items():
        factorial = math.factorial(degree)
        quotient, remainder = divmod(count, degree)
        bound *= pow(factorial, quotient)
        if remainder:
            bound *= _ceil_nth_root(pow(factorial, remainder), degree)
    return bound


def _perfect_matching(row_masks: Sequence[int]) -> Optional[Tuple[int, ...]]:
    """Find a row-to-column perfect matching by augmenting paths."""
    n = len(row_masks)
    match_row = [-1] * n
    match_column = [-1] * n

    for start_row in range(n):
        parent_column = [-1] * n
        seen_rows = [False] * n
        seen_columns = [False] * n
        seen_rows[start_row] = True
        queue = [start_row]
        head = 0
        free_column = -1

        while head < len(queue) and free_column < 0:
            row = queue[head]
            head += 1
            available = row_masks[row]
            while available:
                bit = available & -available
                column = bit.bit_length() - 1
                available ^= bit
                if seen_columns[column]:
                    continue
                seen_columns[column] = True
                parent_column[column] = row
                matched_row = match_column[column]
                if matched_row < 0:
                    free_column = column
                    break
                if not seen_rows[matched_row]:
                    seen_rows[matched_row] = True
                    queue.append(matched_row)

        if free_column < 0:
            return None

        column = free_column
        while column >= 0:
            row = parent_column[column]
            previous_column = match_row[row]
            match_row[row] = column
            match_column[column] = row
            column = previous_column

    return tuple(match_row)


def _strong_components(
    adjacency: Sequence[Sequence[int]],
) -> Tuple[Tuple[int, ...], ...]:
    """Iterative Kosaraju decomposition of a directed graph."""
    n = len(adjacency)
    reverse: List[List[int]] = [[] for _ in range(n)]
    for source, targets in enumerate(adjacency):
        for target in targets:
            reverse[target].append(source)

    seen = [False] * n
    finishing_order = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack = [(start, 0)]
        while stack:
            vertex, next_index = stack[-1]
            if next_index == len(adjacency[vertex]):
                stack.pop()
                finishing_order.append(vertex)
                continue
            # Discover one neighbor at a time, as recursive DFS would. Marking
            # all siblings up front can produce an invalid finishing order
            # when one sibling reaches another, merging independent blocks.
            target = adjacency[vertex][next_index]
            stack[-1] = (vertex, next_index + 1)
            if not seen[target]:
                seen[target] = True
                stack.append((target, 0))

    seen = [False] * n
    components = []
    for start in reversed(finishing_order):
        if seen[start]:
            continue
        seen[start] = True
        component = []
        stack = [start]
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for target in reverse[vertex]:
                if not seen[target]:
                    seen[target] = True
                    stack.append(target)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _matching_blocks(
    row_masks: Sequence[int],
) -> Tuple[bool, Tuple[Tuple[int, ...], ...]]:
    """Split a feasible matrix into matching-induced irreducible blocks.

    After permuting a found perfect matching onto the diagonal, an edge from
    row ``i`` to the matched column of row ``j`` gives a directed edge
    ``i -> j``.  Its strongly connected components are independent square
    blocks; edges between components cannot participate in a perfect matching.
    """
    matching = _perfect_matching(row_masks)
    if matching is None:
        return False, ()
    n = len(row_masks)
    if n == 0:
        return True, ()

    matched_column_to_row = [-1] * n
    for row, column in enumerate(matching):
        matched_column_to_row[column] = row

    adjacency = []
    for mask in row_masks:
        targets = []
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            column = bit.bit_length() - 1
            remaining ^= bit
            targets.append(matched_column_to_row[column])
        adjacency.append(tuple(targets))

    blocks = []
    for component in _strong_components(adjacency):
        columns = [matching[row] for row in component]
        block_rows = []
        for row in component:
            local_mask = 0
            for local_column, original_column in enumerate(columns):
                if row_masks[row] & (1 << original_column):
                    local_mask |= 1 << local_column
            block_rows.append(local_mask)
        blocks.append(tuple(block_rows))
    return True, tuple(blocks)


def _unpack_masks(row_masks: Sequence[int]) -> Tuple[Tuple[int, ...], ...]:
    """Convert packed rows to the normalized nested representation."""
    n = len(row_masks)
    return tuple(
        tuple((mask >> column) & 1 for column in range(n))
        for mask in row_masks
    )


def _is_prime(value: int) -> bool:
    """Deterministic Miller--Rabin primality test for unsigned 64-bit values."""
    if value < 2:
        return False
    for small_prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value == small_prime:
            return True
        if value % small_prime == 0:
            return False

    odd_part = value - 1
    power_of_two = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power_of_two += 1

    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        base %= value
        if base == 0:
            continue
        witness = pow(base, odd_part, value)
        if witness in (1, value - 1):
            continue
        for _ in range(power_of_two - 1):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def _crt_primes_above_bound(bound: int) -> Tuple[int, ...]:
    """Choose descending 31-bit primes with product strictly above a bound.

    The native Montgomery fast path has the same measured cost throughout
    ``p < 2**31``.  Starting at the Mersenne prime ``2**31 - 1`` therefore
    extracts one more CRT bit per exponential traversal than the former
    ascending 30-bit plan.
    """
    if bound < 0:
        raise ValueError("bound must be non-negative")
    primes = []
    product = 1
    candidate = (1 << 31) - 1
    while product <= bound:
        if _is_prime(candidate):
            primes.append(candidate)
            product *= candidate
        candidate -= 2
    return tuple(primes)


def _native_residue(
    matrix: Sequence[Sequence[int]], modulus: int, num_threads: int
) -> int:
    """Call the native modular hook and enforce its residue contract."""
    native_matrix = [list(row) for row in matrix]
    raw_residue = _binary_ext.binary_permanent_mod_p(
        native_matrix, modulus, num_threads
    )
    try:
        residue = operator.index(raw_residue)
    except TypeError as exc:
        raise RuntimeError(
            "native binary-permanent backend returned an invalid residue"
        ) from exc
    if isinstance(raw_residue, (bool, np.bool_)):
        raise RuntimeError(
            "native binary-permanent backend returned an invalid residue"
        )
    residue = int(residue)
    if not 0 <= residue < modulus:
        raise RuntimeError(
            "native binary-permanent backend returned an invalid residue"
        )
    return residue


def _native_exact_permanent(
    matrix: Sequence[Sequence[int]], bound: int, num_threads: int
) -> int:
    """Reconstruct a nonnegative exact permanent from native residues."""
    result = 0
    product = 1
    for prime in _crt_primes_above_bound(bound):
        residue = _native_residue(matrix, prime, num_threads)
        correction = (
            (residue - result) % prime
        ) * pow(product, -1, prime) % prime
        result += product * correction
        product *= prime

    # The planner guarantees uniqueness in [0, bound].  Treat a violation as a
    # backend error rather than returning an apparently exact corrupt answer.
    if result > bound:
        raise RuntimeError(
            "native binary-permanent residues reconstructed outside the bound"
        )
    return result


def get_binary_backend() -> str:
    """Return the backend selected by ``backend='auto'``."""
    return "native" if HAS_BINARY_EXT else "python"


def available_binary_backends() -> Tuple[str, ...]:
    """Return the backend names usable in this process."""
    if HAS_BINARY_EXT:
        return ("native", "python")
    return ("python",)


def _select_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    backend = backend.lower()
    if backend == "auto":
        return get_binary_backend()
    if backend == "python":
        return backend
    if backend == "native":
        if not HAS_BINARY_EXT:
            raise RuntimeError("the native binary-permanent backend is unavailable")
        return backend
    raise ValueError("backend must be one of: 'auto', 'native', 'python'")


def binary_permanent_mod_p(
    matrix,
    p: int,
    *,
    backend: str = "auto",
    parallel: bool = True,
    num_threads: int = 0,
    strict: bool = True,
) -> int:
    """Compute the permanent modulo an odd prime larger than the order.

    The prime restriction permits the native incremental-product kernel to
    invert every possible nonzero row sum.  Exact callers normally use
    :func:`binary_permanent`, which plans these residue primes automatically.
    """
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a boolean")
    num_threads = _validate_num_threads(num_threads)
    normalized = _normalize_binary_matrix(matrix, strict=strict)
    modulus = _validate_residue_prime(p, len(normalized))
    selected = _select_backend(backend)
    automatic = backend.lower() == "auto"

    if not normalized:
        return 1 % modulus
    if len(normalized) == 1:
        return normalized[0][0] % modulus

    feasible, blocks = _matching_blocks(_row_masks(normalized))
    if not feasible:
        return 0

    result = 1
    threads = num_threads if parallel else 1
    for block in blocks:
        if len(block) == 1:
            continue
        block_matrix = _unpack_masks(block)
        if selected == "native" and len(block) < 64:
            residue = _native_residue(block_matrix, modulus, threads)
        else:
            if selected == "native" and not automatic:
                raise ValueError(
                    "native binary-permanent blocks must have order less than 64"
                )
            residue = _permanent_dp(block, modulus)
        result = result * residue % modulus
    return result


def binary_permanent(
    matrix,
    *,
    backend: str = "auto",
    parallel: bool = True,
    num_threads: int = 0,
    strict: bool = True,
) -> int:
    """Compute the exact permanent of a square zero-one matrix.

    In strict mode every entry must be an integral zero or one.  With
    ``strict=False``, an integral matrix is converted to its support matrix.
    The result is always a nonnegative, arbitrary-size Python integer.
    """
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a boolean")
    num_threads = _validate_num_threads(num_threads)
    normalized = _normalize_binary_matrix(matrix, strict=strict)
    selected = _select_backend(backend)
    automatic = backend.lower() == "auto"

    if not normalized:
        return 1
    if len(normalized) == 1:
        return normalized[0][0]

    feasible, blocks = _matching_blocks(_row_masks(normalized))
    if not feasible:
        return 0

    result = 1
    threads = num_threads if parallel else 1
    for block in blocks:
        if len(block) == 1:
            continue
        if automatic:
            frontier_plan = _auto_frontier_plan(block)
            if frontier_plan is not None:
                frontier_masks, frontier_order = frontier_plan
                try:
                    block_result = _frontier_permanent_masks(
                        frontier_masks,
                        frontier_order,
                        _AUTO_FRONTIER_MAX_STATES,
                    )
                except FrontierStateLimitExceeded:
                    pass
                else:
                    result *= block_result
                    continue
        block_matrix = _unpack_masks(block)
        if selected == "native" and len(block) < 64:
            block_result = _native_exact_permanent(
                block_matrix, _normalized_bound(block_matrix), threads
            )
        else:
            if selected == "native" and not automatic:
                raise ValueError(
                    "native binary-permanent blocks must have order less than 64"
                )
            block_result = _permanent_dp(block)
        result *= block_result
    return result


__all__ = [
    "FrontierStateLimitExceeded",
    "HAS_BINARY_EXT",
    "available_binary_backends",
    "binary_permanent",
    "binary_permanent_bound",
    "binary_permanent_frontier",
    "binary_permanent_mod_p",
    "get_binary_backend",
]
