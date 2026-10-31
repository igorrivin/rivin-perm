"""Exact and modular permanents of affine polynomial matrices.

For integer matrices ``C`` and ``L``, :func:`affine_permanent` returns the
exact coefficients of ``per(C + x L)`` in ascending degree order.  It uses a
rigorous absolute coefficient bound and signed or unsigned Chinese-remainder
reconstruction over deterministic 31-bit primes.  :func:`affine_permanent_mod`
performs one modular traversal when an exact result is not required.

:func:`principal_permanent_polynomial` and
:func:`principal_permanent_polynomial_mod` specialize the identity-constant
case ``per(I + x A)``.  Its degree-``k`` coefficient is the sum of the
permanents of all ``k``-by-``k`` principal submatrices of ``A``.

For a zero-one matrix ``A``, :func:`rook_polynomial` returns the exact
matching counts ``r_k`` in ascending cardinality order.  It obtains them from
the coefficients of ``per(J + x A)``, where ``J`` is the all-ones matrix.
A fixed-point Bregman--Minc bound and an independent characteristic-two
determinant identify the exact lift of every odd-modulus CRT residue.

For a nonnegative integer matrix ``W``,
:func:`weighted_rook_polynomial` returns the corresponding weighted matching
sums.  It uses the same affine-permanent identity, coefficientwise bounds,
and characteristic-two lift certificate while preserving arbitrary-size
Python weights.

The portable kernel applies Glynn's formula with one sign fixed and carries a
coefficient vector through each product of affine row sums.

Only odd positive signed-64-bit moduli are accepted because Glynn's
power-of-two normalization must be invertible and the public contract must
also be usable by the optional native backend.  Primality is not required.
"""

from __future__ import annotations

import operator
from functools import lru_cache
from math import factorial
from typing import Sequence


_INT64_MAX = (1 << 63) - 1
_INT32_MAX = (1 << 31) - 1
_CRT_PRIME_START = (1 << 31) - 1
_ROOK_BOUND_SCALE = 1 << 32
_ROOK_NATIVE_MAX_MODULI = 16
_WEIGHTED_ROOK_NATIVE_MAX_ENTRY = (1 << 16) - 1
_UINT32_LIMIT = 1 << 32


try:
    import perm_polynomial_ext as _polynomial_ext
except ImportError:
    _polynomial_ext = None


HAS_POLYNOMIAL_EXT = (
    _polynomial_ext is not None
    and hasattr(_polynomial_ext, "affine_permanent_mod")
)


_POLYNOMIAL_NATIVE_HOOK_NAMES = (
    "affine_permanent_mod",
    "rook_polynomial_mod_many",
    "weighted_rook_polynomial_mod_many",
)


def polynomial_backend_capabilities() -> dict[str, object]:
    """Return the available polynomial backends and native accelerator hooks.

    The returned mapping is a fresh snapshot with the stable keys
    ``available_backends``, ``default_backend``,
    ``native_extension_available``, and ``native_hooks``.  Hook names describe
    optional implementation details that callers such as benchmark drivers can
    test without inspecting the private extension module.  The portable
    ``python`` backend is always available.
    """
    native_available = bool(HAS_POLYNOMIAL_EXT)
    extension = _polynomial_ext if native_available else None
    native_hooks = tuple(
        name
        for name in _POLYNOMIAL_NATIVE_HOOK_NAMES
        if extension is not None and hasattr(extension, name)
    )
    available_backends = (
        ("native", "python") if native_available else ("python",)
    )
    return {
        "available_backends": available_backends,
        "default_backend": "native" if native_available else "python",
        "native_extension_available": native_available,
        "native_hooks": native_hooks,
    }


def _validate_modulus(modulus: int) -> int:
    try:
        value = operator.index(modulus)
    except TypeError as exc:
        raise TypeError("modulus must be an integer") from exc
    if isinstance(modulus, bool):
        raise TypeError("modulus must be an integer")
    value = int(value)
    if value <= 1 or value % 2 == 0:
        raise ValueError("modulus must be an odd integer greater than 1")
    if value > _INT64_MAX:
        raise ValueError(
            "modulus must fit in a positive signed 64-bit integer"
        )
    return value


def _validate_num_threads(num_threads: int) -> int:
    try:
        value = operator.index(num_threads)
    except TypeError as exc:
        raise TypeError("num_threads must be an integer") from exc
    if isinstance(num_threads, bool):
        raise TypeError("num_threads must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError("num_threads must be non-negative")
    if value > _INT32_MAX:
        raise ValueError("num_threads is too large")
    return value


def _snapshot_integer_matrix(
    matrix, name: str
) -> tuple[tuple[int, ...], ...]:
    """Validate and snapshot a square matrix of arbitrary Python integers."""
    declared_shape = getattr(matrix, "shape", None)
    if declared_shape is not None:
        try:
            declared_shape = tuple(declared_shape)
        except TypeError as exc:
            raise ValueError(
                f"{name} must have a two-dimensional square shape"
            ) from exc
        if len(declared_shape) != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if declared_shape[0] != declared_shape[1]:
            raise ValueError(f"{name} must be square")

    try:
        order = len(matrix)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of rows") from exc
    if order >= 64:
        raise ValueError("matrix order must be less than 64")
    if declared_shape is not None and declared_shape != (order, order):
        raise ValueError(f"{name} shape changed during validation")

    normalized = []
    for row_index, row in enumerate(matrix):
        try:
            width = len(row)
        except TypeError as exc:
            raise TypeError(f"{name} rows must be sequences") from exc
        if width != order:
            raise ValueError(f"{name} must be square")

        normalized_row = []
        for column_index, entry in enumerate(row):
            try:
                value = operator.index(entry)
            except TypeError as exc:
                raise TypeError(
                    f"{name} entries must be integers; entry "
                    f"({row_index}, {column_index}) is "
                    f"{type(entry).__name__}"
                ) from exc
            normalized_row.append(int(value))
        if len(normalized_row) != order:
            raise ValueError(f"{name} row width changed during validation")
        normalized.append(tuple(normalized_row))

    if len(normalized) != order:
        raise ValueError(f"{name} row count changed during validation")
    return tuple(normalized)


def _snapshot_binary_matrix(matrix) -> tuple[tuple[int, ...], ...]:
    """Validate and snapshot a square integral zero-one matrix."""
    normalized = _snapshot_integer_matrix(matrix, "matrix")
    for row_index, row in enumerate(normalized):
        for column_index, value in enumerate(row):
            if value not in (0, 1):
                raise ValueError(
                    "matrix entries must be 0 or 1; entry "
                    f"({row_index}, {column_index}) is {value}"
                )
    return normalized


def _snapshot_nonnegative_integer_matrix(
    matrix,
) -> tuple[tuple[int, ...], ...]:
    """Validate and snapshot a square nonnegative integral matrix."""
    normalized = _snapshot_integer_matrix(matrix, "matrix")
    for row_index, row in enumerate(normalized):
        for column_index, value in enumerate(row):
            if value < 0:
                raise ValueError(
                    "matrix entries must be nonnegative; entry "
                    f"({row_index}, {column_index}) is {value}"
                )
    return normalized


def _validate_affine_matrices(
    constant_matrix, linear_matrix
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    constant = _snapshot_integer_matrix(constant_matrix, "constant_matrix")
    linear = _snapshot_integer_matrix(linear_matrix, "linear_matrix")
    if len(constant) != len(linear):
        raise ValueError(
            "constant_matrix and linear_matrix must have the same shape"
        )
    return constant, linear


def _reduce_matrix(
    matrix: Sequence[Sequence[int]], modulus: int
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(entry % modulus for entry in row) for row in matrix
    )


def _select_backend(backend: str) -> str:
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    selected = backend.lower()
    if selected == "auto":
        return "native" if HAS_POLYNOMIAL_EXT else "python"
    if selected == "python":
        return selected
    if selected == "native":
        if not HAS_POLYNOMIAL_EXT:
            raise RuntimeError(
                "the native polynomial-permanent backend is unavailable"
            )
        return selected
    raise ValueError("backend must be one of: 'auto', 'native', 'python'")


def _multiply_affine_factors(
    constants: Sequence[int], linears: Sequence[int], modulus: int
) -> list[int]:
    """Return ``product_i(constants[i] + x*linears[i])`` modulo m."""
    product = [1]
    for constant, linear in zip(constants, linears):
        updated = [0] * (len(product) + 1)
        for degree, coefficient in enumerate(product):
            updated[degree] = (
                updated[degree] + coefficient * constant
            ) % modulus
            updated[degree + 1] = (
                updated[degree + 1] + coefficient * linear
            ) % modulus
        product = updated
    return product


def _affine_permanent_glynn(
    constant_matrix: Sequence[Sequence[int]],
    linear_matrix: Sequence[Sequence[int]],
    modulus: int,
) -> tuple[int, ...]:
    """Coefficient-valued fixed-sign Glynn formula."""
    order = len(constant_matrix)
    if order == 0:
        return (1 % modulus,)
    if order == 1:
        return (constant_matrix[0][0], linear_matrix[0][0])

    constant_sums = [sum(row) % modulus for row in constant_matrix]
    linear_sums = [sum(row) % modulus for row in linear_matrix]
    coefficients = [0] * (order + 1)
    state_sign = 1

    for state in range(1 << (order - 1)):
        if state:
            column = (state & -state).bit_length() - 1
            gray = state ^ (state >> 1)
            subtracting = (gray >> column) & 1
            for row in range(order):
                constant_delta = 2 * constant_matrix[row][column]
                linear_delta = 2 * linear_matrix[row][column]
                if subtracting:
                    constant_sums[row] = (
                        constant_sums[row] - constant_delta
                    ) % modulus
                    linear_sums[row] = (
                        linear_sums[row] - linear_delta
                    ) % modulus
                else:
                    constant_sums[row] = (
                        constant_sums[row] + constant_delta
                    ) % modulus
                    linear_sums[row] = (
                        linear_sums[row] + linear_delta
                    ) % modulus
            state_sign = -state_sign

        product = _multiply_affine_factors(
            constant_sums, linear_sums, modulus
        )
        if state_sign > 0:
            for degree, value in enumerate(product):
                coefficients[degree] = (
                    coefficients[degree] + value
                ) % modulus
        else:
            for degree, value in enumerate(product):
                coefficients[degree] = (
                    coefficients[degree] - value
                ) % modulus

    inverse_normalization = pow(
        pow(2, order - 1, modulus), -1, modulus
    )
    return tuple(
        coefficient * inverse_normalization % modulus
        for coefficient in coefficients
    )


def _is_identity_matrix(matrix: Sequence[Sequence[int]]) -> bool:
    """Return whether ``matrix`` is the exact square identity matrix."""
    return all(
        entry == int(row_index == column_index)
        for row_index, row in enumerate(matrix)
        for column_index, entry in enumerate(row)
    )


def _principal_permanent_glynn(
    matrix: Sequence[Sequence[int]], modulus: int
) -> tuple[int, ...]:
    """Coefficients of ``per(I + x*matrix)`` by direct sign projection.

    In Glynn's formula the product of the constant row factors for ``I``
    equals the outer sign product, so those signs cancel.  If

    ``factor_i = epsilon_i * (matrix * epsilon)_i``, then one fixed-sign
    half-cube traversal can accumulate every elementary-symmetric vector
    directly.  This avoids the general multiplication of affine factors.
    """
    order = len(matrix)
    if order == 0:
        return (1 % modulus,)

    # epsilon_0 is fixed positive; Gray bits 0..order-2 control columns
    # 1..order-1.  Keeping signed Python sums avoids a modulus operation on
    # every row update; all uses below are reduced before multiplication.
    signed_sums = [sum(row) for row in matrix]
    coefficients = [0] * (order + 1)
    gray = 0
    state_count = 1 << (order - 1)

    for state in range(state_count):
        elementary = [0] * (order + 1)
        elementary[0] = 1
        for row in range(order):
            negative = row > 0 and ((gray >> (row - 1)) & 1)
            factor = (-signed_sums[row] if negative else signed_sums[row])
            factor %= modulus
            for degree in range(row + 1, 0, -1):
                elementary[degree] = (
                    elementary[degree]
                    + factor * elementary[degree - 1]
                ) % modulus
        for degree, value in enumerate(elementary):
            coefficients[degree] = (
                coefficients[degree] + value
            ) % modulus

        if state + 1 == state_count:
            break
        transition = state + 1
        bit = (transition & -transition).bit_length() - 1
        column = bit + 1
        old_negative = (gray >> bit) & 1
        direction = 2 if old_negative else -2
        for row in range(order):
            signed_sums[row] += direction * matrix[row][column]
        gray ^= 1 << bit

    inverse_normalization = pow(
        pow(2, order - 1, modulus), -1, modulus
    )
    return tuple(
        coefficient * inverse_normalization % modulus
        for coefficient in coefficients
    )


def _native_affine_permanent(
    constant_matrix: Sequence[Sequence[int]],
    linear_matrix: Sequence[Sequence[int]],
    modulus: int,
    num_threads: int,
) -> tuple[int, ...]:
    native_constant = [list(row) for row in constant_matrix]
    native_linear = [list(row) for row in linear_matrix]
    raw = _polynomial_ext.affine_permanent_mod(
        native_constant, native_linear, modulus, num_threads
    )
    try:
        raw_values = tuple(raw)
    except TypeError as exc:
        raise RuntimeError(
            "native polynomial-permanent backend returned invalid coefficients"
        ) from exc
    if len(raw_values) != len(constant_matrix) + 1:
        raise RuntimeError(
            "native polynomial-permanent backend returned invalid coefficients"
        )

    coefficients = []
    for raw_value in raw_values:
        try:
            value = operator.index(raw_value)
        except TypeError as exc:
            raise RuntimeError(
                "native polynomial-permanent backend returned invalid coefficients"
            ) from exc
        if isinstance(raw_value, bool):
            raise RuntimeError(
                "native polynomial-permanent backend returned invalid coefficients"
            )
        value = int(value)
        if not 0 <= value < modulus:
            raise RuntimeError(
                "native polynomial-permanent backend returned invalid coefficients"
            )
        coefficients.append(value)
    return tuple(coefficients)


def _native_rook_polynomial_many(
    matrix: Sequence[Sequence[int]],
    moduli: Sequence[int],
    num_threads: int,
) -> tuple[tuple[int, ...], ...]:
    """Call and strictly validate the fused native rook kernel."""
    native_matrix = [list(row) for row in matrix]
    raw = _polynomial_ext.rook_polynomial_mod_many(
        native_matrix, tuple(moduli), num_threads
    )
    try:
        raw_rows = tuple(raw)
    except TypeError as exc:
        raise RuntimeError(
            "native polynomial-permanent backend returned invalid "
            "rook residues"
        ) from exc
    if len(raw_rows) != len(moduli):
        raise RuntimeError(
            "native polynomial-permanent backend returned invalid "
            "rook residues"
        )

    result = []
    expected_length = len(matrix) + 1
    for raw_row, modulus in zip(raw_rows, moduli):
        try:
            raw_values = tuple(raw_row)
        except TypeError as exc:
            raise RuntimeError(
                "native polynomial-permanent backend returned invalid "
                "rook residues"
            ) from exc
        if len(raw_values) != expected_length:
            raise RuntimeError(
                "native polynomial-permanent backend returned invalid "
                "rook residues"
            )

        residues = []
        for raw_value in raw_values:
            try:
                value = operator.index(raw_value)
            except TypeError as exc:
                raise RuntimeError(
                    "native polynomial-permanent backend returned invalid "
                    "rook residues"
                ) from exc
            if isinstance(raw_value, bool):
                raise RuntimeError(
                    "native polynomial-permanent backend returned invalid "
                    "rook residues"
                )
            value = int(value)
            if not 0 <= value < modulus:
                raise RuntimeError(
                    "native polynomial-permanent backend returned invalid "
                    "rook residues"
                )
            residues.append(value)
        result.append(tuple(residues))
    return tuple(result)


def _native_weighted_rook_polynomial_many(
    matrix: Sequence[Sequence[int]],
    moduli: Sequence[int],
    num_threads: int,
) -> tuple[tuple[int, ...], ...]:
    """Call and strictly validate the fused native weighted-rook kernel."""
    native_matrix = [list(row) for row in matrix]
    raw = _polynomial_ext.weighted_rook_polynomial_mod_many(
        native_matrix, tuple(moduli), num_threads
    )
    try:
        raw_rows = tuple(raw)
    except TypeError as exc:
        raise RuntimeError(
            "native polynomial-permanent backend returned invalid "
            "weighted-rook residues"
        ) from exc
    if len(raw_rows) != len(moduli):
        raise RuntimeError(
            "native polynomial-permanent backend returned invalid "
            "weighted-rook residues"
        )

    result = []
    expected_length = len(matrix) + 1
    for raw_row, modulus in zip(raw_rows, moduli):
        try:
            raw_values = tuple(raw_row)
        except TypeError as exc:
            raise RuntimeError(
                "native polynomial-permanent backend returned invalid "
                "weighted-rook residues"
            ) from exc
        if len(raw_values) != expected_length:
            raise RuntimeError(
                "native polynomial-permanent backend returned invalid "
                "weighted-rook residues"
            )

        residues = []
        for raw_value in raw_values:
            try:
                value = operator.index(raw_value)
            except TypeError as exc:
                raise RuntimeError(
                    "native polynomial-permanent backend returned invalid "
                    "weighted-rook residues"
                ) from exc
            if isinstance(raw_value, bool):
                raise RuntimeError(
                    "native polynomial-permanent backend returned invalid "
                    "weighted-rook residues"
                )
            value = int(value)
            if not 0 <= value < modulus:
                raise RuntimeError(
                    "native polynomial-permanent backend returned invalid "
                    "weighted-rook residues"
                )
            residues.append(value)
        result.append(tuple(residues))
    return tuple(result)


def _nonnegative_affine_product(
    constants: Sequence[int], linears: Sequence[int]
) -> tuple[int, ...]:
    """Multiply affine factors whose coefficients are nonnegative integers."""
    product = [1]
    for constant, linear in zip(constants, linears):
        updated = [0] * (len(product) + 1)
        for degree, coefficient in enumerate(product):
            updated[degree] += coefficient * constant
            updated[degree + 1] += coefficient * linear
        product = updated
    return tuple(product)


@lru_cache(maxsize=128)
def _ceil_scaled_factorial_root(degree: int, scale: int) -> int:
    """Return ``ceil(scale * (degree!)**(1/degree))`` exactly."""
    if degree == 0:
        return 0

    target = factorial(degree) * pow(scale, degree)
    lower = 0
    # ``degree! <= degree**degree`` makes this a rigorous upper end.
    upper = degree * scale
    while lower + 1 < upper:
        midpoint = (lower + upper) // 2
        if pow(midpoint, degree) >= target:
            upper = midpoint
        else:
            lower = midpoint
    return upper


@lru_cache(maxsize=4096)
def _ceil_scaled_dummy_factor(
    order: int, omitted_rows: int, scale: int
) -> int:
    """Round up ``scale*(order!)**(q/order)/q!`` using integers."""
    order_factorial = factorial(order)
    omitted_factorial = factorial(omitted_rows)
    target = pow(scale, order) * pow(order_factorial, omitted_rows)

    def is_upper_bound(value: int) -> bool:
        return (
            pow(value * omitted_factorial, order) >= target
        )

    lower = 0
    upper = scale
    while not is_upper_bound(upper):
        upper *= 2
    while lower + 1 < upper:
        midpoint = (lower + upper) // 2
        if is_upper_bound(midpoint):
            upper = midpoint
        else:
            lower = midpoint
    return upper


def _elementary_symmetric(values: Sequence[int]) -> tuple[int, ...]:
    """Return all elementary symmetric polynomials of ``values``."""
    coefficients = [1] + [0] * len(values)
    for count, value in enumerate(values, start=1):
        for degree in range(count, 0, -1):
            coefficients[degree] += coefficients[degree - 1] * value
    return tuple(coefficients)


def _rook_polynomial_bounds_validated(
    matrix: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Return rigorous coefficientwise bounds for partial matchings.

    Fix a set ``S`` of ``k`` rows and append ``n-k`` all-one rows.  Every
    matching of ``S`` extends to exactly ``(n-k)!`` permutations of this
    augmented matrix.  Bregman--Minc bounds its permanent by the product of
    ``(d_i!)**(1/d_i)`` over selected rows and ``(n!)**(1/n)`` for every
    appended row.  The roots are rounded upward only after scaling by
    ``2**32``.  Multiplying the resulting fixed-point integers and retaining
    the common denominator avoids the substantial loss from independently
    rounding every factor to a whole integer.  Transposition gives the
    independent column bound.

    The returned values are floors of rigorous rational upper bounds.  This
    remains valid because rook coefficients are integers.
    """
    order = len(matrix)
    if order == 0:
        return (1,)

    scale = _ROOK_BOUND_SCALE
    roots = tuple(
        _ceil_scaled_factorial_root(degree, scale)
        for degree in range(order + 1)
    )
    row_weights = [roots[sum(row)] for row in matrix]
    column_weights = [
        roots[sum(matrix[row][column] for row in range(order))]
        for column in range(order)
    ]
    row_symmetric = _elementary_symmetric(row_weights)
    column_symmetric = _elementary_symmetric(column_weights)

    bounds = []
    scale_power = scale
    for size in range(order + 1):
        omitted_rows = order - size
        dummy_factor = _ceil_scaled_dummy_factor(
            order, omitted_rows, scale
        )
        numerator = min(
            row_symmetric[size], column_symmetric[size]
        ) * dummy_factor
        bounds.append(numerator // scale_power)
        scale_power *= scale
    bounds[0] = 1
    return tuple(bounds)


def _bit_parity(value: int) -> int:
    """Return the parity of a nonnegative integer, including on Python 3.8."""
    return bin(value).count("1") & 1


def _rook_polynomial_parities_validated(
    matrix: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Return all rook coefficients modulo two without a Glynn traversal.

    Over characteristic two, Cauchy--Binet gives

    ``det(I + x*A*A.T) = sum_k r_k*x**k``.

    The characteristic polynomial of ``A*A.T`` is computed with the
    division-free Berkowitz recurrence.  Rows and intermediate vectors are
    bit-packed, so the complete certificate costs polynomial time and works
    for every supported order, including orders larger than the base field.
    """
    order = len(matrix)
    if order == 0:
        return (1,)

    matrix_rows = []
    for row in matrix:
        mask = 0
        for column, value in enumerate(row):
            if value:
                mask |= 1 << column
        matrix_rows.append(mask)

    gram_rows = []
    for left in matrix_rows:
        mask = 0
        for column, right in enumerate(matrix_rows):
            if _bit_parity(left & right):
                mask |= 1 << column
        gram_rows.append(mask)

    # Coefficients of the characteristic polynomial of the trailing
    # principal submatrix, in descending degree order.  In characteristic
    # two these are also the ascending coefficients of det(I + x*A*A.T).
    coefficients = [1]
    full_mask = (1 << order) - 1
    for start in range(order - 1, -1, -1):
        trailing_mask = full_mask ^ ((1 << (start + 1)) - 1)
        off_diagonal = gram_rows[start] & trailing_mask
        terms = [1, (gram_rows[start] >> start) & 1]

        vector = off_diagonal
        trailing_order = len(coefficients) - 1
        for power in range(trailing_order):
            terms.append(_bit_parity(off_diagonal & vector))
            if power + 1 == trailing_order:
                continue
            following = 0
            for row in range(start + 1, order):
                if _bit_parity(gram_rows[row] & vector):
                    following |= 1 << row
            vector = following

        updated = [0] * (len(coefficients) + 1)
        for coefficient_index, coefficient in enumerate(coefficients):
            if not coefficient:
                continue
            available = len(updated) - coefficient_index
            for term_index, term in enumerate(terms[:available]):
                updated[coefficient_index + term_index] ^= term
        coefficients = updated

    return tuple(coefficients)


def _weighted_rook_polynomial_parities_validated(
    matrix: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Return weighted matching coefficients modulo two.

    Reducing each weight modulo two turns a weighted matching product into
    its parity without changing the matching constraints.  The resulting
    binary instance is then handled by the independent characteristic-two
    determinant certificate used for ordinary rook polynomials.
    """
    parity_matrix = tuple(
        tuple(value & 1 for value in row) for row in matrix
    )
    return _rook_polynomial_parities_validated(parity_matrix)


def _affine_permanent_bounds_validated(
    constant_matrix: Sequence[Sequence[int]],
    linear_matrix: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Return rigorous absolute bounds for every permanent coefficient.

    A row-sum relaxation drops the requirement that selected columns be
    distinct, so the coefficient vector of

        product_i (sum_j |C_ij| + x sum_j |L_ij|)

    bounds the permanent coefficientwise.  The analogous column relaxation
    is independent.  Two further bounds replace each entry by its row or
    column maximum and account for the ``n!`` permutations explicitly.  The
    coefficientwise minimum of these four bounds is therefore still valid.
    """
    order = len(constant_matrix)
    if order == 0:
        return (1,)

    row_constant_sums = [
        sum(abs(entry) for entry in row) for row in constant_matrix
    ]
    row_linear_sums = [
        sum(abs(entry) for entry in row) for row in linear_matrix
    ]
    column_constant_sums = [
        sum(abs(constant_matrix[row][column]) for row in range(order))
        for column in range(order)
    ]
    column_linear_sums = [
        sum(abs(linear_matrix[row][column]) for row in range(order))
        for column in range(order)
    ]

    row_constant_maxima = [
        max(abs(entry) for entry in row) for row in constant_matrix
    ]
    row_linear_maxima = [
        max(abs(entry) for entry in row) for row in linear_matrix
    ]
    column_constant_maxima = [
        max(abs(constant_matrix[row][column]) for row in range(order))
        for column in range(order)
    ]
    column_linear_maxima = [
        max(abs(linear_matrix[row][column]) for row in range(order))
        for column in range(order)
    ]

    row_sum_bound = _nonnegative_affine_product(
        row_constant_sums, row_linear_sums
    )
    column_sum_bound = _nonnegative_affine_product(
        column_constant_sums, column_linear_sums
    )
    permutation_count = factorial(order)
    row_max_bound = tuple(
        permutation_count * value
        for value in _nonnegative_affine_product(
            row_constant_maxima, row_linear_maxima
        )
    )
    column_max_bound = tuple(
        permutation_count * value
        for value in _nonnegative_affine_product(
            column_constant_maxima, column_linear_maxima
        )
    )
    return tuple(
        min(candidates)
        for candidates in zip(
            row_sum_bound,
            column_sum_bound,
            row_max_bound,
            column_max_bound,
        )
    )


def _weighted_rook_polynomial_bounds_validated(
    matrix: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Return rigorous bounds for nonnegative weighted matchings.

    If ``J`` is all ones, the degree-``k`` coefficient of
    ``per(J + x*matrix)`` is exactly ``(n-k)!`` times the weighted matching
    coefficient.  Dividing a nonnegative affine-permanent upper bound by
    this positive integer and taking the floor therefore remains rigorous.
    """
    order = len(matrix)
    if order == 0:
        return (1,)
    all_ones = ((1,) * order,) * order
    affine_bounds = _affine_permanent_bounds_validated(all_ones, matrix)
    return tuple(
        bound // factorial(order - degree)
        for degree, bound in enumerate(affine_bounds)
    )


def affine_permanent_bounds(
    constant_matrix, linear_matrix
) -> tuple[int, ...]:
    """Return certified absolute bounds for ``per(C + x*L)`` coefficients.

    The tuple is in ascending degree order and has ``order + 1`` entries.
    Bounds are exact nonnegative Python integers and are inexpensive to
    compute (``O(n**2)`` arithmetic operations after reading the matrices).
    """
    constant, linear = _validate_affine_matrices(
        constant_matrix, linear_matrix
    )
    return _affine_permanent_bounds_validated(constant, linear)


def _is_prime_32(value: int) -> bool:
    """Deterministic Miller--Rabin primality test for a 32-bit integer."""
    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if value % prime == 0:
            return value == prime

    odd_part = value - 1
    power_of_two = 0
    while odd_part % 2 == 0:
        power_of_two += 1
        odd_part //= 2

    # These bases are deterministic far beyond the 31-bit planner range.
    for base in (2, 3, 5, 7, 11, 13, 17):
        if base >= value:
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


def _plan_crt_primes(bound: int, *, signed: bool) -> tuple[int, ...]:
    """Choose descending 31-bit primes giving strict CRT coverage."""
    if bound < 0:
        raise ValueError("CRT bound must be nonnegative")
    target = 2 * bound if signed else bound
    product = 1
    primes = []
    candidate = _CRT_PRIME_START
    while product <= target:
        while candidate >= 3 and not _is_prime_32(candidate):
            candidate -= 2
        if candidate < 3:
            raise RuntimeError("exhausted the deterministic CRT prime range")
        primes.append(candidate)
        product *= candidate
        candidate -= 2
    return tuple(primes)


def _plan_rook_pseudo_mersenne_primes(
    target: int, *, order: int, maximum_degree: int
) -> tuple[int, ...]:
    """Plan a common-width prime set for the fused 32-bit rook kernel.

    The hot update forms ``old + p*abs(E)`` in an unsigned 32-bit lane.
    Choosing the largest common width that is safe even for ``p=2**b-1``
    makes every selected prime safe.  Each candidate also satisfies the
    one-fold pseudo-Mersenne reduction condition checked by the native code.
    """
    if target < 0:
        raise ValueError("CRT bound must be nonnegative")
    if not 0 <= maximum_degree <= order < 64:
        raise ValueError("invalid rook pseudo-Mersenne planner dimensions")

    modulus_bits = 31
    while (
        (maximum_degree + 1) * ((1 << modulus_bits) - 1)
        > _UINT32_LIMIT
    ):
        modulus_bits -= 1
    power_of_two = 1 << modulus_bits

    product = 1
    primes = []
    complement = 1
    while product <= target:
        if complement >= power_of_two // 2:
            raise RuntimeError(
                "exhausted the deterministic pseudo-Mersenne prime range"
            )
        modulus = power_of_two - complement
        raw_safe = (
            (maximum_degree + 1) * modulus <= _UINT32_LIMIT
        )
        fold_safe = (
            (maximum_degree + 2) * complement < power_of_two
        )
        if (
            modulus > order
            and raw_safe
            and fold_safe
            and _is_prime_32(modulus)
        ):
            primes.append(modulus)
            product *= modulus
        complement += 2
    return tuple(primes)


def _plan_weighted_rook_pseudo_mersenne_primes(
    target: int, *, order: int, maximum_signed_sum: int
) -> tuple[int, ...]:
    """Plan primes safe for the fused weighted-rook native kernel.

    The native recurrence has the same pseudo-Mersenne reduction as the
    binary fused kernel, but its signed row sums are bounded by a weighted
    line sum rather than by a zero-one degree.  All selected primes share the
    largest bit width whose raw negative update is safe in 32 bits.
    """
    if target < 0:
        raise ValueError("CRT bound must be nonnegative")
    if not 0 <= order < 64 or maximum_signed_sum < 0:
        raise ValueError("invalid weighted-rook planner dimensions")

    modulus_bits = 31
    while (
        modulus_bits > 1
        and (maximum_signed_sum + 1) * ((1 << modulus_bits) - 1)
        > _UINT32_LIMIT
    ):
        modulus_bits -= 1
    if (
        (maximum_signed_sum + 1) * ((1 << modulus_bits) - 1)
        > _UINT32_LIMIT
    ):
        raise RuntimeError(
            "no native weighted-rook modulus is safe for this matrix"
        )

    power_of_two = 1 << modulus_bits
    product = 1
    primes = []
    complement = 1
    while product <= target:
        # Complements increase monotonically. Once the one-fold reduction
        # bound fails, every later candidate fails too, even if its raw
        # signed update or primality would otherwise be admissible. Stop
        # at the first impossible candidate without changing any accepted
        # prime or the eventual fallback decision.
        if (
            complement >= power_of_two // 2
            or (maximum_signed_sum + 2) * complement >= power_of_two
        ):
            raise RuntimeError(
                "exhausted the deterministic weighted-rook prime range"
            )
        modulus = power_of_two - complement
        raw_safe = (
            (maximum_signed_sum + 1) * modulus <= _UINT32_LIMIT
        )
        fold_safe = (
            (maximum_signed_sum + 2) * complement < power_of_two
        )
        if (
            modulus > order
            and raw_safe
            and fold_safe
            and _is_prime_32(modulus)
        ):
            primes.append(modulus)
            product *= modulus
        complement += 2
    return tuple(primes)


def _affine_permanent_mod_validated(
    constant_matrix: Sequence[Sequence[int]],
    linear_matrix: Sequence[Sequence[int]],
    modulus: int,
    selected_backend: str,
    num_threads: int,
) -> tuple[int, ...]:
    constant = _reduce_matrix(constant_matrix, modulus)
    linear = _reduce_matrix(linear_matrix, modulus)
    if not constant:
        return (1 % modulus,)
    if len(constant) == 1:
        return (constant[0][0], linear[0][0])
    if selected_backend == "native":
        return _native_affine_permanent(
            constant, linear, modulus, num_threads
        )
    if _is_identity_matrix(constant):
        return _principal_permanent_glynn(linear, modulus)
    return _affine_permanent_glynn(constant, linear, modulus)


def affine_permanent_mod(
    constant_matrix,
    linear_matrix,
    modulus: int,
    *,
    backend: str = "auto",
    parallel: bool = True,
    num_threads: int = 0,
) -> tuple[int, ...]:
    """Return coefficients of ``per(constant_matrix + x*linear_matrix)``.

    ``modulus`` must be odd, greater than one, and no larger than the maximum
    signed 64-bit integer; it need not be prime.  Coefficients are canonical
    residues in ascending degree order.  The returned tuple always has
    ``order + 1`` entries, including any trailing zero coefficients.
    """
    modulus = _validate_modulus(modulus)
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a boolean")
    num_threads = _validate_num_threads(num_threads)
    selected = _select_backend(backend)

    constant, linear = _validate_affine_matrices(
        constant_matrix, linear_matrix
    )
    threads = num_threads if parallel else 1
    return _affine_permanent_mod_validated(
        constant, linear, modulus, selected, threads
    )


def _affine_permanent_validated(
    constant: Sequence[Sequence[int]],
    linear: Sequence[Sequence[int]],
    selected_backend: str,
    num_threads: int,
) -> tuple[int, ...]:
    """Exact affine permanent after matrix and option validation."""
    order = len(constant)
    if order == 0:
        return (1,)
    if order == 1:
        return (constant[0][0], linear[0][0])

    bounds = _affine_permanent_bounds_validated(constant, linear)
    maximum_bound = max(bounds)
    if maximum_bound == 0:
        return (0,) * (order + 1)

    nonnegative = all(
        entry >= 0
        for matrix in (constant, linear)
        for row in matrix
        for entry in row
    )
    primes = _plan_crt_primes(maximum_bound, signed=not nonnegative)

    reconstructed = [0] * (order + 1)
    reconstruction_modulus = 1
    for prime in primes:
        residues = _affine_permanent_mod_validated(
            constant, linear, prime, selected_backend, num_threads
        )
        if reconstruction_modulus == 1:
            reconstructed[:] = residues
        else:
            inverse = pow(reconstruction_modulus % prime, -1, prime)
            for degree, residue in enumerate(residues):
                correction = (
                    (residue - reconstructed[degree]) * inverse
                ) % prime
                reconstructed[degree] += (
                    reconstruction_modulus * correction
                )
        reconstruction_modulus *= prime

    if nonnegative:
        exact = tuple(reconstructed)
        valid = all(
            0 <= value <= bound for value, bound in zip(exact, bounds)
        )
    else:
        midpoint = reconstruction_modulus // 2
        exact = tuple(
            value - reconstruction_modulus if value > midpoint else value
            for value in reconstructed
        )
        valid = all(
            abs(value) <= bound for value, bound in zip(exact, bounds)
        )
    if not valid:
        raise RuntimeError(
            "CRT reconstruction lies outside its certified coefficient bound"
        )
    return exact


def affine_permanent(
    constant_matrix,
    linear_matrix,
    *,
    backend: str = "auto",
    parallel: bool = True,
    num_threads: int = 0,
) -> tuple[int, ...]:
    """Return the exact integer coefficients of ``per(C + x*L)``.

    Inputs must be equally sized square integer matrices of order less than
    64.  The implementation evaluates the complete coefficient vector modulo
    deterministic descending 31-bit primes, then reconstructs it with CRT.
    A coefficientwise absolute bound certifies uniqueness.  If both matrices
    are nonnegative, unsigned reconstruction needs a prime product only
    strictly larger than the bound; otherwise centered signed reconstruction
    uses a product strictly larger than twice the bound.

    Coefficients are returned in ascending degree order, including trailing
    zeroes.  The empty permanent is ``(1,)``.
    """
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a boolean")
    num_threads = _validate_num_threads(num_threads)
    selected = _select_backend(backend)
    constant, linear = _validate_affine_matrices(
        constant_matrix, linear_matrix
    )
    threads = num_threads if parallel else 1
    return _affine_permanent_validated(
        constant, linear, selected, threads
    )


def principal_permanent_polynomial_mod(
    matrix,
    modulus: int,
    *,
    backend: str = "auto",
    parallel: bool = True,
    num_threads: int = 0,
) -> tuple[int, ...]:
    """Return the principal-permanent polynomial modulo ``modulus``.

    Entry ``k`` is the sum of ``per(matrix[S, S])`` over every cardinality-
    ``k`` index set ``S``; equivalently, the result contains the coefficients
    of ``per(I + x*matrix)`` as canonical residues.  ``modulus`` must be odd,
    greater than one, and fit in a positive signed 64-bit integer, but it need
    not be prime.

    The portable Python backend uses the direct fixed-sign projection.  The
    native backend uses the general affine-permanent residue kernel.
    """
    modulus = _validate_modulus(modulus)
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a boolean")
    num_threads = _validate_num_threads(num_threads)
    selected = _select_backend(backend)
    normalized = _snapshot_integer_matrix(matrix, "matrix")
    order = len(normalized)
    identity = tuple(
        tuple(int(row == column) for column in range(order))
        for row in range(order)
    )
    threads = num_threads if parallel else 1
    return _affine_permanent_mod_validated(
        identity, normalized, modulus, selected, threads
    )


def principal_permanent_polynomial(
    matrix,
    *,
    backend: str = "auto",
    parallel: bool = True,
    num_threads: int = 0,
) -> tuple[int, ...]:
    """Return the exact principal-permanent generating polynomial.

    Entry ``k`` is the sum of ``per(matrix[S, S])`` over every cardinality-
    ``k`` index set ``S``; equivalently, the result is the coefficient vector
    of ``per(I + x*matrix)``.  Arbitrary square integer matrices of order less
    than 64 are accepted, and the empty result is ``(1,)``.

    The universal implementation uses the certified affine-permanent CRT
    path.  The portable Python backend recognizes the identity constant and
    uses a cheaper direct sign projection in which the two Glynn sign
    products cancel.
    """
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a boolean")
    num_threads = _validate_num_threads(num_threads)
    selected = _select_backend(backend)
    normalized = _snapshot_integer_matrix(matrix, "matrix")
    order = len(normalized)
    identity = tuple(
        tuple(int(row == column) for column in range(order))
        for row in range(order)
    )
    threads = num_threads if parallel else 1
    return _affine_permanent_validated(
        identity, normalized, selected, threads
    )


def rook_polynomial(
    matrix,
    *,
    backend: str = "auto",
    parallel: bool = True,
    num_threads: int = 0,
) -> tuple[int, ...]:
    """Return exact matching counts for a square zero-one matrix.

    Entry ``k`` is the number of ways to choose ``k`` one-entries with no
    shared row or column.  Integral zeroes and ones (including booleans) are
    accepted; all other entries are rejected.  The tuple has ``order + 1``
    entries in ascending ``k`` order, including trailing zeroes.  The empty
    matrix has rook polynomial ``(1,)``.

    Internally, if ``J`` is the all-ones matrix, the coefficient of ``x**k``
    in ``per(J + x*matrix)`` is ``(n-k)!`` times the requested count.  The
    quotient coefficients are reconstructed directly with unsigned CRT.
    An independent characteristic-two determinant supplies their parity,
    halving the required CRT range, while fixed-point coefficientwise
    partial-matching bounds certify the selected lifts.
    """
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a boolean")
    num_threads = _validate_num_threads(num_threads)
    selected = _select_backend(backend)
    normalized = _snapshot_binary_matrix(matrix)

    order = len(normalized)
    if order == 0:
        return (1,)
    if order == 1:
        return (1, normalized[0][0])
    if not any(any(row) for row in normalized):
        return (1,) + (0,) * order

    bounds = _rook_polynomial_bounds_validated(normalized)
    parities = _rook_polynomial_parities_validated(normalized)
    # The product of odd CRT primes changes parity when it is added.  Once
    # every coefficient is below twice that product, its independent parity
    # bit distinguishes the canonical residue from the only possible lift.
    parity_target = max(bounds) // 2
    threads = num_threads if parallel else 1

    fused_native = (
        selected == "native"
        and hasattr(_polynomial_ext, "rook_polynomial_mod_many")
    )
    if fused_native:
        maximum_row_degree = max(sum(row) for row in normalized)
        maximum_column_degree = max(
            sum(normalized[row][column] for row in range(order))
            for column in range(order)
        )
        maximum_degree = min(
            maximum_row_degree, maximum_column_degree
        )
        primes = _plan_rook_pseudo_mersenne_primes(
            parity_target,
            order=order,
            maximum_degree=maximum_degree,
        )
    else:
        primes = _plan_crt_primes(parity_target, signed=False)

    reconstructed = [0] * (order + 1)
    reconstruction_modulus = 1

    residue_rows = []
    if fused_native:
        for start in range(0, len(primes), _ROOK_NATIVE_MAX_MODULI):
            modulus_chunk = primes[
                start:start + _ROOK_NATIVE_MAX_MODULI
            ]
            residue_rows.extend(
                _native_rook_polynomial_many(
                    normalized, modulus_chunk, threads
                )
            )
    else:
        all_ones = ((1,) * order,) * order
        factorials = tuple(
            factorial(order - degree) for degree in range(order + 1)
        )
        for prime in primes:
            scaled_residues = _affine_permanent_mod_validated(
                all_ones, normalized, prime, selected, threads
            )
            residue_rows.append(tuple(
                coefficient * pow(divisor, -1, prime) % prime
                for coefficient, divisor in zip(
                    scaled_residues, factorials
                )
            ))

    for prime, residues in zip(primes, residue_rows):
        if reconstruction_modulus == 1:
            reconstructed[:] = residues
        else:
            inverse = pow(reconstruction_modulus % prime, -1, prime)
            for degree, residue in enumerate(residues):
                correction = (
                    (residue - reconstructed[degree]) * inverse
                ) % prime
                reconstructed[degree] += (
                    reconstruction_modulus * correction
                )
        reconstruction_modulus *= prime

    exact = tuple(
        value
        if value % 2 == parity
        else value + reconstruction_modulus
        for value, parity in zip(reconstructed, parities)
    )
    if not all(
        0 <= value <= bound for value, bound in zip(exact, bounds)
    ):
        raise RuntimeError(
            "CRT reconstruction lies outside its certified rook bound"
        )
    return exact


def _weighted_rook_components_validated(
    matrix: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return nonempty connected components of the bipartite support.

    Row and column vertices are distinct even when their indices agree.
    Isolated vertices contribute the polynomial one and are omitted. Unlike
    perfect-matching SCC factors, these genuinely disconnected components
    also factor every partial-matching coefficient.
    """
    order = len(matrix)
    # A row incident to every column connects all nonisolated rows. This
    # common dense case needs no per-edge union/find bookkeeping.
    if order and all(matrix[0]):
        rows = tuple(row for row, values in enumerate(matrix) if any(values))
        return ((rows, tuple(range(order))),)

    row_columns = []
    column_rows = [0] * order
    unseen_rows = 0
    for row, values in enumerate(matrix):
        row_bit = 1 << row
        columns = 0
        for column, value in enumerate(values):
            if value:
                columns |= 1 << column
                column_rows[column] |= row_bit
        row_columns.append(columns)
        if columns:
            unseen_rows |= row_bit

    def indices(mask):
        result = []
        while mask:
            bit = mask & -mask
            result.append(bit.bit_length() - 1)
            mask ^= bit
        return tuple(result)

    # With at most one incident row per column, each nonempty row is its
    # own star component. In particular, diagonals need no graph traversal.
    if all(mask & (mask - 1) == 0 for mask in column_rows):
        return tuple(((row,), indices(columns))
                     for row, columns in enumerate(row_columns) if columns)

    components = []
    while unseen_rows:
        pending_rows = unseen_rows & -unseen_rows
        unseen_rows ^= pending_rows
        rows = columns = 0
        while pending_rows:
            row_bit = pending_rows & -pending_rows
            pending_rows ^= row_bit
            rows |= row_bit
            new_columns = row_columns[row_bit.bit_length() - 1] & ~columns
            columns |= new_columns
            while new_columns:
                column_bit = new_columns & -new_columns
                new_columns ^= column_bit
                neighbors = column_rows[column_bit.bit_length() - 1] & unseen_rows
                unseen_rows ^= neighbors
                pending_rows |= neighbors
        components.append((indices(rows), indices(columns)))
    return tuple(components)


def _weighted_rook_small_component_validated(matrix, rows, columns):
    """Count one thin support component with at most eight exact DP states.

    The smaller side has at most three vertices. A mask records those already
    matched; processing a vertex on the longer side either skips it or uses
    one free neighbor. Descending mask order prevents using that vertex twice.
    All operations use arbitrary-size integers, without modular reconstruction.
    """
    degree = min(len(rows), len(columns))
    if not 1 <= degree <= 3:
        raise ValueError("small weighted-rook component must have smaller side 1..3")
    if len(columns) <= len(rows):
        values = ([matrix[row][column] for column in columns] for row in rows)
    else:
        values = ([matrix[row][column] for row in rows] for column in columns)
    full_mask = (1 << degree) - 1
    states = [1] + [0] * full_mask
    for weights in values:
        for mask in range(full_mask - 1, -1, -1):
            count = states[mask]
            if not count:
                continue
            available = full_mask ^ mask
            while available:
                bit = available & -available
                available ^= bit
                states[mask | bit] += count * weights[bit.bit_length() - 1]
    coefficients = [0] * (degree + 1)
    # Mask cardinalities for the at most eight states; compatible with 3.8.
    for cardinality, count in zip((0, 1, 1, 2, 1, 2, 2, 3), states):
        coefficients[cardinality] += count
    return tuple(coefficients)


def weighted_rook_polynomial(
    matrix,
    *,
    backend: str = "auto",
    parallel: bool = True,
    num_threads: int = 0,
) -> tuple[int, ...]:
    """Return exact weighted matching sums for a square matrix.

    Matrix entries must be nonnegative integers; arbitrary-size Python
    integers and booleans are accepted.  Entry ``k`` of the returned tuple is
    the sum, over all size-``k`` partial matchings, of the product of their
    selected entries.  The tuple has ``order + 1`` entries in ascending
    cardinality order, including trailing zeroes, and the empty convention is
    ``(1,)``.

    If ``J`` is the all-ones matrix, the coefficient of ``x**k`` in
    ``per(J + x*matrix)`` is ``(n-k)!`` times the requested value.  Exact
    coefficient bounds and an independently computed characteristic-two
    parity vector certify the unsigned CRT reconstruction.  A fused native
    residue kernel is used when available and its input and modulus safety
    contracts are satisfied; otherwise the public affine-permanent backend
    supplies the same residues. Disconnected bipartite support components
    are evaluated separately and their exact polynomials multiplied. A
    one-row or one-column component has polynomial ``1 + sum(weights)*x``;
    this includes each isolated edge of a permuted diagonal matrix. Other
    decomposed components with at most three vertices on their smaller side
    use a bounded exact subset calculation, avoiding repeated CRT setup. Internal
    zero padding handles rectangular components without changing their
    partial matchings. Trailing zeroes retain the original square order.
    """
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a boolean")
    num_threads = _validate_num_threads(num_threads)
    selected = _select_backend(backend)
    normalized = _snapshot_nonnegative_integer_matrix(matrix)

    order = len(normalized)
    if order == 0:
        return (1,)
    if order == 1:
        return (1, normalized[0][0])
    if not any(any(row) for row in normalized):
        return (1,) + (0,) * order

    threads = num_threads if parallel else 1
    components = _weighted_rook_components_validated(normalized)
    if (
        len(components) == 1
        and max(map(len, components[0])) == order
        and min(map(len, components[0])) > 1
    ):
        return _weighted_rook_polynomial_crt_validated(
            normalized, selected, threads
        )

    # A matching splits uniquely across disconnected support components.
    # Multiplication convolves its cardinalities and multiplies its weights.
    # No rounding or CRT is needed for this combination of exact integers.
    coefficients = [1]
    for rows, columns in components:
        degree = min(len(rows), len(columns))
        if degree == 1:
            factor = (1, sum(normalized[row][column]
                             for row in rows for column in columns))
        elif degree <= 3:
            factor = _weighted_rook_small_component_validated(
                normalized, rows, columns
            )
        else:
            size = max(len(rows), len(columns))
            block = tuple(
                tuple(
                    normalized[rows[row]][columns[column]]
                    if row < len(rows) and column < len(columns) else 0
                    for column in range(size)
                )
                for row in range(size)
            )
            exact = _weighted_rook_polynomial_crt_validated(
                block, selected, threads
            )
            if any(exact[degree + 1:]):
                raise RuntimeError(
                    "weighted-rook component exceeds its matching degree"
                )
            factor = exact[:degree + 1]
        product = [0] * (len(coefficients) + len(factor) - 1)
        for left_degree, left in enumerate(coefficients):
            for right_degree, right in enumerate(factor):
                product[left_degree + right_degree] += left * right
        coefficients = product
    return tuple(coefficients) + (0,) * (order + 1 - len(coefficients))


def _weighted_rook_polynomial_crt_validated(
    normalized: Sequence[Sequence[int]], selected: str, threads: int,
) -> tuple[int, ...]:
    """Evaluate one already validated square weighted-rook input by CRT."""
    order = len(normalized)
    bounds = _weighted_rook_polynomial_bounds_validated(normalized)
    parities = _weighted_rook_polynomial_parities_validated(normalized)
    parity_target = max(bounds) // 2

    fused_native = (
        selected == "native"
        and hasattr(
            _polynomial_ext, "weighted_rook_polynomial_mod_many"
        )
        and all(
            value <= _WEIGHTED_ROOK_NATIVE_MAX_ENTRY
            for row in normalized
            for value in row
        )
    )
    primes = None
    if fused_native:
        maximum_row_sum = max(sum(row) for row in normalized)
        maximum_column_sum = max(
            sum(normalized[row][column] for row in range(order))
            for column in range(order)
        )
        maximum_signed_sum = min(
            maximum_row_sum, maximum_column_sum
        )
        try:
            primes = _plan_weighted_rook_pseudo_mersenne_primes(
                parity_target,
                order=order,
                maximum_signed_sum=maximum_signed_sum,
            )
        except RuntimeError:
            # The general affine path accepts arbitrary-size weights and has
            # no common-width pseudo-Mersenne planning restriction.
            fused_native = False
    if not fused_native:
        primes = _plan_crt_primes(parity_target, signed=False)

    reconstructed = [0] * (order + 1)
    reconstruction_modulus = 1
    residue_rows = []
    if fused_native:
        for start in range(0, len(primes), _ROOK_NATIVE_MAX_MODULI):
            modulus_chunk = primes[
                start:start + _ROOK_NATIVE_MAX_MODULI
            ]
            residue_rows.extend(
                _native_weighted_rook_polynomial_many(
                    normalized, modulus_chunk, threads
                )
            )
    else:
        all_ones = ((1,) * order,) * order
        factorials = tuple(
            factorial(order - degree) for degree in range(order + 1)
        )
        for prime in primes:
            scaled_residues = _affine_permanent_mod_validated(
                all_ones, normalized, prime, selected, threads
            )
            residue_rows.append(tuple(
                coefficient * pow(divisor, -1, prime) % prime
                for coefficient, divisor in zip(
                    scaled_residues, factorials
                )
            ))

    for prime, residues in zip(primes, residue_rows):
        if reconstruction_modulus == 1:
            reconstructed[:] = residues
        else:
            inverse = pow(reconstruction_modulus % prime, -1, prime)
            for degree, residue in enumerate(residues):
                correction = (
                    (residue - reconstructed[degree]) * inverse
                ) % prime
                reconstructed[degree] += (
                    reconstruction_modulus * correction
                )
        reconstruction_modulus *= prime

    exact = tuple(
        value
        if value % 2 == parity
        else value + reconstruction_modulus
        for value, parity in zip(reconstructed, parities)
    )
    if not all(
        0 <= value <= bound for value, bound in zip(exact, bounds)
    ):
        raise RuntimeError(
            "CRT reconstruction lies outside its certified "
            "weighted-rook bound"
        )
    return exact


__all__ = [
    "HAS_POLYNOMIAL_EXT",
    "affine_permanent",
    "affine_permanent_bounds",
    "affine_permanent_mod",
    "polynomial_backend_capabilities",
    "principal_permanent_polynomial",
    "principal_permanent_polynomial_mod",
    "rook_polynomial",
    "weighted_rook_polynomial",
]
