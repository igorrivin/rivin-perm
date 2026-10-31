/*
 * Specialized OpenMP backend for permanents of 0-1 matrices.
 *
 * Fixed-sign Glynn halves the state count relative to Ryser.  Columns are
 * stored as uint64_t row masks, the densest column is fixed at +1, and the
 * remaining columns are assigned to Gray-code bits in ascending degree order.
 * For every row we retain its small signed sum.  A zero counter and the
 * product of all nonzero sums let a column toggle update the summand with one
 * modular multiplication per nonzero matrix entry in that column.
 *
 * The fast path uses 32-bit Montgomery arithmetic for p < 2^31.  A 64-bit
 * Montgomery path keeps the exported API correct for every odd prime that
 * fits in a signed 64-bit integer.  Both paths require p > n so all possible
 * nonzero row sums are invertible.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <omp.h>
#include <stdint.h>

#define BINARY_MAX_ORDER 63

static inline uint64_t add_mod_u64(uint64_t a, uint64_t b, uint64_t p) {
    return a >= p - b ? a - (p - b) : a + b;
}

static inline uint64_t sub_mod_u64(uint64_t a, uint64_t b, uint64_t p) {
    return a >= b ? a - b : p - (b - a);
}

static inline uint64_t mul_mod_normal(uint64_t a, uint64_t b, uint64_t p) {
    return (uint64_t)(((__uint128_t)a * (__uint128_t)b) % p);
}

static uint64_t pow_mod_normal(uint64_t base, uint64_t exponent, uint64_t p) {
    uint64_t result = 1 % p;
    while (exponent != 0) {
        if (exponent & 1U) result = mul_mod_normal(result, base, p);
        base = mul_mod_normal(base, base, p);
        exponent >>= 1;
    }
    return result;
}

/* Deterministic Miller--Rabin for the full uint64_t range. */
static int is_prime_u64(uint64_t value) {
    static const uint32_t small_primes[] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    };
    static const uint64_t witnesses[] = {
        UINT64_C(2), UINT64_C(325), UINT64_C(9375), UINT64_C(28178),
        UINT64_C(450775), UINT64_C(9780504), UINT64_C(1795265022)
    };

    if (value < 2) return 0;
    for (size_t index = 0;
         index < sizeof(small_primes) / sizeof(small_primes[0]); index++) {
        uint64_t prime = small_primes[index];
        if (value == prime) return 1;
        if (value % prime == 0) return 0;
    }

    uint64_t odd_part = value - 1;
    int power_of_two = 0;
    while ((odd_part & 1U) == 0) {
        odd_part >>= 1;
        power_of_two++;
    }

    for (size_t index = 0;
         index < sizeof(witnesses) / sizeof(witnesses[0]); index++) {
        uint64_t base = witnesses[index] % value;
        if (base == 0) continue;
        uint64_t current = pow_mod_normal(base, odd_part, value);
        if (current == 1 || current == value - 1) continue;

        int composite = 1;
        for (int round = 1; round < power_of_two; round++) {
            current = mul_mod_normal(current, current, value);
            if (current == value - 1) {
                composite = 0;
                break;
            }
        }
        if (composite) return 0;
    }
    return 1;
}

static inline uint64_t signed_mod(int value, uint64_t p) {
    return value < 0 ? p - (uint64_t)(-value) : (uint64_t)value;
}

typedef struct {
    uint32_t p;
    uint32_t negative_inverse;
    uint32_t one;
} Mont32;

static Mont32 make_mont32(uint32_t p) {
    uint32_t inverse = p;
    inverse *= 2U - p * inverse;
    inverse *= 2U - p * inverse;
    inverse *= 2U - p * inverse;
    inverse *= 2U - p * inverse;
    inverse *= 2U - p * inverse;
    Mont32 result = {
        p,
        (uint32_t)(0U - inverse),
        (uint32_t)((UINT64_C(1) << 32) % p)
    };
    return result;
}

/* p < 2^31 makes product + multiplier*p overflow-free in uint64_t. */
static inline uint32_t mul_mont32(uint32_t a, uint32_t b, Mont32 context) {
    uint64_t product = (uint64_t)a * (uint64_t)b;
    uint32_t multiplier = (uint32_t)product * context.negative_inverse;
    uint64_t reduced =
        (product + (uint64_t)multiplier * context.p) >> 32;
    if (reduced >= context.p) reduced -= context.p;
    return (uint32_t)reduced;
}

static inline uint32_t to_mont32(uint64_t value, Mont32 context) {
    return (uint32_t)((value * context.one) % context.p);
}

typedef struct {
    uint64_t p;
    uint64_t negative_inverse;
    uint64_t one;
} Mont64;

static Mont64 make_mont64(uint64_t p) {
    uint64_t inverse = p;
    inverse *= 2U - p * inverse;
    inverse *= 2U - p * inverse;
    inverse *= 2U - p * inverse;
    inverse *= 2U - p * inverse;
    inverse *= 2U - p * inverse;
    inverse *= 2U - p * inverse;
    Mont64 result = {
        p,
        (uint64_t)(0U - inverse),
        (uint64_t)((((__uint128_t)1) << 64) % p)
    };
    return result;
}

/* p < 2^63 makes the 128-bit Montgomery accumulation overflow-free. */
static inline uint64_t mul_mont64(uint64_t a, uint64_t b, Mont64 context) {
    __uint128_t product = (__uint128_t)a * (__uint128_t)b;
    uint64_t multiplier =
        (uint64_t)product * context.negative_inverse;
    __uint128_t combined =
        product + (__uint128_t)multiplier * context.p;
    uint64_t reduced = (uint64_t)(combined >> 64);
    if (reduced >= context.p) reduced -= context.p;
    return reduced;
}

static inline uint64_t to_mont64(uint64_t value, Mont64 context) {
    return mul_mod_normal(value, context.one, context.p);
}

static int densest_column(const uint64_t *columns, int n) {
    int best = 0;
    int best_degree = __builtin_popcountll(columns[0]);
    for (int column = 1; column < n; column++) {
        int degree = __builtin_popcountll(columns[column]);
        if (degree > best_degree) {
            best = column;
            best_degree = degree;
        }
    }
    return best;
}

static void order_glynn_columns(const uint64_t *input, int n,
                                uint64_t *ordered) {
    int fixed = densest_column(input, n);
    int count = 0;
    for (int column = 0; column < n; column++) {
        if (column != fixed) ordered[count++] = input[column];
    }

    /* Lowest-degree variables occupy the most frequently toggled Gray bits. */
    for (int index = 1; index < count; index++) {
        uint64_t value = ordered[index];
        int degree = __builtin_popcountll(value);
        int cursor = index;
        while (cursor > 0 &&
               __builtin_popcountll(ordered[cursor - 1]) > degree) {
            ordered[cursor] = ordered[cursor - 1];
            cursor--;
        }
        ordered[cursor] = value;
    }
}

static __uint128_t gray_update_cost(const uint64_t *columns, int n) {
    int degrees[BINARY_MAX_ORDER];
    for (int column = 0; column < n; column++) {
        degrees[column] = __builtin_popcountll(columns[column]);
    }
    for (int index = 1; index < n; index++) {
        int value = degrees[index];
        int cursor = index;
        while (cursor > 0 && degrees[cursor - 1] > value) {
            degrees[cursor] = degrees[cursor - 1];
            cursor--;
        }
        degrees[cursor] = value;
    }

    __uint128_t cost = 0;
    for (int bit = 0; bit < n - 1; bit++) {
        cost += (__uint128_t)degrees[bit] << (n - 2 - bit);
    }
    return cost;
}

/* Split [0,total) without computing total + threads - 1. */
static void thread_interval(uint64_t total, int thread_id, int threads,
                            uint64_t *start, uint64_t *end) {
    uint64_t quotient = total / (uint64_t)threads;
    uint64_t remainder = total % (uint64_t)threads;
    uint64_t preceding_extra =
        (uint64_t)thread_id < remainder ? (uint64_t)thread_id : remainder;
    *start = (uint64_t)thread_id * quotient + preceding_extra;
    *end = *start + quotient + ((uint64_t)thread_id < remainder);
}

#define DEFINE_GLYNN_MONT_KERNEL(NAME, WORD, CONTEXT, TO_MONT, MUL_MONT)       \
static uint64_t NAME(const uint64_t *input_columns, int n, uint64_t p,       \
                     int num_threads, CONTEXT context) {                     \
    uint64_t columns[BINARY_MAX_ORDER - 1];                                  \
    order_glynn_columns(input_columns, n, columns);                          \
                                                                              \
    WORD transition[2][2 * BINARY_MAX_ORDER + 1];                            \
    for (int old = -n; old <= n; old++) {                                    \
        for (int direction = 0; direction < 2; direction++) {                \
            int next = old + (direction ? 2 : -2);                           \
            uint64_t ratio = 0;                                               \
            if (next >= -n && next <= n) {                                   \
                if (old == 0) {                                               \
                    ratio = signed_mod(next, p);                              \
                } else {                                                      \
                    uint64_t inverse = pow_mod_normal(                        \
                        signed_mod(old, p), p - 2, p);                        \
                    ratio = next == 0                                         \
                        ? inverse                                             \
                        : mul_mod_normal(signed_mod(next, p), inverse, p);     \
                }                                                             \
            }                                                                 \
            transition[direction][old + n] = TO_MONT(ratio, context);         \
        }                                                                     \
    }                                                                         \
                                                                              \
    uint64_t states = UINT64_C(1) << (n - 1);                                \
    if ((uint64_t)num_threads > states) num_threads = (int)states;            \
    uint64_t result = 0;                                                       \
    _Pragma("omp parallel num_threads(num_threads) shared(result)")          \
    {                                                                         \
        int thread_id = omp_get_thread_num();                                 \
        int threads = omp_get_num_threads();                                  \
        uint64_t start, end;                                                   \
        thread_interval(states, thread_id, threads, &start, &end);            \
        int16_t row_sums[BINARY_MAX_ORDER] = {0};                             \
        WORD local_result = 0;                                                 \
                                                                              \
        for (int column = 0; column < n; column++) {                          \
            uint64_t rows = input_columns[column];                            \
            while (rows != 0) {                                               \
                int row = __builtin_ctzll(rows);                              \
                row_sums[row]++;                                               \
                rows &= rows - 1;                                              \
            }                                                                 \
        }                                                                     \
                                                                              \
        uint64_t gray = start ^ (start >> 1);                                 \
        uint64_t selected = gray;                                              \
        while (selected != 0) {                                                \
            int column = __builtin_ctzll(selected);                           \
            uint64_t rows = columns[column];                                  \
            while (rows != 0) {                                               \
                int row = __builtin_ctzll(rows);                              \
                row_sums[row] -= 2;                                            \
                rows &= rows - 1;                                              \
            }                                                                 \
            selected &= selected - 1;                                          \
        }                                                                     \
                                                                              \
        int zero_count = 0;                                                    \
        WORD product = context.one;                                            \
        for (int row = 0; row < n; row++) {                                   \
            int value = row_sums[row];                                        \
            if (value == 0) zero_count++;                                     \
            else product = MUL_MONT(                                          \
                product, TO_MONT(signed_mod(value, p), context), context);     \
        }                                                                     \
        int state_sign = (__builtin_popcountll(gray) & 1) ? -1 : 1;           \
                                                                              \
        for (uint64_t state = start; state < end; state++) {                  \
            if (state != start) {                                              \
                int column = __builtin_ctzll(state);                          \
                gray = state ^ (state >> 1);                                  \
                int adding = ((gray >> column) & 1U) ? 0 : 1;                 \
                int delta = adding ? 2 : -2;                                  \
                uint64_t rows = columns[column];                              \
                while (rows != 0) {                                           \
                    int row = __builtin_ctzll(rows);                          \
                    int old = row_sums[row];                                  \
                    int next = old + delta;                                   \
                    product = MUL_MONT(                                       \
                        product, transition[adding][old + n], context);        \
                    if (old == 0) zero_count--;                               \
                    if (next == 0) zero_count++;                              \
                    row_sums[row] = (int16_t)next;                            \
                    rows &= rows - 1;                                          \
                }                                                             \
                state_sign = -state_sign;                                     \
            }                                                                 \
            if (zero_count == 0) {                                            \
                local_result = (WORD)(state_sign < 0                          \
                    ? sub_mod_u64(local_result, product, p)                   \
                    : add_mod_u64(local_result, product, p));                 \
            }                                                                 \
        }                                                                     \
                                                                              \
        _Pragma("omp critical(binary_permanent_reduction)")                  \
        result = add_mod_u64(result, local_result, p);                        \
    }                                                                         \
                                                                              \
    uint64_t inverse_two = (p >> 1) + 1;                                     \
    uint64_t scale = pow_mod_normal(inverse_two, (uint64_t)(n - 1), p);       \
    WORD scaled = MUL_MONT(                                                   \
        (WORD)result, TO_MONT(scale, context), context);                      \
    return MUL_MONT(scaled, (WORD)1, context);                               \
}

DEFINE_GLYNN_MONT_KERNEL(
    binary_glynn_mont32, uint32_t, Mont32, to_mont32, mul_mont32)
DEFINE_GLYNN_MONT_KERNEL(
    binary_glynn_mont64, uint64_t, Mont64, to_mont64, mul_mont64)

#undef DEFINE_GLYNN_MONT_KERNEL

static uint64_t binary_permanent_mod_prime(const uint64_t *column_masks,
                                           const uint64_t *row_masks,
                                           int n, uint64_t p,
                                           int num_threads) {
    const uint64_t *oriented = column_masks;
    if (gray_update_cost(row_masks, n) < gray_update_cost(column_masks, n)) {
        oriented = row_masks;
    }
    if (p < (UINT64_C(1) << 31)) {
        return binary_glynn_mont32(
            oriented, n, p, num_threads, make_mont32((uint32_t)p));
    }
    return binary_glynn_mont64(
        oriented, n, p, num_threads, make_mont64(p));
}

static int parse_index_long_long(PyObject *object, const char *name,
                                 long long *result) {
    if (PyBool_Check(object)) {
        PyErr_Format(PyExc_TypeError, "%s must be an integer", name);
        return 0;
    }
    PyObject *index = PyNumber_Index(object);
    if (!index) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError, "%s must be an integer", name);
        return 0;
    }
    long long value = PyLong_AsLongLong(index);
    Py_DECREF(index);
    if (value == -1 && PyErr_Occurred()) return 0;
    *result = value;
    return 1;
}

static PyObject *py_binary_permanent_mod_p(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *matrix_object;
    PyObject *p_object;
    PyObject *threads_object = NULL;
    long long signed_p = 0;
    int num_threads = 0;
    if (!PyArg_ParseTuple(
            args, "OO|O:binary_permanent_mod_p",
            &matrix_object, &p_object, &threads_object)) {
        return NULL;
    }
    if (!parse_index_long_long(p_object, "p", &signed_p)) return NULL;
    if (threads_object) {
        long long signed_threads;
        if (!parse_index_long_long(
                threads_object, "num_threads", &signed_threads)) {
            return NULL;
        }
        if (signed_threads > INT32_MAX) {
            PyErr_SetString(PyExc_ValueError,
                            "num_threads is too large");
            return NULL;
        }
        num_threads = (int)signed_threads;
    }
    if (num_threads < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "num_threads must be non-negative");
        return NULL;
    }

    PyObject *matrix = PySequence_Fast(
        matrix_object, "matrix must be a sequence of rows");
    if (!matrix) return NULL;
    Py_ssize_t python_n = PySequence_Fast_GET_SIZE(matrix);
    if (python_n > BINARY_MAX_ORDER) {
        Py_DECREF(matrix);
        PyErr_Format(PyExc_ValueError,
                     "native binary backend supports order at most %d",
                     BINARY_MAX_ORDER);
        return NULL;
    }
    int n = (int)python_n;

    if (signed_p <= 1) {
        Py_DECREF(matrix);
        PyErr_SetString(PyExc_ValueError, "p must be greater than 1");
        return NULL;
    }
    uint64_t p = (uint64_t)signed_p;
    if (p == 2) {
        Py_DECREF(matrix);
        PyErr_SetString(PyExc_ValueError, "p must be odd");
        return NULL;
    }
    if (p <= (uint64_t)n) {
        Py_DECREF(matrix);
        PyErr_SetString(PyExc_ValueError,
                        "p must be greater than the matrix order");
        return NULL;
    }
    if (!is_prime_u64(p)) {
        Py_DECREF(matrix);
        PyErr_SetString(PyExc_ValueError, "p must be prime");
        return NULL;
    }

    uint64_t column_masks[BINARY_MAX_ORDER] = {0};
    uint64_t row_masks[BINARY_MAX_ORDER] = {0};
    for (int row = 0; row < n; row++) {
        PyObject *row_object = PySequence_Fast(
            PySequence_Fast_GET_ITEM(matrix, row),
            "matrix rows must be sequences");
        if (!row_object) {
            Py_DECREF(matrix);
            return NULL;
        }
        if (PySequence_Fast_GET_SIZE(row_object) != python_n) {
            Py_DECREF(row_object);
            Py_DECREF(matrix);
            PyErr_SetString(PyExc_ValueError, "matrix must be square");
            return NULL;
        }
        for (int column = 0; column < n; column++) {
            PyObject *entry = PySequence_Fast_GET_ITEM(row_object, column);
            long long value = PyLong_AsLongLong(entry);
            if (value == -1 && PyErr_Occurred()) {
                Py_DECREF(row_object);
                Py_DECREF(matrix);
                return NULL;
            }
            if (value != 0 && value != 1) {
                Py_DECREF(row_object);
                Py_DECREF(matrix);
                PyErr_SetString(PyExc_ValueError,
                                "matrix entries must be 0 or 1");
                return NULL;
            }
            if (value == 1) {
                column_masks[column] |= UINT64_C(1) << row;
                row_masks[row] |= UINT64_C(1) << column;
            }
        }
        Py_DECREF(row_object);
    }
    Py_DECREF(matrix);

    if (n == 0) return PyLong_FromLong(1);
    if (n == 1) {
        return PyLong_FromLong((long)(column_masks[0] & UINT64_C(1)));
    }
    for (int index = 0; index < n; index++) {
        if (column_masks[index] == 0 || row_masks[index] == 0) {
            return PyLong_FromLong(0);
        }
    }
    /* Treat an explicit count as a maximum, not permission to ask the
     * OpenMP runtime to create an unbounded number of workers.  Some
     * runtimes terminate the process (rather than returning an error) when
     * an oversized team cannot be created. */
    int max_threads = omp_get_max_threads();
    if (max_threads < 1) max_threads = 1;
    if (num_threads == 0 || num_threads > max_threads) {
        num_threads = max_threads;
    }

    uint64_t result;
    Py_BEGIN_ALLOW_THREADS
    result = binary_permanent_mod_prime(
        column_masks, row_masks, n, p, num_threads);
    Py_END_ALLOW_THREADS
    return PyLong_FromUnsignedLongLong(result);
}

static PyObject *py_get_max_threads(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyLong_FromLong(omp_get_max_threads());
}

static PyMethodDef BinaryPermanentMethods[] = {
    {"binary_permanent_mod_p", py_binary_permanent_mod_p, METH_VARARGS,
     "Compute a 0-1 matrix permanent modulo an odd prime p > n.\n"
     "Args: matrix, p, [num_threads]"},
    {"get_max_threads", py_get_max_threads, METH_NOARGS,
     "Return the maximum OpenMP thread count."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef binary_permanent_module = {
    PyModuleDef_HEAD_INIT,
    "perm_binary_ext",
    "Specialized OpenMP permanent backend for 0-1 matrices.",
    -1,
    BinaryPermanentMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_perm_binary_ext(void) {
    return PyModule_Create(&binary_permanent_module);
}
