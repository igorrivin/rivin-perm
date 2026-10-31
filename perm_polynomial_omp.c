/*
 * Exact coefficient-valued Glynn permanent for affine integer matrices.
 *
 * The setup.py build installs this as perm_polynomial_ext.  A manual build
 * can use, for example:
 *
 *   cc -O3 -std=c11 -fPIC -shared -fopenmp $(python3-config --includes) \
 *      perm_polynomial_omp.c -o perm_polynomial_ext$(python3-config --extension-suffix)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <limits.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__) && defined(__linux__) && \
    defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 10)
#define ROOK_HAVE_RUNTIME_SVE 1
#include <arm_sve.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#else
#define ROOK_HAVE_RUNTIME_SVE 0
#endif

#define POLYNOMIAL_MAX_ORDER 63
#define CACHE_LINE_BYTES 64
#define MERSENNE31 UINT64_C(2147483647)
#define ROOK_MAX_MODULI 16
#define WEIGHTED_ROOK_MAX_ENTRY UINT16_MAX

typedef enum {
    MOD_MERSENNE31,
    MOD_SMALL,
    MOD_WIDE
} ModulusKind;

typedef struct {
    uint64_t modulus;
    ModulusKind kind;
} ModulusContext;

typedef struct {
    uint8_t row;
    uint64_t constant_mod;
    uint64_t linear_mod;
    int32_t constant_signed;
    int32_t linear_signed;
} ColumnUpdate;

typedef struct {
    int n;
    int fixed_column;
    int use_signed_sums;
    int variable_columns[POLYNOMIAL_MAX_ORDER];
    size_t update_offsets[POLYNOMIAL_MAX_ORDER + 1];
    ColumnUpdate *updates;
    const uint64_t *constant_mod;
    const uint64_t *linear_mod;
    const int64_t *constant_signed;
    const int64_t *linear_signed;
    ModulusContext modulus;
} AffineEngine;

static inline uint64_t mersenne31_reduce(uint64_t value) {
    uint64_t result = (value & MERSENNE31) + (value >> 31);
    result = (result & MERSENNE31) + (result >> 31);
    if (result >= MERSENNE31) result -= MERSENNE31;
    return result;
}

static inline uint64_t add_mod(uint64_t left, uint64_t right,
                               const ModulusContext *context) {
    uint64_t modulus = context->modulus;
    return left >= modulus - right ? left - (modulus - right) : left + right;
}

static inline uint64_t sub_mod(uint64_t left, uint64_t right,
                               const ModulusContext *context) {
    uint64_t modulus = context->modulus;
    return left >= right ? left - right : modulus - (right - left);
}

static inline uint64_t mul_mod(uint64_t left, uint64_t right,
                               const ModulusContext *context) {
    if (context->kind == MOD_MERSENNE31) {
        return mersenne31_reduce(left * right);
    }
    if (context->kind == MOD_SMALL) {
        return (left * right) % context->modulus;
    }
    return (uint64_t)(((__uint128_t)left * (__uint128_t)right) %
                      (__uint128_t)context->modulus);
}

static inline uint64_t mul_add_mod(uint64_t first_left,
                                   uint64_t first_right,
                                   uint64_t second_left,
                                   uint64_t second_right,
                                   const ModulusContext *context) {
    if (context->kind == MOD_MERSENNE31) {
        uint64_t value = first_left * first_right +
                         second_left * second_right;
        return mersenne31_reduce(value);
    }
    if (context->kind == MOD_SMALL) {
        uint64_t value = first_left * first_right +
                         second_left * second_right;
        return value % context->modulus;
    }
    __uint128_t value = (__uint128_t)first_left * first_right +
                        (__uint128_t)second_left * second_right;
    return (uint64_t)(value % (__uint128_t)context->modulus);
}

static uint64_t magnitude_i64(int64_t value) {
    if (value >= 0) return (uint64_t)value;
    return (uint64_t)(-(value + 1)) + UINT64_C(1);
}

static uint64_t signed_mod(int64_t value, const ModulusContext *context) {
    uint64_t residue = magnitude_i64(value) % context->modulus;
    if (value >= 0 || residue == 0) return residue;
    return context->modulus - residue;
}

static int64_t centered_representative(uint64_t value, uint64_t modulus) {
    if (value <= modulus / UINT64_C(2)) return (int64_t)value;
    return -(int64_t)(modulus - value);
}

static int parse_index_i64(PyObject *object, const char *name,
                           int64_t *result) {
    if (PyBool_Check(object) || !PyIndex_Check(object)) {
        PyErr_Format(PyExc_TypeError, "%s must be an integer", name);
        return 0;
    }
    PyObject *index = PyNumber_Index(object);
    if (!index) return 0;
    long long value = PyLong_AsLongLong(index);
    Py_DECREF(index);
    if (value == -1 && PyErr_Occurred()) return 0;
    *result = (int64_t)value;
    return 1;
}

static int parse_matrix_entry_i64(PyObject *object, int64_t *result) {
    if (!PyIndex_Check(object)) {
        PyErr_SetString(PyExc_TypeError, "matrix entries must be integers");
        return 0;
    }
    PyObject *index = PyNumber_Index(object);
    if (!index) return 0;
    long long value = PyLong_AsLongLong(index);
    Py_DECREF(index);
    if (value == -1 && PyErr_Occurred()) return 0;
    *result = (int64_t)value;
    return 1;
}

static int parse_matrix(PyObject *object, const char *name, int expected_n,
                        const ModulusContext *context,
                        uint64_t **normalized_out,
                        int64_t **centered_out, int *n_out) {
    if (!PySequence_Check(object)) {
        PyErr_Format(PyExc_TypeError, "%s must be a sequence of rows", name);
        return 0;
    }
    PyObject *outer = PySequence_Tuple(object);
    if (!outer) return 0;
    Py_ssize_t python_n = PyTuple_GET_SIZE(outer);
    if (python_n >= 64) {
        Py_DECREF(outer);
        PyErr_Format(PyExc_ValueError, "%s order must be less than 64", name);
        return 0;
    }
    if (expected_n >= 0 && python_n != expected_n) {
        Py_DECREF(outer);
        PyErr_SetString(PyExc_ValueError,
                        "constant_matrix and linear_matrix must have the same order");
        return 0;
    }
    int n = (int)python_n;

    PyObject *rows = PyTuple_New(python_n);
    if (!rows) {
        Py_DECREF(outer);
        return 0;
    }
    for (int row = 0; row < n; row++) {
        PyObject *row_source = PyTuple_GET_ITEM(outer, row);
        if (!PySequence_Check(row_source)) {
            Py_DECREF(rows);
            Py_DECREF(outer);
            PyErr_SetString(PyExc_TypeError,
                            "matrix rows must be sequences");
            return 0;
        }
        PyObject *row_object = PySequence_Tuple(row_source);
        if (!row_object) {
            Py_DECREF(rows);
            Py_DECREF(outer);
            return 0;
        }
        if (PyTuple_GET_SIZE(row_object) != python_n) {
            Py_DECREF(row_object);
            Py_DECREF(rows);
            Py_DECREF(outer);
            PyErr_Format(PyExc_ValueError, "%s must be square", name);
            return 0;
        }
        PyTuple_SET_ITEM(rows, row, row_object);
    }
    Py_DECREF(outer);

    size_t count = (size_t)n * (size_t)n;
    uint64_t *normalized = NULL;
    int64_t *centered = NULL;
    if (count != 0) {
        normalized = (uint64_t *)malloc(count * sizeof(*normalized));
        centered = (int64_t *)malloc(count * sizeof(*centered));
        if (!normalized || !centered) {
            free(normalized);
            free(centered);
            Py_DECREF(rows);
            PyErr_NoMemory();
            return 0;
        }
    }

    for (int row = 0; row < n; row++) {
        PyObject *row_object = PyTuple_GET_ITEM(rows, row);
        for (int column = 0; column < n; column++) {
            int64_t signed_value;
            if (!parse_matrix_entry_i64(
                    PyTuple_GET_ITEM(row_object, column),
                    &signed_value)) {
                free(normalized);
                free(centered);
                Py_DECREF(rows);
                return 0;
            }
            uint64_t residue = signed_mod(signed_value, context);
            size_t index = (size_t)row * (size_t)n + (size_t)column;
            normalized[index] = residue;
            centered[index] = centered_representative(
                residue, context->modulus);
        }
    }
    Py_DECREF(rows);
    *normalized_out = normalized;
    *centered_out = centered;
    *n_out = n;
    return 1;
}

static int signed_sums_are_small(const int64_t *constant,
                                 const int64_t *linear, int n) {
    for (int row = 0; row < n; row++) {
        uint64_t constant_bound = 0;
        uint64_t linear_bound = 0;
        for (int column = 0; column < n; column++) {
            size_t index = (size_t)row * (size_t)n + (size_t)column;
            uint64_t constant_abs = magnitude_i64(constant[index]);
            uint64_t linear_abs = magnitude_i64(linear[index]);
            if (constant_abs > (uint64_t)INT32_MAX - constant_bound ||
                linear_abs > (uint64_t)INT32_MAX - linear_bound) {
                return 0;
            }
            constant_bound += constant_abs;
            linear_bound += linear_abs;
        }
    }
    return 1;
}

static int column_cost(const uint64_t *constant, const uint64_t *linear,
                       int n, int column) {
    int cost = 0;
    for (int row = 0; row < n; row++) {
        size_t index = (size_t)row * (size_t)n + (size_t)column;
        if (constant[index] != 0 || linear[index] != 0) cost++;
    }
    return cost;
}

static int prepare_engine(AffineEngine *engine, int n,
                          const uint64_t *constant_mod,
                          const uint64_t *linear_mod,
                          const int64_t *constant_signed,
                          const int64_t *linear_signed,
                          ModulusContext modulus) {
    memset(engine, 0, sizeof(*engine));
    engine->n = n;
    engine->constant_mod = constant_mod;
    engine->linear_mod = linear_mod;
    engine->constant_signed = constant_signed;
    engine->linear_signed = linear_signed;
    engine->modulus = modulus;
    engine->use_signed_sums = signed_sums_are_small(
        constant_signed, linear_signed, n);

    int costs[POLYNOMIAL_MAX_ORDER] = {0};
    int fixed = 0;
    for (int column = 0; column < n; column++) {
        costs[column] = column_cost(constant_mod, linear_mod, n, column);
        if (costs[column] > costs[fixed]) fixed = column;
    }
    engine->fixed_column = fixed;

    int variable_count = 0;
    for (int column = 0; column < n; column++) {
        if (column == fixed) continue;
        int position = variable_count;
        while (position > 0) {
            int prior = engine->variable_columns[position - 1];
            if (costs[prior] < costs[column] ||
                (costs[prior] == costs[column] && prior < column)) {
                break;
            }
            engine->variable_columns[position] = prior;
            position--;
        }
        engine->variable_columns[position] = column;
        variable_count++;
    }

    size_t capacity = (size_t)n * (size_t)(n > 0 ? n - 1 : 0);
    if (capacity != 0) {
        engine->updates = (ColumnUpdate *)malloc(
            capacity * sizeof(*engine->updates));
        if (!engine->updates) return 0;
    }
    size_t update_count = 0;
    for (int bit = 0; bit < n - 1; bit++) {
        engine->update_offsets[bit] = update_count;
        int column = engine->variable_columns[bit];
        for (int row = 0; row < n; row++) {
            size_t index = (size_t)row * (size_t)n + (size_t)column;
            if (constant_mod[index] == 0 && linear_mod[index] == 0) continue;
            ColumnUpdate update;
            update.row = (uint8_t)row;
            update.constant_mod = constant_mod[index];
            update.linear_mod = linear_mod[index];
            if (engine->use_signed_sums) {
                update.constant_signed = (int32_t)constant_signed[index];
                update.linear_signed = (int32_t)linear_signed[index];
            } else {
                update.constant_signed = 0;
                update.linear_signed = 0;
            }
            engine->updates[update_count++] = update;
        }
    }
    engine->update_offsets[n - 1] = update_count;
    return 1;
}

static int initialize_modular_sums(const AffineEngine *engine, uint64_t gray,
                                   uint64_t *constant_sums,
                                   uint64_t *linear_sums) {
    int n = engine->n;
    int fixed = engine->fixed_column;
    for (int row = 0; row < n; row++) {
        size_t index = (size_t)row * (size_t)n + (size_t)fixed;
        constant_sums[row] = engine->constant_mod[index];
        linear_sums[row] = engine->linear_mod[index];
    }
    for (int bit = 0; bit < n - 1; bit++) {
        int negative = (int)((gray >> bit) & UINT64_C(1));
        for (size_t index = engine->update_offsets[bit];
             index < engine->update_offsets[bit + 1]; index++) {
            const ColumnUpdate *update = &engine->updates[index];
            int row = (int)update->row;
            if (negative) {
                constant_sums[row] = sub_mod(
                    constant_sums[row], update->constant_mod,
                    &engine->modulus);
                linear_sums[row] = sub_mod(
                    linear_sums[row], update->linear_mod,
                    &engine->modulus);
            } else {
                constant_sums[row] = add_mod(
                    constant_sums[row], update->constant_mod,
                    &engine->modulus);
                linear_sums[row] = add_mod(
                    linear_sums[row], update->linear_mod,
                    &engine->modulus);
            }
        }
    }
    int zero_count = 0;
    for (int row = 0; row < n; row++) {
        zero_count += constant_sums[row] == 0 && linear_sums[row] == 0;
    }
    return zero_count;
}

static int initialize_signed_sums(const AffineEngine *engine, uint64_t gray,
                                  int32_t *constant_sums,
                                  int32_t *linear_sums) {
    int n = engine->n;
    int fixed = engine->fixed_column;
    for (int row = 0; row < n; row++) {
        size_t index = (size_t)row * (size_t)n + (size_t)fixed;
        constant_sums[row] = (int32_t)engine->constant_signed[index];
        linear_sums[row] = (int32_t)engine->linear_signed[index];
    }
    for (int bit = 0; bit < n - 1; bit++) {
        int direction = ((gray >> bit) & UINT64_C(1)) ? -1 : 1;
        for (size_t index = engine->update_offsets[bit];
             index < engine->update_offsets[bit + 1]; index++) {
            const ColumnUpdate *update = &engine->updates[index];
            int row = (int)update->row;
            int64_t constant_value = (int64_t)constant_sums[row] +
                (int64_t)direction * update->constant_signed;
            int64_t linear_value = (int64_t)linear_sums[row] +
                (int64_t)direction * update->linear_signed;
            constant_sums[row] = (int32_t)constant_value;
            linear_sums[row] = (int32_t)linear_value;
        }
    }
    int zero_count = 0;
    for (int row = 0; row < n; row++) {
        zero_count += constant_sums[row] == 0 && linear_sums[row] == 0;
    }
    return zero_count;
}

static int polynomial_term_modular(const uint64_t *constant_sums,
                                   const uint64_t *linear_sums, int n,
                                   const ModulusContext *modulus,
                                   uint64_t *output) {
    uint64_t first[POLYNOMIAL_MAX_ORDER + 1] = {0};
    uint64_t second[POLYNOMIAL_MAX_ORDER + 1] = {0};
    uint64_t *current = first;
    uint64_t *next = second;
    int degree = 0;
    current[0] = UINT64_C(1) % modulus->modulus;

    for (int row = 0; row < n; row++) {
        uint64_t constant = constant_sums[row];
        uint64_t linear = linear_sums[row];
        if (constant == 0 && linear == 0) return 0;
        if (linear == 0) {
            #pragma omp simd
            for (int index = 0; index <= degree; index++) {
                current[index] = mul_mod(current[index], constant, modulus);
            }
            continue;
        }
        if (constant == 0) {
            #pragma omp simd
            for (int index = 0; index <= degree; index++) {
                next[index + 1] = mul_mod(current[index], linear, modulus);
            }
            next[0] = 0;
            degree++;
            uint64_t *swap = current;
            current = next;
            next = swap;
            continue;
        }

        next[0] = mul_mod(current[0], constant, modulus);
        #pragma omp simd
        for (int index = 1; index <= degree; index++) {
            next[index] = mul_add_mod(
                current[index], constant,
                current[index - 1], linear, modulus);
        }
        next[degree + 1] = mul_mod(current[degree], linear, modulus);
        degree++;
        uint64_t *swap = current;
        current = next;
        next = swap;
    }
    memset(output, 0, (size_t)(n + 1) * sizeof(*output));
    memcpy(output, current, (size_t)(degree + 1) * sizeof(*output));
    return 1;
}

/* Return -1 for modular fallback, 0 for a zero term, and 1 for success. */
static int polynomial_term_signed(const int32_t *constant_sums,
                                  const int32_t *linear_sums, int n,
                                  const ModulusContext *modulus,
                                  uint64_t *output) {
    uint64_t bound = 1;
    uint64_t scalar = UINT64_C(1) % modulus->modulus;
    int shift = 0;
    for (int row = 0; row < n; row++) {
        int64_t constant = constant_sums[row];
        int64_t linear = linear_sums[row];
        if (constant == 0 && linear == 0) return 0;
        if (constant == 0 || linear == 0) {
            int64_t factor = constant == 0 ? linear : constant;
            scalar = mul_mod(scalar, signed_mod(factor, modulus), modulus);
            if (constant == 0) shift++;
            if (scalar == 0) return 0;
            continue;
        }
        uint64_t norm = magnitude_i64(constant) + magnitude_i64(linear);
        if (bound > (uint64_t)INT64_MAX / norm) return -1;
        bound *= norm;
    }

    int64_t first[POLYNOMIAL_MAX_ORDER + 1] = {0};
    int64_t second[POLYNOMIAL_MAX_ORDER + 1] = {0};
    int64_t *current = first;
    int64_t *next = second;
    int degree = 0;
    current[0] = 1;
    for (int row = 0; row < n; row++) {
        int64_t constant = constant_sums[row];
        int64_t linear = linear_sums[row];
        if (constant == 0 || linear == 0) continue;
        next[0] = current[0] * constant;
        #pragma omp simd
        for (int index = 1; index <= degree; index++) {
            next[index] = current[index] * constant +
                          current[index - 1] * linear;
        }
        next[degree + 1] = current[degree] * linear;
        degree++;
        int64_t *swap = current;
        current = next;
        next = swap;
    }

    memset(output, 0, (size_t)(n + 1) * sizeof(*output));
    for (int index = 0; index <= degree; index++) {
        uint64_t coefficient = signed_mod(current[index], modulus);
        output[index + shift] = mul_mod(coefficient, scalar, modulus);
    }
    return 1;
}

static int update_modular_sums(const AffineEngine *engine, int bit,
                               int old_negative, uint64_t *constant_sums,
                               uint64_t *linear_sums) {
    int zero_delta = 0;
    for (size_t index = engine->update_offsets[bit];
         index < engine->update_offsets[bit + 1]; index++) {
        const ColumnUpdate *update = &engine->updates[index];
        int row = (int)update->row;
        int was_zero = constant_sums[row] == 0 && linear_sums[row] == 0;
        uint64_t constant_twice = add_mod(
            update->constant_mod, update->constant_mod, &engine->modulus);
        uint64_t linear_twice = add_mod(
            update->linear_mod, update->linear_mod, &engine->modulus);
        if (old_negative) {
            constant_sums[row] = add_mod(
                constant_sums[row], constant_twice, &engine->modulus);
            linear_sums[row] = add_mod(
                linear_sums[row], linear_twice, &engine->modulus);
        } else {
            constant_sums[row] = sub_mod(
                constant_sums[row], constant_twice, &engine->modulus);
            linear_sums[row] = sub_mod(
                linear_sums[row], linear_twice, &engine->modulus);
        }
        int is_zero = constant_sums[row] == 0 && linear_sums[row] == 0;
        zero_delta += is_zero - was_zero;
    }
    return zero_delta;
}

static int update_signed_sums(const AffineEngine *engine, int bit,
                              int old_negative, int32_t *constant_sums,
                              int32_t *linear_sums) {
    int direction = old_negative ? 2 : -2;
    int zero_delta = 0;
    for (size_t index = engine->update_offsets[bit];
         index < engine->update_offsets[bit + 1]; index++) {
        const ColumnUpdate *update = &engine->updates[index];
        int row = (int)update->row;
        int was_zero = constant_sums[row] == 0 && linear_sums[row] == 0;
        int64_t constant_value = (int64_t)constant_sums[row] +
            (int64_t)direction * update->constant_signed;
        int64_t linear_value = (int64_t)linear_sums[row] +
            (int64_t)direction * update->linear_signed;
        constant_sums[row] = (int32_t)constant_value;
        linear_sums[row] = (int32_t)linear_value;
        int is_zero = constant_sums[row] == 0 && linear_sums[row] == 0;
        zero_delta += is_zero - was_zero;
    }
    return zero_delta;
}

static void accumulate_term(uint64_t *accumulator, const uint64_t *term,
                            int n, int negative,
                            const ModulusContext *modulus) {
    if (negative) {
        #pragma omp simd
        for (int degree = 0; degree <= n; degree++) {
            accumulator[degree] = sub_mod(
                accumulator[degree], term[degree], modulus);
        }
    } else {
        #pragma omp simd
        for (int degree = 0; degree <= n; degree++) {
            accumulator[degree] = add_mod(
                accumulator[degree], term[degree], modulus);
        }
    }
}

static void evaluate_modular_range(const AffineEngine *engine,
                                   uint64_t start, uint64_t end,
                                   uint64_t *accumulator) {
    uint64_t constant_sums[POLYNOMIAL_MAX_ORDER] = {0};
    uint64_t linear_sums[POLYNOMIAL_MAX_ORDER] = {0};
    uint64_t term[POLYNOMIAL_MAX_ORDER + 1] = {0};
    uint64_t gray = start ^ (start >> 1);
    int zero_count = initialize_modular_sums(
        engine, gray, constant_sums, linear_sums);

    for (uint64_t index = start; index < end; index++) {
        if (zero_count == 0 && polynomial_term_modular(
                constant_sums, linear_sums, engine->n,
                &engine->modulus, term)) {
            accumulate_term(accumulator, term, engine->n,
                            (int)(index & UINT64_C(1)), &engine->modulus);
        }
        if (index + UINT64_C(1) == end) break;
        int bit = __builtin_ctzll(index + UINT64_C(1));
        int old_negative = (int)((gray >> bit) & UINT64_C(1));
        zero_count += update_modular_sums(
            engine, bit, old_negative, constant_sums, linear_sums);
        gray ^= UINT64_C(1) << bit;
    }
}

static void evaluate_signed_range(const AffineEngine *engine,
                                  uint64_t start, uint64_t end,
                                  uint64_t *accumulator) {
    int32_t constant_sums[POLYNOMIAL_MAX_ORDER] = {0};
    int32_t linear_sums[POLYNOMIAL_MAX_ORDER] = {0};
    uint64_t constant_mod[POLYNOMIAL_MAX_ORDER] = {0};
    uint64_t linear_mod[POLYNOMIAL_MAX_ORDER] = {0};
    uint64_t term[POLYNOMIAL_MAX_ORDER + 1] = {0};
    uint64_t gray = start ^ (start >> 1);
    int zero_count = initialize_signed_sums(
        engine, gray, constant_sums, linear_sums);

    for (uint64_t index = start; index < end; index++) {
        int status = 0;
        if (zero_count == 0) {
            status = polynomial_term_signed(
                constant_sums, linear_sums, engine->n,
                &engine->modulus, term);
        }
        if (status < 0) {
            for (int row = 0; row < engine->n; row++) {
                constant_mod[row] = signed_mod(
                    constant_sums[row], &engine->modulus);
                linear_mod[row] = signed_mod(
                    linear_sums[row], &engine->modulus);
            }
            status = polynomial_term_modular(
                constant_mod, linear_mod, engine->n,
                &engine->modulus, term);
        }
        if (status > 0) {
            accumulate_term(accumulator, term, engine->n,
                            (int)(index & UINT64_C(1)), &engine->modulus);
        }
        if (index + UINT64_C(1) == end) break;
        int bit = __builtin_ctzll(index + UINT64_C(1));
        int old_negative = (int)((gray >> bit) & UINT64_C(1));
        zero_count += update_signed_sums(
            engine, bit, old_negative, constant_sums, linear_sums);
        gray ^= UINT64_C(1) << bit;
    }
}

static int affine_permanent_compute(const AffineEngine *engine,
                                    int num_threads, uint64_t *result) {
    int n = engine->n;
    uint64_t states = UINT64_C(1) << (n - 1);
    if ((uint64_t)num_threads > states) num_threads = (int)states;
    if (num_threads < 1) num_threads = 1;
    size_t result_stride = (size_t)n + 1;
    size_t stride_bytes = result_stride * sizeof(uint64_t);
    stride_bytes = ((stride_bytes + CACHE_LINE_BYTES - 1) /
                    CACHE_LINE_BYTES) * CACHE_LINE_BYTES;
    size_t thread_stride = stride_bytes / sizeof(uint64_t);
    if ((size_t)num_threads > SIZE_MAX / thread_stride) return 0;
    size_t allocation_entries = (size_t)num_threads * thread_stride;
    if (allocation_entries >
            (SIZE_MAX - (CACHE_LINE_BYTES - 1)) / sizeof(uint64_t)) {
        return 0;
    }
    size_t allocation_bytes = allocation_entries * sizeof(uint64_t) +
                              (CACHE_LINE_BYTES - 1);
    unsigned char *thread_allocation =
        (unsigned char *)calloc(allocation_bytes, 1);
    if (!thread_allocation) return 0;
    uintptr_t aligned_address =
        ((uintptr_t)thread_allocation + (CACHE_LINE_BYTES - 1)) &
        ~((uintptr_t)CACHE_LINE_BYTES - (uintptr_t)1);
    uint64_t *thread_results = (uint64_t *)aligned_address;

    uint64_t task_limit = (uint64_t)(unsigned int)num_threads *
                          UINT64_C(1024);
    uint64_t chunks_of_1024 =
        (states + UINT64_C(1023)) / UINT64_C(1024);
    uint64_t task_count = task_limit < chunks_of_1024
        ? task_limit : chunks_of_1024;
    if (task_count < (uint64_t)(unsigned int)num_threads) {
        task_count = (uint64_t)(unsigned int)num_threads;
    }

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        uint64_t *accumulator =
            thread_results + (size_t)thread_id * thread_stride;

        #pragma omp for schedule(dynamic, 1)
        for (uint64_t task = 0; task < task_count; task++) {
            uint64_t start = (uint64_t)(
                ((__uint128_t)states * (__uint128_t)task) /
                (__uint128_t)task_count);
            uint64_t end = (uint64_t)(
                ((__uint128_t)states * (__uint128_t)(task + UINT64_C(1))) /
                (__uint128_t)task_count);
            if (engine->use_signed_sums) {
                evaluate_signed_range(engine, start, end, accumulator);
            } else {
                evaluate_modular_range(engine, start, end, accumulator);
            }
        }
    }

    memset(result, 0, result_stride * sizeof(*result));
    for (int thread = 0; thread < num_threads; thread++) {
        const uint64_t *accumulator =
            thread_results + (size_t)thread * thread_stride;
        for (int degree = 0; degree <= n; degree++) {
            result[degree] = add_mod(
                result[degree], accumulator[degree], &engine->modulus);
        }
    }
    free(thread_allocation);

    uint64_t inverse_two = engine->modulus.modulus / UINT64_C(2) +
                           UINT64_C(1);
    uint64_t scale = UINT64_C(1) % engine->modulus.modulus;
    for (int count = 1; count < n; count++) {
        scale = mul_mod(scale, inverse_two, &engine->modulus);
    }
    for (int degree = 0; degree <= n; degree++) {
        result[degree] = mul_mod(
            result[degree], scale, &engine->modulus);
    }
    return 1;
}

/* Specialized multi-modulus rook-polynomial engine.
 *
 * Residues are flattened as [degree][modulus].  Updating a coefficient from
 * the preceding degree is therefore a fixed displacement by modulus_count.
 * This layout fills SIMD vectors even when the modulus count is not a multiple
 * of the hardware lane count, and needs no cross-lane permutation.
 */

typedef struct {
    int n;
    int modulus_count;
    int modulus_bits;
    int entries;
    uint32_t modulus[ROOK_MAX_MODULI];
    uint32_t complement[ROOK_MAX_MODULI];
    uint32_t *lane_modulus;
    uint32_t *lane_complement;
    int32_t initial_column_sum[POLYNOMIAL_MAX_ORDER];
    uint8_t adjacency_count[POLYNOMIAL_MAX_ORDER - 1];
    struct RookUpdate *adjacency;
} RookEngine;

typedef struct RookUpdate {
    uint8_t column;
    uint16_t weight;
} RookUpdate;

static inline uint32_t rook_reduce_pseudo_mersenne(
        uint32_t value, uint32_t modulus, uint32_t complement,
        int modulus_bits) {
    uint32_t mask = (UINT32_C(1) << modulus_bits) - UINT32_C(1);
    uint32_t reduced = (value & mask) +
        (value >> modulus_bits) * complement;
    return reduced >= modulus ? reduced - modulus : reduced;
}

static inline void rook_polynomial_term_scalar(
        const RookEngine *engine, const int32_t *signed_column_sums,
        uint32_t *term) {
    int modulus_count = engine->modulus_count;
    memset(term, 0, (size_t)engine->entries * sizeof(*term));
    for (int modulus_index = 0; modulus_index < modulus_count;
         modulus_index++) {
        term[modulus_index] = UINT32_C(1);
    }

    int degree = 0;
    for (int column = 0; column < engine->n; column++) {
        int32_t signed_value = signed_column_sums[column];
        if (signed_value == 0) continue;
        uint32_t magnitude = signed_value < 0
            ? (uint32_t)(-(int64_t)signed_value)
            : (uint32_t)signed_value;
        int end = (degree + 2) * modulus_count;
        for (int position = end - 1; position >= modulus_count;
             position--) {
            uint32_t modulus = engine->lane_modulus[position];
            uint32_t product = term[position - modulus_count] * magnitude;
            uint32_t combined;
            if (signed_value > 0) {
                combined = term[position] + product;
            } else {
                combined = term[position] + modulus * magnitude - product;
            }
            term[position] = rook_reduce_pseudo_mersenne(
                combined, modulus, engine->lane_complement[position],
                engine->modulus_bits);
        }
        degree++;
    }
}

static inline void rook_layer_add_scalar(
        const RookEngine *engine, uint32_t *layer, const uint32_t *term) {
    for (int position = 0; position < engine->entries; position++) {
        uint32_t sum = layer[position] + term[position];
        uint32_t modulus = engine->lane_modulus[position];
        layer[position] = sum >= modulus ? sum - modulus : sum;
    }
}

static void rook_initialize_state(const RookEngine *engine, uint64_t gray,
                                  int32_t *signed_column_sums,
                                  int *negative_count) {
    for (int column = 0; column < engine->n; column++) {
        signed_column_sums[column] =
            engine->initial_column_sum[column];
    }
    *negative_count = __builtin_popcountll(gray);
    uint64_t remaining = gray;
    while (remaining != 0) {
        int bit = __builtin_ctzll(remaining);
        const RookUpdate *updates =
            engine->adjacency + (size_t)bit * (size_t)engine->n;
        for (int index = 0; index < engine->adjacency_count[bit]; index++) {
            int column = (int)updates[index].column;
            signed_column_sums[column] = (int32_t)(
                (int64_t)signed_column_sums[column] -
                INT64_C(2) * (int64_t)updates[index].weight);
        }
        remaining &= remaining - UINT64_C(1);
    }
}

static void rook_evaluate_range_scalar(const RookEngine *engine,
                                       uint64_t start, uint64_t end,
                                       uint32_t *layers) {
    int32_t signed_column_sums[POLYNOMIAL_MAX_ORDER];
    uint32_t term[(POLYNOMIAL_MAX_ORDER + 1) * ROOK_MAX_MODULI];
    uint64_t gray = start ^ (start >> 1);
    int negative_count;
    rook_initialize_state(
        engine, gray, signed_column_sums, &negative_count);

    for (uint64_t state = start; state < end; state++) {
        rook_polynomial_term_scalar(engine, signed_column_sums, term);
        rook_layer_add_scalar(
            engine,
            layers + (size_t)negative_count * (size_t)engine->entries,
            term);
        if (state + UINT64_C(1) == end) break;

        int bit = __builtin_ctzll(state + UINT64_C(1));
        int was_negative = (int)((gray >> bit) & UINT64_C(1));
        int delta = was_negative ? 2 : -2;
        const RookUpdate *updates =
            engine->adjacency + (size_t)bit * (size_t)engine->n;
        for (int index = 0; index < engine->adjacency_count[bit]; index++) {
            int column = (int)updates[index].column;
            signed_column_sums[column] = (int32_t)(
                (int64_t)signed_column_sums[column] +
                (int64_t)delta * (int64_t)updates[index].weight);
        }
        negative_count += was_negative ? -1 : 1;
        gray ^= UINT64_C(1) << bit;
    }
}

#if ROOK_HAVE_RUNTIME_SVE

#define ROOK_SVE_TARGET \
    __attribute__((target("arch=armv8.2-a+sve")))

ROOK_SVE_TARGET
static inline svuint32_t rook_reduce_pseudo_mersenne_sve(
        svbool_t predicate, svuint32_t value, svuint32_t modulus,
        svuint32_t complement, int modulus_bits) {
    svuint32_t mask = svdup_n_u32(
        (UINT32_C(1) << modulus_bits) - UINT32_C(1));
    svuint32_t low = svand_u32_x(predicate, value, mask);
    svuint32_t high = svlsr_u32_x(
        predicate, value, svdup_n_u32((uint32_t)modulus_bits));
    svuint32_t reduced = svmla_u32_x(
        predicate, low, high, complement);
    svbool_t at_least_modulus = svcmpge_u32(
        predicate, reduced, modulus);
    return svsub_u32_m(at_least_modulus, reduced, modulus);
}

ROOK_SVE_TARGET
static inline void rook_polynomial_term_sve(
        const RookEngine *engine, const int32_t *signed_column_sums,
        uint32_t *term) {
    int modulus_count = engine->modulus_count;
    memset(term, 0, (size_t)engine->entries * sizeof(*term));
    for (int modulus_index = 0; modulus_index < modulus_count;
         modulus_index++) {
        term[modulus_index] = UINT32_C(1);
    }

    int degree = 0;
    int lanes = (int)svcntw();
    for (int column = 0; column < engine->n; column++) {
        int32_t signed_value = signed_column_sums[column];
        if (signed_value == 0) continue;
        uint32_t magnitude = signed_value < 0
            ? (uint32_t)(-(int64_t)signed_value)
            : (uint32_t)signed_value;
        svuint32_t magnitude_vector = svdup_n_u32(magnitude);
        int end = (degree + 2) * modulus_count;
        while (end > modulus_count) {
            int start = end - lanes;
            if (start < modulus_count) start = modulus_count;
            svbool_t predicate = svwhilelt_b32(
                (uint64_t)start, (uint64_t)end);
            svuint32_t old_value = svld1_u32(predicate, term + start);
            svuint32_t shifted = svld1_u32(
                predicate, term + start - modulus_count);
            svuint32_t modulus = svld1_u32(
                predicate, engine->lane_modulus + start);
            svuint32_t complement = svld1_u32(
                predicate, engine->lane_complement + start);
            svuint32_t product = svmul_u32_x(
                predicate, shifted, magnitude_vector);
            svuint32_t combined;
            if (signed_value > 0) {
                combined = svadd_u32_x(predicate, old_value, product);
            } else {
                svuint32_t bias = svmul_u32_x(
                    predicate, modulus, magnitude_vector);
                combined = svsub_u32_x(
                    predicate,
                    svadd_u32_x(predicate, old_value, bias),
                    product);
            }
            svst1_u32(
                predicate, term + start,
                rook_reduce_pseudo_mersenne_sve(
                    predicate, combined, modulus, complement,
                    engine->modulus_bits));
            end = start;
        }
        degree++;
    }
}

ROOK_SVE_TARGET
static inline void rook_layer_add_sve(
        const RookEngine *engine, uint32_t *layer, const uint32_t *term) {
    int lanes = (int)svcntw();
    for (int start = 0; start < engine->entries; start += lanes) {
        svbool_t predicate = svwhilelt_b32(
            (uint64_t)start, (uint64_t)engine->entries);
        svuint32_t left = svld1_u32(predicate, layer + start);
        svuint32_t right = svld1_u32(predicate, term + start);
        svuint32_t modulus = svld1_u32(
            predicate, engine->lane_modulus + start);
        svuint32_t sum = svadd_u32_x(predicate, left, right);
        svbool_t at_least_modulus = svcmpge_u32(
            predicate, sum, modulus);
        svst1_u32(
            predicate, layer + start,
            svsub_u32_m(at_least_modulus, sum, modulus));
    }
}

ROOK_SVE_TARGET
static void rook_evaluate_range_sve(const RookEngine *engine,
                                    uint64_t start, uint64_t end,
                                    uint32_t *layers) {
    int32_t signed_column_sums[POLYNOMIAL_MAX_ORDER];
    uint32_t term[(POLYNOMIAL_MAX_ORDER + 1) * ROOK_MAX_MODULI];
    uint64_t gray = start ^ (start >> 1);
    int negative_count;
    rook_initialize_state(
        engine, gray, signed_column_sums, &negative_count);

    for (uint64_t state = start; state < end; state++) {
        rook_polynomial_term_sve(engine, signed_column_sums, term);
        rook_layer_add_sve(
            engine,
            layers + (size_t)negative_count * (size_t)engine->entries,
            term);
        if (state + UINT64_C(1) == end) break;

        int bit = __builtin_ctzll(state + UINT64_C(1));
        int was_negative = (int)((gray >> bit) & UINT64_C(1));
        int delta = was_negative ? 2 : -2;
        const RookUpdate *updates =
            engine->adjacency + (size_t)bit * (size_t)engine->n;
        for (int index = 0; index < engine->adjacency_count[bit]; index++) {
            int column = (int)updates[index].column;
            signed_column_sums[column] = (int32_t)(
                (int64_t)signed_column_sums[column] +
                (int64_t)delta * (int64_t)updates[index].weight);
        }
        negative_count += was_negative ? -1 : 1;
        gray ^= UINT64_C(1) << bit;
    }
}

static int rook_runtime_has_sve(void) {
    return (getauxval(AT_HWCAP) & (unsigned long)HWCAP_SVE) != 0;
}

#endif

static uint32_t rook_mod_pow(uint32_t base, uint64_t exponent,
                             uint32_t modulus) {
    uint64_t result = UINT64_C(1);
    uint64_t power = base % modulus;
    while (exponent != 0) {
        if (exponent & UINT64_C(1)) result = result * power % modulus;
        power = power * power % modulus;
        exponent >>= 1;
    }
    return (uint32_t)result;
}

static void rook_finish_residues(
        const RookEngine *engine, const uint32_t *layers,
        uint32_t result[ROOK_MAX_MODULI][POLYNOMIAL_MAX_ORDER + 1]) {
    for (int modulus_index = 0;
         modulus_index < engine->modulus_count;
         modulus_index++) {
        uint32_t modulus = engine->modulus[modulus_index];
        uint32_t inverse_two_power = rook_mod_pow(
            rook_mod_pow(UINT32_C(2), (uint64_t)(engine->n - 1), modulus),
            (uint64_t)modulus - UINT64_C(2), modulus);
        uint32_t factorial[POLYNOMIAL_MAX_ORDER + 1];
        factorial[0] = UINT32_C(1);
        for (int value = 1; value <= engine->n; value++) {
            factorial[value] = (uint32_t)(
                (uint64_t)factorial[value - 1] * (uint32_t)value % modulus);
        }

        for (int degree = 0; degree <= engine->n; degree++) {
            uint32_t sum = 0;
            for (int negative_count = 0;
                 negative_count < engine->n;
                 negative_count++) {
                int signed_total = engine->n - 2 * negative_count;
                uint32_t total_modulus = signed_total >= 0
                    ? (uint32_t)signed_total
                    : modulus - (uint32_t)(-signed_total);
                uint32_t total_power = rook_mod_pow(
                    total_modulus,
                    (uint64_t)(engine->n - degree), modulus);
                uint32_t layer_value = layers[
                    (size_t)negative_count * (size_t)engine->entries +
                    (size_t)degree * (size_t)engine->modulus_count +
                    (size_t)modulus_index];
                uint32_t term = (uint32_t)(
                    (uint64_t)layer_value * total_power % modulus);
                if (negative_count & 1) {
                    sum = sum >= term ? sum - term : sum + modulus - term;
                } else {
                    sum = sum >= modulus - term
                        ? sum - (modulus - term)
                        : sum + term;
                }
            }
            uint32_t inverse_factorial = rook_mod_pow(
                factorial[engine->n - degree],
                (uint64_t)modulus - UINT64_C(2), modulus);
            result[modulus_index][degree] = (uint32_t)(
                (uint64_t)sum * inverse_two_power % modulus *
                inverse_factorial % modulus);
        }
    }
}

static int rook_polynomial_compute_many(
        const RookEngine *engine, int num_threads,
        uint32_t result[ROOK_MAX_MODULI][POLYNOMIAL_MAX_ORDER + 1]) {
    uint64_t states = UINT64_C(1) << (engine->n - 1);
    if ((uint64_t)num_threads > states) num_threads = (int)states;
    if (num_threads < 1) num_threads = 1;

    size_t layer_entries = (size_t)engine->n * (size_t)engine->entries;
    size_t stride_bytes = layer_entries * sizeof(uint32_t);
    stride_bytes = ((stride_bytes + CACHE_LINE_BYTES - 1) /
                    CACHE_LINE_BYTES) * CACHE_LINE_BYTES;
    size_t thread_stride = stride_bytes / sizeof(uint32_t);
    if ((size_t)num_threads > SIZE_MAX / thread_stride) return 0;
    size_t allocation_entries = (size_t)num_threads * thread_stride;
    if (allocation_entries >
            (SIZE_MAX - (CACHE_LINE_BYTES - 1)) / sizeof(uint32_t)) {
        return 0;
    }
    size_t allocation_bytes = allocation_entries * sizeof(uint32_t) +
                              (CACHE_LINE_BYTES - 1);
    unsigned char *thread_allocation =
        (unsigned char *)calloc(allocation_bytes, 1);
    if (!thread_allocation) return 0;
    uintptr_t aligned_address =
        ((uintptr_t)thread_allocation + (CACHE_LINE_BYTES - 1)) &
        ~((uintptr_t)CACHE_LINE_BYTES - (uintptr_t)1);
    uint32_t *thread_layers = (uint32_t *)aligned_address;

    uint64_t maximum_tasks =
        (uint64_t)(unsigned int)num_threads * UINT64_C(1024);
    uint64_t chunks_of_1024 =
        (states + UINT64_C(1023)) / UINT64_C(1024);
    uint64_t task_count = maximum_tasks < chunks_of_1024
        ? maximum_tasks : chunks_of_1024;
    if (task_count < (uint64_t)(unsigned int)num_threads) {
        task_count = (uint64_t)(unsigned int)num_threads;
    }

#if ROOK_HAVE_RUNTIME_SVE
    int use_sve = rook_runtime_has_sve();
#endif

    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        uint32_t *layers =
            thread_layers + (size_t)thread_id * thread_stride;

        #pragma omp for schedule(dynamic, 1)
        for (uint64_t task = 0; task < task_count; task++) {
            uint64_t start = (uint64_t)(
                ((__uint128_t)states * (__uint128_t)task) /
                (__uint128_t)task_count);
            uint64_t end = (uint64_t)(
                ((__uint128_t)states *
                 (__uint128_t)(task + UINT64_C(1))) /
                (__uint128_t)task_count);
#if ROOK_HAVE_RUNTIME_SVE
            if (use_sve) {
                rook_evaluate_range_sve(engine, start, end, layers);
            } else
#endif
            {
                rook_evaluate_range_scalar(engine, start, end, layers);
            }
        }
    }

    uint32_t *global_layers = (uint32_t *)calloc(
        layer_entries, sizeof(uint32_t));
    if (!global_layers) {
        free(thread_allocation);
        return 0;
    }
    for (int thread = 0; thread < num_threads; thread++) {
        const uint32_t *source =
            thread_layers + (size_t)thread * thread_stride;
        for (int negative_count = 0;
             negative_count < engine->n;
             negative_count++) {
            uint32_t *destination = global_layers +
                (size_t)negative_count * (size_t)engine->entries;
            const uint32_t *term = source +
                (size_t)negative_count * (size_t)engine->entries;
#if ROOK_HAVE_RUNTIME_SVE
            if (use_sve) {
                rook_layer_add_sve(engine, destination, term);
            } else
#endif
            {
                rook_layer_add_scalar(engine, destination, term);
            }
        }
    }

    rook_finish_residues(engine, global_layers, result);
    free(global_layers);
    free(thread_allocation);
    return 1;
}

static PyObject *py_affine_permanent_mod(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *constant_object;
    PyObject *linear_object;
    PyObject *modulus_object;
    PyObject *threads_object = NULL;
    if (!PyArg_ParseTuple(
            args, "OOO|O:affine_permanent_mod", &constant_object,
            &linear_object, &modulus_object, &threads_object)) {
        return NULL;
    }

    int64_t signed_modulus;
    if (!parse_index_i64(modulus_object, "modulus", &signed_modulus)) {
        return NULL;
    }
    if (signed_modulus <= 1 || (signed_modulus & INT64_C(1)) == 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "modulus must be an odd signed 64-bit integer greater than 1");
        return NULL;
    }
    ModulusContext modulus;
    modulus.modulus = (uint64_t)signed_modulus;
    if (modulus.modulus == MERSENNE31) {
        modulus.kind = MOD_MERSENNE31;
    } else if (modulus.modulus < (UINT64_C(1) << 31)) {
        modulus.kind = MOD_SMALL;
    } else {
        modulus.kind = MOD_WIDE;
    }

    int requested_threads = 0;
    if (threads_object) {
        int64_t signed_threads;
        if (!parse_index_i64(threads_object, "num_threads", &signed_threads)) {
            return NULL;
        }
        if (signed_threads < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "num_threads must be non-negative");
            return NULL;
        }
        if (signed_threads > INT_MAX) {
            PyErr_SetString(PyExc_ValueError,
                            "num_threads is too large");
            return NULL;
        }
        requested_threads = (int)signed_threads;
    }
    int max_threads = omp_get_max_threads();
    if (max_threads < 1) max_threads = 1;
    if (requested_threads == 0 || requested_threads > max_threads) {
        requested_threads = max_threads;
    }

    uint64_t *constant_mod = NULL;
    uint64_t *linear_mod = NULL;
    int64_t *constant_signed = NULL;
    int64_t *linear_signed = NULL;
    int n = 0;
    int linear_n = 0;
    if (!parse_matrix(constant_object, "constant_matrix", -1, &modulus,
                      &constant_mod, &constant_signed, &n)) {
        return NULL;
    }
    if (!parse_matrix(linear_object, "linear_matrix", n, &modulus,
                      &linear_mod, &linear_signed, &linear_n)) {
        free(constant_mod);
        free(constant_signed);
        return NULL;
    }
    (void)linear_n;

    if (n == 0) {
        free(constant_mod);
        free(linear_mod);
        free(constant_signed);
        free(linear_signed);
        PyObject *tuple = PyTuple_New(1);
        if (!tuple) return NULL;
        PyObject *one = PyLong_FromUnsignedLongLong(
            UINT64_C(1) % modulus.modulus);
        if (!one) {
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, 0, one);
        return tuple;
    }
    if (n == 1) {
        PyObject *tuple = PyTuple_New(2);
        if (!tuple) {
            free(constant_mod);
            free(linear_mod);
            free(constant_signed);
            free(linear_signed);
            return NULL;
        }
        PyObject *constant_value = PyLong_FromUnsignedLongLong(constant_mod[0]);
        PyObject *linear_value = PyLong_FromUnsignedLongLong(linear_mod[0]);
        free(constant_mod);
        free(linear_mod);
        free(constant_signed);
        free(linear_signed);
        if (!constant_value || !linear_value) {
            Py_XDECREF(constant_value);
            Py_XDECREF(linear_value);
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, 0, constant_value);
        PyTuple_SET_ITEM(tuple, 1, linear_value);
        return tuple;
    }

    AffineEngine engine;
    if (!prepare_engine(&engine, n, constant_mod, linear_mod,
                        constant_signed, linear_signed, modulus)) {
        free(constant_mod);
        free(linear_mod);
        free(constant_signed);
        free(linear_signed);
        PyErr_NoMemory();
        return NULL;
    }
    uint64_t result[POLYNOMIAL_MAX_ORDER + 1] = {0};
    int success;
    Py_BEGIN_ALLOW_THREADS
    success = affine_permanent_compute(&engine, requested_threads, result);
    Py_END_ALLOW_THREADS

    free(engine.updates);
    free(constant_mod);
    free(linear_mod);
    free(constant_signed);
    free(linear_signed);
    if (!success) {
        PyErr_NoMemory();
        return NULL;
    }

    PyObject *tuple = PyTuple_New(n + 1);
    if (!tuple) return NULL;
    for (int degree = 0; degree <= n; degree++) {
        PyObject *value = PyLong_FromUnsignedLongLong(result[degree]);
        if (!value) {
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, degree, value);
    }
    return tuple;
}

static int parse_rook_matrix(PyObject *object, int binary_only,
                             uint16_t **matrix_out, int *n_out) {
    if (!PySequence_Check(object)) {
        PyErr_SetString(PyExc_TypeError, "matrix must be a sequence of rows");
        return 0;
    }
    PyObject *outer = PySequence_Tuple(object);
    if (!outer) return 0;
    Py_ssize_t python_n = PyTuple_GET_SIZE(outer);
    if (python_n >= 64) {
        Py_DECREF(outer);
        PyErr_SetString(PyExc_ValueError,
                        "matrix order must be less than 64");
        return 0;
    }
    int n = (int)python_n;
    PyObject *rows = PyTuple_New(python_n);
    if (!rows) {
        Py_DECREF(outer);
        return 0;
    }
    for (int row = 0; row < n; row++) {
        PyObject *source = PyTuple_GET_ITEM(outer, row);
        if (!PySequence_Check(source)) {
            Py_DECREF(rows);
            Py_DECREF(outer);
            PyErr_SetString(PyExc_TypeError,
                            "matrix rows must be sequences");
            return 0;
        }
        PyObject *snapshot = PySequence_Tuple(source);
        if (!snapshot) {
            Py_DECREF(rows);
            Py_DECREF(outer);
            return 0;
        }
        if (PyTuple_GET_SIZE(snapshot) != python_n) {
            Py_DECREF(snapshot);
            Py_DECREF(rows);
            Py_DECREF(outer);
            PyErr_SetString(PyExc_ValueError, "matrix must be square");
            return 0;
        }
        PyTuple_SET_ITEM(rows, row, snapshot);
    }
    Py_DECREF(outer);

    size_t count = (size_t)n * (size_t)n;
    uint16_t *matrix = count == 0
        ? NULL : (uint16_t *)malloc(count * sizeof(*matrix));
    if (count != 0 && !matrix) {
        Py_DECREF(rows);
        PyErr_NoMemory();
        return 0;
    }
    for (int row = 0; row < n; row++) {
        PyObject *snapshot = PyTuple_GET_ITEM(rows, row);
        for (int column = 0; column < n; column++) {
            PyObject *entry = PyTuple_GET_ITEM(snapshot, column);
            if (!PyIndex_Check(entry)) {
                free(matrix);
                Py_DECREF(rows);
                PyErr_SetString(PyExc_TypeError,
                                "matrix entries must be integers");
                return 0;
            }
            PyObject *index = PyNumber_Index(entry);
            if (!index) {
                free(matrix);
                Py_DECREF(rows);
                return 0;
            }
            long long value = PyLong_AsLongLong(index);
            Py_DECREF(index);
            if (value == -1 && PyErr_Occurred()) {
                free(matrix);
                Py_DECREF(rows);
                return 0;
            }
            if (binary_only && value != 0 && value != 1) {
                free(matrix);
                Py_DECREF(rows);
                PyErr_SetString(PyExc_ValueError,
                                "matrix entries must be 0 or 1");
                return 0;
            }
            if (!binary_only &&
                (value < 0 || value > (long long)WEIGHTED_ROOK_MAX_ENTRY)) {
                free(matrix);
                Py_DECREF(rows);
                PyErr_Format(
                    PyExc_ValueError,
                    "matrix entries must be between 0 and %u",
                    (unsigned)WEIGHTED_ROOK_MAX_ENTRY);
                return 0;
            }
            matrix[(size_t)row * (size_t)n + (size_t)column] =
                (uint16_t)value;
        }
    }
    Py_DECREF(rows);
    *matrix_out = matrix;
    *n_out = n;
    return 1;
}

static int parse_binary_matrix(PyObject *object, uint16_t **matrix_out,
                               int *n_out) {
    return parse_rook_matrix(object, 1, matrix_out, n_out);
}

static int parse_weighted_rook_matrix(PyObject *object,
                                      uint16_t **matrix_out, int *n_out) {
    return parse_rook_matrix(object, 0, matrix_out, n_out);
}

static int rook_is_prime_u32(uint32_t value) {
    if (value < UINT32_C(2)) return 0;
    if ((value & UINT32_C(1)) == 0) return value == UINT32_C(2);
    for (uint32_t divisor = UINT32_C(3);
         divisor <= value / divisor;
         divisor += UINT32_C(2)) {
        if (value % divisor == 0) return 0;
    }
    return 1;
}

static int parse_rook_moduli(PyObject *object, int n,
                             unsigned maximum_signed_sum,
                             RookEngine *engine) {
    if (!PySequence_Check(object)) {
        PyErr_SetString(PyExc_TypeError, "moduli must be a sequence");
        return 0;
    }
    PyObject *snapshot = PySequence_Tuple(object);
    if (!snapshot) return 0;
    Py_ssize_t count = PyTuple_GET_SIZE(snapshot);
    if (count < 1 || count > ROOK_MAX_MODULI) {
        Py_DECREF(snapshot);
        PyErr_Format(
            PyExc_ValueError, "moduli must contain between 1 and %d values",
            ROOK_MAX_MODULI);
        return 0;
    }

    int common_bits = -1;
    for (Py_ssize_t index = 0; index < count; index++) {
        int64_t signed_modulus;
        if (!parse_index_i64(
                PyTuple_GET_ITEM(snapshot, index), "modulus",
                &signed_modulus)) {
            Py_DECREF(snapshot);
            return 0;
        }
        if (signed_modulus <= n ||
            signed_modulus >= (INT64_C(1) << 31) ||
            (signed_modulus & INT64_C(1)) == 0) {
            Py_DECREF(snapshot);
            PyErr_SetString(
                PyExc_ValueError,
                "each modulus must be an odd prime larger than the matrix "
                "order and smaller than 2**31");
            return 0;
        }
        uint32_t modulus = (uint32_t)signed_modulus;
        if (!rook_is_prime_u32(modulus)) {
            Py_DECREF(snapshot);
            PyErr_SetString(PyExc_ValueError,
                            "each modulus must be prime");
            return 0;
        }
        for (Py_ssize_t prior = 0; prior < index; prior++) {
            if (engine->modulus[prior] == modulus) {
                Py_DECREF(snapshot);
                PyErr_SetString(PyExc_ValueError,
                                "moduli must be distinct");
                return 0;
            }
        }

        int bits = 0;
        uint64_t power_of_two = UINT64_C(1);
        while (power_of_two <= modulus) {
            power_of_two <<= 1;
            bits++;
        }
        if (common_bits < 0) common_bits = bits;
        if (bits != common_bits) {
            Py_DECREF(snapshot);
            PyErr_SetString(
                PyExc_ValueError,
                "all moduli must have the same pseudo-Mersenne bit width");
            return 0;
        }
        uint32_t complement = (uint32_t)(power_of_two - modulus);

        /* The negative update forms old + modulus*|E| before subtracting.
         * Its exact safety condition is (max|E|+1)*modulus <= 2**32.
         * One pseudo-Mersenne fold followed by one subtraction is sufficient
         * under the conservative condition (max|E|+2)*c < 2**bits.
         */
        if (((uint64_t)maximum_signed_sum + UINT64_C(1)) * modulus >
                (UINT64_C(1) << 32) ||
            ((uint64_t)maximum_signed_sum + UINT64_C(2)) * complement >=
                power_of_two) {
            Py_DECREF(snapshot);
            PyErr_SetString(
                PyExc_ValueError,
                "moduli are not safe for 32-bit pseudo-Mersenne reduction "
                "at this matrix degree");
            return 0;
        }
        engine->modulus[index] = modulus;
        engine->complement[index] = complement;
    }
    Py_DECREF(snapshot);
    engine->modulus_count = (int)count;
    engine->modulus_bits = common_bits;
    return 1;
}

static int prepare_rook_engine(RookEngine *engine, const uint16_t *matrix,
                               int n, PyObject *moduli_object) {
    memset(engine, 0, sizeof(*engine));
    engine->n = n;

    uint32_t row_weight_sum[POLYNOMIAL_MAX_ORDER] = {0};
    uint32_t column_weight_sum[POLYNOMIAL_MAX_ORDER] = {0};
    unsigned row_nonzero_count[POLYNOMIAL_MAX_ORDER] = {0};
    unsigned column_nonzero_count[POLYNOMIAL_MAX_ORDER] = {0};
    uint32_t maximum_row_weight_sum = 0;
    uint32_t maximum_column_weight_sum = 0;
    for (int row = 0; row < n; row++) {
        for (int column = 0; column < n; column++) {
            uint16_t entry =
                matrix[(size_t)row * (size_t)n + (size_t)column];
            row_weight_sum[row] += (uint32_t)entry;
            column_weight_sum[column] += (uint32_t)entry;
            row_nonzero_count[row] += (unsigned)(entry != 0);
            column_nonzero_count[column] += (unsigned)(entry != 0);
        }
        if (row_weight_sum[row] > maximum_row_weight_sum) {
            maximum_row_weight_sum = row_weight_sum[row];
        }
    }
    for (int column = 0; column < n; column++) {
        if (column_weight_sum[column] > maximum_column_weight_sum) {
            maximum_column_weight_sum = column_weight_sum[column];
        }
    }

    int transpose =
        maximum_row_weight_sum < maximum_column_weight_sum;
    unsigned maximum_signed_sum = (unsigned)(transpose
        ? maximum_row_weight_sum : maximum_column_weight_sum);
    /* Orders zero and one return directly without the pseudo-Mersenne hot
     * loop, so their entry range need not satisfy its intermediate bound. */
    unsigned reduction_maximum = n <= 1 ? 0 : maximum_signed_sum;
    if (!parse_rook_moduli(
            moduli_object, n, reduction_maximum, engine)) {
        return 0;
    }

    engine->entries = (n + 1) * engine->modulus_count;
    size_t lane_bytes = (size_t)engine->entries * sizeof(uint32_t);
    engine->lane_modulus = (uint32_t *)malloc(lane_bytes);
    engine->lane_complement = (uint32_t *)malloc(lane_bytes);
    if (!engine->lane_modulus || !engine->lane_complement) {
        free(engine->lane_modulus);
        free(engine->lane_complement);
        engine->lane_modulus = NULL;
        engine->lane_complement = NULL;
        PyErr_NoMemory();
        return 0;
    }
    for (int position = 0; position < engine->entries; position++) {
        int modulus_index = position % engine->modulus_count;
        engine->lane_modulus[position] = engine->modulus[modulus_index];
        engine->lane_complement[position] =
            engine->complement[modulus_index];
    }

    int fixed_row = 0;
    unsigned fixed_update_count = 0;
    for (int oriented_row = 0; oriented_row < n; oriented_row++) {
        unsigned update_count = transpose
            ? column_nonzero_count[oriented_row]
            : row_nonzero_count[oriented_row];
        if (update_count > fixed_update_count) {
            fixed_update_count = update_count;
            fixed_row = oriented_row;
        }
    }
    for (int oriented_column = 0; oriented_column < n;
         oriented_column++) {
        engine->initial_column_sum[oriented_column] = (int32_t)(transpose
            ? row_weight_sum[oriented_column]
            : column_weight_sum[oriented_column]);
    }

    size_t adjacency_entries = n > 1
        ? (size_t)(n - 1) * (size_t)n : 0;
    engine->adjacency = adjacency_entries == 0
        ? NULL : (RookUpdate *)malloc(
            adjacency_entries * sizeof(*engine->adjacency));
    if (adjacency_entries != 0 && !engine->adjacency) {
        free(engine->lane_modulus);
        free(engine->lane_complement);
        engine->lane_modulus = NULL;
        engine->lane_complement = NULL;
        PyErr_NoMemory();
        return 0;
    }
    int bit = 0;
    for (int oriented_row = 0; oriented_row < n; oriented_row++) {
        if (oriented_row == fixed_row) continue;
        RookUpdate *updates =
            engine->adjacency + (size_t)bit * (size_t)n;
        for (int oriented_column = 0; oriented_column < n;
             oriented_column++) {
            int source_row = transpose ? oriented_column : oriented_row;
            int source_column = transpose ? oriented_row : oriented_column;
            uint16_t weight = matrix[
                (size_t)source_row * (size_t)n + (size_t)source_column];
            if (weight != 0) {
                int index = (int)engine->adjacency_count[bit]++;
                updates[index].column = (uint8_t)oriented_column;
                updates[index].weight = weight;
            }
        }
        bit++;
    }
    return 1;
}

static void destroy_rook_engine(RookEngine *engine) {
    free(engine->adjacency);
    free(engine->lane_modulus);
    free(engine->lane_complement);
    engine->adjacency = NULL;
    engine->lane_modulus = NULL;
    engine->lane_complement = NULL;
}

static PyObject *rook_polynomial_mod_many_common(
        PyObject *matrix_object, PyObject *moduli_object,
        PyObject *threads_object, int binary_only) {
    int requested_threads = 0;
    if (threads_object) {
        int64_t signed_threads;
        if (!parse_index_i64(threads_object, "num_threads", &signed_threads)) {
            return NULL;
        }
        if (signed_threads < 0 || signed_threads > INT_MAX) {
            PyErr_SetString(PyExc_ValueError,
                            "num_threads must be a nonnegative C integer");
            return NULL;
        }
        requested_threads = (int)signed_threads;
    }
    int maximum_threads = omp_get_max_threads();
    if (maximum_threads < 1) maximum_threads = 1;
    if (requested_threads == 0 || requested_threads > maximum_threads) {
        requested_threads = maximum_threads;
    }

    uint16_t *matrix = NULL;
    int n;
    int parsed = binary_only
        ? parse_binary_matrix(matrix_object, &matrix, &n)
        : parse_weighted_rook_matrix(matrix_object, &matrix, &n);
    if (!parsed) return NULL;

    RookEngine engine;
    if (!prepare_rook_engine(&engine, matrix, n, moduli_object)) {
        free(matrix);
        return NULL;
    }
    free(matrix);

    uint32_t result[ROOK_MAX_MODULI][POLYNOMIAL_MAX_ORDER + 1] = {{0}};
    int success = 1;
    if (n == 0) {
        for (int modulus_index = 0;
             modulus_index < engine.modulus_count;
             modulus_index++) {
            result[modulus_index][0] = UINT32_C(1);
        }
    } else if (n == 1) {
        for (int modulus_index = 0;
             modulus_index < engine.modulus_count;
             modulus_index++) {
            result[modulus_index][0] = UINT32_C(1);
            result[modulus_index][1] = (uint32_t)(
                (uint64_t)(uint32_t)engine.initial_column_sum[0] %
                (uint64_t)engine.modulus[modulus_index]);
        }
    } else {
        Py_BEGIN_ALLOW_THREADS
        success = rook_polynomial_compute_many(
            &engine, requested_threads, result);
        Py_END_ALLOW_THREADS
    }
    destroy_rook_engine(&engine);
    if (!success) {
        PyErr_NoMemory();
        return NULL;
    }

    PyObject *outer = PyTuple_New(engine.modulus_count);
    if (!outer) return NULL;
    for (int modulus_index = 0;
         modulus_index < engine.modulus_count;
         modulus_index++) {
        PyObject *coefficients = PyTuple_New(n + 1);
        if (!coefficients) {
            Py_DECREF(outer);
            return NULL;
        }
        for (int degree = 0; degree <= n; degree++) {
            PyObject *value = PyLong_FromUnsignedLong(
                result[modulus_index][degree]);
            if (!value) {
                Py_DECREF(coefficients);
                Py_DECREF(outer);
                return NULL;
            }
            PyTuple_SET_ITEM(coefficients, degree, value);
        }
        PyTuple_SET_ITEM(outer, modulus_index, coefficients);
    }
    return outer;
}

static PyObject *py_rook_polynomial_mod_many(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *matrix_object;
    PyObject *moduli_object;
    PyObject *threads_object = NULL;
    if (!PyArg_ParseTuple(
            args, "OO|O:rook_polynomial_mod_many", &matrix_object,
            &moduli_object, &threads_object)) {
        return NULL;
    }
    return rook_polynomial_mod_many_common(
        matrix_object, moduli_object, threads_object, 1);
}

static PyObject *py_weighted_rook_polynomial_mod_many(
        PyObject *self, PyObject *args) {
    (void)self;
    PyObject *matrix_object;
    PyObject *moduli_object;
    PyObject *threads_object = NULL;
    if (!PyArg_ParseTuple(
            args, "OO|O:weighted_rook_polynomial_mod_many", &matrix_object,
            &moduli_object, &threads_object)) {
        return NULL;
    }
    return rook_polynomial_mod_many_common(
        matrix_object, moduli_object, threads_object, 0);
}

static PyMethodDef PolynomialMethods[] = {
    {"affine_permanent_mod", py_affine_permanent_mod, METH_VARARGS,
     "Return coefficients of per(constant_matrix + X*linear_matrix) mod m.\n"
     "Args: constant_matrix, linear_matrix, modulus, [num_threads]"},
    {"rook_polynomial_mod_many", py_rook_polynomial_mod_many, METH_VARARGS,
     "Return rook-polynomial residues for several pseudo-Mersenne primes.\n"
     "Args: binary_matrix, moduli, [num_threads]"},
    {"weighted_rook_polynomial_mod_many",
     py_weighted_rook_polynomial_mod_many, METH_VARARGS,
     "Return weighted rook-polynomial residues for pseudo-Mersenne primes.\n"
     "Matrix entries must be integers from 0 through 65535.\n"
     "Args: weight_matrix, moduli, [num_threads]"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef polynomial_module = {
    PyModuleDef_HEAD_INIT,
    "perm_polynomial_ext",
    "OpenMP affine-matrix permanent polynomial backend.",
    -1,
    PolynomialMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_perm_polynomial_ext(void) {
    return PyModule_Create(&polynomial_module);
}
