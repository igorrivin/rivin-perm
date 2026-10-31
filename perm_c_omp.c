/*
 * OpenMP-parallelized C extension for fast permanent computation.
 *
 * This version parallelizes Ryser and Glynn across contiguous Gray-code
 * blocks, and the prime-order Schur necklace quotient across FKM subtrees.
 *
 * Compile with: python setup.py build_ext --inplace
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

/* All operands passed to these helpers are in [0, p). */
static inline int64_t normalize_mod_i64(int64_t value, int64_t p) {
    int64_t result = value % p;
    return result < 0 ? result + p : result;
}

static inline int64_t add_mod_i64(int64_t a, int64_t b, int64_t p) {
    return a >= p - b ? a - (p - b) : a + b;
}

static inline int64_t sub_mod_i64(int64_t a, int64_t b, int64_t p) {
    return a >= b ? a - b : p - (b - a);
}

static inline int64_t mul_mod_i64(int64_t a, int64_t b, int64_t p) {
    if (a == 0 || b == 0) return 0;
    if (a <= INT64_MAX / b) return (a * b) % p;
    return (int64_t)(((__int128)a * (__int128)b) % (__int128)p);
}

#define DIRECT_MUL_MOD_LIMIT INT64_C(3037000499)

typedef int64_t (*row_product_mod_p_fn)(const int64_t *, int, int64_t);

static int64_t row_product_mod_p_direct(const int64_t *values, int n,
                                        int64_t p) {
    int64_t product = 1 % p;
    for (int row = 0; row < n; row++) {
        product = (product * values[row]) % p;
    }
    return product;
}

static int64_t row_product_mod_p_safe(const int64_t *values, int n,
                                      int64_t p) {
    int64_t product = 1 % p;
    for (int row = 0; row < n; row++) {
        product = mul_mod_i64(product, values[row], p);
    }
    return product;
}

static row_product_mod_p_fn select_row_product_mod_p(int64_t p) {
    return p <= DIRECT_MUL_MOD_LIMIT
        ? row_product_mod_p_direct
        : row_product_mod_p_safe;
}

static int64_t pow_mod_i64(int64_t base, uint64_t exponent, int64_t p) {
    int64_t result = 1 % p;
    while (exponent) {
        if (exponent & 1U) result = mul_mod_i64(result, base, p);
        base = mul_mod_i64(base, base, p);
        exponent >>= 1;
    }
    return result;
}

#define NECKLACE_SERIAL_MIN_ORDER 11
#define NECKLACE_PARALLEL_MIN_ORDER 19

/* See perm_c.c for the group action and orbit-weight derivation. */

typedef void (*necklace_orbit_callback_fn)(uint64_t, int, void *);

typedef struct {
    int length;
    int max_weight;
    uint8_t a[64];
    necklace_orbit_callback_fn callback;
    void *callback_context;
} NecklaceGenerator;

static uint64_t rotate_word_left(uint64_t word, int shift, int length) {
    if (shift == 0) return word;
    uint64_t mask = (UINT64_C(1) << length) - 1;
    return ((word << shift) & mask) | (word >> (length - shift));
}

static uint64_t least_word_rotation(uint64_t word, int length) {
    uint64_t result = word;
    for (int shift = 1; shift < length; shift++) {
        uint64_t rotated = rotate_word_left(word, shift, length);
        if (rotated < result) result = rotated;
    }
    return result;
}

static void emit_necklace_orbit(NecklaceGenerator *generator, uint64_t word,
                                int period, int weight) {
    int half_weight = generator->length / 2;
    if (weight < half_weight) {
        generator->callback(
            word, 2 * period, generator->callback_context);
        return;
    }

    uint64_t mask = (UINT64_C(1) << generator->length) - 1;
    uint64_t complement_rep = least_word_rotation(
        mask ^ word, generator->length);
    if (word > complement_rep) return;
    generator->callback(
        word,
        word == complement_rep ? period : 2 * period,
        generator->callback_context);
}

static void necklace_recurse(NecklaceGenerator *generator, int t, int period,
                             int weight, uint64_t prefix) {
    if (t > generator->length) {
        if (generator->length % period == 0) {
            emit_necklace_orbit(generator, prefix, period, weight);
        }
        return;
    }

    uint8_t bit = generator->a[t - period];
    if (weight + bit <= generator->max_weight) {
        generator->a[t] = bit;
        necklace_recurse(generator, t + 1, period, weight + bit,
                         (prefix << 1) | bit);
    }
    if (bit == 0 && weight + 1 <= generator->max_weight) {
        generator->a[t] = 1;
        necklace_recurse(generator, t + 1, t, weight + 1,
                         (prefix << 1) | 1U);
    }
}

static void enumerate_necklace_orbits(
    int length, necklace_orbit_callback_fn callback, void *context) {
    NecklaceGenerator generator = {0};
    generator.length = length;
    generator.max_weight = length / 2;
    generator.callback = callback;
    generator.callback_context = context;
    necklace_recurse(&generator, 1, 1, 0, 0);
}

static int is_prime_order(int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if ((n & 1) == 0) return 0;
    for (int divisor = 3; divisor <= n / divisor; divisor += 2) {
        if (n % divisor == 0) return 0;
    }
    return 1;
}

static int primitive_root_prime_order(int prime) {
    int phi = prime - 1;
    int remaining = phi;
    int factors[16];
    int factor_count = 0;

    for (int divisor = 2; divisor <= remaining / divisor; divisor++) {
        if (remaining % divisor == 0) {
            factors[factor_count++] = divisor;
            while (remaining % divisor == 0) remaining /= divisor;
        }
    }
    if (remaining > 1) factors[factor_count++] = remaining;

    for (int candidate = 2; candidate < prime; candidate++) {
        int primitive = 1;
        for (int index = 0; index < factor_count; index++) {
            if (pow_mod_i64(candidate, (uint64_t)(phi / factors[index]),
                            prime) == 1) {
                primitive = 0;
                break;
            }
        }
        if (primitive) return candidate;
    }
    return -1;
}

static int build_circulant_schur_minor(int64_t *minor, int order,
                                       int64_t omega, int64_t p) {
    int length = order - 1;
    int primitive = primitive_root_prime_order(order);
    if (primitive < 0) return 0;

    int64_t first_row[64];
    int exponent = 1;
    for (int column = 0; column < length; column++) {
        first_row[column] = pow_mod_i64(omega, (uint64_t)exponent, p);
        exponent = (exponent * primitive) % order;
    }

    for (int row = 0; row < length; row++) {
        for (int column = 0; column < length; column++) {
            int index = column - row;
            if (index < 0) index += length;
            minor[row * length + column] = first_row[index];
        }
    }
    return 1;
}

typedef struct {
    const int64_t *matrix;
    int length;
    int64_t p;
    row_product_mod_p_fn row_product;
    int64_t row_sums[64];
    uint64_t previous_word;
    int64_t raw_sum;
    uint64_t representatives;
    uint64_t words;
} NecklaceEvaluation;

static void initialize_necklace_evaluation(NecklaceEvaluation *evaluation,
                                           const int64_t *matrix, int length,
                                           int64_t p) {
    evaluation->matrix = matrix;
    evaluation->length = length;
    evaluation->p = p;
    evaluation->row_product = select_row_product_mod_p(p);
    evaluation->previous_word = 0;
    evaluation->raw_sum = 0;
    evaluation->representatives = 0;
    evaluation->words = 0;

    for (int row = 0; row < length; row++) {
        int64_t sum = 0;
        for (int column = 0; column < length; column++) {
            sum = add_mod_i64(sum, matrix[row * length + column], p);
        }
        evaluation->row_sums[row] = sum;
    }
}

static void evaluate_necklace_orbit(uint64_t word, int orbit_size,
                                    void *context) {
    NecklaceEvaluation *evaluation = (NecklaceEvaluation *)context;
    uint64_t difference = evaluation->previous_word ^ word;

    while (difference) {
        int low_bit = __builtin_ctzll(difference);
        int column = evaluation->length - 1 - low_bit;
        int subtracting = (int)((word >> low_bit) & 1U);
        for (int row = 0; row < evaluation->length; row++) {
            int64_t entry = evaluation->matrix[
                row * evaluation->length + column];
            int64_t twice_entry = add_mod_i64(entry, entry, evaluation->p);
            evaluation->row_sums[row] = subtracting
                ? sub_mod_i64(evaluation->row_sums[row], twice_entry,
                              evaluation->p)
                : add_mod_i64(evaluation->row_sums[row], twice_entry,
                              evaluation->p);
        }
        difference &= difference - 1;
    }

    int64_t product = evaluation->row_product(
        evaluation->row_sums, evaluation->length, evaluation->p);
    if (__builtin_popcountll(word) & 1) {
        product = product == 0 ? 0 : evaluation->p - product;
    }
    int64_t weighted = mul_mod_i64(
        (int64_t)((uint64_t)orbit_size % (uint64_t)evaluation->p),
        product, evaluation->p);
    evaluation->raw_sum = add_mod_i64(
        evaluation->raw_sum, weighted, evaluation->p);
    evaluation->previous_word = word;
    evaluation->representatives++;
    evaluation->words += (uint64_t)orbit_size;
}

static int64_t necklace_cofactor_mod_p_serial(
    const int64_t *matrix, int length, int64_t p,
    uint64_t *representatives, uint64_t *words) {
    NecklaceEvaluation evaluation;
    initialize_necklace_evaluation(&evaluation, matrix, length, p);
    enumerate_necklace_orbits(
        length, evaluate_necklace_orbit, &evaluation);

    if (representatives) *representatives = evaluation.representatives;
    if (words) *words = evaluation.words;

    int64_t inverse_two = (p >> 1) + 1;
    int64_t scale = pow_mod_i64(inverse_two, (uint64_t)length, p);
    return mul_mod_i64(evaluation.raw_sum, scale, p);
}

typedef struct {
    uint64_t representatives;
    uint64_t words;
} NecklaceOrbitStats;

static void count_necklace_orbit(uint64_t word, int orbit_size, void *context) {
    (void)word;
    NecklaceOrbitStats *stats = (NecklaceOrbitStats *)context;
    stats->representatives++;
    stats->words += (uint64_t)orbit_size;
}

static NecklaceOrbitStats necklace_orbit_stats(int length) {
    NecklaceOrbitStats stats = {0, 0};
    enumerate_necklace_orbits(length, count_necklace_orbit, &stats);
    return stats;
}

typedef struct {
    uint8_t a[64];
    int t;
    int period;
    int weight;
    uint64_t prefix;
} NecklaceSubtree;

/*
 * Capture complete FKM recursion states at a moderate prefix depth. Each
 * subtree is an independent dynamic work item: it initializes its own row
 * sums once, then keeps incremental representative-to-representative updates.
 */
typedef struct {
    NecklaceSubtree *items;
    size_t count;
    size_t capacity;
    int length;
    int max_weight;
    int split_depth;
    int allocation_failed;
} NecklaceSubtreeList;

static void append_necklace_subtree(NecklaceSubtreeList *list,
                                    const uint8_t *a, int t, int period,
                                    int weight, uint64_t prefix) {
    if (list->allocation_failed) return;
    if (list->count == list->capacity) {
        if (list->capacity > SIZE_MAX / 2) {
            list->allocation_failed = 1;
            return;
        }
        size_t new_capacity = list->capacity ? 2 * list->capacity : 64;
        if (new_capacity > SIZE_MAX / sizeof(NecklaceSubtree)) {
            list->allocation_failed = 1;
            return;
        }
        NecklaceSubtree *new_items = (NecklaceSubtree *)realloc(
            list->items, new_capacity * sizeof(NecklaceSubtree));
        if (!new_items) {
            list->allocation_failed = 1;
            return;
        }
        list->items = new_items;
        list->capacity = new_capacity;
    }

    NecklaceSubtree *subtree = &list->items[list->count++];
    for (int index = 0; index < 64; index++) subtree->a[index] = a[index];
    subtree->t = t;
    subtree->period = period;
    subtree->weight = weight;
    subtree->prefix = prefix;
}

static void collect_necklace_subtrees_recurse(
    NecklaceSubtreeList *list, uint8_t *a, int t, int period, int weight,
    uint64_t prefix) {
    if (list->allocation_failed) return;
    if (t > list->split_depth || t > list->length) {
        append_necklace_subtree(list, a, t, period, weight, prefix);
        return;
    }

    uint8_t bit = a[t - period];
    if (weight + bit <= list->max_weight) {
        a[t] = bit;
        collect_necklace_subtrees_recurse(
            list, a, t + 1, period, weight + bit,
            (prefix << 1) | bit);
    }
    if (bit == 0 && weight + 1 <= list->max_weight) {
        a[t] = 1;
        collect_necklace_subtrees_recurse(
            list, a, t + 1, t, weight + 1,
            (prefix << 1) | 1U);
    }
}

static NecklaceSubtreeList collect_necklace_subtrees(int length,
                                                      int split_depth) {
    NecklaceSubtreeList list = {0};
    uint8_t a[64] = {0};
    list.length = length;
    list.max_weight = length / 2;
    list.split_depth = split_depth;
    collect_necklace_subtrees_recurse(&list, a, 1, 1, 0, 0);
    return list;
}

typedef struct {
    int64_t raw_sum;
    uint64_t representatives;
    uint64_t words;
} NecklaceSubtreeResult;

static NecklaceSubtreeResult evaluate_necklace_subtree(
    const NecklaceSubtree *subtree, const int64_t *matrix, int length,
    int64_t p) {
    NecklaceEvaluation evaluation;
    NecklaceGenerator generator = {0};
    initialize_necklace_evaluation(&evaluation, matrix, length, p);
    generator.length = length;
    generator.max_weight = length / 2;
    for (int index = 0; index < 64; index++) {
        generator.a[index] = subtree->a[index];
    }
    generator.callback = evaluate_necklace_orbit;
    generator.callback_context = &evaluation;
    necklace_recurse(
        &generator, subtree->t, subtree->period, subtree->weight,
        subtree->prefix);

    NecklaceSubtreeResult result = {
        evaluation.raw_sum,
        evaluation.representatives,
        evaluation.words,
    };
    return result;
}

static int64_t necklace_cofactor_mod_p_parallel(
    const int64_t *matrix, int length, int64_t p, int num_threads,
    uint64_t *representatives, uint64_t *words) {
    int log_threads = 0;
    int64_t rounded_threads = 1;
    while (rounded_threads < num_threads && log_threads < 20) {
        rounded_threads <<= 1;
        log_threads++;
    }
    int split_depth = 6 + log_threads;
    if (split_depth > length - 2) split_depth = length - 2;
    if (split_depth < 1) {
        return necklace_cofactor_mod_p_serial(
            matrix, length, p, representatives, words);
    }

    NecklaceSubtreeList subtrees = collect_necklace_subtrees(
        length, split_depth);
    if (subtrees.allocation_failed || subtrees.count == 0 ||
        subtrees.count > (size_t)INT64_MAX) {
        free(subtrees.items);
        return -1;
    }

    NecklaceSubtreeResult *results = (NecklaceSubtreeResult *)calloc(
        subtrees.count, sizeof(NecklaceSubtreeResult));
    if (!results) {
        free(subtrees.items);
        return -1;
    }

    int64_t task_count = (int64_t)subtrees.count;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (int64_t task_index = 0; task_index < task_count; task_index++) {
        results[task_index] = evaluate_necklace_subtree(
            &subtrees.items[task_index], matrix, length, p);
    }

    int64_t raw_sum = 0;
    uint64_t total_representatives = 0;
    uint64_t total_words = 0;
    for (size_t index = 0; index < subtrees.count; index++) {
        raw_sum = add_mod_i64(raw_sum, results[index].raw_sum, p);
        total_representatives += results[index].representatives;
        total_words += results[index].words;
    }

    free(results);
    free(subtrees.items);
    if (representatives) *representatives = total_representatives;
    if (words) *words = total_words;

    int64_t inverse_two = (p >> 1) + 1;
    int64_t scale = pow_mod_i64(inverse_two, (uint64_t)length, p);
    return mul_mod_i64(raw_sum, scale, p);
}

static int64_t necklace_cofactor_mod_p(
    const int64_t *matrix, int length, int64_t p, int num_threads,
    uint64_t *representatives, uint64_t *words) {
    if (length <= 16 || num_threads <= 1) {
        return necklace_cofactor_mod_p_serial(
            matrix, length, p, representatives, words);
    }
    return necklace_cofactor_mod_p_parallel(
        matrix, length, p, num_threads, representatives, words);
}

/* Glynn's formula with the last column sign fixed at +1. */
static int64_t glynn_permanent_mod_p_serial(const int64_t *matrix, int n,
                                             int64_t p) {
    uint64_t num_states = 1ULL << (n - 1);
    row_product_mod_p_fn row_product = select_row_product_mod_p(p);
    int64_t *row_sums = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    if (!row_sums) return -1;

    for (int row = 0; row < n; row++) {
        int64_t sum = 0;
        for (int column = 0; column < n; column++) {
            sum = add_mod_i64(sum, matrix[row * n + column], p);
        }
        row_sums[row] = sum;
    }

    int64_t result = 0;
    int state_sign = 1;

    for (uint64_t state = 0; state < num_states; state++) {
        if (state != 0) {
            int column = __builtin_ctzll(state);
            uint64_t gray = state ^ (state >> 1);
            int subtracting = (int)((gray >> column) & 1U);
            for (int row = 0; row < n; row++) {
                int64_t entry = matrix[row * n + column];
                int64_t twice_entry = add_mod_i64(entry, entry, p);
                row_sums[row] = subtracting
                    ? sub_mod_i64(row_sums[row], twice_entry, p)
                    : add_mod_i64(row_sums[row], twice_entry, p);
            }
            state_sign = -state_sign;
        }

        int64_t product = row_product(row_sums, n, p);
        result = state_sign < 0
            ? sub_mod_i64(result, product, p)
            : add_mod_i64(result, product, p);
    }

    free(row_sums);

    int64_t inverse_two = (p >> 1) + 1;
    int64_t scale = pow_mod_i64(inverse_two, (uint64_t)(n - 1), p);
    return mul_mod_i64(result, scale, p);
}

static int64_t glynn_permanent_mod_p_parallel(const int64_t *matrix, int n,
                                               int64_t p, int num_threads) {
    uint64_t num_states = 1ULL << (n - 1);
    row_product_mod_p_fn row_product = select_row_product_mod_p(p);
    int64_t result = 0;
    int allocation_failed = 0;

    #pragma omp parallel num_threads(num_threads) shared(result, allocation_failed)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        uint64_t block_size = num_states / (uint64_t)nthreads;
        if (num_states % (uint64_t)nthreads) block_size++;
        uint64_t start = (uint64_t)tid * block_size;
        uint64_t end = start + block_size;
        if (end > num_states) end = num_states;
        if (start >= num_states) start = end;

        int64_t local_result = 0;
        int64_t *row_sums =
            (int64_t *)malloc((size_t)n * sizeof(int64_t));

        if (!row_sums) {
            #pragma omp atomic write
            allocation_failed = 1;
        } else if (start < end) {
            for (int row = 0; row < n; row++) {
                int64_t sum = 0;
                for (int column = 0; column < n; column++) {
                    sum = add_mod_i64(sum, matrix[row * n + column], p);
                }
                row_sums[row] = sum;
            }

            uint64_t gray = start ^ (start >> 1);
            for (int column = 0; column < n - 1; column++) {
                if ((gray >> column) & 1U) {
                    for (int row = 0; row < n; row++) {
                        int64_t entry = matrix[row * n + column];
                        int64_t twice_entry = add_mod_i64(entry, entry, p);
                        row_sums[row] = sub_mod_i64(
                            row_sums[row], twice_entry, p);
                    }
                }
            }

            int state_sign =
                (__builtin_popcountll(gray) & 1) ? -1 : 1;

            for (uint64_t state = start; state < end; state++) {
                if (state != start) {
                    int column = __builtin_ctzll(state);
                    gray = state ^ (state >> 1);
                    int subtracting = (int)((gray >> column) & 1U);
                    for (int row = 0; row < n; row++) {
                        int64_t entry = matrix[row * n + column];
                        int64_t twice_entry = add_mod_i64(entry, entry, p);
                        row_sums[row] = subtracting
                            ? sub_mod_i64(row_sums[row], twice_entry, p)
                            : add_mod_i64(row_sums[row], twice_entry, p);
                    }
                    state_sign = -state_sign;
                }

                int64_t product = row_product(row_sums, n, p);
                local_result = state_sign < 0
                    ? sub_mod_i64(local_result, product, p)
                    : add_mod_i64(local_result, product, p);
            }
        }

        if (row_sums) free(row_sums);
        #pragma omp critical
        {
            result = add_mod_i64(result, local_result, p);
        }
    }

    if (allocation_failed) return -1;

    int64_t inverse_two = (p >> 1) + 1;
    int64_t scale = pow_mod_i64(inverse_two, (uint64_t)(n - 1), p);
    return mul_mod_i64(result, scale, p);
}

static int64_t glynn_permanent_mod_p(const int64_t *matrix, int n, int64_t p,
                                     int num_threads) {
    if (n <= 16 || num_threads <= 1) {
        return glynn_permanent_mod_p_serial(matrix, n, p);
    }
    return glynn_permanent_mod_p_parallel(matrix, n, p, num_threads);
}

/*
 * Compute permanent of matrix mod p using parallelized Ryser's algorithm.
 *
 * Uses Gray code block parallelization (arXiv:2407.20205 Section 3.1):
 * - Split 2^n states into contiguous Gray code blocks per thread
 * - Each thread initializes row sums for its starting state (O(n²))
 * - Then uses O(n) Gray code iteration within its block
 *
 * This gives nearly perfect parallelization!
 *
 * matrix: flattened n x n matrix (row-major)
 * n: matrix dimension
 * p: prime modulus
 * num_threads: number of OpenMP threads to use
 *
 * Returns: permanent mod p
 */
static int64_t ryser_permanent_mod_p_parallel(const int64_t *matrix, int n,
                                               int64_t p, int num_threads) {
    if (n == 0) return 1 % p;
    if (n == 1) return matrix[0];

    uint64_t num_subsets = 1ULL << n;
    row_product_mod_p_fn row_product = select_row_product_mod_p(p);
    int64_t result = 0;
    int allocation_failed = 0;

    #pragma omp parallel num_threads(num_threads) shared(result, allocation_failed)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        /* Each thread handles a contiguous block of Gray code states */
        uint64_t block_size = num_subsets / (uint64_t)nthreads;
        if (num_subsets % (uint64_t)nthreads) block_size++;
        uint64_t start = (uint64_t)tid * block_size;
        uint64_t end = start + block_size;
        if (end > num_subsets) end = num_subsets;
        if (start >= num_subsets) start = end;

        /* Skip state 0 (empty subset) */
        if (start == 0) start = 1;

        int64_t local_result = 0;
        int64_t *row_sums = (int64_t *)malloc(n * sizeof(int64_t));

        if (!row_sums) {
            #pragma omp atomic write
            allocation_failed = 1;
        } else if (start < end) {
            /* Initialize row sums for the starting Gray code state */
            uint64_t gray_start = start ^ (start >> 1);

            for (int row = 0; row < n; row++) {
                row_sums[row] = 0;
            }

            int popcount_start = 0;
            for (int col = 0; col < n; col++) {
                if ((gray_start >> col) & 1) {
                    for (int row = 0; row < n; row++) {
                        row_sums[row] = add_mod_i64(
                            row_sums[row], matrix[row * n + col], p);
                    }
                    popcount_start++;
                }
            }

            /* Process starting state */
            int subset_sign = (popcount_start & 1) ? -1 : 1;

            int64_t prod = row_product(row_sums, n, p);
            local_result = subset_sign < 0
                ? sub_mod_i64(local_result, prod, p)
                : add_mod_i64(local_result, prod, p);

            /* Gray code iteration within this block */
            for (uint64_t i = start + 1; i < end; i++) {
                /* Find which column changed from i-1 to i in Gray code */
                uint64_t gray_i = i ^ (i >> 1);
                uint64_t gray_prev = (i - 1) ^ ((i - 1) >> 1);
                uint64_t diff = gray_i ^ gray_prev;
                int col = __builtin_ctzll(diff);

                if ((gray_i >> col) & 1) {
                    /* Column turned on: add it */
                    for (int row = 0; row < n; row++) {
                        row_sums[row] = add_mod_i64(
                            row_sums[row], matrix[row * n + col], p);
                    }
                    subset_sign = -subset_sign;
                } else {
                    /* Column turned off: subtract it */
                    for (int row = 0; row < n; row++) {
                        row_sums[row] = sub_mod_i64(
                            row_sums[row], matrix[row * n + col], p);
                    }
                    subset_sign = -subset_sign;
                }

                /* Compute product of row sums mod p */
                prod = row_product(row_sums, n, p);

                local_result = subset_sign < 0
                    ? sub_mod_i64(local_result, prod, p)
                    : add_mod_i64(local_result, prod, p);
            }
        }

        if (row_sums) free(row_sums);
        #pragma omp critical
        {
            result = add_mod_i64(result, local_result, p);
        }
    }

    if (allocation_failed) return -1;

    /* Apply (-1)^n factor */
    if (n % 2 == 1) {
        result = result == 0 ? 0 : p - result;
    }

    return result;
}

/*
 * Serial version with Gray code (for comparison / small n).
 */
static int64_t ryser_permanent_mod_p_serial(const int64_t *matrix, int n, int64_t p) {
    if (n == 0) return 1 % p;
    if (n == 1) return matrix[0];

    int64_t *row_sums = (int64_t *)calloc(n, sizeof(int64_t));
    if (!row_sums) return -1;

    int64_t result = 0;
    int sign = 1;
    row_product_mod_p_fn row_product = select_row_product_mod_p(p);

    uint64_t num_subsets = 1ULL << n;

    for (uint64_t i = 1; i < num_subsets; i++) {
        int j = __builtin_ctzll(i);
        uint64_t gray_i = i ^ (i >> 1);

        if ((gray_i >> j) & 1) {
            for (int row = 0; row < n; row++) {
                row_sums[row] = add_mod_i64(
                    row_sums[row], matrix[row * n + j], p);
            }
        } else {
            for (int row = 0; row < n; row++) {
                row_sums[row] = sub_mod_i64(
                    row_sums[row], matrix[row * n + j], p);
            }
        }

        sign = -sign;

        int64_t prod = row_product(row_sums, n, p);

        result = sign < 0
            ? sub_mod_i64(result, prod, p)
            : add_mod_i64(result, prod, p);
    }

    free(row_sums);

    if (n % 2 == 1) {
        result = result == 0 ? 0 : p - result;
    }

    return result;
}

/*
 * Choose between serial and parallel based on n and available threads.
 */
static int64_t ryser_permanent_mod_p(const int64_t *matrix, int n,
                                      int64_t p, int num_threads) {
    /* For small n, serial Gray code is faster due to lower overhead */
    if (n <= 16 || num_threads <= 1) {
        return ryser_permanent_mod_p_serial(matrix, n, p);
    }
    return ryser_permanent_mod_p_parallel(matrix, n, p, num_threads);
}

/* Glynn only divides by powers of two, so every odd modulus is valid,
 * including composite moduli. This is the full matrix identity; it does
 * not use the Schur cofactor or necklace reductions.
 *
 * Restrict automatic use to the direct-product arithmetic tier. With wide
 * moduli, Ryser's intermediate zero products can avoid costly 128-bit
 * reductions on sparse/structured inputs, while Glynn's signed sums need
 * not vanish. Fewer states therefore do not imply uniformly less work in
 * that tier. Keep Ryser for tiny matrices as well, where setup dominates. */
static int64_t permanent_mod_p(const int64_t *matrix, int n,
                                int64_t p, int num_threads) {
    if (n == 0) return 1 % p;
    if (n == 1) return matrix[0];
    if (n >= 4 && p > 1 && (p & 1) && p <= DIRECT_MUL_MOD_LIMIT) {
        return glynn_permanent_mod_p(matrix, n, p, num_threads);
    }
    return ryser_permanent_mod_p(matrix, n, p, num_threads);
}

/* Keep input conversion and validation identical for the explicit Ryser
 * entry point and the general algorithm-selecting entry point. */
static PyObject *py_matrix_permanent_mod_p(PyObject *args,
                                           int select_algorithm) {
    PyObject *matrix_obj;
    long long p;
    int num_threads = 0;  /* 0 = auto-detect */

    if (!PyArg_ParseTuple(args, "OL|i", &matrix_obj, &p, &num_threads)) {
        return NULL;
    }

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    if (!PyList_Check(matrix_obj)) {
        PyErr_SetString(PyExc_TypeError, "matrix must be a list of lists");
        return NULL;
    }

    Py_ssize_t n = PyList_Size(matrix_obj);
    if (p <= 0) {
        PyErr_SetString(PyExc_ValueError, "p must be a positive signed 64-bit integer");
        return NULL;
    }
    if (n >= 64) {
        PyErr_SetString(PyExc_ValueError, "matrix dimension must be less than 64");
        return NULL;
    }
    if (n == 0) {
        return PyLong_FromLongLong(1 % p);
    }

    int64_t *matrix = (int64_t *)malloc(n * n * sizeof(int64_t));
    if (!matrix) {
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *row = PyList_GetItem(matrix_obj, i);
        if (!PyList_Check(row) || PyList_Size(row) != n) {
            free(matrix);
            PyErr_SetString(PyExc_ValueError, "matrix must be square");
            return NULL;
        }

        for (Py_ssize_t j = 0; j < n; j++) {
            PyObject *item = PyList_GetItem(row, j);
            int64_t value = PyLong_AsLongLong(item);
            if (PyErr_Occurred()) {
                free(matrix);
                return NULL;
            }
            matrix[i * n + j] = normalize_mod_i64(value, (int64_t)p);
        }
    }

    int64_t result;
    Py_BEGIN_ALLOW_THREADS
    result = select_algorithm
        ? permanent_mod_p(matrix, (int)n, (int64_t)p, num_threads)
        : ryser_permanent_mod_p(matrix, (int)n, (int64_t)p, num_threads);
    Py_END_ALLOW_THREADS
    free(matrix);

    if (result < 0) {
        PyErr_NoMemory();
        return NULL;
    }

    return PyLong_FromLongLong(result);
}

static PyObject *py_ryser_permanent_mod_p(PyObject *self, PyObject *args) {
    return py_matrix_permanent_mod_p(args, 0);
}

static PyObject *py_permanent_mod_p(PyObject *self, PyObject *args) {
    return py_matrix_permanent_mod_p(args, 1);
}

/*
 * Build Schur matrix and compute permanent mod p.
 */
static PyObject *py_schur_permanent_mod_p(PyObject *self, PyObject *args) {
    int n;
    long long omega, p;
    int num_threads = 0;

    if (!PyArg_ParseTuple(args, "iLL|i", &n, &omega, &p, &num_threads)) {
        return NULL;
    }

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    if (n < 0 || n >= 64) {
        PyErr_SetString(PyExc_ValueError, "n must satisfy 0 <= n < 64");
        return NULL;
    }
    if (p <= 0) {
        PyErr_SetString(PyExc_ValueError, "p must be a positive signed 64-bit integer");
        return NULL;
    }

    if (n == 0) return PyLong_FromLongLong(1 % p);
    if (n == 1) return PyLong_FromLongLong(1 % p);
    if (p == 1) return PyLong_FromLongLong(0);

    int64_t omega_mod = normalize_mod_i64((int64_t)omega, (int64_t)p);
    int64_t result;

    /*
     * For odd n and omega^n = 1, expansion along the all-ones first row gives
     * n equal permanental cofactors. Glynn on the order-(n-1) (0,0) minor
     * fixes one sign, so this path visits 2^(n-2) states.
     */
    if (n >= 3 && (n & 1) && (p & 1) &&
        pow_mod_i64(omega_mod, (uint64_t)n, (int64_t)p) == 1 % p) {
        int minor_n = n - 1;
        int use_necklace =
            n >= NECKLACE_SERIAL_MIN_ORDER && is_prime_order(n) &&
            (num_threads <= 1 || minor_n <= 16 ||
             n >= NECKLACE_PARALLEL_MIN_ORDER);
        size_t minor_count = (size_t)minor_n * (size_t)minor_n;
        int64_t *minor = (int64_t *)malloc(minor_count * sizeof(int64_t));
        if (!minor) {
            PyErr_NoMemory();
            return NULL;
        }

        if (use_necklace) {
            use_necklace = build_circulant_schur_minor(
                minor, n, omega_mod, (int64_t)p);
        }
        if (!use_necklace) {
            int64_t omega_j = omega_mod;
            for (int row = 0; row < minor_n; row++) {
                int64_t power = omega_j;
                for (int column = 0; column < minor_n; column++) {
                    minor[row * minor_n + column] = power;
                    power = mul_mod_i64(power, omega_j, (int64_t)p);
                }
                omega_j = mul_mod_i64(
                    omega_j, omega_mod, (int64_t)p);
            }
        }

        Py_BEGIN_ALLOW_THREADS
        if (use_necklace) {
            uint64_t represented_words = 0;
            result = necklace_cofactor_mod_p(
                minor, minor_n, (int64_t)p, num_threads, NULL,
                &represented_words);
            if (result < 0 ||
                represented_words != (UINT64_C(1) << minor_n)) {
                result = glynn_permanent_mod_p(
                    minor, minor_n, (int64_t)p, num_threads);
            }
        } else {
            result = glynn_permanent_mod_p(
                minor, minor_n, (int64_t)p, num_threads);
        }
        if (result >= 0) {
            result = mul_mod_i64(result, (int64_t)n % (int64_t)p,
                                 (int64_t)p);
        }
        Py_END_ALLOW_THREADS
        free(minor);
    } else {
        size_t matrix_count = (size_t)n * (size_t)n;
        int64_t *matrix =
            (int64_t *)malloc(matrix_count * sizeof(int64_t));
        if (!matrix) {
            PyErr_NoMemory();
            return NULL;
        }

        int64_t omega_j = 1 % p;
        for (int row = 0; row < n; row++) {
            int64_t power = 1 % p;
            for (int column = 0; column < n; column++) {
                matrix[row * n + column] = power;
                power = mul_mod_i64(power, omega_j, (int64_t)p);
            }
            omega_j = mul_mod_i64(omega_j, omega_mod, (int64_t)p);
        }

        Py_BEGIN_ALLOW_THREADS
        result = ryser_permanent_mod_p(
            matrix, n, (int64_t)p, num_threads);
        Py_END_ALLOW_THREADS
        free(matrix);
    }

    if (result < 0) {
        PyErr_NoMemory();
        return NULL;
    }

    return PyLong_FromLongLong(result);
}

static PyObject *py_schur_necklace_orbit_stats(PyObject *self,
                                                PyObject *args) {
    int n;
    if (!PyArg_ParseTuple(args, "i", &n)) return NULL;
    if (n < 3 || n >= 64 || !(n & 1) || !is_prime_order(n)) {
        PyErr_SetString(
            PyExc_ValueError,
            "n must be an odd prime satisfying 3 <= n < 64");
        return NULL;
    }

    NecklaceOrbitStats stats;
    Py_BEGIN_ALLOW_THREADS
    stats = necklace_orbit_stats(n - 1);
    Py_END_ALLOW_THREADS
    return Py_BuildValue(
        "(KK)", (unsigned long long)stats.representatives,
        (unsigned long long)stats.words);
}

/*
 * Get/set number of threads.
 */
static PyObject *py_get_max_threads(PyObject *self, PyObject *args) {
    return PyLong_FromLong(omp_get_max_threads());
}

static PyObject *py_set_num_threads(PyObject *self, PyObject *args) {
    int num_threads;
    if (!PyArg_ParseTuple(args, "i", &num_threads)) {
        return NULL;
    }
    if (num_threads <= 0) {
        PyErr_SetString(PyExc_ValueError, "num_threads must be positive");
        return NULL;
    }
    omp_set_num_threads(num_threads);
    Py_RETURN_NONE;
}

/* Module method table */
static PyMethodDef PermanentMethods[] = {
    {"permanent_mod_p", py_permanent_mod_p, METH_VARARGS,
     "Compute permanent of integer matrix mod p using Glynn or Ryser.\n"
     "Select Glynn for n >= 4 and odd 1 < p <= 3037000499; otherwise Ryser.\n"
     "Args: matrix, p, [num_threads]"},
    {"ryser_permanent_mod_p", py_ryser_permanent_mod_p, METH_VARARGS,
     "Compute permanent of integer matrix mod p using Ryser's algorithm.\n"
     "Args: matrix, p, [num_threads]"},
    {"schur_permanent_mod_p", py_schur_permanent_mod_p, METH_VARARGS,
     "Compute permanent of n x n Schur matrix mod p.\n"
     "Args: n, omega, p, [num_threads]"},
    {"schur_necklace_orbit_stats", py_schur_necklace_orbit_stats,
     METH_VARARGS,
     "Return (representatives, represented words) for prime Schur order n."},
    {"get_max_threads", py_get_max_threads, METH_NOARGS,
     "Get maximum number of OpenMP threads."},
    {"set_num_threads", py_set_num_threads, METH_VARARGS,
     "Set number of OpenMP threads."},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef permanentmodule = {
    PyModuleDef_HEAD_INIT,
    "perm_c_ext",
    "Fast permanent computation using Ryser and Glynn (OpenMP-parallelized C extension)",
    -1,
    PermanentMethods
};

/* Module initialization */
PyMODINIT_FUNC PyInit_perm_c_ext(void) {
    return PyModule_Create(&permanentmodule);
}
