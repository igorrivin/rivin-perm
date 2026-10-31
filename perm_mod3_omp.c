/*
 * Fast permanent mod 3 using bipedal F2 encoding with OpenMP parallelization.
 *
 * Based on: arXiv:2407.20205 "Fast computation of permanents over F_3 via F_2 arithmetic"
 *
 * Encoding: F_3 = {0, 1, -1} -> (mag, sgn) in F_2 x F_2
 *   0  -> (0, 0)
 *   1  -> (1, 0)
 *   -1 -> (1, 1)
 *
 * For n < 64, pack all n components into 64-bit integers.  The enumeration
 * uses 1ULL << n, so order 64 cannot be represented by the loop bound.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <omp.h>

/* Bipedal addition: (m1,s1) + (m2,s2) */
static inline void bipedal_add(uint64_t m1, uint64_t s1,
                                uint64_t m2, uint64_t s2,
                                uint64_t *mag_out, uint64_t *sgn_out) {
    uint64_t t = m1 ^ s1 ^ s2;
    *mag_out = (m2 & t) | (m1 ^ m2);
    *sgn_out = (m2 & t) ^ s1;
}

/* Bipedal subtraction: (m1,s1) - (m2,s2) */
static inline void bipedal_sub(uint64_t m1, uint64_t s1,
                                uint64_t m2, uint64_t s2,
                                uint64_t *mag_out, uint64_t *sgn_out) {
    uint64_t t = s1 ^ s2;
    *mag_out = (m1 & t) | (m1 ^ m2);
    *sgn_out = (m1 & t) ^ (m2 ^ s2);
}

/* Count set bits (popcount) */
static inline int popcount64(uint64_t x) {
    return __builtin_popcountll(x);
}

/*
 * Parallel Ryser algorithm with bipedal encoding.
 *
 * Key optimization from arXiv:2407.20205 Section 3.1:
 * - Split 2^n states into contiguous Gray code blocks
 * - Each thread initializes row sums for its starting state (O(n))
 * - Then uses Gray code iteration within its block (O(1) per step)
 *
 * This gives nearly perfect parallelization!
 */
static int permanent_mod3_parallel(uint64_t *col_mags, uint64_t *col_sgns,
                                    int n, int num_threads) {
    uint64_t mask = ((uint64_t)1 << n) - 1;
    uint64_t num_subsets = (uint64_t)1 << n;

    int64_t global_sum = 0;

    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }

    #pragma omp parallel num_threads(num_threads) reduction(+:global_sum)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        /* Each thread handles a contiguous block of Gray code states */
        uint64_t block_size = (num_subsets + nthreads - 1) / nthreads;
        uint64_t start = (uint64_t)tid * block_size;
        uint64_t end = start + block_size;
        if (end > num_subsets) end = num_subsets;
        if (start >= num_subsets) start = end;  /* Empty block */

        /* Skip state 0 (empty subset) */
        if (start == 0) start = 1;

        int64_t local_sum = 0;

        if (start < end) {
            /* Initialize row sums for the starting Gray code state */
            uint64_t gray_start = start ^ (start >> 1);
            uint64_t row_mag = 0;
            uint64_t row_sgn = 0;
            int popcount_start = 0;

            for (int j = 0; j < n; j++) {
                if ((gray_start >> j) & 1) {
                    uint64_t new_mag, new_sgn;
                    bipedal_add(row_mag, row_sgn, col_mags[j], col_sgns[j],
                               &new_mag, &new_sgn);
                    row_mag = new_mag;
                    row_sgn = new_sgn;
                    popcount_start++;
                }
            }

            /* Process starting state */
            int subset_sign = (popcount_start & 1) ? -1 : 1;

            if ((row_mag & mask) == mask) {
                int prod_sgn = popcount64(row_sgn & mask) & 1;
                int final_sgn = prod_sgn ^ (subset_sign < 0 ? 1 : 0);
                local_sum += (final_sgn == 0) ? 1 : -1;
            }

            /* Gray code iteration within this block */
            for (uint64_t i = start + 1; i < end; i++) {
                /* Find which bit flipped from i-1 to i in Gray code */
                /* Gray(i) = i ^ (i >> 1) */
                /* Gray(i) ^ Gray(i-1) has exactly one bit set */
                uint64_t gray_i = i ^ (i >> 1);
                uint64_t gray_prev = (i - 1) ^ ((i - 1) >> 1);
                uint64_t diff = gray_i ^ gray_prev;
                int j = __builtin_ctzll(diff);  /* Position of changed bit */

                uint64_t new_mag, new_sgn;
                if ((gray_i >> j) & 1) {
                    /* Bit j turned on: add column j */
                    bipedal_add(row_mag, row_sgn, col_mags[j], col_sgns[j],
                               &new_mag, &new_sgn);
                    subset_sign = -subset_sign;
                } else {
                    /* Bit j turned off: subtract column j */
                    bipedal_sub(row_mag, row_sgn, col_mags[j], col_sgns[j],
                               &new_mag, &new_sgn);
                    subset_sign = -subset_sign;
                }
                row_mag = new_mag;
                row_sgn = new_sgn;

                /* Check if product is nonzero */
                if ((row_mag & mask) == mask) {
                    int prod_sgn = popcount64(row_sgn & mask) & 1;
                    int final_sgn = prod_sgn ^ (subset_sign < 0 ? 1 : 0);
                    local_sum += (final_sgn == 0) ? 1 : -1;
                }
            }
        }

        global_sum += local_sum;
    }

    /* Apply (-1)^n factor */
    if (n & 1) {
        global_sum = -global_sum;
    }

    /* Convert to mod 3 in {0, 1, 2} */
    int result = (int)(global_sum % 3);
    if (result < 0) result += 3;

    return result;
}

/*
 * Serial Gray code version for comparison/small matrices.
 */
static int permanent_mod3_serial(uint64_t *col_mags, uint64_t *col_sgns, int n) {
    uint64_t mask = ((uint64_t)1 << n) - 1;

    uint64_t row_mag = 0;
    uint64_t row_sgn = 0;
    int64_t result = 0;
    int subset_sign = 1;

    for (uint64_t i = 1; i < ((uint64_t)1 << n); i++) {
        /* Find lowest set bit position */
        int j = __builtin_ctzll(i);

        /* Gray code */
        uint64_t gray_i = i ^ (i >> 1);

        uint64_t new_mag, new_sgn;
        if ((gray_i >> j) & 1) {
            bipedal_add(row_mag, row_sgn, col_mags[j], col_sgns[j],
                       &new_mag, &new_sgn);
        } else {
            bipedal_sub(row_mag, row_sgn, col_mags[j], col_sgns[j],
                       &new_mag, &new_sgn);
        }
        row_mag = new_mag;
        row_sgn = new_sgn;

        subset_sign = -subset_sign;

        /* Check if product is nonzero */
        if ((row_mag & mask) == mask) {
            int prod_sgn = popcount64(row_sgn & mask) & 1;

            /* Multiply by subset_sign */
            if (subset_sign < 0) {
                prod_sgn = 1 - prod_sgn;
            }

            /* Add to result */
            if (prod_sgn == 0) {
                result += 1;
            } else {
                result -= 1;
            }
        }
    }

    /* Apply (-1)^n */
    if (n & 1) {
        result = -result;
    }

    int r = (int)(result % 3);
    if (r < 0) r += 3;
    return r;
}

/*
 * Python interface: permanent_mod3(matrix, num_threads=-1)
 */
static PyObject* py_permanent_mod3(PyObject *self, PyObject *args) {
    PyArrayObject *matrix;
    int num_threads = -1;

    if (!PyArg_ParseTuple(args, "O!|i", &PyArray_Type, &matrix, &num_threads)) {
        return NULL;
    }

    /* Check dimensions */
    if (PyArray_NDIM(matrix) != 2) {
        PyErr_SetString(PyExc_ValueError, "Matrix must be 2-dimensional");
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(matrix);
    if (dims[0] != dims[1]) {
        PyErr_SetString(PyExc_ValueError, "Matrix must be square");
        return NULL;
    }

    if (dims[0] >= 64) {
        PyErr_SetString(PyExc_ValueError, "Matrix size must be less than 64");
        return NULL;
    }

    int n = (int)dims[0];

    if (n == 0) {
        return PyLong_FromLong(1);
    }

    /* Convert matrix to contiguous int64 array */
    PyArrayObject *mat_contig = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)matrix, NPY_INT64, 2, 2);
    if (mat_contig == NULL) {
        return NULL;
    }

    int64_t *data = (int64_t*)PyArray_DATA(mat_contig);

    /* Encode columns as bipedal pairs */
    uint64_t col_mags[64];
    uint64_t col_sgns[64];

    for (int j = 0; j < n; j++) {
        uint64_t mag = 0;
        uint64_t sgn = 0;

        for (int i = 0; i < n; i++) {
            int64_t val = data[i * n + j];
            /* Reduce mod 3 */
            int v = (int)((val % 3 + 3) % 3);

            if (v != 0) {
                mag |= ((uint64_t)1 << i);
                if (v == 2) {  /* -1 in F_3 */
                    sgn |= ((uint64_t)1 << i);
                }
            }
        }

        col_mags[j] = mag;
        col_sgns[j] = sgn;
    }

    Py_DECREF(mat_contig);

    int result;

    /* Choose serial vs parallel based on size and thread count */
    int actual_threads = (num_threads <= 0) ? omp_get_max_threads() : num_threads;

    Py_BEGIN_ALLOW_THREADS
    if (n <= 20 || actual_threads == 1) {
        result = permanent_mod3_serial(col_mags, col_sgns, n);
    } else {
        result = permanent_mod3_parallel(col_mags, col_sgns, n, num_threads);
    }
    Py_END_ALLOW_THREADS

    return PyLong_FromLong(result);
}

/* Module methods */
static PyMethodDef methods[] = {
    {"permanent_mod3", py_permanent_mod3, METH_VARARGS,
     "Compute permanent of integer matrix mod 3 using bipedal encoding.\n\n"
     "Args:\n"
     "    matrix: n x n integer matrix (n < 64)\n"
     "    num_threads: number of threads (-1 for all available)\n\n"
     "Returns:\n"
     "    perm(matrix) mod 3, in {0, 1, 2}"},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "perm_mod3_omp",
    "Fast permanent mod 3 with OpenMP parallelization",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_perm_mod3_omp(void) {
    import_array();
    return PyModule_Create(&module);
}
