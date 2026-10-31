/*
 * Parallel permanent kernel for a single large matrix (n up to 56+).
 *
 * Uses Glynn's formula (2010), which naturally iterates over 2^(n-1)
 * sign vectors — a 2x speedup over Ryser's 2^n subsets.
 *
 * Glynn's formula:
 *   perm(A) = (1/2^{n-1}) sum_{δ: δ_{n-1}=1} (∏_j δ_j) ∏_i (∑_j δ_j a_{ij})
 *
 * where δ ∈ {±1}^n with one component fixed at +1.  We fix δ_{n-1}=+1 and
 * iterate over 2^{n-1} sign patterns for columns 0..n-2 via Gray code.
 *
 * Each thread handles a contiguous block of the 2^{n-1} iterations with
 * Kahan-compensated accumulation.  Templated for real vs complex.
 *
 * Compile library:
 *   nvcc -shared -Xcompiler -fPIC -O3 -arch=all-major \
 *        -o perm_parallel_cuda.so perm_parallel_cuda.cu
 * Test binary:
 *   nvcc -O3 -DTEST_MAIN -arch=all-major \
 *        -o perm_parallel_cuda_test perm_parallel_cuda.cu -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK_GOTO(call, label) do {                                  \
    cudaError_t cuda_check_error__ = (call);                               \
    if (cuda_check_error__ != cudaSuccess) {                               \
        fprintf(stderr, "%s failed: %s\n", #call,                         \
                cudaGetErrorString(cuda_check_error__));                   \
        goto label;                                                        \
    }                                                                      \
} while (0)

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

__device__ __forceinline__ uint64_t gray_code(uint64_t x) {
    return x ^ (x >> 1);
}

/* ------------------------------------------------------------------ */
/*  Glynn permanent kernel                                            */
/* ------------------------------------------------------------------ */

/*
 * Each thread processes sign-vector indices s in [start, end).
 * total = 2^(n-1).  start = tid * iters_per_thread (0-based).
 *
 * For index s, the sign vector δ has:
 *   δ_j = +1 if bit j of gray(s) is 0,  -1 if bit j is 1  (j=0..n-2)
 *   δ_{n-1} = +1 always
 *
 * Row sums: rs_i = sum_j δ_j * a_{ij}
 * Sign factor: prod(δ_j) = (-1)^{popcount(gray(s))} = (-1)^s
 *
 * Gray-code update: when bit col flips in gray(s),
 *   if bit becomes 1 (δ_col: +1 → -1): rs_i -= 2*a_{i,col}
 *   if bit becomes 0 (δ_col: -1 → +1): rs_i += 2*a_{i,col}
 *
 * Host divides final sum by 2^{n-1}.  No (-1)^n correction needed.
 */
template <bool IS_COMPLEX>
__global__ void permanent_glynn_kernel(
    const double * __restrict__ matrix_re,   /* n×n row-major */
    const double * __restrict__ matrix_im,   /* n×n row-major (NULL if real) */
    int n,
    uint64_t total,            /* 2^(n-1) */
    uint64_t iters_per_thread,
    double * __restrict__ partial_re,
    double * __restrict__ partial_im)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t start = tid * iters_per_thread;  /* 0-based */
    if (start >= total) {
        partial_re[tid] = 0.0;
        if (IS_COMPLEX) partial_im[tid] = 0.0;
        return;
    }
    uint64_t end = start + iters_per_thread;
    if (end > total) end = total;

    int ncols = n - 1;  /* variable sign bits: columns 0..n-2 */

    /* Compute full row sums: rs[i] = sum_{j=0}^{n-1} a_{ij} */
    double rs_re[64];
    double rs_im[64];
    for (int i = 0; i < n; i++) {
        double s_re = 0.0;
        double s_im = 0.0;
        for (int j = 0; j < n; j++) {
            s_re += __ldg(&matrix_re[i * n + j]);
            if (IS_COMPLEX) s_im += __ldg(&matrix_im[i * n + j]);
        }
        rs_re[i] = s_re;
        if (IS_COMPLEX) rs_im[i] = s_im;
    }

    /* Adjust for gray(start): subtract 2*column_j for each set bit j */
    uint64_t g_prev = gray_code(start);
    for (int c = 0; c < ncols; c++) {
        if ((g_prev >> c) & 1ULL) {
            for (int i = 0; i < n; i++) {
                rs_re[i] -= 2.0 * __ldg(&matrix_re[i * n + c]);
                if (IS_COMPLEX) rs_im[i] -= 2.0 * __ldg(&matrix_im[i * n + c]);
            }
        }
    }

    /* Kahan-compensated accumulator */
    double sum_re = 0.0, comp_re = 0.0;
    double sum_im = 0.0, comp_im = 0.0;

    /* Sign for index s: (-1)^s */
    int sign_val = (start & 1) ? -1 : 1;

    /* Process starting index */
    {
        double prod_re = rs_re[0];
        double prod_im;
        if (IS_COMPLEX) {
            prod_im = rs_im[0];
            for (int i = 1; i < n; i++) {
                double new_re = prod_re * rs_re[i] - prod_im * rs_im[i];
                double new_im = prod_re * rs_im[i] + prod_im * rs_re[i];
                prod_re = new_re;
                prod_im = new_im;
            }
        } else {
            for (int i = 1; i < n; i++) {
                prod_re *= rs_re[i];
            }
        }

        {
            double y = __dsub_rn(sign_val * prod_re, comp_re);
            double t = __dadd_rn(sum_re, y);
            comp_re = __dsub_rn(__dsub_rn(t, sum_re), y);
            sum_re = t;
        }
        if (IS_COMPLEX) {
            double y = __dsub_rn(sign_val * prod_im, comp_im);
            double t = __dadd_rn(sum_im, y);
            comp_im = __dsub_rn(__dsub_rn(t, sum_im), y);
            sum_im = t;
        }
    }

    /* Gray-code iterate for s = start+1 .. end-1 */
    for (uint64_t s = start + 1; s < end; s++) {
        /* Column that flips: position of lowest set bit of s */
        int col = __ffsll((long long)s) - 1;

        /* Direction: check if bit col is set in gray(s).
         * If set (δ_col = -1): subtract 2*column
         * If clear (δ_col = +1): add 2*column */
        uint64_t g = gray_code(s);
        if ((g >> col) & 1ULL) {
            for (int i = 0; i < n; i++) {
                rs_re[i] -= 2.0 * __ldg(&matrix_re[i * n + col]);
                if (IS_COMPLEX) rs_im[i] -= 2.0 * __ldg(&matrix_im[i * n + col]);
            }
        } else {
            for (int i = 0; i < n; i++) {
                rs_re[i] += 2.0 * __ldg(&matrix_re[i * n + col]);
                if (IS_COMPLEX) rs_im[i] += 2.0 * __ldg(&matrix_im[i * n + col]);
            }
        }

        sign_val = -sign_val;

        double prod_re = rs_re[0];
        double prod_im;
        if (IS_COMPLEX) {
            prod_im = rs_im[0];
            for (int i = 1; i < n; i++) {
                double new_re = prod_re * rs_re[i] - prod_im * rs_im[i];
                double new_im = prod_re * rs_im[i] + prod_im * rs_re[i];
                prod_re = new_re;
                prod_im = new_im;
            }
        } else {
            for (int i = 1; i < n; i++) {
                prod_re *= rs_re[i];
            }
        }

        {
            double y = __dsub_rn(sign_val * prod_re, comp_re);
            double t = __dadd_rn(sum_re, y);
            comp_re = __dsub_rn(__dsub_rn(t, sum_re), y);
            sum_re = t;
        }
        if (IS_COMPLEX) {
            double y = __dsub_rn(sign_val * prod_im, comp_im);
            double t = __dadd_rn(sum_im, y);
            comp_im = __dsub_rn(__dsub_rn(t, sum_im), y);
            sum_im = t;
        }
    }

    partial_re[tid] = sum_re;
    if (IS_COMPLEX) partial_im[tid] = sum_im;
}

/* ------------------------------------------------------------------ */
/*  Reduction kernel                                                  */
/* ------------------------------------------------------------------ */

template <bool IS_COMPLEX>
__global__ void reduce_kernel(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    uint64_t count,
    double * __restrict__ out_re,
    double * __restrict__ out_im)
{
    extern __shared__ double sdata[];
    double *s_re = sdata;
    double *s_im = IS_COMPLEX ? (sdata + blockDim.x) : nullptr;

    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;

    s_re[lid] = (gid < count) ? in_re[gid] : 0.0;
    if (IS_COMPLEX) s_im[lid] = (gid < count) ? in_im[gid] : 0.0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            s_re[lid] = __dadd_rn(s_re[lid], s_re[lid + stride]);
            if (IS_COMPLEX) {
                s_im[lid] = __dadd_rn(s_im[lid], s_im[lid + stride]);
            }
        }
        __syncthreads();
    }

    if (lid == 0) {
        out_re[blockIdx.x] = s_re[0];
        if (IS_COMPLEX) out_im[blockIdx.x] = s_im[0];
    }
}

/* ------------------------------------------------------------------ */
/*  Host-side multi-pass reduction                                    */
/* ------------------------------------------------------------------ */

static int reduce_host(
    double *d_data_re, double *d_data_im,
    uint64_t count, bool is_complex,
    double *result_re, double *result_im)
{
    const int RED_BLOCK = 256;
    double *d_scratch1_re = nullptr, *d_scratch1_im = nullptr;
    double *d_scratch2_re = nullptr, *d_scratch2_im = nullptr;
    double *d_in_re = d_data_re, *d_in_im = d_data_im;
    double *d_out_re = nullptr, *d_out_im = nullptr;
    uint64_t cur_count = count;
    uint64_t max_blocks = (count + RED_BLOCK - 1) / RED_BLOCK;
    int status = -1;

    if (!d_data_re || !result_re || count == 0 ||
        (is_complex && (!d_data_im || !result_im))) {
        fprintf(stderr, "reduce_host: invalid argument\n");
        return -1;
    }

    if (count > 1) {
        CUDA_CHECK_GOTO(
            cudaMalloc(&d_scratch1_re, max_blocks * sizeof(double)), cleanup);
        if (is_complex) {
            CUDA_CHECK_GOTO(
                cudaMalloc(&d_scratch1_im, max_blocks * sizeof(double)), cleanup);
        }
        if (max_blocks > 1) {
            CUDA_CHECK_GOTO(
                cudaMalloc(&d_scratch2_re, max_blocks * sizeof(double)), cleanup);
            if (is_complex) {
                CUDA_CHECK_GOTO(
                    cudaMalloc(&d_scratch2_im, max_blocks * sizeof(double)), cleanup);
            }
        }
        d_out_re = d_scratch1_re;
        d_out_im = d_scratch1_im;
    }

    while (cur_count > 1) {
        uint64_t num_blocks = (cur_count + RED_BLOCK - 1) / RED_BLOCK;

        size_t smem = is_complex ? 2 * RED_BLOCK * sizeof(double)
                                 : RED_BLOCK * sizeof(double);

        if (is_complex) {
            reduce_kernel<true><<<(int)num_blocks, RED_BLOCK, smem>>>(
                d_in_re, d_in_im, cur_count, d_out_re, d_out_im);
        } else {
            reduce_kernel<false><<<(int)num_blocks, RED_BLOCK, smem>>>(
                d_in_re, nullptr, cur_count, d_out_re, nullptr);
        }
        CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

        d_in_re = d_out_re;
        d_in_im = d_out_im;
        if (d_out_re == d_scratch1_re) {
            d_out_re = d_scratch2_re;
            d_out_im = d_scratch2_im;
        } else {
            d_out_re = d_scratch1_re;
            d_out_im = d_scratch1_im;
        }
        cur_count = num_blocks;
    }

    CUDA_CHECK_GOTO(cudaDeviceSynchronize(), cleanup);
    CUDA_CHECK_GOTO(
        cudaMemcpy(result_re, d_in_re, sizeof(double), cudaMemcpyDeviceToHost), cleanup);
    if (is_complex) {
        CUDA_CHECK_GOTO(
            cudaMemcpy(result_im, d_in_im, sizeof(double), cudaMemcpyDeviceToHost), cleanup);
    }
    status = 0;

cleanup:
    if (d_scratch1_re) cudaFree(d_scratch1_re);
    if (d_scratch1_im) cudaFree(d_scratch1_im);
    if (d_scratch2_re) cudaFree(d_scratch2_re);
    if (d_scratch2_im) cudaFree(d_scratch2_im);
    return status;
}

/* ------------------------------------------------------------------ */
/*  C interface for Python ctypes                                     */
/* ------------------------------------------------------------------ */

extern "C" {

int permanent_parallel_real(const double *matrix_re, int n, double *result_out)
{
    uint64_t total;
    uint64_t iters_per_thread;
    int device = 0, num_sms = 0;
    int threads_per_block = 128, total_threads = 0, num_blocks = 0;
    size_t mat_size, partial_size;
    double *d_matrix_re = nullptr, *d_partial_re = nullptr;
    double result = 0.0;
    int status = -1;

    if (!matrix_re || !result_out) {
        fprintf(stderr, "permanent_parallel_real: null argument\n");
        return -1;
    }
    if (n <= 0 || n > 63) {
        fprintf(stderr, "permanent_parallel_real: n=%d out of range [1,63]\n", n);
        return -1;
    }
    if (n == 1) {
        *result_out = matrix_re[0];
        return 0;
    }

    total = 1ULL << (n - 1);  /* Glynn: 2^(n-1) sign vectors */
    CUDA_CHECK_GOTO(cudaGetDevice(&device), cleanup);
    CUDA_CHECK_GOTO(
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device),
        cleanup);
    if (n <= 20) {
        total_threads = 1024;
    } else if (n <= 30) {
        total_threads = num_sms * 128;
    } else {
        total_threads = num_sms * 256;
    }
    num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    total_threads = num_blocks * threads_per_block;

    iters_per_thread = (total + (uint64_t)total_threads - 1)
                       / (uint64_t)total_threads;

    mat_size = (size_t)n * n * sizeof(double);
    partial_size = (size_t)total_threads * sizeof(double);

    CUDA_CHECK_GOTO(cudaMalloc(&d_matrix_re, mat_size), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(&d_partial_re, partial_size), cleanup);
    CUDA_CHECK_GOTO(
        cudaMemcpy(d_matrix_re, matrix_re, mat_size, cudaMemcpyHostToDevice), cleanup);

    permanent_glynn_kernel<false><<<num_blocks, threads_per_block>>>(
        d_matrix_re, nullptr, n, total, iters_per_thread,
        d_partial_re, nullptr);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    CUDA_CHECK_GOTO(cudaDeviceSynchronize(), cleanup);

    if (reduce_host(d_partial_re, nullptr, (uint64_t)total_threads,
                    false, &result, nullptr) != 0) {
        goto cleanup;
    }

    /* Glynn: divide by 2^{n-1} */
    result = ldexp(result, -(n - 1));

    *result_out = result;
    status = 0;

cleanup:
    if (d_matrix_re) cudaFree(d_matrix_re);
    if (d_partial_re) cudaFree(d_partial_re);
    return status;
}

int permanent_parallel_complex(const double *matrix_re, const double *matrix_im,
                               int n, double *result_re_out, double *result_im_out)
{
    uint64_t total;
    uint64_t iters_per_thread;
    int device = 0, num_sms = 0;
    int threads_per_block = 128, total_threads = 0, num_blocks = 0;
    size_t mat_size, partial_size;
    double *d_mat_re = nullptr, *d_mat_im = nullptr;
    double *d_partial_re = nullptr, *d_partial_im = nullptr;
    double res_re = 0.0, res_im = 0.0;
    int status = -1;

    if (!matrix_re || !matrix_im || !result_re_out || !result_im_out) {
        fprintf(stderr, "permanent_parallel_complex: null argument\n");
        return -1;
    }
    if (n <= 0 || n > 63) {
        fprintf(stderr, "permanent_parallel_complex: n=%d out of range [1,63]\n", n);
        return -1;
    }
    if (n == 1) {
        *result_re_out = matrix_re[0];
        *result_im_out = matrix_im[0];
        return 0;
    }

    total = 1ULL << (n - 1);
    CUDA_CHECK_GOTO(cudaGetDevice(&device), cleanup);
    CUDA_CHECK_GOTO(
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device),
        cleanup);
    if (n <= 20) {
        total_threads = 1024;
    } else if (n <= 30) {
        total_threads = num_sms * 64;
    } else {
        total_threads = num_sms * 128;
    }
    num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    total_threads = num_blocks * threads_per_block;

    iters_per_thread = (total + (uint64_t)total_threads - 1)
                       / (uint64_t)total_threads;

    mat_size = (size_t)n * n * sizeof(double);
    partial_size = (size_t)total_threads * sizeof(double);

    CUDA_CHECK_GOTO(cudaMalloc(&d_mat_re, mat_size), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(&d_mat_im, mat_size), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(&d_partial_re, partial_size), cleanup);
    CUDA_CHECK_GOTO(cudaMalloc(&d_partial_im, partial_size), cleanup);

    CUDA_CHECK_GOTO(
        cudaMemcpy(d_mat_re, matrix_re, mat_size, cudaMemcpyHostToDevice), cleanup);
    CUDA_CHECK_GOTO(
        cudaMemcpy(d_mat_im, matrix_im, mat_size, cudaMemcpyHostToDevice), cleanup);

    permanent_glynn_kernel<true><<<num_blocks, threads_per_block>>>(
        d_mat_re, d_mat_im, n, total, iters_per_thread,
        d_partial_re, d_partial_im);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    CUDA_CHECK_GOTO(cudaDeviceSynchronize(), cleanup);

    if (reduce_host(d_partial_re, d_partial_im, (uint64_t)total_threads,
                    true, &res_re, &res_im) != 0) {
        goto cleanup;
    }

    res_re = ldexp(res_re, -(n - 1));
    res_im = ldexp(res_im, -(n - 1));

    *result_re_out = res_re;
    *result_im_out = res_im;
    status = 0;

cleanup:
    if (d_mat_re) cudaFree(d_mat_re);
    if (d_mat_im) cudaFree(d_mat_im);
    if (d_partial_re) cudaFree(d_partial_re);
    if (d_partial_im) cudaFree(d_partial_im);
    return status;
}

int get_gpu_info(int *num_sms, int *max_threads_per_sm) {
    int device = 0;
    cudaError_t err;
    if (!num_sms || !max_threads_per_sm) return -1;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) return -1;
    err = cudaDeviceGetAttribute(num_sms, cudaDevAttrMultiProcessorCount, device);
    if (err != cudaSuccess) return -1;
    err = cudaDeviceGetAttribute(
        max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device);
    if (err != cudaSuccess) return -1;
    return 0;
}

}  /* extern "C" */


#ifdef TEST_MAIN
#include <string.h>

static double factorial(int n) {
    double f = 1.0;
    for (int i = 2; i <= n; i++) f *= i;
    return f;
}

int main() {
    printf("=== Parallel Glynn Permanent Test ===\n\n");

    /* Test 1: Identity matrix (perm = 1) */
    {
        int n = 4;
        double mat[16];
        memset(mat, 0, sizeof(mat));
        for (int i = 0; i < n; i++) mat[i * n + i] = 1.0;

        double result;
        int rc = permanent_parallel_real(mat, n, &result);
        printf("Identity %dx%d: perm = %.6f (expected 1.0) [%s]\n",
               n, n, result, (rc == 0 && fabs(result - 1.0) < 1e-10) ? "PASS" : "FAIL");
    }

    /* Test 2: All-ones matrix (perm = n!) */
    for (int n = 2; n <= 15; n++) {
        double *mat = (double *)malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++) mat[i] = 1.0;

        double result;
        permanent_parallel_real(mat, n, &result);
        double expected = factorial(n);
        double relerr = fabs(result - expected) / expected;
        printf("Ones %2dx%2d: perm = %.1f, n! = %.1f, relerr = %.2e [%s]\n",
               n, n, result, expected, relerr,
               relerr < 1e-10 ? "PASS" : "FAIL");
        free(mat);
    }

    /* Test 3: Complex identity (perm = 1+0i) */
    {
        int n = 4;
        double re[16], im[16];
        memset(re, 0, sizeof(re));
        memset(im, 0, sizeof(im));
        for (int i = 0; i < n; i++) re[i * n + i] = 1.0;

        double res_re, res_im;
        permanent_parallel_complex(re, im, n, &res_re, &res_im);
        printf("\nComplex identity 4x4: perm = %.6f + %.6fi (expected 1+0i) [%s]\n",
               res_re, res_im,
               (fabs(res_re - 1.0) < 1e-10 && fabs(res_im) < 1e-10) ? "PASS" : "FAIL");
    }

    /* Test 4: DFT(5)/sqrt(5) */
    {
        int n = 5;
        double re[25], im[25];
        double sqrtn = sqrt((double)n);
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                double angle = 2.0 * M_PI * j * k / n;
                re[j * n + k] = cos(angle) / sqrtn;
                im[j * n + k] = sin(angle) / sqrtn;
            }
        }
        double res_re, res_im;
        permanent_parallel_complex(re, im, n, &res_re, &res_im);
        double absperm = sqrt(res_re * res_re + res_im * res_im);
        printf("DFT(5)/sqrt(5): |perm| = %.10f, perm = %.10f + %.10fi\n",
               absperm, res_re, res_im);
    }

    /* Test 5: Larger all-ones with timing */
    for (int n = 20; n <= 30; n += 5) {
        double *mat = (double *)malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++) mat[i] = 1.0;

        double result;
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        permanent_parallel_real(mat, n, &result);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms;
        cudaEventElapsedTime(&ms, t0, t1);

        double expected = factorial(n);
        double relerr = fabs(result - expected) / fabs(expected);
        printf("Ones %2dx%2d: perm = %.6e, n! = %.6e, relerr = %.2e, time = %.3f s [%s]\n",
               n, n, result, expected, relerr, ms / 1000.0,
               relerr < 1e-6 ? "PASS" : "FAIL");
        free(mat);
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }

    printf("\nDone.\n");
    return 0;
}
#endif
