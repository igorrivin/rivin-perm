/*
 * CUDA implementation of fast permanent mod 3 using bipedal F2 encoding.
 *
 * Based on: arXiv:2407.20205 "Fast computation of permanents over F_3 via F_2 arithmetic"
 *
 * Compile with:
 *   nvcc -O3 -o perm_mod3_cuda perm_mod3_cuda.cu
 *
 * Or as Python extension:
 *   nvcc -shared -Xcompiler -fPIC -o perm_mod3_cuda.so perm_mod3_cuda.cu \
 *        $(python3 -c "import numpy; print('-I' + numpy.get_include())") \
 *        $(python3-config --includes)
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

static int cuda_call_succeeded(cudaError_t error, const char *operation) {
    if (error == cudaSuccess) return 1;
    fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
    return 0;
}

/* Bipedal encoding:
 *   0  -> (mag=0, sgn=0)
 *   1  -> (mag=1, sgn=0)
 *  -1  -> (mag=1, sgn=1)
 */

__device__ __forceinline__ void bipedal_add(uint64_t m1, uint64_t s1,
                                             uint64_t m2, uint64_t s2,
                                             uint64_t *mag_out, uint64_t *sgn_out) {
    uint64_t t = m1 ^ s1 ^ s2;
    *mag_out = (m2 & t) | (m1 ^ m2);
    *sgn_out = (m2 & t) ^ s1;
}

__device__ __forceinline__ void bipedal_sub(uint64_t m1, uint64_t s1,
                                             uint64_t m2, uint64_t s2,
                                             uint64_t *mag_out, uint64_t *sgn_out) {
    uint64_t t = s1 ^ s2;
    *mag_out = (m1 & t) | (m1 ^ m2);
    *sgn_out = (m1 & t) ^ (m2 ^ s2);
}

/*
 * Each CUDA thread processes a contiguous block of Gray code states.
 * Uses Gray code iteration within block for O(1) per step.
 */
__global__ void permanent_mod3_kernel(const uint64_t *col_mags,
                                       const uint64_t *col_sgns,
                                       int n,
                                       uint64_t num_subsets,
                                       uint64_t block_size,
                                       uint64_t output_count,
                                       int64_t *partial_sums) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= output_count) return;

    uint64_t start = tid * block_size;
    uint64_t end = start + block_size;
    if (end > num_subsets) end = num_subsets;
    if (start >= num_subsets) start = end;

    /* Skip state 0 (empty subset) */
    if (start == 0) start = 1;

    uint64_t mask = ((uint64_t)1 << n) - 1;
    int64_t local_sum = 0;

    if (start < end) {
        /* Initialize row sums for starting Gray code state */
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
            int prod_sgn = __popcll(row_sgn & mask) & 1;
            int final_sgn = prod_sgn ^ (subset_sign < 0 ? 1 : 0);
            local_sum += (final_sgn == 0) ? 1 : -1;
        }

        /* Gray code iteration within this block */
        for (uint64_t i = start + 1; i < end; i++) {
            uint64_t gray_i = i ^ (i >> 1);
            uint64_t gray_prev = (i - 1) ^ ((i - 1) >> 1);
            uint64_t diff = gray_i ^ gray_prev;
            int j = __ffsll(diff) - 1;  /* Position of changed bit */

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

            if ((row_mag & mask) == mask) {
                int prod_sgn = __popcll(row_sgn & mask) & 1;
                int final_sgn = prod_sgn ^ (subset_sign < 0 ? 1 : 0);
                local_sum += (final_sgn == 0) ? 1 : -1;
            }
        }
    }

    partial_sums[tid] = local_sum;
}

/*
 * Reduction kernel to sum partial results
 */
__global__ void reduce_sum(const int64_t *input, int64_t *output, uint64_t n) {
    extern __shared__ int64_t sdata[];

    uint64_t tid = threadIdx.x;
    uint64_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int64_t sum = 0;
    if (i < n) sum = input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

extern "C" int cuda_runtime_available(void) {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}

/*
 * Host function to compute permanent mod 3
 */
extern "C" int permanent_mod3_cuda(const uint64_t *h_col_mags,
                                    const uint64_t *h_col_sgns,
                                    int n) {
    if (n < 0 || n >= 64) {
        fprintf(stderr, "n must satisfy 0 <= n < 64\n");
        return -1;
    }
    if (n == 0) return 1;
    if (!h_col_mags || !h_col_sgns) {
        fprintf(stderr, "column encodings must not be NULL\n");
        return -1;
    }
    if (n == 1) {
        return (h_col_mags[0] & 1) == 0
            ? 0
            : (((h_col_sgns[0] & 1) == 0) ? 1 : 2);
    }

    uint64_t num_subsets = (uint64_t)1 << n;

    /* Determine grid/block dimensions */
    int threads_per_block = 256;
    uint64_t total_threads;
    uint64_t block_size;

    /* For large n, use more threads with smaller blocks */
    if (n >= 30) {
        total_threads = 1ULL << 20;  /* 1M threads */
        block_size = (num_subsets + total_threads - 1) / total_threads;
    } else if (n >= 20) {
        total_threads = 1ULL << 16;  /* 64K threads */
        block_size = (num_subsets + total_threads - 1) / total_threads;
    } else {
        total_threads = num_subsets;
        block_size = 1;
    }

    uint64_t num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    /* Allocate device memory */
    uint64_t *d_col_mags = NULL;
    uint64_t *d_col_sgns = NULL;
    int64_t *d_partial_sums = NULL;
    int64_t *d_reduce_sums = NULL;
    int64_t *reduce_input = NULL;
    int64_t *reduce_output = NULL;
    uint64_t reduce_size = total_threads;
    int64_t result = 0;
    int return_value = -1;

    if (!cuda_call_succeeded(
            cudaMalloc(&d_col_mags, n * sizeof(uint64_t)),
            "cudaMalloc(col_mags)")) goto cleanup;
    if (!cuda_call_succeeded(
            cudaMalloc(&d_col_sgns, n * sizeof(uint64_t)),
            "cudaMalloc(col_sgns)")) goto cleanup;
    if (!cuda_call_succeeded(
            cudaMalloc(&d_partial_sums, total_threads * sizeof(int64_t)),
            "cudaMalloc(partial_sums)")) goto cleanup;
    if (total_threads > 1 && !cuda_call_succeeded(
            cudaMalloc(&d_reduce_sums, total_threads * sizeof(int64_t)),
            "cudaMalloc(reduce_sums)")) goto cleanup;

    if (!cuda_call_succeeded(
            cudaMemcpy(d_col_mags, h_col_mags, n * sizeof(uint64_t),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(col_mags H2D)")) goto cleanup;
    if (!cuda_call_succeeded(
            cudaMemcpy(d_col_sgns, h_col_sgns, n * sizeof(uint64_t),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(col_sgns H2D)")) goto cleanup;

    /* Launch kernel */
    permanent_mod3_kernel<<<num_blocks, threads_per_block>>>(
        d_col_mags, d_col_sgns, n, num_subsets, block_size, total_threads,
        d_partial_sums);
    if (!cuda_call_succeeded(cudaGetLastError(),
                             "permanent_mod3_kernel launch")) goto cleanup;
    if (!cuda_call_succeeded(cudaDeviceSynchronize(),
                             "permanent_mod3_kernel execution")) goto cleanup;

    /* Reduce partial sums on GPU */
    reduce_input = d_partial_sums;
    reduce_output = d_reduce_sums;
    while (reduce_size > 1) {
        uint64_t reduce_blocks = (reduce_size + threads_per_block * 2 - 1) / (threads_per_block * 2);
        reduce_sum<<<reduce_blocks, threads_per_block, threads_per_block * sizeof(int64_t)>>>(
            reduce_input, reduce_output, reduce_size);
        if (!cuda_call_succeeded(cudaGetLastError(),
                                 "reduce_sum launch")) goto cleanup;
        if (!cuda_call_succeeded(cudaDeviceSynchronize(),
                                 "reduce_sum execution")) goto cleanup;
        reduce_size = reduce_blocks;
        int64_t *tmp = reduce_input;
        reduce_input = reduce_output;
        reduce_output = tmp;
    }

    /* Copy result back */
    if (!cuda_call_succeeded(
            cudaMemcpy(&result, reduce_input, sizeof(int64_t),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(result D2H)")) goto cleanup;

    /* Apply (-1)^n factor */
    if (n & 1) {
        result = -result;
    }

    /* Convert to mod 3 in {0, 1, 2} */
    return_value = (int)(result % 3);
    if (return_value < 0) return_value += 3;

cleanup:
    if (d_col_mags && !cuda_call_succeeded(cudaFree(d_col_mags),
                                            "cudaFree(col_mags)")) {
        return_value = -1;
    }
    if (d_col_sgns && !cuda_call_succeeded(cudaFree(d_col_sgns),
                                            "cudaFree(col_sgns)")) {
        return_value = -1;
    }
    if (d_partial_sums && !cuda_call_succeeded(cudaFree(d_partial_sums),
                                                "cudaFree(partial_sums)")) {
        return_value = -1;
    }
    if (d_reduce_sums && !cuda_call_succeeded(cudaFree(d_reduce_sums),
                                               "cudaFree(reduce_sums)")) {
        return_value = -1;
    }
    return return_value;
}

/*
 * Encode matrix and compute permanent - standalone version
 */
extern "C" int permanent_mod3_matrix(const int *matrix, int n) {
    if (n < 0 || n >= 64) {
        fprintf(stderr, "n must satisfy 0 <= n < 64\n");
        return -1;
    }
    if (n == 0) return 1;
    if (!matrix) {
        fprintf(stderr, "matrix must not be NULL\n");
        return -1;
    }
    if (n == 1) {
        int value = matrix[0] % 3;
        return value < 0 ? value + 3 : value;
    }

    /* Encode columns */
    uint64_t *col_mags = (uint64_t*)malloc(n * sizeof(uint64_t));
    uint64_t *col_sgns = (uint64_t*)malloc(n * sizeof(uint64_t));
    if (!col_mags || !col_sgns) {
        fprintf(stderr, "failed to allocate column encodings\n");
        free(col_mags);
        free(col_sgns);
        return -1;
    }

    for (int j = 0; j < n; j++) {
        uint64_t mag = 0, sgn = 0;
        for (int i = 0; i < n; i++) {
            int val = ((matrix[i * n + j] % 3) + 3) % 3;
            if (val != 0) {
                mag |= ((uint64_t)1 << i);
                if (val == 2) {
                    sgn |= ((uint64_t)1 << i);
                }
            }
        }
        col_mags[j] = mag;
        col_sgns[j] = sgn;
    }

    int result = permanent_mod3_cuda(col_mags, col_sgns, n);

    free(col_mags);
    free(col_sgns);

    return result;
}

/* Simple test main */
#ifdef TEST_MAIN
int main(int argc, char **argv) {
    int n = 30;
    if (argc > 1) n = atoi(argv[1]);

    printf("Testing permanent mod 3 for random %dx%d matrix\n", n, n);

    /* Generate random matrix */
    srand(42);
    int *matrix = (int*)malloc(n * n * sizeof(int));
    for (int i = 0; i < n * n; i++) {
        matrix[i] = rand() % 3;
    }

    /* Time the computation */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int result = permanent_mod3_matrix(matrix, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Result: perm mod 3 = %d\n", result);
    printf("Time: %.3f ms\n", milliseconds);

    free(matrix);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
#endif
