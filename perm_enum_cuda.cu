/*
 * GPU enumeration of permanents mod p for all n×n matrices over F_p.
 *
 * For the Scheinerman equidistribution conjecture: counts how many
 * matrices have perm ≡ r (mod p) for each residue r ∈ {0,...,p-1}.
 *
 * Each GPU thread handles one matrix (identified by global index).
 * Index → matrix via base-p digit extraction.
 * Perm computed via Ryser formula with Gray code.
 * Results accumulated into shared-memory histogram per block,
 * then atomicAdd to global counts.
 *
 * Compile:
 *   nvcc -shared -Xcompiler -fPIC -O3 -o perm_enum_cuda.so perm_enum_cuda.cu
 * Test binary:
 *   nvcc -O3 -DTEST_MAIN -o perm_enum_cuda perm_enum_cuda.cu
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda_runtime.h>

#define MAX_N 6
#define MAX_P 23

static int is_prime_int(int p) {
    if (p < 2) return 0;
    for (int d = 2; d <= p / d; d++) {
        if (p % d == 0) return 0;
    }
    return 1;
}

/*
 * Kernel: one thread per matrix.
 * Each thread:
 *   1. Converts its global index to an n×n matrix over F_p
 *   2. Computes perm mod p via Ryser/Gray code
 *   3. Increments shared histogram[residue], then atomicAdd to global
 *
 * For n<=6, p<=17, the Ryser loop has at most 2^6=64 subsets,
 * so each thread is very lightweight.
 */
__global__ void enumerate_perm_kernel(int n, int p,
                                       unsigned long long total_matrices,
                                       unsigned long long *global_counts)
{
    /* Shared-memory histogram for this block */
    extern __shared__ unsigned long long shist[];

    int tid = threadIdx.x;

    /* Initialize shared histogram */
    for (int r = tid; r < p; r += blockDim.x) {
        shist[r] = 0;
    }
    __syncthreads();

    /* Grid-stride loop over all matrices */
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + tid;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    for (unsigned long long mat_idx = idx; mat_idx < total_matrices; mat_idx += stride) {
        /* Decode mat_idx into n×n matrix over F_p (base-p digits) */
        int matrix[MAX_N * MAX_N];
        unsigned long long tmp = mat_idx;
        for (int i = 0; i < n * n; i++) {
            matrix[i] = (int)(tmp % p);
            tmp /= p;
        }

        /* Ryser formula with Gray code: perm = (-1)^n Σ_{S⊆[n],S≠∅} (-1)^|S| Π_i (Σ_{j∈S} a_{ij}) */
        int row_sums[MAX_N];
        for (int i = 0; i < n; i++) row_sums[i] = 0;

        long long ryser_sum = 0;
        int num_subsets = 1 << n;

        /* Use simple Gray code iteration */
        int sign = 1;  /* (-1)^|S|, starts at -1 for |S|=1 but we track via sign flip */

        for (int s = 1; s < num_subsets; s++) {
            /* Find column that changed (lowest set bit of s) */
            int col = __ffs(s) - 1;

            /* Gray code: determine if adding or removing */
            int gray = s ^ (s >> 1);
            if ((gray >> col) & 1) {
                for (int row = 0; row < n; row++)
                    row_sums[row] += matrix[row * n + col];
            } else {
                for (int row = 0; row < n; row++)
                    row_sums[row] -= matrix[row * n + col];
            }
            sign = -sign;

            /* Product of row sums mod p */
            long long prod = 1;
            for (int row = 0; row < n; row++) {
                int val = row_sums[row] % p;
                if (val < 0) val += p;
                prod = (prod * val) % p;
            }

            ryser_sum += sign * prod;
        }

        /* (-1)^n factor */
        if (n & 1) ryser_sum = -ryser_sum;

        /* Final mod p */
        int residue = (int)(ryser_sum % p);
        if (residue < 0) residue += p;

        /* Accumulate into shared histogram */
        atomicAdd(&shist[residue], 1ULL);
    }

    __syncthreads();

    /* Flush shared histogram to global counts */
    for (int r = tid; r < p; r += blockDim.x) {
        if (shist[r] > 0)
            atomicAdd(&global_counts[r], shist[r]);
    }
}


/*
 * Host API: enumerate all n×n matrices over F_p and count permanents.
 * counts must point to p elements of long long, zeroed by caller or here.
 * Returns 0 on success, -1 on error.
 */
extern "C" int enumerate_permanent_modp(int n, int p, long long *counts) {
    if (!counts) {
        fprintf(stderr, "enumerate_permanent_modp: counts must not be null\n");
        return -1;
    }
    if (n < 1 || n > MAX_N) {
        fprintf(stderr, "enumerate_permanent_modp: n=%d out of range [1,%d]\n", n, MAX_N);
        return -1;
    }
    if (p < 2 || p > MAX_P) {
        fprintf(stderr, "enumerate_permanent_modp: p=%d out of range [2,%d]\n", p, MAX_P);
        return -1;
    }
    if (!is_prime_int(p)) {
        fprintf(stderr, "enumerate_permanent_modp: p=%d is not prime\n", p);
        return -1;
    }

    /* Total matrices = p^(n²) */
    unsigned long long total = 1;
    for (int i = 0; i < n * n; i++) {
        if (total > ULLONG_MAX / (unsigned long long)p) {
            fprintf(stderr,
                    "enumerate_permanent_modp: p^(n^2) does not fit in uint64 "
                    "for n=%d, p=%d\n", n, p);
            return -1;
        }
        total *= (unsigned long long)p;
    }

    /* Allocate device histogram */
    unsigned long long *d_counts = NULL;
    cudaError_t err = cudaMalloc(&d_counts, p * sizeof(unsigned long long));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMemset(d_counts, 0, p * sizeof(unsigned long long));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_counts);
        return -1;
    }

    /* Choose launch config */
    int threads_per_block = 256;
    /* Use enough blocks to saturate the GPU but cap for small problems */
    unsigned long long total_threads_needed = total;
    int max_blocks = 65535;  /* conservative, works on all GPUs */

    /* For H100: aim for ~100K+ threads for good occupancy */
    int num_blocks;
    if (total_threads_needed <= (unsigned long long)threads_per_block) {
        num_blocks = 1;
    } else {
        unsigned long long nb = (total_threads_needed + threads_per_block - 1) / threads_per_block;
        num_blocks = (nb > (unsigned long long)max_blocks) ? max_blocks : (int)nb;
    }

    /* Shared memory: p * sizeof(unsigned long long) */
    int shared_mem = p * sizeof(unsigned long long);

    /* Launch */
    enumerate_perm_kernel<<<num_blocks, threads_per_block, shared_mem>>>(
        n, p, total, d_counts);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_counts);
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(d_counts);
        return -1;
    }

    /* Copy back */
    unsigned long long h_counts[MAX_P];
    err = cudaMemcpy(
        h_counts, d_counts, p * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_counts);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    for (int r = 0; r < p; r++) {
        if (h_counts[r] > (unsigned long long)LLONG_MAX) {
            fprintf(stderr, "enumerate_permanent_modp: count does not fit in int64\n");
            return -1;
        }
        counts[r] = (long long)h_counts[r];
    }

    return 0;
}


/* ====================================================================
 * Batched sampling kernel: compute perm mod p for many random matrices.
 * One thread per matrix, Ryser with Gray code.
 * For n up to 30 (2^30 subsets per thread).
 * ==================================================================== */

#define MAX_N_SAMPLE 30

__global__ void batch_perm_modp_kernel(
    const int *matrices,   /* batch_size * n * n, row-major, entries in [0,p) */
    int n, int p,
    int batch_size,
    unsigned long long *global_counts)  /* p bins */
{
    extern __shared__ unsigned long long bhist[];

    int tid = threadIdx.x;
    for (int r = tid; r < p; r += blockDim.x)
        bhist[r] = 0;
    __syncthreads();

    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + (uint64_t)tid;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    for (uint64_t mat_idx = idx; mat_idx < (uint64_t)batch_size; mat_idx += stride) {
        const int *mat = matrices + (size_t)mat_idx * (size_t)n * (size_t)n;

        int row_sums[MAX_N_SAMPLE];
        for (int i = 0; i < n; i++) row_sums[i] = 0;

        long long ryser_sum = 0;
        unsigned int num_subsets = 1u << n;
        int sign = 1;

        for (unsigned int s = 1; s < num_subsets; s++) {
            int col = __ffs(s) - 1;
            unsigned int gray = s ^ (s >> 1);
            if ((gray >> col) & 1) {
                for (int row = 0; row < n; row++)
                    row_sums[row] += mat[row * n + col];
            } else {
                for (int row = 0; row < n; row++)
                    row_sums[row] -= mat[row * n + col];
            }
            sign = -sign;

            /* Product with early break on zero row_sum */
            long long prod = 1;
            for (int row = 0; row < n; row++) {
                int val = row_sums[row] % p;
                if (val < 0) val += p;
                if (val == 0) { prod = 0; break; }
                prod = (prod * val) % p;
            }

            ryser_sum += sign * prod;
        }

        if (n & 1) ryser_sum = -ryser_sum;
        int residue = (int)(ryser_sum % p);
        if (residue < 0) residue += p;

        atomicAdd(&bhist[residue], 1ULL);
    }

    __syncthreads();
    for (int r = tid; r < p; r += blockDim.x) {
        if (bhist[r] > 0)
            atomicAdd(&global_counts[r], bhist[r]);
    }
}


/*
 * Host API: compute perm mod p for a batch of matrices and accumulate counts.
 * h_matrices: batch_size * n * n ints, row-major, entries in [0,p).
 * counts: p elements, OUTPUT (zeroed internally).
 * Returns 0 on success.
 */
extern "C" int sample_permanent_modp(
    int n, int p,
    const int *h_matrices,
    int batch_size,
    long long *counts)
{
    if (!h_matrices || !counts) {
        fprintf(stderr, "sample_permanent_modp: input and output pointers must not be null\n");
        return -1;
    }
    if (n < 1 || n > MAX_N_SAMPLE) {
        fprintf(stderr, "sample_permanent_modp: n=%d out of range [1,%d]\n", n, MAX_N_SAMPLE);
        return -1;
    }
    if (p < 2 || p > MAX_P) {
        fprintf(stderr, "sample_permanent_modp: p=%d out of range [2,%d]\n", p, MAX_P);
        return -1;
    }
    if (!is_prime_int(p)) {
        fprintf(stderr, "sample_permanent_modp: p=%d is not prime\n", p);
        return -1;
    }
    if (batch_size <= 0) {
        fprintf(stderr, "sample_permanent_modp: batch_size must be positive\n");
        return -1;
    }

    size_t entries_per_matrix = (size_t)n * (size_t)n;
    if ((size_t)batch_size > SIZE_MAX / entries_per_matrix / sizeof(int)) {
        fprintf(stderr, "sample_permanent_modp: matrix batch is too large\n");
        return -1;
    }
    size_t mat_bytes = (size_t)batch_size * entries_per_matrix * sizeof(int);

    int rc = -1;
    int *d_matrices = NULL;
    unsigned long long *d_counts = NULL;
    unsigned long long h_counts[MAX_P];
    cudaError_t err;
    int threads_per_block = (n <= 16) ? 256 : 128;
    size_t blocks_needed = (
        (size_t)batch_size + (size_t)threads_per_block - 1
    ) / (size_t)threads_per_block;
    int num_blocks = blocks_needed > 65535 ? 65535 : (int)blocks_needed;
    int shared_mem = p * sizeof(unsigned long long);

    err = cudaMalloc(&d_matrices, mat_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc matrices failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    err = cudaMalloc(&d_counts, p * sizeof(unsigned long long));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc counts failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    err = cudaMemcpy(d_matrices, h_matrices, mat_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy matrices failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    err = cudaMemset(d_counts, 0, p * sizeof(unsigned long long));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset counts failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    /* Launch config: fewer threads per block for large n (more registers) */
    batch_perm_modp_kernel<<<num_blocks, threads_per_block, shared_mem>>>(
        d_matrices, n, p, batch_size, d_counts);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    err = cudaMemcpy(
        h_counts, d_counts, p * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy counts failed: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    for (int r = 0; r < p; r++) {
        if (h_counts[r] > (unsigned long long)LLONG_MAX) {
            fprintf(stderr, "sample_permanent_modp: count does not fit in int64\n");
            goto cleanup;
        }
    }
    rc = 0;

cleanup:
    if (d_matrices) {
        err = cudaFree(d_matrices);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaFree matrices failed: %s\n", cudaGetErrorString(err));
            rc = -1;
        }
    }
    if (d_counts) {
        err = cudaFree(d_counts);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaFree counts failed: %s\n", cudaGetErrorString(err));
            rc = -1;
        }
    }
    if (rc == 0) {
        for (int r = 0; r < p; r++) {
            counts[r] = (long long)h_counts[r];
        }
    }
    return rc;
}


#ifdef TEST_MAIN
int main() {
    /* Test n=2, p=3: 3^4 = 81 matrices */
    int n = 2, p = 3;
    long long counts[17] = {0};

    printf("Enumerating %dx%d matrices over F_%d...\n", n, n, p);
    int rc = enumerate_permanent_modp(n, p, counts);
    if (rc != 0) {
        printf("Error!\n");
        return 1;
    }

    long long total = 0;
    for (int r = 0; r < p; r++) {
        printf("  perm ≡ %d: %lld\n", r, counts[r]);
        total += counts[r];
    }
    printf("  Total: %lld (expected %lld)\n", total, (long long)(p * p * p * p));

    /* n=3, p=3 */
    n = 3; p = 3;
    printf("\nEnumerating %dx%d matrices over F_%d...\n", n, n, p);
    rc = enumerate_permanent_modp(n, p, counts);
    if (rc != 0) {
        printf("Error!\n");
        return 1;
    }

    total = 0;
    for (int r = 0; r < p; r++) {
        printf("  perm ≡ %d: %lld\n", r, counts[r]);
        total += counts[r];
    }
    printf("  Total: %lld (expected %lld)\n", total, (long long)19683);

    /* Check nonzero residues are equal */
    if (counts[1] == counts[2]) {
        printf("  ✓ Nonzero residues equal\n");
    } else {
        printf("  ✗ Nonzero residues NOT equal!\n");
    }

    return 0;
}
#endif
