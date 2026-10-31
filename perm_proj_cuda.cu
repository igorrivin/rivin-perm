/*
 * Projective multiset enumeration of 4×4 permanents mod p.
 *
 * Exploits row-permutation and row-scaling invariance of perm=0:
 *   N_0^perm(4,p) = L_row(4,p) + (p-1)^4 × W_0
 * where W_0 is the weighted count of projective row multisets with perm≡0.
 *
 * Projective rows: nonzero vectors in F_p^4 with first nonzero entry = 1.
 * Count: R = (p^4-1)/(p-1) = p^3+p^2+p+1.
 *
 * Multisets of size 4: C(R+3, 4) total.
 * Each thread decodes its index to (i0 ≤ i1 ≤ i2 ≤ i3),
 * computes perm via fully unrolled Ryser, and if perm=0,
 * adds multinomial weight 4!/(m1!...mk!) to W_0.
 *
 * Compile:
 *   nvcc -shared -Xcompiler -fPIC -O3 -o perm_proj_cuda.so perm_proj_cuda.cu
 * Test:
 *   nvcc -O3 -DTEST_MAIN -o perm_proj_cuda perm_proj_cuda.cu
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <cuda_runtime.h>

#define MAX_P 97
#define N_COLS 4

static int host_is_prime(int p) {
    if (p < 2) return 0;
    for (int d = 2; d <= p / d; d++) {
        if (p % d == 0) return 0;
    }
    return 1;
}

/* ----------------------------------------------------------------
 * Device: decode multiset index → (i0 ≤ i1 ≤ i2 ≤ i3)
 *
 * Bijection: multiset {i0,i1,i2,i3} ↔ combination {j0<j1<j2<j3}
 * where jk = ik + k.  Domain size = C(R+3, 4).
 *
 * Combinatorial number system: idx = C(j3,4)+C(j2,3)+C(j1,2)+C(j0,1)
 * Decode by finding largest jk with C(jk,k+1) ≤ remaining.
 * ---------------------------------------------------------------- */

__device__ __forceinline__
unsigned long long comb2(int n) {
    return (n >= 2) ? (unsigned long long)n * (n - 1) / 2 : 0;
}

__device__ __forceinline__
unsigned long long comb3(int n) {
    return (n >= 3) ? (unsigned long long)n * (n - 1) * (n - 2) / 6 : 0;
}

__device__ __forceinline__
unsigned long long comb4(int n) {
    return (n >= 4) ? (unsigned long long)n * (n - 1) * (n - 2) * (n - 3) / 24 : 0;
}

__device__ void decode_multiset(unsigned long long idx, int R,
                                int *i0, int *i1, int *i2, int *i3) {
    /* Find j3: largest j with C(j,4) ≤ idx */
    int lo = 3, hi = R + 2;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;
        if (comb4(mid) <= idx) lo = mid;
        else hi = mid - 1;
    }
    int j3 = lo;
    idx -= comb4(j3);

    /* Find j2: largest j with C(j,3) ≤ idx, j < j3 */
    lo = 2; hi = j3 - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;
        if (comb3(mid) <= idx) lo = mid;
        else hi = mid - 1;
    }
    int j2 = lo;
    idx -= comb3(j2);

    /* Find j1: largest j with C(j,2) ≤ idx, j < j2 */
    lo = 1; hi = j2 - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;
        if (comb2(mid) <= idx) lo = mid;
        else hi = mid - 1;
    }
    int j1 = lo;
    idx -= comb2(j1);

    int j0 = (int)idx;

    /* Convert back: ik = jk - k */
    *i0 = j0;
    *i1 = j1 - 1;
    *i2 = j2 - 2;
    *i3 = j3 - 3;
}

/* ----------------------------------------------------------------
 * Device: fully unrolled 4×4 permanent via column-subset DP.
 *
 * dp[S] = Σ_{j∈S} dp[S\{j}] · a[|S|-1][j]
 *
 * Key insight: max value at layer 4 is 24·(p-1)^4 ≤ 5.6M for p=23,
 * which fits in a 32-bit int.  So we do ALL arithmetic without any
 * intermediate % p, and reduce once at the very end.
 * 28 multiplications + 17 additions + 1 single mod.
 * ---------------------------------------------------------------- */

__device__ __forceinline__
int perm_4x4_dp_modp(const int *r0, const int *r1,
                      const int *r2, const int *r3, int p) {
    /* Layer 1: dp[{j}] = r0[j]  (entries already in [0, p-1]) */
    int d0 = r0[0], d1 = r0[1], d2 = r0[2], d3 = r0[3];

    /* Layer 2: dp[{i,j}] = d_i * a1_j + d_j * a1_i  (12 mults, 6 adds) */
    int a10 = r1[0], a11 = r1[1], a12 = r1[2], a13 = r1[3];
    int d01 = d0 * a11 + d1 * a10;
    int d02 = d0 * a12 + d2 * a10;
    int d03 = d0 * a13 + d3 * a10;
    int d12 = d1 * a12 + d2 * a11;
    int d13 = d1 * a13 + d3 * a11;
    int d23 = d2 * a13 + d3 * a12;

    /* Layer 3: dp[{i,j,k}] = Σ dp[S\{l}] * a2_l  (12 mults, 8 adds) */
    int a20 = r2[0], a21 = r2[1], a22 = r2[2], a23 = r2[3];
    int d012 = d12 * a20 + d02 * a21 + d01 * a22;
    int d013 = d13 * a20 + d03 * a21 + d01 * a23;
    int d023 = d23 * a20 + d03 * a22 + d02 * a23;
    int d123 = d23 * a21 + d13 * a22 + d12 * a23;

    /* Layer 4: perm = Σ dp[S\{l}] * a3_l  (4 mults, 3 adds) */
    int a30 = r3[0], a31 = r3[1], a32 = r3[2], a33 = r3[3];
    int perm = d123 * a30 + d023 * a31 + d013 * a32 + d012 * a33;

    /* Single mod at the very end */
    perm %= p;
    if (perm < 0) perm += p;
    return perm;
}

/* ----------------------------------------------------------------
 * Device: multinomial weight 4!/(m1!...mk!) from sorted indices
 * ---------------------------------------------------------------- */

__device__ __forceinline__
int compute_weight_4(int i0, int i1, int i2, int i3) {
    /* Sorted: i0 ≤ i1 ≤ i2 ≤ i3 */
    if (i0 == i3) return 1;           /* all same: 4!/4! = 1 */
    if (i0 == i2) return 4;           /* triple + single: 4!/3! = 4 */
    if (i1 == i3) return 4;           /* single + triple: 4!/3! = 4 */
    if (i0 == i1 && i2 == i3) return 6;  /* two pairs: 4!/(2!2!) = 6 */
    if (i0 == i1) return 12;          /* one pair (at start): 4!/2! = 12 */
    if (i1 == i2) return 12;          /* one pair (middle): 4!/2! = 12 */
    if (i2 == i3) return 12;          /* one pair (at end): 4!/2! = 12 */
    return 24;                         /* all distinct: 4! = 24 */
}

/* ----------------------------------------------------------------
 * Kernel: projective multiset permanent enumeration
 *
 * Block-based: each thread gets a contiguous range of multisets.
 * Decode once at the start, then iterate sequentially — avoids
 * 3 binary searches per multiset (the old bottleneck: decode was
 * ~4× more expensive than the permanent itself).
 * ---------------------------------------------------------------- */

__device__ __forceinline__
void next_multiset(int *i0, int *i1, int *i2, int *i3, int R) {
    /* Increment multiset in colex order (matching combinatorial number system).
     * j0 is least significant: try incrementing i0 first, cascade right. */
    if (++(*i0) <= *i1) return;           /* most common case */
    *i0 = 0; if (++(*i1) <= *i2) return;
    *i0 = 0; *i1 = 0; if (++(*i2) <= *i3) return;
    *i0 = 0; *i1 = 0; *i2 = 0; ++(*i3);  /* i3 < R guaranteed by caller */
}

__global__ void proj_multiset_perm_kernel(
    const int *proj_rows,         /* R × 4, projective row table */
    int R,                         /* number of projective rows */
    int p,                         /* prime */
    unsigned long long total_ms,   /* total multisets = C(R+3, 4) */
    unsigned long long block_size, /* multisets per thread */
    unsigned long long *global_w0) /* output: weighted count of perm=0 */
{
    __shared__ unsigned long long block_w0;
    if (threadIdx.x == 0) block_w0 = 0;
    __syncthreads();

    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long start = tid * block_size;
    unsigned long long end = start + block_size;
    if (end > total_ms) end = total_ms;
    if (start >= total_ms) start = end;

    unsigned long long local_w0 = 0;

    if (start < end) {
        /* Decode starting position once */
        int i0, i1, i2, i3;
        decode_multiset(start, R, &i0, &i1, &i2, &i3);

        for (unsigned long long midx = start; midx < end; midx++) {
            /* Load 4 rows via read-only cache (__ldg) for better random access */
            int row0[4], row1[4], row2[4], row3[4];
            const int *r0p = proj_rows + i0 * 4;
            const int *r1p = proj_rows + i1 * 4;
            const int *r2p = proj_rows + i2 * 4;
            const int *r3p = proj_rows + i3 * 4;
            row0[0] = __ldg(r0p); row0[1] = __ldg(r0p+1); row0[2] = __ldg(r0p+2); row0[3] = __ldg(r0p+3);
            row1[0] = __ldg(r1p); row1[1] = __ldg(r1p+1); row1[2] = __ldg(r1p+2); row1[3] = __ldg(r1p+3);
            row2[0] = __ldg(r2p); row2[1] = __ldg(r2p+1); row2[2] = __ldg(r2p+2); row2[3] = __ldg(r2p+3);
            row3[0] = __ldg(r3p); row3[1] = __ldg(r3p+1); row3[2] = __ldg(r3p+2); row3[3] = __ldg(r3p+3);

            int perm = perm_4x4_dp_modp(row0, row1, row2, row3, p);

            if (perm == 0) {
                local_w0 += compute_weight_4(i0, i1, i2, i3);
            }

            /* Advance to next multiset (cheap: just increment + cascade) */
            if (midx + 1 < end) {
                next_multiset(&i0, &i1, &i2, &i3, R);
            }
        }
    }

    if (local_w0 > 0) {
        atomicAdd(&block_w0, local_w0);
    }
    __syncthreads();

    if (threadIdx.x == 0 && block_w0 > 0) {
        atomicAdd(global_w0, block_w0);
    }
}

/* ----------------------------------------------------------------
 * Host: generate projective rows in F_p^4
 *
 * Nonzero vectors with first nonzero entry = 1.
 * For leading position k: entries [0..k-1] = 0, entry k = 1,
 *                          entries [k+1..3] ∈ F_p
 * ---------------------------------------------------------------- */

static int generate_projective_rows(int p, int *table) {
    int count = 0;
    for (int k = 0; k < 4; k++) {
        /* Number of free entries after position k */
        int free = 3 - k;
        int num_tails = 1;
        for (int i = 0; i < free; i++) num_tails *= p;

        for (int t = 0; t < num_tails; t++) {
            int *row = table + count * 4;
            /* Zero prefix */
            for (int j = 0; j < k; j++) row[j] = 0;
            /* Leading 1 */
            row[k] = 1;
            /* Free tail in base p */
            int tmp = t;
            for (int j = k + 1; j < 4; j++) {
                row[j] = tmp % p;
                tmp /= p;
            }
            count++;
        }
    }
    return count;
}

/* ----------------------------------------------------------------
 * Host: compute C(n, 4) as unsigned long long
 * ---------------------------------------------------------------- */

static unsigned long long host_comb4(long long n) {
    if (n < 4) return 0;
    __uint128_t value = (__uint128_t)n * (n - 1) * (n - 2) * (n - 3) / 24;
    return value > ULLONG_MAX ? ULLONG_MAX : (unsigned long long)value;
}

/* ----------------------------------------------------------------
 * Host API: compute W_0 for given prime p
 *
 * Returns 0 on success, -1 on error.
 * w0_out receives the weighted count of projective multisets with perm≡0.
 * total_w_out receives total weight (for validation: should = R^4).
 * ---------------------------------------------------------------- */

extern "C" int proj_multiset_perm_n4(int p,
                                      unsigned long long *w0_out,
                                      unsigned long long *total_w_out) {
    int *h_table = NULL;
    int *d_table = NULL;
    unsigned long long *d_w0 = NULL;
    cudaError_t err;

    if (!w0_out) {
        fprintf(stderr, "proj_multiset_perm_n4: w0_out must not be null\n");
        return -1;
    }
    if (p < 2 || p > MAX_P) {
        fprintf(stderr, "proj_multiset_perm_n4: p=%d out of range\n", p);
        return -1;
    }
    if (!host_is_prime(p)) {
        fprintf(stderr, "proj_multiset_perm_n4: p=%d is not prime\n", p);
        return -1;
    }

    /* Compute R = (p^4-1)/(p-1) = p^3+p^2+p+1 */
    int R = p * p * p + p * p + p + 1;

    /* Generate projective row table */
    __uint128_t expected_total_w_128 =
        (__uint128_t)R * (unsigned)R * (unsigned)R * (unsigned)R;
    if (expected_total_w_128 > ULLONG_MAX) {
        fprintf(stderr,
                "proj_multiset_perm_n4: result counts exceed uint64 for p=%d\n", p);
        return -1;
    }

    h_table = (int *)malloc((size_t)R * 4 * sizeof(int));
    if (!h_table) {
        fprintf(stderr, "proj_multiset_perm_n4: host allocation failed\n");
        return -1;
    }
    int actual_R = generate_projective_rows(p, h_table);
    if (actual_R != R) {
        fprintf(stderr, "Expected R=%d, got %d\n", R, actual_R);
        free(h_table);
        return -1;
    }

    /* Total multisets = C(R+3, 4) */
    unsigned long long total_ms = host_comb4((long long)R + 3);
    if (total_ms == ULLONG_MAX) {
        fprintf(stderr,
                "proj_multiset_perm_n4: multiset count exceeds uint64 for p=%d\n", p);
        free(h_table);
        return -1;
    }

    fprintf(stderr, "p=%d: R=%d, multisets=%llu\n", p, R, total_ms);

    /* Also compute total weight for validation */
    /* Sum of weights over all multisets = R^4 */
    /* (since each ordered 4-tuple maps to exactly one multiset with that weight) */
    unsigned long long expected_total_w = (unsigned long long)expected_total_w_128;

    /* Upload table to GPU */
    err = cudaMalloc(&d_table, (size_t)R * 4 * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_table);
        return -1;
    }
    err = cudaMemcpy(
        d_table, h_table, (size_t)R * 4 * sizeof(int), cudaMemcpyHostToDevice);
    free(h_table);
    h_table = NULL;
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_table);
        return -1;
    }

    /* Allocate output */
    err = cudaMalloc(&d_w0, sizeof(unsigned long long));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_table);
        return -1;
    }
    err = cudaMemset(d_w0, 0, sizeof(unsigned long long));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_table);
        cudaFree(d_w0);
        return -1;
    }

    /* Launch config: use enough threads for good occupancy.
     * Each thread gets a contiguous block of multisets to iterate
     * sequentially (decode once, increment cheaply thereafter). */
    int threads_per_block = 256;
    unsigned long long total_threads = 16ULL * 1024 * 1024;  /* 16M threads */
    unsigned long long block_size = (total_ms + total_threads - 1) / total_threads;
    if (block_size < 1) block_size = 1;
    /* Recompute actual thread count needed */
    total_threads = (total_ms + block_size - 1) / block_size;
    int num_blocks = (int)((total_threads + threads_per_block - 1) / threads_per_block);
    if (num_blocks < 1) num_blocks = 1;

    fprintf(stderr, "Launching: %d blocks × %d threads, block_size=%llu\n",
            num_blocks, threads_per_block, block_size);

    proj_multiset_perm_kernel<<<num_blocks, threads_per_block>>>(
        d_table, R, p, total_ms, block_size, d_w0);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_table);
        cudaFree(d_w0);
        return -1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(d_table);
        cudaFree(d_w0);
        return -1;
    }

    /* Copy back */
    unsigned long long h_w0;
    err = cudaMemcpy(&h_w0, d_w0, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(d_table);
    cudaFree(d_w0);

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    *w0_out = h_w0;
    if (total_w_out) *total_w_out = expected_total_w;
    return 0;
}

#ifdef TEST_MAIN
int main() {
    /* Validate for p=3 */
    unsigned long long w0, total_w;

    printf("=== Projective multiset enumeration for 4×4 permanents ===\n\n");

    int primes[] = {3, 5, 7};
    int nprimes = 3;

    for (int i = 0; i < nprimes; i++) {
        int p = primes[i];
        int R = p*p*p + p*p + p + 1;
        unsigned long long total_ms = host_comb4((long long)R + 3);

        printf("p=%d: R=%d, multisets=%llu\n", p, R, total_ms);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        int rc = proj_multiset_perm_n4(p, &w0, &total_w);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        if (rc != 0) {
            printf("  ERROR\n");
            continue;
        }

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        /* Reconstruct N_0 */
        long long pm1_4 = (long long)(p-1)*(p-1)*(p-1)*(p-1);
        /* L_row(4,p) = 4p^12 - 6p^8 + 4p^4 - 1 */
        long long p4 = (long long)p*p*p*p;
        long long p8 = p4*p4;
        long long p12 = p8*p4;
        long long L = 4*p12 - 6*p8 + 4*p4 - 1;
        long long N0 = L + pm1_4 * (long long)w0;
        long long p16 = p12*p4;
        long long p15 = p16/p;
        long long E = N0 - p15;

        printf("  W_0 = %llu\n", w0);
        printf("  L_row = %lld\n", L);
        printf("  N_0 = %lld\n", N0);
        printf("  p^15 = %lld\n", p15);
        printf("  E^perm(4,%d) = %lld\n", p, E);
        printf("  Time: %.3f s\n\n", ms / 1000.0);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}
#endif
