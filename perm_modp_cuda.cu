/*
 * CUDA implementation of permanent mod p for general prime p.
 *
 * Uses Gray code block parallelization with 64-bit arithmetic.
 * Suitable for Schur matrix permanents with small primes.
 *
 * Compile: nvcc -O3 -DTEST_MAIN -o perm_modp_cuda perm_modp_cuda.cu
 * Library: nvcc -shared -Xcompiler -fPIC -O3 -o perm_modp_cuda.so perm_modp_cuda.cu
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

static int64_t normalize_mod_host(int64_t value, int64_t p) {
    int64_t result = value % p;
    return result < 0 ? result + p : result;
}

static int64_t mul_mod_host(int64_t a, int64_t b, int64_t p) {
    if (a == 0 || b == 0) return 0;
    if (a <= INT64_MAX / b) return (a * b) % p;
    return (int64_t)(((__int128)a * (__int128)b) % (__int128)p);
}

static int64_t pow_mod_host(int64_t base, uint64_t exponent, int64_t p) {
    int64_t result = 1 % p;
    base = normalize_mod_host(base, p);
    while (exponent != 0) {
        if (exponent & 1U) result = mul_mod_host(result, base, p);
        exponent >>= 1;
        if (exponent != 0) base = mul_mod_host(base, base, p);
    }
    return result;
}

struct NecklacePrefixTask {
    uint64_t word;
    uint32_t period;
    uint32_t weight;
};

struct NecklaceFrontierBuilder {
    NecklacePrefixTask *tasks;
    size_t count;
    size_t capacity;
    int minor_size;
    int depth;
    int failed;
    uint8_t bits[64];
};

static int append_necklace_task(NecklaceFrontierBuilder *builder,
                                uint64_t word,
                                int period,
                                int weight) {
    if (builder->count >= UINT32_MAX) {
        builder->failed = 1;
        return 0;
    }
    if (builder->count == builder->capacity) {
        size_t new_capacity = builder->capacity
            ? builder->capacity * 2 : 1024;
        if (new_capacity <= builder->capacity ||
            new_capacity > UINT32_MAX ||
            new_capacity > SIZE_MAX / sizeof(NecklacePrefixTask)) {
            builder->failed = 1;
            return 0;
        }
        void *resized = realloc(
            builder->tasks, new_capacity * sizeof(NecklacePrefixTask));
        if (!resized) {
            builder->failed = 1;
            return 0;
        }
        builder->tasks = (NecklacePrefixTask*)resized;
        builder->capacity = new_capacity;
    }
    builder->tasks[builder->count++] = {
        word, (uint32_t)period, (uint32_t)weight};
    return 1;
}

static void build_necklace_frontier_recursive(
        NecklaceFrontierBuilder *builder,
        int position,
        int period,
        int weight,
        uint64_t prefix) {
    if (builder->failed) return;
    if (position > builder->depth) {
        append_necklace_task(
            builder, prefix << (builder->minor_size - builder->depth),
            period, weight);
        return;
    }

    uint8_t copied_bit = builder->bits[position - period];
    if (weight + copied_bit <= builder->minor_size / 2) {
        builder->bits[position] = copied_bit;
        build_necklace_frontier_recursive(
            builder, position + 1, period, weight + copied_bit,
            (prefix << 1) | copied_bit);
    }
    if (!builder->failed && copied_bit == 0 &&
        weight + 1 <= builder->minor_size / 2) {
        builder->bits[position] = 1;
        build_necklace_frontier_recursive(
            builder, position + 1, position, weight + 1,
            (prefix << 1) | 1U);
    }
}

static NecklacePrefixTask *build_necklace_frontier(int minor_size,
                                                    int depth,
                                                    uint32_t *task_count) {
    *task_count = 0;
    if (minor_size <= 0 || minor_size >= 64 ||
        depth < 0 || depth > minor_size) {
        return NULL;
    }
    NecklaceFrontierBuilder builder = {};
    builder.minor_size = minor_size;
    builder.depth = depth;
    build_necklace_frontier_recursive(&builder, 1, 1, 0, 0);
    if (builder.failed || builder.count == 0 || builder.count > UINT32_MAX) {
        free(builder.tasks);
        return NULL;
    }
    *task_count = (uint32_t)builder.count;
    return builder.tasks;
}

static int is_prime_order_host(int value) {
    if (value < 2) return 0;
    if ((value & 1) == 0) return value == 2;
    for (int divisor = 3; divisor <= value / divisor; divisor += 2) {
        if (value % divisor == 0) return 0;
    }
    return 1;
}

/* Only called for the validated prime Schur orders below 64. */
static int primitive_root_prime_order_host(int prime) {
    for (int candidate = 2; candidate < prime; candidate++) {
        int value = 1;
        int primitive = 1;
        for (int exponent = 1; exponent < prime - 1; exponent++) {
            value = (value * candidate) % prime;
            if (value == 1) {
                primitive = 0;
                break;
            }
        }
        if (primitive) return candidate;
    }
    return -1;
}

/* All operands passed to these device helpers are in [0, modulus). */
__device__ __forceinline__ uint64_t add_mod_u64(uint64_t a, uint64_t b,
                                                uint64_t modulus) {
    return a >= modulus - b ? a - (modulus - b) : a + b;
}

__device__ __forceinline__ uint64_t sub_mod_u64(uint64_t a, uint64_t b,
                                                uint64_t modulus) {
    return a >= b ? a - b : modulus - (b - a);
}

__device__ __forceinline__ uint64_t mul_mod_u64(uint64_t a, uint64_t b,
                                                uint64_t modulus) {
    if (a == 0 || b == 0) return 0;
    if (a <= UINT64_MAX / b) return (a * b) % modulus;

    /* Overflow-safe fallback for moduli too large for a 64-bit product. */
    uint64_t result = 0;
    while (b != 0) {
        if (b & 1) result = add_mod_u64(result, a, modulus);
        b >>= 1;
        if (b != 0) a = add_mod_u64(a, a, modulus);
    }
    return result;
}

/*
 * Arithmetic for p = 2^32 - c, 1 <= c <= 2^16 - 1.  For x < 2^64,
 * replace each high 32-bit limb by c because 2^32 == c (mod p):
 *
 *   t0 = low(x)  + high(x)  * c
 *   t1 = low(t0) + high(t0) * c.
 *
 * The first fold is below (c+1)2^32, hence high(t0) <= c.  Therefore
 * t1 <= 2^32-1+c^2 < 2p for c <= 2^16-1, so one subtraction completes
 * the reduction.  All intermediate values fit in uint64_t.
 */
__device__ __forceinline__ uint32_t mul_mod_pseudo_mersenne_u32(
        uint32_t a, uint32_t b, uint32_t p, uint32_t c) {
    uint64_t product = (uint64_t)a * (uint64_t)b;
    uint64_t folded = (uint32_t)product + (product >> 32) * (uint64_t)c;
    folded = (uint32_t)folded + (folded >> 32) * (uint64_t)c;
    if (folded >= p) folded -= p;
    return (uint32_t)folded;
}

__device__ __forceinline__ uint32_t add_mod_u32(uint32_t a, uint32_t b,
                                                uint32_t p) {
    uint64_t sum = (uint64_t)a + (uint64_t)b;
    if (sum >= p) sum -= p;
    return (uint32_t)sum;
}

__device__ __forceinline__ uint32_t sub_mod_u32(uint32_t a, uint32_t b,
                                                uint32_t p) {
    return a >= b ? a - b : p - (b - a);
}

/*
 * Each CUDA thread processes a contiguous block of Gray code states.
 * Row sums are stored per-thread, updated via Gray code iteration.
 */
__global__ void permanent_modp_kernel(const int64_t *matrix,  /* n x n, row-major */
                                       int n,
                                       uint64_t p,
                                       uint64_t num_subsets,
                                       uint64_t block_size,
                                       uint64_t output_count,
                                       uint64_t *partial_sums) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= output_count) return;

    uint64_t start = tid * block_size;
    uint64_t end = start + block_size;
    if (end > num_subsets) end = num_subsets;
    if (start >= num_subsets) start = end;

    /* Skip state 0 (empty subset) */
    if (start == 0) start = 1;

    uint64_t local_sum = 0;

    /* Allocate row_sums in registers/local memory */
    /* For n <= 64, we use a fixed-size array */
    uint64_t row_sums[64];

    if (start < end) {
        /* Initialize row sums to zero */
        for (int i = 0; i < n; i++) {
            row_sums[i] = 0;
        }

        /* Initialize for starting Gray code state */
        uint64_t gray_start = start ^ (start >> 1);
        int popcount_start = 0;

        for (int col = 0; col < n; col++) {
            if ((gray_start >> col) & 1) {
                for (int row = 0; row < n; row++) {
                    row_sums[row] = add_mod_u64(
                        row_sums[row], (uint64_t)matrix[row * n + col], p);
                }
                popcount_start++;
            }
        }

        /* Process starting state */
        int subset_sign = (popcount_start & 1) ? -1 : 1;

        uint64_t prod = 1 % p;
        for (int row = 0; row < n; row++) {
            prod = mul_mod_u64(prod, row_sums[row], p);
        }
        local_sum = subset_sign < 0
            ? sub_mod_u64(local_sum, prod, p)
            : add_mod_u64(local_sum, prod, p);

        /* Gray code iteration within this block */
        for (uint64_t i = start + 1; i < end; i++) {
            uint64_t gray_i = i ^ (i >> 1);
            uint64_t gray_prev = (i - 1) ^ ((i - 1) >> 1);
            uint64_t diff = gray_i ^ gray_prev;
            int col = __ffsll(diff) - 1;

            if ((gray_i >> col) & 1) {
                /* Add column */
                for (int row = 0; row < n; row++) {
                    row_sums[row] = add_mod_u64(
                        row_sums[row], (uint64_t)matrix[row * n + col], p);
                }
            } else {
                /* Subtract column */
                for (int row = 0; row < n; row++) {
                    row_sums[row] = sub_mod_u64(
                        row_sums[row], (uint64_t)matrix[row * n + col], p);
                }
            }
            subset_sign = -subset_sign;

            /* Compute product of row sums mod p without signed overflow. */
            prod = 1 % p;
            for (int row = 0; row < n; row++) {
                prod = mul_mod_u64(prod, row_sums[row], p);
            }
            local_sum = subset_sign < 0
                ? sub_mod_u64(local_sum, prod, p)
                : add_mod_u64(local_sum, prod, p);
        }
    }

    partial_sums[tid] = local_sum;
}

/*
 * Glynn's formula with the last column sign fixed at +1.  Gray-code bit
 * ``col`` represents delta[col] == -1, so each state contributes
 *
 *   product(delta) * product(row sums).
 *
 * Fixing one sign leaves 2^(n-1) states and the host applies 2^(1-n).
 */
__global__ void permanent_glynn_modp_kernel(
        const int64_t *matrix,  /* n x n, row-major */
        int n,
        uint64_t p,
        uint64_t num_states,
        uint64_t block_size,
        uint64_t output_count,
        uint64_t *partial_sums) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= output_count) return;

    uint64_t start = tid * block_size;
    uint64_t end = start + block_size;
    if (end > num_states) end = num_states;

    uint64_t local_sum = 0;
    uint64_t row_sums[64];

    if (start < end) {
        /* Start from delta=(+1,...,+1). */
        for (int row = 0; row < n; row++) {
            uint64_t sum = 0;
            for (int col = 0; col < n; col++) {
                sum = add_mod_u64(
                    sum, (uint64_t)matrix[row * n + col], p);
            }
            row_sums[row] = sum;
        }

        uint64_t gray = start ^ (start >> 1);
        for (int col = 0; col < n - 1; col++) {
            if ((gray >> col) & 1U) {
                for (int row = 0; row < n; row++) {
                    uint64_t entry = (uint64_t)matrix[row * n + col];
                    uint64_t twice_entry = add_mod_u64(entry, entry, p);
                    row_sums[row] = sub_mod_u64(
                        row_sums[row], twice_entry, p);
                }
            }
        }

        int state_sign = (__popcll(gray) & 1) ? -1 : 1;
        uint64_t prod = 1 % p;
        for (int row = 0; row < n; row++) {
            prod = mul_mod_u64(prod, row_sums[row], p);
        }
        local_sum = state_sign < 0
            ? sub_mod_u64(local_sum, prod, p)
            : add_mod_u64(local_sum, prod, p);

        for (uint64_t state = start + 1; state < end; state++) {
            /* Consecutive binary-reflected Gray states differ at ffs(state). */
            int col = __ffsll(state) - 1;
            gray = state ^ (state >> 1);

            for (int row = 0; row < n; row++) {
                uint64_t entry = (uint64_t)matrix[row * n + col];
                uint64_t twice_entry = add_mod_u64(entry, entry, p);
                row_sums[row] = ((gray >> col) & 1U)
                    ? sub_mod_u64(row_sums[row], twice_entry, p)
                    : add_mod_u64(row_sums[row], twice_entry, p);
            }
            state_sign = -state_sign;

            prod = 1 % p;
            for (int row = 0; row < n; row++) {
                prod = mul_mod_u64(prod, row_sums[row], p);
            }
            local_sum = state_sign < 0
                ? sub_mod_u64(local_sum, prod, p)
                : add_mod_u64(local_sum, prod, p);
        }
    }

    partial_sums[tid] = local_sum;
}

/*
 * Register-resident specialization for the CRT primes immediately below
 * 2^32.  N is compile-time constant so every row operation and product is
 * unrolled; ptxas can consequently scalarize row_sums instead of placing the
 * generic 64-element array in per-thread local memory.
 */
template<int N>
__global__ void permanent_glynn_modp_pm32_kernel(
        const uint32_t *matrix,  /* N x N, row-major */
        uint32_t p,
        uint32_t c,
        uint64_t num_states,
        uint64_t block_size,
        uint64_t output_count,
        uint64_t *partial_sums) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (tid >= output_count) return;

    uint64_t start = tid * block_size;
    uint64_t end = start + block_size;
    if (end > num_states) end = num_states;

    uint32_t local_sum = 0;
    uint32_t row_sums[N];

    if (start < end) {
        uint64_t gray = start ^ (start >> 1);

        /* The last column has fixed Glynn sign +1. */
#pragma unroll
        for (int row = 0; row < N; row++) {
            uint32_t sum = matrix[row * N + (N - 1)];
#pragma unroll
            for (int col = 0; col < N - 1; col++) {
                uint32_t entry = matrix[row * N + col];
                sum = ((gray >> col) & 1U)
                    ? sub_mod_u32(sum, entry, p)
                    : add_mod_u32(sum, entry, p);
            }
            row_sums[row] = sum;
        }

        int state_sign = (__popcll(gray) & 1) ? -1 : 1;
        uint32_t prod = 1;
#pragma unroll
        for (int row = 0; row < N; row++) {
            prod = mul_mod_pseudo_mersenne_u32(
                prod, row_sums[row], p, c);
        }
        local_sum = state_sign < 0
            ? sub_mod_u32(local_sum, prod, p)
            : add_mod_u32(local_sum, prod, p);

        for (uint64_t state = start + 1; state < end; state++) {
            int col = __ffsll(state) - 1;
            gray = state ^ (state >> 1);

#pragma unroll
            for (int row = 0; row < N; row++) {
                uint32_t entry = matrix[row * N + col];
                uint32_t twice_entry = add_mod_u32(entry, entry, p);
                row_sums[row] = ((gray >> col) & 1U)
                    ? sub_mod_u32(row_sums[row], twice_entry, p)
                    : add_mod_u32(row_sums[row], twice_entry, p);
            }
            state_sign = -state_sign;

            prod = 1;
#pragma unroll
            for (int row = 0; row < N; row++) {
                prod = mul_mod_pseudo_mersenne_u32(
                    prod, row_sums[row], p, c);
            }
            local_sum = state_sign < 0
                ? sub_mod_u32(local_sum, prod, p)
                : add_mod_u32(local_sum, prod, p);
        }
    }

    partial_sums[tid] = local_sum;
}

/*
 * One byte per remaining FKM recursion level: six bits of period and two
 * bits of DFS phase.  Keeping the words as named scalars (rather than a
 * dynamically indexed array) lets ptxas retain them in registers.
 */
struct PackedNecklaceDfsState {
    uint64_t first;
    uint64_t second;
    uint64_t third;

    __device__ __forceinline__ uint8_t get(int slot) const {
        int shift = (slot & 7) * 8;
        switch (slot >> 3) {
            case 0: return (uint8_t)(first >> shift);
            case 1: return (uint8_t)(second >> shift);
            default: return (uint8_t)(third >> shift);
        }
    }

    __device__ __forceinline__ void set(int slot, uint8_t value) {
        int shift = (slot & 7) * 8;
        uint64_t mask = UINT64_C(0xff) << shift;
        uint64_t inserted = (uint64_t)value << shift;
        switch (slot >> 3) {
            case 0:
                first = (first & ~mask) | inserted;
                break;
            case 1:
                second = (second & ~mask) | inserted;
                break;
            default:
                third = (third & ~mask) | inserted;
                break;
        }
    }
};

template<int M>
__device__ __forceinline__ uint64_t rotate_necklace_word(uint64_t word,
                                                         int shift) {
    constexpr uint64_t mask = (UINT64_C(1) << M) - 1;
    return ((word << shift) & mask) | (word >> (M - shift));
}

template<int M>
__device__ __forceinline__ uint64_t least_necklace_rotation(uint64_t word) {
    uint64_t best = word;
#pragma unroll
    for (int shift = 1; shift < M; shift++) {
        uint64_t rotated = rotate_necklace_word<M>(word, shift);
        if (rotated < best) best = rotated;
    }
    return best;
}

template<int M, int DEPTH>
__device__ __forceinline__ bool next_necklace_representative(
        int &position,
        uint64_t &word,
        PackedNecklaceDfsState &stack,
        uint64_t &output_word,
        uint32_t &orbit_size) {
    constexpr int start = DEPTH + 1;
    constexpr uint64_t word_mask = (UINT64_C(1) << M) - 1;
    static_assert(M < 64, "necklace words must fit in uint64_t");
    static_assert(DEPTH >= 0 && DEPTH <= M, "invalid necklace depth");
    static_assert(M - DEPTH + 1 <= 24,
                  "packed necklace stack has at most 24 entries");

    while (position >= start) {
        uint8_t packed = stack.get(position - start);
        int period = packed & 63;

        if (position == M + 1) {
            int weight = __popcll(word);
            --position;
            if (M % period != 0) continue;

            if (weight < M / 2) {
                output_word = word;
                orbit_size = 2 * period;
                return true;
            }

            uint64_t complement_rep =
                least_necklace_rotation<M>(word_mask ^ word);
            if (word > complement_rep) continue;
            output_word = word;
            orbit_size = word == complement_rep ? period : 2 * period;
            return true;
        }

        int phase = packed >> 6;
        int source = position - period;
        int copied_bit = (word >> (M - source)) & 1U;
        int weight = __popcll(word >> (M - position + 1));

        if (phase == 0) {
            stack.set(position - start,
                      (uint8_t)(period | (1U << 6)));
            if (weight + copied_bit <= M / 2) {
                uint64_t bit = UINT64_C(1) << (M - position);
                word = copied_bit ? word | bit : word & ~bit;
                stack.set(position + 1 - start, (uint8_t)period);
                ++position;
                continue;
            }
        }

        if (phase <= 1) {
            stack.set(position - start,
                      (uint8_t)(period | (2U << 6)));
            if (copied_bit == 0 && weight + 1 <= M / 2) {
                word |= UINT64_C(1) << (M - position);
                stack.set(position + 1 - start, (uint8_t)position);
                ++position;
                continue;
            }
        }
        --position;
    }
    return false;
}

/*
 * Each thread advances complete FKM prefix subtrees.  The compile-time minor
 * size scalarizes all row sums, while a persistent atomic queue balances the
 * unequal subtrees.  The orbit group consists of cyclic column rotations and
 * global sign complementation; each emitted representative is weighted by
 * its exact orbit size.
 */
template<int M, int DEPTH>
__global__ void schur_necklace_pm32_kernel(
        const NecklacePrefixTask *tasks,
        uint32_t task_count,
        const uint32_t *base,
        uint32_t base_sum,
        uint32_t p,
        uint32_t c,
        uint32_t *next_task,
        unsigned long long *raw_sum) {
    uint32_t local_sum = 0;

    while (true) {
        uint32_t task_index = atomicAdd(next_task, 1U);
        if (task_index >= task_count) break;

        NecklacePrefixTask task = tasks[task_index];
        uint32_t row_sums[M];
#pragma unroll
        for (int row = 0; row < M; row++) row_sums[row] = base_sum;

        int position = DEPTH + 1;
        uint64_t generator_word = task.word;
        PackedNecklaceDfsState stack = {0, 0, 0};
        stack.set(0, (uint8_t)task.period);
        uint64_t previous_word = 0;

        while (true) {
            uint64_t word;
            uint32_t orbit_size;
            if (!next_necklace_representative<M, DEPTH>(
                    position, generator_word, stack, word, orbit_size)) {
                break;
            }

            uint64_t difference = previous_word ^ word;
            while (difference) {
                int low_bit = __ffsll((long long)difference) - 1;
                int column = M - 1 - low_bit;
                bool subtracting = (word >> low_bit) & 1U;
#pragma unroll
                for (int row = 0; row < M; row++) {
                    int base_index = column - row;
                    if (base_index < 0) base_index += M;
                    uint32_t entry = base[base_index];
                    uint32_t twice_entry = add_mod_u32(entry, entry, p);
                    row_sums[row] = subtracting
                        ? sub_mod_u32(row_sums[row], twice_entry, p)
                        : add_mod_u32(row_sums[row], twice_entry, p);
                }
                difference &= difference - 1;
            }
            previous_word = word;

            uint32_t product = 1;
#pragma unroll
            for (int row = 0; row < M; row++) {
                product = mul_mod_pseudo_mersenne_u32(
                    product, row_sums[row], p, c);
            }
            if (__popcll(word) & 1U) {
                product = product ? p - product : 0;
            }
            uint32_t contribution = mul_mod_pseudo_mersenne_u32(
                product, orbit_size, p, c);
            local_sum = add_mod_u32(local_sum, contribution, p);
        }
    }

    extern __shared__ uint32_t block_sums[];
    block_sums[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset; offset >>= 1) {
        if (threadIdx.x < offset) {
            block_sums[threadIdx.x] = add_mod_u32(
                block_sums[threadIdx.x],
                block_sums[threadIdx.x + offset], p);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(raw_sum, (unsigned long long)block_sums[0]);
    }
}

/*
 * Reduction kernel
 */
__global__ void reduce_sum_modp(const uint64_t *input, uint64_t *output,
                                uint64_t n, uint64_t p) {
    extern __shared__ uint64_t sdata[];

    uint64_t tid = threadIdx.x;
    uint64_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    uint64_t sum = 0;
    if (i < n) sum = input[i];
    if (i + blockDim.x < n) {
        sum = add_mod_u64(sum, input[i + blockDim.x], p);
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = add_mod_u64(sdata[tid], sdata[tid + s], p);
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
 * Host function to compute permanent mod p
 */
extern "C" int64_t permanent_modp_cuda(const int64_t *h_matrix,
                                        int n,
                                        int64_t p) {
    if (n < 0 || n >= 64) {
        fprintf(stderr, "n must satisfy 0 <= n < 64\n");
        return -1;
    }
    if (p <= 0) {
        fprintf(stderr, "p must be a positive signed 64-bit integer\n");
        return -1;
    }
    if (n == 0) return 1 % p;
    if (!h_matrix) {
        fprintf(stderr, "matrix must not be NULL\n");
        return -1;
    }
    if (n == 1) return normalize_mod_host(h_matrix[0], p);

    uint64_t num_subsets = (uint64_t)1 << n;

    /* Determine grid/block dimensions */
    int threads_per_block = 256;
    uint64_t total_threads;
    uint64_t block_size;

    if (n >= 30) {
        total_threads = 1ULL << 18;  /* 256K threads - fewer than mod3 due to more registers */
        block_size = (num_subsets + total_threads - 1) / total_threads;
    } else if (n >= 20) {
        total_threads = 1ULL << 14;
        block_size = (num_subsets + total_threads - 1) / total_threads;
    } else {
        total_threads = num_subsets;
        block_size = 1;
    }

    uint64_t num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    /* Allocate device memory */
    int64_t *h_normalized = (int64_t*)malloc(
        (size_t)n * (size_t)n * sizeof(int64_t));
    int64_t *d_matrix = NULL;
    uint64_t *d_partial_sums = NULL;
    uint64_t *d_reduce_sums = NULL;
    uint64_t *reduce_input = NULL;
    uint64_t *reduce_output = NULL;
    uint64_t reduce_size = total_threads;
    uint64_t result = 0;
    int64_t return_value = -1;

    if (!h_normalized) {
        fprintf(stderr, "failed to allocate normalized matrix\n");
        return -1;
    }
    for (int i = 0; i < n * n; i++) {
        h_normalized[i] = normalize_mod_host(h_matrix[i], p);
    }

    if (!cuda_call_succeeded(
            cudaMalloc(&d_matrix, (size_t)n * (size_t)n * sizeof(int64_t)),
            "cudaMalloc(matrix)")) goto cleanup;
    if (!cuda_call_succeeded(
            cudaMalloc(&d_partial_sums, total_threads * sizeof(uint64_t)),
            "cudaMalloc(partial_sums)")) goto cleanup;
    if (total_threads > 1 && !cuda_call_succeeded(
            cudaMalloc(&d_reduce_sums, total_threads * sizeof(uint64_t)),
            "cudaMalloc(reduce_sums)")) goto cleanup;

    if (!cuda_call_succeeded(
            cudaMemcpy(d_matrix, h_normalized,
                       (size_t)n * (size_t)n * sizeof(int64_t),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(matrix H2D)")) goto cleanup;

    /* Launch kernel */
    permanent_modp_kernel<<<num_blocks, threads_per_block>>>(
        d_matrix, n, (uint64_t)p, num_subsets, block_size, total_threads,
        d_partial_sums);
    if (!cuda_call_succeeded(cudaGetLastError(),
                             "permanent_modp_kernel launch")) goto cleanup;
    if (!cuda_call_succeeded(cudaDeviceSynchronize(),
                             "permanent_modp_kernel execution")) goto cleanup;

    /* Reduce partial sums */
    reduce_input = d_partial_sums;
    reduce_output = d_reduce_sums;
    while (reduce_size > 1) {
        uint64_t reduce_blocks = (reduce_size + threads_per_block * 2 - 1) / (threads_per_block * 2);
        reduce_sum_modp<<<reduce_blocks, threads_per_block, threads_per_block * sizeof(uint64_t)>>>(
            reduce_input, reduce_output, reduce_size, (uint64_t)p);
        if (!cuda_call_succeeded(cudaGetLastError(),
                                 "reduce_sum_modp launch")) goto cleanup;
        if (!cuda_call_succeeded(cudaDeviceSynchronize(),
                                 "reduce_sum_modp execution")) goto cleanup;
        reduce_size = reduce_blocks;
        uint64_t *tmp = reduce_input;
        reduce_input = reduce_output;
        reduce_output = tmp;
    }

    /* Copy result back */
    if (!cuda_call_succeeded(
            cudaMemcpy(&result, reduce_input, sizeof(uint64_t),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(result D2H)")) goto cleanup;

    /* Apply (-1)^n factor */
    if (n & 1) {
        result = result == 0 ? 0 : (uint64_t)p - result;
    }

    return_value = (int64_t)result;

cleanup:
    free(h_normalized);
    if (d_matrix && !cuda_call_succeeded(cudaFree(d_matrix),
                                          "cudaFree(matrix)")) {
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

static int glynn_pm32_size_supported(int n) {
    switch (n) {
        case 28:
        case 30:
        case 32:
        case 34:
        case 36:
        case 38:
        case 40:
        case 42:
        case 44:
            return 1;
        default:
            return 0;
    }
}

static void launch_glynn_pm32_kernel(const uint32_t *matrix,
                                     int n,
                                     uint32_t p,
                                     uint32_t c,
                                     uint64_t num_states,
                                     uint64_t block_size,
                                     uint64_t output_count,
                                     uint64_t num_blocks,
                                     int threads_per_block,
                                     uint64_t *partial_sums) {
#define LAUNCH_GLYNN_PM32(N)                                                \
    permanent_glynn_modp_pm32_kernel<N><<<num_blocks, threads_per_block>>>( \
        matrix, p, c, num_states, block_size, output_count, partial_sums)

    switch (n) {
        case 28: LAUNCH_GLYNN_PM32(28); break;
        case 30: LAUNCH_GLYNN_PM32(30); break;
        case 32: LAUNCH_GLYNN_PM32(32); break;
        case 34: LAUNCH_GLYNN_PM32(34); break;
        case 36: LAUNCH_GLYNN_PM32(36); break;
        case 38: LAUNCH_GLYNN_PM32(38); break;
        case 40: LAUNCH_GLYNN_PM32(40); break;
        case 42: LAUNCH_GLYNN_PM32(42); break;
        case 44: LAUNCH_GLYNN_PM32(44); break;
    }

#undef LAUNCH_GLYNN_PM32
}

struct NecklaceKernelConfig {
    int depth;
    int threads_per_block;
    int blocks_per_multiprocessor;
};

static int necklace_kernel_config(int minor_size,
                                  NecklaceKernelConfig *config) {
    switch (minor_size) {
        case 40: *config = {24, 256, 4}; return 1;
        case 46: *config = {28, 256, 4}; return 1;
        default: return 0;
    }
}

static void launch_schur_necklace_kernel(
        int minor_size,
        const NecklacePrefixTask *tasks,
        uint32_t task_count,
        const uint32_t *base,
        uint32_t base_sum,
        uint32_t p,
        uint32_t c,
        uint32_t *next_task,
        unsigned long long *raw_sum,
        int blocks,
        int threads_per_block) {
#define LAUNCH_NECKLACE(M, DEPTH)                                      \
    schur_necklace_pm32_kernel<M, DEPTH>                               \
        <<<blocks, threads_per_block,                                  \
           threads_per_block * sizeof(uint32_t)>>>(                    \
            tasks, task_count, base, base_sum, p, c, next_task, raw_sum)

    switch (minor_size) {
        case 40: LAUNCH_NECKLACE(40, 24); break;
        case 46: LAUNCH_NECKLACE(46, 28); break;
    }

#undef LAUNCH_NECKLACE
}

static int schur_necklace_preconditions(int n, int64_t omega, int64_t p) {
    NecklaceKernelConfig config;
    if (n < 3 || !(n & 1) || !is_prime_order_host(n) ||
        !necklace_kernel_config(n - 1, &config) || !(p & 1) ||
        (uint64_t)p > UINT32_MAX) {
        return 0;
    }
    uint64_t gap = (1ULL << 32) - (uint64_t)p;
    return gap >= 1 && gap <= UINT16_MAX &&
           pow_mod_host(omega, (uint64_t)n, p) == 1 % p;
}

/* Returns the full n x n Schur permanent, including cofactor factor n. */
static int64_t schur_necklace_modp_cuda_internal(int n,
                                                 int64_t omega,
                                                 int64_t p) {
    int minor_size = n - 1;
    NecklaceKernelConfig config;
    uint32_t task_count = 0;
    NecklacePrefixTask *tasks = NULL;
    NecklacePrefixTask *d_tasks = NULL;
    uint32_t *d_base = NULL;
    uint32_t *d_next_task = NULL;
    unsigned long long *d_raw_sum = NULL;
    unsigned long long raw_sum = 0;
    uint32_t base[64];
    uint64_t base_sum64 = 0;
    uint32_t base_sum = 0;
    uint32_t modulus = (uint32_t)p;
    uint32_t gap = (uint32_t)((1ULL << 32) - (uint64_t)p);
    size_t task_bytes = 0;
    int64_t return_value = -1;

    if (!necklace_kernel_config(minor_size, &config)) return -1;
    tasks = build_necklace_frontier(
        minor_size, config.depth, &task_count);
    if (!tasks) {
        fprintf(stderr, "failed to build Schur necklace frontier\n");
        return -1;
    }
    if ((size_t)task_count > SIZE_MAX / sizeof(NecklacePrefixTask)) {
        fprintf(stderr, "Schur necklace frontier is too large\n");
        goto cleanup;
    }

    {
        int generator = primitive_root_prime_order_host(n);
        if (generator < 0) {
            fprintf(stderr, "failed to find primitive root for Schur order\n");
            goto cleanup;
        }
        int exponent = 1;
        for (int index = 0; index < minor_size; index++) {
            base[index] = (uint32_t)pow_mod_host(
                omega, (uint64_t)exponent, p);
            base_sum64 += base[index];
            exponent = (exponent * generator) % n;
        }
    }
    base_sum = (uint32_t)(base_sum64 % (uint64_t)p);
    task_bytes = (size_t)task_count * sizeof(NecklacePrefixTask);

    if (!cuda_call_succeeded(cudaMalloc(&d_tasks, task_bytes),
                             "cudaMalloc(necklace tasks)")) goto cleanup;
    if (!cuda_call_succeeded(
            cudaMalloc(&d_base, (size_t)minor_size * sizeof(uint32_t)),
            "cudaMalloc(necklace base)")) goto cleanup;
    if (!cuda_call_succeeded(cudaMalloc(&d_next_task, sizeof(uint32_t)),
                             "cudaMalloc(necklace queue)")) goto cleanup;
    if (!cuda_call_succeeded(
            cudaMalloc(&d_raw_sum, sizeof(unsigned long long)),
            "cudaMalloc(necklace sum)")) goto cleanup;

    if (!cuda_call_succeeded(
            cudaMemcpy(d_tasks, tasks, task_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(necklace tasks H2D)")) goto cleanup;
    if (!cuda_call_succeeded(
            cudaMemcpy(d_base, base,
                       (size_t)minor_size * sizeof(uint32_t),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(necklace base H2D)")) goto cleanup;
    if (!cuda_call_succeeded(cudaMemset(d_next_task, 0, sizeof(uint32_t)),
                             "cudaMemset(necklace queue)")) goto cleanup;
    if (!cuda_call_succeeded(
            cudaMemset(d_raw_sum, 0, sizeof(unsigned long long)),
            "cudaMemset(necklace sum)")) goto cleanup;

    {
        int device = 0;
        int multiprocessor_count = 0;
        if (!cuda_call_succeeded(cudaGetDevice(&device),
                                 "cudaGetDevice(necklace)")) goto cleanup;
        if (!cuda_call_succeeded(
                cudaDeviceGetAttribute(&multiprocessor_count,
                                       cudaDevAttrMultiProcessorCount,
                                       device),
                "cudaDeviceGetAttribute(multiprocessor count)")) {
            goto cleanup;
        }
        if (multiprocessor_count <= 0 ||
            multiprocessor_count >
                INT32_MAX / config.blocks_per_multiprocessor) {
            fprintf(stderr, "invalid CUDA multiprocessor count\n");
            goto cleanup;
        }
        int blocks =
            multiprocessor_count * config.blocks_per_multiprocessor;
        launch_schur_necklace_kernel(
            minor_size, d_tasks, task_count, d_base, base_sum, modulus, gap,
            d_next_task, d_raw_sum, blocks, config.threads_per_block);
    }
    if (!cuda_call_succeeded(cudaGetLastError(),
                             "schur_necklace_pm32_kernel launch")) {
        goto cleanup;
    }
    if (!cuda_call_succeeded(cudaDeviceSynchronize(),
                             "schur_necklace_pm32_kernel execution")) {
        goto cleanup;
    }
    if (!cuda_call_succeeded(
            cudaMemcpy(&raw_sum, d_raw_sum, sizeof(unsigned long long),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(necklace result D2H)")) goto cleanup;

    raw_sum %= (uint64_t)p;
    {
        int64_t inverse_two = (p >> 1) + 1;
        int64_t scale = pow_mod_host(
            inverse_two, (uint64_t)minor_size, p);
        int64_t minor_permanent = mul_mod_host(
            (int64_t)raw_sum, scale, p);
        return_value = mul_mod_host(
            minor_permanent, normalize_mod_host((int64_t)n, p), p);
    }

cleanup:
    free(tasks);
    if (d_tasks && !cuda_call_succeeded(cudaFree(d_tasks),
                                         "cudaFree(necklace tasks)")) {
        return_value = -1;
    }
    if (d_base && !cuda_call_succeeded(cudaFree(d_base),
                                        "cudaFree(necklace base)")) {
        return_value = -1;
    }
    if (d_next_task && !cuda_call_succeeded(
            cudaFree(d_next_task), "cudaFree(necklace queue)")) {
        return_value = -1;
    }
    if (d_raw_sum && !cuda_call_succeeded(
            cudaFree(d_raw_sum), "cudaFree(necklace sum)")) {
        return_value = -1;
    }
    return return_value;
}

/*
 * Compute an n x n permanent with Glynn's formula.  The caller supplies an
 * odd modulus, making 2 invertible, and n >= 2.  This is kept private because
 * the exported general-matrix API deliberately retains Ryser's formula for
 * all supported moduli.
 */
static int64_t permanent_glynn_modp_cuda_internal(const int64_t *h_matrix,
                                                   int n,
                                                   int64_t p) {
    uint64_t num_states = (uint64_t)1 << (n - 1);
    const int threads_per_block = 256;
    uint64_t total_threads;
    uint64_t block_size;

    /* Keep enough independent blocks resident on large GPUs. */
    if (n >= 28) {
        total_threads = 1ULL << 18;
        block_size = (num_states + total_threads - 1) / total_threads;
    } else if (n >= 20) {
        total_threads = 1ULL << 14;
        block_size = (num_states + total_threads - 1) / total_threads;
    } else {
        total_threads = num_states;
        block_size = 1;
    }

    uint64_t num_blocks =
        (total_threads + threads_per_block - 1) / threads_per_block;
    uint64_t pm_gap = 0;
    int use_pm32 = 0;
    if ((uint64_t)p <= UINT32_MAX && glynn_pm32_size_supported(n)) {
        pm_gap = (1ULL << 32) - (uint64_t)p;
        use_pm32 = pm_gap >= 1 && pm_gap <= UINT16_MAX;
    }
    size_t matrix_entries = (size_t)n * (size_t)n;
    size_t matrix_bytes = matrix_entries *
        (use_pm32 ? sizeof(uint32_t) : sizeof(int64_t));

    void *h_normalized = malloc(matrix_bytes);
    void *d_matrix = NULL;
    uint64_t *d_partial_sums = NULL;
    uint64_t *d_reduce_sums = NULL;
    uint64_t *reduce_input = NULL;
    uint64_t *reduce_output = NULL;
    uint64_t reduce_size = total_threads;
    uint64_t result = 0;
    int64_t return_value = -1;

    if (!h_normalized) {
        fprintf(stderr, "failed to allocate normalized Glynn matrix\n");
        return -1;
    }
    if (use_pm32) {
        uint32_t *normalized32 = (uint32_t*)h_normalized;
        for (size_t i = 0; i < matrix_entries; i++) {
            normalized32[i] = (uint32_t)normalize_mod_host(h_matrix[i], p);
        }
    } else {
        int64_t *normalized64 = (int64_t*)h_normalized;
        for (size_t i = 0; i < matrix_entries; i++) {
            normalized64[i] = normalize_mod_host(h_matrix[i], p);
        }
    }

    if (!cuda_call_succeeded(cudaMalloc(&d_matrix, matrix_bytes),
                             "cudaMalloc(Glynn matrix)")) goto cleanup;
    if (!cuda_call_succeeded(
            cudaMalloc(&d_partial_sums,
                       total_threads * sizeof(uint64_t)),
            "cudaMalloc(Glynn partial_sums)")) goto cleanup;
    if (total_threads > 1 && !cuda_call_succeeded(
            cudaMalloc(&d_reduce_sums,
                       total_threads * sizeof(uint64_t)),
            "cudaMalloc(Glynn reduce_sums)")) goto cleanup;
    if (!cuda_call_succeeded(
            cudaMemcpy(d_matrix, h_normalized, matrix_bytes,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(Glynn matrix H2D)")) goto cleanup;

    if (use_pm32) {
        launch_glynn_pm32_kernel(
            (const uint32_t*)d_matrix, n, (uint32_t)p, (uint32_t)pm_gap,
            num_states, block_size, total_threads, num_blocks,
            threads_per_block, d_partial_sums);
    } else {
        permanent_glynn_modp_kernel<<<num_blocks, threads_per_block>>>(
            (const int64_t*)d_matrix, n, (uint64_t)p, num_states, block_size,
            total_threads, d_partial_sums);
    }
    if (!cuda_call_succeeded(cudaGetLastError(),
                             "permanent_glynn_modp_kernel launch")) {
        goto cleanup;
    }
    if (!cuda_call_succeeded(cudaDeviceSynchronize(),
                             "permanent_glynn_modp_kernel execution")) {
        goto cleanup;
    }

    reduce_input = d_partial_sums;
    reduce_output = d_reduce_sums;
    while (reduce_size > 1) {
        uint64_t reduce_blocks =
            (reduce_size + threads_per_block * 2 - 1) /
            (threads_per_block * 2);
        reduce_sum_modp<<<reduce_blocks, threads_per_block,
                          threads_per_block * sizeof(uint64_t)>>>(
            reduce_input, reduce_output, reduce_size, (uint64_t)p);
        if (!cuda_call_succeeded(cudaGetLastError(),
                                 "Glynn reduce_sum_modp launch")) {
            goto cleanup;
        }
        if (!cuda_call_succeeded(cudaDeviceSynchronize(),
                                 "Glynn reduce_sum_modp execution")) {
            goto cleanup;
        }
        reduce_size = reduce_blocks;
        uint64_t *tmp = reduce_input;
        reduce_input = reduce_output;
        reduce_output = tmp;
    }

    if (!cuda_call_succeeded(
            cudaMemcpy(&result, reduce_input, sizeof(uint64_t),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy(Glynn result D2H)")) goto cleanup;

    /* Glynn's normalization is 2^(1-n). */
    {
        int64_t inverse_two = (p >> 1) + 1;
        int64_t scale = pow_mod_host(inverse_two, (uint64_t)(n - 1), p);
        return_value = mul_mod_host((int64_t)result, scale, p);
    }

cleanup:
    free(h_normalized);
    if (d_matrix && !cuda_call_succeeded(cudaFree(d_matrix),
                                          "cudaFree(Glynn matrix)")) {
        return_value = -1;
    }
    if (d_partial_sums && !cuda_call_succeeded(
            cudaFree(d_partial_sums), "cudaFree(Glynn partial_sums)")) {
        return_value = -1;
    }
    if (d_reduce_sums && !cuda_call_succeeded(
            cudaFree(d_reduce_sums), "cudaFree(Glynn reduce_sums)")) {
        return_value = -1;
    }
    return return_value;
}

/*
 * Build and compute Schur matrix permanent mod p
 */
extern "C" int64_t schur_permanent_modp_cuda(int n, int64_t omega, int64_t p) {
    if (n < 0 || n >= 64) {
        fprintf(stderr, "n must satisfy 0 <= n < 64\n");
        return -1;
    }
    if (p <= 0) {
        fprintf(stderr, "p must be a positive signed 64-bit integer\n");
        return -1;
    }
    if (n == 0) return 1 % p;
    if (n == 1) return 1 % p;

    if (p == 1) return 0;

    int64_t omega_mod = normalize_mod_host(omega, p);

    /*
     * For odd n and omega^n = 1, expansion along the all-ones row gives
     *
     *   per(S_n) = n * per(S_n with row 0 and column 0 removed).
     *
     * Indeed, translating a removed column to column 0 scales row j by
     * omega^(j*c); their product is omega^(c*n*(n-1)/2) = 1 for odd n.
     * The minor has order n-1, and Glynn then fixes one sign, leaving
     * 2^(n-2) states.  An odd modulus is required to divide by 2.
     */
    if (n >= 3 && (n & 1) && (p & 1) &&
        pow_mod_host(omega_mod, (uint64_t)n, p) == 1 % p) {
#ifndef PERM_DISABLE_SCHUR_NECKLACE
        if (schur_necklace_preconditions(n, omega_mod, p)) {
            return schur_necklace_modp_cuda_internal(n, omega_mod, p);
        }
#endif
        int minor_n = n - 1;
        size_t minor_count = (size_t)minor_n * (size_t)minor_n;
        int64_t *minor = (int64_t*)malloc(minor_count * sizeof(int64_t));
        if (!minor) {
            fprintf(stderr, "failed to allocate Schur minor\n");
            return -1;
        }

        /* Rows and columns 1,...,n-1 of S_n. */
        int64_t omega_j = omega_mod;
        for (int row = 0; row < minor_n; row++) {
            int64_t power = omega_j;
            for (int col = 0; col < minor_n; col++) {
                minor[row * minor_n + col] = power;
                power = mul_mod_host(power, omega_j, p);
            }
            omega_j = mul_mod_host(omega_j, omega_mod, p);
        }

        int64_t minor_permanent =
            permanent_glynn_modp_cuda_internal(minor, minor_n, p);
        free(minor);
        if (minor_permanent < 0) return -1;
        return mul_mod_host(minor_permanent,
                            normalize_mod_host((int64_t)n, p), p);
    }

    /* Build Schur matrix: S[j,k] = omega^(j*k) mod p */
    int64_t *matrix = (int64_t*)malloc(
        (size_t)n * (size_t)n * sizeof(int64_t));
    if (!matrix) {
        fprintf(stderr, "failed to allocate Schur matrix\n");
        return -1;
    }

    int64_t omega_j = 1 % p;
    for (int j = 0; j < n; j++) {
        int64_t power = 1 % p;
        for (int k = 0; k < n; k++) {
            matrix[j * n + k] = power;
            power = mul_mod_host(power, omega_j, p);
        }
        omega_j = mul_mod_host(omega_j, omega_mod, p);
    }

    int64_t result = permanent_modp_cuda(matrix, n, p);

    free(matrix);
    return result;
}

#ifdef TEST_MAIN
/* Find primitive root mod p */
int64_t primitive_root(int64_t p) {
    for (int64_t g = 2; g < p; g++) {
        int64_t val = 1;
        int is_primitive = 1;
        for (int64_t i = 1; i < p - 1; i++) {
            val = (val * g) % p;
            if (val == 1) {
                is_primitive = 0;
                break;
            }
        }
        if (is_primitive) return g;
    }
    return -1;
}

int main(int argc, char **argv) {
    int n = 37;
    if (argc > 1) n = atoi(argv[1]);

    /* Find smallest prime p ≡ 1 (mod n) */
    int64_t p = n + 1;
    while (1) {
        int is_prime = 1;
        for (int64_t i = 2; i * i <= p; i++) {
            if (p % i == 0) { is_prime = 0; break; }
        }
        if (is_prime && (p - 1) % n == 0) break;
        p++;
    }

    int64_t g = primitive_root(p);
    int64_t omega = 1;
    int64_t exp = (p - 1) / n;
    for (int64_t i = 0; i < exp; i++) {
        omega = (omega * g) % p;
    }

    printf("Schur permanent mod %lld for n=%d\n", (long long)p, n);
    printf("omega = %lld (primitive %d-th root of unity)\n", (long long)omega, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int64_t result = schur_permanent_modp_cuda(n, omega, p);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Result: %lld\n", (long long)result);
    printf("Time: %.3f s\n", milliseconds / 1000.0);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
#endif
