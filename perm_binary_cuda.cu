/*
 * Experimental CUDA backend for permanents of zero-one matrices.
 *
 * This file is intentionally source-only: it is not part of the Python
 * package build.  Fixed-sign Glynn leaves 2^(n-1) sign states.  Columns are
 * stored as sparse row lists, with low-degree columns on the frequently
 * toggled Gray bits and a densest column fixed at +1.  Per-thread row sums
 * are packed signed bytes.  A zero counter and the product of all nonzero
 * row sums update each term by ratios instead of rebuilding an n-factor
 * product.
 *
 * Arithmetic uses 32-bit Montgomery residues.  The ABI therefore accepts
 * only odd primes n < p < 2^31, exactly matching the planner in
 * perm_binary.py.  binary_permanent_modp_cuda returns a residue in [0,p) on
 * success and -1 on every validation, runtime, launch, or cleanup failure.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <new>
#include <vector>

namespace {

constexpr int kMaxOrder = 63;
constexpr int kValueCount = 2 * kMaxOrder + 1;
constexpr int kMaxNonzeros = kMaxOrder * kMaxOrder;
constexpr int kThreadsPerBlock = 256;
constexpr uint64_t kTargetTasks = UINT64_C(1) << 20;

/* Constant memory is shared by launches, so the host ABI serializes calls. */
std::mutex call_mutex;

__constant__ uint16_t constant_offsets[kMaxOrder];
__constant__ uint8_t constant_rows[kMaxNonzeros];
__constant__ uint64_t constant_base_words[8];
__constant__ uint32_t constant_value_mont[kValueCount];
__constant__ uint32_t constant_transition[2][kValueCount];

struct PackedSignedBytes {
    uint64_t w0, w1, w2, w3, w4, w5, w6, w7;

    __device__ __forceinline__ uint64_t get_word(int index) const {
        switch (index) {
            case 0: return w0;
            case 1: return w1;
            case 2: return w2;
            case 3: return w3;
            case 4: return w4;
            case 5: return w5;
            case 6: return w6;
            default: return w7;
        }
    }

    __device__ __forceinline__ void set_word(int index, uint64_t value) {
        switch (index) {
            case 0: w0 = value; break;
            case 1: w1 = value; break;
            case 2: w2 = value; break;
            case 3: w3 = value; break;
            case 4: w4 = value; break;
            case 5: w5 = value; break;
            case 6: w6 = value; break;
            default: w7 = value; break;
        }
    }

    __device__ __forceinline__ int get(int row) const {
        int shift = (row & 7) * 8;
        return (int)(int8_t)(uint8_t)(get_word(row >> 3) >> shift);
    }

    __device__ __forceinline__ void set(int row, int value) {
        int word_index = row >> 3;
        int shift = (row & 7) * 8;
        uint64_t mask = UINT64_C(0xff) << shift;
        uint64_t word = get_word(word_index);
        word = (word & ~mask) |
            ((uint64_t)(uint8_t)(int8_t)value << shift);
        set_word(word_index, word);
    }
};

__device__ __forceinline__ uint32_t add_mod(uint32_t a, uint32_t b,
                                             uint32_t p) {
    /* p < 2^31 makes this uint32_t addition overflow-free. */
    uint32_t sum = a + b;
    return sum >= p ? sum - p : sum;
}

__device__ __forceinline__ uint32_t sub_mod(uint32_t a, uint32_t b,
                                             uint32_t p) {
    return a >= b ? a - b : p - (b - a);
}

__device__ __forceinline__ uint32_t multiply_montgomery(
        uint32_t a, uint32_t b, uint32_t p, uint32_t negative_inverse) {
    uint64_t product = (uint64_t)a * (uint64_t)b;
    uint32_t multiplier = (uint32_t)product * negative_inverse;
    uint64_t reduced =
        (product + (uint64_t)multiplier * (uint64_t)p) >> 32;
    if (reduced >= p) reduced -= p;
    return (uint32_t)reduced;
}

__global__ void binary_glynn_ratio_kernel(
        int n, uint32_t p, uint32_t negative_inverse, uint32_t one_mont,
        uint64_t states, uint64_t states_per_task, uint64_t task_count,
        unsigned long long *block_sum) {
    __shared__ uint32_t reductions[kThreadsPerBlock];
    uint64_t task = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t local_sum = 0;

    if (task < task_count) {
        uint64_t start = task * states_per_task;
        uint64_t end = start + states_per_task;
        if (end > states) end = states;

        PackedSignedBytes sums = {
            constant_base_words[0], constant_base_words[1],
            constant_base_words[2], constant_base_words[3],
            constant_base_words[4], constant_base_words[5],
            constant_base_words[6], constant_base_words[7]
        };

        /* Initialize the signed row sums at this task's first Gray state. */
        uint64_t gray = start ^ (start >> 1);
        uint64_t selected = gray;
        while (selected != 0) {
            int column = __ffsll((long long)selected) - 1;
            uint16_t begin = constant_offsets[column];
            uint16_t finish = constant_offsets[column + 1];
            for (uint16_t index = begin; index < finish; index++) {
                int row = constant_rows[index];
                sums.set(row, sums.get(row) - 2);
            }
            selected &= selected - 1;
        }

        int zero_count = 0;
        uint32_t product = one_mont;
        for (int row = 0; row < n; row++) {
            int value = sums.get(row);
            if (value == 0) {
                zero_count++;
            } else {
                product = multiply_montgomery(
                    product, constant_value_mont[value + kMaxOrder], p,
                    negative_inverse);
            }
        }

        int state_sign =
            (__popcll((unsigned long long)gray) & 1) ? -1 : 1;
        for (uint64_t state = start; state < end; state++) {
            if (state != start) {
                /* BRGC state i flips bit ctz(i). */
                int column = __ffsll((long long)state) - 1;
                gray = state ^ (state >> 1);
                int direction = ((gray >> column) & 1U) ? 0 : 1;
                int delta = direction ? 2 : -2;
                uint16_t begin = constant_offsets[column];
                uint16_t finish = constant_offsets[column + 1];
                for (uint16_t index = begin; index < finish; index++) {
                    int row = constant_rows[index];
                    int old_value = sums.get(row);
                    int new_value = old_value + delta;
                    product = multiply_montgomery(
                        product,
                        constant_transition[direction]
                                           [old_value + kMaxOrder],
                        p, negative_inverse);
                    if (old_value == 0) zero_count--;
                    if (new_value == 0) zero_count++;
                    sums.set(row, new_value);
                }
                state_sign = -state_sign;
            }

            if (zero_count == 0) {
                local_sum = state_sign < 0
                    ? sub_mod(local_sum, product, p)
                    : add_mod(local_sum, product, p);
            }
        }
    }

    reductions[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = kThreadsPerBlock / 2; stride != 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reductions[threadIdx.x] = add_mod(
                reductions[threadIdx.x], reductions[threadIdx.x + stride], p);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(block_sum, (unsigned long long)reductions[0]);
    }
}

uint64_t multiply_mod_host(uint64_t a, uint64_t b, uint64_t p) {
    return (uint64_t)((__uint128_t)a * (__uint128_t)b % p);
}

uint64_t power_mod_host(uint64_t base, uint64_t exponent, uint64_t p) {
    uint64_t result = 1 % p;
    while (exponent != 0) {
        if (exponent & 1U) result = multiply_mod_host(result, base, p);
        base = multiply_mod_host(base, base, p);
        exponent >>= 1;
    }
    return result;
}

bool is_prime_u32(uint32_t value) {
    if (value < 2) return false;
    if ((value & 1U) == 0) return value == 2;
    for (uint32_t factor = 3;
         (uint64_t)factor * (uint64_t)factor <= value; factor += 2) {
        if (value % factor == 0) return false;
    }
    return true;
}

uint32_t montgomery_negative_inverse(uint32_t p) {
    uint32_t inverse = p;
    inverse *= 2U - p * inverse;
    inverse *= 2U - p * inverse;
    inverse *= 2U - p * inverse;
    inverse *= 2U - p * inverse;
    inverse *= 2U - p * inverse;
    return 0U - inverse;
}

uint32_t multiply_montgomery_host(uint32_t a, uint32_t b, uint32_t p,
                                  uint32_t negative_inverse) {
    uint64_t product = (uint64_t)a * (uint64_t)b;
    uint32_t multiplier = (uint32_t)product * negative_inverse;
    uint64_t reduced =
        (product + (uint64_t)multiplier * (uint64_t)p) >> 32;
    if (reduced >= p) reduced -= p;
    return (uint32_t)reduced;
}

struct SparseOrientation {
    std::vector<uint16_t> offsets;
    std::vector<uint8_t> rows;
    uint64_t base_words[8] = {};
    __uint128_t gray_cost = 0;
};

SparseOrientation make_orientation(const std::vector<uint8_t> &matrix,
                                   int n, bool transpose) {
    std::vector<std::vector<uint8_t>> columns(n);
    std::vector<int> row_degrees(n, 0);
    for (int row = 0; row < n; row++) {
        for (int column = 0; column < n; column++) {
            uint8_t entry = transpose
                ? matrix[(size_t)column * n + row]
                : matrix[(size_t)row * n + column];
            if (entry) {
                columns[column].push_back((uint8_t)row);
                row_degrees[row]++;
            }
        }
    }

    int fixed = 0;
    for (int column = 1; column < n; column++) {
        if (columns[column].size() > columns[fixed].size()) fixed = column;
    }
    std::vector<int> variables;
    variables.reserve(n - 1);
    for (int column = 0; column < n; column++) {
        if (column != fixed) variables.push_back(column);
    }
    std::stable_sort(variables.begin(), variables.end(), [&](int a, int b) {
        return columns[a].size() < columns[b].size();
    });

    SparseOrientation output;
    output.offsets.reserve(n);
    output.rows.reserve(matrix.size());
    output.offsets.push_back(0);
    for (int bit = 0; bit < n - 1; bit++) {
        const std::vector<uint8_t> &list = columns[variables[bit]];
        output.rows.insert(output.rows.end(), list.begin(), list.end());
        output.offsets.push_back((uint16_t)output.rows.size());
        output.gray_cost +=
            (__uint128_t)list.size() << (n - 2 - bit);
    }
    for (int row = 0; row < n; row++) {
        int word = row >> 3;
        int shift = (row & 7) * 8;
        output.base_words[word] |=
            (uint64_t)(uint8_t)(int8_t)row_degrees[row] << shift;
    }
    return output;
}

bool cuda_succeeded(cudaError_t error, const char *operation) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(error));
    return false;
}

int64_t binary_permanent_modp_cuda_impl(const uint8_t *matrix, int n,
                                        int64_t signed_p) {
    if (n < 0 || n > kMaxOrder) {
        std::fprintf(stderr, "n must satisfy 0 <= n <= %d\n", kMaxOrder);
        return -1;
    }
    if (signed_p <= n || signed_p <= 2 ||
        signed_p >= (INT64_C(1) << 31) ||
        !is_prime_u32((uint32_t)signed_p)) {
        std::fprintf(
            stderr,
            "p must be an odd prime satisfying n < p < 2^31\n");
        return -1;
    }
    uint32_t p = (uint32_t)signed_p;
    if (n == 0) return 1 % p;
    if (!matrix) {
        std::fprintf(stderr, "matrix must not be NULL\n");
        return -1;
    }

    std::vector<uint8_t> input((size_t)n * (size_t)n);
    std::vector<int> row_degrees(n, 0);
    std::vector<int> column_degrees(n, 0);
    for (int row = 0; row < n; row++) {
        for (int column = 0; column < n; column++) {
            size_t index = (size_t)row * n + column;
            uint8_t entry = matrix[index];
            if (entry > 1) {
                std::fprintf(stderr, "matrix entries must be 0 or 1\n");
                return -1;
            }
            input[index] = entry;
            row_degrees[row] += entry;
            column_degrees[column] += entry;
        }
    }
    if (n == 1) return input[0];
    if (std::find(row_degrees.begin(), row_degrees.end(), 0) !=
            row_degrees.end() ||
        std::find(column_degrees.begin(), column_degrees.end(), 0) !=
            column_degrees.end()) {
        return 0;
    }

    SparseOrientation normal = make_orientation(input, n, false);
    SparseOrientation transposed = make_orientation(input, n, true);
    const SparseOrientation &oriented =
        transposed.gray_cost < normal.gray_cost ? transposed : normal;

    uint16_t offsets[kMaxOrder] = {};
    uint8_t rows[kMaxNonzeros] = {};
    for (size_t index = 0; index < oriented.offsets.size(); index++) {
        offsets[index] = oriented.offsets[index];
    }
    for (size_t index = 0; index < oriented.rows.size(); index++) {
        rows[index] = oriented.rows[index];
    }

    uint32_t negative_inverse = montgomery_negative_inverse(p);
    uint32_t one_mont = (uint32_t)((UINT64_C(1) << 32) % p);
    uint32_t values[kValueCount] = {};
    uint32_t transitions[2][kValueCount] = {};
    for (int value = -kMaxOrder; value <= kMaxOrder; value++) {
        uint64_t normalized =
            value < 0 ? p - (uint32_t)(-value) : (uint32_t)value;
        values[value + kMaxOrder] =
            (uint32_t)(normalized * one_mont % p);
        for (int direction = 0; direction < 2; direction++) {
            int next = value + (direction ? 2 : -2);
            if (next < -kMaxOrder || next > kMaxOrder) continue;
            uint64_t normalized_next =
                next < 0 ? p - (uint32_t)(-next) : (uint32_t)next;
            uint64_t ratio;
            if (value == 0) {
                ratio = normalized_next;
            } else {
                uint64_t inverse = power_mod_host(normalized, p - 2, p);
                ratio = next == 0
                    ? inverse
                    : multiply_mod_host(normalized_next, inverse, p);
            }
            transitions[direction][value + kMaxOrder] =
                (uint32_t)(ratio * one_mont % p);
        }
    }

    /* Constant symbols make warp accesses broadcast efficiently. */
    if (!cuda_succeeded(cudaMemcpyToSymbol(
            constant_offsets, offsets, sizeof(offsets)),
            "cudaMemcpyToSymbol(binary offsets)")) return -1;
    if (!cuda_succeeded(cudaMemcpyToSymbol(
            constant_rows, rows, sizeof(rows)),
            "cudaMemcpyToSymbol(binary rows)")) return -1;
    if (!cuda_succeeded(cudaMemcpyToSymbol(
            constant_base_words, oriented.base_words,
            sizeof(oriented.base_words)),
            "cudaMemcpyToSymbol(binary base sums)")) return -1;
    if (!cuda_succeeded(cudaMemcpyToSymbol(
            constant_value_mont, values, sizeof(values)),
            "cudaMemcpyToSymbol(binary values)")) return -1;
    if (!cuda_succeeded(cudaMemcpyToSymbol(
            constant_transition, transitions, sizeof(transitions)),
            "cudaMemcpyToSymbol(binary transitions)")) return -1;

    uint64_t states = UINT64_C(1) << (n - 1);
    uint64_t task_count = std::min(states, kTargetTasks);
    uint64_t states_per_task = states / task_count;
    uint64_t blocks =
        (task_count + kThreadsPerBlock - 1) / kThreadsPerBlock;

    unsigned long long *device_sum = nullptr;
    unsigned long long host_sum = 0;
    int64_t return_value = -1;
    if (!cuda_succeeded(cudaMalloc(&device_sum, sizeof(*device_sum)),
                        "cudaMalloc(binary sum)")) goto cleanup;
    if (!cuda_succeeded(cudaMemset(device_sum, 0, sizeof(*device_sum)),
                        "cudaMemset(binary sum)")) goto cleanup;

    binary_glynn_ratio_kernel<<<(uint32_t)blocks, kThreadsPerBlock>>>(
        n, p, negative_inverse, one_mont, states, states_per_task,
        task_count, device_sum);
    if (!cuda_succeeded(cudaGetLastError(),
                        "binary_glynn_ratio_kernel launch")) goto cleanup;
    if (!cuda_succeeded(cudaDeviceSynchronize(),
                        "binary_glynn_ratio_kernel execution")) goto cleanup;
    if (!cuda_succeeded(cudaMemcpy(
            &host_sum, device_sum, sizeof(host_sum), cudaMemcpyDeviceToHost),
            "cudaMemcpy(binary sum D2H)")) goto cleanup;

    {
        /* At most 4096 block residues were atomically accumulated. */
        uint32_t summed_mont = (uint32_t)(host_sum % p);
        uint32_t unscaled = multiply_montgomery_host(
            summed_mont, 1, p, negative_inverse);
        uint32_t inverse_two = (p >> 1) + 1;
        uint32_t scale = (uint32_t)power_mod_host(
            inverse_two, (uint64_t)(n - 1), p);
        return_value = (int64_t)multiply_mod_host(unscaled, scale, p);
    }

cleanup:
    if (device_sum &&
        !cuda_succeeded(cudaFree(device_sum), "cudaFree(binary sum)")) {
        return_value = -1;
    }
    return return_value;
}

}  // namespace

extern "C" int cuda_runtime_available(void) {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}

extern "C" int64_t binary_permanent_modp_cuda(
        const uint8_t *matrix, int n, int64_t p) {
    try {
        std::lock_guard<std::mutex> lock(call_mutex);
        return binary_permanent_modp_cuda_impl(matrix, n, p);
    } catch (const std::bad_alloc &) {
        std::fprintf(stderr, "binary CUDA host allocation failed\n");
    } catch (...) {
        std::fprintf(stderr, "binary CUDA backend failed unexpectedly\n");
    }
    return -1;
}
