// Exact modular weighted partial-matching frontier (research baseline).
//
// Build, for example:
//   c++ -std=c++17 -O3 -fPIC -shared weighted_matching_frontier_native.cpp
//       -o weighted_matching_frontier_native.so   (on the same command line)
//
// ABI version 1. Rows have already been ordered, and global column identities
// have been assigned reusable one-hot uint64 slots by the caller. A slot can
// be reused only after its previous column was forgotten. This permits more
// than 64 global columns, but at most 64 columns during any individual row.
// edge_bits and canonical weights describe the ORIGINAL nonzero support:
// a weight that becomes zero modulo p must still retain its structural edge.
// live_after gives the slots retained after each row, including unused slots.
//
// Each layer maps a used-slot mask to a truncated cardinality polynomial.
// Skip-row and choose-one-unused-edge transitions are both allowed; forgetting
// a column never requires it to have been matched. Structural states are kept
// even if their whole polynomial is zero modulo p (or beyond the truncation).
//
// max_bytes caps the sum of requested allocation sizes for all live native DP
// allocations: hash nodes, hash bucket arrays, and coefficient vectors. Both
// layers and transient allocations during rehash/growth share this one cap.
// It excludes allocator bookkeeping, caller-owned input/output buffers, stack,
// and code/runtime memory; it is not a process RSS limit. max_states separately
// limits the number of structural states in each layer.
//
// Return: 0 success, 1 invalid input, 2 requested-byte limit, 3 state limit,
//         4 underlying allocation failure, 5 unexpected internal failure.
// stats[0..4]: peak requested bytes, peak successful states in any layer,
//              completed rows, failed processing row (UINT64_MAX if none or
//              before processing), maximum attempted states in any layer.
// On success attempted states equals peak states. On state-limit failure it
// equals max_states + 1. All counters describe one modular pass.
//
// The caller supplies correctly sized accessible arrays (as for any C ABI).
// offsets has rows+1 items even when rows is zero; out has degree+1 items;
// stats has five items. out and stats must not overlap. Null edge arrays are
// allowed only when edge_count is zero; null live_after only when rows is zero.
// The output polynomial stays untouched on every error. No exception escapes
// this ABI. There is no mutable global state.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::uint64_t no_row = std::numeric_limits<std::uint64_t>::max();

struct ByteLimit {};
struct StateLimit {};
struct InternalFailure {};

struct Budget {
    explicit Budget(std::uint64_t maximum) noexcept : limit(maximum) {}
    std::uint64_t limit;
    std::uint64_t live = 0;
    std::uint64_t peak = 0;
};

// unordered_map rebinds this allocator for BOTH nodes and buckets. The vector
// allocator receives the same Budget pointer explicitly; no DP container uses
// its default allocator. Charge before allocation and roll back on failure.
template <class T>
class BudgetAllocator {
public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    Budget* budget = nullptr;

    BudgetAllocator() noexcept = default;
    explicit BudgetAllocator(Budget* value) noexcept : budget(value) {}

    template <class U>
    BudgetAllocator(const BudgetAllocator<U>& other) noexcept
        : budget(other.budget) {}

    T* allocate(std::size_t count) {
        if (budget == nullptr) {
            throw InternalFailure{};
        }
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T) ||
            count > std::numeric_limits<std::uint64_t>::max() / sizeof(T)) {
            throw ByteLimit{};
        }
        const auto bytes = static_cast<std::uint64_t>(count) * sizeof(T);
        if (bytes > budget->limit - budget->live) {
            throw ByteLimit{};
        }
        budget->live += bytes;
        try {
            T* result = std::allocator<T>{}.allocate(count);
            budget->peak = std::max(budget->peak, budget->live);
            return result;
        } catch (...) {
            budget->live -= bytes;
            throw;
        }
    }

    void deallocate(T* pointer, std::size_t count) noexcept {
        std::allocator<T>{}.deallocate(pointer, count);
        budget->live -= static_cast<std::uint64_t>(count) * sizeof(T);
    }

    template <class U>
    bool operator==(const BudgetAllocator<U>& other) const noexcept {
        return budget == other.budget;
    }

    template <class U>
    bool operator!=(const BudgetAllocator<U>& other) const noexcept {
        return !(*this == other);
    }
};

using Polynomial = std::vector<std::uint32_t, BudgetAllocator<std::uint32_t>>;
using Entry = std::pair<const std::uint64_t, Polynomial>;
using States = std::unordered_map<std::uint64_t, Polynomial,
                                  std::hash<std::uint64_t>,
                                  std::equal_to<std::uint64_t>,
                                  BudgetAllocator<Entry>>;

template <class T>
bool representable_array(std::uint64_t count) noexcept {
    // Reject overflow not only in element counts but also byte offsets and
    // pointer differences before reading caller-owned arrays.
    return count <= std::numeric_limits<std::size_t>::max() / sizeof(T) &&
           count <= static_cast<std::uint64_t>(
                        std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(T);
}

bool output_stats_overlap(const std::uint32_t* out, std::uint64_t degree,
                          const std::uint64_t* stats) noexcept {
    const auto output_address = reinterpret_cast<std::uintptr_t>(out);
    const auto stats_address = reinterpret_cast<std::uintptr_t>(stats);
    const auto output_bytes = static_cast<std::uintptr_t>(degree + 1) *
                              sizeof(std::uint32_t);
    const auto stats_bytes = std::uintptr_t{5} * sizeof(std::uint64_t);
    // Differences avoid wrapping an end address. Counts were checked first.
    return output_address <= stats_address
               ? stats_address - output_address < output_bytes
               : output_address - stats_address < stats_bytes;
}

bool valid_plan(std::uint64_t rows, std::uint64_t degree,
                std::uint64_t edge_count, const std::uint64_t* offsets,
                const std::uint64_t* edge_bits, const std::uint32_t* weights,
                const std::uint64_t* live_after, std::uint32_t modulus,
                std::uint64_t max_states, std::uint64_t max_bytes) noexcept {
    if (rows == no_row || degree > rows || max_states == 0 || max_bytes == 0 ||
        modulus < 3 || modulus >= (std::uint32_t{1} << 31) ||
        (modulus & 1U) == 0 || offsets == nullptr ||
        (rows != 0 && live_after == nullptr) ||
        (edge_count != 0 && (edge_bits == nullptr || weights == nullptr)) ||
        !representable_array<std::uint64_t>(rows + 1) ||
        !representable_array<std::uint32_t>(degree + 1) ||
        !representable_array<std::uint64_t>(edge_count) ||
        !representable_array<std::uint32_t>(edge_count)) {
        return false;
    }
    if (offsets[0] != 0 || offsets[rows] != edge_count) {
        return false;
    }
    std::uint64_t live = 0;
    for (std::uint64_t row = 0; row < rows; ++row) {
        const auto begin = offsets[row];
        const auto end = offsets[row + 1];
        if (begin > end || end > edge_count || end - begin > 64) {
            return false;
        }
        std::uint64_t row_bits = 0;
        for (auto edge = begin; edge < end; ++edge) {
            const auto bit = edge_bits[edge];
            if (bit == 0 || (bit & (bit - 1)) != 0 ||
                (bit & row_bits) != 0 || weights[edge] >= modulus) {
                return false;
            }
            row_bits |= bit;
        }
        if ((live_after[row] & ~(live | row_bits)) != 0) {
            return false;
        }
        live = live_after[row];
    }
    return live == 0;
}

// Allocate a new state even when the contribution is zero. In particular,
// modular zero weights must not change the reachable-state count across CRT
// primes, nor hide a structural state-limit violation.
void accumulate(States& next, std::uint64_t key, const Polynomial& source,
                std::size_t shift, std::uint32_t weight, std::size_t length,
                std::uint32_t modulus, std::uint64_t max_states,
                Budget& budget, std::uint64_t& peak_states,
                std::uint64_t& attempted_states) {
    auto found = next.find(key);
    if (found == next.end()) {
        const auto candidate = static_cast<std::uint64_t>(next.size()) + 1;
        attempted_states = std::max(attempted_states, candidate);
        if (candidate > max_states) {
            throw StateLimit{};
        }
        found = next.emplace(
            std::piecewise_construct, std::forward_as_tuple(key),
            std::forward_as_tuple(length, std::uint32_t{0},
                                  BudgetAllocator<std::uint32_t>(&budget)))
                    .first;
        peak_states = std::max(peak_states,
                               static_cast<std::uint64_t>(next.size()));
    }
    auto& target = found->second;
    for (std::size_t k = 0; k < source.size() && k + shift < length; ++k) {
        // modulus < 2^31 makes product + accumulator strictly less than 2^62.
        // There is no floating-point arithmetic and no integer approximation.
        const auto value = static_cast<std::uint64_t>(source[k]) * weight +
                           target[k + shift];
        target[k + shift] = static_cast<std::uint32_t>(value % modulus);
    }
}

void publish_stats(std::uint64_t* output, const Budget& budget,
                   std::uint64_t peak_states, std::uint64_t processed_rows,
                   std::uint64_t failure_row,
                   std::uint64_t attempted_states) noexcept {
    if (output != nullptr) {
        output[0] = budget.peak;
        output[1] = peak_states;
        output[2] = processed_rows;
        output[3] = failure_row;
        output[4] = attempted_states;
    }
}

}  // namespace

extern "C" std::uint32_t weighted_matching_frontier_abi_version() noexcept {
    return 1;
}

extern "C" int weighted_matching_frontier_mod(
    std::uint64_t rows, std::uint64_t degree, std::uint64_t edge_count,
    const std::uint64_t* offsets, const std::uint64_t* edge_bits,
    const std::uint32_t* weights, const std::uint64_t* live_after,
    std::uint32_t modulus, std::uint64_t max_states, std::uint64_t max_bytes,
    std::uint32_t* out, std::uint64_t* stats) noexcept {
    Budget budget(max_bytes);
    std::uint64_t peak_states = 0;
    std::uint64_t processed_rows = 0;
    std::uint64_t failure_row = no_row;
    std::uint64_t attempted_states = 0;

    // Detect overlap before publishing diagnostic counters, so even malformed
    // aliases cannot alter the polynomial output on this error path.
    if (out != nullptr && stats != nullptr && degree != no_row &&
        representable_array<std::uint32_t>(degree + 1) &&
        output_stats_overlap(out, degree, stats)) {
        return 1;
    }
    if (out == nullptr || stats == nullptr ||
        !valid_plan(rows, degree, edge_count, offsets, edge_bits, weights,
                    live_after, modulus, max_states, max_bytes)) {
        publish_stats(stats, budget, 0, 0, no_row, 0);
        return 1;
    }

    int status = 0;
    try {
        if (rows == 0) {
            // The empty polynomial needs no dynamic storage.
            out[0] = 1;
            publish_stats(stats, budget, 1, 0, no_row, 1);
            return 0;
        }
        States states(0, std::hash<std::uint64_t>{},
                      std::equal_to<std::uint64_t>{},
                      BudgetAllocator<Entry>(&budget));
        attempted_states = 1;
        auto initial = states.emplace(
            std::piecewise_construct, std::forward_as_tuple(std::uint64_t{0}),
            std::forward_as_tuple(std::size_t{1}, std::uint32_t{1},
                                  BudgetAllocator<std::uint32_t>(&budget)));
        if (!initial.second) {
            throw InternalFailure{};
        }
        peak_states = 1;

        for (std::uint64_t row = 0; row < rows; ++row) {
            failure_row = row;
            const auto length = static_cast<std::size_t>(
                std::min(row + 1, degree) + 1);
            States next(0, std::hash<std::uint64_t>{},
                        std::equal_to<std::uint64_t>{},
                        BudgetAllocator<Entry>(&budget));
            for (const auto& entry : states) {
                const auto used = entry.first;
                const auto& polynomial = entry.second;
                accumulate(next, used & live_after[row], polynomial, 0, 1,
                           length, modulus, max_states, budget, peak_states,
                           attempted_states);
                for (auto edge = offsets[row]; edge < offsets[row + 1]; ++edge) {
                    const auto bit = edge_bits[edge];
                    if ((used & bit) == 0) {
                        accumulate(next, (used | bit) & live_after[row],
                                   polynomial, 1, weights[edge], length,
                                   modulus, max_states, budget, peak_states,
                                   attempted_states);
                    }
                }
            }
            states.swap(next);
            ++processed_rows;
            // Destruction of next releases the old layer before the next row.
        }
        failure_row = no_row;
        const auto result = states.find(0);
        if (states.size() != 1 || result == states.end() ||
            result->second.size() != degree + 1) {
            throw InternalFailure{};
        }
        std::copy(result->second.begin(), result->second.end(), out);
    } catch (const ByteLimit&) {
        status = 2;
    } catch (const StateLimit&) {
        status = 3;
    } catch (const std::bad_alloc&) {
        status = 4;
    } catch (...) {
        status = 5;
    }
    publish_stats(stats, budget, peak_states, processed_rows,
                  failure_row, attempted_states);
    return status;
}
