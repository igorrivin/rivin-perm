// Independent occupied-column DP for an exact weighted matching query.
// For activity a/b, each unmatched row contributes b and each matched edge
// contributes a*Aij. No modular inverse of b is ever required.
// Descending masks let one array carry old and new row layers safely.
#include <cstdint>
#include <new>
#include <vector>

namespace {
struct Field {
    std::uint32_t prime, complement;
    bool folded;
    explicit Field(std::uint32_t p)
        : prime(p), complement((std::uint32_t(1) << 31) - p),
          folded(complement <= 32768) {}

    std::uint32_t mul(std::uint32_t a, std::uint32_t b) const {
        std::uint64_t x = std::uint64_t(a) * b;
        if (!folded) return std::uint32_t(x % prime);
        // a,b<p<2^31 and c=2^31-p<=32768. After two folds,
        // x <= 2^31-1+c^2 < 2*p, so one subtraction suffices.
        constexpr std::uint64_t mask = (std::uint64_t(1) << 31) - 1;
        x = (x & mask) + (x >> 31) * complement;
        x = (x & mask) + (x >> 31) * complement;
        return std::uint32_t(x >= prime ? x - prime : x);
    }

    std::uint32_t add(std::uint32_t a, std::uint32_t b) const {
        const std::uint32_t value = a + b; // <=2*p-2<2^32
        return value >= prime ? value - prime : value;
    }
};
}

// Return 0 on success, 1 for invalid inputs, 2 for allocation failure,
// 3 for another exception. Matrix entries and activity values are canonical
// residues; the caller supplies a valid prime. The field arithmetic itself
// works for every odd modulus in the accepted range.
extern "C" int weighted_matching_partition_mod(
    const std::uint32_t* matrix, int n, std::uint32_t prime,
    std::uint32_t numerator, std::uint32_t denominator,
    std::uint32_t* out) noexcept {
    if (n < 0 || n > 20 || !out || (n && !matrix) || prime < 3 ||
        prime >= (std::uint32_t(1) << 31) || !(prime & 1) ||
        numerator >= prime || denominator >= prime) return 1;
    for (int i = 0; i < n * n; ++i) if (matrix[i] >= prime) return 1;
    try {
        const Field field(prime);
        const std::uint32_t count = std::uint32_t(1) << n;
        const std::uint32_t full = count - 1;
        std::vector<std::uint32_t> dp(count, 0);
        std::vector<std::uint32_t> edges(n);
        dp[0] = 1;
        for (int row = 0; row < n; ++row) {
            std::uint32_t support = 0;
            for (int col = 0; col < n; ++col) {
                edges[col] = field.mul(numerator, matrix[row * n + col]);
                if (edges[col]) support |= std::uint32_t(1) << col;
            }
            for (std::uint32_t mask = count; mask-- > 0;) {
                const std::uint32_t old = dp[mask];
                if (!old) continue;
                dp[mask] = field.mul(denominator, old);
                std::uint32_t available = (full ^ mask) & support;
                while (available) {
                    const std::uint32_t bit = available & (0u - available);
                    const unsigned col = __builtin_ctz(available);
                    available ^= bit;
                    const std::uint32_t target = mask | bit;
                    dp[target] = field.add(dp[target], field.mul(edges[col], old));
                }
            }
        }
        std::uint32_t zero = 0, first = 0, second = 0;
        for (std::uint32_t mask = 0; mask < count; ++mask) {
            const std::uint32_t value = dp[mask];
            const unsigned k = __builtin_popcount(mask);
            zero = field.add(zero, value);
            first = field.add(first, field.mul(k % prime, value));
            second = field.add(second, field.mul((k * k) % prime, value));
        }
        out[0] = zero;
        out[1] = first;
        out[2] = second;
        return 0;
    } catch (const std::bad_alloc&) {
        return 2;
    } catch (...) {
        return 3;
    }
}
