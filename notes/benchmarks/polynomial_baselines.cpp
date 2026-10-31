// Independent CPU baselines for complete modular permanent coefficient vectors.
// Polynomial-valued Ryser is established prior art; see Gao et al. (2009), III.C.
// This implementation does not call the production polynomial implementation.
#include <algorithm>
#include <cstdint>
#include <new>
#include <vector>

namespace {
constexpr uint32_t M31 = 2147483647U;

template<bool mersenne> struct Field {
    uint32_t p;
    uint32_t reduce(uint64_t x) const {
        if constexpr (mersenne) {
            x = (x & M31) + (x >> 31);
            x = (x & M31) + (x >> 31);
            return uint32_t(x >= M31 ? x - M31 : x);
        }
        return uint32_t(x % p);
    }
    uint32_t add(uint32_t a, uint32_t b) const {
        uint32_t x = a + b; // p <= 2^31-1, so addition fits uint32_t.
        return x >= p ? x - p : x;
    }
    uint32_t sub(uint32_t a, uint32_t b) const {
        return a >= b ? a - b : p - (b - a);
    }
    uint32_t mul(uint32_t a, uint32_t b) const {
        return reduce(uint64_t(a) * b);
    }
};

template<bool mersenne>
void ryser(const uint32_t* c, const uint32_t* l, int n, uint32_t p,
           uint32_t* out) {
    Field<mersenne> f{p};
    std::fill(out, out + n + 1, 0);
    if (!n) { out[0] = 1; return; }
    std::vector<uint32_t> cs(n), ls(n), product(n + 1);
    uint64_t previous = 0;
    for (uint64_t index = 1; index < (uint64_t(1) << n); ++index) {
        uint64_t gray = index ^ (index >> 1);
        uint64_t changed = gray ^ previous;
        int column = __builtin_ctzll(changed);
        bool plus = (gray & changed) != 0;
        for (int row = 0; row < n; ++row) {
            uint32_t a = c[row*n + column], b = l[row*n + column];
            if (a) cs[row] = plus ? f.add(cs[row], a) : f.sub(cs[row], a);
            if (b) ls[row] = plus ? f.add(ls[row], b) : f.sub(ls[row], b);
        }
        previous = gray;
        std::fill(product.begin(), product.end(), 0);
        product[0] = 1;
        int degree = 0;
        bool zero = false;
        for (int row = 0; row < n; ++row) {
            uint32_t a = cs[row], b = ls[row];
            if (!a && !b) { zero = true; break; }
            if (!b) {
                if (a != 1) for (int k = 0; k <= degree; ++k)
                    product[k] = f.mul(product[k], a);
            } else if (!a) {
                for (int k = degree + 1; k > 0; --k)
                    product[k] = f.mul(product[k - 1], b);
                product[0] = 0;
                ++degree;
            } else {
                product[degree + 1] = f.mul(product[degree], b);
                for (int k = degree; k > 0; --k)
                    product[k] = f.reduce(uint64_t(product[k]) * a +
                                          uint64_t(product[k - 1]) * b);
                product[0] = f.mul(product[0], a);
                ++degree;
            }
        }
        if (zero) continue;
        // Subset size parity equals binary reflected Gray index parity.
        bool negative = ((n + __builtin_popcountll(gray)) & 1) != 0;
        for (int k = 0; k <= degree; ++k)
            out[k] = negative ? f.sub(out[k], product[k])
                              : f.add(out[k], product[k]);
    }
}

template<bool mersenne>
void rook_dp(const uint32_t* a, int n, uint32_t p, uint32_t* out) {
    Field<mersenne> f{p};
    const uint64_t states = uint64_t(1) << n;
    std::vector<uint32_t> dp(states, 0);
    dp[0] = 1;
    for (int row = 0; row < n; ++row) {
        std::vector<int> columns;
        for (int col = 0; col < n; ++col)
            if (a[row*n + col]) columns.push_back(col);
        // Descending destinations have already been processed, so an in-place
        // update never matches two entries from the same row. Skips retain dp.
        for (uint64_t mask = states; mask-- > 0;) {
            uint32_t count = dp[mask];
            if (!count) continue;
            for (int col : columns) {
                uint64_t bit = uint64_t(1) << col;
                if (!(mask & bit)) {
                    uint64_t target = mask | bit;
                    dp[target] = f.add(dp[target], f.mul(count, a[row*n+col]));
                }
            }
        }
    }
    std::fill(out, out + n + 1, 0);
    for (uint64_t mask = 0; mask < states; ++mask) {
        int degree = __builtin_popcountll(mask);
        out[degree] = f.add(out[degree], dp[mask]);
    }
}
}

// Return codes: 0 success, 1 invalid dimensions/modulus, 2 allocation failure.
// Inputs are row-major canonical residues; Python performs contract validation.
extern "C" int polynomial_ryser(const uint32_t* c, const uint32_t* l, int n,
                                 uint32_t p, uint32_t* out) {
    if (n < 0 || n > 24 || p < 3 || p > M31 || !(p & 1)) return 1;
    try {
        if (p == M31) ryser<true>(c, l, n, p, out);
        else ryser<false>(c, l, n, p, out);
    } catch (const std::bad_alloc&) { return 2; }
    return 0;
}

extern "C" int weighted_rook_dp(const uint32_t* a, int n, uint32_t p,
                                 uint32_t* out) {
    if (n < 0 || n > 24 || p < 3 || p > M31 || !(p & 1)) return 1;
    try {
        if (p == M31) rook_dp<true>(a, n, p, out);
        else rook_dp<false>(a, n, p, out);
    } catch (const std::bad_alloc&) { return 2; }
    return 0;
}
