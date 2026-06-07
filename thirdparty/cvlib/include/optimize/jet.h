// Forward-mode autodiff dual number type (header-only).

#ifndef CVLIB_OPTIMIZE_JET_H_
#define CVLIB_OPTIMIZE_JET_H_

#include "types.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cvlib {
namespace optimize {

// Jet<N>: scalar value plus N partial derivatives propagated by overloads.

template <int32_t N>
struct Jet {
    float64_t           a;
    std::array<float64_t, N> v;

    Jet() : a(0.0) { v.fill(0.0); }
    explicit Jet(float64_t scalar) : a(scalar) { v.fill(0.0); }
    Jet(float64_t scalar, int32_t k) : a(scalar) {
        v.fill(0.0);
        v[k] = 1.0;
    }
    Jet(float64_t scalar, const std::array<float64_t, N>& dv)
        : a(scalar), v(dv) {}
};

// Helpers for elementwise tangent arithmetic.
template <int32_t N>
inline std::array<float64_t, N> add_v(const std::array<float64_t, N>& x,
                                       const std::array<float64_t, N>& y) {
    std::array<float64_t, N> r{};
    for (int32_t i = 0; i < N; ++i) { r[i] = x[i] + y[i]; }
    return r;
}

template <int32_t N>
inline std::array<float64_t, N> sub_v(const std::array<float64_t, N>& x,
                                       const std::array<float64_t, N>& y) {
    std::array<float64_t, N> r{};
    for (int32_t i = 0; i < N; ++i) { r[i] = x[i] - y[i]; }
    return r;
}

template <int32_t N>
inline std::array<float64_t, N> scale_v(float64_t s,
                                         const std::array<float64_t, N>& x) {
    std::array<float64_t, N> r{};
    for (int32_t i = 0; i < N; ++i) { r[i] = s * x[i]; }
    return r;
}

template <int32_t N>
inline std::array<float64_t, N> neg_v(const std::array<float64_t, N>& x) {
    std::array<float64_t, N> r{};
    for (int32_t i = 0; i < N; ++i) { r[i] = -x[i]; }
    return r;
}

// Basic arithmetic on Jet<N>.
template <int32_t N>
inline Jet<N> operator+(const Jet<N>& x, const Jet<N>& y) {
    return Jet<N>(x.a + y.a, add_v<N>(x.v, y.v));
}

template <int32_t N>
inline Jet<N> operator-(const Jet<N>& x, const Jet<N>& y) {
    return Jet<N>(x.a - y.a, sub_v<N>(x.v, y.v));
}

template <int32_t N>
inline Jet<N> operator-(const Jet<N>& x) {
    return Jet<N>(-x.a, neg_v<N>(x.v));
}

template <int32_t N>
inline Jet<N> operator*(const Jet<N>& x, const Jet<N>& y) {
    return Jet<N>(x.a * y.a,
                  add_v<N>(scale_v<N>(y.a, x.v), scale_v<N>(x.a, y.v)));
}

template <int32_t N>
inline Jet<N> operator/(const Jet<N>& x, const Jet<N>& y) {
    const float64_t inv = 1.0 / y.a;
    const float64_t a = x.a * inv;
    return Jet<N>(a, scale_v<N>(inv, sub_v<N>(x.v, scale_v<N>(a, y.v))));
}

// Mixed scalar/Jet arithmetic.
template <int32_t N>
inline Jet<N> operator+(const Jet<N>& x, float64_t s) {
    return Jet<N>(x.a + s, x.v);
}

template <int32_t N>
inline Jet<N> operator+(float64_t s, const Jet<N>& x) { return x + s; }

template <int32_t N>
inline Jet<N> operator-(const Jet<N>& x, float64_t s) {
    return Jet<N>(x.a - s, x.v);
}

template <int32_t N>
inline Jet<N> operator-(float64_t s, const Jet<N>& x) {
    return Jet<N>(s - x.a, neg_v<N>(x.v));
}

template <int32_t N>
inline Jet<N> operator*(const Jet<N>& x, float64_t s) {
    return Jet<N>(x.a * s, scale_v<N>(s, x.v));
}

template <int32_t N>
inline Jet<N> operator*(float64_t s, const Jet<N>& x) { return x * s; }

template <int32_t N>
inline Jet<N> operator/(const Jet<N>& x, float64_t s) {
    const float64_t inv = 1.0 / s;
    return Jet<N>(x.a * inv, scale_v<N>(inv, x.v));
}

template <int32_t N>
inline Jet<N> operator/(float64_t s, const Jet<N>& x) {
    const float64_t inv = 1.0 / x.a;
    const float64_t a = s * inv;
    return Jet<N>(a, scale_v<N>(-a * inv, x.v));
}

// Comparisons (only the scalar component).
template <int32_t N>
inline bool operator<(const Jet<N>& x, const Jet<N>& y)  { return x.a <  y.a; }
template <int32_t N>
inline bool operator>(const Jet<N>& x, const Jet<N>& y)  { return x.a >  y.a; }
template <int32_t N>
inline bool operator<=(const Jet<N>& x, const Jet<N>& y) { return x.a <= y.a; }
template <int32_t N>
inline bool operator>=(const Jet<N>& x, const Jet<N>& y) { return x.a >= y.a; }
template <int32_t N>
inline bool operator==(const Jet<N>& x, const Jet<N>& y) { return x.a == y.a; }
template <int32_t N>
inline bool operator!=(const Jet<N>& x, const Jet<N>& y) { return x.a != y.a; }

// Elementary functions.
template <int32_t N>
inline Jet<N> sqrt(const Jet<N>& x) {
    const float64_t s = std::sqrt(x.a);
    const float64_t inv = 0.5 / s;
    return Jet<N>(s, scale_v<N>(inv, x.v));
}

template <int32_t N>
inline Jet<N> sin(const Jet<N>& x) {
    return Jet<N>(std::sin(x.a), scale_v<N>(std::cos(x.a), x.v));
}

template <int32_t N>
inline Jet<N> cos(const Jet<N>& x) {
    return Jet<N>(std::cos(x.a), scale_v<N>(-std::sin(x.a), x.v));
}

template <int32_t N>
inline Jet<N> exp(const Jet<N>& x) {
    const float64_t e = std::exp(x.a);
    return Jet<N>(e, scale_v<N>(e, x.v));
}

template <int32_t N>
inline Jet<N> log(const Jet<N>& x) {
    return Jet<N>(std::log(x.a), scale_v<N>(1.0 / x.a, x.v));
}

template <int32_t N>
inline Jet<N> atan2(const Jet<N>& y, const Jet<N>& x) {
    const float64_t s = x.a * x.a + y.a * y.a;
    return Jet<N>(std::atan2(y.a, x.a),
                  add_v<N>(scale_v<N>( x.a / s, y.v),
                           scale_v<N>(-y.a / s, x.v)));
}

template <int32_t N>
inline Jet<N> abs(const Jet<N>& x) {
    return (x.a >= 0.0) ? x : -x;
}

}  // namespace optimize
}  // namespace cvlib

#endif  // CVLIB_OPTIMIZE_JET_H_
