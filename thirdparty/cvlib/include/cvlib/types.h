// Matrix, vector, and complex buffer types.

#ifndef CVLIB_TYPES_H_
#define CVLIB_TYPES_H_

#include "cvlib/error_codes.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace cvlib {

using float64_t = double;
using float32_t = float;

// Row-major matrix (heap-allocated).
struct Matrix {
    float64_t* data;
    int32_t rows;
    int32_t cols;
};

struct Vector {
    float64_t* data;
    int32_t size;
};

struct Complex {
    float64_t real;
    float64_t imag;
};

struct CMatrix {
    Complex* data;
    int32_t rows;
    int32_t cols;
};

struct CVector {
    Complex* data;
    int32_t size;
};

// Matrix allocation and accessors.

/*
Computes a safe matrix element count for allocation.

@param rows Matrix row count.
@param cols Matrix column count.
@param element_size Size of one element in bytes.
@param count Output element count.
@returns ErrorCode.
*/

inline ErrorCode checked_matrix_element_count(int32_t rows, int32_t cols,
                                              std::size_t element_size,
                                              std::size_t* count) {
    ErrorCode ec = ErrorCode::kSuccess;
    if (count == nullptr) {
        ec = ErrorCode::kNullPointer;
    } else {
        *count = 0U;
        if (rows <= 0 || cols <= 0 || element_size == 0U) {
            ec = ErrorCode::kInvalidDimension;
        } else {
            const std::size_t r = static_cast<std::size_t>(rows);
            const std::size_t c = static_cast<std::size_t>(cols);
            const std::size_t max_count =
                static_cast<std::size_t>(
                    std::numeric_limits<int32_t>::max());
            if (r > std::numeric_limits<std::size_t>::max() / c) {
                ec = ErrorCode::kInvalidDimension;
            } else {
                const std::size_t n = r * c;
                if (n > max_count ||
                    n > std::numeric_limits<std::size_t>::max() /
                            element_size) {
                    ec = ErrorCode::kInvalidDimension;
                } else {
                    *count = n;
                }
            }
        }
    }
    return ec;
}

/*
Computes a safe vector element count for allocation.

@param size Vector length.
@param element_size Size of one element in bytes.
@param count Output element count.
@returns ErrorCode.
*/

inline ErrorCode checked_vector_element_count(int32_t size,
                                              std::size_t element_size,
                                              std::size_t* count) {
    ErrorCode ec = ErrorCode::kSuccess;
    if (count == nullptr) {
        ec = ErrorCode::kNullPointer;
    } else {
        *count = 0U;
        if (size <= 0 || element_size == 0U) {
            ec = ErrorCode::kInvalidDimension;
        } else {
            const std::size_t n = static_cast<std::size_t>(size);
            if (n > std::numeric_limits<std::size_t>::max() / element_size) {
                ec = ErrorCode::kInvalidDimension;
            } else {
                *count = n;
            }
        }
    }
    return ec;
}

/*
Allocates a zero-filled row-major matrix with checked dimensions.

@param rows Matrix row count.
@param cols Matrix column count.
@param out Output matrix.
@returns ErrorCode.
*/

inline ErrorCode matrix_create_checked(int32_t rows, int32_t cols,
                                       Matrix* out) {
    ErrorCode ec = ErrorCode::kSuccess;
    if (out == nullptr) {
        ec = ErrorCode::kNullPointer;
    } else {
        out->data = nullptr;
        out->rows = rows;
        out->cols = cols;
        std::size_t count = 0U;
        ec = checked_matrix_element_count(rows, cols, sizeof(float64_t),
                                          &count);
        if (ec == ErrorCode::kSuccess) {
            out->data = static_cast<float64_t*>(
                std::calloc(count, sizeof(float64_t)));
            if (out->data == nullptr) {
                ec = ErrorCode::kNullPointer;
            }
        }
    }
    return ec;
}

// Allocates a zero-filled row-major matrix on the heap.
inline Matrix matrix_create(int32_t rows, int32_t cols) {
    Matrix m;
    (void)matrix_create_checked(rows, cols, &m);
    return m;
}

// Frees the matrix buffer if owned; safe on null or already-freed.
inline void matrix_destroy(Matrix* m) {
    if (m != nullptr && m->data != nullptr) {
        std::free(m->data);
        m->data = nullptr;
    }
}

// Returns element at (r, c) without bounds checking.
inline float64_t matrix_get(const Matrix* m, int32_t r, int32_t c) {
    return m->data[r * m->cols + c];
}

// Writes v into element (r, c) without bounds checking.
inline void matrix_set(Matrix* m, int32_t r, int32_t c, float64_t v) {
    m->data[r * m->cols + c] = v;
}

/*
Copies a matrix into a checked heap allocation.

@param src Source matrix.
@param dst Output matrix.
@returns ErrorCode.
*/

inline ErrorCode matrix_copy_checked(const Matrix* src, Matrix* dst) {
    ErrorCode ec = ErrorCode::kSuccess;
    std::size_t count = 0U;
    if (dst == nullptr || src == nullptr || src->data == nullptr) {
        ec = ErrorCode::kNullPointer;
    } else {
        dst->data = nullptr;
        dst->rows = src->rows;
        dst->cols = src->cols;
        ec = checked_matrix_element_count(src->rows, src->cols,
                                          sizeof(float64_t), &count);
    }
    if (ec == ErrorCode::kSuccess) {
        ec = matrix_create_checked(src->rows, src->cols, dst);
    }
    if (ec == ErrorCode::kSuccess) {
        std::memcpy(dst->data, src->data, count * sizeof(float64_t));
    }
    return ec;
}

// Returns a deep copy of src.
inline Matrix matrix_copy(const Matrix* src) {
    Matrix dst = {nullptr, 0, 0};
    (void)matrix_copy_checked(src, &dst);
    return dst;
}

/*
Allocates an n-by-n identity matrix with checked dimensions.

@param n Matrix size.
@param out Output matrix.
@returns ErrorCode.
*/

inline ErrorCode matrix_identity_checked(int32_t n, Matrix* out) {
    ErrorCode ec = matrix_create_checked(n, n, out);
    if (ec == ErrorCode::kSuccess) {
        for (int32_t i = 0; i < n; ++i) {
            matrix_set(out, i, i, 1.0);
        }
    }
    return ec;
}

// Returns an n-by-n identity matrix.
inline Matrix matrix_identity(int32_t n) {
    Matrix m;
    (void)matrix_identity_checked(n, &m);
    return m;
}

// Vector allocation and accessors.

/*
Allocates a zero-filled vector with checked length.

@param size Vector length.
@param out Output vector.
@returns ErrorCode.
*/

inline ErrorCode vector_create_checked(int32_t size, Vector* out) {
    ErrorCode ec = ErrorCode::kSuccess;
    if (out == nullptr) {
        ec = ErrorCode::kNullPointer;
    } else {
        out->data = nullptr;
        out->size = size;
        std::size_t count = 0U;
        ec = checked_vector_element_count(size, sizeof(float64_t), &count);
        if (ec == ErrorCode::kSuccess) {
            out->data = static_cast<float64_t*>(
                std::calloc(count, sizeof(float64_t)));
            if (out->data == nullptr) {
                ec = ErrorCode::kNullPointer;
            }
        }
    }
    return ec;
}

// Allocates a zero-filled vector of the given length.
inline Vector vector_create(int32_t size) {
    Vector v;
    (void)vector_create_checked(size, &v);
    return v;
}

// Frees the vector buffer if owned; safe on null or already-freed.
inline void vector_destroy(Vector* v) {
    if (v != nullptr && v->data != nullptr) {
        std::free(v->data);
        v->data = nullptr;
    }
}

/*
Copies a vector into a checked heap allocation.

@param src Source vector.
@param dst Output vector.
@returns ErrorCode.
*/

inline ErrorCode vector_copy_checked(const Vector* src, Vector* dst) {
    ErrorCode ec = ErrorCode::kSuccess;
    std::size_t count = 0U;
    if (dst == nullptr || src == nullptr || src->data == nullptr) {
        ec = ErrorCode::kNullPointer;
    } else {
        dst->data = nullptr;
        dst->size = src->size;
        ec = checked_vector_element_count(src->size, sizeof(float64_t),
                                          &count);
    }
    if (ec == ErrorCode::kSuccess) {
        ec = vector_create_checked(src->size, dst);
    }
    if (ec == ErrorCode::kSuccess) {
        std::memcpy(dst->data, src->data, count * sizeof(float64_t));
    }
    return ec;
}

// Returns a deep copy of src.
inline Vector vector_copy(const Vector* src) {
    Vector dst = {nullptr, 0};
    (void)vector_copy_checked(src, &dst);
    return dst;
}

// Complex scalars.

// Constructs a Complex from real and imaginary parts.
inline Complex complex_make(float64_t r, float64_t i) {
    Complex c;
    c.real = r;
    c.imag = i;
    return c;
}

// Returns |c| = sqrt(real^2 + imag^2).
inline float64_t complex_abs(Complex c) {
    return std::sqrt(c.real * c.real + c.imag * c.imag);
}

// Returns a + b.
inline Complex complex_add(Complex a, Complex b) {
    return complex_make(a.real + b.real, a.imag + b.imag);
}

// Returns a - b.
inline Complex complex_sub(Complex a, Complex b) {
    return complex_make(a.real - b.real, a.imag - b.imag);
}

// Returns a * b.
inline Complex complex_mul(Complex a, Complex b) {
    return complex_make(a.real * b.real - a.imag * b.imag,
                        a.real * b.imag + a.imag * b.real);
}

// Returns a / b (no zero-denominator guard).
inline Complex complex_div(Complex a, Complex b) {
    const float64_t denom = b.real * b.real + b.imag * b.imag;
    return complex_make((a.real * b.real + a.imag * b.imag) / denom,
                        (a.imag * b.real - a.real * b.imag) / denom);
}

// Returns the complex conjugate of c.
inline Complex complex_conj(Complex c) {
    return complex_make(c.real, -c.imag);
}

// Returns the principal square root of c.
inline Complex complex_sqrt(Complex c) {
    const float64_t r = complex_abs(c);
    if (r < 1e-15) {
        return complex_make(0.0, 0.0);
    }
    const float64_t t = std::sqrt((std::fabs(c.real) + r) * 0.5);
    if (c.real >= 0.0) {
        return complex_make(t, c.imag / (2.0 * t));
    }
    return complex_make(std::fabs(c.imag) / (2.0 * t),
                        (c.imag >= 0.0) ? t : -t);
}

// Complex matrices.

/*
Allocates a zero-filled row-major complex matrix with checked dimensions.

@param rows Matrix row count.
@param cols Matrix column count.
@param out Output complex matrix.
@returns ErrorCode.
*/

inline ErrorCode cmatrix_create_checked(int32_t rows, int32_t cols,
                                        CMatrix* out) {
    ErrorCode ec = ErrorCode::kSuccess;
    if (out == nullptr) {
        ec = ErrorCode::kNullPointer;
    } else {
        out->data = nullptr;
        out->rows = rows;
        out->cols = cols;
        std::size_t count = 0U;
        ec = checked_matrix_element_count(rows, cols, sizeof(Complex),
                                          &count);
        if (ec == ErrorCode::kSuccess) {
            out->data = static_cast<Complex*>(
                std::calloc(count, sizeof(Complex)));
            if (out->data == nullptr) {
                ec = ErrorCode::kNullPointer;
            }
        }
    }
    return ec;
}

// Allocates a zero-filled row-major complex matrix.
inline CMatrix cmatrix_create(int32_t rows, int32_t cols) {
    CMatrix m;
    (void)cmatrix_create_checked(rows, cols, &m);
    return m;
}

// Frees the complex matrix buffer if owned.
inline void cmatrix_destroy(CMatrix* m) {
    if (m != nullptr && m->data != nullptr) {
        std::free(m->data);
        m->data = nullptr;
    }
}

// Returns element at (r, c) without bounds checking.
inline Complex cmatrix_get(const CMatrix* m, int32_t r, int32_t c) {
    return m->data[r * m->cols + c];
}

// Writes v into element (r, c) without bounds checking.
inline void cmatrix_set(CMatrix* m, int32_t r, int32_t c, Complex v) {
    m->data[r * m->cols + c] = v;
}

/*
Allocates an n-by-n complex identity matrix with checked dimensions.

@param n Matrix size.
@param out Output complex matrix.
@returns ErrorCode.
*/

inline ErrorCode cmatrix_identity_checked(int32_t n, CMatrix* out) {
    ErrorCode ec = cmatrix_create_checked(n, n, out);
    if (ec == ErrorCode::kSuccess) {
        for (int32_t i = 0; i < n; ++i) {
            cmatrix_set(out, i, i, complex_make(1.0, 0.0));
        }
    }
    return ec;
}

// Returns an n-by-n complex identity matrix.
inline CMatrix cmatrix_identity(int32_t n) {
    CMatrix m;
    (void)cmatrix_identity_checked(n, &m);
    return m;
}

/*
Promotes a real matrix to a checked complex matrix allocation.

@param src Source real matrix.
@param dst Output complex matrix.
@returns ErrorCode.
*/

inline ErrorCode cmatrix_from_matrix_checked(const Matrix* src,
                                             CMatrix* dst) {
    ErrorCode ec = ErrorCode::kSuccess;
    std::size_t count = 0U;
    if (dst == nullptr || src == nullptr || src->data == nullptr) {
        ec = ErrorCode::kNullPointer;
    } else {
        dst->data = nullptr;
        dst->rows = src->rows;
        dst->cols = src->cols;
        ec = checked_matrix_element_count(src->rows, src->cols,
                                          sizeof(Complex), &count);
    }
    if (ec == ErrorCode::kSuccess) {
        ec = cmatrix_create_checked(src->rows, src->cols, dst);
    }
    if (ec == ErrorCode::kSuccess) {
        for (std::size_t i = 0U; i < count; ++i) {
            dst->data[i] = complex_make(src->data[i], 0.0);
        }
    }
    return ec;
}

// Promotes a real matrix to complex (zero imaginary parts).
inline CMatrix cmatrix_from_matrix(const Matrix* src) {
    CMatrix dst = {nullptr, 0, 0};
    (void)cmatrix_from_matrix_checked(src, &dst);
    return dst;
}

/*
Copies a complex matrix into a checked heap allocation.

@param src Source complex matrix.
@param dst Output complex matrix.
@returns ErrorCode.
*/

inline ErrorCode cmatrix_copy_checked(const CMatrix* src, CMatrix* dst) {
    ErrorCode ec = ErrorCode::kSuccess;
    std::size_t count = 0U;
    if (dst == nullptr || src == nullptr || src->data == nullptr) {
        ec = ErrorCode::kNullPointer;
    } else {
        dst->data = nullptr;
        dst->rows = src->rows;
        dst->cols = src->cols;
        ec = checked_matrix_element_count(src->rows, src->cols,
                                          sizeof(Complex), &count);
    }
    if (ec == ErrorCode::kSuccess) {
        ec = cmatrix_create_checked(src->rows, src->cols, dst);
    }
    if (ec == ErrorCode::kSuccess) {
        std::memcpy(dst->data, src->data, count * sizeof(Complex));
    }
    return ec;
}

// Returns a deep copy of a complex matrix.
inline CMatrix cmatrix_copy(const CMatrix* src) {
    CMatrix dst = {nullptr, 0, 0};
    (void)cmatrix_copy_checked(src, &dst);
    return dst;
}

/*
Allocates a zero-filled complex vector with checked length.

@param size Vector length.
@param out Output complex vector.
@returns ErrorCode.
*/

inline ErrorCode cvector_create_checked(int32_t size, CVector* out) {
    ErrorCode ec = ErrorCode::kSuccess;
    if (out == nullptr) {
        ec = ErrorCode::kNullPointer;
    } else {
        out->data = nullptr;
        out->size = size;
        std::size_t count = 0U;
        ec = checked_vector_element_count(size, sizeof(Complex), &count);
        if (ec == ErrorCode::kSuccess) {
            out->data = static_cast<Complex*>(
                std::calloc(count, sizeof(Complex)));
            if (out->data == nullptr) {
                ec = ErrorCode::kNullPointer;
            }
        }
    }
    return ec;
}

// Allocates a zero-filled complex vector of the given length.
inline CVector cvector_create(int32_t size) {
    CVector v;
    (void)cvector_create_checked(size, &v);
    return v;
}

// Frees the complex vector buffer if owned.
inline void cvector_destroy(CVector* v) {
    if (v != nullptr && v->data != nullptr) {
        std::free(v->data);
        v->data = nullptr;
    }
}

}  // namespace cvlib

#endif  // CVLIB_TYPES_H_
