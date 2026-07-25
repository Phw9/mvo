// Matrix, vector, and complex buffer types.

#ifndef CVLIB_TYPES_H_
#define CVLIB_TYPES_H_

#include "error_codes.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace cvlib {

using float64_t = double;
using float32_t = float;

// Element depth tags (k8U == 0 so a zero-initialized tag is 8-bit
// unsigned). A full buffer type combines a depth with a channel count
// via mat_type().
static constexpr int32_t k8U  = 0;
static constexpr int32_t k8S  = 1;
static constexpr int32_t k16U = 2;
static constexpr int32_t k16S = 3;
static constexpr int32_t k32S = 4;
static constexpr int32_t k32F = 5;
static constexpr int32_t k64F = 6;

static constexpr int32_t kDepthShift = 3;   // channels occupy bits >= 3
static constexpr int32_t kDepthMask  = 0x7;

// Combines a depth tag and channel count (1..) into a buffer type:
// type = depth | ((channels - 1) << 3).
inline int32_t mat_type(int32_t depth, int32_t channels) {
    return (depth & kDepthMask) | ((channels - 1) << kDepthShift);
}
// Extracts the depth tag from a buffer type.
inline int32_t mat_depth(int32_t type) { return type & kDepthMask; }
// Extracts the channel count from a buffer type.
inline int32_t mat_channels(int32_t type) {
    return (type >> kDepthShift) + 1;
}
// Bytes per single element of the depth (not per pixel): 8U->1, 16->2,
// 32->4, 64F->8; 0 for an unknown depth.
inline int32_t mat_depth_bytes(int32_t type) {
    int32_t bytes = 0;
    switch (mat_depth(type)) {
        case k8U:
        case k8S:  bytes = 1; break;
        case k16U:
        case k16S: bytes = 2; break;
        case k32S:
        case k32F: bytes = 4; break;
        case k64F: bytes = 8; break;
        default:   bytes = 0; break;
    }
    return bytes;
}

// Single-channel buffer type constants.
static constexpr int32_t k8UC1  = k8U;
static constexpr int32_t k8SC1  = k8S;
static constexpr int32_t k16UC1 = k16U;
static constexpr int32_t k16SC1 = k16S;
static constexpr int32_t k32SC1 = k32S;
static constexpr int32_t k32FC1 = k32F;
static constexpr int32_t k64FC1 = k64F;

struct Matrix;

// Non-owning read-only view of a rectangular sub-block of a Matrix.
// Blocks borrow the parent buffer (no allocation, no ownership) and are
// unchecked like Matrix::operator(): the caller guarantees the block
// lies inside the parent, shapes match on assignment, and the parent
// outlives the view.
struct ConstMatrixBlock {
    const float64_t* data;
    int32_t stride;
    int32_t row0;
    int32_t col0;
    int32_t rows;
    int32_t cols;

    float64_t operator()(int32_t r, int32_t c) const {
        return data[(row0 + r) * stride + (col0 + c)];
    }
};

// Mutable block view; assignment copies *elements* (Eigen-block style):
//   m.block(6, 6, 3, 3) = exp_w;      // Matrix -> block
//   m.block(0, 3, 3, 3) = 0.0;        // fill
//   a.block(...) = b.block(...);      // block -> block (no overlap)
struct MatrixBlock {
    float64_t* data;
    int32_t stride;
    int32_t row0;
    int32_t col0;
    int32_t rows;
    int32_t cols;

    float64_t& operator()(int32_t r, int32_t c) {
        return data[(row0 + r) * stride + (col0 + c)];
    }
    float64_t operator()(int32_t r, int32_t c) const {
        return data[(row0 + r) * stride + (col0 + c)];
    }

    operator ConstMatrixBlock() const {
        return ConstMatrixBlock{data, stride, row0, col0, rows, cols};
    }

    MatrixBlock& operator=(float64_t v) {
        for (int32_t r = 0; r < rows; ++r) {
            for (int32_t c = 0; c < cols; ++c) {
                (*this)(r, c) = v;
            }
        }
        return *this;
    }
    MatrixBlock& operator=(const ConstMatrixBlock& src) {
        for (int32_t r = 0; r < rows; ++r) {
            for (int32_t c = 0; c < cols; ++c) {
                (*this)(r, c) = src(r, c);
            }
        }
        return *this;
    }
    MatrixBlock& operator=(const MatrixBlock& src) {
        return *this = static_cast<ConstMatrixBlock>(src);
    }
    MatrixBlock& operator=(const Matrix& src);
};

// Row-major matrix (heap-allocated).
//
// Element access convention: Matrix lvalues (locals, struct members,
// dereferenced results) use the call operator --
//   m(i, j) = v;   const float64_t x = m(i, j);
// -- while pointer-typed call sites keep the matrix_get/matrix_set free
// functions, which read better than (*m)(i, j). Both are unchecked, like
// the free functions they mirror. Member functions keep the struct an
// aggregate: brace initialization and the C-style API are unchanged.
struct Matrix {
    // Untyped buffer so a matrix can hold any depth (see the type tag).
    // The element accessors below are the f64 fast path and require the
    // matrix to be k64FC1 -- every numeric matrix is. Other depths use
    // ptr<T>() with a T matching the buffer depth.
    void* data;
    int32_t rows;
    int32_t cols;
    // Buffer type tag and row stride (in samples). Defaulted so existing
    // aggregate initialization -- Matrix m = {nullptr, 0, 0}; -- still
    // compiles and yields a dense k64FC1 matrix. The f64 accessors assume
    // dense storage (stride == cols); strided access reads ptr<T>() with
    // the stride directly.
    int32_t stride = 0;        // 0 means dense: treat as cols
    int32_t type   = k64FC1;

    // Typed data pointer: reinterprets the untyped buffer as T*. The
    // caller guarantees T matches the buffer depth (T == float64_t for
    // the k64FC1 fast path the accessors below use).
    template <typename T> T* ptr() { return static_cast<T*>(data); }
    template <typename T> const T* ptr() const {
        return static_cast<const T*>(data);
    }

    float64_t& operator()(int32_t r, int32_t c) {
        return ptr<float64_t>()[r * cols + c];
    }
    float64_t operator()(int32_t r, int32_t c) const {
        return ptr<float64_t>()[r * cols + c];
    }

    // Writes v into every element without reallocating; unchecked like
    // operator() (the caller guarantees data is non-null and k64FC1).
    void fill(float64_t v) {
        float64_t* p = ptr<float64_t>();
        const int32_t n = rows * cols;
        for (int32_t i = 0; i < n; ++i) {
            p[i] = v;
        }
    }
    void set_zero() { fill(0.0); }
    // Writes 1 on the main diagonal and 0 elsewhere. Rectangular shapes
    // are supported: the min(rows, cols) leading diagonal entries get 1.
    void set_identity() {
        set_zero();
        float64_t* p = ptr<float64_t>();
        const int32_t k = (rows < cols) ? rows : cols;
        for (int32_t i = 0; i < k; ++i) {
            p[i * cols + i] = 1.0;
        }
    }
    // m = v fills every element with v; the implicit Matrix-to-Matrix
    // assignment stays the usual shallow copy.
    Matrix& operator=(float64_t v) {
        fill(v);
        return *this;
    }

    // Borrow a rows-by-cols view anchored at (r0, c0); see MatrixBlock
    // for the unchecked contract.
    MatrixBlock block(int32_t r0, int32_t c0, int32_t block_rows,
                      int32_t block_cols) {
        return MatrixBlock{ptr<float64_t>(), cols, r0, c0, block_rows, block_cols};
    }
    ConstMatrixBlock block(int32_t r0, int32_t c0, int32_t block_rows,
                           int32_t block_cols) const {
        return ConstMatrixBlock{ptr<float64_t>(), cols, r0, c0, block_rows,
                                block_cols};
    }

    // Crop a block's elements into this pre-allocated matrix (shapes
    // must match; unchecked).
    Matrix& operator=(const ConstMatrixBlock& src) {
        for (int32_t r = 0; r < rows; ++r) {
            for (int32_t c = 0; c < cols; ++c) {
                (*this)(r, c) = src(r, c);
            }
        }
        return *this;
    }

    // Deep-copies src's elements into this pre-allocated matrix of the
    // same shape (unchecked); the shallow Matrix-to-Matrix assignment
    // stays untouched.
    void copy_from(const Matrix& src) {
        float64_t* dst = ptr<float64_t>();
        const float64_t* s = src.ptr<float64_t>();
        const int32_t n = rows * cols;
        for (int32_t i = 0; i < n; ++i) {
            dst[i] = s[i];
        }
    }
};

inline MatrixBlock& MatrixBlock::operator=(const Matrix& src) {
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t c = 0; c < cols; ++c) {
            (*this)(r, c) = src(r, c);
        }
    }
    return *this;
}

// Element access mirrors Matrix: Vector lvalues use v[i] (unchecked);
// pointer-typed call sites keep v->data[i].
struct Vector {
    float64_t* data;
    int32_t size;

    float64_t& operator[](int32_t i) { return data[i]; }
    float64_t operator[](int32_t i) const { return data[i]; }

    // Writes v into every element without reallocating; unchecked like
    // operator[].
    void fill(float64_t v) {
        for (int32_t i = 0; i < size; ++i) {
            data[i] = v;
        }
    }
    void set_zero() { fill(0.0); }
    // v = x fills every element with x; Vector-to-Vector assignment
    // stays the usual shallow copy.
    Vector& operator=(float64_t v) {
        fill(v);
        return *this;
    }
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
        out->stride = cols;
        out->type = k64FC1;
        std::size_t count = 0U;
        ec = checked_matrix_element_count(rows, cols, sizeof(float64_t),
                                          &count);
        if (ec == ErrorCode::kSuccess) {
            out->data = std::calloc(count, sizeof(float64_t));
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
    return m->ptr<float64_t>()[r * m->cols + c];
}

// Writes v into element (r, c) without bounds checking.
inline void matrix_set(Matrix* m, int32_t r, int32_t c, float64_t v) {
    m->ptr<float64_t>()[r * m->cols + c] = v;
}

// Row stride in samples: the stored stride, or the dense default
// (cols * channels) when stride is 0.
inline int32_t mat_stride(const Matrix* m) {
    const int32_t cn = mat_channels(m->type);
    return (m->stride != 0) ? m->stride : m->cols * cn;
}

// Typed element access honoring stride and channels; T must match the
// buffer depth. ch selects the channel (0 for single-channel matrices).
// The index is computed in 64 bits so multi-channel buffers do not
// overflow the sample offset.
template <typename T>
inline T& mat_at(Matrix* m, int32_t r, int32_t c, int32_t ch = 0) {
    const std::int64_t cn = mat_channels(m->type);
    const std::int64_t stride = mat_stride(m);
    return m->ptr<T>()[static_cast<std::int64_t>(r) * stride +
                       static_cast<std::int64_t>(c) * cn + ch];
}
template <typename T>
inline const T& mat_at(const Matrix* m, int32_t r, int32_t c,
                       int32_t ch = 0) {
    const std::int64_t cn = mat_channels(m->type);
    const std::int64_t stride = mat_stride(m);
    return m->ptr<T>()[static_cast<std::int64_t>(r) * stride +
                       static_cast<std::int64_t>(c) * cn + ch];
}

/*
Allocates a zero-filled row-major matrix of a given buffer type.

@param rows Matrix row count.
@param cols Matrix column count.
@param type Buffer type (mat_type(depth, channels)).
@param out Output matrix (dense: stride == cols * channels).
@returns ErrorCode.
*/

inline ErrorCode matrix_create_typed_checked(int32_t rows, int32_t cols,
                                             int32_t type, Matrix* out) {
    ErrorCode ec = ErrorCode::kSuccess;
    if (out == nullptr) {
        ec = ErrorCode::kNullPointer;
    } else {
        const int32_t cn = mat_channels(type);
        const int32_t bytes = mat_depth_bytes(type);
        out->data = nullptr;
        out->rows = rows;
        out->cols = cols;
        out->stride = cols * cn;
        out->type = type;
        if (bytes == 0 || cn < 1) {
            ec = ErrorCode::kInvalidArgument;
        } else {
            const std::size_t elem = static_cast<std::size_t>(bytes) *
                                     static_cast<std::size_t>(cn);
            std::size_t count = 0U;
            ec = checked_matrix_element_count(rows, cols, elem, &count);
            if (ec == ErrorCode::kSuccess) {
                out->data = std::calloc(count, elem);
                if (out->data == nullptr) {
                    ec = ErrorCode::kNullPointer;
                }
            }
        }
    }
    return ec;
}

// Allocates a zero-filled typed matrix on the heap.
inline Matrix matrix_create_typed(int32_t rows, int32_t cols, int32_t type) {
    Matrix m;
    (void)matrix_create_typed_checked(rows, cols, type, &m);
    return m;
}

// Non-owning read-only view of a strided, typed buffer (the borrow
// counterpart of Matrix; see matrix_view). The parent buffer must
// outlive the view.
struct MatrixView {
    const void* data;
    int32_t rows;
    int32_t cols;
    int32_t stride;   // samples per row
    int32_t type;

    template <typename T> const T* ptr() const {
        return static_cast<const T*>(data);
    }
};

// Borrows a matrix as a read-only view (dense stride resolved).
inline MatrixView matrix_view(const Matrix* m) {
    return MatrixView{m->data, m->rows, m->cols, mat_stride(m), m->type};
}

// Typed element access on a view; T must match the buffer depth.
template <typename T>
inline const T& mat_at(const MatrixView* v, int32_t r, int32_t c,
                       int32_t ch = 0) {
    const int32_t cn = mat_channels(v->type);
    return v->ptr<T>()[r * v->stride + c * cn + ch];
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
    // Deep copy preserves the source buffer type; the element size is the
    // per-sample depth times the channel count so a typed (non-k64FC1)
    // matrix is not read past its buffer.
    const std::size_t elem =
        (src != nullptr)
            ? static_cast<std::size_t>(mat_depth_bytes(src->type)) *
                  static_cast<std::size_t>(mat_channels(src->type))
            : 0U;
    if (dst == nullptr || src == nullptr || src->data == nullptr ||
        elem == 0U) {
        ec = (dst == nullptr || src == nullptr || src->data == nullptr)
                 ? ErrorCode::kNullPointer
                 : ErrorCode::kInvalidArgument;
    } else {
        dst->data = nullptr;
        ec = checked_matrix_element_count(src->rows, src->cols, elem,
                                          &count);
    }
    if (ec == ErrorCode::kSuccess) {
        ec = matrix_create_typed_checked(src->rows, src->cols, src->type,
                                         dst);
    }
    if (ec == ErrorCode::kSuccess) {
        std::memcpy(dst->data, src->data, count * elem);
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
        out->set_identity();
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
    Complex result;
    const float64_t r = complex_abs(c);
    if (r < 1e-15) {
        result = complex_make(0.0, 0.0);
    } else {
        const float64_t t = std::sqrt((std::fabs(c.real) + r) * 0.5);
        if (c.real >= 0.0) {
            result = complex_make(t, c.imag / (2.0 * t));
        } else {
            result = complex_make(std::fabs(c.imag) / (2.0 * t),
                                  (c.imag >= 0.0) ? t : -t);
        }
    }
    return result;
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
    } else if (src->type != k64FC1) {
        // Real-to-complex promotion reads the source as f64; a typed
        // (image) buffer is not a valid numeric input.
        ec = ErrorCode::kInvalidArgument;
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
        const float64_t* s = src->ptr<float64_t>();
        for (std::size_t i = 0U; i < count; ++i) {
            dst->data[i] = complex_make(s[i], 0.0);
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
