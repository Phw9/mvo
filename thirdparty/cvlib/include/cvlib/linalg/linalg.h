// Dense matrix and vector linear algebra.

#ifndef CVLIB_LINALG_LINALG_H_
#define CVLIB_LINALG_LINALG_H_

#include "cvlib/types.h"
#include "cvlib/error_codes.h"

#include <cstdint>

namespace cvlib {
namespace linalg {

static constexpr int32_t kNormFrobenius = 0;
static constexpr int32_t kNormSpectral  = 1;
static constexpr int32_t kNormNuclear  = 2;

static constexpr int32_t kInversePseudo      = 0;
static constexpr int32_t kInverseGaussJordan = 1;

// Default tolerance for SVD-based rank/pinv routines.
static constexpr float64_t kRankDefaultTolerance = 1e-10;

/*
Computes the matrix product C = A B.

@param a Left factor, m-by-k.
@param b Right factor, k-by-n.
@param result Output, m-by-n; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode matmul(const Matrix* a, const Matrix* b, Matrix* result);

/*
Computes C = A B^T with B stored as n-by-k (B^T is k-by-n).

@param a Left factor, m-by-k.
@param b Right factor before transpose, n-by-k.
@param result Output, m-by-n; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode matmul_nt(const Matrix* a, const Matrix* b, Matrix* result);

/*
Computes the Euclidean norm of a vector.

@param v Input vector.
@param result Output scalar norm.
@returns ErrorCode.

*/

ErrorCode vec_norm(const Vector* v, float64_t* result);

/*
Computes the matrix transpose out of place (rectangular allowed).

@param m Input matrix.
@param result Output, cols(m)-by-rows(m); must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode transpose(const Matrix* m, Matrix* result);

/*
Inverts a square matrix using Gauss-Jordan elimination.

@param m Input n-by-n matrix.
@param result Output n-by-n inverse; must be pre-allocated.
@returns ErrorCode: kSuccess, or singular matrix or invalid shape.

*/

ErrorCode inv_gj(const Matrix* m, Matrix* result);

/*
Computes the Moore-Penrose pseudoinverse via SVD.
Singular values at or below tol * max(singular values) are discarded.

@param m Input matrix of arbitrary shape.
@param result Output; must be pre-allocated with compatible dimensions.
@param tol Relative SV cutoff (default kRankDefaultTolerance).
@returns ErrorCode.

*/

ErrorCode pinv(const Matrix* m, Matrix* result,
               float64_t tol = kRankDefaultTolerance);

/*
Computes an inverse using either SVD pseudoinverse or Gauss-Jordan.
The default SVD pseudoinverse accepts rectangular matrices and returns the
Moore-Penrose inverse. Gauss-Jordan is square-only.

@param m Input matrix.
@param result Output cols(m)-by-rows(m) for pseudoinverse, n-by-n for GJ.
@param method kInversePseudo (default) or kInverseGaussJordan.
@returns ErrorCode.

*/

ErrorCode inv(const Matrix* m, Matrix* result,
              int32_t method = kInversePseudo);

/*
Computes the numerical rank from singular values.
Singular values at or below tol * max(singular values) are discarded.

@param m Input matrix.
@param result Output rank.
@param tol Relative SV cutoff (default kRankDefaultTolerance).
@returns ErrorCode.

*/

ErrorCode rank(const Matrix* m, int32_t* result,
               float64_t tol = kRankDefaultTolerance);

/*
Computes nullity as number of columns minus numerical rank.

@param m Input matrix.
@param result Output nullity.
@param tol Relative SV cutoff (default kRankDefaultTolerance).
@returns ErrorCode.

*/

ErrorCode nullity(const Matrix* m, int32_t* result,
                  float64_t tol = kRankDefaultTolerance);

/*
Tests whether the matrix is full rank in the numerical sense.

@param m Input matrix.
@param result Output true if rank equals min(rows, cols).
@param tol Relative SV cutoff (default kRankDefaultTolerance).
@returns ErrorCode.

*/

ErrorCode is_full_rank(const Matrix* m, bool* result,
                       float64_t tol = kRankDefaultTolerance);

/*
Computes the trace of a square matrix.

@param m Input n-by-n matrix.
@param result Output scalar trace.
@returns ErrorCode.

*/

ErrorCode trace(const Matrix* m, float64_t* result);

/*
Computes Frobenius, spectral, or nuclear norm selected by norm_type.

@param m Input matrix.
@param result Output scalar norm.
@param norm_type kNormFrobenius (default), kNormSpectral, or kNormNuclear.
@returns ErrorCode.

*/

ErrorCode mat_norm(const Matrix* m, float64_t* result,
                   int32_t norm_type = kNormFrobenius);

/*
Computes the dot product of two vectors of equal length.

@param a First input vector.
@param b Second input vector.
@param result Output scalar dot product.
@returns ErrorCode.

*/

ErrorCode dot(const Vector* a, const Vector* b, float64_t* result);

/*
Computes the 3D cross product result = a x b.

@param a First input length-3 vector.
@param b Second input length-3 vector.
@param result Output length-3 vector; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode cross(const Vector* a, const Vector* b, Vector* result);

/*
Computes the outer product result = a b^T.

@param a Left input vector of length m.
@param b Right input vector of length n.
@param result Output m-by-n matrix; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode outer(const Vector* a, const Vector* b, Matrix* result);

/*
Computes the BLAS-style update result = alpha * x + y for vectors.

@param alpha Scalar multiplier applied to x.
@param x Input vector.
@param y Input vector of the same length as x.
@param result Output vector of the same length; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode axpy(float64_t alpha, const Vector* x, const Vector* y,
               Vector* result);

/*
Computes the elementwise sum result = a + b for matrices of the same shape.

@param a Left input matrix.
@param b Right input matrix of the same shape as a.
@param result Output matrix of the same shape; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode mat_add(const Matrix* a, const Matrix* b, Matrix* result);

/*
Computes the elementwise difference result = a - b for matrices of equal shape.

@param a Left input matrix.
@param b Right input matrix of the same shape as a.
@param result Output matrix of the same shape; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode mat_sub(const Matrix* a, const Matrix* b, Matrix* result);

/*
Computes the scalar multiple result = alpha * a.

@param a Input matrix.
@param alpha Scalar multiplier.
@param result Output matrix of the same shape as a; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode mat_scale(const Matrix* a, float64_t alpha, Matrix* result);

/*
Builds an n-by-n diagonal matrix from a length-n vector.

@param v Input length-n vector providing diagonal entries.
@param result Output n-by-n matrix; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode diag(const Vector* v, Matrix* result);

/*
Extracts the main diagonal of a matrix into a vector of length min(rows, cols).

@param m Input matrix of arbitrary shape.
@param result Output vector of length min(rows, cols); must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode diag_of(const Matrix* m, Vector* result);

}  // namespace linalg
}  // namespace cvlib

#endif  // CVLIB_LINALG_LINALG_H_
