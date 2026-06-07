// Real-valued matrix decompositions: QR, eigen (symmetric), determinant.

#ifndef CVLIB_LINALG_DECOMPOSITION_H_
#define CVLIB_LINALG_DECOMPOSITION_H_

#include "types.h"
#include "error_codes.h"

#include <cstdint>

namespace cvlib {
namespace linalg {

/*
Sorts indices so values[i] appear in descending magnitude order.

@param values Input array of length n.
@param n Length of values and indices.
@param indices Output permutation of indices, length n.
@returns ErrorCode.
*/

ErrorCode argsort_desc(const float64_t* values, int32_t n, int32_t* indices);

/*
Computes the determinant of a real square matrix via cofactor expansion.

@param m Input n-by-n matrix.
@param result Output determinant.
@returns ErrorCode.
*/

ErrorCode det(const Matrix* m, float64_t* result);

/*
Real QR factorization A = Q R via Modified Gram-Schmidt (reduced form).
R diagonal entries are non-negative, matching numpy.linalg.qr conventions.

@param a Input m-by-n matrix with m >= n.
@param q Output m-by-n with orthonormal columns; pre-allocated.
@param r Output n-by-n upper triangular; pre-allocated.
@returns ErrorCode.
*/

ErrorCode qr(const Matrix* a, Matrix* q, Matrix* r);

// Default Jacobi-eigen iteration cap and convergence tolerance.
static constexpr int32_t   kEighDefaultMaxIter   = 1000;
static constexpr float64_t kEighDefaultTolerance = 1e-12;

/*
Symmetric eigendecomposition A = V D V^T via cyclic Jacobi rotations.
Inputs are auto-symmetrized; eigenvalues sorted in descending order.
Output matches numpy.linalg.eigh / cv::eigen for symmetric matrices.

@param a Input n-by-n (treated as symmetric).
@param eigenvalues Output length-n vector, descending order; pre-allocated.
@param eigenvectors Output n-by-n orthogonal matrix (columns); pre-allocated.
@param max_iter Maximum Jacobi sweeps (default kEighDefaultMaxIter).
@param tol Off-diagonal convergence tolerance (default kEighDefaultTolerance).
@returns ErrorCode.
*/

ErrorCode eigh(const Matrix* a, Vector* eigenvalues, Matrix* eigenvectors,
               int32_t max_iter = kEighDefaultMaxIter,
               float64_t tol = kEighDefaultTolerance);

}  // namespace linalg
}  // namespace cvlib

#endif  // CVLIB_LINALG_DECOMPOSITION_H_
