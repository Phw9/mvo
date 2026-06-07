// Singular value decomposition (full U, S, Vt).

#ifndef CVLIB_LINALG_SVD_H_
#define CVLIB_LINALG_SVD_H_

#include "types.h"
#include "defs.h"
#include "error_codes.h"

#include <cstdint>

namespace cvlib {
namespace linalg {

// Default SVD convergence tolerance.
static constexpr float64_t kSvdDefaultTolerance = 1e-10;

/*
Computes the factorization A = U S V^T with a one-sided Jacobi method
(no external LAPACK). U and Vt are optional and independent: pass
nullptr to skip allocating and writing the corresponding factor.

@param a Input matrix, m-by-n.
@param s Output singular values on the diagonal; rows/cols can be any
  size since only entries up to min(rows, cols, min(m, n)) are written.
@param u Optional m-by-m left singular vectors; null skips computation.
@param vt Optional n-by-n transpose of right singular vectors; null skips.
@param max_iter Upper bound on Jacobi sweeps (default kSvdMaxIterations).
@param tol User convergence tolerance (default kSvdDefaultTolerance).
@returns ErrorCode.

*/

ErrorCode svd(const Matrix* a, Matrix* s,
              Matrix* u = nullptr, Matrix* vt = nullptr,
              int32_t max_iter = kSvdMaxIterations,
              float64_t tol = kSvdDefaultTolerance);

}  // namespace linalg
}  // namespace cvlib

#endif  // CVLIB_LINALG_SVD_H_
