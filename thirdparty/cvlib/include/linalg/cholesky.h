// Cholesky factorization for symmetric positive-definite systems.

#ifndef CVLIB_LINALG_CHOLESKY_H_
#define CVLIB_LINALG_CHOLESKY_H_

#include "types.h"
#include "error_codes.h"

#include <cstdint>

namespace cvlib {
namespace linalg {

/*
Computes the lower-triangular Cholesky factor L such that A = L L^T.
Strict upper triangle of result is zeroed.

@param a Input symmetric positive-definite n-by-n matrix.
@param l Output n-by-n lower-triangular factor; must be pre-allocated.
@returns ErrorCode: kSuccess, kInvalidShape, or kNotPositiveDefinite.

*/

ErrorCode cholesky(const Matrix* a, Matrix* l);

/*
Solves A x = b given L from cholesky(A) by forward and back substitution.

@param l Lower-triangular Cholesky factor produced by cholesky.
@param b Right-hand side vector of length n.
@param x Output solution vector of length n; must be pre-allocated.
@returns ErrorCode.

*/

ErrorCode cholesky_solve(const Matrix* l, const Vector* b, Vector* x);

}  // namespace linalg
}  // namespace cvlib

#endif  // CVLIB_LINALG_CHOLESKY_H_
