// Input validation helpers.
//
// Shape/null validators are cheap and run on every call. The deep
// validators (finite, symmetric covariance, orthonormal rotation,
// camera matrix) run on user-supplied per-call inputs -- measurements,
// noise matrices, intrinsics -- but not on filter-evolved state or on
// large point arrays, whose per-element scans would dominate hot paths.

#ifndef CVLIB_VALIDATORS_H_
#define CVLIB_VALIDATORS_H_

#include "types.h"
#include "defs.h"
#include "error_codes.h"

#include <cmath>

namespace cvlib {

// Orthonormality tolerance for user-supplied rotation measurements.
static constexpr float64_t kRotationOrthoTolerance = 1e-6;
// Symmetry tolerance for user-supplied covariance/noise matrices,
// relative to the largest |entry| (floored at 1.0 so zero matrices pass).
static constexpr float64_t kCovarianceSymmetryTol = 1e-9;

/*
Validates that a matrix pointer and dimensions are usable for computation.

@param m Candidate matrix.
@returns ErrorCode: kSuccess, kNullPointer, or kInvalidDimension.

*/

inline ErrorCode validate_matrix(const Matrix* m) {
    ErrorCode result = ErrorCode::kSuccess;
    if (m == nullptr || m->data == nullptr) {
        result = ErrorCode::kNullPointer;
    } else if (m->rows <= 0 || m->cols <= 0) {
        result = ErrorCode::kInvalidDimension;
    } else if (m->type != k64FC1) {
        // Numeric kernels operate on single-channel f64 buffers; a matrix
        // carrying image pixels (other depths/channels) is rejected here.
        result = ErrorCode::kInvalidArgument;
    }
    return result;
}

/*
Like validate_matrix and additionally requires a square matrix.

@param m Candidate matrix.
@returns ErrorCode: kSuccess, kNullPointer, kInvalidDimension, or kInvalidShape.

*/

inline ErrorCode validate_square_matrix(const Matrix* m) {
    ErrorCode result = validate_matrix(m);
    if (result == ErrorCode::kSuccess && m->rows != m->cols) {
        result = ErrorCode::kInvalidShape;
    }
    return result;
}

/*
Validates that a vector pointer and length are usable.

@param v Candidate vector.
@returns ErrorCode: kSuccess, kNullPointer, or kInvalidDimension.

*/

inline ErrorCode validate_vector(const Vector* v) {
    ErrorCode result = ErrorCode::kSuccess;
    if (v == nullptr || v->data == nullptr) {
        result = ErrorCode::kNullPointer;
    } else if (v->size <= 0) {
        result = ErrorCode::kInvalidDimension;
    }
    return result;
}

/*
Validates exact matrix shape after validate_matrix.

@param m Candidate matrix.
@param expected_rows Required row count.
@param expected_cols Required column count.
@returns ErrorCode: kSuccess, kNullPointer, kInvalidDimension, or kInvalidShape.

*/

inline ErrorCode validate_matrix_dimensions(const Matrix* m,
                                            int32_t expected_rows,
                                            int32_t expected_cols) {
    ErrorCode result = validate_matrix(m);
    if (result == ErrorCode::kSuccess) {
        if (m->rows != expected_rows || m->cols != expected_cols) {
            result = ErrorCode::kInvalidShape;
        }
    }
    return result;
}

/*
Validates a relative tolerance expected in the open interval (0, 1).

@param tol User-supplied threshold.
@returns ErrorCode: kSuccess or kInvalidArgument.

*/

inline ErrorCode validate_tolerance(float64_t tol) {
    ErrorCode result = ErrorCode::kSuccess;
    if (tol <= 0.0 || tol >= 1.0) {
        result = ErrorCode::kInvalidArgument;
    }
    return result;
}

/*
Validates a positive iteration budget.

@param max_iter Maximum iteration count.
@returns ErrorCode: kSuccess or kInvalidArgument.

*/

inline ErrorCode validate_max_iterations(int32_t max_iter) {
    ErrorCode result = ErrorCode::kSuccess;
    if (max_iter <= 0) {
        result = ErrorCode::kInvalidArgument;
    }
    return result;
}

/*
Validates shape-only for a 3-by-3 rotation matrix buffer.

@param m Candidate matrix.
@returns ErrorCode: kSuccess, kNullPointer, or kInvalidShape.

*/

inline ErrorCode validate_rotation_matrix(const Matrix* m) {
    ErrorCode result = ErrorCode::kSuccess;
    if (m == nullptr || m->data == nullptr) {
        result = ErrorCode::kNullPointer;
    } else if (m->rows != kRotationMatrixSize ||
               m->cols != kRotationMatrixSize) {
        result = ErrorCode::kInvalidShape;
    }
    return result;
}

/*
Validates shape-only for a length-3 translation vector.

@param v Candidate vector.
@returns ErrorCode: kSuccess, kNullPointer, or kInvalidShape.

*/

inline ErrorCode validate_translation_vector(const Vector* v) {
    ErrorCode result = ErrorCode::kSuccess;
    if (v == nullptr || v->data == nullptr) {
        result = ErrorCode::kNullPointer;
    } else if (v->size != kTranslationVectorSize) {
        result = ErrorCode::kInvalidShape;
    }
    return result;
}

/*
Validates shape-only for a 4-by-4 homogeneous transform buffer.

@param m Candidate matrix.
@returns ErrorCode: kSuccess, kNullPointer, or kInvalidShape.

*/

inline ErrorCode validate_transform_matrix(const Matrix* m) {
    ErrorCode result = ErrorCode::kSuccess;
    if (m == nullptr || m->data == nullptr) {
        result = ErrorCode::kNullPointer;
    } else if (m->rows != kTransformMatrixSize ||
               m->cols != kTransformMatrixSize) {
        result = ErrorCode::kInvalidShape;
    }
    return result;
}

/*
Validates that every matrix entry is finite (rejects NaN/Inf).

@param m Candidate matrix.
@returns ErrorCode: kSuccess, kNullPointer, kInvalidDimension, or
  kInvalidArgument on a non-finite entry.

*/

inline ErrorCode validate_finite_matrix(const Matrix* m) {
    ErrorCode result = validate_matrix(m);
    if (result == ErrorCode::kSuccess) {
        const float64_t* p = m->ptr<float64_t>();
        const int32_t n = m->rows * m->cols;
        for (int32_t i = 0; i < n; ++i) {
            if (!std::isfinite(p[i])) {
                result = ErrorCode::kInvalidArgument;
                break;
            }
        }
    }
    return result;
}

/*
Validates that every vector entry is finite (rejects NaN/Inf).

@param v Candidate vector.
@returns ErrorCode: kSuccess, kNullPointer, kInvalidDimension, or
  kInvalidArgument on a non-finite entry.

*/

inline ErrorCode validate_finite_vector(const Vector* v) {
    ErrorCode result = validate_vector(v);
    if (result == ErrorCode::kSuccess) {
        for (int32_t i = 0; i < v->size; ++i) {
            if (!std::isfinite(v->data[i])) {
                result = ErrorCode::kInvalidArgument;
                break;
            }
        }
    }
    return result;
}

/*
Deep validation for a user-supplied covariance / noise matrix: square,
finite, and symmetric within kCovarianceSymmetryTol relative to the
largest |entry|. Positive semi-definiteness is intentionally NOT
checked: zero variances are legal and the SPD consumers already fall
back to pseudo-inversion.

@param m Candidate covariance matrix.
@returns ErrorCode: kSuccess, kNullPointer, kInvalidDimension,
  kInvalidShape, or kInvalidArgument (non-finite or asymmetric).

*/

inline ErrorCode validate_noise_covariance(const Matrix* m) {
    ErrorCode result = validate_square_matrix(m);
    if (result == ErrorCode::kSuccess) {
        result = validate_finite_matrix(m);
    }
    if (result == ErrorCode::kSuccess) {
        float64_t max_abs = 0.0;
        const float64_t* p = m->ptr<float64_t>();
        const int32_t n = m->rows * m->cols;
        for (int32_t i = 0; i < n; ++i) {
            const float64_t a = std::fabs(p[i]);
            if (a > max_abs) {
                max_abs = a;
            }
        }
        const float64_t tol =
            kCovarianceSymmetryTol * ((max_abs > 1.0) ? max_abs : 1.0);
        for (int32_t i = 0;
             i < m->rows && result == ErrorCode::kSuccess; ++i) {
            for (int32_t j = i + 1; j < m->cols; ++j) {
                if (std::fabs(matrix_get(m, i, j) -
                              matrix_get(m, j, i)) > tol) {
                    result = ErrorCode::kInvalidArgument;
                    break;
                }
            }
        }
    }
    return result;
}

/*
Deep validation for a user-supplied rotation measurement: 3x3, finite,
orthonormal within kRotationOrthoTolerance, and right-handed (det > 0).
Filter-evolved state rotations keep the shape-only
validate_rotation_matrix.

@param m Candidate rotation matrix.
@returns ErrorCode: kSuccess, kNullPointer, kInvalidShape, or
  kInvalidArgument (non-finite, non-orthonormal, or a reflection).

*/

inline ErrorCode validate_orthonormal_rotation(const Matrix* m) {
    ErrorCode result = validate_rotation_matrix(m);
    if (result == ErrorCode::kSuccess) {
        result = validate_finite_matrix(m);
    }
    if (result == ErrorCode::kSuccess) {
        for (int32_t i = 0;
             i < kRotationMatrixSize && result == ErrorCode::kSuccess;
             ++i) {
            for (int32_t j = i; j < kRotationMatrixSize; ++j) {
                float64_t dot = 0.0;
                for (int32_t r = 0; r < kRotationMatrixSize; ++r) {
                    dot += matrix_get(m, r, i) * matrix_get(m, r, j);
                }
                const float64_t expected = (i == j) ? 1.0 : 0.0;
                if (std::fabs(dot - expected) > kRotationOrthoTolerance) {
                    result = ErrorCode::kInvalidArgument;
                    break;
                }
            }
        }
    }
    if (result == ErrorCode::kSuccess) {
        const float64_t det =
            matrix_get(m, 0, 0) * (matrix_get(m, 1, 1) * matrix_get(m, 2, 2) -
                                   matrix_get(m, 1, 2) * matrix_get(m, 2, 1)) -
            matrix_get(m, 0, 1) * (matrix_get(m, 1, 0) * matrix_get(m, 2, 2) -
                                   matrix_get(m, 1, 2) * matrix_get(m, 2, 0)) +
            matrix_get(m, 0, 2) * (matrix_get(m, 1, 0) * matrix_get(m, 2, 1) -
                                   matrix_get(m, 1, 1) * matrix_get(m, 2, 0));
        if (det <= 0.0) {
            result = ErrorCode::kInvalidArgument;
        }
    }
    return result;
}

/*
Deep validation for a 3x3 camera intrinsic matrix: finite entries and
positive focal lengths K(0,0) and K(1,1).

@param k Candidate intrinsic matrix.
@returns ErrorCode: kSuccess, kNullPointer, kInvalidShape, or
  kInvalidArgument (non-finite or non-positive focal length).

*/

inline ErrorCode validate_camera_matrix(const Matrix* k) {
    ErrorCode result = validate_matrix(k);
    if (result == ErrorCode::kSuccess &&
        (k->rows != kRotationMatrixSize ||
         k->cols != kRotationMatrixSize)) {
        result = ErrorCode::kInvalidShape;
    }
    if (result == ErrorCode::kSuccess) {
        result = validate_finite_matrix(k);
    }
    if (result == ErrorCode::kSuccess &&
        (matrix_get(k, 0, 0) <= 0.0 || matrix_get(k, 1, 1) <= 0.0)) {
        result = ErrorCode::kInvalidArgument;
    }
    return result;
}

/*
Validates an N-by-3 point cloud with positive N.

@param m Candidate matrix.
@returns ErrorCode: kSuccess, kNullPointer, or kInvalidShape.

*/

inline ErrorCode validate_points_3d(const Matrix* m) {
    ErrorCode result = ErrorCode::kSuccess;
    if (m == nullptr || m->data == nullptr) {
        result = ErrorCode::kNullPointer;
    } else if (m->rows <= 0 || m->cols != kPoint3dSize) {
        result = ErrorCode::kInvalidShape;
    }
    return result;
}

/*
Validates an N-by-2 point set with positive N.

@param m Candidate matrix.
@returns ErrorCode: kSuccess, kNullPointer, or kInvalidShape.

*/

inline ErrorCode validate_points_2d(const Matrix* m) {
    ErrorCode result = ErrorCode::kSuccess;
    if (m == nullptr || m->data == nullptr) {
        result = ErrorCode::kNullPointer;
    } else if (m->rows <= 0 || m->cols != kPoint2dSize) {
        result = ErrorCode::kInvalidShape;
    }
    return result;
}

/*
Validates a non-null generic pointer.

@param ptr Candidate pointer.
@returns ErrorCode: kSuccess or kNullPointer.

*/

inline ErrorCode validate_output_ptr(const void* ptr) {
    ErrorCode result = ErrorCode::kSuccess;
    if (ptr == nullptr) {
        result = ErrorCode::kNullPointer;
    }
    return result;
}

/*
Validates that an output matrix has a non-null data pointer.

@param m Candidate output matrix.
@returns ErrorCode: kSuccess or kNullPointer.

*/

inline ErrorCode validate_result_matrix(const Matrix* m) {
    ErrorCode result = ErrorCode::kSuccess;
    if (m == nullptr || m->data == nullptr) {
        result = ErrorCode::kNullPointer;
    }
    return result;
}

/*
Validates that an output vector has a non-null data pointer.

@param v Candidate output vector.
@returns ErrorCode: kSuccess or kNullPointer.

*/

inline ErrorCode validate_result_vector(const Vector* v) {
    ErrorCode result = ErrorCode::kSuccess;
    if (v == nullptr || v->data == nullptr) {
        result = ErrorCode::kNullPointer;
    }
    return result;
}

}  // namespace cvlib

#endif  // CVLIB_VALIDATORS_H_
