// Input validation helpers.

#ifndef CVLIB_VALIDATORS_H_
#define CVLIB_VALIDATORS_H_

#include "../types.h"
#include "../defs.h"
#include "../error_codes.h"

namespace cvlib {

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
