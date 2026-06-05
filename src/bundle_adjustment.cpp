#include "mvo/bundle_adjustment.h"

#include "mvo/converter.h"
#include "mvo/visualization.h"

#include "cvlib/calib3d/bundle_adjustment.h"
#include "cvlib/optimize/loss.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace mvo {
namespace {

void set_pose_row(cvlib::Matrix* poses, int32_t row, const Pose& pose) {
    for (int32_t i = 0; i < 9; ++i) {
        cvlib::matrix_set(poses, row, i, pose.r[i]);
    }
    for (int32_t i = 0; i < 3; ++i) {
        cvlib::matrix_set(poses, row, 9 + i, pose.t[i]);
    }
}

void get_pose_row(const cvlib::Matrix& poses, int32_t row, Pose* pose) {
    for (int32_t i = 0; i < 9; ++i) {
        pose->r[i] = cvlib::matrix_get(&poses, row, i);
    }
    for (int32_t i = 0; i < 3; ++i) {
        pose->t[i] = cvlib::matrix_get(&poses, row, 9 + i);
    }
}

double point_distance(const cv::Point3f& a, const cv::Point3f& b) {
    const double dx = static_cast<double>(a.x - b.x);
    const double dy = static_cast<double>(a.y - b.y);
    const double dz = static_cast<double>(a.z - b.z);
    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    return distance;
}

void compute_anchor_similarity(const Pose& target_reference,
                               const Pose& optimized_reference,
                               const Pose& target_current,
                               const Pose& optimized_current,
                               const BundleAdjustmentParameters& parameters,
                               double* scale,
                               double q[9],
                               double c[3]) {
    const cv::Point3f target_ref_center =
        camera_center_from_pose(target_reference);
    const cv::Point3f target_cur_center =
        camera_center_from_pose(target_current);
    const cv::Point3f opt_ref_center =
        camera_center_from_pose(optimized_reference);
    const cv::Point3f opt_cur_center =
        camera_center_from_pose(optimized_current);
    const double target_baseline =
        point_distance(target_ref_center, target_cur_center);
    const double optimized_baseline =
        point_distance(opt_ref_center, opt_cur_center);

    *scale = 1.0;
    if (target_baseline > parameters.min_baseline &&
        optimized_baseline > parameters.min_baseline) {
        *scale = target_baseline / optimized_baseline;
    }

    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            double value = 0.0;
            for (int32_t k = 0; k < 3; ++k) {
                value += target_reference.r[k * 3 + row] *
                         optimized_reference.r[k * 3 + col];
            }
            q[row * 3 + col] = value;
        }
    }

    for (int32_t row = 0; row < 3; ++row) {
        double value = 0.0;
        for (int32_t k = 0; k < 3; ++k) {
            value += target_reference.r[k * 3 + row] *
                     ((*scale * optimized_reference.t[k]) -
                      target_reference.t[k]);
        }
        c[row] = value;
    }
}

Pose anchor_pose_to_reference(const Pose& optimized_pose,
                              double scale,
                              const double q[9],
                              const double c[3]) {
    Pose anchored;
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            double value = 0.0;
            for (int32_t k = 0; k < 3; ++k) {
                value += optimized_pose.r[row * 3 + k] * q[col * 3 + k];
            }
            anchored.r[row * 3 + col] = value;
        }
    }
    for (int32_t row = 0; row < 3; ++row) {
        double rc = 0.0;
        for (int32_t k = 0; k < 3; ++k) {
            rc += anchored.r[row * 3 + k] * c[k];
        }
        anchored.t[row] = scale * optimized_pose.t[row] - rc;
    }
    return anchored;
}

cv::Point3f anchor_point_to_reference(const cv::Point3f& point,
                                      double scale,
                                      const double q[9],
                                      const double c[3]) {
    const double x = static_cast<double>(point.x);
    const double y = static_cast<double>(point.y);
    const double z = static_cast<double>(point.z);
    const cv::Point3f anchored(
        static_cast<float>(scale * (q[0] * x + q[1] * y + q[2] * z) + c[0]),
        static_cast<float>(scale * (q[3] * x + q[4] * y + q[5] * z) + c[1]),
        static_cast<float>(scale * (q[6] * x + q[7] * y + q[8] * z) + c[2]));
    return anchored;
}

bool ba_points_have_positive_depth(const std::vector<cv::Point3f>& points,
                                   const Pose& reference_pose,
                                   const Pose& current_pose) {
    bool ok = true;
    for (const cv::Point3f& point : points) {
        if (depth_in_pose(point, reference_pose) <= 1.0e-6 ||
            depth_in_pose(point, current_pose) <= 1.0e-6) {
            ok = false;
            break;
        }
    }
    return ok;
}

}  // namespace

bool run_two_view_bundle_adjustment(
    const Pose& reference_pose,
    Pose* current_pose,
    const CameraIntrinsics& camera,
    const std::vector<cv::Point2f>& reference_points,
    const std::vector<cv::Point2f>& current_points,
    std::vector<cv::Point3f>* map_points,
    const BundleAdjustmentParameters& parameters,
    const std::string& tag,
    bool debug_geometry) {
    bool ok = false;
    const int32_t aligned_count = static_cast<int32_t>(
        std::min({reference_points.size(), current_points.size(),
                  map_points->size()}));
    const int32_t n_ba = std::min(aligned_count, parameters.max_points);

    if (n_ba >= parameters.min_points) {
        std::vector<cv::Point3f> input_points;
        std::vector<cv::Point2f> input_ref_points;
        std::vector<cv::Point2f> input_cur_points;
        input_points.reserve(static_cast<std::size_t>(n_ba));
        input_ref_points.reserve(static_cast<std::size_t>(n_ba));
        input_cur_points.reserve(static_cast<std::size_t>(n_ba));
        for (int32_t i = 0; i < n_ba; ++i) {
            input_points.push_back((*map_points)[static_cast<std::size_t>(i)]);
            input_ref_points.push_back(
                reference_points[static_cast<std::size_t>(i)]);
            input_cur_points.push_back(
                current_points[static_cast<std::size_t>(i)]);
        }

        const ReprojectionStats before_ref = compute_reprojection_stats(
            input_points, input_ref_points, reference_pose, camera);
        const ReprojectionStats before_cur = compute_reprojection_stats(
            input_points, input_cur_points, *current_pose, camera);

        cvlib::Matrix k = make_camera_matrix(camera);
        cvlib::Matrix poses = cvlib::matrix_create(2, 12);
        cvlib::Matrix ba_points = cvlib::matrix_create(n_ba, 3);
        cvlib::Matrix observations = cvlib::matrix_create(2 * n_ba, 4);
        set_pose_row(&poses, 0, reference_pose);
        set_pose_row(&poses, 1, *current_pose);
        for (int32_t i = 0; i < n_ba; ++i) {
            const cv::Point3f& point = input_points[static_cast<std::size_t>(i)];
            cvlib::matrix_set(&ba_points, i, 0, point.x);
            cvlib::matrix_set(&ba_points, i, 1, point.y);
            cvlib::matrix_set(&ba_points, i, 2, point.z);
            cvlib::matrix_set(&observations, 2 * i, 0, 0.0);
            cvlib::matrix_set(&observations, 2 * i, 1, i);
            cvlib::matrix_set(&observations, 2 * i, 2,
                              input_ref_points[static_cast<std::size_t>(i)].x);
            cvlib::matrix_set(&observations, 2 * i, 3,
                              input_ref_points[static_cast<std::size_t>(i)].y);
            cvlib::matrix_set(&observations, 2 * i + 1, 0, 1.0);
            cvlib::matrix_set(&observations, 2 * i + 1, 1, i);
            cvlib::matrix_set(&observations, 2 * i + 1, 2,
                              input_cur_points[static_cast<std::size_t>(i)].x);
            cvlib::matrix_set(&observations, 2 * i + 1, 3,
                              input_cur_points[static_cast<std::size_t>(i)].y);
        }

        cvlib::calib3d::BAOptions options =
            cvlib::calib3d::default_ba_options();
        options.lm.max_iter = parameters.max_iterations;
        options.lm.loss.type = cvlib::optimize::kLossHuber;
        options.lm.loss.scale = parameters.loss_scale;
        cvlib::calib3d::BAData data = {
            &poses, &ba_points, &observations, &k, nullptr};
        cvlib::optimize::OptimizeReport report = {};
        const cvlib::ErrorCode ba_ec =
            cvlib::calib3d::bundle_adjustment(&data, &options, &report);

        Pose optimized_reference;
        Pose optimized_current;
        get_pose_row(poses, 0, &optimized_reference);
        get_pose_row(poses, 1, &optimized_current);
        double scale = 1.0;
        double q[9];
        double c[3];
        compute_anchor_similarity(reference_pose, optimized_reference,
                                  *current_pose, optimized_current,
                                  parameters,
                                  &scale, q, c);
        Pose anchored_current =
            anchor_pose_to_reference(optimized_current, scale, q, c);
        std::vector<cv::Point3f> optimized_points = input_points;
        for (int32_t i = 0; i < n_ba; ++i) {
            const cv::Point3f point(
                static_cast<float>(cvlib::matrix_get(&ba_points, i, 0)),
                static_cast<float>(cvlib::matrix_get(&ba_points, i, 1)),
                static_cast<float>(cvlib::matrix_get(&ba_points, i, 2)));
            optimized_points[static_cast<std::size_t>(i)] =
                anchor_point_to_reference(point, scale, q, c);
        }

        const ReprojectionStats after_ref = compute_reprojection_stats(
            optimized_points, input_ref_points, reference_pose, camera);
        const ReprojectionStats after_cur = compute_reprojection_stats(
            optimized_points, input_cur_points, anchored_current, camera);
        const double max_ref_p90 =
            std::max(before_ref.p90, parameters.max_reprojection_p90);
        const double max_cur_p90 =
            std::max(before_cur.p90, parameters.max_reprojection_p90);
        const bool cost_ok =
            report.final_cost <= report.initial_cost *
                                 parameters.max_cost_growth;
        const bool reprojection_ok =
            after_ref.valid == n_ba && after_cur.valid == n_ba &&
            after_ref.p90 <= max_ref_p90 && after_cur.p90 <= max_cur_p90;
        const bool depth_ok = ba_points_have_positive_depth(
            optimized_points, reference_pose, anchored_current);
        ok = ba_ec == cvlib::ErrorCode::kSuccess && cost_ok &&
             reprojection_ok && depth_ok;

        if (ok) {
            *current_pose = anchored_current;
            for (int32_t i = 0; i < n_ba; ++i) {
                (*map_points)[static_cast<std::size_t>(i)] =
                    optimized_points[static_cast<std::size_t>(i)];
            }
        }

        if (debug_geometry || ba_ec != cvlib::ErrorCode::kSuccess) {
            std::cout << tag << "_ba status=" << static_cast<int32_t>(ba_ec)
                      << " accepted=" << ok
                      << " points=" << n_ba
                      << " cost=" << report.initial_cost << "->"
                      << report.final_cost
                      << " p90_ref=" << before_ref.p90 << "->"
                      << after_ref.p90
                      << " p90_cur=" << before_cur.p90 << "->"
                      << after_cur.p90
                      << std::endl;
        }

        cvlib::matrix_destroy(&k);
        cvlib::matrix_destroy(&poses);
        cvlib::matrix_destroy(&ba_points);
        cvlib::matrix_destroy(&observations);
    }

    return ok;
}

}  // namespace mvo
