#include "pose_estimation.h"

#include "converter.h"
#include "visualization.h"

#include <calib3d/pnp.h>
#include <optimize/lm.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>

namespace mvo {

bool solve_pnp_once(const std::vector<cv::Point3f>& map_points,
                    const std::vector<cv::Point2f>& image_points,
                    const CameraIntrinsics& camera,
                    const Pose& initial_pose,
                    Pose* pose,
                    cvlib::optimize::OptimizeReport* report,
                    cvlib::ErrorCode* error_code) {
    bool ok = false;
    cvlib::Matrix world = points3f_to_matrix(map_points);
    cvlib::Matrix image = points2f_to_matrix(image_points);
    cvlib::Matrix k = make_camera_matrix(camera);
    cvlib::Matrix r = cvlib::matrix_create(3, 3);
    cvlib::Vector t = cvlib::vector_create(3);
    cvlib::Matrix r_init = pose_rotation_to_matrix(initial_pose);
    cvlib::Vector t_init = pose_translation_to_vector(initial_pose);
    *error_code = cvlib::calib3d::solve_pnp(
        &world, &image, &k, &r, &t, nullptr, &r_init, &t_init, nullptr,
        report);
    if (*error_code == cvlib::ErrorCode::kSuccess) {
        copy_cvlib_pose(&r, &t, pose);
        ok = true;
    }
    cvlib::matrix_destroy(&world);
    cvlib::matrix_destroy(&image);
    cvlib::matrix_destroy(&k);
    cvlib::matrix_destroy(&r);
    cvlib::vector_destroy(&t);
    cvlib::matrix_destroy(&r_init);
    cvlib::vector_destroy(&t_init);
    return ok;
}

void select_indexed_points(const std::vector<cv::Point3f>& map_points,
                           const std::vector<cv::Point2f>& image_points,
                           const std::vector<int32_t>& indices,
                           std::vector<cv::Point3f>* selected_map_points,
                           std::vector<cv::Point2f>* selected_image_points) {
    selected_map_points->clear();
    selected_image_points->clear();
    selected_map_points->reserve(indices.size());
    selected_image_points->reserve(indices.size());
    for (const int32_t index : indices) {
        selected_map_points->push_back(
            map_points[static_cast<std::size_t>(index)]);
        selected_image_points->push_back(
            image_points[static_cast<std::size_t>(index)]);
    }
}

bool run_pnp_ransac(const std::vector<cv::Point3f>& map_points,
                    const std::vector<cv::Point2f>& image_points,
                    const CameraIntrinsics& camera,
                    const Pose& initial_pose,
                    const PnpParameters& parameters,
                    bool debug_geometry,
                    Pose* best_pose,
                    std::vector<cv::Point3f>* best_map_points,
                    std::vector<cv::Point2f>* best_image_points,
                    cvlib::optimize::OptimizeReport* best_report,
                    cvlib::ErrorCode* best_error_code,
                    int32_t* best_inlier_count) {
    const int32_t total = static_cast<int32_t>(map_points.size());
    const int32_t sample_size =
        std::min(parameters.ransac_sample_size, total);
    if (parameters.ransac_iterations <= 0 ||
        total < parameters.min_tracks ||
        sample_size < parameters.min_tracks) {
        return false;
    }

    std::vector<int32_t> all_indices(static_cast<std::size_t>(total));
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::mt19937 rng(0x4D564F50);

    int32_t best_score = -1;
    double best_p90 = std::numeric_limits<double>::infinity();
    Pose candidate_pose = initial_pose;
    for (int32_t iter = 0; iter < parameters.ransac_iterations; ++iter) {
        std::shuffle(all_indices.begin(), all_indices.end(), rng);
        std::vector<int32_t> sample_indices(
            all_indices.begin(), all_indices.begin() + sample_size);
        std::vector<cv::Point3f> sample_map_points;
        std::vector<cv::Point2f> sample_image_points;
        select_indexed_points(map_points, image_points, sample_indices,
                              &sample_map_points, &sample_image_points);

        Pose hypothesis = initial_pose;
        cvlib::optimize::OptimizeReport report = {};
        cvlib::ErrorCode ec = cvlib::ErrorCode::kUnknownMethod;
        if (!solve_pnp_once(sample_map_points, sample_image_points, camera,
                            initial_pose, &hypothesis, &report, &ec)) {
            continue;
        }

        std::vector<cv::Point3f> inlier_map_points;
        std::vector<cv::Point2f> inlier_image_points;
        filter_by_reprojection(map_points, image_points, hypothesis, camera,
                               parameters.reprojection_inlier_threshold,
                               &inlier_map_points, &inlier_image_points);
        const int32_t inliers =
            static_cast<int32_t>(inlier_image_points.size());
        if (inliers < parameters.min_stable_inliers) {
            continue;
        }
        const ReprojectionStats stats = compute_reprojection_stats(
            inlier_map_points, inlier_image_points, hypothesis, camera);
        if (inliers > best_score ||
            (inliers == best_score && stats.p90 < best_p90)) {
            best_score = inliers;
            best_p90 = stats.p90;
            candidate_pose = hypothesis;
            *best_map_points = inlier_map_points;
            *best_image_points = inlier_image_points;
            *best_report = report;
            *best_error_code = ec;
        }
    }

    bool ok = best_score >= parameters.min_stable_inliers;
    if (ok) {
        Pose refined_pose = candidate_pose;
        cvlib::optimize::OptimizeReport refined_report = {};
        cvlib::ErrorCode refined_ec = cvlib::ErrorCode::kUnknownMethod;
        const bool refined_ok = solve_pnp_once(
            *best_map_points, *best_image_points, camera, candidate_pose,
            &refined_pose, &refined_report, &refined_ec);
        if (refined_ok) {
            std::vector<cv::Point3f> refined_map_points;
            std::vector<cv::Point2f> refined_image_points;
            filter_by_reprojection(
                *best_map_points, *best_image_points, refined_pose, camera,
                parameters.reprojection_inlier_threshold, &refined_map_points,
                &refined_image_points);
            if (static_cast<int32_t>(refined_image_points.size()) >=
                parameters.min_stable_inliers) {
                *best_pose = refined_pose;
                *best_map_points = refined_map_points;
                *best_image_points = refined_image_points;
                *best_report = refined_report;
                *best_error_code = refined_ec;
                *best_inlier_count =
                    static_cast<int32_t>(refined_image_points.size());
            } else {
                *best_pose = candidate_pose;
                *best_inlier_count = best_score;
            }
        } else {
            *best_pose = candidate_pose;
            *best_inlier_count = best_score;
        }
    }
    if (debug_geometry) {
        std::cout << "pnp_ransac input=" << total
                  << " iterations=" << parameters.ransac_iterations
                  << " sample_size=" << sample_size
                  << " best_inliers=" << best_score
                  << " best_p90=" << best_p90
                  << " ok=" << ok << std::endl;
    }
    return ok;
}

bool run_pnp(std::vector<cv::Point3f>* map_points,
             std::vector<cv::Point2f>* image_points,
             const CameraIntrinsics& camera,
             const Pose& initial_pose,
             const PnpParameters& parameters,
             bool debug_geometry,
             Pose* pose) {
    bool ok = false;
    cvlib::optimize::OptimizeReport report = {};
    cvlib::ErrorCode ec = cvlib::ErrorCode::kUnknownMethod;
    Pose candidate = initial_pose;
    std::string reject_reason;
    int32_t inlier_count = 0;
    if (static_cast<int32_t>(map_points->size()) >= parameters.min_tracks &&
        parameters.ransac_iterations > 0) {
        std::vector<cv::Point3f> ransac_map_points;
        std::vector<cv::Point2f> ransac_image_points;
        ok = run_pnp_ransac(
            *map_points, *image_points, camera, initial_pose, parameters,
            debug_geometry, &candidate, &ransac_map_points,
            &ransac_image_points, &report, &ec, &inlier_count);
        if (ok) {
            *map_points = ransac_map_points;
            *image_points = ransac_image_points;
        }
    }
    if (!ok &&
        static_cast<int32_t>(map_points->size()) >= parameters.min_tracks) {
        ok = solve_pnp_once(*map_points, *image_points, camera, initial_pose,
                            &candidate, &report, &ec);
    }
    if (ok) {
        std::vector<cv::Point3f> inlier_map_points;
        std::vector<cv::Point2f> inlier_image_points;
        filter_by_reprojection(*map_points, *image_points, candidate, camera,
                               parameters.reprojection_inlier_threshold,
                               &inlier_map_points, &inlier_image_points);
        inlier_count = static_cast<int32_t>(inlier_map_points.size());
        if (inlier_count >= parameters.min_stable_inliers) {
            if (inlier_map_points.size() < map_points->size()) {
                cvlib::optimize::OptimizeReport refined_report = {};
                cvlib::ErrorCode refined_ec = cvlib::ErrorCode::kUnknownMethod;
                Pose refined_pose = candidate;
                const bool refined_ok = solve_pnp_once(
                    inlier_map_points, inlier_image_points, camera, candidate,
                    &refined_pose, &refined_report, &refined_ec);
                if (refined_ok) {
                    candidate = refined_pose;
                    report = refined_report;
                    ec = refined_ec;
                    *map_points = inlier_map_points;
                    *image_points = inlier_image_points;
                } else {
                    ok = false;
                    reject_reason = "refine_failed";
                    ec = refined_ec;
                }
            }
        } else {
            ok = false;
            reject_reason = "too_few_inliers";
        }
    }
    if (ok) {
        const ReprojectionStats stats = compute_reprojection_stats(
            *map_points, *image_points, candidate, camera);
        if (stats.p90 > parameters.max_reprojection_p90) {
            ok = false;
            reject_reason = "high_reprojection";
        }
    }
    if (ok) {
        *pose = candidate;
        const ReprojectionStats stats = compute_reprojection_stats(
            *map_points, *image_points, *pose, camera);
        std::cout << "pnp tracks=" << image_points->size()
                  << " cost=" << report.final_cost
                  << " reproj_rmse=" << stats.rmse
                  << " reproj_median=" << stats.median
                  << " reproj_p90=" << stats.p90
                  << std::endl;
    } else if (!reject_reason.empty()) {
        std::cout << "pnp_rejected reason=" << reject_reason
                  << " status=" << static_cast<int32_t>(ec)
                  << " tracks=" << image_points->size()
                  << " inliers=" << inlier_count << std::endl;
    } else {
        std::cout << "pnp_failed status=" << static_cast<int32_t>(ec)
                  << " tracks=" << image_points->size() << std::endl;
    }
    if (debug_geometry && ok) {
        const cv::Point3f center = camera_center_from_pose(*pose);
        std::cout << "pnp_debug camera_center=[" << center.x << ","
                  << center.y << "," << center.z << "]"
                  << " pose_t=[" << pose->t[0] << "," << pose->t[1]
                  << "," << pose->t[2] << "]" << std::endl;
    }
    return ok;
}

}  // namespace mvo
