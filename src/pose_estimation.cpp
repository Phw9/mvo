#include "pose_estimation.h"

#include "converter.h"
#include "visualization.h"

#include <calib3d/pnp.h>
#include <optimize/lm.h>

#include <iostream>
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
    if (static_cast<int32_t>(map_points->size()) >= parameters.min_tracks) {
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
