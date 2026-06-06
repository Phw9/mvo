#include "init.h"

#include "bundle_adjustment.h"
#include "config.h"
#include "converter.h"
#include "map_data.h"
#include "visualization.h"

#include <calib3d/multiview.h>

#include <iostream>

#include <opencv2/calib3d.hpp>

namespace mvo {

bool select_orb_two_view_matches(const std::vector<cv::Point2f>& points0,
                                 const std::vector<cv::Point2f>& points1,
                                 const CameraIntrinsics& camera,
                                 const MvoParameters& parameters,
                                 bool debug_geometry,
                                 TwoViewSelection* selection) {
    bool ok = false;
    selection->points0.clear();
    selection->points1.clear();
    selection->indices.clear();
    selection->fundamental_inliers = 0;
    selection->homography_inliers = 0;
    selection->homography_ratio = 0.0;
    selection->selected_model = "none";

    const cv::Mat k = (cv::Mat_<double>(3, 3) << camera.fx, 0.0, camera.cx,
                       0.0, camera.fy, camera.cy, 0.0, 0.0, 1.0);
    cv::Mat mask;
    const cv::Mat e = cv::findEssentialMat(
        points0, points1, k, cv::RANSAC, 0.999,
        parameters.feature.orb_ransac_threshold, mask);
    if (!e.empty()) {
        cv::Mat r;
        cv::Mat t;
        cv::recoverPose(e, points0, points1, k, r, t, mask);
        for (int32_t i = 0; i < mask.rows; ++i) {
            if (mask.at<uchar>(i, 0) != 0) {
                selection->points0.push_back(
                    points0[static_cast<std::size_t>(i)]);
                selection->points1.push_back(
                    points1[static_cast<std::size_t>(i)]);
                selection->indices.push_back(i);
            }
        }
        selection->fundamental_inliers =
            static_cast<int32_t>(selection->points0.size());
    }

    if (selection->fundamental_inliers >=
        parameters.initializer.min_tracks) {
        selection->selected_model = "orb_fundamental";
        ok = true;
    }
    if (debug_geometry) {
        std::cout << "initializer_orb_model input=" << points0.size()
                  << " essential_inliers="
                  << selection->fundamental_inliers
                  << " selected=" << selection->selected_model
                  << " threshold=" << parameters.feature.orb_ransac_threshold
                  << std::endl;
    }
    return ok;
}

bool select_two_view_matches(const std::vector<cv::Point2f>& points0,
                             const std::vector<cv::Point2f>& points1,
                             const InitializerParameters& parameters,
                             bool debug_geometry,
                             TwoViewSelection* selection) {
    bool ok = false;
    selection->points0.clear();
    selection->points1.clear();
    selection->indices.clear();
    selection->fundamental_inliers = 0;
    selection->homography_inliers = 0;
    selection->homography_ratio = 0.0;
    selection->selected_model = "none";
    cvlib::Matrix pix0 = points2f_to_matrix(points0);
    cvlib::Matrix pix1 = points2f_to_matrix(points1);
    cvlib::Matrix f = cvlib::matrix_create(3, 3);
    cvlib::Matrix h = cvlib::matrix_create(3, 3);
    std::vector<int32_t> f_mask(points0.size(), 0);
    std::vector<int32_t> h_mask(points0.size(), 0);
    cvlib::calib3d::RansacParams f_params;
    f_params.max_iters = parameters.ransac_max_iters;
    f_params.inlier_thresh = parameters.fundamental_inlier_threshold;
    f_params.min_inliers = parameters.min_tracks;
    f_params.seed = 0x4D564F;
    cvlib::calib3d::RansacParams h_params;
    h_params.max_iters = parameters.ransac_max_iters;
    h_params.inlier_thresh = parameters.homography_inlier_threshold;
    h_params.min_inliers = parameters.min_tracks;
    h_params.seed = 0x484D564F;

    const cvlib::ErrorCode f_ec = cvlib::calib3d::find_fundamental_ransac(
        &pix0, &pix1, f_params, &f, f_mask.data(),
        &selection->fundamental_inliers);
    const cvlib::ErrorCode h_ec = cvlib::calib3d::find_homography_ransac(
        &pix0, &pix1, h_params, &h, h_mask.data(),
        &selection->homography_inliers);

    const int32_t total_model_inliers =
        selection->fundamental_inliers + selection->homography_inliers;
    if (total_model_inliers > 0) {
        selection->homography_ratio =
            static_cast<double>(selection->homography_inliers) /
            static_cast<double>(total_model_inliers);
    }

    if (f_ec == cvlib::ErrorCode::kSuccess &&
        selection->fundamental_inliers >= parameters.min_tracks &&
        selection->homography_ratio <= parameters.homography_model_ratio) {
        for (int32_t i = 0; i < static_cast<int32_t>(f_mask.size()); ++i) {
            if (f_mask[static_cast<std::size_t>(i)] != 0) {
                selection->points0.push_back(
                    points0[static_cast<std::size_t>(i)]);
                selection->points1.push_back(
                    points1[static_cast<std::size_t>(i)]);
                selection->indices.push_back(i);
            }
        }
        selection->selected_model = "fundamental";
        ok = static_cast<int32_t>(selection->points0.size()) >=
             parameters.min_tracks;
    } else if (f_ec == cvlib::ErrorCode::kSuccess &&
               selection->fundamental_inliers >= parameters.min_tracks) {
        for (int32_t i = 0; i < static_cast<int32_t>(f_mask.size()); ++i) {
            if (f_mask[static_cast<std::size_t>(i)] != 0) {
                selection->points0.push_back(
                    points0[static_cast<std::size_t>(i)]);
                selection->points1.push_back(
                    points1[static_cast<std::size_t>(i)]);
                selection->indices.push_back(i);
            }
        }
        selection->selected_model = "fundamental_quality_checked";
        ok = static_cast<int32_t>(selection->points0.size()) >=
             parameters.min_tracks;
    }
    if (debug_geometry) {
        std::cout << "initializer_model f_status="
                  << static_cast<int32_t>(f_ec)
                  << " h_status=" << static_cast<int32_t>(h_ec)
                  << " input=" << points0.size()
                  << " f_inliers=" << selection->fundamental_inliers
                  << " h_inliers=" << selection->homography_inliers
                  << " h_ratio=" << selection->homography_ratio
                  << " selected=" << selection->selected_model
                  << std::endl;
    }

    cvlib::matrix_destroy(&pix0);
    cvlib::matrix_destroy(&pix1);
    cvlib::matrix_destroy(&f);
    cvlib::matrix_destroy(&h);
    return ok;
}

bool initialize_two_view(const std::vector<cv::Point2f>& points0,
                         const std::vector<cv::Point2f>& points1,
                         const std::vector<cv::Mat>* point1_descriptors,
                         const CameraIntrinsics& camera,
                         const MvoParameters& parameters,
                         bool run_ba,
                         bool debug_geometry,
                         TrackState* state) {
    bool ok = false;
    TwoViewSelection selection;
    const bool model_ok =
        parameters.feature.frontend_mode == 1
            ? select_orb_two_view_matches(points0, points1, camera,
                                          parameters, debug_geometry,
                                          &selection)
            : select_two_view_matches(points0, points1,
                                      parameters.initializer,
                                      debug_geometry, &selection);
    cvlib::Matrix k = make_camera_matrix(camera);
    cvlib::Matrix pix0 = points2f_to_matrix(selection.points0);
    cvlib::Matrix pix1 = points2f_to_matrix(selection.points1);
    cvlib::Matrix norm0 = points2f_to_normalized_matrix(
        selection.points0, camera);
    cvlib::Matrix norm1 = points2f_to_normalized_matrix(
        selection.points1, camera);
    cvlib::Matrix e = cvlib::matrix_create(3, 3);
    cvlib::Matrix r = cvlib::matrix_create(3, 3);
    cvlib::Vector t = cvlib::vector_create(3);
    Pose pose0;
    Pose pose1;
    InitialMap initial_map;
    std::vector<cv::Mat> initial_descriptors;
    set_identity_pose(&pose0);
    set_identity_pose(&pose1);

    cvlib::ErrorCode ec = model_ok ? cvlib::ErrorCode::kSuccess
                                   : cvlib::ErrorCode::kNotConverged;
    if (ec == cvlib::ErrorCode::kSuccess) {
        ec = cvlib::calib3d::find_essential_matrix(&pix0, &pix1, &k, &e);
    }
    if (ec == cvlib::ErrorCode::kSuccess) {
        ec = cvlib::calib3d::recover_pose_from_essential(
            &e, &norm0, &norm1, &r, &t);
    }
    if (ec == cvlib::ErrorCode::kSuccess) {
        copy_cvlib_pose(&r, &t, &pose1);
        cvlib::Matrix p0 = pose_to_projection(pose0);
        cvlib::Matrix p1 = pose_to_projection(pose1);
        cvlib::Matrix points3d = cvlib::matrix_create(pix0.rows, 3);
        ec = cvlib::calib3d::triangulate_points(
            &p0, &p1, &norm0, &norm1, &points3d);

        if (ec == cvlib::ErrorCode::kSuccess) {
            state->prev_points.clear();
            state->map_points.clear();
            for (int32_t i = 0; i < points3d.rows; ++i) {
                const double z = cvlib::matrix_get(&points3d, i, 2);
                const cv::Point3f point3d(
                    static_cast<float>(cvlib::matrix_get(&points3d, i, 0)),
                    static_cast<float>(cvlib::matrix_get(&points3d, i, 1)),
                    static_cast<float>(z));
                const double z2 = depth_in_pose(point3d, pose1);
                if (z > 1.0e-6 && z2 > 1.0e-6) {
                    initial_map.points0.push_back(selection.points0[i]);
                    initial_map.points1.push_back(selection.points1[i]);
                    initial_map.points3d.push_back(point3d);
                    if (point1_descriptors != nullptr &&
                        i < static_cast<int32_t>(selection.indices.size()) &&
                        selection.indices[static_cast<std::size_t>(i)] <
                            static_cast<int32_t>(
                                point1_descriptors->size())) {
                        initial_descriptors.push_back(
                            (*point1_descriptors)[static_cast<std::size_t>(
                                selection.indices[static_cast<std::size_t>(i)])]);
                    } else {
                        initial_descriptors.push_back(cv::Mat());
                    }
                    state->prev_points.push_back(selection.points1[i]);
                    state->map_points.push_back(make_map_point(
                        point3d, 1, 2, 0.0, parameters.mapping,
                        initial_descriptors.back()));
                }
            }
            const ReprojectionStats stats0 = compute_reprojection_stats(
                map_point_positions(state->map_points), initial_map.points0,
                pose0, camera);
            const ReprojectionStats stats1 = compute_reprojection_stats(
                map_point_positions(state->map_points), initial_map.points1,
                pose1, camera);
            const double parallax_deg = median_parallax_deg(
                map_point_positions(state->map_points), pose0, pose1);
            const double min_parallax_deg =
                parameters.feature.frontend_mode == 1
                    ? parameters.initializer.orb_min_parallax_deg
                    : parameters.initializer.min_parallax_deg;
            const double max_triangulation_p90 =
                parameters.feature.frontend_mode == 1
                    ? parameters.initializer.orb_max_triangulation_p90
                    : parameters.initializer.max_triangulation_p90;
            const bool quality_ok =
                static_cast<int32_t>(state->map_points.size()) >=
                    parameters.initializer.min_tracks &&
                parallax_deg >= min_parallax_deg &&
                stats0.p90 <= max_triangulation_p90 &&
                stats1.p90 <= max_triangulation_p90;
            if (debug_geometry) {
                std::cout << "triangulation_debug input="
                          << selection.points0.size()
                          << " positive_depth=" << state->map_points.size()
                          << " parallax_deg=" << parallax_deg
                          << " reproj0_rmse=" << stats0.rmse
                          << " reproj1_rmse=" << stats1.rmse
                          << " reproj1_p90=" << stats1.p90
                          << " pose_t=[" << pose1.t[0] << ","
                          << pose1.t[1] << "," << pose1.t[2] << "]"
                          << std::endl;
            }
            if (!quality_ok) {
                ec = cvlib::ErrorCode::kNotConverged;
            }
        }

        if (ec == cvlib::ErrorCode::kSuccess && run_ba &&
            static_cast<int32_t>(state->map_points.size()) >=
                parameters.initializer.min_map_points) {
            run_two_view_bundle_adjustment(
                pose0, &pose1, camera, initial_map.points0,
                initial_map.points1, &initial_map.points3d,
                parameters.bundle_adjustment,
                "initial", debug_geometry);
            state->map_points.clear();
            state->map_points.reserve(initial_map.points3d.size());
            for (int32_t i = 0;
                 i < static_cast<int32_t>(initial_map.points3d.size());
                 ++i) {
                state->map_points.push_back(make_map_point(
                    initial_map.points3d[static_cast<std::size_t>(i)],
                    1, 2, 0.0, parameters.mapping,
                    initial_descriptors[static_cast<std::size_t>(i)]));
            }
        }

        cvlib::matrix_destroy(&p0);
        cvlib::matrix_destroy(&p1);
        cvlib::matrix_destroy(&points3d);
    }

    if (debug_geometry && ec != cvlib::ErrorCode::kSuccess) {
        std::cout << "initialization_failed status="
                  << static_cast<int32_t>(ec) << std::endl;
    }
    if (ec == cvlib::ErrorCode::kSuccess &&
        static_cast<int32_t>(state->map_points.size()) >=
            parameters.initializer.min_map_points) {
        state->last_pose = pose1;
        state->keyframes = 2;
        ok = true;
    }

    cvlib::matrix_destroy(&k);
    cvlib::matrix_destroy(&pix0);
    cvlib::matrix_destroy(&pix1);
    cvlib::matrix_destroy(&norm0);
    cvlib::matrix_destroy(&norm1);
    cvlib::matrix_destroy(&e);
    cvlib::matrix_destroy(&r);
    cvlib::vector_destroy(&t);
    return ok;
}

}  // namespace mvo
