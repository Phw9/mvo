#include "loop_closure.h"

#include "converter.h"

#include <calib3d/alignment.h>
#include <calib3d/multiview.h>

#include <opencv2/features2d.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

namespace mvo {
namespace {

// Below this many keyframes the sequential scan is cheaper than spawning
// worker threads.
constexpr std::size_t kParallelScanMinKeyframes = 256;

struct ScanBest {
    double score = 0.0;
    std::size_t index = 0;
    bool found = false;
};

ScanBest scan_candidate_range(const BowDatabase& db,
                              const DBoW2::BowVector& bow,
                              int32_t frame_id,
                              const LoopClosureParameters& parameters,
                              std::size_t begin, std::size_t end) {
    ScanBest best;
    for (std::size_t i = begin; i < end; ++i) {
        const LoopKeyframe& keyframe = db.keyframes[i];
        if (frame_id - keyframe.frame_id > parameters.recent_exclusion) {
            const double score = db.vocabulary.score(bow, keyframe.bow);
            if (score > best.score) {
                best.score = score;
                best.index = i;
                best.found = true;
            }
        }
    }
    return best;
}

// The per-pair scores are computed exactly as in the sequential scan and the
// chunk results are reduced in index order with a strict ">", so the winner
// (first keyframe with the maximal score) is identical to the sequential
// result regardless of thread scheduling.
const LoopKeyframe* find_best_candidate(const BowDatabase& db,
                                        const DBoW2::BowVector& bow,
                                        int32_t frame_id,
                                        const LoopClosureParameters& parameters,
                                        double* best_score) {
    *best_score = 0.0;
    const std::size_t total = db.keyframes.size();
    const std::size_t worker_count = std::min<std::size_t>(
        std::max(1U, std::thread::hardware_concurrency()),
        total / (kParallelScanMinKeyframes / 2) + 1);
    ScanBest best;
    if (total < kParallelScanMinKeyframes || worker_count < 2) {
        best = scan_candidate_range(db, bow, frame_id, parameters, 0, total);
    } else {
        std::vector<std::future<ScanBest>> futures;
        futures.reserve(worker_count);
        const std::size_t chunk = (total + worker_count - 1) / worker_count;
        for (std::size_t begin = 0; begin < total; begin += chunk) {
            const std::size_t end = std::min(total, begin + chunk);
            futures.push_back(std::async(
                std::launch::async, scan_candidate_range, std::cref(db),
                std::cref(bow), frame_id, std::cref(parameters), begin,
                end));
        }
        for (std::future<ScanBest>& future : futures) {
            const ScanBest chunk_best = future.get();
            if (chunk_best.found && chunk_best.score > best.score) {
                best = chunk_best;
            }
        }
    }
    if (!best.found) {
        return nullptr;
    }
    *best_score = best.score;
    return &db.keyframes[best.index];
}

// Ratio-test match that keeps the keypoint indices. The indices are what lets
// a loop correspondence be joined with the locally triangulated depth of the
// same keypoint, which is where the metric scale comes from.
std::vector<std::pair<int32_t, int32_t>> match_descriptor_indices(
    const cv::Mat& query_descriptors,
    const cv::Mat& train_descriptors,
    double match_ratio) {
    std::vector<std::pair<int32_t, int32_t>> matches;
    if (!query_descriptors.empty() && !train_descriptors.empty()) {
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher.knnMatch(query_descriptors, train_descriptors, knn_matches, 2);
        matches.reserve(knn_matches.size());
        for (const std::vector<cv::DMatch>& match : knn_matches) {
            if (match.size() == 2U &&
                match[0].distance <
                    static_cast<float>(match_ratio) * match[1].distance) {
                matches.emplace_back(match[0].queryIdx, match[0].trainIdx);
            }
        }
    }
    return matches;
}

int32_t match_keyframe_points(const cv::Mat& query_descriptors,
                              const std::vector<cv::KeyPoint>& query_keypoints,
                              const LoopKeyframe& candidate,
                              double match_ratio,
                              std::vector<cv::Point2f>* query_points,
                              std::vector<cv::Point2f>* match_points,
                              std::vector<std::pair<int32_t, int32_t>>*
                                  match_indices) {
    query_points->clear();
    match_points->clear();
    *match_indices = match_descriptor_indices(query_descriptors,
                                              candidate.descriptors,
                                              match_ratio);
    query_points->reserve(match_indices->size());
    match_points->reserve(match_indices->size());
    for (const std::pair<int32_t, int32_t>& match : *match_indices) {
        query_points->push_back(
            query_keypoints[static_cast<std::size_t>(match.first)].pt);
        match_points->push_back(
            candidate.keypoints[static_cast<std::size_t>(match.second)].pt);
    }
    return static_cast<int32_t>(query_points->size());
}

double center_distance(const cv::Point3f& a, const cv::Point3f& b) {
    const cv::Point3f d = a - b;
    return std::sqrt(static_cast<double>(d.x * d.x + d.y * d.y + d.z * d.z));
}

// A revisit produces the same constraint frame after frame. Once one closure
// is accepted, keep rejecting the same place until the query has travelled
// past the distance threshold or the frame gap has expired; duplicated edges
// only re-pull the pose graph toward an already applied correction.
bool is_duplicate_site(const BowDatabase& db,
                       int32_t frame_id,
                       const cv::Point3f& query_center,
                       const LoopKeyframe& candidate,
                       const LoopClosureParameters& parameters,
                       const LoopClosureSite** site) {
    const LoopClosureSite* hit = nullptr;
    for (std::size_t i = db.closed_sites.size(); i > 0U && hit == nullptr;
         --i) {
        const LoopClosureSite& closed =
            db.closed_sites[i - 1U];
        if (frame_id - closed.query_frame <= parameters.duplicate_frame_gap) {
            // Match by keyframe index as well as by position: monocular
            // scale drift can stretch the stored centers apart even when the
            // revisit is the same one.
            const bool near_match_frame =
                std::abs(candidate.frame_id - closed.match_frame) <=
                parameters.duplicate_match_window;
            const bool near_centers =
                center_distance(query_center, closed.query_center) <=
                    parameters.duplicate_distance &&
                center_distance(candidate.camera_center, closed.match_center) <=
                    parameters.duplicate_distance;
            if (near_match_frame || near_centers) {
                hit = &closed;
            }
        }
    }
    if (site != nullptr) {
        *site = hit;
    }
    return hit != nullptr;
}

bool recover_loop_relative_pose(const std::vector<cv::Point2f>& match_points,
                                const std::vector<cv::Point2f>& query_points,
                                const CameraIntrinsics& camera,
                                Pose* relative_pose) {
    cvlib::Matrix pix0 = points2f_to_matrix(match_points);
    cvlib::Matrix pix1 = points2f_to_matrix(query_points);
    cvlib::Matrix norm0 = points2f_to_normalized_matrix(match_points, camera);
    cvlib::Matrix norm1 = points2f_to_normalized_matrix(query_points, camera);
    cvlib::Matrix k = make_camera_matrix(camera);
    cvlib::Matrix e = cvlib::matrix_create(3, 3);
    cvlib::Matrix r = cvlib::matrix_create(3, 3);
    cvlib::Vector t = cvlib::vector_create(3);

    cvlib::ErrorCode ec =
        cvlib::calib3d::find_essential_matrix(&pix0, &pix1, &k, &e);
    if (ec == cvlib::ErrorCode::kSuccess) {
        ec = cvlib::calib3d::recover_pose_from_essential(
            &e, &norm0, &norm1, &r, &t);
    }
    const bool ok = ec == cvlib::ErrorCode::kSuccess;
    if (ok) {
        copy_cvlib_pose(&r, &t, relative_pose);
    }

    cvlib::matrix_destroy(&pix0);
    cvlib::matrix_destroy(&pix1);
    cvlib::matrix_destroy(&norm0);
    cvlib::matrix_destroy(&norm1);
    cvlib::matrix_destroy(&k);
    cvlib::matrix_destroy(&e);
    cvlib::matrix_destroy(&r);
    cvlib::vector_destroy(&t);
    return ok;
}

bool verify_loop_geometry(const std::vector<cv::Point2f>& match_points,
                          const std::vector<cv::Point2f>& query_points,
                          const CameraIntrinsics& camera,
                          const LoopClosureParameters& parameters,
                          int32_t* inliers,
                          Pose* relative_pose,
                          std::vector<cv::Point2f>* inlier_match_out,
                          std::vector<cv::Point2f>* inlier_query_out,
                          std::vector<int32_t>* inlier_indices) {
    bool ok = false;
    *inliers = 0;
    inlier_match_out->clear();
    inlier_query_out->clear();
    inlier_indices->clear();
    cvlib::Matrix pix0 = points2f_to_matrix(match_points);
    cvlib::Matrix pix1 = points2f_to_matrix(query_points);
    cvlib::Matrix f = cvlib::matrix_create(3, 3);
    std::vector<int32_t> mask(match_points.size(), 0);
    cvlib::calib3d::RansacParams ransac_params;
    ransac_params.max_iters = parameters.ransac_max_iters;
    ransac_params.inlier_thresh = parameters.inlier_threshold;
    ransac_params.min_inliers = parameters.min_inliers;
    ransac_params.seed = 0x4C4F4F50;

    const cvlib::ErrorCode f_ec = cvlib::calib3d::find_fundamental_ransac(
        &pix0, &pix1, ransac_params, &f, mask.data(), inliers);
    if (f_ec == cvlib::ErrorCode::kSuccess &&
        *inliers >= parameters.min_inliers) {
        std::vector<cv::Point2f> inlier_match_points;
        std::vector<cv::Point2f> inlier_query_points;
        inlier_match_points.reserve(static_cast<std::size_t>(*inliers));
        inlier_query_points.reserve(static_cast<std::size_t>(*inliers));
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] != 0) {
                inlier_match_points.push_back(match_points[i]);
                inlier_query_points.push_back(query_points[i]);
                inlier_indices->push_back(static_cast<int32_t>(i));
            }
        }
        ok = recover_loop_relative_pose(inlier_match_points,
                                        inlier_query_points, camera,
                                        relative_pose);
        if (ok) {
            *inlier_match_out = inlier_match_points;
            *inlier_query_out = inlier_query_points;
        }
        if (ok) {
            // A degenerate essential decomposition can hand back a
            // non-rotation R; a pose-graph edge built from it hides the
            // loop translation instead of closing it.
            const double rotation_error =
                rotation_orthonormality_error(*relative_pose);
            const double det = rotation_determinant(*relative_pose);
            if (rotation_error > parameters.max_rotation_error ||
                std::abs(det - 1.0) > parameters.max_rotation_error) {
                std::cout << "loop_rotation_rejected rot_err="
                          << rotation_error << " det=" << det << std::endl;
                ok = false;
            }
        }
    }

    cvlib::matrix_destroy(&pix0);
    cvlib::matrix_destroy(&pix1);
    cvlib::matrix_destroy(&f);
    return ok;
}

bool triangulate_pair(const Pose& pose_a, const Pose& pose_b,
                      const cv::Point2f& pixel_a, const cv::Point2f& pixel_b,
                      const CameraIntrinsics& camera, cv::Point3f* out) {
    const std::vector<cv::Point2f> pixels_a = {pixel_a};
    const std::vector<cv::Point2f> pixels_b = {pixel_b};
    cvlib::Matrix pa = pose_to_projection(pose_a);
    cvlib::Matrix pb = pose_to_projection(pose_b);
    cvlib::Matrix xa = points2f_to_normalized_matrix(pixels_a, camera);
    cvlib::Matrix xb = points2f_to_normalized_matrix(pixels_b, camera);
    cvlib::Matrix triangulated = cvlib::matrix_create(1, 3);
    const bool ok = cvlib::calib3d::triangulate_points(
        &pa, &pb, &xa, &xb, &triangulated) == cvlib::ErrorCode::kSuccess;
    if (ok) {
        *out = cv::Point3f(
            static_cast<float>(cvlib::matrix_get(&triangulated, 0, 0)),
            static_cast<float>(cvlib::matrix_get(&triangulated, 0, 1)),
            static_cast<float>(cvlib::matrix_get(&triangulated, 0, 2)));
    }
    cvlib::matrix_destroy(&pa);
    cvlib::matrix_destroy(&pb);
    cvlib::matrix_destroy(&xa);
    cvlib::matrix_destroy(&xb);
    cvlib::matrix_destroy(&triangulated);
    return ok;
}

cv::Point3f world_to_camera(const cv::Point3f& point, const Pose& pose) {
    const double x = static_cast<double>(point.x);
    const double y = static_cast<double>(point.y);
    const double z = static_cast<double>(point.z);
    double out[3];
    for (int32_t row = 0; row < 3; ++row) {
        out[row] = pose.t[row] + pose.r[row * 3 + 0] * x +
                   pose.r[row * 3 + 1] * y + pose.r[row * 3 + 2] * z;
    }
    return cv::Point3f(static_cast<float>(out[0]),
                       static_cast<float>(out[1]),
                       static_cast<float>(out[2]));
}

/*
Depth for the keypoints of one keyframe, expressed in that keyframe's own
camera frame and in that keyframe's own local scale. The neighbour keyframe
supplies the baseline, so the result carries the VO scale as it was at that
moment - comparing the two sides of a loop is what makes the relative scale
observable.
*/
std::unordered_map<int32_t, cv::Point3f> triangulate_local_points(
    const LoopKeyframe& keyframe,
    const LoopKeyframe& neighbor,
    const CameraIntrinsics& camera,
    const LoopClosureParameters& parameters,
    bool debug_geometry) {
    std::unordered_map<int32_t, cv::Point3f> local_points;
    const std::vector<std::pair<int32_t, int32_t>> matches =
        match_descriptor_indices(keyframe.descriptors, neighbor.descriptors,
                                 parameters.match_ratio);
    local_points.reserve(matches.size());
    int32_t rejected_cheirality = 0;
    int32_t rejected_parallax = 0;
    int32_t rejected_reprojection = 0;
    for (const std::pair<int32_t, int32_t>& match : matches) {
        const cv::Point2f& pixel =
            keyframe.keypoints[static_cast<std::size_t>(match.first)].pt;
        const cv::Point2f& neighbor_pixel =
            neighbor.keypoints[static_cast<std::size_t>(match.second)].pt;
        cv::Point3f world;
        if (!triangulate_pair(keyframe.pose, neighbor.pose, pixel,
                              neighbor_pixel, camera, &world)) {
            continue;
        }
        if (depth_in_pose(world, keyframe.pose) <= 0.0 ||
            depth_in_pose(world, neighbor.pose) <= 0.0) {
            ++rejected_cheirality;
        } else if (parallax_deg_for_point(world, keyframe.pose,
                                          neighbor.pose) <
                   parameters.metric_min_parallax_deg) {
            ++rejected_parallax;
        } else if (reprojection_residual(world, pixel, keyframe.pose,
                                         camera) >
                       parameters.metric_max_reprojection_error ||
                   reprojection_residual(world, neighbor_pixel,
                                         neighbor.pose, camera) >
                       parameters.metric_max_reprojection_error) {
            ++rejected_reprojection;
        } else {
            local_points[match.first] = world_to_camera(world, keyframe.pose);
        }
    }
    if (debug_geometry) {
        std::cout << "loop_local_depth frame=" << keyframe.frame_id
                  << " neighbor=" << neighbor.frame_id
                  << " matches=" << matches.size()
                  << " kept=" << local_points.size()
                  << " cheirality=" << rejected_cheirality
                  << " parallax=" << rejected_parallax
                  << " reproj=" << rejected_reprojection << std::endl;
    }
    return local_points;
}

// Closed-form similarity transform x_to = scale * R * x_from + t (Umeyama),
// delegated to cvlib's similarity_align: the same centered cross-covariance
// SVD construction with a determinant sign correction and the
// singular-value-sum / source-variance scale.
bool solve_similarity(const std::vector<cv::Point3f>& from_points,
                      const std::vector<cv::Point3f>& to_points,
                      Pose* transform, double* scale) {
    const std::size_t count = from_points.size();
    bool ok = count >= 3U && to_points.size() == count;
    if (ok) {
        cvlib::Matrix src = points3f_to_matrix(from_points);
        cvlib::Matrix dst = points3f_to_matrix(to_points);
        cvlib::Matrix rotation = cvlib::matrix_create(3, 3);
        cvlib::Vector translation = cvlib::vector_create(3);
        cvlib::float64_t fitted_scale = 0.0;
        const cvlib::ErrorCode ec = cvlib::calib3d::similarity_align(
            &src, &dst, &rotation, &translation, &fitted_scale);
        ok = ec == cvlib::ErrorCode::kSuccess &&
             std::isfinite(fitted_scale) && fitted_scale > 0.0;
        if (ok) {
            *scale = fitted_scale;
            for (int32_t row = 0; row < 3; ++row) {
                for (int32_t col = 0; col < 3; ++col) {
                    transform->r[row * 3 + col] =
                        cvlib::matrix_get(&rotation, row, col);
                }
                transform->t[row] = translation[row];
            }
        }
        cvlib::matrix_destroy(&src);
        cvlib::matrix_destroy(&dst);
        cvlib::matrix_destroy(&rotation);
        cvlib::vector_destroy(&translation);
    }
    return ok;
}

double similarity_residual(const cv::Point3f& from_point,
                           const cv::Point3f& to_point,
                           const Pose& transform, double scale) {
    const cv::Point3f mapped = world_to_camera(
        cv::Point3f(static_cast<float>(scale * from_point.x),
                    static_cast<float>(scale * from_point.y),
                    static_cast<float>(scale * from_point.z)),
        transform);
    const cv::Point3f error = to_point - mapped;
    return std::sqrt(static_cast<double>(error.dot(error)));
}

/*
RANSAC similarity fit over the loop correspondences that have local depth on
both sides. The inlier test is relative to the point distance because a
monocular map has no unit: a fixed metric threshold would accept everything
near the camera and reject everything far from it.
*/
bool estimate_metric_loop_transform(
    const std::vector<cv::Point3f>& from_points,
    const std::vector<cv::Point3f>& to_points,
    const LoopClosureParameters& parameters,
    bool debug_geometry,
    Pose* transform, double* scale, int32_t* inliers) {
    const std::size_t count = from_points.size();
    bool ok = false;
    int32_t best_seen = 0;
    *inliers = 0;
    if (count >= static_cast<std::size_t>(parameters.metric_min_inliers)) {
        cv::RNG rng(0x53494D33);
        std::vector<cv::Point3f> sample_from(3);
        std::vector<cv::Point3f> sample_to(3);
        std::vector<int32_t> best_inlier_indices;
        for (int32_t iteration = 0;
             iteration < parameters.metric_ransac_iters; ++iteration) {
            for (int32_t s = 0; s < 3; ++s) {
                const int32_t index =
                    rng.uniform(0, static_cast<int32_t>(count));
                sample_from[static_cast<std::size_t>(s)] =
                    from_points[static_cast<std::size_t>(index)];
                sample_to[static_cast<std::size_t>(s)] =
                    to_points[static_cast<std::size_t>(index)];
            }
            Pose candidate = {};
            double candidate_scale = 1.0;
            if (solve_similarity(sample_from, sample_to, &candidate,
                                 &candidate_scale)) {
                std::vector<int32_t> inlier_indices;
                for (std::size_t i = 0; i < count; ++i) {
                    const cv::Point3f& target = to_points[i];
                    const double norm = std::sqrt(
                        static_cast<double>(target.dot(target)));
                    if (similarity_residual(from_points[i], target, candidate,
                                            candidate_scale) <=
                        parameters.metric_inlier_ratio * norm) {
                        inlier_indices.push_back(static_cast<int32_t>(i));
                    }
                }
                if (inlier_indices.size() > best_inlier_indices.size()) {
                    best_inlier_indices = inlier_indices;
                }
                best_seen = std::max(best_seen,
                                     static_cast<int32_t>(
                                         inlier_indices.size()));
            }
        }
        if (best_inlier_indices.size() >=
            static_cast<std::size_t>(parameters.metric_min_inliers)) {
            std::vector<cv::Point3f> refit_from;
            std::vector<cv::Point3f> refit_to;
            refit_from.reserve(best_inlier_indices.size());
            refit_to.reserve(best_inlier_indices.size());
            for (const int32_t index : best_inlier_indices) {
                refit_from.push_back(
                    from_points[static_cast<std::size_t>(index)]);
                refit_to.push_back(to_points[static_cast<std::size_t>(index)]);
            }
            ok = solve_similarity(refit_from, refit_to, transform, scale);
            if (ok) {
                // A revisit cannot legitimately report an order-of-magnitude
                // scale ratio; such a fit is a RANSAC accident, and feeding it
                // to the graph is worse than having no measurement.
                ok = *scale <= parameters.metric_max_scale_ratio &&
                     *scale >= 1.0 / parameters.metric_max_scale_ratio;
                if (!ok && debug_geometry) {
                    std::cout << "loop_similarity_scale_rejected scale="
                              << *scale << std::endl;
                }
            }
            if (ok) {
                *inliers = static_cast<int32_t>(best_inlier_indices.size());
            }
        }
    }
    if (debug_geometry) {
        std::cout << "loop_similarity pairs=" << count
                  << " best_consensus=" << best_seen
                  << " required=" << parameters.metric_min_inliers
                  << std::endl;
    }
    return ok;
}

/*
Collects the 3D-3D correspondences of a verified closure: each loop inlier
keypoint that also has local depth on both sides contributes one pair, the
match keyframe's point in its own camera frame and the query frame's point in
its own. The similarity between the two sets is the loop's Sim(3).
*/
bool recover_loop_similarity(const BowDatabase& db,
                             const LoopKeyframe& query_keyframe,
                             const LoopKeyframe& candidate,
                             std::size_t candidate_index,
                             const std::vector<std::pair<int32_t, int32_t>>&
                                 matches,
                             const std::vector<int32_t>& inlier_indices,
                             const CameraIntrinsics& camera,
                             const LoopClosureParameters& parameters,
                             bool debug_geometry,
                             Pose* transform, double* scale,
                             int32_t* inliers) {
    bool ok = false;
    const std::size_t gap =
        static_cast<std::size_t>(parameters.metric_neighbor_gap);
    const bool query_neighbor_ok = db.keyframes.size() > gap;
    const bool candidate_neighbor_ok =
        candidate_index >= gap || candidate_index + gap < db.keyframes.size();
    if (query_neighbor_ok && candidate_neighbor_ok) {
        const LoopKeyframe& query_neighbor =
            db.keyframes[db.keyframes.size() - gap];
        const LoopKeyframe& candidate_neighbor =
            candidate_index >= gap
                ? db.keyframes[candidate_index - gap]
                : db.keyframes[candidate_index + gap];
        const std::unordered_map<int32_t, cv::Point3f> query_local =
            triangulate_local_points(query_keyframe, query_neighbor, camera,
                                     parameters, debug_geometry);
        const std::unordered_map<int32_t, cv::Point3f> candidate_local =
            triangulate_local_points(candidate, candidate_neighbor, camera,
                                     parameters, debug_geometry);
        std::vector<cv::Point3f> from_points;
        std::vector<cv::Point3f> to_points;
        for (const int32_t index : inlier_indices) {
            const std::pair<int32_t, int32_t>& match =
                matches[static_cast<std::size_t>(index)];
            const auto query_it = query_local.find(match.first);
            const auto candidate_it = candidate_local.find(match.second);
            if (query_it != query_local.end() &&
                candidate_it != candidate_local.end()) {
                from_points.push_back(candidate_it->second);
                to_points.push_back(query_it->second);
            }
        }
        ok = estimate_metric_loop_transform(from_points, to_points, parameters,
                                            debug_geometry, transform, scale,
                                            inliers);
        if (debug_geometry) {
            std::cout << "loop_metric query=" << query_keyframe.frame_id
                      << " match=" << candidate.frame_id
                      << " query_local=" << query_local.size()
                      << " match_local=" << candidate_local.size()
                      << " pairs=" << from_points.size()
                      << " inliers=" << *inliers
                      << " scale=" << (ok ? *scale : 0.0)
                      << " ok=" << ok << std::endl;
        }
    }
    return ok;
}

}  // namespace

bool load_vocabulary(const std::string& path, BowDatabase* db) {
    bool ok = false;
    if (std::filesystem::exists(path)) {
        db->vocabulary.load(path);
        db->vocabulary_loaded = !db->vocabulary.empty();
        ok = db->vocabulary_loaded;
    }
    return ok;
}

bool query_and_add_loop(const cv::Mat& image, BowDatabase* db,
                        int32_t frame_id,
                        const Pose& current_pose,
                        const CameraIntrinsics& camera,
                        const LoopClosureParameters& parameters,
                        bool debug_geometry,
                        LoopClosureEvent* closure) {
    bool closed = false;
    if (db->orb.empty()) {
        db->orb = cv::ORB::create(parameters.orb_features);
    }
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    db->orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    if (db->vocabulary_loaded && !descriptors.empty()) {
        LoopKeyframe keyframe;
        keyframe.frame_id = frame_id;
        keyframe.keypoints = std::move(keypoints);
        keyframe.descriptors = descriptors;
        keyframe.camera_center = camera_center_from_pose(current_pose);
        keyframe.pose = current_pose;
        keyframe.vo_pose = current_pose;
        const std::vector<cv::Mat> rows = descriptor_rows(descriptors);
        db->vocabulary.transform(rows, keyframe.bow);

        double best_score = 0.0;
        const LoopKeyframe* candidate = find_best_candidate(
            *db, keyframe.bow, frame_id, parameters, &best_score);
        const int32_t best_id = candidate != nullptr ? candidate->frame_id
                                                     : -1;

        const LoopClosureSite* duplicate = nullptr;
        const bool suppressed =
            candidate != nullptr && best_score >= parameters.min_score &&
            is_duplicate_site(*db, frame_id, keyframe.camera_center,
                              *candidate, parameters, &duplicate);

        if (suppressed) {
            db->consistent_detections = 0;
            db->last_candidate_frame = -1;
            if (debug_geometry) {
                std::cout << "loop_duplicate_skipped query=" << frame_id
                          << " match=" << best_id
                          << " closed_query=" << duplicate->query_frame
                          << " closed_match=" << duplicate->match_frame
                          << std::endl;
            }
        } else if (candidate != nullptr && best_score >= parameters.min_score) {
            // A single high BoW score is not trusted on its own: the same
            // place must win over consecutive queries before geometry runs.
            const bool same_place =
                db->last_candidate_frame >= 0 &&
                std::abs(candidate->frame_id - db->last_candidate_frame) <=
                    parameters.consistency_window;
            db->consistent_detections =
                same_place ? db->consistent_detections + 1 : 1;
            db->last_candidate_frame = candidate->frame_id;
            if (db->consistent_detections >=
                parameters.min_consecutive_detections) {
                std::vector<cv::Point2f> query_points;
                std::vector<cv::Point2f> match_points;
                std::vector<std::pair<int32_t, int32_t>> match_indices;
                const int32_t matches = match_keyframe_points(
                    keyframe.descriptors, keyframe.keypoints, *candidate,
                    parameters.match_ratio, &query_points, &match_points,
                    &match_indices);
                int32_t inliers = 0;
                Pose relative_pose = {};
                std::vector<cv::Point2f> inlier_match_points;
                std::vector<cv::Point2f> inlier_query_points;
                std::vector<int32_t> inlier_indices;
                const bool verified =
                    matches >= parameters.min_matches &&
                    verify_loop_geometry(match_points, query_points, camera,
                                         parameters, &inliers,
                                         &relative_pose,
                                         &inlier_match_points,
                                         &inlier_query_points,
                                         &inlier_indices);
                // The essential matrix fixes the loop rotation but leaves the
                // translation without a unit; the 3D-3D fit supplies both.
                Pose metric_pose = {};
                double metric_scale = 1.0;
                int32_t metric_inliers = 0;
                const bool has_metric =
                    verified &&
                    recover_loop_similarity(
                        *db, keyframe, *candidate,
                        static_cast<std::size_t>(candidate -
                                                 db->keyframes.data()),
                        match_indices, inlier_indices, camera, parameters,
                        debug_geometry, &metric_pose, &metric_scale,
                        &metric_inliers);
                // Without a measured Sim(3) the edge carries only the
                // scale-free revisit assumption, which the graph can satisfy
                // by moving the gauge instead of correcting the trajectory.
                // Recording no closure lets the next frame try again.
                const bool metric_ok =
                    has_metric || parameters.metric_required == 0;
                if (verified && !metric_ok && debug_geometry) {
                    std::cout << "loop_metric_missing query=" << frame_id
                              << " match=" << best_id << std::endl;
                }
                if (verified && metric_ok) {
                    LoopClosureEvent event;
                    event.query_frame = frame_id;
                    event.match_frame = candidate->frame_id;
                    event.score = best_score;
                    event.matches = matches;
                    event.inliers = inliers;
                    event.relative_pose =
                        has_metric ? metric_pose : relative_pose;
                    event.relative_scale = has_metric ? metric_scale : 1.0;
                    event.has_metric_transform = has_metric;
                    event.metric_inliers = metric_inliers;
                    event.match_inlier_points = std::move(inlier_match_points);
                    event.query_inlier_points = std::move(inlier_query_points);
                    event.query_center = keyframe.camera_center;
                    event.match_center = candidate->camera_center;
                    db->closures.push_back(event);
                    LoopClosureSite site;
                    site.query_frame = frame_id;
                    site.match_frame = candidate->frame_id;
                    site.query_center = keyframe.camera_center;
                    site.match_center = candidate->camera_center;
                    db->closed_sites.push_back(site);
                    if (closure != nullptr) {
                        *closure = event;
                    }
                    closed = true;
                    db->consistent_detections = 0;
                    db->last_candidate_frame = -1;
                    std::cout << "loop_closure query=" << frame_id
                              << " match=" << event.match_frame
                              << " score=" << best_score
                              << " matches=" << matches
                              << " inliers=" << inliers
                              << " metric=" << event.has_metric_transform
                              << " metric_inliers=" << event.metric_inliers
                              << " scale=" << event.relative_scale
                              << std::endl;
                } else if (!verified && debug_geometry) {
                    std::cout << "loop_rejected query=" << frame_id
                              << " match=" << best_id
                              << " score=" << best_score
                              << " matches=" << matches
                              << " inliers=" << inliers << std::endl;
                }
            }
        } else {
            db->consistent_detections = 0;
            db->last_candidate_frame = -1;
        }

        std::cout << "loop_query frame=" << frame_id
                  << " best_id=" << best_id
                  << " score=" << best_score
                  << " streak=" << db->consistent_detections << std::endl;
        db->keyframes.push_back(std::move(keyframe));
    }
    return closed;
}

}  // namespace mvo
