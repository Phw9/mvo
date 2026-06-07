#include "feature.h"

#include "converter.h"
#include "feature2d/klt.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

namespace mvo {
namespace {

bool frontend_is_orb(const FeatureParameters& parameters) {
    return parameters.frontend_mode == 1;
}

bool frontend_is_superpoint(const FeatureParameters& parameters) {
    return parameters.frontend_mode == 2;
}

const char* frontend_name(const FeatureParameters& parameters) {
    if (parameters.frontend_mode == 1) {
        return "orb";
    }
    if (parameters.frontend_mode == 2) {
        return "superpoint_superglue";
    }
    return "klt";
}

cv::Ptr<cv::ORB> create_orb(const FeatureParameters& parameters) {
    return cv::ORB::create(
        parameters.orb_features,
        static_cast<float>(parameters.orb_scale_factor),
        parameters.orb_levels,
        parameters.orb_edge_threshold,
        0,
        2,
        cv::ORB::HARRIS_SCORE,
        parameters.orb_patch_size,
        parameters.orb_fast_threshold);
}

bool superpoint_unavailable(const FeatureParameters& parameters,
                            const std::string& tag) {
    // TODO: Add ONNX Runtime inference for SuperPoint keypoints/descriptors
    // and SuperGlue/LightGlue matching. Public ONNX exports exist, but this
    // repository does not yet bundle model files or an inference runtime.
    std::cout << "frontend_unavailable tag=" << tag
              << " frontend_mode=" << parameters.frontend_mode
              << " frontend=" << frontend_name(parameters)
              << " reason=superpoint_superglue_runtime_not_configured"
              << " superpoint_model=" << parameters.superpoint_model
              << " superglue_model=" << parameters.superglue_model
              << std::endl;
    return false;
}

void make_existing_mask(const cv::Mat& image,
                        const std::vector<cv::Point2f>& existing,
                        double min_distance,
                        cv::Mat* mask) {
    *mask = cv::Mat(image.size(), CV_8UC1, cv::Scalar(255));
    for (const cv::Point2f& point : existing) {
        cv::circle(*mask, point, static_cast<int32_t>(min_distance),
                   cv::Scalar(0), cv::FILLED);
    }
}

bool detect_orb_points(const cv::Mat& image,
                       const std::vector<cv::Point2f>& existing,
                       int32_t max_points,
                       const FeatureParameters& parameters,
                       std::vector<cv::Point2f>* points) {
    cv::Mat mask;
    make_existing_mask(image, existing, parameters.orb_min_distance, &mask);
    cv::Ptr<cv::ORB> orb = create_orb(parameters);
    std::vector<cv::KeyPoint> keypoints;
    orb->detect(image, keypoints, mask);
    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response > b.response;
              });

    points->clear();
    std::vector<cv::Point2f> occupied = existing;
    for (const cv::KeyPoint& keypoint : keypoints) {
        if (static_cast<int32_t>(points->size()) >= max_points) {
            break;
        }
        if (point_is_far_from_existing(keypoint.pt, occupied,
                                       parameters.orb_min_distance)) {
            points->push_back(keypoint.pt);
            occupied.push_back(keypoint.pt);
        }
    }
    return !points->empty();
}

struct OrbMatchResult {
    std::vector<cv::Point2f> tracked_prev;
    std::vector<cv::Point2f> tracked_next;
    std::vector<int32_t> tracked_indices;
    std::vector<cv::Mat> tracked_descriptors;
    int32_t prev_descriptors = 0;
    int32_t next_keypoints = 0;
    int32_t match_candidates = 0;
    int32_t geometry_inliers = 0;
};

struct OrbProjectionCandidate {
    int32_t map_index = -1;
    int32_t keypoint_index = -1;
    int32_t descriptor_distance = 0;
};

void filter_orb_matches_by_geometry(const FeatureParameters& parameters,
                                    OrbMatchResult* result) {
    std::vector<cv::Point2f> finite_prev;
    std::vector<cv::Point2f> finite_next;
    std::vector<int32_t> finite_indices;
    std::vector<cv::Mat> finite_descriptors;
    const int32_t finite_count = std::min(
        static_cast<int32_t>(result->tracked_prev.size()),
        static_cast<int32_t>(result->tracked_next.size()));
    finite_prev.reserve(result->tracked_prev.size());
    finite_next.reserve(result->tracked_next.size());
    finite_indices.reserve(result->tracked_indices.size());
    finite_descriptors.reserve(result->tracked_descriptors.size());
    for (int32_t i = 0; i < finite_count; ++i) {
        const cv::Point2f& prev =
            result->tracked_prev[static_cast<std::size_t>(i)];
        const cv::Point2f& next =
            result->tracked_next[static_cast<std::size_t>(i)];
        if (std::isfinite(prev.x) && std::isfinite(prev.y) &&
            std::isfinite(next.x) && std::isfinite(next.y)) {
            finite_prev.push_back(prev);
            finite_next.push_back(next);
            if (i < static_cast<int32_t>(result->tracked_indices.size())) {
                finite_indices.push_back(
                    result->tracked_indices[static_cast<std::size_t>(i)]);
            }
            if (i < static_cast<int32_t>(
                        result->tracked_descriptors.size())) {
                finite_descriptors.push_back(
                    result->tracked_descriptors[static_cast<std::size_t>(i)]);
            }
        }
    }
    if (finite_next.size() != result->tracked_next.size()) {
        result->tracked_prev = finite_prev;
        result->tracked_next = finite_next;
        result->tracked_indices = finite_indices;
        result->tracked_descriptors = finite_descriptors;
    }
    result->geometry_inliers = static_cast<int32_t>(
        result->tracked_next.size());
    if (static_cast<int32_t>(result->tracked_next.size()) <
        parameters.min_init_tracks) {
        return;
    }

    std::vector<uchar> mask;
    cv::Mat fundamental;
    try {
        fundamental = cv::findFundamentalMat(
            result->tracked_prev, result->tracked_next, cv::FM_RANSAC,
            parameters.orb_ransac_threshold, 0.99, mask);
    } catch (const cv::Exception&) {
        return;
    }
    if (fundamental.empty()) {
        return;
    }

    std::vector<cv::Point2f> filtered_prev;
    std::vector<cv::Point2f> filtered_next;
    std::vector<int32_t> filtered_indices;
    std::vector<cv::Mat> filtered_descriptors;
    filtered_prev.reserve(result->tracked_prev.size());
    filtered_next.reserve(result->tracked_next.size());
    filtered_indices.reserve(result->tracked_indices.size());
    filtered_descriptors.reserve(result->tracked_descriptors.size());
    const int32_t filter_count = std::min(
        static_cast<int32_t>(mask.size()),
        static_cast<int32_t>(result->tracked_next.size()));
    for (int32_t i = 0; i < filter_count; ++i) {
        if (mask[static_cast<std::size_t>(i)] != 0) {
            filtered_prev.push_back(result->tracked_prev[
                static_cast<std::size_t>(i)]);
            filtered_next.push_back(result->tracked_next[
                static_cast<std::size_t>(i)]);
            filtered_indices.push_back(result->tracked_indices[
                static_cast<std::size_t>(i)]);
            if (static_cast<int32_t>(result->tracked_descriptors.size()) >
                i) {
                filtered_descriptors.push_back(
                    result->tracked_descriptors[static_cast<std::size_t>(i)]);
            }
        }
    }

    result->geometry_inliers = static_cast<int32_t>(filtered_next.size());
    if (result->geometry_inliers >= parameters.min_init_tracks) {
        result->tracked_prev = filtered_prev;
        result->tracked_next = filtered_next;
        result->tracked_indices = filtered_indices;
        result->tracked_descriptors = filtered_descriptors;
    }
}

void run_orb_match_pass(const cv::Mat& prev_image,
                        const cv::Mat& image,
                        const std::vector<cv::Point2f>& prev_points,
                        const FeatureParameters& parameters,
                        bool anchor_prev_points,
                        OrbMatchResult* result) {
    result->tracked_prev.clear();
    result->tracked_next.clear();
    result->tracked_indices.clear();
    result->tracked_descriptors.clear();
    result->prev_descriptors = 0;
    result->next_keypoints = 0;
    result->match_candidates = 0;
    result->geometry_inliers = 0;

    if (prev_points.empty()) {
        return;
    }

    cv::Ptr<cv::ORB> orb = create_orb(parameters);
    cv::Mat prev_descriptors;
    cv::Mat next_descriptors;
    std::vector<cv::KeyPoint> prev_keypoints;
    std::vector<cv::KeyPoint> next_keypoints;
    if (anchor_prev_points) {
        prev_keypoints.reserve(prev_points.size());
        for (int32_t i = 0; i < static_cast<int32_t>(prev_points.size()); ++i) {
            cv::KeyPoint keypoint(
                prev_points[static_cast<std::size_t>(i)],
                static_cast<float>(parameters.orb_patch_size));
            keypoint.class_id = i;
            prev_keypoints.push_back(keypoint);
        }
        orb->compute(prev_image, prev_keypoints, prev_descriptors);
    } else {
        orb->detectAndCompute(prev_image, cv::noArray(), prev_keypoints,
                              prev_descriptors);
    }
    orb->detectAndCompute(image, cv::noArray(), next_keypoints,
                          next_descriptors);
    result->prev_descriptors = prev_descriptors.rows;
    result->next_keypoints = static_cast<int32_t>(next_keypoints.size());

    if (prev_descriptors.empty() || next_descriptors.empty()) {
        return;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> forward_matches;
    std::vector<std::vector<cv::DMatch>> reverse_matches;
    matcher.knnMatch(prev_descriptors, next_descriptors, forward_matches,
                     std::min(2, next_descriptors.rows));
    matcher.knnMatch(next_descriptors, prev_descriptors, reverse_matches,
                     std::min(2, prev_descriptors.rows));

    std::vector<int32_t> reverse_best(
        static_cast<std::size_t>(next_descriptors.rows), -1);
    for (const std::vector<cv::DMatch>& matches : reverse_matches) {
        if (matches.empty()) {
            continue;
        }
        const cv::DMatch& best = matches[0];
        const bool ratio_ok =
            matches.size() < 2 ||
            best.distance <= static_cast<float>(parameters.orb_match_ratio) *
                                 matches[1].distance;
        if (ratio_ok &&
            best.distance <=
                static_cast<float>(parameters.orb_max_match_distance)) {
            reverse_best[static_cast<std::size_t>(best.queryIdx)] =
                best.trainIdx;
        }
    }

    std::set<int32_t> used_prev_indices;
    std::set<int32_t> used_next_indices;
    const double max_distance_sq =
        parameters.orb_min_distance * parameters.orb_min_distance;
    for (const std::vector<cv::DMatch>& matches : forward_matches) {
        if (matches.empty()) {
            continue;
        }
        const cv::DMatch& best = matches[0];
        const bool ratio_ok =
            matches.size() < 2 ||
            best.distance <= static_cast<float>(parameters.orb_match_ratio) *
                                 matches[1].distance;
        if (!ratio_ok ||
            best.distance >
                static_cast<float>(parameters.orb_max_match_distance)) {
            continue;
        }
        if (best.trainIdx < 0 ||
            best.trainIdx >= static_cast<int32_t>(reverse_best.size())) {
            continue;
        }
        if (reverse_best[static_cast<std::size_t>(best.trainIdx)] !=
            best.queryIdx) {
            continue;
        }
        if (best.queryIdx < 0 || best.queryIdx >= prev_descriptors.rows ||
            best.queryIdx >= static_cast<int32_t>(prev_keypoints.size()) ||
            best.trainIdx < 0 || best.trainIdx >= next_descriptors.rows ||
            best.trainIdx >= static_cast<int32_t>(next_keypoints.size())) {
            continue;
        }
        ++result->match_candidates;
        const cv::KeyPoint& prev_keypoint =
            prev_keypoints[static_cast<std::size_t>(best.queryIdx)];
        const cv::KeyPoint& next_keypoint =
            next_keypoints[static_cast<std::size_t>(best.trainIdx)];
        int32_t prev_index = prev_keypoint.class_id;
        if (!anchor_prev_points) {
            prev_index = -1;
            double best_distance_sq = max_distance_sq;
            for (int32_t i = 0;
                 i < static_cast<int32_t>(prev_points.size()); ++i) {
                const cv::Point2f& point =
                    prev_points[static_cast<std::size_t>(i)];
                const double dx = static_cast<double>(
                    prev_keypoint.pt.x - point.x);
                const double dy = static_cast<double>(
                    prev_keypoint.pt.y - point.y);
                const double distance_sq = dx * dx + dy * dy;
                if (distance_sq <= best_distance_sq) {
                    best_distance_sq = distance_sq;
                    prev_index = i;
                }
            }
        }
        if (prev_index < 0 ||
            prev_index >= static_cast<int32_t>(prev_points.size()) ||
            used_prev_indices.count(prev_index) != 0 ||
            used_next_indices.count(best.trainIdx) != 0) {
            continue;
        }
        const cv::Point2f& original_prev =
            prev_points[static_cast<std::size_t>(prev_index)];
        const double dx = static_cast<double>(prev_keypoint.pt.x -
                                             original_prev.x);
        const double dy = static_cast<double>(prev_keypoint.pt.y -
                                             original_prev.y);
        if (dx * dx + dy * dy > max_distance_sq) {
            continue;
        }
        used_prev_indices.insert(prev_index);
        used_next_indices.insert(best.trainIdx);
        result->tracked_prev.push_back(anchor_prev_points ? original_prev
                                                          : prev_keypoint.pt);
        result->tracked_next.push_back(next_keypoint.pt);
        result->tracked_indices.push_back(prev_index);
        result->tracked_descriptors.push_back(
            next_descriptors.row(best.trainIdx).clone());
    }
    filter_orb_matches_by_geometry(parameters, result);
}

bool match_orb_descriptors_to_current(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& prev_points,
    const std::vector<MapPoint>& map_points,
    const Pose& predicted_pose,
    const CameraIntrinsics& camera,
    const FeatureParameters& parameters,
    bool use_projection_gate,
    OrbMatchResult* result) {
    result->tracked_prev.clear();
    result->tracked_next.clear();
    result->tracked_indices.clear();
    result->tracked_descriptors.clear();
    result->prev_descriptors = 0;
    result->next_keypoints = 0;
    result->match_candidates = 0;
    result->geometry_inliers = 0;

    cv::Mat map_descriptors;
    std::vector<int32_t> descriptor_indices;
    for (int32_t i = 0; i < static_cast<int32_t>(map_points.size()); ++i) {
        if (i >= static_cast<int32_t>(prev_points.size())) {
            break;
        }
        const cv::Mat& descriptor =
            map_points[static_cast<std::size_t>(i)].descriptor;
        if (!descriptor.empty()) {
            map_descriptors.push_back(descriptor.reshape(1, 1));
            descriptor_indices.push_back(i);
        }
    }
    result->prev_descriptors = map_descriptors.rows;
    if (map_descriptors.empty()) {
        return false;
    }

    cv::Ptr<cv::ORB> orb = create_orb(parameters);
    std::vector<cv::KeyPoint> next_keypoints;
    cv::Mat next_descriptors;
    orb->detectAndCompute(image, cv::noArray(), next_keypoints,
                          next_descriptors);
    result->next_keypoints = static_cast<int32_t>(next_keypoints.size());
    if (next_descriptors.empty()) {
        return false;
    }

    if (use_projection_gate) {
        std::vector<OrbProjectionCandidate> candidates;
        const double projection_radius_sq =
            parameters.orb_projection_radius *
            parameters.orb_projection_radius;
        for (int32_t descriptor_row = 0; descriptor_row < map_descriptors.rows;
             ++descriptor_row) {
            const int32_t map_index =
                descriptor_indices[static_cast<std::size_t>(descriptor_row)];
            const cv::Point3f& map_point =
                map_points[static_cast<std::size_t>(map_index)].position;
            if (depth_in_pose(map_point, predicted_pose) <= 1.0e-6) {
                continue;
            }
            const double x_cam =
                predicted_pose.r[0] * static_cast<double>(map_point.x) +
                predicted_pose.r[1] * static_cast<double>(map_point.y) +
                predicted_pose.r[2] * static_cast<double>(map_point.z) +
                predicted_pose.t[0];
            const double y_cam =
                predicted_pose.r[3] * static_cast<double>(map_point.x) +
                predicted_pose.r[4] * static_cast<double>(map_point.y) +
                predicted_pose.r[5] * static_cast<double>(map_point.z) +
                predicted_pose.t[1];
            const double z_cam =
                predicted_pose.r[6] * static_cast<double>(map_point.x) +
                predicted_pose.r[7] * static_cast<double>(map_point.y) +
                predicted_pose.r[8] * static_cast<double>(map_point.z) +
                predicted_pose.t[2];
            if (z_cam <= 1.0e-6) {
                continue;
            }
            const cv::Point2f projected(
                static_cast<float>(camera.fx * x_cam / z_cam + camera.cx),
                static_cast<float>(camera.fy * y_cam / z_cam + camera.cy));
            if (projected.x < 0.0F ||
                projected.x >= static_cast<float>(image.cols) ||
                projected.y < 0.0F ||
                projected.y >= static_cast<float>(image.rows)) {
                continue;
            }

            int32_t best_keypoint = -1;
            int32_t best_distance = std::numeric_limits<int32_t>::max();
            int32_t second_distance = std::numeric_limits<int32_t>::max();
            const int32_t searchable_keypoints = std::min(
                static_cast<int32_t>(next_keypoints.size()),
                next_descriptors.rows);
            for (int32_t keypoint_index = 0;
                 keypoint_index < searchable_keypoints; ++keypoint_index) {
                const cv::Point2f& keypoint =
                    next_keypoints[static_cast<std::size_t>(keypoint_index)]
                        .pt;
                const double dx =
                    static_cast<double>(keypoint.x - projected.x);
                const double dy =
                    static_cast<double>(keypoint.y - projected.y);
                if (dx * dx + dy * dy > projection_radius_sq) {
                    continue;
                }
                const int32_t distance = static_cast<int32_t>(cv::norm(
                    map_descriptors.row(descriptor_row),
                    next_descriptors.row(keypoint_index), cv::NORM_HAMMING));
                if (distance < best_distance) {
                    second_distance = best_distance;
                    best_distance = distance;
                    best_keypoint = keypoint_index;
                } else if (distance < second_distance) {
                    second_distance = distance;
                }
            }
            const bool ratio_ok =
                second_distance == std::numeric_limits<int32_t>::max() ||
                static_cast<double>(best_distance) <=
                    parameters.orb_match_ratio *
                        static_cast<double>(second_distance);
            if (best_keypoint >= 0 && ratio_ok &&
                best_distance <=
                    static_cast<int32_t>(parameters.orb_max_match_distance)) {
                candidates.push_back(
                    {map_index, best_keypoint, best_distance});
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const OrbProjectionCandidate& lhs,
                     const OrbProjectionCandidate& rhs) {
                      return lhs.descriptor_distance <
                             rhs.descriptor_distance;
                  });
        std::set<int32_t> used_map_indices;
        std::set<int32_t> used_next_indices;
        for (const OrbProjectionCandidate& candidate : candidates) {
            if (used_map_indices.count(candidate.map_index) != 0 ||
                used_next_indices.count(candidate.keypoint_index) != 0 ||
                candidate.keypoint_index < 0 ||
                candidate.keypoint_index >= next_descriptors.rows ||
                candidate.keypoint_index >=
                    static_cast<int32_t>(next_keypoints.size())) {
                continue;
            }
            used_map_indices.insert(candidate.map_index);
            used_next_indices.insert(candidate.keypoint_index);
            ++result->match_candidates;
            result->tracked_prev.push_back(
                prev_points[static_cast<std::size_t>(candidate.map_index)]);
            result->tracked_next.push_back(
                next_keypoints[static_cast<std::size_t>(
                                   candidate.keypoint_index)]
                    .pt);
            result->tracked_indices.push_back(candidate.map_index);
            result->tracked_descriptors.push_back(
                next_descriptors.row(candidate.keypoint_index).clone());
        }
        filter_orb_matches_by_geometry(parameters, result);
        return static_cast<int32_t>(result->tracked_next.size()) >=
               parameters.min_init_tracks;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> forward_matches;
    std::vector<std::vector<cv::DMatch>> reverse_matches;
    matcher.knnMatch(map_descriptors, next_descriptors, forward_matches,
                     std::min(2, next_descriptors.rows));
    matcher.knnMatch(next_descriptors, map_descriptors, reverse_matches,
                     std::min(2, map_descriptors.rows));

    std::vector<int32_t> reverse_best(
        static_cast<std::size_t>(next_descriptors.rows), -1);
    for (const std::vector<cv::DMatch>& matches : reverse_matches) {
        if (matches.empty()) {
            continue;
        }
        const cv::DMatch& best = matches[0];
        const bool ratio_ok =
            matches.size() < 2 ||
            best.distance <= static_cast<float>(parameters.orb_match_ratio) *
                                 matches[1].distance;
        if (ratio_ok &&
            best.distance <=
                static_cast<float>(parameters.orb_max_match_distance)) {
            reverse_best[static_cast<std::size_t>(best.queryIdx)] =
                best.trainIdx;
        }
    }

    std::set<int32_t> used_map_indices;
    std::set<int32_t> used_next_indices;
    const double projection_radius_sq =
        parameters.orb_projection_radius * parameters.orb_projection_radius;
    for (const std::vector<cv::DMatch>& matches : forward_matches) {
        if (matches.empty()) {
            continue;
        }
        const cv::DMatch& best = matches[0];
        const bool ratio_ok =
            matches.size() < 2 ||
            best.distance <= static_cast<float>(parameters.orb_match_ratio) *
                                 matches[1].distance;
        if (!ratio_ok ||
            best.distance >
                static_cast<float>(parameters.orb_max_match_distance) ||
            reverse_best[static_cast<std::size_t>(best.trainIdx)] !=
                best.queryIdx) {
            continue;
        }
        const int32_t map_index =
            descriptor_indices[static_cast<std::size_t>(best.queryIdx)];
        const cv::Point2f& matched_point =
            next_keypoints[static_cast<std::size_t>(best.trainIdx)].pt;
        if (use_projection_gate) {
            const cv::Point3f& map_point =
                map_points[static_cast<std::size_t>(map_index)].position;
            if (depth_in_pose(map_point, predicted_pose) <= 1.0e-6) {
                continue;
            }
            const double x_cam =
                predicted_pose.r[0] * static_cast<double>(map_point.x) +
                predicted_pose.r[1] * static_cast<double>(map_point.y) +
                predicted_pose.r[2] * static_cast<double>(map_point.z) +
                predicted_pose.t[0];
            const double y_cam =
                predicted_pose.r[3] * static_cast<double>(map_point.x) +
                predicted_pose.r[4] * static_cast<double>(map_point.y) +
                predicted_pose.r[5] * static_cast<double>(map_point.z) +
                predicted_pose.t[1];
            const double z_cam =
                predicted_pose.r[6] * static_cast<double>(map_point.x) +
                predicted_pose.r[7] * static_cast<double>(map_point.y) +
                predicted_pose.r[8] * static_cast<double>(map_point.z) +
                predicted_pose.t[2];
            const cv::Point2f projected(
                static_cast<float>(camera.fx * x_cam / z_cam + camera.cx),
                static_cast<float>(camera.fy * y_cam / z_cam + camera.cy));
            if (projected.x < 0.0F ||
                projected.x >= static_cast<float>(image.cols) ||
                projected.y < 0.0F ||
                projected.y >= static_cast<float>(image.rows)) {
                continue;
            }
            const double dx =
                static_cast<double>(matched_point.x - projected.x);
            const double dy =
                static_cast<double>(matched_point.y - projected.y);
            if (dx * dx + dy * dy > projection_radius_sq) {
                continue;
            }
        }
        if (used_map_indices.count(map_index) != 0 ||
            used_next_indices.count(best.trainIdx) != 0) {
            continue;
        }
        ++result->match_candidates;
        used_map_indices.insert(map_index);
        used_next_indices.insert(best.trainIdx);
        result->tracked_prev.push_back(
            prev_points[static_cast<std::size_t>(map_index)]);
        result->tracked_next.push_back(
            matched_point);
        result->tracked_indices.push_back(map_index);
        result->tracked_descriptors.push_back(
            next_descriptors.row(best.trainIdx).clone());
    }
    filter_orb_matches_by_geometry(parameters, result);
    return static_cast<int32_t>(result->tracked_next.size()) >=
           parameters.min_init_tracks;
}

}  // namespace

bool detect_initial_points(const cv::Mat& image,
                           const FeatureParameters& parameters,
                           std::vector<cv::Point2f>* points) {
    if (frontend_is_superpoint(parameters)) {
        return superpoint_unavailable(parameters, "detect_initial");
    }
    if (frontend_is_orb(parameters)) {
        const std::vector<cv::Point2f> empty_existing;
        return detect_orb_points(image, empty_existing, parameters.max_features,
                                 parameters,
                                 points) &&
               static_cast<int32_t>(points->size()) >=
                   parameters.min_init_tracks;
    }
    bool ok = false;
    points->clear();
    cv::goodFeaturesToTrack(image, *points, parameters.max_init_tracks,
                            parameters.klt_quality,
                            parameters.klt_min_distance);
    ok = static_cast<int32_t>(points->size()) >=
         parameters.min_init_tracks;
    return ok;
}

bool point_is_far_from_existing(const cv::Point2f& point,
                                const std::vector<cv::Point2f>& existing,
                                double min_distance) {
    bool far = true;
    const double min_dist_sq = min_distance * min_distance;
    for (const cv::Point2f& other : existing) {
        const double dx = static_cast<double>(point.x - other.x);
        const double dy = static_cast<double>(point.y - other.y);
        if (dx * dx + dy * dy < min_dist_sq) {
            far = false;
            break;
        }
    }
    return far;
}

bool detect_refresh_points(const cv::Mat& image,
                           const std::vector<cv::Point2f>& existing,
                           int32_t max_points,
                           const FeatureParameters& parameters,
                           std::vector<cv::Point2f>* points) {
    if (frontend_is_superpoint(parameters)) {
        return superpoint_unavailable(parameters, "detect_refresh");
    }
    if (frontend_is_orb(parameters)) {
        return detect_orb_points(image, existing, max_points, parameters,
                                 points);
    }
    bool ok = false;
    points->clear();
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(255));
    std::vector<cv::Point2f> occupied = existing;
    for (const cv::Point2f& point : existing) {
        cv::circle(mask, point,
                   static_cast<int32_t>(parameters.klt_min_distance),
                   cv::Scalar(0), cv::FILLED);
    }

    const int32_t grid_cells = parameters.refresh_grid_rows *
                               parameters.refresh_grid_cols;
    const int32_t max_per_cell = std::max(
        1, (max_points + grid_cells - 1) / grid_cells);
    for (int32_t row = 0; row < parameters.refresh_grid_rows &&
                          static_cast<int32_t>(points->size()) < max_points;
         ++row) {
        for (int32_t col = 0; col < parameters.refresh_grid_cols &&
                              static_cast<int32_t>(points->size()) <
                                  max_points;
             ++col) {
            const int32_t x0 = col * image.cols /
                               parameters.refresh_grid_cols;
            const int32_t x1 = (col + 1) * image.cols /
                               parameters.refresh_grid_cols;
            const int32_t y0 = row * image.rows /
                               parameters.refresh_grid_rows;
            const int32_t y1 = (row + 1) * image.rows /
                               parameters.refresh_grid_rows;
            const cv::Rect cell_rect(x0, y0, x1 - x0, y1 - y0);
            if (cell_rect.width > 0 && cell_rect.height > 0) {
                std::vector<cv::Point2f> cell_points;
                cv::goodFeaturesToTrack(image(cell_rect), cell_points,
                                        max_per_cell, parameters.klt_quality,
                                        parameters.klt_min_distance,
                                        mask(cell_rect));
                for (const cv::Point2f& cell_point : cell_points) {
                    const cv::Point2f point(
                        cell_point.x + static_cast<float>(x0),
                        cell_point.y + static_cast<float>(y0));
                    if (static_cast<int32_t>(points->size()) < max_points &&
                        point_is_far_from_existing(point, occupied,
                                                   parameters.klt_min_distance)) {
                        points->push_back(point);
                        occupied.push_back(point);
                        cv::circle(mask, point,
                                   static_cast<int32_t>(
                                       parameters.klt_min_distance),
                                   cv::Scalar(0), cv::FILLED);
                    }
                }
            }
        }
    }

    if (static_cast<int32_t>(points->size()) < max_points) {
        std::vector<cv::Point2f> extra_points;
        cv::goodFeaturesToTrack(image, extra_points,
                                max_points -
                                    static_cast<int32_t>(points->size()),
                                parameters.klt_quality,
                                parameters.klt_min_distance, mask);
        for (const cv::Point2f& point : extra_points) {
            if (static_cast<int32_t>(points->size()) < max_points &&
                point_is_far_from_existing(point, occupied,
                                           parameters.klt_min_distance)) {
                points->push_back(point);
                occupied.push_back(point);
            }
        }
    }

    ok = !points->empty();
    return ok;
}

cvlib::feature2d::KltImageView make_cvlib_klt_image_view(
    const cv::Mat& image,
    std::vector<cvlib::float64_t>* pixels) {
    cvlib::feature2d::KltImageView view;
    pixels->clear();
    if (!image.empty()) {
        cv::Mat gray;
        if (image.channels() == 1) {
            gray = image;
        } else {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        }
        cv::Mat gray64;
        gray.convertTo(gray64, CV_64F);
        pixels->resize(static_cast<std::size_t>(gray64.rows * gray64.cols));
        for (int32_t row = 0; row < gray64.rows; ++row) {
            const double* src = gray64.ptr<double>(row);
            for (int32_t col = 0; col < gray64.cols; ++col) {
                (*pixels)[static_cast<std::size_t>(
                    row * gray64.cols + col)] = src[col];
            }
        }
        view.data = pixels->data();
        view.rows = gray64.rows;
        view.cols = gray64.cols;
        view.stride = gray64.cols;
    }
    return view;
}

cvlib::feature2d::KltParameters make_cvlib_klt_parameters(
    const FeatureParameters& parameters,
    int32_t window_size,
    int32_t pyramid_levels) {
    cvlib::feature2d::KltParameters klt_parameters =
        cvlib::feature2d::klt_default_parameters();
    klt_parameters.window_width = window_size;
    klt_parameters.window_height = window_size;
    klt_parameters.max_level = pyramid_levels;
    klt_parameters.max_iterations = parameters.klt_max_iterations;
    klt_parameters.epsilon = parameters.klt_epsilon;
    klt_parameters.min_eig_threshold = parameters.klt_min_eig_threshold;
    return klt_parameters;
}

cvlib::ErrorCode track_points_with_cvlib_klt(
    const cv::Mat& prev_image,
    const cv::Mat& image,
    const std::vector<cv::Point2f>& prev_points,
    const FeatureParameters& parameters,
    int32_t window_size,
    int32_t pyramid_levels,
    std::vector<cv::Point2f>* next_points,
    std::vector<uint8_t>* status) {
    cvlib::ErrorCode ec = cvlib::ErrorCode::kSuccess;
    next_points->clear();
    status->clear();
    if (!prev_points.empty()) {
        std::vector<cvlib::float64_t> prev_pixels;
        std::vector<cvlib::float64_t> image_pixels;
        const cvlib::feature2d::KltImageView prev_view =
            make_cvlib_klt_image_view(prev_image, &prev_pixels);
        const cvlib::feature2d::KltImageView image_view =
            make_cvlib_klt_image_view(image, &image_pixels);
        std::vector<cvlib::feature2d::KltPoint> input_points(
            prev_points.size());
        std::vector<cvlib::feature2d::KltPoint> output_points(
            prev_points.size());
        std::vector<cvlib::float64_t> errors(prev_points.size());
        status->resize(prev_points.size(), 0U);
        for (int32_t i = 0; i < static_cast<int32_t>(prev_points.size());
             ++i) {
            input_points[static_cast<std::size_t>(i)].x =
                static_cast<double>(prev_points[static_cast<std::size_t>(i)].x);
            input_points[static_cast<std::size_t>(i)].y =
                static_cast<double>(prev_points[static_cast<std::size_t>(i)].y);
        }
        const cvlib::feature2d::KltParameters klt_parameters =
            make_cvlib_klt_parameters(parameters, window_size,
                                      pyramid_levels);
        ec = cvlib::feature2d::klt_track(
            &prev_view, &image_view, input_points.data(),
            static_cast<int32_t>(input_points.size()), &klt_parameters,
            output_points.data(), status->data(), errors.data());
        if (ec == cvlib::ErrorCode::kSuccess) {
            next_points->resize(output_points.size());
            for (int32_t i = 0;
                 i < static_cast<int32_t>(output_points.size()); ++i) {
                (*next_points)[static_cast<std::size_t>(i)] = cv::Point2f(
                    static_cast<float>(
                        output_points[static_cast<std::size_t>(i)].x),
                    static_cast<float>(
                        output_points[static_cast<std::size_t>(i)].y));
            }
        } else {
            status->clear();
        }
    }
    return ec;
}

double adaptive_forward_backward_threshold(
    const std::vector<cv::Point2f>& prev_points,
    const std::vector<cv::Point2f>& next_points,
    const std::vector<uint8_t>& status,
    const FeatureParameters& parameters,
    const cv::Size& image_size) {
    double threshold = parameters.max_forward_backward_error;
    std::vector<double> motion_pixels;
    for (int32_t i = 0; i < static_cast<int32_t>(status.size()); ++i) {
        const bool in_bounds =
            next_points[i].x >= 0.0F &&
            next_points[i].x < static_cast<float>(image_size.width) &&
            next_points[i].y >= 0.0F &&
            next_points[i].y < static_cast<float>(image_size.height);
        if (status[i] != 0 && in_bounds) {
            const double dx = static_cast<double>(
                next_points[i].x - prev_points[i].x);
            const double dy = static_cast<double>(
                next_points[i].y - prev_points[i].y);
            motion_pixels.push_back(std::sqrt(dx * dx + dy * dy));
        }
    }
    if (!motion_pixels.empty()) {
        const size_t median_index = motion_pixels.size() / 2U;
        std::nth_element(motion_pixels.begin(),
                         motion_pixels.begin() + median_index,
                         motion_pixels.end());
        const double motion_threshold =
            motion_pixels[median_index] *
            parameters.forward_backward_motion_ratio;
        threshold = std::max(parameters.max_forward_backward_error,
                             std::min(
                                      parameters.max_adaptive_forward_backward_error,
                                      motion_threshold));
    }
    return threshold;
}

struct KltPassResult {
    std::vector<cv::Point2f> tracked_prev;
    std::vector<cv::Point2f> tracked_next;
    std::vector<int32_t> tracked_indices;
    int32_t raw_kept = 0;
    int32_t cvlib_error = 0;
    double fb_threshold = 1.0;
    int32_t window_size = 21;
    int32_t pyramid_levels = 3;
    bool used_wide_search = false;
};

void run_klt_pass(const cv::Mat& prev_image,
                  const cv::Mat& image,
                  const std::vector<cv::Point2f>& prev_points,
                  const FeatureParameters& parameters,
                  int32_t window_size,
                  int32_t pyramid_levels,
                  bool adaptive_fb,
                  KltPassResult* result) {
    result->tracked_prev.clear();
    result->tracked_next.clear();
    result->tracked_indices.clear();
    result->raw_kept = 0;
    result->cvlib_error = 0;
    result->fb_threshold = parameters.max_forward_backward_error;
    result->window_size = window_size;
    result->pyramid_levels = pyramid_levels;
    result->used_wide_search = adaptive_fb;

    std::vector<uint8_t> status;
    std::vector<uint8_t> backward_status;
    std::vector<cv::Point2f> next_points;
    std::vector<cv::Point2f> backward_points;
    if (!prev_points.empty()) {
        cvlib::ErrorCode ec = track_points_with_cvlib_klt(
            prev_image, image, prev_points, parameters, window_size,
            pyramid_levels, &next_points, &status);
        if (ec == cvlib::ErrorCode::kSuccess) {
            ec = track_points_with_cvlib_klt(
                image, prev_image, next_points, parameters, window_size,
                pyramid_levels, &backward_points, &backward_status);
        }
        result->cvlib_error = static_cast<int32_t>(ec);
        if (ec == cvlib::ErrorCode::kSuccess) {
            if (adaptive_fb) {
                result->fb_threshold = adaptive_forward_backward_threshold(
                    prev_points, next_points, status, parameters,
                    image.size());
            }
            const int32_t track_count = std::min(
                std::min(static_cast<int32_t>(status.size()),
                         static_cast<int32_t>(backward_status.size())),
                std::min(static_cast<int32_t>(next_points.size()),
                         static_cast<int32_t>(backward_points.size())));
            for (int32_t i = 0; i < track_count; ++i) {
                if (status[static_cast<std::size_t>(i)] != 0U) {
                    ++result->raw_kept;
                }
                const bool in_bounds =
                    next_points[static_cast<std::size_t>(i)].x >= 0.0F &&
                    next_points[static_cast<std::size_t>(i)].x <
                        static_cast<float>(image.cols) &&
                    next_points[static_cast<std::size_t>(i)].y >= 0.0F &&
                    next_points[static_cast<std::size_t>(i)].y <
                        static_cast<float>(image.rows);
                const double dx = static_cast<double>(
                    backward_points[static_cast<std::size_t>(i)].x -
                    prev_points[static_cast<std::size_t>(i)].x);
                const double dy = static_cast<double>(
                    backward_points[static_cast<std::size_t>(i)].y -
                    prev_points[static_cast<std::size_t>(i)].y);
                const double fb_error = std::sqrt(dx * dx + dy * dy);
                if (status[static_cast<std::size_t>(i)] != 0U &&
                    backward_status[static_cast<std::size_t>(i)] != 0U &&
                    in_bounds && fb_error <= result->fb_threshold) {
                    result->tracked_prev.push_back(
                        prev_points[static_cast<std::size_t>(i)]);
                    result->tracked_next.push_back(
                        next_points[static_cast<std::size_t>(i)]);
                    result->tracked_indices.push_back(i);
                }
            }
        }
        if (static_cast<int32_t>(result->tracked_next.size()) >
            parameters.max_init_tracks) {
            result->tracked_prev.resize(parameters.max_init_tracks);
            result->tracked_next.resize(parameters.max_init_tracks);
            result->tracked_indices.resize(parameters.max_init_tracks);
        }
    }
}

bool track_points(const cv::Mat& prev_image, const cv::Mat& image,
                  const std::vector<cv::Point2f>& prev_points,
                  const FeatureParameters& parameters,
                  std::vector<cv::Point2f>* tracked_prev,
                  std::vector<cv::Point2f>* tracked_next,
                  std::vector<int32_t>* tracked_indices,
                  bool debug_geometry,
                  const std::string& tag,
                  bool wide_search,
                  std::vector<cv::Mat>* tracked_descriptors) {
    if (frontend_is_superpoint(parameters)) {
        return superpoint_unavailable(parameters, tag);
    }
    if (frontend_is_orb(parameters)) {
        OrbMatchResult orb_result;
        const bool anchor_prev_points = tag.rfind("frame_", 0) == 0;
        try {
            run_orb_match_pass(prev_image, image, prev_points, parameters,
                               anchor_prev_points, &orb_result);
        } catch (const cv::Exception& e) {
            tracked_prev->clear();
            tracked_next->clear();
            if (tracked_indices != nullptr) {
                tracked_indices->clear();
            }
            if (tracked_descriptors != nullptr) {
                tracked_descriptors->clear();
            }
            std::cout << "orb_match_exception tag=" << tag
                      << " code=" << e.code
                      << " message=" << e.what() << std::endl;
            return false;
        }
        *tracked_prev = orb_result.tracked_prev;
        *tracked_next = orb_result.tracked_next;
        if (tracked_indices != nullptr) {
            *tracked_indices = orb_result.tracked_indices;
        }
        if (tracked_descriptors != nullptr) {
            *tracked_descriptors = orb_result.tracked_descriptors;
        }
        if (debug_geometry) {
            std::cout << "orb_debug tag=" << tag
                      << " input=" << prev_points.size()
                      << " prev_desc=" << orb_result.prev_descriptors
                      << " next_keypoints=" << orb_result.next_keypoints
                      << " candidates=" << orb_result.match_candidates
                      << " geometry_inliers="
                      << orb_result.geometry_inliers
                      << " kept=" << tracked_next->size()
                      << " ratio=" << parameters.orb_match_ratio
                      << " max_distance="
                      << parameters.orb_max_match_distance
                      << " ransac_threshold="
                      << parameters.orb_ransac_threshold
                      << " anchored=" << anchor_prev_points
                      << " wide_search=" << wide_search
                      << std::endl;
        }
        return static_cast<int32_t>(tracked_next->size()) >=
               parameters.min_init_tracks;
    }
    bool ok = false;
    KltPassResult result;
    run_klt_pass(prev_image, image, prev_points, parameters,
                 parameters.klt_init_window_size,
                 parameters.klt_init_pyramid_levels, false, &result);
    const int32_t retry_threshold = std::min(
        parameters.min_init_tracks, static_cast<int32_t>(prev_points.size()));
    if (wide_search &&
        static_cast<int32_t>(result.tracked_next.size()) < retry_threshold) {
        KltPassResult retry_result;
        run_klt_pass(prev_image, image, prev_points, parameters,
                     parameters.klt_window_size,
                     parameters.klt_pyramid_levels, true, &retry_result);
        if (retry_result.tracked_next.size() > result.tracked_next.size()) {
            result = retry_result;
        }
    }

    *tracked_prev = result.tracked_prev;
    *tracked_next = result.tracked_next;
    if (tracked_indices != nullptr) {
        *tracked_indices = result.tracked_indices;
    }
    if (debug_geometry) {
        std::cout << "klt_debug tag=" << tag
                  << " input=" << prev_points.size()
                  << " status=" << result.raw_kept
                  << " fb_kept=" << tracked_next->size()
                  << " cvlib_error=" << result.cvlib_error
                  << " fb_thresh=" << result.fb_threshold
                  << " lk_window=" << result.window_size
                  << " lk_levels=" << result.pyramid_levels
                  << " lk_retry=" << result.used_wide_search
                  << std::endl;
    }
    ok = static_cast<int32_t>(tracked_next->size()) >=
         parameters.min_init_tracks;
    return ok;
}

bool match_orb_map_points(const cv::Mat& image,
                          const std::vector<cv::Point2f>& prev_points,
                          const std::vector<MapPoint>& map_points,
                          const Pose& predicted_pose,
                          const CameraIntrinsics& camera,
                          const FeatureParameters& parameters,
                          bool debug_geometry,
                          const std::string& tag,
                          std::vector<cv::Point2f>* tracked_prev,
                          std::vector<cv::Point2f>* tracked_next,
                          std::vector<int32_t>* tracked_indices,
                          std::vector<cv::Mat>* tracked_descriptors) {
    OrbMatchResult result;
    bool used_projection_gate = true;
    double projection_radius = parameters.orb_projection_radius;
    bool ok = match_orb_descriptors_to_current(
        image, prev_points, map_points, predicted_pose, camera, parameters,
        true, &result);
    if (static_cast<int32_t>(result.tracked_next.size()) <
        parameters.min_init_tracks) {
        FeatureParameters wide_parameters = parameters;
        wide_parameters.orb_projection_radius =
            parameters.orb_projection_fallback_radius;
        OrbMatchResult wide_result;
        const bool wide_ok = match_orb_descriptors_to_current(
            image, prev_points, map_points, predicted_pose, camera,
            wide_parameters, true, &wide_result);
        if (wide_result.tracked_next.size() > result.tracked_next.size()) {
            result = wide_result;
            ok = wide_ok;
            projection_radius = wide_parameters.orb_projection_radius;
        }
    }
    if (static_cast<int32_t>(result.tracked_next.size()) <
        parameters.min_init_tracks) {
        OrbMatchResult fallback_result;
        const bool fallback_ok = match_orb_descriptors_to_current(
            image, prev_points, map_points, predicted_pose, camera,
            parameters, false, &fallback_result);
        if (fallback_result.tracked_next.size() >
            result.tracked_next.size()) {
            result = fallback_result;
            ok = fallback_ok;
            used_projection_gate = false;
        }
    }
    *tracked_prev = result.tracked_prev;
    *tracked_next = result.tracked_next;
    if (tracked_indices != nullptr) {
        *tracked_indices = result.tracked_indices;
    }
    if (tracked_descriptors != nullptr) {
        *tracked_descriptors = result.tracked_descriptors;
    }
    if (debug_geometry) {
        std::cout << "orb_map_debug tag=" << tag
                  << " input=" << map_points.size()
                  << " map_desc=" << result.prev_descriptors
                  << " next_keypoints=" << result.next_keypoints
                  << " candidates=" << result.match_candidates
                  << " geometry_inliers=" << result.geometry_inliers
                  << " kept=" << tracked_next->size()
                  << " ratio=" << parameters.orb_match_ratio
                  << " max_distance=" << parameters.orb_max_match_distance
                  << " ransac_threshold="
                  << parameters.orb_ransac_threshold
                  << " projection_radius="
                  << projection_radius
                  << " projection_fallback_radius="
                  << parameters.orb_projection_fallback_radius
                  << " projection_gate=" << used_projection_gate
                  << std::endl;
    }
    return ok;
}

}  // namespace mvo
