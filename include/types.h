#pragma once

#include "constants.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace mvo {

enum class InputType {
    kImageSequence,
    kVideo,
};

struct CameraIntrinsics {
    double fx = 7.188560000000e+02;
    double fy = 7.188560000000e+02;
    double cx = 6.071928000000e+02;
    double cy = 1.852157000000e+02;
};

struct Pose {
    double r[9];
    double t[3];
};

struct TrackState {
    cv::Mat prev_image;
    std::vector<cv::Point2f> prev_points;
    std::vector<cv::Point3f> map_points;
    std::vector<cv::Point3f> all_map_points;
    Pose prev_pose;
    Pose last_pose;
    int32_t frames_processed = 0;
    int32_t keyframes = 0;
    int32_t loop_queries = 0;
    int32_t pnp_success = 0;
};

struct InitialMap {
    std::vector<cv::Point2f> points0;
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point3f> points3d;
};

struct ReprojectionStats {
    int32_t valid = 0;
    double rmse = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double max = 0.0;
};

struct TwoViewSelection {
    std::vector<cv::Point2f> points0;
    std::vector<cv::Point2f> points1;
    int32_t fundamental_inliers = 0;
    int32_t homography_inliers = 0;
    double homography_ratio = 0.0;
    std::string selected_model = "none";
};

}  // namespace mvo
