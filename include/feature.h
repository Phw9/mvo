#pragma once

#include "parameters.h"
#include "types.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace mvo {

bool detect_initial_points(const cv::Mat& image,
                           const FeatureParameters& parameters,
                           std::vector<cv::Point2f>* points);
bool point_is_far_from_existing(const cv::Point2f& point,
                                const std::vector<cv::Point2f>& existing,
                                double min_distance);
bool detect_refresh_points(const cv::Mat& image,
                           const std::vector<cv::Point2f>& existing,
                           int32_t max_points,
                           const FeatureParameters& parameters,
                           std::vector<cv::Point2f>* points);
bool track_points(const cv::Mat& prev_image, const cv::Mat& image,
                  const std::vector<cv::Point2f>& prev_points,
                  const FeatureParameters& parameters,
                  std::vector<cv::Point2f>* tracked_prev,
                  std::vector<cv::Point2f>* tracked_next,
                  std::vector<int32_t>* tracked_indices,
                  bool debug_geometry,
                  const std::string& tag,
                  bool wide_search,
                  std::vector<cv::Mat>* tracked_descriptors = nullptr);
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
                          std::vector<cv::Mat>* tracked_descriptors);

}  // namespace mvo
