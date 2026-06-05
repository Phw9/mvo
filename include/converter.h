#pragma once

#include "types.h"

#include "../thirdparty/cvlib/include/types.h"

#include <opencv2/core.hpp>

#include <vector>

namespace mvo {

cvlib::Matrix make_camera_matrix(const CameraIntrinsics& camera);
cvlib::Matrix points2f_to_matrix(const std::vector<cv::Point2f>& points);
cvlib::Matrix points2f_to_normalized_matrix(
    const std::vector<cv::Point2f>& points,
    const CameraIntrinsics& camera);
cvlib::Matrix points3f_to_matrix(const std::vector<cv::Point3f>& points);
cvlib::Matrix pose_to_projection(const Pose& pose);
void copy_cvlib_pose(const cvlib::Matrix* r, const cvlib::Vector* t,
                     Pose* pose);
cvlib::Matrix pose_rotation_to_matrix(const Pose& pose);
cvlib::Vector pose_translation_to_vector(const Pose& pose);
double depth_in_pose(const cv::Point3f& point, const Pose& pose);
double reprojection_residual(const cv::Point3f& point,
                             const cv::Point2f& observation,
                             const Pose& pose,
                             const CameraIntrinsics& camera);
ReprojectionStats compute_reprojection_stats(
    const std::vector<cv::Point3f>& points3d,
    const std::vector<cv::Point2f>& observations,
    const Pose& pose,
    const CameraIntrinsics& camera);
void filter_by_reprojection(const std::vector<cv::Point3f>& map_points,
                            const std::vector<cv::Point2f>& image_points,
                            const Pose& pose,
                            const CameraIntrinsics& camera,
                            double threshold,
                            std::vector<cv::Point3f>* filtered_map_points,
                            std::vector<cv::Point2f>* filtered_image_points);
std::vector<cv::Mat> descriptor_rows(const cv::Mat& descriptors);

}  // namespace mvo
