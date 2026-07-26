#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mvo {

enum class InputType {
    kImageSequence,
    kVideo,
};

enum class SensorMode {
    kMonocular,
    kStereo,
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

struct MapPoint {
    int32_t id = -1;
    cv::Point3f position;
    cv::Point2f anchor_observation;
    Pose anchor_pose;
    int32_t created_frame = 0;
    int32_t last_seen_frame = 0;
    int32_t anchor_frame = 0;
    int32_t age = 0;
    int32_t track_length = 1;
    double last_reprojection_error = 0.0;
    // Consecutive frames whose reprojection error stayed above the cull
    // threshold; reset whenever the point reprojects cleanly. Drives the
    // lifetime map-point cull.
    int32_t consecutive_bad_frames = 0;
    bool has_position = true;
    bool has_anchor = false;
    bool candidate = true;
};

// Per-frame record of a PnP-inlier observation of a persistent map point,
// kept for global bundle adjustment after loop closures.
struct MapObservation {
    int32_t frame_id = 0;
    int32_t point_id = 0;
    cv::Point2f pixel;
};

// Right-image match recorded when a map point is stereo-triangulated. The
// known baseline turns this into a metric depth constraint for bundle
// adjustment, which mono-only reprojection rows cannot provide.
struct StereoObservation {
    int32_t frame_id = 0;
    int32_t point_id = 0;
    cv::Point2f pixel;
    float right_x = 0.0F;
};

struct MapArchive {
    std::vector<MapObservation> observations;
    std::vector<StereoObservation> stereo_observations;
    // Indices into the vectors above keyed by frame, so windowed BA can
    // gather a few frames without scanning the whole archive.
    std::unordered_map<int32_t, std::vector<int32_t>> observations_by_frame;
    std::unordered_map<int32_t, std::vector<int32_t>>
        stereo_observations_by_frame;
    std::unordered_map<int32_t, cv::Point3f> positions;
    std::unordered_map<int32_t, int32_t> last_seen;
};

// Recent per-frame world->camera pose, kept in a bounded ring buffer so the
// mono windowed local BA can refine the last few poses jointly with the map
// points their KLT tracks observe.
struct FramePose {
    int32_t frame_id = 0;
    Pose pose = {};
};

struct TrackState {
    cv::Mat prev_image;
    std::vector<cv::Point2f> prev_points;
    std::vector<MapPoint> map_points;
    std::vector<cv::Point3f> all_map_points;
    // Sliding window of recent committed poses feeding run_mono_local_ba;
    // trimmed to the configured window size after every successful frame.
    std::vector<FramePose> pose_window;
    Pose prev_pose;
    Pose last_pose;
    // Pose from two frames back, used to build a constant-velocity motion
    // prediction that seeds KLT (use_initial_flow) so fast forward motion
    // does not drop tracks early.
    Pose pose_before_last;
    bool has_pose_before_last = false;
    int32_t frames_processed = 0;
    int32_t keyframes = 0;
    int32_t loop_queries = 0;
    int32_t pnp_success = 0;
    // Reference for mono keyframe selection: the frame id, world->camera pose,
    // and positioned-map-point count captured at the last inserted keyframe.
    // Tracking updates last_pose every frame; only keyframes advance these.
    int32_t last_keyframe_frame = 0;
    int32_t last_keyframe_tracked = 0;
    Pose last_keyframe_pose = {};
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
