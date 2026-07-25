#include "bundle_adjustment.h"

#include "converter.h"
#include "loop_closure.h"

#include <calib3d/bundle_adjustment.h>
#include <optimize/loss.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

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

double pose_baseline_length(const Pose& reference_pose,
                            const Pose& current_pose) {
    const cv::Point3f reference_center =
        camera_center_from_pose(reference_pose);
    const cv::Point3f current_center = camera_center_from_pose(current_pose);
    const double baseline = point_distance(reference_center, current_center);
    return baseline;
}

double scale_change_ratio(double scale) {
    double ratio = std::numeric_limits<double>::infinity();
    if (std::isfinite(scale) && scale > 0.0) {
        ratio = std::max(scale, 1.0 / scale);
    }
    return ratio;
}

void compute_anchor_similarity(const Pose& target_reference,
                               const Pose& optimized_reference,
                               const Pose& target_current,
                               const Pose& optimized_current,
                               const BundleAdjustmentParameters& parameters,
                               double* scale,
                               double q[9],
                               double c[3]) {
    const double target_baseline = pose_baseline_length(
        target_reference, target_current);
    const double optimized_baseline = pose_baseline_length(
        optimized_reference, optimized_current);

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

// Compact mono windowed-BA problem: the surviving window cameras (by original
// window-row index), the map points kept, and the mono reprojection rows
// [cam_compact_idx, point_idx, u, v]. No stereo rows and no metric scale.
struct MonoBaProblem {
    std::vector<int32_t> camera_rows;
    std::vector<int32_t> point_ids;
    std::vector<cv::Point3f> point_positions;
    std::vector<std::array<double, 4>> observation_rows;
};

/*
Gathers archived observations restricted to the window frames, keeps the
best-supported points that already have a triangulated position, gates every
observation on positive depth and a bounded seed residual, then iteratively
drops cameras whose surviving support falls below min_camera_observations.
An under-constrained free camera would make the cvlib BA reject the whole
problem (kInvalidDimension) or leave the SE(3) manifold, so the pruning must
run before the solve, exactly like the stereo builder.
*/

bool build_mono_ba_problem(const MapArchive& archive,
                           const std::unordered_map<int32_t, int32_t>&
                               frame_row,
                           const std::vector<Pose>& row_poses,
                           const CameraIntrinsics& camera,
                           int32_t min_observations,
                           int32_t min_camera_observations,
                           int32_t max_points,
                           double loss_scale,
                           MonoBaProblem* problem) {
    std::vector<int32_t> window_frames;
    window_frames.reserve(frame_row.size());
    for (const auto& entry : frame_row) {
        window_frames.push_back(entry.first);
    }
    std::sort(window_frames.begin(), window_frames.end());
    std::unordered_map<int32_t, std::vector<std::size_t>> groups;
    for (const int32_t frame_id : window_frames) {
        const auto frame_it = archive.observations_by_frame.find(frame_id);
        if (frame_it == archive.observations_by_frame.end()) {
            continue;
        }
        for (const int32_t i : frame_it->second) {
            groups[archive.observations[static_cast<std::size_t>(i)].point_id]
                .push_back(static_cast<std::size_t>(i));
        }
    }
    std::vector<std::pair<int32_t, int32_t>> eligible;
    eligible.reserve(groups.size());
    for (const auto& entry : groups) {
        const int32_t n = static_cast<int32_t>(entry.second.size());
        if (n >= min_observations &&
            archive.positions.find(entry.first) != archive.positions.end()) {
            eligible.push_back({entry.first, n});
        }
    }
    std::sort(eligible.begin(), eligible.end(),
              [](const std::pair<int32_t, int32_t>& a,
                 const std::pair<int32_t, int32_t>& b) {
                  return a.second != b.second ? a.second > b.second
                                              : a.first < b.first;
              });
    if (static_cast<int32_t>(eligible.size()) > max_points) {
        eligible.resize(static_cast<std::size_t>(max_points));
    }

    struct CandidateObservation {
        int32_t row = 0;
        double u = 0.0;
        double v = 0.0;
    };
    struct CandidatePoint {
        int32_t id = 0;
        cv::Point3f position;
        std::vector<CandidateObservation> observations;
    };
    std::vector<CandidatePoint> candidates;
    const double seed_gate = 3.0 * loss_scale;
    for (const std::pair<int32_t, int32_t>& entry : eligible) {
        CandidatePoint candidate;
        candidate.id = entry.first;
        candidate.position = archive.positions.at(entry.first);
        for (const std::size_t obs_idx : groups[entry.first]) {
            const MapObservation& obs = archive.observations[obs_idx];
            const int32_t row = frame_row.at(obs.frame_id);
            const Pose& pose = row_poses[static_cast<std::size_t>(row)];
            const double residual = reprojection_residual(
                candidate.position, obs.pixel, pose, camera);
            if (depth_in_pose(candidate.position, pose) > 1.0e-6 &&
                std::isfinite(residual) && residual <= seed_gate) {
                CandidateObservation cand_obs;
                cand_obs.row = row;
                cand_obs.u = static_cast<double>(obs.pixel.x);
                cand_obs.v = static_cast<double>(obs.pixel.y);
                candidate.observations.push_back(cand_obs);
            }
        }
        if (static_cast<int32_t>(candidate.observations.size()) >=
            min_observations) {
            candidates.push_back(std::move(candidate));
        }
    }

    std::vector<char> row_active(row_poses.size(), 1);
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<int32_t> row_support(row_poses.size(), 0);
        for (const CandidatePoint& candidate : candidates) {
            if (static_cast<int32_t>(candidate.observations.size()) <
                min_observations) {
                continue;
            }
            for (const CandidateObservation& obs : candidate.observations) {
                ++row_support[static_cast<std::size_t>(obs.row)];
            }
        }
        for (std::size_t row = 0; row < row_poses.size(); ++row) {
            if (row_active[row] != 0 &&
                row_support[row] < min_camera_observations) {
                row_active[row] = 0;
                changed = true;
                for (CandidatePoint& candidate : candidates) {
                    std::vector<CandidateObservation> kept;
                    kept.reserve(candidate.observations.size());
                    for (const CandidateObservation& obs :
                         candidate.observations) {
                        if (obs.row != static_cast<int32_t>(row)) {
                            kept.push_back(obs);
                        }
                    }
                    candidate.observations = std::move(kept);
                }
            }
        }
    }

    std::vector<int32_t> compact_index(row_poses.size(), -1);
    for (std::size_t row = 0; row < row_poses.size(); ++row) {
        if (row_active[row] != 0) {
            compact_index[row] =
                static_cast<int32_t>(problem->camera_rows.size());
            problem->camera_rows.push_back(static_cast<int32_t>(row));
        }
    }
    for (CandidatePoint& candidate : candidates) {
        if (static_cast<int32_t>(candidate.observations.size()) <
            min_observations) {
            continue;
        }
        const int32_t point_row =
            static_cast<int32_t>(problem->point_positions.size());
        problem->point_ids.push_back(candidate.id);
        problem->point_positions.push_back(candidate.position);
        for (const CandidateObservation& obs : candidate.observations) {
            const double cam_idx = static_cast<double>(
                compact_index[static_cast<std::size_t>(obs.row)]);
            problem->observation_rows.push_back(
                {cam_idx, static_cast<double>(point_row), obs.u, obs.v});
        }
    }
    return problem->camera_rows.size() >= 2U &&
           !problem->point_positions.empty() &&
           problem->observation_rows.size() >= 2U;
}

/*
Runs the cvlib BA over the compact window with the leading camera held fixed
(fixed_pose_count = 1), which anchors the SE(3) gauge to the oldest surviving
window pose so the refinement stays continuous with the already-published
trajectory. No stereo baseline and no re-anchoring: the fixed camera keeps the
solver output in the same frame the window came in. Accepts only on kSuccess
and a non-increasing cost.
*/

bool solve_mono_ba(const CameraIntrinsics& camera,
                   const std::vector<Pose>& active_poses,
                   int32_t max_iterations,
                   double loss_scale,
                   int32_t solver,
                   int32_t min_camera_observations,
                   MonoBaProblem* problem,
                   std::vector<Pose>* optimized_poses,
                   cvlib::optimize::OptimizeReport* report,
                   int32_t* status) {
    const int32_t cam_count = static_cast<int32_t>(active_poses.size());
    const int32_t point_count =
        static_cast<int32_t>(problem->point_positions.size());
    const int32_t observation_count =
        static_cast<int32_t>(problem->observation_rows.size());

    cvlib::Matrix k = make_camera_matrix(camera);
    cvlib::Matrix poses = cvlib::matrix_create(cam_count, 12);
    cvlib::Matrix points = cvlib::matrix_create(point_count, 3);
    cvlib::Matrix observations = cvlib::matrix_create(observation_count, 4);
    for (int32_t c = 0; c < cam_count; ++c) {
        set_pose_row(&poses, c, active_poses[static_cast<std::size_t>(c)]);
    }
    for (int32_t p = 0; p < point_count; ++p) {
        const cv::Point3f& point =
            problem->point_positions[static_cast<std::size_t>(p)];
        cvlib::matrix_set(&points, p, 0, static_cast<double>(point.x));
        cvlib::matrix_set(&points, p, 1, static_cast<double>(point.y));
        cvlib::matrix_set(&points, p, 2, static_cast<double>(point.z));
    }
    for (int32_t o = 0; o < observation_count; ++o) {
        for (int32_t c = 0; c < 4; ++c) {
            cvlib::matrix_set(
                &observations, o, c,
                problem->observation_rows[static_cast<std::size_t>(o)][
                    static_cast<std::size_t>(c)]);
        }
    }

    cvlib::calib3d::BAOptions options =
        cvlib::calib3d::default_ba_options();
    options.solver = solver == 1 ? cvlib::calib3d::kBASolverSchur
                                 : cvlib::calib3d::kBASolverDense;
    options.jacobian_mode = cvlib::calib3d::kBAJacobianAnalytic;
    options.min_camera_observations = min_camera_observations;
    options.fixed_pose_count = 1;
    options.lm.max_iter = max_iterations;
    options.lm.loss.type = cvlib::optimize::kLossHuber;
    options.lm.loss.scale = loss_scale;
    cvlib::calib3d::BAData data = {&poses, &points, &observations, &k,
                                   nullptr};
    const cvlib::ErrorCode ec =
        cvlib::calib3d::bundle_adjustment(&data, &options, report);
    *status = static_cast<int32_t>(ec);

    const bool accepted = ec == cvlib::ErrorCode::kSuccess &&
                          report->final_cost <= report->initial_cost;
    if (accepted) {
        optimized_poses->clear();
        optimized_poses->reserve(static_cast<std::size_t>(cam_count));
        for (int32_t c = 0; c < cam_count; ++c) {
            Pose pose;
            get_pose_row(poses, c, &pose);
            optimized_poses->push_back(pose);
        }
        for (int32_t p = 0; p < point_count; ++p) {
            problem->point_positions[static_cast<std::size_t>(p)] =
                cv::Point3f(
                    static_cast<float>(cvlib::matrix_get(&points, p, 0)),
                    static_cast<float>(cvlib::matrix_get(&points, p, 1)),
                    static_cast<float>(cvlib::matrix_get(&points, p, 2)));
        }
    }

    cvlib::matrix_destroy(&k);
    cvlib::matrix_destroy(&poses);
    cvlib::matrix_destroy(&points);
    cvlib::matrix_destroy(&observations);
    return accepted;
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
        options.solver = parameters.solver == 1
                             ? cvlib::calib3d::kBASolverSchur
                             : cvlib::calib3d::kBASolverDense;
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
        const Pose target_current_pose = *current_pose;
        double scale = 1.0;
        double q[9];
        double c[3];
        compute_anchor_similarity(reference_pose, optimized_reference,
                                  target_current_pose, optimized_current,
                                  parameters,
                                  &scale, q, c);
        Pose anchored_current =
            anchor_pose_to_reference(optimized_current, scale, q, c);
        const double target_baseline = pose_baseline_length(
            reference_pose, target_current_pose);
        const double optimized_baseline = pose_baseline_length(
            optimized_reference, optimized_current);
        const double anchored_baseline = pose_baseline_length(
            reference_pose, anchored_current);
        const double anchor_scale_change = scale_change_ratio(scale);
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
        const bool baseline_ok =
            target_baseline > parameters.min_baseline &&
            optimized_baseline > parameters.min_baseline &&
            anchor_scale_change <= parameters.max_anchor_scale_change;
        ok = ba_ec == cvlib::ErrorCode::kSuccess && cost_ok &&
             reprojection_ok && depth_ok && baseline_ok;

        if (ok) {
            *current_pose = anchored_current;
            for (int32_t i = 0; i < n_ba; ++i) {
                (*map_points)[static_cast<std::size_t>(i)] =
                    optimized_points[static_cast<std::size_t>(i)];
            }
        }

        if (debug_geometry || ba_ec != cvlib::ErrorCode::kSuccess ||
            !baseline_ok) {
            std::cout << tag << "_ba status=" << static_cast<int32_t>(ba_ec)
                      << " accepted=" << ok
                      << " points=" << n_ba
                      << " cost=" << report.initial_cost << "->"
                      << report.final_cost
                      << " p90_ref=" << before_ref.p90 << "->"
                      << after_ref.p90
                      << " p90_cur=" << before_cur.p90 << "->"
                      << after_cur.p90
                      << " baseline=" << target_baseline << "->"
                      << optimized_baseline << "->" << anchored_baseline
                      << " anchor_scale=" << scale
                      << " anchor_scale_change=" << anchor_scale_change
                      << " baseline_ok=" << baseline_ok
                      << std::endl;
        }

        cvlib::matrix_destroy(&k);
        cvlib::matrix_destroy(&poses);
        cvlib::matrix_destroy(&ba_points);
        cvlib::matrix_destroy(&observations);
    }

    return ok;
}

bool run_mono_local_ba(const CameraIntrinsics& camera,
                       const MonoLocalBaParameters& parameters,
                       int32_t frame_id,
                       bool debug_geometry,
                       MapArchive* archive,
                       BowDatabase* bow_db,
                       TrackState* state) {
    bool accepted = false;
    const int32_t total = static_cast<int32_t>(state->pose_window.size());
    const int32_t window = std::min(parameters.window, total);
    if (window < 2) {
        return accepted;
    }
    const int32_t base = total - window;
    std::unordered_map<int32_t, int32_t> frame_row;
    std::vector<Pose> row_poses;
    std::vector<int32_t> row_frame_ids;
    frame_row.reserve(static_cast<std::size_t>(window));
    row_poses.reserve(static_cast<std::size_t>(window));
    row_frame_ids.reserve(static_cast<std::size_t>(window));
    for (int32_t r = 0; r < window; ++r) {
        const FramePose& entry =
            state->pose_window[static_cast<std::size_t>(base + r)];
        frame_row[entry.frame_id] = r;
        row_poses.push_back(entry.pose);
        row_frame_ids.push_back(entry.frame_id);
    }

    MonoBaProblem problem;
    const bool built = build_mono_ba_problem(
        *archive, frame_row, row_poses, camera, parameters.min_observations,
        parameters.min_camera_observations, parameters.max_points,
        parameters.loss_scale, &problem);
    cvlib::optimize::OptimizeReport report = {};
    int32_t status = -1;
    if (built) {
        std::vector<Pose> active_poses;
        active_poses.reserve(problem.camera_rows.size());
        for (const int32_t row : problem.camera_rows) {
            active_poses.push_back(row_poses[static_cast<std::size_t>(row)]);
        }
        std::vector<Pose> optimized;
        accepted = solve_mono_ba(camera, active_poses,
                                 parameters.max_iterations,
                                 parameters.loss_scale, parameters.solver,
                                 parameters.min_camera_observations, &problem,
                                 &optimized, &report, &status);
        if (accepted) {
            std::unordered_map<int32_t, Pose> refined_poses;
            refined_poses.reserve(problem.camera_rows.size());
            for (std::size_t c = 0; c < problem.camera_rows.size(); ++c) {
                const int32_t row = problem.camera_rows[c];
                const int32_t fid =
                    row_frame_ids[static_cast<std::size_t>(row)];
                state->pose_window[static_cast<std::size_t>(base + row)]
                    .pose = optimized[c];
                refined_poses[fid] = optimized[c];
                // Tracking continues from the refined current-frame estimate.
                if (row == window - 1) {
                    state->last_pose = optimized[c];
                    state->prev_pose = state->last_pose;
                }
            }
            // The trajectory dump reads bow_db keyframe poses, so the refined
            // poses must reach the matching keyframes or the ATE never sees
            // the improvement.
            if (bow_db != nullptr) {
                for (LoopKeyframe& keyframe : bow_db->keyframes) {
                    const auto it = refined_poses.find(keyframe.frame_id);
                    if (it != refined_poses.end()) {
                        keyframe.pose = it->second;
                        keyframe.camera_center =
                            camera_center_from_pose(it->second);
                    }
                }
            }
            std::unordered_map<int32_t, cv::Point3f> refined_points;
            refined_points.reserve(problem.point_ids.size());
            for (std::size_t p = 0; p < problem.point_ids.size(); ++p) {
                refined_points[problem.point_ids[p]] =
                    problem.point_positions[p];
                archive->positions[problem.point_ids[p]] =
                    problem.point_positions[p];
            }
            for (MapPoint& point : state->map_points) {
                if (point.has_position) {
                    const auto it = refined_points.find(point.id);
                    if (it != refined_points.end()) {
                        point.position = it->second;
                    }
                }
            }
        }
    }

    std::cout << "mono_local_ba frame=" << frame_id
              << " cams=" << problem.camera_rows.size() << "/" << window
              << " points=" << problem.point_positions.size()
              << " observations=" << problem.observation_rows.size()
              << " status=" << status
              << " accepted=" << (accepted ? 1 : 0);
    if (debug_geometry) {
        std::cout << " cost=" << report.initial_cost << "->"
                  << report.final_cost
                  << " iterations=" << report.iterations
                  << " term=" << report.termination;
    }
    std::cout << std::endl;
    return accepted;
}

}  // namespace mvo
