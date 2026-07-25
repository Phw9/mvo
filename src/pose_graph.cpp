#include "pose_graph.h"

#include "converter.h"

#include <calib3d/bundle_adjustment.h>
#include <calib3d/multiview.h>
#include <calib3d/sophus.h>
#include <calib3d/transforms.h>
#include <linalg/linalg.h>
#include <optimize/loss.h>
#include <optimize/problem.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

namespace mvo {
namespace {

constexpr int32_t kPoseParamSize = cvlib::calib3d::kSe3ParamSize;
constexpr int32_t kPoseLocalSize = cvlib::calib3d::kSe3LocalSize;
constexpr int32_t kEdgeCols = 2 + kPoseParamSize;

// Scratch buffers reused across residual evaluations so the numerical
// Jacobian does not allocate per call.
struct PoseGraphContext {
    const PoseGraphData* data;
    int32_t fixed_pose_count;
    cvlib::Matrix t_from;
    cvlib::Matrix t_to;
    cvlib::Matrix t_from_inv;
    cvlib::Matrix z;
    cvlib::Matrix z_inv;
    cvlib::Matrix t_rel;
    cvlib::Matrix t_err;
    cvlib::Vector xi;
};

bool create_context_scratch(PoseGraphContext* ctx) {
    ctx->t_from = cvlib::matrix_create(4, 4);
    ctx->t_to = cvlib::matrix_create(4, 4);
    ctx->t_from_inv = cvlib::matrix_create(4, 4);
    ctx->z = cvlib::matrix_create(4, 4);
    ctx->z_inv = cvlib::matrix_create(4, 4);
    ctx->t_rel = cvlib::matrix_create(4, 4);
    ctx->t_err = cvlib::matrix_create(4, 4);
    ctx->xi = cvlib::vector_create(kPoseLocalSize);
    return ctx->t_from.data != nullptr && ctx->t_to.data != nullptr &&
           ctx->t_from_inv.data != nullptr && ctx->z.data != nullptr &&
           ctx->z_inv.data != nullptr && ctx->t_rel.data != nullptr &&
           ctx->t_err.data != nullptr && ctx->xi.data != nullptr;
}

void destroy_context_scratch(PoseGraphContext* ctx) {
    cvlib::matrix_destroy(&ctx->t_from);
    cvlib::matrix_destroy(&ctx->t_to);
    cvlib::matrix_destroy(&ctx->t_from_inv);
    cvlib::matrix_destroy(&ctx->z);
    cvlib::matrix_destroy(&ctx->z_inv);
    cvlib::matrix_destroy(&ctx->t_rel);
    cvlib::matrix_destroy(&ctx->t_err);
    cvlib::vector_destroy(&ctx->xi);
}

void params12_to_transform(const cvlib::float64_t* block, cvlib::Matrix* t) {
    t->set_identity();
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            (*t)(row, col) = block[row * 3 + col];
        }
        (*t)(row, 3) = block[9 + row];
    }
}

void transform_to_params12(const cvlib::Matrix& t, cvlib::float64_t* block) {
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            block[row * 3 + col] = t(row, col);
        }
        block[9 + row] = t(row, 3);
    }
}

void load_pose_transform(const PoseGraphContext& ctx,
                         const cvlib::float64_t* params,
                         int32_t pose_idx, cvlib::Matrix* t) {
    if (pose_idx < ctx.fixed_pose_count) {
        params12_to_transform(
            ctx.data->poses->ptr<cvlib::float64_t>() + pose_idx * kPoseParamSize, t);
    } else {
        params12_to_transform(
            params +
                (pose_idx - ctx.fixed_pose_count) * kPoseParamSize, t);
    }
}

bool pose_graph_residuals(const cvlib::float64_t* params, int32_t n_params,
                          cvlib::float64_t* residuals, int32_t n_residuals,
                          void* user_data) {
    (void)n_params;
    PoseGraphContext* ctx = static_cast<PoseGraphContext*>(user_data);
    const cvlib::Matrix* edges = ctx->data->edges;
    const cvlib::Matrix* weights = ctx->data->weights;
    bool ok = n_residuals == kPoseLocalSize * edges->rows;
    for (int32_t k = 0; ok && k < edges->rows; ++k) {
        const cvlib::float64_t* edge_row = edges->ptr<cvlib::float64_t>() + k * kEdgeCols;
        const int32_t from_idx = static_cast<int32_t>(edge_row[0]);
        const int32_t to_idx = static_cast<int32_t>(edge_row[1]);
        load_pose_transform(*ctx, params, from_idx, &ctx->t_from);
        load_pose_transform(*ctx, params, to_idx, &ctx->t_to);
        params12_to_transform(edge_row + 2, &ctx->z);
        ok = cvlib::calib3d::inv_transform(&ctx->z, &ctx->z_inv) ==
                 cvlib::ErrorCode::kSuccess &&
             cvlib::calib3d::inv_transform(&ctx->t_from,
                                           &ctx->t_from_inv) ==
                 cvlib::ErrorCode::kSuccess &&
             cvlib::linalg::matmul(&ctx->t_to, &ctx->t_from_inv,
                                   &ctx->t_rel) ==
                 cvlib::ErrorCode::kSuccess &&
             cvlib::linalg::matmul(&ctx->z_inv, &ctx->t_rel,
                                   &ctx->t_err) ==
                 cvlib::ErrorCode::kSuccess &&
             cvlib::calib3d::se3_log(&ctx->t_err, &ctx->xi) ==
                 cvlib::ErrorCode::kSuccess;
        if (ok) {
            const cvlib::float64_t translation_weight =
                weights != nullptr ? cvlib::matrix_get(weights, k, 0) : 1.0;
            const cvlib::float64_t rotation_weight =
                weights != nullptr ? cvlib::matrix_get(weights, k, 1) : 1.0;
            for (int32_t i = 0; i < 3; ++i) {
                residuals[kPoseLocalSize * k + i] =
                    translation_weight * ctx->xi[i];
                residuals[kPoseLocalSize * k + 3 + i] =
                    rotation_weight * ctx->xi[3 + i];
            }
        }
    }
    return ok;
}

void pose_graph_plus(const cvlib::float64_t* x, int32_t n_params,
                     const cvlib::float64_t* delta, int32_t n_local,
                     cvlib::float64_t* x_plus, void* user_data) {
    (void)n_local;
    (void)user_data;
    const int32_t n_free = n_params / kPoseParamSize;
    for (int32_t b = 0; b < n_free; ++b) {
        cvlib::calib3d::se3_plus_left(
            x + b * kPoseParamSize, kPoseParamSize,
            delta + b * kPoseLocalSize, kPoseLocalSize,
            x_plus + b * kPoseParamSize, nullptr);
    }
}

cvlib::ErrorCode validate_pose_graph_data(const PoseGraphData* data) {
    cvlib::ErrorCode ec = cvlib::ErrorCode::kSuccess;
    if (data == nullptr || data->poses == nullptr ||
        data->edges == nullptr || data->poses->data == nullptr ||
        data->edges->data == nullptr) {
        ec = cvlib::ErrorCode::kNullPointer;
    } else if (data->poses->cols != kPoseParamSize ||
               data->poses->rows < 2 || data->edges->cols != kEdgeCols ||
               data->edges->rows < 1 ||
               (data->weights != nullptr &&
                (data->weights->rows != data->edges->rows ||
                 data->weights->cols != 2))) {
        ec = cvlib::ErrorCode::kInvalidDimension;
    } else {
        for (int32_t k = 0; k < data->edges->rows; ++k) {
            const int32_t from_idx = static_cast<int32_t>(
                cvlib::matrix_get(data->edges, k, 0));
            const int32_t to_idx = static_cast<int32_t>(
                cvlib::matrix_get(data->edges, k, 1));
            if (from_idx < 0 || from_idx >= data->poses->rows ||
                to_idx < 0 || to_idx >= data->poses->rows ||
                from_idx == to_idx) {
                ec = cvlib::ErrorCode::kOutOfBounds;
                break;
            }
        }
    }
    return ec;
}

}  // namespace

PoseGraphOptions default_pose_graph_options() {
    PoseGraphOptions options;
    options.fixed_pose_count = 1;
    options.lm = cvlib::optimize::default_lm_options();
    return options;
}

cvlib::ErrorCode pose_graph_optimization(
    PoseGraphData* data,
    const PoseGraphOptions* options,
    cvlib::optimize::OptimizeReport* report) {
    const PoseGraphOptions default_options = default_pose_graph_options();
    const PoseGraphOptions* used_options =
        options != nullptr ? options : &default_options;
    cvlib::ErrorCode ec = validate_pose_graph_data(data);
    if (ec == cvlib::ErrorCode::kSuccess &&
        (used_options->fixed_pose_count < 0 ||
         used_options->fixed_pose_count >= data->poses->rows)) {
        ec = cvlib::ErrorCode::kInvalidArgument;
    }
    if (ec == cvlib::ErrorCode::kSuccess) {
        const int32_t n_free =
            data->poses->rows - used_options->fixed_pose_count;
        cvlib::Vector params =
            cvlib::vector_create(n_free * kPoseParamSize);
        PoseGraphContext ctx = {};
        ctx.data = data;
        ctx.fixed_pose_count = used_options->fixed_pose_count;
        if (params.data == nullptr || !create_context_scratch(&ctx)) {
            ec = cvlib::ErrorCode::kNullPointer;
        } else {
            for (int32_t i = 0; i < n_free * kPoseParamSize; ++i) {
                params[i] = data->poses->ptr<cvlib::float64_t>()[
                    used_options->fixed_pose_count * kPoseParamSize + i];
            }
            cvlib::optimize::Problem problem;
            problem.n_params = n_free * kPoseParamSize;
            problem.n_local = n_free * kPoseLocalSize;
            problem.n_residuals = kPoseLocalSize * data->edges->rows;
            problem.residual_fn = pose_graph_residuals;
            problem.jacobian_fn = nullptr;
            problem.plus_fn = pose_graph_plus;
            problem.user_data = &ctx;
            ec = cvlib::optimize::levenberg_marquardt(
                &problem, &params, &used_options->lm, report);
            if (ec == cvlib::ErrorCode::kSuccess) {
                for (int32_t i = 0; i < n_free * kPoseParamSize; ++i) {
                    data->poses->ptr<cvlib::float64_t>()[
                        used_options->fixed_pose_count * kPoseParamSize +
                        i] = params[i];
                }
            }
        }
        destroy_context_scratch(&ctx);
        cvlib::vector_destroy(&params);
    }
    return ec;
}

namespace {

void pose_to_transform(const Pose& pose, cvlib::Matrix* t) {
    t->set_identity();
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            (*t)(row, col) = pose.r[row * 3 + col];
        }
        (*t)(row, 3) = pose.t[row];
    }
}

Pose transform_to_pose(const cvlib::Matrix& t) {
    Pose pose;
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            pose.r[row * 3 + col] = t(row, col);
        }
        pose.t[row] = t(row, 3);
    }
    return pose;
}

// Re-evaluates the total cost directly from the optimized poses so the
// solver report can be cross-checked against the written-back state.
double recompute_pose_graph_cost(const PoseGraphData& data,
                                 int32_t fixed_pose_count) {
    double cost = -1.0;
    const int32_t n_free = data.poses->rows - fixed_pose_count;
    cvlib::Vector params = cvlib::vector_create(n_free * kPoseParamSize);
    std::vector<cvlib::float64_t> residuals(
        static_cast<std::size_t>(kPoseLocalSize * data.edges->rows), 0.0);
    PoseGraphContext ctx = {};
    ctx.data = &data;
    ctx.fixed_pose_count = fixed_pose_count;
    if (params.data != nullptr && create_context_scratch(&ctx)) {
        for (int32_t i = 0; i < n_free * kPoseParamSize; ++i) {
            params[i] = data.poses->ptr<cvlib::float64_t>()[
                fixed_pose_count * kPoseParamSize + i];
        }
        if (pose_graph_residuals(params.data, params.size, residuals.data(),
                                 static_cast<int32_t>(residuals.size()),
                                 &ctx)) {
            cost = 0.0;
            for (const cvlib::float64_t r : residuals) {
                cost += 0.5 * r * r;
            }
        }
    }
    destroy_context_scratch(&ctx);
    cvlib::vector_destroy(&params);
    return cost;
}

Pose param_row_to_pose(const cvlib::Matrix& poses, int32_t row) {
    Pose pose;
    for (int32_t i = 0; i < 9; ++i) {
        pose.r[i] = cvlib::matrix_get(&poses, row, i);
    }
    for (int32_t i = 0; i < 3; ++i) {
        pose.t[i] = cvlib::matrix_get(&poses, row, 9 + i);
    }
    return pose;
}

// Evaluates one edge's error transform E = Z^-1 * T_to * T_from^-1 at the
// optimized state and reports the raw translation norm, rotation angle,
// and the se3_log twist norms, to cross-check the residual the solver saw
// against the geometric center gap.
void debug_loop_edge_error(const cvlib::Matrix& poses,
                           const std::array<double, 14>& edge_row) {
    const int32_t from_row = static_cast<int32_t>(edge_row[0]);
    const int32_t to_row = static_cast<int32_t>(edge_row[1]);
    cvlib::Matrix t_from = cvlib::matrix_create(4, 4);
    cvlib::Matrix t_to = cvlib::matrix_create(4, 4);
    cvlib::Matrix t_from_inv = cvlib::matrix_create(4, 4);
    cvlib::Matrix z = cvlib::matrix_create(4, 4);
    cvlib::Matrix z_inv = cvlib::matrix_create(4, 4);
    cvlib::Matrix t_rel = cvlib::matrix_create(4, 4);
    cvlib::Matrix t_err = cvlib::matrix_create(4, 4);
    cvlib::Vector xi = cvlib::vector_create(kPoseLocalSize);
    params12_to_transform(poses.ptr<cvlib::float64_t>() + from_row * kPoseParamSize, &t_from);
    params12_to_transform(poses.ptr<cvlib::float64_t>() + to_row * kPoseParamSize, &t_to);
    params12_to_transform(edge_row.data() + 2, &z);
    const bool ok =
        cvlib::calib3d::inv_transform(&z, &z_inv) ==
            cvlib::ErrorCode::kSuccess &&
        cvlib::calib3d::inv_transform(&t_from, &t_from_inv) ==
            cvlib::ErrorCode::kSuccess &&
        cvlib::linalg::matmul(&t_to, &t_from_inv, &t_rel) ==
            cvlib::ErrorCode::kSuccess &&
        cvlib::linalg::matmul(&z_inv, &t_rel, &t_err) ==
            cvlib::ErrorCode::kSuccess &&
        cvlib::calib3d::se3_log(&t_err, &xi) == cvlib::ErrorCode::kSuccess;
    if (ok) {
        const double t_e_norm = std::sqrt(
            t_err(0, 3) * t_err(0, 3) + t_err(1, 3) * t_err(1, 3) +
            t_err(2, 3) * t_err(2, 3));
        const double t_rel_norm = std::sqrt(
            t_rel(0, 3) * t_rel(0, 3) + t_rel(1, 3) * t_rel(1, 3) +
            t_rel(2, 3) * t_rel(2, 3));
        double trace = t_err(0, 0) + t_err(1, 1) + t_err(2, 2);
        trace = std::min(3.0, std::max(-1.0, trace));
        const double theta = std::acos((trace - 1.0) / 2.0);
        const double rho_norm = std::sqrt(
            xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]);
        const double phi_norm = std::sqrt(
            xi[3] * xi[3] + xi[4] * xi[4] + xi[5] * xi[5]);
        const Pose from_pose = param_row_to_pose(poses, from_row);
        const Pose to_pose = param_row_to_pose(poses, to_row);
        const cv::Point3f c_from = camera_center_from_pose(from_pose);
        const cv::Point3f c_to = camera_center_from_pose(to_pose);
        std::cout << "pose_graph_loop_debug rows=" << from_row << "->"
                  << to_row << " t_e=" << t_e_norm
                  << " t_rel=" << t_rel_norm
                  << " theta=" << theta
                  << " rho=" << rho_norm
                  << " phi=" << phi_norm
                  << " c_from=" << c_from.x << "," << c_from.y << ","
                  << c_from.z
                  << " c_to=" << c_to.x << "," << c_to.y << "," << c_to.z
                  << " rot_err_from="
                  << rotation_orthonormality_error(from_pose)
                  << " rot_err_to="
                  << rotation_orthonormality_error(to_pose) << std::endl;
    }
    cvlib::matrix_destroy(&t_from);
    cvlib::matrix_destroy(&t_to);
    cvlib::matrix_destroy(&t_from_inv);
    cvlib::matrix_destroy(&z);
    cvlib::matrix_destroy(&z_inv);
    cvlib::matrix_destroy(&t_rel);
    cvlib::matrix_destroy(&t_err);
    cvlib::vector_destroy(&xi);
}

double center_distance(const Pose& a, const Pose& b) {
    const cv::Point3f center_a = camera_center_from_pose(a);
    const cv::Point3f center_b = camera_center_from_pose(b);
    const double dx = static_cast<double>(center_a.x - center_b.x);
    const double dy = static_cast<double>(center_a.y - center_b.y);
    const double dz = static_cast<double>(center_a.z - center_b.z);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::vector<int32_t> select_graph_nodes(
    const BowDatabase& db,
    const std::unordered_map<int32_t, int32_t>& keyframe_index,
    int32_t max_graph_poses) {
    std::set<int32_t> nodes;
    const int32_t count = static_cast<int32_t>(db.keyframes.size());
    nodes.insert(0);
    nodes.insert(count - 1);
    for (const LoopClosureEvent& event : db.closures) {
        const auto query_it = keyframe_index.find(event.query_frame);
        const auto match_it = keyframe_index.find(event.match_frame);
        if (query_it != keyframe_index.end()) {
            nodes.insert(query_it->second);
        }
        if (match_it != keyframe_index.end()) {
            nodes.insert(match_it->second);
        }
    }
    const int32_t stride = std::max(
        1, (count + max_graph_poses - 1) / max_graph_poses);
    for (int32_t i = 0; i < count; i += stride) {
        nodes.insert(i);
    }
    return std::vector<int32_t>(nodes.begin(), nodes.end());
}

// Sim(3) state x_cam = exp(log_s) * R * x_w + t. SE(3) cannot absorb the
// monocular scale drift that dominates the loop gap, so the pose graph is
// optimized over Sim(3); loop edges leave scale unconstrained and the
// sequential edges' soft scale-equality redistributes drift along the
// chain instead of bending the trajectory.
constexpr int32_t kSim3ParamSize = 13;
constexpr int32_t kSim3LocalSize = 7;

struct Sim3 {
    double r[9];
    double t[3];
    double log_s;
};

Sim3 sim3_from_pose(const Pose& pose) {
    Sim3 s;
    for (int32_t i = 0; i < 9; ++i) {
        s.r[i] = pose.r[i];
    }
    for (int32_t i = 0; i < 3; ++i) {
        s.t[i] = pose.t[i];
    }
    s.log_s = 0.0;
    return s;
}

// The SE(3) pose with the same camera center and rotation: t_se3 = t / s.
Pose sim3_to_pose(const Sim3& s) {
    Pose pose;
    const double inv_scale = std::exp(-s.log_s);
    for (int32_t i = 0; i < 9; ++i) {
        pose.r[i] = s.r[i];
    }
    for (int32_t i = 0; i < 3; ++i) {
        pose.t[i] = s.t[i] * inv_scale;
    }
    return pose;
}

Sim3 sim3_inverse(const Sim3& s) {
    Sim3 out;
    out.log_s = -s.log_s;
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            out.r[row * 3 + col] = s.r[col * 3 + row];
        }
    }
    const double inv_scale = std::exp(-s.log_s);
    for (int32_t row = 0; row < 3; ++row) {
        double value = 0.0;
        for (int32_t k = 0; k < 3; ++k) {
            value += out.r[row * 3 + k] * s.t[k];
        }
        out.t[row] = -inv_scale * value;
    }
    return out;
}

Sim3 sim3_compose(const Sim3& a, const Sim3& b) {
    Sim3 out;
    out.log_s = a.log_s + b.log_s;
    const double scale_a = std::exp(a.log_s);
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            double value = 0.0;
            for (int32_t k = 0; k < 3; ++k) {
                value += a.r[row * 3 + k] * b.r[k * 3 + col];
            }
            out.r[row * 3 + col] = value;
        }
        double t = a.t[row];
        for (int32_t k = 0; k < 3; ++k) {
            t += scale_a * a.r[row * 3 + k] * b.t[k];
        }
        out.t[row] = t;
    }
    return out;
}

struct Sim3Context {
    const cvlib::Matrix* edges;    // K x 14, R/t measurements
    const cvlib::Matrix* weights;  // K x 3 [w_t, w_r, w_s]
    // Measured log scale per edge; 0 for VO edges and for loop edges whose
    // scale could not be measured. Kept beside the edge matrix so the SE(3)
    // pose graph API keeps its K x 14 contract.
    const std::vector<double>* log_scales;
    const std::vector<Sim3>* base; // all node states (fixed reads)
    int32_t fixed_pose_count;
    cvlib::Matrix r_mat;
    cvlib::Matrix rot_out;
    cvlib::Vector vec3;
};

Sim3 sim3_from_params(const cvlib::float64_t* block) {
    Sim3 s;
    for (int32_t i = 0; i < 9; ++i) {
        s.r[i] = block[i];
    }
    for (int32_t i = 0; i < 3; ++i) {
        s.t[i] = block[9 + i];
    }
    s.log_s = block[12];
    return s;
}

void sim3_to_params(const Sim3& s, cvlib::float64_t* block) {
    for (int32_t i = 0; i < 9; ++i) {
        block[i] = s.r[i];
    }
    for (int32_t i = 0; i < 3; ++i) {
        block[9 + i] = s.t[i];
    }
    block[12] = s.log_s;
}

bool sim3_residuals(const cvlib::float64_t* params, int32_t n_params,
                    cvlib::float64_t* residuals, int32_t n_residuals,
                    void* user_data) {
    (void)n_params;
    Sim3Context* ctx = static_cast<Sim3Context*>(user_data);
    const cvlib::Matrix* edges = ctx->edges;
    bool ok = n_residuals == kSim3LocalSize * edges->rows;
    for (int32_t k = 0; ok && k < edges->rows; ++k) {
        const cvlib::float64_t* edge_row = edges->ptr<cvlib::float64_t>() + k * kEdgeCols;
        const int32_t from_idx = static_cast<int32_t>(edge_row[0]);
        const int32_t to_idx = static_cast<int32_t>(edge_row[1]);
        auto load = [&](int32_t idx) {
            return idx < ctx->fixed_pose_count
                       ? (*ctx->base)[static_cast<std::size_t>(idx)]
                       : sim3_from_params(
                             params + (idx - ctx->fixed_pose_count) *
                                          kSim3ParamSize);
        };
        const Sim3 s_from = load(from_idx);
        const Sim3 s_to = load(to_idx);
        Sim3 z;
        for (int32_t i = 0; i < 9; ++i) {
            z.r[i] = edge_row[2 + i];
        }
        for (int32_t i = 0; i < 3; ++i) {
            z.t[i] = edge_row[11 + i];
        }
        z.log_s = (*ctx->log_scales)[static_cast<std::size_t>(k)];
        const Sim3 error = sim3_compose(
            sim3_inverse(z), sim3_compose(s_to, sim3_inverse(s_from)));
        for (int32_t i = 0; i < 9; ++i) {
            ctx->r_mat.ptr<cvlib::float64_t>()[i] = error.r[i];
        }
        ok = cvlib::calib3d::so3_log(&ctx->r_mat, &ctx->vec3) ==
             cvlib::ErrorCode::kSuccess;
        if (ok) {
            const double w_t = cvlib::matrix_get(ctx->weights, k, 0);
            const double w_r = cvlib::matrix_get(ctx->weights, k, 1);
            const double w_s = cvlib::matrix_get(ctx->weights, k, 2);
            for (int32_t i = 0; i < 3; ++i) {
                residuals[kSim3LocalSize * k + i] = w_t * error.t[i];
                residuals[kSim3LocalSize * k + 3 + i] =
                    w_r * ctx->vec3[i];
            }
            residuals[kSim3LocalSize * k + 6] = w_s * error.log_s;
        }
    }
    return ok;
}

void sim3_plus(const cvlib::float64_t* x, int32_t n_params,
               const cvlib::float64_t* delta, int32_t n_local,
               cvlib::float64_t* x_plus, void* user_data) {
    (void)n_local;
    Sim3Context* ctx = static_cast<Sim3Context*>(user_data);
    const int32_t n_free = n_params / kSim3ParamSize;
    for (int32_t b = 0; b < n_free; ++b) {
        Sim3 s = sim3_from_params(x + b * kSim3ParamSize);
        const cvlib::float64_t* d = delta + b * kSim3LocalSize;
        for (int32_t i = 0; i < 3; ++i) {
            s.t[i] += d[i];
            ctx->vec3[i] = d[3 + i];
        }
        s.log_s += d[6];
        if (cvlib::calib3d::so3_exp(&ctx->vec3, &ctx->rot_out) ==
            cvlib::ErrorCode::kSuccess) {
            double rotated[9];
            for (int32_t row = 0; row < 3; ++row) {
                for (int32_t col = 0; col < 3; ++col) {
                    double value = 0.0;
                    for (int32_t k = 0; k < 3; ++k) {
                        value += ctx->rot_out.ptr<cvlib::float64_t>()[row * 3 + k] *
                                 s.r[k * 3 + col];
                    }
                    rotated[row * 3 + col] = value;
                }
            }
            for (int32_t i = 0; i < 9; ++i) {
                s.r[i] = rotated[i];
            }
        }
        sim3_to_params(s, x_plus + b * kSim3ParamSize);
    }
}

cvlib::ErrorCode optimize_sim3_pose_graph(
    std::vector<Sim3>* node_states,
    const cvlib::Matrix& edges,
    const cvlib::Matrix& weights,
    const std::vector<double>& log_scales,
    int32_t fixed_pose_count,
    const cvlib::optimize::LMOptions& lm,
    cvlib::optimize::OptimizeReport* report) {
    const int32_t node_count = static_cast<int32_t>(node_states->size());
    const int32_t n_free = node_count - fixed_pose_count;
    cvlib::ErrorCode ec = n_free >= 1 ? cvlib::ErrorCode::kSuccess
                                      : cvlib::ErrorCode::kInvalidArgument;
    if (ec == cvlib::ErrorCode::kSuccess) {
        const std::vector<Sim3> base = *node_states;
        Sim3Context ctx = {};
        ctx.edges = &edges;
        ctx.weights = &weights;
        ctx.log_scales = &log_scales;
        ctx.base = &base;
        ctx.fixed_pose_count = fixed_pose_count;
        ctx.r_mat = cvlib::matrix_create(3, 3);
        ctx.rot_out = cvlib::matrix_create(3, 3);
        ctx.vec3 = cvlib::vector_create(3);
        cvlib::Vector params =
            cvlib::vector_create(n_free * kSim3ParamSize);
        if (ctx.r_mat.data == nullptr || ctx.rot_out.data == nullptr ||
            ctx.vec3.data == nullptr || params.data == nullptr) {
            ec = cvlib::ErrorCode::kNullPointer;
        } else {
            for (int32_t b = 0; b < n_free; ++b) {
                sim3_to_params(
                    base[static_cast<std::size_t>(fixed_pose_count + b)],
                    params.data + b * kSim3ParamSize);
            }
            cvlib::optimize::Problem problem;
            problem.n_params = n_free * kSim3ParamSize;
            problem.n_local = n_free * kSim3LocalSize;
            problem.n_residuals = kSim3LocalSize * edges.rows;
            problem.residual_fn = sim3_residuals;
            problem.jacobian_fn = nullptr;
            problem.plus_fn = sim3_plus;
            problem.user_data = &ctx;
            ec = cvlib::optimize::levenberg_marquardt(&problem, &params,
                                                      &lm, report);
            if (ec == cvlib::ErrorCode::kSuccess) {
                for (int32_t b = 0; b < n_free; ++b) {
                    (*node_states)[static_cast<std::size_t>(
                        fixed_pose_count + b)] = sim3_from_params(
                        params.data + b * kSim3ParamSize);
                }
            }
        }
        cvlib::matrix_destroy(&ctx.r_mat);
        cvlib::matrix_destroy(&ctx.rot_out);
        cvlib::vector_destroy(&ctx.vec3);
        cvlib::vector_destroy(&params);
    }
    return ec;
}

bool append_sequential_edge(const Pose& from_pose, const Pose& to_pose,
                            int32_t from_row, int32_t to_row,
                            double scale_weight,
                            std::vector<std::array<double, 14>>* edge_rows,
                            std::vector<std::array<double, 3>>* weight_rows,
                            std::vector<double>* edge_log_scales) {
    cvlib::Matrix t_from = cvlib::matrix_create(4, 4);
    cvlib::Matrix t_to = cvlib::matrix_create(4, 4);
    cvlib::Matrix t_from_inv = cvlib::matrix_create(4, 4);
    cvlib::Matrix z = cvlib::matrix_create(4, 4);
    pose_to_transform(from_pose, &t_from);
    pose_to_transform(to_pose, &t_to);
    const bool ok =
        cvlib::calib3d::inv_transform(&t_from, &t_from_inv) ==
            cvlib::ErrorCode::kSuccess &&
        cvlib::linalg::matmul(&t_to, &t_from_inv, &z) ==
            cvlib::ErrorCode::kSuccess;
    if (ok) {
        std::array<double, 14> row = {};
        row[0] = static_cast<double>(from_row);
        row[1] = static_cast<double>(to_row);
        transform_to_params12(z, row.data() + 2);
        edge_rows->push_back(row);
        // The scale-equality weight decides how stiff the chain is against
        // the loop edges' free scale: too soft and the optimizer pays for the
        // loop gap with a scale jump instead of a trajectory correction.
        weight_rows->push_back({1.0, 1.0, scale_weight});
        // VO neighbours share the frontend's scale by construction.
        edge_log_scales->push_back(0.0);
    }
    cvlib::matrix_destroy(&t_from);
    cvlib::matrix_destroy(&t_to);
    cvlib::matrix_destroy(&t_from_inv);
    cvlib::matrix_destroy(&z);
    return ok;
}

void append_loop_edge(const LoopClosureEvent& event,
                      int32_t from_row, int32_t to_row,
                      const LoopClosureParameters& parameters,
                      std::vector<std::array<double, 14>>* edge_rows,
                      std::vector<std::array<double, 3>>* weight_rows,
                      std::vector<double>* edge_log_scales) {
    std::array<double, 14> row = {};
    row[0] = static_cast<double>(from_row);
    row[1] = static_cast<double>(to_row);
    for (int32_t i = 0; i < 9; ++i) {
        row[static_cast<std::size_t>(2 + i)] = event.relative_pose.r[i];
    }
    if (event.has_metric_transform) {
        // The 3D-3D fit measured the full Sim(3): translation in the map's
        // own units and the scale ratio between the two revisits. With that
        // in the edge the loop can no longer be satisfied by moving the
        // gauge, which is the whole point of measuring it.
        for (int32_t i = 0; i < 3; ++i) {
            row[static_cast<std::size_t>(11 + i)] = event.relative_pose.t[i];
        }
        edge_log_scales->push_back(std::log(event.relative_scale));
        weight_rows->push_back({parameters.pgo_loop_translation_weight,
                                parameters.pgo_loop_rotation_weight,
                                parameters.pgo_loop_scale_weight});
    } else {
        // Fallback: rotation only, translation from the zero-baseline revisit
        // assumption, scale left free because nothing measured it.
        edge_log_scales->push_back(0.0);
        weight_rows->push_back({parameters.pgo_loop_translation_weight,
                                parameters.pgo_loop_rotation_weight, 0.0});
    }
    edge_rows->push_back(row);
}

/*
Maps every keyframe onto the optimized trajectory. Node keyframes take
their optimized pose directly; keyframes between two nodes apply the
Lie-interpolated right correction C = T_old^-1 * T_new of the surrounding
nodes, so the full trajectory stays smooth.
*/
bool compute_optimized_centers(const std::vector<Pose>& base_poses,
                               const std::vector<int32_t>& nodes,
                               const cvlib::Matrix& optimized_poses,
                               std::vector<cv::Point3f>* centers,
                               std::vector<Pose>* corrected_poses) {
    const int32_t node_count = static_cast<int32_t>(nodes.size());
    const int32_t keyframe_count = static_cast<int32_t>(base_poses.size());
    bool ok = true;
    std::vector<cvlib::Matrix> corrections(
        static_cast<std::size_t>(node_count));
    cvlib::Matrix t_old = cvlib::matrix_create(4, 4);
    cvlib::Matrix t_old_inv = cvlib::matrix_create(4, 4);
    cvlib::Matrix t_new = cvlib::matrix_create(4, 4);
    cvlib::Matrix correction = cvlib::matrix_create(4, 4);
    cvlib::Matrix t_corrected = cvlib::matrix_create(4, 4);

    for (int32_t p = 0; ok && p < node_count; ++p) {
        corrections[static_cast<std::size_t>(p)] = cvlib::matrix_create(4, 4);
        pose_to_transform(
            base_poses[static_cast<std::size_t>(nodes[
                static_cast<std::size_t>(p)])], &t_old);
        params12_to_transform(
            optimized_poses.ptr<cvlib::float64_t>() + p * kPoseParamSize, &t_new);
        ok = cvlib::calib3d::inv_transform(&t_old, &t_old_inv) ==
                 cvlib::ErrorCode::kSuccess &&
             cvlib::linalg::matmul(
                 &t_old_inv, &t_new,
                 &corrections[static_cast<std::size_t>(p)]) ==
                 cvlib::ErrorCode::kSuccess;
    }

    centers->clear();
    centers->reserve(static_cast<std::size_t>(keyframe_count));
    if (corrected_poses != nullptr) {
        corrected_poses->clear();
        corrected_poses->reserve(static_cast<std::size_t>(keyframe_count));
    }
    int32_t segment = 0;
    for (int32_t i = 0; ok && i < keyframe_count; ++i) {
        while (segment + 1 < node_count && nodes[
                   static_cast<std::size_t>(segment + 1)] <= i) {
            ++segment;
        }
        pose_to_transform(base_poses[static_cast<std::size_t>(i)], &t_old);
        const cvlib::Matrix* applied =
            &corrections[static_cast<std::size_t>(segment)];
        if (i != nodes[static_cast<std::size_t>(segment)] &&
            segment + 1 < node_count) {
            const int32_t span = nodes[static_cast<std::size_t>(segment + 1)]
                - nodes[static_cast<std::size_t>(segment)];
            const double u = static_cast<double>(
                i - nodes[static_cast<std::size_t>(segment)]) /
                static_cast<double>(span);
            ok = cvlib::calib3d::se3_interp(
                     &corrections[static_cast<std::size_t>(segment)],
                     &corrections[static_cast<std::size_t>(segment + 1)],
                     u, &correction) == cvlib::ErrorCode::kSuccess;
            applied = &correction;
        }
        if (ok) {
            ok = cvlib::linalg::matmul(&t_old, applied, &t_corrected) ==
                 cvlib::ErrorCode::kSuccess;
        }
        if (ok) {
            const Pose corrected = transform_to_pose(t_corrected);
            centers->push_back(camera_center_from_pose(corrected));
            if (corrected_poses != nullptr) {
                corrected_poses->push_back(corrected);
            }
        }
    }

    for (cvlib::Matrix& node_correction : corrections) {
        cvlib::matrix_destroy(&node_correction);
    }
    cvlib::matrix_destroy(&t_old);
    cvlib::matrix_destroy(&t_old_inv);
    cvlib::matrix_destroy(&t_new);
    cvlib::matrix_destroy(&correction);
    cvlib::matrix_destroy(&t_corrected);
    return ok;
}

Sim3 sim3_from_correction(const LoopCorrection& correction) {
    Sim3 s;
    for (int32_t i = 0; i < 9; ++i) {
        s.r[i] = correction.r[i];
    }
    for (int32_t i = 0; i < 3; ++i) {
        s.t[i] = correction.t[i];
    }
    s.log_s = std::log(correction.scale);
    return s;
}

LoopCorrection correction_from_sim3(const Sim3& s) {
    LoopCorrection correction;
    for (int32_t i = 0; i < 9; ++i) {
        correction.r[i] = s.r[i];
    }
    for (int32_t i = 0; i < 3; ++i) {
        correction.t[i] = s.t[i];
    }
    correction.scale = std::exp(s.log_s);
    correction.valid = true;
    return correction;
}

// Keeps the keyframe database on the optimized trajectory so the next graph
// is built from corrected poses instead of re-deriving the same correction,
// and so the duplicate-site centers stay comparable to live camera centers.
void rebase_keyframes(BowDatabase* db,
                      const std::vector<Pose>& corrected_poses) {
    if (corrected_poses.size() == db->keyframes.size()) {
        std::unordered_map<int32_t, cv::Point3f> centers;
        centers.reserve(db->keyframes.size());
        for (std::size_t i = 0; i < db->keyframes.size(); ++i) {
            LoopKeyframe& keyframe = db->keyframes[i];
            keyframe.pose = corrected_poses[i];
            keyframe.camera_center = camera_center_from_pose(keyframe.pose);
            centers[keyframe.frame_id] = keyframe.camera_center;
        }
        for (LoopClosureSite& site : db->closed_sites) {
            const auto query_it = centers.find(site.query_frame);
            const auto match_it = centers.find(site.match_frame);
            if (query_it != centers.end()) {
                site.query_center = query_it->second;
            }
            if (match_it != centers.end()) {
                site.match_center = match_it->second;
            }
        }
    }
}

}  // namespace

Pose correct_pose(const LoopCorrection& correction, const Pose& pose) {
    Pose corrected = pose;
    if (correction.valid) {
        corrected = sim3_to_pose(sim3_compose(sim3_from_pose(pose),
                                              sim3_from_correction(correction)));
    }
    return corrected;
}

cv::Point3f correct_point(const LoopCorrection& correction,
                          const cv::Point3f& point) {
    cv::Point3f corrected = point;
    if (correction.valid) {
        // A world point moves by the inverse correction so that the camera
        // observation s*R*x_w + t it produced stays the same after the pose
        // itself has been pulled into the optimized frame.
        const Sim3 inverse = sim3_inverse(sim3_from_correction(correction));
        const double scale = std::exp(inverse.log_s);
        const double x = static_cast<double>(point.x);
        const double y = static_cast<double>(point.y);
        const double z = static_cast<double>(point.z);
        double out[3];
        for (int32_t row = 0; row < 3; ++row) {
            out[row] = inverse.t[row] +
                       scale * (inverse.r[row * 3 + 0] * x +
                                inverse.r[row * 3 + 1] * y +
                                inverse.r[row * 3 + 2] * z);
        }
        corrected = cv::Point3f(static_cast<float>(out[0]),
                                static_cast<float>(out[1]),
                                static_cast<float>(out[2]));
    }
    return corrected;
}

void apply_loop_correction(const LoopCorrection& correction,
                           TrackState* state,
                           MapArchive* archive) {
    if (correction.valid) {
        state->last_pose = correct_pose(correction, state->last_pose);
        state->prev_pose = correct_pose(correction, state->prev_pose);
        for (MapPoint& point : state->map_points) {
            if (point.has_position) {
                point.position = correct_point(correction, point.position);
            }
            if (point.has_anchor) {
                point.anchor_pose = correct_pose(correction, point.anchor_pose);
            }
            if (archive != nullptr && point.id >= 0 && point.has_position) {
                const auto it = archive->positions.find(point.id);
                if (it != archive->positions.end()) {
                    it->second = point.position;
                }
            }
        }
    }
}

bool run_pose_graph_optimization(BowDatabase* db,
                                 const LoopClosureParameters& parameters,
                                 int32_t frame_id,
                                 bool force,
                                 bool debug_geometry,
                                 std::vector<cv::Point3f>* optimized_centers,
                                 std::vector<Pose>* corrected_poses,
                                 LoopCorrection* correction) {
    bool accepted = false;
    optimized_centers->clear();
    if (correction != nullptr) {
        *correction = LoopCorrection();
    }
    const int32_t keyframe_count = static_cast<int32_t>(db->keyframes.size());
    const int32_t pending_count =
        static_cast<int32_t>(db->closures.size()) -
        db->optimized_closure_count;
    const bool pending = pending_count > 0;
    // Hysteresis: while the revisit episode keeps producing verified
    // closures the constraints only accumulate; optimize once the episode
    // has gone quiet, or immediately on the end-of-run flush. Waiting out
    // the gap delays the correction, so enough pending closures also fire
    // the optimization right away.
    const bool episode_open =
        pending &&
        pending_count < parameters.pgo_pending_trigger &&
        frame_id - db->closures.back().query_frame <
            parameters.pgo_episode_end_gap;
    if (episode_open && !force) {
        if (debug_geometry) {
            std::cout << "pose_graph_deferred frame=" << frame_id
                      << " pending="
                      << db->closures.size() -
                             static_cast<std::size_t>(
                                 db->optimized_closure_count)
                      << " last_closure="
                      << db->closures.back().query_frame << std::endl;
        }
    } else if (pending && keyframe_count >= 2) {
        db->last_pgo_frame = frame_id;
        db->optimized_closure_count =
            static_cast<int32_t>(db->closures.size());
        std::unordered_map<int32_t, int32_t> keyframe_index;
        keyframe_index.reserve(static_cast<std::size_t>(keyframe_count));
        for (int32_t i = 0; i < keyframe_count; ++i) {
            keyframe_index[db->keyframes[
                static_cast<std::size_t>(i)].frame_id] = i;
        }
        const std::vector<int32_t> nodes = select_graph_nodes(
            *db, keyframe_index, parameters.pgo_max_graph_poses);
        std::unordered_map<int32_t, int32_t> node_row;
        node_row.reserve(nodes.size());
        for (int32_t p = 0; p < static_cast<int32_t>(nodes.size()); ++p) {
            node_row[nodes[static_cast<std::size_t>(p)]] = p;
        }

        std::vector<std::array<double, 14>> edge_rows;
        std::vector<std::array<double, 3>> weight_rows;
        std::vector<double> edge_log_scales;
        bool edges_ok = true;
        for (std::size_t p = 0; edges_ok && p + 1 < nodes.size(); ++p) {
            edges_ok = append_sequential_edge(
                db->keyframes[static_cast<std::size_t>(nodes[p])].pose,
                db->keyframes[static_cast<std::size_t>(nodes[p + 1])].pose,
                static_cast<int32_t>(p), static_cast<int32_t>(p + 1),
                parameters.pgo_scale_weight, &edge_rows, &weight_rows,
                &edge_log_scales);
        }
        int32_t loop_edges = 0;
        int32_t earliest_match_row = static_cast<int32_t>(nodes.size());
        for (const LoopClosureEvent& event : db->closures) {
            const auto match_it = keyframe_index.find(event.match_frame);
            const auto query_it = keyframe_index.find(event.query_frame);
            if (match_it != keyframe_index.end() &&
                query_it != keyframe_index.end() &&
                match_it->second != query_it->second) {
                const int32_t match_row = node_row[match_it->second];
                append_loop_edge(event, match_row,
                                 node_row[query_it->second], parameters,
                                 &edge_rows, &weight_rows, &edge_log_scales);
                earliest_match_row = std::min(earliest_match_row, match_row);
                ++loop_edges;
            }
        }

        if (edges_ok && loop_edges > 0 && nodes.size() >= 2U) {
            const int32_t node_count = static_cast<int32_t>(nodes.size());
            const int32_t edge_count = static_cast<int32_t>(edge_rows.size());
            cvlib::Matrix poses = cvlib::matrix_create(
                node_count, kPoseParamSize);
            cvlib::Matrix edges = cvlib::matrix_create(edge_count, kEdgeCols);
            cvlib::Matrix weights = cvlib::matrix_create(edge_count, 3);
            std::vector<Sim3> node_states;
            node_states.reserve(static_cast<std::size_t>(node_count));
            for (int32_t p = 0; p < node_count; ++p) {
                node_states.push_back(sim3_from_pose(
                    db->keyframes[static_cast<std::size_t>(
                        nodes[static_cast<std::size_t>(p)])].pose));
            }
            for (int32_t k = 0; k < edge_count; ++k) {
                for (int32_t c = 0; c < kEdgeCols; ++c) {
                    cvlib::matrix_set(&edges, k, c, edge_rows[
                        static_cast<std::size_t>(k)][
                        static_cast<std::size_t>(c)]);
                }
                for (int32_t c = 0; c < 3; ++c) {
                    cvlib::matrix_set(&weights, k, c, weight_rows[
                        static_cast<std::size_t>(k)][
                        static_cast<std::size_t>(c)]);
                }
            }

            // Poses recorded before the earliest loop anchor carry no loop
            // information; freeze them so the correction stays inside the
            // loop segment instead of bending the pre-loop trajectory.
            const int32_t fixed_pose_count = std::min(
                node_count - 1, std::max(1, earliest_match_row + 1));
            cvlib::optimize::LMOptions lm =
                cvlib::optimize::default_lm_options();
            lm.max_iter = parameters.pgo_max_iterations;
            lm.loss.type = parameters.pgo_loss_type;
            lm.loss.scale = parameters.pgo_loss_scale;
            cvlib::optimize::OptimizeReport report = {};
            const cvlib::ErrorCode ec = optimize_sim3_pose_graph(
                &node_states, edges, weights, edge_log_scales,
                fixed_pose_count, lm,
                &report);
            // Downstream consumers (diagnostics, interpolation, global BA)
            // work on the SE(3) equivalents of the optimized Sim(3) states.
            for (int32_t p = 0; p < node_count; ++p) {
                const Pose pose = sim3_to_pose(
                    node_states[static_cast<std::size_t>(p)]);
                for (int32_t i = 0; i < 9; ++i) {
                    cvlib::matrix_set(&poses, p, i, pose.r[i]);
                }
                for (int32_t i = 0; i < 3; ++i) {
                    cvlib::matrix_set(&poses, p, 9 + i, pose.t[i]);
                }
            }

            // The newest node carries the correction the frontend has to
            // follow; a wild scale there means the optimizer absorbed the
            // loop error into the gauge instead of closing the loop, and
            // handing that to tracking is worse than skipping the closure.
            const Sim3 newest_correction = sim3_compose(
                sim3_inverse(sim3_from_pose(
                    db->keyframes[static_cast<std::size_t>(
                        nodes[static_cast<std::size_t>(node_count - 1)])]
                        .pose)),
                node_states[static_cast<std::size_t>(node_count - 1)]);
            const double correction_scale = std::exp(newest_correction.log_s);
            const bool scale_ok =
                correction_scale <= parameters.pgo_max_scale_change &&
                correction_scale >= 1.0 / parameters.pgo_max_scale_change;
            const bool cost_ok = report.final_cost <= report.initial_cost;
            if (ec == cvlib::ErrorCode::kSuccess && cost_ok && scale_ok) {
                std::vector<Pose> base_poses;
                base_poses.reserve(static_cast<std::size_t>(keyframe_count));
                for (const LoopKeyframe& keyframe : db->keyframes) {
                    base_poses.push_back(keyframe.pose);
                }
                accepted = compute_optimized_centers(base_poses, nodes,
                                                     poses,
                                                     optimized_centers,
                                                     corrected_poses);
            } else if (!scale_ok) {
                std::cout << "pose_graph_scale_rejected frame=" << frame_id
                          << " scale=" << correction_scale
                          << " limit=" << parameters.pgo_max_scale_change
                          << std::endl;
            }
            if (accepted) {
                ++db->pgo_runs;
                if (correction != nullptr) {
                    *correction = correction_from_sim3(newest_correction);
                }
                if (corrected_poses != nullptr) {
                    rebase_keyframes(db, *corrected_poses);
                }
            }
            if (ec == cvlib::ErrorCode::kSuccess) {
                // Loop-gap diagnostic: distance between the two loop
                // endpoint centers before and after optimization shows
                // whether the loop actually closed.
                for (int32_t k = node_count - 1; k < edge_count; ++k) {
                    const int32_t from_row = static_cast<int32_t>(
                        edge_rows[static_cast<std::size_t>(k)][0]);
                    const int32_t to_row = static_cast<int32_t>(
                        edge_rows[static_cast<std::size_t>(k)][1]);
                    const LoopKeyframe& from_keyframe = db->keyframes[
                        static_cast<std::size_t>(nodes[
                            static_cast<std::size_t>(from_row)])];
                    const LoopKeyframe& to_keyframe = db->keyframes[
                        static_cast<std::size_t>(nodes[
                            static_cast<std::size_t>(to_row)])];
                    const double gap_before = center_distance(
                        from_keyframe.pose, to_keyframe.pose);
                    const double gap_after = center_distance(
                        param_row_to_pose(poses, from_row),
                        param_row_to_pose(poses, to_row));
                    std::cout << "pose_graph_loop from="
                              << from_keyframe.frame_id
                              << " to=" << to_keyframe.frame_id
                              << " gap=" << gap_before << "->" << gap_after
                              << std::endl;
                    debug_loop_edge_error(
                        poses, edge_rows[static_cast<std::size_t>(k)]);
                }
            }
            const double rotation_error =
                ec == cvlib::ErrorCode::kSuccess
                    ? rotation_orthonormality_error(
                          param_row_to_pose(poses, node_count - 1))
                    : -1.0;
            const double last_log_scale =
                node_states[static_cast<std::size_t>(node_count - 1)].log_s;
            std::cout << "pose_graph frame=" << frame_id
                      << " nodes=" << node_count
                      << " fixed=" << fixed_pose_count
                      << " edges=" << edge_count
                      << " loop_edges=" << loop_edges
                      << " status=" << static_cast<int32_t>(ec)
                      << " accepted=" << accepted
                      << " cost=" << report.initial_cost << "->"
                      << report.final_cost
                      << " rot_err=" << rotation_error
                      << " last_scale=" << std::exp(last_log_scale)
                      << " iterations=" << report.iterations
                      << " term=" << report.termination << std::endl;

            cvlib::matrix_destroy(&poses);
            cvlib::matrix_destroy(&edges);
            cvlib::matrix_destroy(&weights);
        } else if (debug_geometry) {
            std::cout << "pose_graph_skipped frame=" << frame_id
                      << " edges_ok=" << edges_ok
                      << " loop_edges=" << loop_edges
                      << " nodes=" << nodes.size() << std::endl;
        }
    }
    return accepted;
}

namespace {

// Triangulates one correspondence from two poses in the corrected frame.
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

// Composes a * b for Poses treated as rigid transforms.
Pose compose_pose(const Pose& a, const Pose& b) {
    Pose out;
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            double value = 0.0;
            for (int32_t k = 0; k < 3; ++k) {
                value += a.r[row * 3 + k] * b.r[k * 3 + col];
            }
            out.r[row * 3 + col] = value;
        }
        double t = a.t[row];
        for (int32_t k = 0; k < 3; ++k) {
            t += a.r[row * 3 + k] * b.t[k];
        }
        out.t[row] = t;
    }
    return out;
}

Pose invert_pose(const Pose& pose) {
    Pose out;
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            out.r[row * 3 + col] = pose.r[col * 3 + row];
        }
    }
    for (int32_t row = 0; row < 3; ++row) {
        double t = 0.0;
        for (int32_t k = 0; k < 3; ++k) {
            t += out.r[row * 3 + k] * pose.t[k];
        }
        out.t[row] = -t;
    }
    return out;
}

std::vector<int32_t> select_ba_cameras(
    const BowDatabase& db,
    const std::unordered_map<int32_t, int32_t>& keyframe_index,
    int32_t max_cameras) {
    std::set<int32_t> cams;
    const int32_t count = static_cast<int32_t>(db.keyframes.size());
    cams.insert(0);
    cams.insert(count - 1);
    for (const LoopClosureEvent& event : db.closures) {
        const auto query_it = keyframe_index.find(event.query_frame);
        const auto match_it = keyframe_index.find(event.match_frame);
        if (query_it != keyframe_index.end()) {
            cams.insert(query_it->second);
        }
        if (match_it != keyframe_index.end()) {
            cams.insert(match_it->second);
        }
    }
    const int32_t stride = std::max(
        1, (count + max_cameras - 1) / max_cameras);
    for (int32_t i = 0; i < count; i += stride) {
        cams.insert(i);
    }
    return std::vector<int32_t>(cams.begin(), cams.end());
}

}  // namespace

bool run_loop_global_ba(const BowDatabase& db,
                        const std::vector<Pose>& corrected_poses,
                        const MapArchive& archive,
                        const CameraIntrinsics& camera,
                        const LoopClosureParameters& parameters,
                        int32_t frame_id,
                        bool debug_geometry,
                        std::vector<cv::Point3f>* optimized_centers) {
    bool accepted = false;
    optimized_centers->clear();
    const int32_t keyframe_count = static_cast<int32_t>(db.keyframes.size());
    if (db.closures.empty() ||
        corrected_poses.size() != static_cast<std::size_t>(keyframe_count)) {
        return accepted;
    }

    std::unordered_map<int32_t, int32_t> keyframe_index;
    keyframe_index.reserve(static_cast<std::size_t>(keyframe_count));
    for (int32_t i = 0; i < keyframe_count; ++i) {
        keyframe_index[db.keyframes[static_cast<std::size_t>(i)].frame_id] =
            i;
    }
    const std::vector<int32_t> cams = select_ba_cameras(
        db, keyframe_index, parameters.gba_max_cameras);
    const int32_t cam_count = static_cast<int32_t>(cams.size());
    std::unordered_map<int32_t, int32_t> cam_row;
    cam_row.reserve(cams.size());
    for (int32_t c = 0; c < cam_count; ++c) {
        cam_row[cams[static_cast<std::size_t>(c)]] = c;
    }

    // Archived observations restricted to the selected cameras, grouped by
    // persistent point id.
    struct PointGroup {
        std::vector<std::size_t> observation_indices;
    };
    std::unordered_map<int32_t, PointGroup> groups;
    for (std::size_t i = 0; i < archive.observations.size(); ++i) {
        const MapObservation& obs = archive.observations[i];
        const auto kf_it = keyframe_index.find(obs.frame_id);
        if (kf_it != keyframe_index.end() &&
            cam_row.find(kf_it->second) != cam_row.end()) {
            groups[obs.point_id].observation_indices.push_back(i);
        }
    }
    std::vector<std::pair<int32_t, int32_t>> eligible;
    for (const auto& entry : groups) {
        const int32_t n = static_cast<int32_t>(
            entry.second.observation_indices.size());
        if (n >= parameters.gba_min_observations) {
            eligible.push_back({entry.first, n});
        }
    }
    std::sort(eligible.begin(), eligible.end(),
              [](const std::pair<int32_t, int32_t>& a,
                 const std::pair<int32_t, int32_t>& b) {
                  return a.second != b.second ? a.second > b.second
                                              : a.first < b.first;
              });
    if (static_cast<int32_t>(eligible.size()) > parameters.gba_max_points) {
        eligible.resize(static_cast<std::size_t>(parameters.gba_max_points));
    }

    // Segment-wise corrections are not globally rigid, so transported map
    // positions would start the BA tens of pixels off. Re-triangulate each
    // point from its widest-baseline pair of corrected cameras instead so
    // the initial residuals are consistent with the corrected geometry.
    std::vector<cv::Point3f> point_positions;
    std::vector<std::array<double, 4>> observation_rows;
    point_positions.reserve(eligible.size());
    for (const std::pair<int32_t, int32_t>& entry : eligible) {
        std::vector<std::pair<int32_t, cv::Point2f>> views;
        views.reserve(groups[entry.first].observation_indices.size());
        for (const std::size_t obs_idx :
             groups[entry.first].observation_indices) {
            const MapObservation& obs = archive.observations[obs_idx];
            views.push_back({keyframe_index[obs.frame_id], obs.pixel});
        }
        std::sort(views.begin(), views.end(),
                  [](const std::pair<int32_t, cv::Point2f>& a,
                     const std::pair<int32_t, cv::Point2f>& b) {
                      return a.first < b.first;
                  });
        if (views.size() < 2U || views.front().first == views.back().first) {
            continue;
        }
        // Widest baseline gives the best-conditioned depth but fails when
        // the endpoints straddle different correction segments; fall back
        // to the adjacent pair, which shares a segment, and let Huber
        // weight the farther observations.
        const std::array<std::pair<std::size_t, std::size_t>, 2> pairs = {
            {{0U, views.size() - 1U}, {0U, 1U}}};
        cv::Point3f point;
        bool point_ok = false;
        for (const std::pair<std::size_t, std::size_t>& pair : pairs) {
            const Pose& pose_a =
                corrected_poses[static_cast<std::size_t>(
                    views[pair.first].first)];
            const Pose& pose_b =
                corrected_poses[static_cast<std::size_t>(
                    views[pair.second].first)];
            cv::Point3f candidate;
            if (!triangulate_pair(pose_a, pose_b, views[pair.first].second,
                                  views[pair.second].second, camera,
                                  &candidate)) {
                continue;
            }
            const double residual_a = reprojection_residual(
                candidate, views[pair.first].second, pose_a, camera);
            const double residual_b = reprojection_residual(
                candidate, views[pair.second].second, pose_b, camera);
            // Huber down-weights moderate mismatch, so the seed only has
            // to be roughly consistent; a tight gate here starves the BA
            // of track points.
            const double seed_gate = 3.0 * parameters.gba_loss_scale;
            if (depth_in_pose(candidate, pose_a) > 1.0e-6 &&
                depth_in_pose(candidate, pose_b) > 1.0e-6 &&
                std::isfinite(residual_a) && std::isfinite(residual_b) &&
                residual_a <= seed_gate && residual_b <= seed_gate) {
                point = candidate;
                point_ok = true;
                break;
            }
        }
        if (!point_ok) {
            continue;
        }
        std::vector<std::array<double, 4>> point_observations;
        for (const std::size_t obs_idx :
             groups[entry.first].observation_indices) {
            const MapObservation& obs = archive.observations[obs_idx];
            const int32_t kf = keyframe_index[obs.frame_id];
            const Pose& view_pose =
                corrected_poses[static_cast<std::size_t>(kf)];
            const double residual = reprojection_residual(
                point, obs.pixel, view_pose, camera);
            if (depth_in_pose(point, view_pose) > 1.0e-6 &&
                std::isfinite(residual)) {
                point_observations.push_back(
                    {static_cast<double>(cam_row[kf]), 0.0,
                     static_cast<double>(obs.pixel.x),
                     static_cast<double>(obs.pixel.y)});
            }
        }
        if (static_cast<int32_t>(point_observations.size()) <
            parameters.gba_min_observations) {
            continue;
        }
        const int32_t point_row =
            static_cast<int32_t>(point_positions.size());
        point_positions.push_back(point);
        for (std::array<double, 4>& row : point_observations) {
            row[1] = static_cast<double>(point_row);
            observation_rows.push_back(row);
        }
    }
    // Loop-pair points: triangulated from the PGO-corrected poses so both
    // frames' reprojection terms couple the loop in the BA.
    int32_t loop_points = 0;
    for (const LoopClosureEvent& event : db.closures) {
        const auto match_it = keyframe_index.find(event.match_frame);
        const auto query_it = keyframe_index.find(event.query_frame);
        if (match_it == keyframe_index.end() ||
            query_it == keyframe_index.end() ||
            cam_row.find(match_it->second) == cam_row.end() ||
            cam_row.find(query_it->second) == cam_row.end()) {
            continue;
        }
        const Pose& match_pose =
            corrected_poses[static_cast<std::size_t>(match_it->second)];
        const Pose& query_pose =
            corrected_poses[static_cast<std::size_t>(query_it->second)];
        const std::size_t pair_count = std::min(
            {event.match_inlier_points.size(),
             event.query_inlier_points.size(),
             static_cast<std::size_t>(parameters.gba_max_loop_points)});
        if (pair_count == 0U) {
            continue;
        }
        std::vector<cv::Point2f> match_pixels(
            event.match_inlier_points.begin(),
            event.match_inlier_points.begin() +
                static_cast<std::ptrdiff_t>(pair_count));
        std::vector<cv::Point2f> query_pixels(
            event.query_inlier_points.begin(),
            event.query_inlier_points.begin() +
                static_cast<std::ptrdiff_t>(pair_count));
        cvlib::Matrix p1 = pose_to_projection(match_pose);
        cvlib::Matrix p2 = pose_to_projection(query_pose);
        cvlib::Matrix x1 = points2f_to_normalized_matrix(match_pixels,
                                                         camera);
        cvlib::Matrix x2 = points2f_to_normalized_matrix(query_pixels,
                                                         camera);
        cvlib::Matrix triangulated = cvlib::matrix_create(
            static_cast<int32_t>(pair_count), 3);
        const cvlib::ErrorCode tri_ec = cvlib::calib3d::triangulate_points(
            &p1, &p2, &x1, &x2, &triangulated);
        if (tri_ec == cvlib::ErrorCode::kSuccess) {
            for (std::size_t i = 0; i < pair_count; ++i) {
                const cv::Point3f point(
                    static_cast<float>(cvlib::matrix_get(
                        &triangulated, static_cast<int32_t>(i), 0)),
                    static_cast<float>(cvlib::matrix_get(
                        &triangulated, static_cast<int32_t>(i), 1)),
                    static_cast<float>(cvlib::matrix_get(
                        &triangulated, static_cast<int32_t>(i), 2)));
                const double residual_match = reprojection_residual(
                    point, match_pixels[i], match_pose, camera);
                const double residual_query = reprojection_residual(
                    point, query_pixels[i], query_pose, camera);
                if (depth_in_pose(point, match_pose) > 1.0e-6 &&
                    depth_in_pose(point, query_pose) > 1.0e-6 &&
                    std::isfinite(residual_match) &&
                    std::isfinite(residual_query) &&
                    residual_match <= parameters.gba_loss_scale &&
                    residual_query <= parameters.gba_loss_scale) {
                    const int32_t point_row = static_cast<int32_t>(
                        point_positions.size());
                    point_positions.push_back(point);
                    observation_rows.push_back(
                        {static_cast<double>(cam_row[match_it->second]),
                         static_cast<double>(point_row),
                         static_cast<double>(match_pixels[i].x),
                         static_cast<double>(match_pixels[i].y)});
                    observation_rows.push_back(
                        {static_cast<double>(cam_row[query_it->second]),
                         static_cast<double>(point_row),
                         static_cast<double>(query_pixels[i].x),
                         static_cast<double>(query_pixels[i].y)});
                    ++loop_points;
                }
            }
        }
        cvlib::matrix_destroy(&p1);
        cvlib::matrix_destroy(&p2);
        cvlib::matrix_destroy(&x1);
        cvlib::matrix_destroy(&x2);
        cvlib::matrix_destroy(&triangulated);
    }

    const int32_t point_count = static_cast<int32_t>(point_positions.size());
    const int32_t observation_count =
        static_cast<int32_t>(observation_rows.size());
    if (point_count < 1 || observation_count < 2 || loop_points == 0) {
        if (debug_geometry) {
            std::cout << "global_ba_skipped frame=" << frame_id
                      << " points=" << point_count
                      << " loop_points=" << loop_points
                      << " observations=" << observation_count << std::endl;
        }
        return accepted;
    }

    cvlib::Matrix k = make_camera_matrix(camera);
    cvlib::Matrix poses = cvlib::matrix_create(cam_count, 12);
    cvlib::Matrix points = cvlib::matrix_create(point_count, 3);
    cvlib::Matrix observations = cvlib::matrix_create(observation_count, 4);
    for (int32_t c = 0; c < cam_count; ++c) {
        const Pose& pose =
            corrected_poses[static_cast<std::size_t>(cams[
                static_cast<std::size_t>(c)])];
        for (int32_t i = 0; i < 9; ++i) {
            cvlib::matrix_set(&poses, c, i, pose.r[i]);
        }
        for (int32_t i = 0; i < 3; ++i) {
            cvlib::matrix_set(&poses, c, 9 + i, pose.t[i]);
        }
    }
    for (int32_t p = 0; p < point_count; ++p) {
        const cv::Point3f& point =
            point_positions[static_cast<std::size_t>(p)];
        cvlib::matrix_set(&points, p, 0, static_cast<double>(point.x));
        cvlib::matrix_set(&points, p, 1, static_cast<double>(point.y));
        cvlib::matrix_set(&points, p, 2, static_cast<double>(point.z));
    }
    for (int32_t o = 0; o < observation_count; ++o) {
        for (int32_t c = 0; c < 4; ++c) {
            cvlib::matrix_set(&observations, o, c, observation_rows[
                static_cast<std::size_t>(o)][static_cast<std::size_t>(c)]);
        }
    }

    cvlib::calib3d::BAOptions options =
        cvlib::calib3d::default_ba_options();
    options.solver = cvlib::calib3d::kBASolverSchur;
    options.lm.max_iter = parameters.gba_max_iterations;
    options.lm.loss.type = cvlib::optimize::kLossHuber;
    options.lm.loss.scale = parameters.gba_loss_scale;
    cvlib::calib3d::BAData data = {&poses, &points, &observations, &k,
                                   nullptr};
    cvlib::optimize::OptimizeReport report = {};
    const cvlib::ErrorCode ec =
        cvlib::calib3d::bundle_adjustment(&data, &options, &report);

    const bool cost_ok = report.final_cost <= report.initial_cost;
    if (ec == cvlib::ErrorCode::kSuccess && cost_ok) {
        // Re-anchor the gauge to the earliest camera so the fixed pre-loop
        // segment stays put: T'' = T' * D with D = T_first'^-1 * T_first.
        const Pose first_before =
            corrected_poses[static_cast<std::size_t>(cams[0])];
        const Pose first_after = param_row_to_pose(poses, 0);
        const Pose anchor = compose_pose(invert_pose(first_after),
                                         first_before);
        for (int32_t c = 0; c < cam_count; ++c) {
            const Pose anchored = compose_pose(param_row_to_pose(poses, c),
                                               anchor);
            for (int32_t i = 0; i < 9; ++i) {
                cvlib::matrix_set(&poses, c, i, anchored.r[i]);
            }
            for (int32_t i = 0; i < 3; ++i) {
                cvlib::matrix_set(&poses, c, 9 + i, anchored.t[i]);
            }
        }
        accepted = compute_optimized_centers(corrected_poses, cams, poses,
                                             optimized_centers, nullptr);
    }
    std::cout << "global_ba frame=" << frame_id
              << " cams=" << cam_count
              << " points=" << point_count
              << " loop_points=" << loop_points
              << " observations=" << observation_count
              << " status=" << static_cast<int32_t>(ec)
              << " accepted=" << accepted
              << " cost=" << report.initial_cost << "->"
              << report.final_cost
              << " iterations=" << report.iterations
              << " term=" << report.termination << std::endl;

    cvlib::matrix_destroy(&k);
    cvlib::matrix_destroy(&poses);
    cvlib::matrix_destroy(&points);
    cvlib::matrix_destroy(&observations);
    return accepted;
}

}  // namespace mvo
