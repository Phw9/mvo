#include "config.h"

#include <opencv2/core/persistence.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

namespace mvo {

void set_identity_pose(Pose* pose) {
    for (int32_t i = 0; i < 9; ++i) {
        pose->r[i] = 0.0;
    }
    pose->r[0] = 1.0;
    pose->r[4] = 1.0;
    pose->r[8] = 1.0;
    pose->t[0] = 0.0;
    pose->t[1] = 0.0;
    pose->t[2] = 0.0;
}

std::string input_type_name(InputType input_type) {
    std::string name = "image_sequence";
    if (input_type == InputType::kVideo) {
        name = "video";
    }
    return name;
}

bool parse_input_type(const std::string& name, InputType* input_type) {
    bool ok = false;
    if (name == "image" || name == "images" || name == "image_sequence") {
        *input_type = InputType::kImageSequence;
        ok = true;
    } else if (name == "video") {
        *input_type = InputType::kVideo;
        ok = true;
    }
    return ok;
}

void read_string_node(const cv::FileNode& node, const std::string& key,
                      std::string* value) {
    const cv::FileNode child = node[key];
    if (!child.empty()) {
        *value = static_cast<std::string>(child);
    }
}

void read_int_node(const cv::FileNode& node, const std::string& key,
                   int32_t* value) {
    const cv::FileNode child = node[key];
    if (!child.empty()) {
        *value = static_cast<int32_t>(child);
    }
}

void read_bool_node(const cv::FileNode& node, const std::string& key,
                    bool* value) {
    const cv::FileNode child = node[key];
    if (!child.empty()) {
        *value = static_cast<int32_t>(child) != 0;
    }
}

bool load_json_config(const std::string& path, AppConfig* config) {
    bool ok = false;
    cv::FileStorage fs(path, cv::FileStorage::READ |
                                 cv::FileStorage::FORMAT_JSON);
    if (fs.isOpened()) {
        const cv::FileNode input = fs["input"];
        const cv::FileNode root = input.empty() ? fs.root() : input;
        std::string input_type;
        read_string_node(root, "type", &input_type);
        if (!input_type.empty()) {
            InputType parsed_type = config->input_type;
            if (parse_input_type(input_type, &parsed_type)) {
                config->input_type = parsed_type;
            }
        }
        read_string_node(root, "path", &config->input_path);
        read_string_node(root, "parameter_dir", &config->parameter_dir);
        read_string_node(root, "parameters_dir", &config->parameter_dir);
        read_string_node(root, "calib", &config->calib_path);
        read_string_node(root, "calibration", &config->calib_path);
        read_string_node(root, "vocabulary", &config->vocabulary);
        read_int_node(root, "max_frames", &config->max_frames);
        read_bool_node(root, "run_ba", &config->run_ba);
        read_bool_node(root, "no_gui", &config->no_gui);
        read_bool_node(root, "debug_geometry", &config->debug_geometry);

        const cv::FileNode rerun = root["rerun"];
        if (!rerun.empty()) {
            read_bool_node(rerun, "spawn", &config->rerun_spawn);
            read_string_node(rerun, "save", &config->rerun_save_path);
        }
        const cv::FileNode parameters = root["parameters"];
        if (!parameters.empty()) {
            read_string_node(parameters, "directory",
                             &config->parameter_dir);
            read_string_node(parameters, "dir", &config->parameter_dir);
        }
        config->max_frames = std::max(2, config->max_frames);
        ok = true;
    }
    return ok;
}

AppConfig parse_args(int argc, char** argv) {
    AppConfig config;
    for (int32_t i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input-config" && i + 1 < argc) {
            ++i;
            config.input_config_path = argv[i];
            if (!load_json_config(config.input_config_path, &config)) {
                std::cout << "input_config=failed path="
                          << config.input_config_path << std::endl;
            }
        }
    }
    for (int32_t i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--image-dir" && i + 1 < argc) {
            ++i;
            config.input_type = InputType::kImageSequence;
            config.input_path = argv[i];
        } else if (arg == "--video" && i + 1 < argc) {
            ++i;
            config.input_type = InputType::kVideo;
            config.input_path = argv[i];
        } else if (arg == "--input-path" && i + 1 < argc) {
            ++i;
            config.input_path = argv[i];
        } else if (arg == "--input-type" && i + 1 < argc) {
            ++i;
            InputType parsed_type = config.input_type;
            if (parse_input_type(argv[i], &parsed_type)) {
                config.input_type = parsed_type;
            }
        } else if (arg == "--input-config" && i + 1 < argc) {
            ++i;
        } else if (arg == "--parameter-dir" && i + 1 < argc) {
            ++i;
            config.parameter_dir = argv[i];
        } else if (arg == "--vocabulary" && i + 1 < argc) {
            ++i;
            config.vocabulary = argv[i];
        } else if (arg == "--calib" && i + 1 < argc) {
            ++i;
            config.calib_path = argv[i];
        } else if (arg == "--max-frames" && i + 1 < argc) {
            ++i;
            config.max_frames = std::max(2, std::stoi(argv[i]));
        } else if (arg == "--no-gui") {
            config.no_gui = true;
        } else if (arg == "--no-ba") {
            config.run_ba = false;
        } else if (arg == "--debug-geometry") {
            config.debug_geometry = true;
        } else if (arg == "--rerun-spawn") {
            config.rerun_spawn = true;
        } else if (arg == "--no-rerun") {
            config.rerun_spawn = false;
            config.rerun_save_path.clear();
        } else if (arg == "--rerun-save" && i + 1 < argc) {
            ++i;
            config.rerun_save_path = argv[i];
        }
    }
    load_parameter_configs(config.parameter_dir, &config.parameters);
    return config;
}

bool load_kitti_calibration(const std::string& path,
                            CameraIntrinsics* camera) {
    bool ok = false;
    std::ifstream file(path);
    std::string line;
    while (!ok && std::getline(file, line)) {
        std::istringstream iss(line);
        std::string label;
        double p[12] = {};
        iss >> label;
        if (label == "P0:" || label == "P0") {
            bool parsed = true;
            for (double& value : p) {
                if (!(iss >> value)) {
                    parsed = false;
                }
            }
            if (parsed && p[0] > 0.0 && p[5] > 0.0) {
                camera->fx = p[0];
                camera->fy = p[5];
                camera->cx = p[2];
                camera->cy = p[6];
                ok = true;
            }
        }
    }
    return ok;
}

}  // namespace mvo
