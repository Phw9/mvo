#pragma once

#include "mvo/parameters.h"
#include "mvo/types.h"

#include <cstdint>
#include <string>

namespace mvo {

struct AppConfig {
    InputType input_type = InputType::kImageSequence;
    std::string input_path = "image/image_0";
    std::string input_config_path;
    std::string parameter_dir = "configs/parameters";
    std::string calib_path = "image/calib.txt";
    std::string vocabulary = "image/KITTI_00_phphww_voc.yml.gz";
    CameraIntrinsics camera;
    int32_t max_frames = 20;
    bool no_gui = true;
    bool run_ba = true;
    bool debug_geometry = false;
    bool rerun_spawn = true;
    std::string rerun_save_path;
    MvoParameters parameters;
};

void set_identity_pose(Pose* pose);
std::string input_type_name(InputType input_type);
AppConfig parse_args(int argc, char** argv);
bool load_kitti_calibration(const std::string& path,
                            CameraIntrinsics* camera);

}  // namespace mvo
