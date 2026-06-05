#pragma once

#include "mvo/config.h"

#include <opencv2/videoio.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace mvo {

struct FrameSource {
    InputType input_type = InputType::kImageSequence;
    std::vector<std::string> images;
    cv::VideoCapture video;
    int32_t next_index = 0;
    int32_t total_frames = 0;
    bool opened = false;
};

bool open_frame_source(const AppConfig& config, FrameSource* source);
bool read_next_frame(FrameSource* source, cv::Mat* gray);

}  // namespace mvo
