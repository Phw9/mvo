#pragma once

#include "parameters.h"

#include "DBoW2/BowVector.h"
#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace mvo {

using OrbVocabulary =
    DBoW2::TemplatedVocabulary<cv::Mat, DBoW2::FORB>;

struct BowDatabase {
    OrbVocabulary vocabulary;
    bool vocabulary_loaded = false;
    std::vector<DBoW2::BowVector> bows;
};

bool load_vocabulary(const std::string& path, BowDatabase* db);
bool query_and_add_loop(const cv::Mat& image, BowDatabase* db,
                        int32_t frame_id,
                        const LoopClosureParameters& parameters);

}  // namespace mvo
