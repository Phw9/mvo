#include "loop_closure.h"

#include "converter.h"

#include <opencv2/features2d.hpp>

#include <filesystem>
#include <iostream>

namespace mvo {

bool load_vocabulary(const std::string& path, BowDatabase* db) {
    bool ok = false;
    if (std::filesystem::exists(path)) {
        db->vocabulary.load(path);
        db->vocabulary_loaded = !db->vocabulary.empty();
        ok = db->vocabulary_loaded;
    }
    return ok;
}

bool query_and_add_loop(const cv::Mat& image, BowDatabase* db,
                        int32_t frame_id,
                        const LoopClosureParameters& parameters) {
    bool ok = false;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(parameters.orb_features);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

    if (db->vocabulary_loaded && !descriptors.empty()) {
        DBoW2::BowVector bow;
        std::vector<cv::Mat> rows = descriptor_rows(descriptors);
        db->vocabulary.transform(rows, bow);

        int32_t best_id = -1;
        double best_score = 0.0;
        for (int32_t i = 0; i < static_cast<int32_t>(db->bows.size()); ++i) {
            if (frame_id - i > parameters.recent_exclusion) {
                const double score = db->vocabulary.score(bow, db->bows[i]);
                if (score > best_score) {
                    best_score = score;
                    best_id = i;
                }
            }
        }

        std::cout << "loop_query frame=" << frame_id
                  << " best_id=" << best_id
                  << " score=" << best_score << std::endl;
        db->bows.push_back(bow);
        ok = true;
    }
    return ok;
}

}  // namespace mvo
