#include <string>
#include <iostream>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "PoseEstimator.h"

using namespace std::filesystem;
using namespace cv;

int main(int argc, char* argv[])
{
    int min_hessian = 500; // 400;
    std::string query_image_path = "../data/query.png";
    std::string view_images_path = "../data/view_images";
    std::string metadata_path = "../data/calib_data.json";
    cpp_practicing::PoseEstimator pose_estimator(
        query_image_path,
        metadata_path,
        view_images_path,
        min_hessian
    );
    pose_estimator.estimate();

    // cpp_practicing::testKeypointDetector(query_image_path);
}
