
#include <string>
#include <iostream>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>

#include "PoseEstimator.h"

using namespace std::filesystem;
using namespace cv;

using namespace cpp_practicing;

int main() {
    std::cout << "Hello" << std::endl;

    std::string query_image_path = "query.jpg";
    std::string view_images_path = "view_images";
    PoseEstimator pose_estimator;
    pose_estimator.estimate(query_image_path, view_images_path);
}