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

namespace cpp_practicing {
    // void generate_images() {
    //     // image 1
    //     Mat img1 (300, 250, CV_8UC1, Scalar(0, 0, 0));
    //     //imwrite("img1.ppg", img1);
    //     std::cout << "image was saved" << std::endl;
    // }

    TEST(PoseEstimatorTest, Initial) {
        int min_hessian = 400;
        std::string query_image_path = "query.png";
        std::string view_images_path = "view_images";
        std::string metadata_path = "calib_data.json";
        PoseEstimator pose_estimator(
            query_image_path, 
            metadata_path, 
            view_images_path, 
            min_hessian
        );
        pose_estimator.estimate();
        ASSERT_TRUE(true);

    }

    TEST(PoseEstimatorTest, LoadQueryImage) {
        int min_hessian = 400;
        std::string query_image_path = "query.png";
        std::string view_images_path = "view_images";
        std::string metadata_path = "calib_data.json";
        PoseEstimator pose_estimator(
            query_image_path, 
            metadata_path, 
            view_images_path,
            min_hessian);
        pose_estimator.estimate();
        auto query_image = pose_estimator.getQueryImage();
        ASSERT_EQ(query_image.file_name, "query.png");

    }

    TEST(PoseEstimatorTest, LoadViewImages) {
        int min_hessian = 400;
        std::string query_image_path = "query.png";
        std::string view_images_path = "view_images";
        std::string metadata_path = "calib_data.json";
        PoseEstimator pose_estimator(
            query_image_path, 
            metadata_path, 
            view_images_path,
            min_hessian);
        pose_estimator.estimate();
        auto view_images = pose_estimator.getViewImages();
        ASSERT_EQ(static_cast<int>(view_images.size()), 6);

    }

    TEST(PoseEstimatorTest, LoadQueryImageMetadata) {
        int min_hessian = 400;
        std::string query_image_path = "query.png";
        std::string view_images_path = "view_images";
        std::string metadata_path = "calib_data.json";
        PoseEstimator pose_estimator(
            query_image_path, 
            metadata_path, 
            view_images_path,
            min_hessian);
        pose_estimator.estimate();
        auto query_image_metadata = pose_estimator.getQueryImageMetadata();
        auto pose = query_image_metadata.pose;
        ASSERT_EQ(static_cast<int>(pose.translation.size()), 3);

    }

    TEST(PoseEstimatorTest, TestQueryKeypoints) {
        int min_hessian = 400;
        std::string query_image_path = "query.png";
        std::string view_images_path = "view_images";
        std::string metadata_path = "calib_data.json";
        PoseEstimator pose_estimator(
            query_image_path, 
            metadata_path, 
            view_images_path,
            min_hessian);
        pose_estimator.estimate();
        auto keypoints = pose_estimator.getQueryImageKeypoints();
        EXPECT_TRUE(static_cast<int>(keypoints.size()) > 0);

    }
}