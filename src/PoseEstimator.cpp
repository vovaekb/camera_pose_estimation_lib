#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <chrono>
#include <thread>
#include <mutex>
#include <functional>
#include <algorithm>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <fmt/core.h>
 
#include <Eigen/Core>
#include "PoseEstimator.h"
 
using namespace std::filesystem;
using namespace cv;

namespace cpp_practicing {
    using string_vector = std::vector<std::string>;
    using float_vector = std::vector<float>;
    using json = nlohmann::json;
    
    namespace {
        float mae(const float_vector& predictions, const float_vector& targets) {
            if (predictions.empty()) return 0.0f;
            float_vector differences;
            differences.reserve(predictions.size());
            std::transform(
                predictions.begin(), 
                predictions.end(), 
                targets.begin(),
                std::back_inserter(differences),
                [](float pred, float target) { return std::abs(pred - target); }
            );
            float diff_sum = std::accumulate(differences.begin(), differences.end(), 0.0f);
            return diff_sum / static_cast<float>(differences.size());
        }

        float mae([[maybe_unused]] const Eigen::MatrixXf& predictions, [[maybe_unused]] const Eigen::MatrixXf& targets) {
            return 0.0f;
        }

        Eigen::MatrixXf convertQuaternionToMatrix(PoseEstimator::Rotation rotation) {
            auto [w, x, y, z] = rotation;
            Eigen::MatrixXf result(3, 3);
            result(0, 0) = 1.0f - 2.0f * y * y - 2.0f * z * z;
            result(0, 1) = 2.0f * x * y - 2.0f * w * z;
            result(0, 2) = 2.0f * x * z + 2.0f * w * y;
            result(1, 0) = 2.0f * x * y + 2.0f * w * z;
            result(1, 1) = 1.0f - 2.0f * x * x - 2.0f * z * z;
            result(1, 2) = 2.0f * y * z - 2.0f * w * x;
            result(2, 0) = 2.0f * x * z - 2.0f * w * y;
            result(2, 1) = 2.0f * y * z + 2.0f * w * x;
            result(2, 2) = 1.0f - 2.0f * x * x - 2.0f * y * y;
            return result;
        }
    }

    void testKeypointDetector(std::string_view imgPath)
    {
        fmt::print("testKeypointDetector called\n");

        fmt::print("Loading query image ...\n");
        Mat image = imread(static_cast<std::string>(imgPath), IMREAD_COLOR);
        PoseEstimator::ImageSample query_image {static_cast<std::string>(imgPath), image, {}, {}, 0};
        fmt::print("query image size: {} x {}\n", query_image.image_data.rows, query_image.image_data.cols);

        int min_hessian = 500;
        Ptr<Feature2D> detector = ORB::create(min_hessian);
        fmt::print("Calculate image descriptors ...\n");
        detector->detectAndCompute(query_image.image_data, noArray(), query_image.keypoints, query_image.descriptors);
        fmt::print("Descriptors for query image calculated\n");
        fmt::print("keypoints number: {}\n", query_image.keypoints.size());
    }

    PoseEstimator::PoseEstimator(
        const std::string& image_file_path, 
        const std::string& metadata_file_path, 
        const std::string& view_files_path,
        int min_hessian) : 
            m_query_image_file(image_file_path), 
            m_query_metadata_file(metadata_file_path),
            m_view_files_path(view_files_path),
            m_min_hessian(min_hessian),
            matcher(cv::BFMatcher::create(cv::NORM_HAMMING)),
            detector(ORB::create(m_min_hessian)),
            camera_matrix(Eigen::Array33f::Zero()) {
        view_images.reserve(MAX_VIEWS_NUMBER);

        // detector = ORB::create(m_min_hessian);
    }

    void PoseEstimator::estimate()
    {
        fmt::print("start estimate ...\n");
        loadQueryImage();
        loadImageMetadata();
        camera_matrix(0, 0) = query_image_metadata.calibration_data.fx;
        camera_matrix(0, 2) = query_image_metadata.calibration_data.cx;
        camera_matrix(1, 1) = query_image_metadata.calibration_data.fy;
        camera_matrix(1, 2) = query_image_metadata.calibration_data.cy;
        camera_matrix(2, 2) = 1;
        
        loadViewImages();
        chunk_size = static_cast<int>(view_images.size() / THREADS_NUMBER);
        findImageDescriptors();
        match();
    }

    void PoseEstimator::loadImageMetadata()
    {
        fmt::print("Loading image metadata ...\n");
        std::ifstream ifs(m_query_metadata_file);
        json json_data = json::parse(ifs);
        auto calibration_info = json_data.at("calibration");
        float fx = calibration_info.at("fx");
        float fy = calibration_info.at("fy");
        float cx = calibration_info.at("cx");
        float cy = calibration_info.at("cy");
        auto pose_json = json_data.at("pose");
        auto origin = pose_json.at("origin");
        auto rotation_json = pose_json.at("rotation");
        PoseEstimator::Rotation rotation = {
            rotation_json.at("w"),
            rotation_json.at("x"),
            rotation_json.at("y"),
            rotation_json.at("z")
        };
        float_vector translation;
        translation.reserve(3);
        for (size_t i = 0; i < 3; ++i) translation.emplace_back(origin[i]);
        PoseEstimator::TransformPose pose = {rotation, translation}; 
        PoseEstimator::CalibrationData calibration_data = {fx, fy, cx, cy};
        query_image_metadata = ImageMetadata { calibration_data, pose };
    }

    void PoseEstimator::loadQueryImage()
    {
        fmt::print("Loading query image ...\n");
        Mat image = imread(m_query_image_file, IMREAD_COLOR);
        query_image = ImageSample {m_query_image_file, image, {}, {}, 0};
    }

    void PoseEstimator::loadViewImages()
    {
        fmt::print("Loading view images ...\n");
        path dir_path = m_view_files_path;
        if (!exists(dir_path)) return;
        for (auto& file : directory_iterator(dir_path))
        {
            auto file_path = file.path();
            if (file_path.extension() == ".jpg" || file_path.extension() == ".png") {
                Mat image = imread(file_path, IMREAD_COLOR);
                view_images.emplace_back(ImageSample {file_path.filename().string(), image, {}, {}, 0});
            }
        }
        fmt::print("Complete\n");
    }

    void PoseEstimator::findImageDescriptors()
    {
        fmt::print("Looking for image descriptors ...\n");
        detector->detectAndCompute(query_image.image_data, noArray(), query_image.keypoints, query_image.descriptors);
        fmt::print("Descriptors for query image calculated\n");
        fmt::print("keypoints number: {}\n", query_image.keypoints.size());
        std::vector<std::thread> threads;
        for (size_t i = 0; i < THREADS_NUMBER; ++i)
        {
            threads.emplace_back(std::thread([&](int thread_idx) {
                int start_index = chunk_size * thread_idx;
                int end_index = (thread_idx == THREADS_NUMBER - 1) ? static_cast<int>(view_images.size()) : (thread_idx + 1) * chunk_size;
                for (int j = start_index; j < end_index; ++j) {
                    std::lock_guard<std::mutex> l(m);
                    detector->detectAndCompute(view_images[j].image_data, noArray(), view_images[j].keypoints, view_images[j].descriptors);
                }
            }, i));
        }
        for (auto &th : threads) if (th.joinable()) th.join();
        fmt::print("Complete\n");
    }

    void PoseEstimator::match()
    {
        fmt::print("Start matching ...\n");
        std::vector<view_matches_vector> views_matches(view_images.size());
        std::vector<std::thread> threads;
        for (size_t i = 0; i < THREADS_NUMBER; ++i) {
            threads.emplace_back(std::thread([&](int thread_idx) {
                int start_index = chunk_size * thread_idx;
                int end_index = (thread_idx == THREADS_NUMBER - 1) ? static_cast<int>(view_images.size()) : (thread_idx + 1) * chunk_size;
                for (int j = start_index; j < end_index; ++j) {
                    view_matches_vector local_matches;
                    matcher->match(view_images[j].descriptors, query_image.descriptors, local_matches);
                    if (!local_matches.empty()) {
                        auto min_it = std::min_element(local_matches.begin(), local_matches.end(), [](const auto& a, const auto& b){ return a.distance < b.distance; });
                        float min_dist = min_it->distance;
                        local_matches.erase(std::remove_if(local_matches.begin(), local_matches.end(), [min_dist](const auto& m){ return m.distance > 2 * min_dist; }), local_matches.end());
                    }
                    std::lock_guard<std::mutex> l(m);
                    views_matches[j] = std::move(local_matches);
                }
            }, i));
        }
        for (auto &th : threads) if (th.joinable()) th.join();

        std::vector<int> views_matches_inliers(view_images.size());
        for (size_t i = 0; i < view_images.size(); ++i) {
            std::vector<Point2d> pts1, pts2;
            for (auto& match : views_matches[i]) {
                pts1.push_back(view_images[i].keypoints[match.queryIdx].pt);
                pts2.push_back(query_image.keypoints[match.trainIdx].pt);
            }
            if (pts1.size() >= 4) {
                Mat inliers;
                findHomography(pts1, pts2, RANSAC, 3, inliers);
                views_matches_inliers[i] = static_cast<int>(cv::countNonZero(inliers));
            } else {
                views_matches_inliers[i] = 0;
            }
        }
    }

    void PoseEstimator::matchTwoImages() const {}
    void PoseEstimator::calculateTransformation() {}
    void PoseEstimator::getPoseError() {
        auto gt_translation = query_image_metadata.pose.translation;
        auto gt_rotation = query_image_metadata.pose.rotation;
        auto gt_rotation_matrix = convertQuaternionToMatrix(gt_rotation);
        [[maybe_unused]] auto translation_error = mae(gt_translation, result_pose_translation);
        [[maybe_unused]] auto rotation_error = mae(gt_rotation_matrix, result_pose_rotation);
    }

    auto PoseEstimator::getQueryImageKeypoints() const -> keypoints_vector { return query_image.keypoints; }
    auto PoseEstimator::getQueryImage() const -> ImageSample { return query_image; }
    auto PoseEstimator::getViewImages() const -> std::vector<ImageSample> { return view_images; }
    auto PoseEstimator::getQueryImageMetadata() const -> ImageMetadata { return query_image_metadata; }
}
