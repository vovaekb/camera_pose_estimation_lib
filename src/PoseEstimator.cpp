#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <Eigen/Core>

#include "PoseEstimator.h"

// using fs = std::filesystem;
using namespace std::filesystem;
using namespace cv;

namespace cpp_practicing {
    using string_vector = std::vector<std::string>;
    using float_vector = std::vector<float>;
    using json = nlohmann::json;

    namespace {

        /**
         * @brief Calculate MAE (mean absolute error) error for translation component between predictions pose and ground truth pose 
         * @param[in] predictions Predicted translation vector  
         * @param[in] targets Ground truth translation vector
         * @return Mean absolute error
         * */
        float mae(const float_vector& predictions, const float_vector& targets)
        {
            float result = 0;
            float_vector differences;
            std::transform(
                predictions.begin(), 
                predictions.end(), 
                targets.begin(),
                differences.begin(),
                [&](const auto& pred, const auto& target) {
                    return abs(pred - target);
                });
            auto diff_sum = std::accumulate(differences.begin(), differences.end(), 0.0);
            result = static_cast<float>(diff_sum / differences.size());
            return result;
        }

        /**
         * @brief Calculate MAE (mean absolute error) error for rotation component between predictions pose and ground truth pose 
         * @param[in] predictions Predicted rotation matrix  
         * @param[in] targets Ground truth rotation matrix
         * @return Mean absolute error
         * */
        float mae(const Eigen::MatrixXf& predictions, const Eigen::MatrixXf& targets) {
            return 0.0;
        }

        /**
         * @brief Convert rotation quaternion to matrix form 
         * @param[in] rotation Rotation quaternion  
         * @return Matrix of rotation
         * */
        Eigen::MatrixXf convertQuaternionToMatrix(PoseEstimator::Rotation rotation)
        {
            auto [w, x, y, z] = rotation;
            Eigen::MatrixXf result(3, 3);
            result(0, 0) = 1.0 - 2.0 * y * y - 2.0 * z * z;
            result(0, 1) = 2.0 * x * y - 2.0 * w * z;
            result(0, 2) = 2.0 * x * z + 2.0 * w * y;
            result(1, 0) = 2.0 * x * y + 2.0 * w * z;
            result(1, 1) = 1.0 - 2.0 * x * x - 2.0 * z * z;
            result(1, 2) = 2.0 * y * z - 2.0 * w * x;
            result(2, 0) = 2.0 * x * z - 2.0 * w * y;
            result(2, 0) = 2.0 * y * z + 2.0 * w * x;
            result(2, 0) = 1.0 - 2.0 * x * x - 2.0 * y * y;
            return result;
        }
    }

    PoseEstimator::PoseEstimator(const std::string& image_file_path, 
        const std::string& metadata_file_path, 
        const std::string& view_files_path) : 
            m_query_image_file(image_file_path), 
            m_query_metadata_file(metadata_file_path),
            m_view_files_path(view_files_path), 
            m_detector(SIFT::create(min_hessian)),
            m_matcher(cv::BFMatcher::create(cv::NORM_L2)),
            m_camera_matrix(Eigen::Array33f::Zero()) {

        m_view_images.reserve(MAX_VIEWS_NUMBER);
    }

    void PoseEstimator::estimate() {
        loadQueryImage();
        
        loadImageMetadata();
        m_camera_matrix(0, 0) = m_query_image_metadata.calibration_data.fx;
        m_camera_matrix(0, 2) = m_query_image_metadata.calibration_data.cx;
        m_camera_matrix(1, 1) = m_query_image_metadata.calibration_data.fy;
        m_camera_matrix(1, 2) = m_query_image_metadata.calibration_data.cy;
        m_camera_matrix(2, 2) = 1;
        
        loadViewImages();

        findImageDescriptors();

        match();
    }

    void PoseEstimator::loadImageMetadata() {
        // load query pose
        std::ifstream ifs(m_query_metadata_file);
        json json_data = json::parse(ifs);

        auto calibration_info = json_data.at("calibration");
        auto fx = static_cast<float>(calibration_info.at("fx"));
        auto fy = static_cast<float>(calibration_info.at("fy"));
        auto cx = static_cast<float>(calibration_info.at("cx"));
        auto cy = static_cast<float>(calibration_info.at("cy"));
        // load query pose
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
        for (size_t i = 0; i < 3; ++i)
        {
            translation.emplace_back(origin[i]);
        }
        PoseEstimator::TransformPose pose = {rotation, translation};

        PoseEstimator::CalibrationData calibration_data = {fx, fy, cx, cy};

        m_query_image_metadata = ImageMetadata { calibration_data, pose };
    }

    void PoseEstimator::loadQueryImage() {
        Mat image = imread(m_query_image_file, IMREAD_COLOR);
        m_query_image = ImageSample {.file_name = m_query_image_file, .image_data = image};
        auto image_size = image.size();
    }

    void PoseEstimator::loadViewImages() {
        path dir_path = m_view_files_path;

        for (const auto& file : directory_iterator(dir_path))
        {
            auto file_path = file.path();
            if (file_path.extension() == ".jpg" || file_path.extension() == ".png") {
                Mat image = imread(file_path, IMREAD_COLOR);
                m_view_images.emplace_back(ImageSample {.file_name = file_path.filename(), .image_data = image}); // view_image);
            }
        }
    }

    void PoseEstimator::findImageDescriptors() {
        // Find keypoints for query image
        m_detector->detectAndCompute(m_query_image.image_data, noArray(), m_query_image.keypoints, m_query_image.descriptors);

        for (auto &&view_img : m_view_images)
        {
            m_detector->detect(view_img.image_data, view_img.keypoints);
            m_detector->detectAndCompute(view_img.image_data, noArray(), view_img.keypoints, view_img.descriptors);
        }
    }
    
    void PoseEstimator::match() const {
        std::vector<view_matches_vector> views_matches;

        for (auto &&view_img : m_view_images)
        {
            view_matches_vector matches;
            m_matcher->match(view_img.descriptors, m_query_image.descriptors, matches);

            // reject weak matches
            double min_dist = 100.0;
            for (const auto& match: matches)
            {
                if (match.distance < min_dist)
                    min_dist = match.distance;
            }

            matches.erase(std::remove_if(matches.begin(),
                matches.end(), [&min_dist](const auto &match){
                    return (match.distance > 2 * min_dist);
                }), matches.end());

            views_matches.emplace_back(matches);
        }

        // TODO: calculate number of inliers using homography
        std::vector<int> views_matches_inliers;
        views_matches_inliers.reserve(m_view_images.size());

        for (size_t i = 0; i < m_view_images.size(); ++i)
        {
            auto view_image = m_view_images[i];
            Mat inliers;
            auto view_matches = views_matches[i];
            std::vector<cv::Point2d> matched_pts1, matched_pts2;
            for (auto& match : view_matches)
            {
                matched_pts1.emplace_back(view_image.keypoints[match.queryIdx].pt);
                matched_pts2.emplace_back(m_query_image.keypoints[match.trainIdx].pt);
            }
            Mat H = findHomography(matched_pts1, matched_pts2, cv::FM_RANSAC, 3, inliers);

            auto inliers_number = cv::sum(inliers)[0];
            views_matches_inliers.emplace_back(inliers_number);
        }

        // Find best match
        auto best_match_index = -1;
        auto max_inliers_number = 0;
        for (size_t i = 0; i < m_view_images.size(); ++i)
        {
            if (views_matches_inliers[i] > max_inliers_number) {
                max_inliers_number = views_matches_inliers[i];
                best_match_index = i;
            }
        }
    }
    
    void PoseEstimator::calculateTransformation() {}
    
    void PoseEstimator::getPoseError() {
        auto gt_translation = m_query_image_metadata.pose.translation;
        // w, x, y, z
        auto gt_rotation = m_query_image_metadata.pose.rotation;
        auto gt_rotation_matrix = convertQuaternionToMatrix(gt_rotation);

        auto translation_error = mae(gt_translation, m_result_pose_translation);
        auto rotation_error = mae(
            gt_rotation_matrix,
            m_result_pose_rotation
        );
    }

    auto PoseEstimator::getQueryImageKeypoints() const -> keypoints_vector {
        return m_query_image.keypoints;
    }

    auto PoseEstimator::getQueryImage() const -> ImageSample {
        return m_query_image;
    }

    auto PoseEstimator::getViewImages() const -> std::vector<ImageSample> {
        return m_view_images;
    }

    auto PoseEstimator::getQueryImageMetadata() const -> ImageMetadata {
        return m_query_image_metadata;
    }

}