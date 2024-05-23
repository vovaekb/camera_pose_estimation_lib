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
    using json = nlohmann::json;

    namespace {

        /**
         * @brief Calculate MAE (mean absolute error) error for translation component between predictions pose and ground truth pose 
         * @param[in] predictions Predicted translation vector  
         * @param[in] targets Ground truth translation vector
         * @return Mean absolute error
         * */
        float mae(const std::vector<float>& predictions, const std::vector<float>& targets)
        {
            float result = 0;
            std::vector<float> differences;
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
            detector(SIFT::create(min_hessian)),
            matcher(cv::BFMatcher::create(cv::NORM_L2)),
            camera_matrix(Eigen::Array33f::Zero()) {
        // detector = SIFT::create(min_hessian);
        // matcher = cv::BFMatcher::create(cv::NORM_L2);

        view_images.reserve(MAX_VIEWS_NUMBER);
    }

    void PoseEstimator::estimate() {
        std::cout << "run pose estimation" << std::endl;

        loadQueryImage();
        
        loadImageMetadata();
        camera_matrix(0, 0) = query_image_metadata.calibration_data.fx;
        camera_matrix(0, 2) = query_image_metadata.calibration_data.cx;
        camera_matrix(1, 1) = query_image_metadata.calibration_data.fy;
        camera_matrix(1, 2) = query_image_metadata.calibration_data.cy;
        camera_matrix(2, 2) = 1;
        
        loadViewImages();

        findImageDescriptors();

        match();
    }

    void PoseEstimator::loadImageMetadata() {
        std::cout << "load image metadata" << std::endl;
        // load query pose
        std::ifstream ifs(m_query_metadata_file);
        json json_data = json::parse(ifs);

        std::cout << "line 96" << std::endl;
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
        // std::cout << "rotation: " << rotation.w << ", " << rotation.x << " " << std::endl;
        
        std::vector<float> translation;
        PoseEstimator::TransformPose pose = {rotation, translation}; 

        PoseEstimator::CalibrationData calibration_data = {fx, fy, cx, cy};

        query_image_metadata = ImageMetadata { calibration_data, pose };
    }

    void PoseEstimator::loadQueryImage() {
        std::cout << "load query image" << std::endl;

        Mat image = imread(m_query_image_file, IMREAD_COLOR);
        query_image = ImageSample {.file_name = m_query_image_file, .image_data = image};
        auto image_size = image.size();
        // std::cout << "image_size: " << image_size.width << " x " << image_size.height << std::endl;
    }

    void PoseEstimator::loadViewImages() {
        std::cout << "load view images" << std::endl;

        path dir_path = m_view_files_path;

        for (auto& file : directory_iterator(dir_path))
        {
            auto file_path = file.path();
            auto file_name = file_path.filename();
            // std::cout << "file " << file_path.filename() << ", " << file_path.extension() << std::endl;
            if (file_path.extension() == ".jpg" || file_path.extension() == ".png") {
                Mat image = imread(file_path, IMREAD_COLOR);
                view_images.emplace_back(ImageSample {.file_name = file_name, .image_data = image}); // view_image);

            }
        }

    }

    void PoseEstimator::findImageDescriptors() {
        // Find keypoints for query image
        // std::cout << "Find keypoints for query image" << std::endl;
        detector->detectAndCompute(query_image.image_data, noArray(), query_image.keypoints, query_image.descriptors);

        for (auto &&view_img : view_images)
        {
            detector->detect(view_img.image_data, view_img.keypoints);
            detector->detectAndCompute(view_img.image_data, noArray(), view_img.keypoints, view_img.descriptors);

        }

    }
    
    void PoseEstimator::match() const {
        std::cout << "Match keypoints for query image" << std::endl;

        std::vector<view_matches_vector> views_matches;

        for (auto &&view_img : view_images)
        {
            view_matches_vector matches;
            matcher->match(view_img.descriptors, query_image.descriptors, matches);
            // cout << "matches number: " << knn_matches.size() << endl;
            std::cout << "matches number: " << matches.size() << std::endl;

            // reject weak matches
            double min_dist = 100.0;
            for (const auto& match: matches)
            {
                if (match.distance < min_dist)
                    min_dist = match.distance;
            }
            // std::cout << "min_dist: " << min_dist << std::endl;

            matches.erase(std::remove_if(matches.begin(),
                matches.end(), [&min_dist](const auto &match){
                    return (match.distance > 2 * min_dist);
                }), matches.end());

            std::cout << "filtered matches number: " << matches.size() << std::endl;

            views_matches.emplace_back(matches);

        }

        // TODO: calculate number of inliers using homography
        std::vector<int> views_matches_inliers;
        views_matches_inliers.reserve(view_images.size());

        for (size_t i = 0; i < view_images.size(); ++i)
        {
            auto view_image = view_images[i];
            Mat inliers;
            auto view_matches = views_matches[i];
            std::vector<cv::Point2d> matched_pts1, matched_pts2;
            for (auto& match : view_matches)
            {
                matched_pts1.emplace_back(view_image.keypoints[match.queryIdx].pt);
                matched_pts2.emplace_back(query_image.keypoints[match.trainIdx].pt);
            }
            Mat H = findHomography(matched_pts1, matched_pts2, cv::FM_RANSAC, 3, inliers);

            auto inliers_number = cv::sum(inliers)[0];
            views_matches_inliers.emplace_back(inliers_number);
            std::cout << "inliers number: " << cv::sum(inliers)[0] << std::endl;

        }
        
        // Test loadImageMetadata method
        // loadImageMetadata();
    }
    
    void PoseEstimator::calculateTransformation() {}
    
    void PoseEstimator::getPoseError() {
        auto gt_translation = query_image_metadata.pose.translation;
        // // gt_t = np.array(gt_origin)
        // w, x, y, z
        auto gt_rotation = query_image_metadata.pose.rotation;
        auto gt_rotation_matrix = convertQuaternionToMatrix(gt_rotation);

        //auto prediction_translation = result_pose.rotation;
        auto translation_error = mae(gt_translation, result_pose_translation);
        // // print('pose error: ', t_error)
        auto rotation_error = mae(
            gt_rotation_matrix,
            result_pose_rotation // result_pose.rotation
        );
    }

}