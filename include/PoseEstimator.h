/**
 * @file PoseEstimator.h
 * @brief Class for pipeline of the alignment pose of a rigid object in a scene
 */

#include <vector>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/xfeatures2d.hpp"

#include <Eigen/Dense>

#include <nlohmann/json.hpp>


using namespace cv;

namespace cpp_practicing {
    
    // using std::vector<view_matches>;

    const int MAX_VIEWS_NUMBER = 20;

    /**
     * @brief Pipeline for camera pose estimation
     */
    class PoseEstimator
    {
    public:
        using string_vector = std::vector<std::string>;
        using float_vector = std::vector<float>;
        using view_matches_vector = std::vector<DMatch>;

        /**
         * @brief Struct for representing Rotation quaternion
         */
        struct Rotation
        {
            /**!< w parameter */
            float w;
            /**!< x parameter */
            float x;
            /**!< y parameter */
            float y;
            /**!< z parameter */
            float z;
        };

        /**
         * @brief Struct for representing camera calibration data
         */
        struct CalibrationData
        {
            /**!< focal length in x axis */
            float fx;
            /**!< focal length in y axis */
            float fy;
            /**!< x coordinate of optical center */
            float cx;
            /**!< y coordinate of optical center */
            float cy;
        };

        /**
         * @brief Struct for representing transformation matrix of view image
         */
        struct TransformPose
        {
            /**!< rotation matrix */
            Rotation rotation;
            /**!< translation vector */
            float_vector translation;
        };

        /**
         * @brief Struct for representing image metadata (intrinsic and extrinsic parameters)
         */
        struct ImageMetadata
        {
            /**!< camera calibration data */
            CalibrationData calibration_data;
            /**!< pose of query image */
            TransformPose pose;
        };

        /**
         * @brief Struct for representing single image sample
         */
        struct ImageSample
        {
            /**!< file name */
            std::string file_name;
            /**!< image matrix */
            Mat image_data;
            /**!< vector of keypoints */
            std::vector<cv::KeyPoint> keypoints;
            /**!< matrix of feature descriptors */
            Mat descriptors;
            /**!< number of inliers in match between the view image and the query image */
            int inliers_number;
        };
        
        PoseEstimator(const std::string& image_file_path, const std::string& metadata_file_path, const std::string& view_files_path);
        /**
         * @brief Start pipeline for the alignment pose of a rigid object in a scene 
         * */
        void estimate();

        /**
         * @brief Get keypoints for query image  
         * @return Vector of keypoints
         * */
        auto getQueryImageKeypoints() const -> keypoints_vector;

        /**
         * @brief Get query image  
         * @return Image sample
         * */
        auto getQueryImage() const -> ImageSample;

        /**
         * @brief Get view images 
         * @return Vector of view images
         * */
        auto getViewImages() const -> std::vector<ImageSample>;

        /**
         * @brief Get metadata for query image
         * @return Image metadata
         * */
        auto getQueryImageMetadata() const -> ImageMetadata;
        
    private:
        /// Query image data
        ImageSample query_image;
        /// vector of view images data
        std::vector<ImageSample> view_images;
        /// SIFT keypoint detector and feature descriptor
        Ptr<SIFT> detector;
        /// Descriptor matcher
        Ptr<DescriptorMatcher> matcher;
        // TransformPose result_pose;
        /// Result pose rotation
        Eigen::MatrixXf result_pose_rotation;
        /// Camera matrix
        Eigen::Array33f camera_matrix;
        /// Result pose translation
        float_vector result_pose_translation;
        /// Metadata of query image 
        ImageMetadata query_image_metadata;
        /// Query image file
        std::string m_query_image_file;
        /// Metadata of query image file
        std::string m_query_metadata_file;
        /// File path to view images 
        std::string m_view_files_path;
        /// min_hessian for SIFT feature descriptor
        int min_hessian = 400;
        
        /**
         * @brief Load image metadata (intrinsic and extrinsic parameters) from file 
         * */
        void loadImageMetadata();
        /**
         * @brief Load query image from disk 
         * */
        void loadQueryImage();
        /**
         * @brief Load view images from disk 
         * */
        void loadViewImages();
        /**
         * @brief Calculate feature descriptors for both query and all the view images 
         * */
        void findImageDescriptors();
        /**
         * @brief Perform match between the query image and all the view images 
         * */
        void match() const;

        /**
         * @brief Calculate pose transformation between the query image and the best matched view image 
         * */
        void calculateTransformation();
        /**
         * @brief Calculate pose error the view images 
         * */
        void getPoseError();
    };

} // namespace cpp_practicing