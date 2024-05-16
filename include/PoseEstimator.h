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
// using Mat = cv::Mat;

namespace cpp_practicing {
    
    // using std::vector<view_matches>;

    const int MAX_VIEWS_NUMBER = 20;

    class PoseEstimator
    {
    public:
        using string_vector = std::vector<std::string>;
        using view_matches_vector = std::vector<DMatch>;

        struct Rotation
        {
            float w;
            float x;
            float y;
            float z;

            // Rotation& operator=(const Rotation& other) {
            //     w = other.w;
            //     x = other.x;
            //     y = other.y;
            //     z = other.z;

            //     return *this;
            // }
        };

        struct CalibrationData
        {
            float fx;
            float fy;
            float cx;
            float cy;

            // CalibrationData& operator=(const CalibrationData& other) {
            //     fx = other.fx;
            //     fy = other.fy;
            //     cx = other.cx;
            //     cy = other.cy;

            //     return *this;
            // }
        };

        struct TransformPose
        {
            Rotation rotation;
            std::vector<float> translation;

            // TransformPose& operator=(const TransformPose& other) {
            //     rotation = other.rotation;
            //     // std::copy()

            //     return *this;
            // }
        };

        struct ImageMetadata
        {
            CalibrationData calibration_data;
            TransformPose pose;
        };

        struct ImageSample
        {
            std::string file_name;
            Mat image_data;
            std::vector<cv::KeyPoint> keypoints;
            Mat descriptors;
            int inliers_number;
        };
        
        PoseEstimator(const std::string& image_file_path, const std::string& metadata_file_path, const std::string& view_files_path);
        void estimate();
        
    private:
        ImageSample query_image;
        std::vector<ImageSample> view_images;
        Ptr<SIFT> detector;
        Ptr<DescriptorMatcher> matcher;
        // TransformPose result_pose;
        Eigen::MatrixXf result_pose_rotation;
        Eigen::Array33f camera_matrix;
        std::vector<float> result_pose_translation;
        ImageMetadata query_image_metadata;
        std::string m_query_image_file;
        std::string m_query_metadata_file;
        std::string m_view_files_path;
        int min_hessian = 400;
        
        void loadImageMetadata();
        void loadQueryImage();
        void loadViewImages();
        void findImageDescriptors();
        void match() const;
        void matchTwoImages() const;
        void calculateTransformation();
        void getPoseError();
    };

} // namespace cpp_practicing