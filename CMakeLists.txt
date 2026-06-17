cmake_minimum_required(VERSION 3.16.0)

project(camera_pose_estimation_lib)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -O2 -lopencv_imgproc -lopencv_core")

# OpenCV

find_package(OpenCV REQUIRED)
set(OPENCV_ENABLE_NONFREE "Enable non-free algorithms" ON)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories("include")


add_library(PoseEstimator SHARED src/PoseEstimator.cpp)

add_executable(app src/main.cpp)

target_link_libraries(
        PoseEstimator PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
)

target_link_libraries(
        app PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
)