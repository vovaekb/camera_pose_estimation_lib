cmake_minimum_required(VERSION 3.16.0)

project(camera_pose_estimation_lib)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenCV
find_package(OpenCV REQUIRED)

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(fmt CONFIG REQUIRED)

# Header files
include_directories(include)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

add_library(PoseEstimator SHARED src/PoseEstimator.cpp)
#target_compile_options(PoseEstimator PRIVATE -Wall -Wextra -Werror -O2)
target_link_libraries(PoseEstimator PRIVATE ${OpenCV_LIBS} Eigen3::Eigen fmt::fmt)

add_executable(app src/main.cpp)
#target_compile_options(app PRIVATE -Wall -Wextra -Werror -O2)
target_link_libraries(app PRIVATE PoseEstimator ${OpenCV_LIBS} Eigen3::Eigen fmt::fmt)

#add_subdirectory(test)