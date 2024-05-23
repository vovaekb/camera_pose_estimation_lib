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

# Conan

message( "CMAKE_BINARY_DIR : " ${CMAKE_BINARY_DIR} )
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)

conan_basic_setup(TARGETS)

include_directories(${CONAN_INCLUDE_DIRS} "include")

# add source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(
        ${PROJECT_NAME} PUBLIC
        ${OpenCV_LIBS}
        Eigen3::Eigen
        CONAN_PKG::nlohmann_json
) 

# add tests
add_subdirectory(test)