enable_testing()

find_package(GTest REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

add_executable(library_test LibraryTest.cpp)

target_include_directories(
        library_test PUBLIC
        ${CMAKE_BINARY_DIR}
        ${GTEST_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS})

add_dependencies(library_test PoseEstimator)

target_link_libraries(library_test PUBLIC 
    PoseEstimator 
    GTest::GTest 
    GTest::Main
    Eigen3::Eigen
    ${OpenCV_LIBS}
)
add_test(NAME LibraryTest COMMAND library_test)
