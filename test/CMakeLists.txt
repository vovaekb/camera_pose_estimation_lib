enable_testing()

add_executable(library_test LibraryTest.cpp ${Test_SOURCES})

target_include_directories(
	library_test PUBLIC
	${CMAKE_BINARY_DIR})

add_dependencies(library_test ${PROJECT_NAME})

target_link_libraries(library_test PUBLIC ${PROJECT_NAME})
