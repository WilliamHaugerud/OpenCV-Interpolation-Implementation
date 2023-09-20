cmake_minimum_required(VERSION 3.0.0)
project(OPENCV_TEST VERSION 0.1.0 LANGUAGES C CXX)
set (CMAKE_CXX_STANDARD 17)
include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(OPENCV_TEST main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(OPENCV_TEST ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
