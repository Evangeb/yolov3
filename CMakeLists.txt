cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(darknet)

find_package(Torch REQUIRED)

set (CMAKE_CXX_STANDARD 11)
include_directories(include)
file(GLOB SOURCES "src/*.cpp")
add_executable(darknet ${SOURCES})
target_link_libraries(darknet "${TORCH_LIBRARIES}")
set(EXECUTABLE_OUTPUT_PATH ..)
