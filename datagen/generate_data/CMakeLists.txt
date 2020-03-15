project (generator)

## CMake version
cmake_minimum_required(VERSION 2.8)


## Set root directory
set(ROOT $ENV{WUKONG_ROOT})


## Use C++11 features
add_definitions(-std=c++11)

## Set include paths
include_directories(core)

## Source code
file(GLOB SOURCES "core/*.hpp")

add_executable(generator ${SOURCES} "core/generate_data.cpp")

## Build
target_link_libraries(generator tbb)
