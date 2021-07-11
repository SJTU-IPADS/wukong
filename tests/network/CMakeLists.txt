project (tests)

## CMake version
cmake_minimum_required(VERSION 2.8)


## Set root directory of Wukong
set(ROOT $ENV{WUKONG_ROOT})


## Use C++11 features
add_definitions(-std=c++11)


## Set dependencies
set(CMAKE_CXX_COMPILER ${ROOT}/deps/openmpi-1.6.5-install/bin/mpic++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp -lboost_system")
set(BOOST_LIB "${ROOT}/deps/boost_1_67_0-install/lib")

## Set include paths
include_directories(${ROOT}/deps/boost_1_67_0-install/include)
include_directories(${ROOT}/core)
include_directories(${ROOT}/utils)
include_directories(tests/network)

## Source code
file(GLOB SOURCES "core/*.hpp" "utils/*.hpp" "tests/network/*.hpp")
add_executable(network ${SOURCES} "network.cpp")

# Build Wukong 
target_link_libraries(network nanomsg zmq rt ibverbs tbb hwloc ${BOOST_LIB}/libboost_system.a ${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a ${BOOST_LIB}/libboost_program_options.a)


