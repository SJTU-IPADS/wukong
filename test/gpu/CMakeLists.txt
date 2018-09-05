project (gputest)

## CMake version
cmake_minimum_required(VERSION 2.8)

## Set root directory of Wukong
set(ROOT $ENV{WUKONG_ROOT})

## Use C++11 features
add_definitions(-std=c++11)

## Set dependencies
set(CMAKE_CXX_COMPILER ${ROOT}/deps/openmpi-1.6.5-install/bin/mpic++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(BOOST_LIB "${ROOT}/deps/boost_1_67_0-install/lib")

## Set include paths
include_directories(${ROOT}/deps/boost_1_67_0-install/include)
include_directories(${ROOT}/core)
include_directories(${ROOT}/utils)
include_directories(${ROOT}/test/gpu)
include_directories(${ROOT}/rdma_lib)

## Source code
file(GLOB SOURCES "core/*.hpp" "utils/*.hpp" "rdma_lib/*.hpp" "test/gpu/*.hpp")
add_executable(gputest ${SOURCES} "test-adaptor.cpp")

## Set options (cached)
## usage: cmake .. -DUSE_RDMA=OFF -DUSE_HADOOP=ON

#### RDMA
option (USE_RDMA "enable RDMA support" ON)
if(USE_RDMA)
  add_definitions(-DHAS_RDMA)
endif(USE_RDMA)

#### GPU (NOT YET READY)
option (USE_GPU "enable GPU support" ON)
if(USE_GPU)
  add_definitions(-DUSE_GPU)
  find_package(CUDA REQUIRED)
  set(WUKONG_LIBS ${WUKONG_LIBS} cudart)
endif(USE_GPU)

set(WUKONG_LIBS ${WUKONG_LIBS} zmq rt ibverbs tbb hwloc)

## Build Wukong
target_link_libraries(gputest ${WUKONG_LIBS} ${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a)
