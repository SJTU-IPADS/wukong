project (wukong)

## We require the most recent version of cmake and automatically
## install the correct version when using the cmake lists
cmake_minimum_required(VERSION 2.8)

## Determine where wukong source code are defined
set(ROOT $ENV{WUKONG_ROOT})

## Use C++11 features
add_definitions(-std=c++11)

set(CMAKE_CXX_COMPILER /usr/bin/mpic++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(BOOST_LIB "${ROOT}/deps/boost_1_58_0-install/lib" )

## Uncomment two lines below to enble HDFS support.
#add_definitions(-DHAS_HADOOP)
#target_link_libraries(wukong hdfs)

## Uncomment the line below to enable RDMA support.
add_definitions(-DHAS_RDMA)

## Uncomment the line below to allow the variable at predicate.
#add_definitions(-DVAR_PREDICATE)

## Set include path
include_directories(deps/boost_1_58_0-install/include)
include_directories(core)
include_directories(utils)

file(GLOB SOURCES "core/*.hpp" "utils/*.hpp")


## Wukong
add_executable(wukong ${SOURCES} "core/wukong.cpp")
target_link_libraries(wukong zmq rt ibverbs tbb ${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a)

