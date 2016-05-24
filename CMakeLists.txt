cmake_minimum_required(VERSION 2.8)
project (wukong)

set(WUKONG_ROOT "/home/sjx/graph-query" )



ADD_DEFINITIONS( -std=c++11)
set(CMAKE_CXX_COMPILER /usr/bin/mpic++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(BOOST_LIB "${WUKONG_ROOT}/deps/boost_1_58_0-install/lib" )
include_directories(deps/boost_1_58_0-install/include)
include_directories(include)
include_directories(utils)
file(GLOB SOURCES "src/*.cpp")

add_executable(wukong ${SOURCES} "test/main.cpp")
target_link_libraries( wukong zmq rt ibverbs tbb ${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a)

# add_executable(test_rdma ${SOURCES} "test/test_rdma.cpp")
# target_link_libraries( test_rdma zmq rt ibverbs tbb ${BOOST_ROOT}/lib/libboost_mpi.a ${BOOST_ROOT}/lib/libboost_serialization.a)
