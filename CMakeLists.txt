cmake_minimum_required(VERSION 2.8)
project (wukong)

set(ROOT $ENV{WUKONG_ROOT})

ADD_DEFINITIONS(-std=c++11)
set(CMAKE_CXX_COMPILER /usr/bin/mpic++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(BOOST_LIB "${ROOT}/deps/boost_1_58_0-install/lib" )
include_directories(deps/boost_1_58_0-install/include)
include_directories(include)
include_directories(utils)
file(GLOB SOURCES "src/*.cpp")

## Wukong (RDMA-enbled)
add_executable(wukong ${SOURCES} "test/main.cpp")
target_link_libraries(wukong zmq rt ibverbs tbb ${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a)

## Wukong (RDMA-disabled)
# add_executable(wukong_zmq ${SOURCES} "test/main_zmq.cpp")
# target_link_libraries(wukong_zmq zmq rt tbb ${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a)
# target_compile_definitions(wukong_zmq PRIVATE USE_ZEROMQ=1)

## RDMA performance evaluation
# add_executable(wukong_test ${SOURCES} "test/test_rdma.cpp")
# target_link_libraries(wukong_test zmq rt ibverbs tbb ${BOOST_ROOT}/lib/libboost_mpi.a ${BOOST_ROOT}/lib/libboost_serialization.a)
