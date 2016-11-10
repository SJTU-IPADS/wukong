cmake_minimum_required(VERSION 2.8)
project (wukong-client)

set(ROOT $ENV{WUKONG_ROOT})

add_definitions(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(BOOST_LIB "${ROOT}/deps/boost_1_58_0-install/lib")
include_directories(include)
include_directories(${ROOT}/deps/boost_1_58_0-install/include)
file(GLOB SOURCES "src/*.cpp")

add_executable(wukong-client ${SOURCES} "test/example.cpp")
target_link_libraries(wukong-client zmq readline ${BOOST_LIB}/libboost_serialization.a)