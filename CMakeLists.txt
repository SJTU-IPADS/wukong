project (wukong)

## CMake version
cmake_minimum_required(VERSION 2.8)


## Set root directory of Wukong
set(ROOT $ENV{WUKONG_ROOT})


## Use C++11 features
add_definitions(-std=c++11)


## Set dependencies
set(CMAKE_CXX_COMPILER ${ROOT}/deps/openmpi-1.6.5-install/bin/mpic++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fopenmp -g")
set(BOOST_LIB "${ROOT}/deps/boost_1_58_0-install/lib")


## Set include paths
include_directories(deps/boost_1_58_0-install/include)
include_directories(core)
include_directories(utils)
include_directories(rdma_lib)


## Source code
file(GLOB SOURCES "core/*.hpp" "utils/*.hpp" "rdma_lib/*.hpp")
add_executable(wukong ${SOURCES} "core/wukong.cpp")


## Set options (cached)
## usage: cmake .. -DUSE_RDMA=OFF -DUSE_HADOOP=ON

#### RDMA
option (USE_RDMA "enable RDMA support" ON)
if(USE_RDMA)
  add_definitions(-DHAS_RDMA)
##  target_link_libraries(wukong rdma)
endif(USE_RDMA)

#### HDFS
option (USE_HADOOP "enable HDFS support" OFF)
if(USE_HADOOP)
  add_definitions(-DHAS_HADOOP)
  target_link_libraries(wukong hdfs)
endif(USE_HADOOP)

#### Dynamic GStore
option (USE_DYNAMIC_GSTORE "enable dynamic gstore" ON)
if(USE_DYNAMIC_GSTORE)
  add_definitions(-DDYNAMIC_GSTORE)
endif(USE_DYNAMIC_GSTORE)

#### Verstile queries (e.g., ?S ?P ?O)
option (USE_VERSATILE "support versatile queries" OFF)
if(USE_VERSATILE)
  add_definitions(-DVERSATILE)
endif(USE_VERSATILE)

#### 64-bit ID (32-bit ID by default)
option (USE_DTYPE_64BIT "use 64-bit ID" OFF)
if(USE_DTYPE_64BIT)
  add_definitions(-DDTYPE_64BIT)
endif(USE_DTYPE_64BIT)


## Build Wukong 
target_link_libraries(wukong zmq rt ibverbs tbb hwloc ${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a)

