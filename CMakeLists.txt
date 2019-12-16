project (wukong)

## CMake version
cmake_minimum_required(VERSION 2.8)


## Set root directory of Wukong
set(ROOT $ENV{WUKONG_ROOT})


## Use C++11 features
add_definitions(-std=c++11)


## Set dependencies
set(CMAKE_CXX_COMPILER ${ROOT}/deps/openmpi-1.6.5-install/bin/mpic++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fopenmp")
set(BOOST_LIB "${ROOT}/deps/boost_1_67_0-install/lib")

## Set include paths
include_directories(deps/boost_1_67_0-install/include)
include_directories(core)
include_directories(utils)
include_directories(rdma_lib)
include_directories(deps/googletest/googletest/include)

## Add sub-directories
add_subdirectory(${ROOT}/deps/googletest)


## Source code
file(GLOB SOURCES "core/*.hpp" "utils/*.hpp" "rdma_lib/*.hpp")


## Set options (cached)
## usage: cmake .. -DUSE_RDMA=OFF -DUSE_HADOOP=ON

#### RDMA
option (USE_RDMA "enable RDMA support" ON)
if(USE_RDMA)
  add_definitions(-DHAS_RDMA)
  set(WUKONG_LIBS ${WUKONG_LIBS} ibverbs)
endif(USE_RDMA)

#### GPU (NOT YET READY)
option (USE_GPU "enable GPU support" OFF)
if(USE_GPU)
  add_definitions(-DUSE_GPU)
  find_package(CUDA REQUIRED)

  ## CUDA configs
  set(CUDA_NVCC_FLAGS -arch=sm_35;-std=c++11)
  set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(WUKONG_LIBS ${WUKONG_LIBS} cudart)
  cuda_add_executable(wukong ${SOURCES} "core/wukong.cpp" "core/gpu/gpu_hash.cu")

else(USE_GPU)
  add_executable(wukong ${SOURCES} "core/wukong.cpp")

endif(USE_GPU)

#### HDFS
option (USE_HADOOP "enable HDFS support" OFF)
if(USE_HADOOP)
  add_definitions(-DHAS_HADOOP)
  target_link_libraries(wukong hdfs)
  set(WUKONG_LIBS ${WUKONG_LIBS} hdfs)
endif(USE_HADOOP)

#### JEMALLOC
## Currently, all malloc APIs(e.g. malloc/free, new/delete) are jemalloc versions by default.
## To change it, go to deps/deps.sh, modify configuration.
target_link_libraries(wukong jemalloc)
set(WUKONG_LIBS ${WUKONG_LIBS} jemalloc)

option (USE_JEMALLOC "enable jemalloc support" ON)
if(USE_JEMALLOC)
  add_definitions(-DUSE_JEMALLOC)
endif(USE_JEMALLOC)

#### Dynamic GStore
option (USE_DYNAMIC_GSTORE "enable dynamic gstore" OFF)
if(USE_DYNAMIC_GSTORE)
  add_definitions(-DDYNAMIC_GSTORE)
endif(USE_DYNAMIC_GSTORE)

#### Verstile queries (e.g., ?S ?P ?O)
option (USE_VERSATILE "support versatile queries" ON)
if(USE_VERSATILE)
  add_definitions(-DVERSATILE)
endif(USE_VERSATILE)

#### 64-bit ID (32-bit ID by default)
option (USE_DTYPE_64BIT "use 64-bit ID" OFF)
if(USE_DTYPE_64BIT)
  add_definitions(-DDTYPE_64BIT)
endif(USE_DTYPE_64BIT)


set(WUKONG_LIBS ${WUKONG_LIBS} zmq rt tbb hwloc)

## Build Wukong
target_link_libraries(wukong ${WUKONG_LIBS} ${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a ${BOOST_LIB}/libboost_program_options.a)

file(GLOB TS  "${ROOT}/tests/*.cc")
add_executable(coretest ${TS})
target_link_libraries(coretest gtest gtest_main ${WUKONG_LIBS} ${BOOST_LIB}/libboost_mpi.a ${BOOST_LIB}/libboost_serialization.a ${BOOST_LIB}/libboost_program_options.a)

## tests
enable_testing()

add_test(NAME test COMMAND coretest)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose DEPENDS coretest)


