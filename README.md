# Dependence

### Boost+mpi

    cd /home/sjx/install/boost_1_58_0/  
    tar jxvf  boost_1_58_0.tar.bz2  
    cd /home/sjx/install/boost_1_58_0/boost_1_58_0  
    ./bootstrap.sh --prefix=/home/sjx/install/boost_1_58_0/boost_1_58_0-install  
add following lines in project-config.jam  

    using mpi ;  

install

    ./b2 install  

### intel-TBB


    cd /home/sjx/install  
    tar -xzf tbb44_20151115oss_src.tgz  
    cd tbb44_20151115oss;  
    make;  

add following lines in ~/.bashrc  

    source /home/sjx/install/tbb44_20151115oss/build/linux_intel64_gcc_cc4.8_libc2.19_kernel3.18.24+_release/tbbvars.sh

### zeroMQ+ cpp-binding
### RDMA

# Compile

1. copy wukong/ to all slaves  
2. modify CMakeLists.txt to set BOOST_ROOT and CMAKE_CXX_COMPILER
3. modify tools/sync.sh to set root_dir  
4. modify tools/mpd.hosts to set ip addresses of slaves  
5. compile and sync  


    cd tools;
    ./make.sh
    ./sync.sh

# Execution

    ./run.sh nmachine

# Input Data

TODO
