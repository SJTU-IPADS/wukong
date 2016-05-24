
# Dependence

export WUKONG_ROOT=/home/sjx/graph-query

### Boost+mpi

    cd  deps/
    tar jxvf  boost_1_58_0.tar.bz2  
    mkdir boost_1_58_0-install
    ./bootstrap.sh --prefix=../boost_1_58_0-install  

add following lines in project-config.jam  

    using mpi ;  

install

    ./b2 install  


### intel-TBB


    cd /home/sjx/install  
    tar -xzf tbb44_20151115oss_src.tgz  
    cd tbb44_20151115oss;  
    make;  

add following lines in ~/.bashrc (replace WUKONG_ROOT with real dirname)

    source ${WUKONG_ROOT}/deps/tbb44_20151115oss/build/linux_intel64_gcc_cc4.8_libc2.19_kernel3.18.24+_release/tbbvars.sh

### zeroMQ+ cpp-binding

    http://zeromq.org/

    cd  deps/
    tar -zxvf zeromq-4.0.5.tar.gz
    cd zeromq-4.0.5-install
    ./configure --prefix=${WUKONG_ROOT}/deps/zeromq-4.0.5-install/
    make
    make install
    cd ..
    cp zmq.hpp  zeromq-4.0.5-install/include/

add following lines in ~/.bashrc (replace WUKONG_ROOT with real dirname)   

    export CPATH=${WUKONG_ROOT}/deps/zeromq-4.0.5-install/include:$CPATH
    export LIBRARY_PATH=${WUKONG_ROOT}/deps/zeromq-4.0.5-install/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=${WUKONG_ROOT}/deps/zeromq-4.0.5-install/lib:$LD_LIBRARY_PATH

### RDMA

# Compile

1. copy wukong/ to all slaves  
2. modify CMakeLists.txt to set CMAKE_CXX_COMPILER
3. modify tools/sync.sh to set root_dir  
4. modify tools/mpd.hosts to set ip addresses of slaves  
5. compile and sync  


    cd tools;
    ./make.sh
    ./sync.sh

# Execution

    ./run.sh nmachine

# Input Data

1. If there is space at the raw_data, convert it to underline first

        cat raw_file | sed -e 's/ /_/g’ > convert_file
2. use generate_data to convert raw_data into id_data .

        ./generate_data lubm_raw_40/ id_lubm_40/
3. put id_data to NFS , and set the global_input_folder at config file

4. use str_normal_minimal if loading str_normal causes too much time
    grep "<http://www.Department0.University0.edu>" str_normal >> str_normal_minimal
