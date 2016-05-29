
# Dependence

"add the root path of wukong in ~/.bashrc"

# Wukong configuration
export WUKONG_ROOT=/path/to/wukong
e.g. export WUKONG_ROOT=/home/rchen/wukong


### Boost

prepare boost

    cd  deps/
    tar jxvf boost_1_58_0.tar.bz2  
    mkdir boost_1_58_0-install
    cd boost_1_58_0/
    ./bootstrap.sh --prefix=../boost_1_58_0-install  

configure boost

    "add the following MPI configuration to project-config.jam"

    # MPI configuration
    using mpi ;  

install boost

    ./b2 install  


### Intel TBB

make and install TBB

    cd deps/  
    tar zxvf tbb44_20151115oss_src.tgz  
    cd tbb44_20151115oss/
    make

configure TBB

    "add following lines in ~/.bashrc"

    # Intel TBB configuration
    source $WUKONG_ROOT/deps/tbb44_20151115oss/build/linux_intel64_gcc_cc4.8_libc2.19_kernel3.14.27_release/tbbvars.sh


### ZeroMQ (http://zeromq.org/)

build and install zeroMQ

    cd  deps/
    tar zxvf zeromq-4.0.5.tar.gz
    mkdir zeromq-4.0.5-install
    cd zeromq-4.0.5/
    ./configure --prefix=$WUKONG_ROOT/deps/zeromq-4.0.5-install/
    make
    make install
    cd ..
    cp zmq.hpp  zeromq-4.0.5-install/include/

configure ZeroMQ

    "add following lines in ~/.bashrc"
 
    export CPATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/include:$CPATH
    export LIBRARY_PATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/lib:$LD_LIBRARY_PATH



# Compile

1. modify CMakeLists.txt to set CMAKE_CXX_COMPILER (e.g., /usr/bin/mpic++)

2. build wukong 
    cd tools/
    ./make.sh

3. modify tools/mpd.hosts to set ip addresses of slaves  

    e.g.,
    10.0.0.100
    10.0.0.101
    10.0.0.102

4. sync wukong/ to all slaves 

    cd tools/
    ./sync.sh


# Execution

    ./run.sh [#nodes]
    e.g., ./run.sh 6



# Input Data

1. if there is space at the raw_data, convert it to underline first

    cat raw_file | sed -e 's/ /_/g’ > convert_file

2. use generate_data.cpp to convert raw_data into id_data

    ./generate_data lubm_raw_40/ id_lubm_40/

3. put id_data to distributed storage (e.g., NFS) , and set the global_input_folder at tools/config

4. use str_normal_minimal if loading str_normal causes too much time
    e.g., grep "<http://www.Department0.University0.edu>" str_normal >> str_normal_minimal

