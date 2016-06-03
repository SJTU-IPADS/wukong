
# Wukong

Wukong, a distributed graph-based RDF store that leverages efficient graph exploration to provide highly concurrent and low-latency queries over large data sets.

### Feature Highlights

- High-performance and Scalable

- Support RDMA

- xx


### License

GraphLab PowerGraph is released under the Apache 2 license.

If you use GraphLab PowerGraph in your research, please cite our paper: xx


### Building

The current version of Wukong was tested on Ubuntu Linux 64-bit 14.04.
It requires a 64-bit operating system.




# Dependencies

To simplify installation, Wukong currently downloads and builds most of its required dependencies using CMake’s External Project feature. This also means the first build could take a long time.

There are however, a few dependencies which must be manually satisfied.

- g++: Required for compiling Wukong.
- *nix build tools (e.g., patch, make)
- MPICH2: Required for running Wukong distributed.

### Satisfying Dependencies on Ubuntu

All the dependencies can be satisfied from the repository:

    $sudo apt-get update
    $sudo apt-get install gcc g++ build-essential libopenmpi-dev openmpi-bin cmake git

### Install Wukong on One Machine

Add the root path of Wukong (e.g., `/home/rchen/wukong`) to bash script (i.e., `~/.bashrc`).

    # Wukong configuration
    export WUKONG_ROOT=[/path/to/wukong]   

#### Install Boost

    $cd  ${WUKONG_ROOT}/deps/
    $tar jxvf boost_1_58_0.tar.bz2  
    $mkdir boost_1_58_0-install
    $cd boost_1_58_0/
    $./bootstrap.sh --prefix=../boost_1_58_0-install  

Add the following MPI configuration to `project-config.jam`

    # MPI configuration
    using mpi ;  

    $./b2 install  


#### Install Intel TBB

    $cd ${WUKONG_ROOT}/deps/  
    $tar zxvf tbb44_20151115oss_src.tgz  
    $cd tbb44_20151115oss/
    $make

Add below settings to bash script (i.e., `~/.bashrc`).
 
    # Intel TBB configuration
    source $WUKONG_ROOT/deps/tbb44_20151115oss/build/[version]/tbbvars.sh

For example:

	source $WUKONG_ROOT/deps/tbb44_20151115oss/build/linux_intel64_gcc_cc4.8_libc2.19_kernel3.14.27_release/tbbvars.sh


#### Install ZeroMQ (http://zeromq.org/)

    $cd ${WUKONG_ROOT}/deps/
    $tar zxvf zeromq-4.0.5.tar.gz
    $mkdir zeromq-4.0.5-install
    $cd zeromq-4.0.5/
    $./configure --prefix=$WUKONG_ROOT/deps/zeromq-4.0.5-install/
    $make
    $make install
    $cd ..
    $cp zmq.hpp  zeromq-4.0.5-install/include/

Add below settings to bash script (i.e., `~/.bashrc`).
 
    ＃ ZeroMQ configuration
    export CPATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/include:$CPATH
    export LIBRARY_PATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/lib:$LD_LIBRARY_PATH


### Copy Wukong Dependencies to All Machines

1) Setup password-less SSH between the master node and all other machines.

2) Create a file in `tools` directory called `mpd.hosts` with the IP address of all machines.

For example:

    $cat ${WUKONG_ROOT}/tools/mpd.hosts
    10.0.0.100
    10.0.0.101
    10.0.0.102

3) Run the following commands to copy Wukong dependencies to the rest of the machines:

	$cd ${WUKONG_ROOT}/tools
	$./syncdeps.sh ../deps/dependencies mpd.hosts



# Compiling and Running

1) Modify CMakeLists.txt to set CMAKE_CXX_COMPILER (e.g., `/usr/bin/mpic++`)

    set(CMAKE_CXX_COMPILER /usr/bin/mpic++)

#### Enable/disable RDMA Feature

Currently, Wukong will enable RDMA feature by default, and suppose the driver has been well installed and configured. If you want to disable RDMA, you need manually modify `CMakeLists.txt` to compile and build `wukong-zmq` instead of `wukong`.

2) Build wukong 

    $cd ${WUKONG_ROOT}/tools
    $./build.sh

Synchronize all executable files (e.g., `build/wukong`) to all machines 

    $cd ${WUKONG_ROOT}/tools
    $./sync.sh

Running sever and a naive client console  

    $cd ${WUKONG_ROOT}/tools
    $./run.sh [#nodes]
    (e.g., ./run.sh 3)



# Prepare RDF Data

If there is space at the raw_data, convert it to underline first

    $cat raw_file | sed -e 's/ /_/g’ > convert_file


Use generate_data.cpp to convert raw_data into id_data

    $cd ${WUKONG_ROOT}/generate
    $g++ -std=c++11 generate_data.cpp -o generate_data
    $./generate_data lubm_raw_40 id_lubm_40


Put `id_data` to distributed storage (e.g., NFS) , and set the `global_input_folder` at `tools/config`


Create a file (`str_normal_minimal`) with minimal id mappings when loading `str_normal` is too lengthy

    $grep "<http://www.Department0.University0.edu>" str_normal >> str_normal_minimal
    ...

