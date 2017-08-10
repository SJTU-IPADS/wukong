# Wukong v0.8

Wukong, a distributed graph-based RDF store that leverages efficient graph exploration to provide highly concurrent and low-latency queries over large data sets.

### Feature Highlights

- High-performance and Scalable RDF Store

- Concurrent SPARQL query processing

- Enabling RDMA feature of InfiniBand

- Support evolving graphs (not included)

For more details on the Wukong see http://ipads.se.sjtu.edu.cn/projects/wukong.html, including new features, instructions, etc.


### License

Wukong is released under the [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0.html).
If you use Wukong in your research, please cite our paper:
   
    @inproceedings{Shi:osdi2016wukong,
     author = {Shi, Jiaxin and Yao, Youyang and Chen, Rong and Chen, Haibo and Li, Feifei},
     title = {Fast and Concurrent RDF Queries with RDMA-based Distributed Graph Exploration},
     booktitle = {12th USENIX Symposium on Operating Systems Design and Implementation},
     series = {OSDI '16},
     year = {2016},
     month = Nov,
     isbn = {978-1-931971-33-1},
     address = {GA},
     pages = {317--332},
     url = {https://www.usenix.org/conference/osdi16/technical-sessions/presentation/shi},
     publisher = {USENIX Association},
    }


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
    $sudo apt-get install gcc g++ build-essential cmake git libreadline6-dev


### Install Wukong on One Machine

Add the root path of Wukong (e.g., `/home/rchen/wukong`) to bash script (i.e., `~/.bashrc`).

    # Wukong configuration
    export WUKONG_ROOT=[/path/to/wukong]   


#### Install OpenMPI

    $cd  $WUKONG_ROOT/deps/
    $tar zxvf openmpi-1.6.5.tar.gz
    $mkdir openmpi-1.6.5-install
    $cd openmpi-1.6.5/
    $./configure --prefix=$WUKONG_ROOT/deps/openmpi-1.6.5-install
    $make all
    $make install


#### Install Boost

    $cd  $WUKONG_ROOT/deps/
    $tar jxvf boost_1_58_0.tar.bz2  
    $mkdir boost_1_58_0-install
    $cd boost_1_58_0/
    $./bootstrap.sh --prefix=../boost_1_58_0-install  

Add the following MPI configuration to `project-config.jam`

    # MPI configuration
    using mpi : $WUKONG_ROOT/deps/openmpi-1.6.5-install/bin/mpicc ;

    $./b2 install  


#### Install Intel TBB

    $cd $WUKONG_ROOT/deps/  
    $tar zxvf tbb44_20151115oss_src.tgz  
    $cd tbb44_20151115oss/
    $make

Add below settings to bash script (i.e., `~/.bashrc`).
 
    # Intel TBB configuration
    source $WUKONG_ROOT/deps/tbb44_20151115oss/build/[version]/tbbvars.sh

For example:

    source $WUKONG_ROOT/deps/tbb44_20151115oss/build/linux_intel64_gcc_cc4.8_libc2.19_kernel3.14.27_release/tbbvars.sh


#### Install ZeroMQ (http://zeromq.org/)

    $cd $WUKONG_ROOT/deps/
    $tar zxvf zeromq-4.0.5.tar.gz
    $mkdir zeromq-4.0.5-install
    $cd zeromq-4.0.5/
    $./configure --prefix=$WUKONG_ROOT/deps/zeromq-4.0.5-install/
    $make
    $make install
    $cd ..
    $cp zmq.hpp  zeromq-4.0.5-install/include/
    $cp zhelpers.hpp  zeromq-4.0.5-install/include/

Add below settings to bash script (i.e., `~/.bashrc`).
 
    ＃ ZeroMQ configuration
    export CPATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/include:$CPATH
    export LIBRARY_PATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/lib:$LD_LIBRARY_PATH


#### Install librdma (Optional)

    $cd $WUKONG_ROOT/deps/
    $tar zxvf librdma-1.0.0.tar.gz
    $cd librdma-1.0.0/
    $./configure --prefix=$WUKONG_ROOT/deps/librdma-1.0.0-install/
    $make
    $make install
    $cd ..

Add below settings to bash script (i.e., `~/.bashrc`).
 
    # librdma configuration
    export CPATH=$WUKONG_ROOT/deps/librdma-1.0.0-install/include:$CPATH
    export LIBRARY_PATH=$WUKONG_ROOT/deps/librdma-1.0.0-install/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$WUKONG_ROOT/deps/librdma-1.0.0-install/lib:$LD_LIBRARY_PATH


#### Install HDFS support (Optional)

We assume that Hadoop/HDFS has been installed on your cluster. The ENV variable for Hadoop should be set correctly.

    # Build hadoop.jar (assume that the Hadoop has been added on this machine) 
    $cd $WUKONG_ROOT/deps/hadoop/
    $./hadoop_deps.sh

Add below settings to bash script (i.e., `~/.bashrc`) according to the installation of Hadoop on your machine.

    # Haddop configuration
    export HADOOP_HOME=/usr/local/hadoop
    export PATH=$HADOOP_HOME/bin:$PATH
    export CPATH=$HADOOP_HOME/include:$CPATH

    # LibJVM configuration
    export LIBRARY_PATH=/usr/lib/jvm/default-java/jre/lib/amd64/server:$LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/lib/jvm/default-java/jre/lib/amd64/server:$LD_LIBRARY_PATH

    # LibHDFS configuration
    export LIBRARY_PATH=$HADOOP_HOME/lib/native:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native:$LD_LIBRARY_PATH

    export CLASSPATH=$WUKONG_ROOT/deps/hadoop/hadoop.jar

Enable the compile option in CMakeLists.txt  

    # Uncomment two lines below to enble HDFS support.
    add_definitions(-DHAS_HADOOP)
    target_link_libraries(wukong hdfs)


NOTE: if the `global_input_folder` start with `hdfs:`, then Wukong will read the files from HDFS.



### Copy Wukong Dependencies to All Machines

1) Setup password-less SSH between the master node and all other machines.

2) Create a file in `scripts` directory called `mpd.hosts` with the IP address of all machines.

For example:

    $cat ${WUKONG_ROOT}/scripts/mpd.hosts
    10.0.0.100
    10.0.0.101
    10.0.0.102

3) Run the following commands to copy Wukong dependencies to the rest of the machines:

    $cd ${WUKONG_ROOT}/scripts
    $./syncdeps.sh ../deps/dependencies mpd.hosts



# Compiling and Running

### Configure wukong

#### Enable/disable RDMA Feature

Currently, Wukong will enable RDMA feature by default, and suppose the driver has been well installed and configured. If you want to disable RDMA, you need manually comment `add_definitions(-DHAS_RDMA)` in `CMakeLists.txt`.


#### Enable/disable the support to versatile queries

To support versatile queries (e.g., ?S ?P ?O), you need manually uncomment `#add_definitions(-DVERSATILE)` in `CMakeLists.txt`. It should be noted that enabling this feature will use more main memory to store RDF graph.


### Build wukong 

    $cd ${WUKONG_ROOT}/scripts
    $./build.sh


### Synchronize all executable files (e.g., `build/wukong`) to all machines 

    $cd ${WUKONG_ROOT}/scripts
    $./sync.sh


### Running sever with a builtin local console for testing

    $cd ${WUKONG_ROOT}/scripts
    $./run.sh [#nodes]
    (e.g., ./run.sh 3)



# Prepare RDF Data

#### Install LUBM-generator (include a bug fix about linux paths)
    
    $cd ~;
    $wget http://swat.cse.lehigh.edu/projects/lubm/uba1.7.zip
    $wget http://swat.cse.lehigh.edu/projects/lubm/GeneratorLinuxFix.zip
    $unzip uba1.7.zip -d ./uba1.7
    $unzip GeneratorLinuxFix.zip -d .
    $cp Generator.java uba1.7/src/edu/lehigh/swat/bench/uba/Generator.java
    $cd uba1.7/src/
    $javac edu/lehigh/swat/bench/uba/Generator.java
    $cd ~
    $cp uba1.7/src/edu/lehigh/swat/bench/uba/*.class uba1.7/classes/edu/lehigh/swat/bench/uba/

#### Install Jena (Convert LUBM data format)
    
    $cd ~;
    $wget http://archive.apache.org/dist/jena/binaries/apache-jena-2.7.4.zip
    $unzip apache-jena-2.7.4.zip -d .
    $export JENA_HOME=~/apache-jena-2.7.4/

#### Generate LUBM-1 ( .nt format)
    
    $cd uba1.7
    $java -cp ./classes/ edu.lehigh.swat.bench.uba.Generator -univ 1 -onto http://swat.cse.lehigh.edu/onto/univ-bench.owl
    $find . -type f -name "University*.owl" -exec $JENA_HOME/bin/rdfcat -out N-TRIPLE -x {} >> uni0.nt \;
    # Now we get NT format of LUBM data
    # each row consists of S P O and '.'
    

#### Generate LUBM (wukong internal format)
    
    $cd ${WUKONG_ROOT}/datagen;
    $g++ -std=c++11 generate_data.cpp -o generate_data
    $mkdir lubm_raw_1
    $mv ~/uba1.7/uni0.nt lubm_raw_1/
    ./generate_data lubm_raw_1 id_lubm_1



