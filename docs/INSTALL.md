# Installing Wukong

## Table of Contents

* [Installing dependencies](#dep)
* [Building and running](#run)
* [Preparing RDF datasets](#data)


<a name="dep"></a>
## Installing dependencies

> Note: the current version of Wukong was tested on Ubuntu Linux 64-bit 14.04.
It requires a 64-bit operating system.

### Install Wukong dependecies on one of your cluster machines

##### Step 0: *Install build tools and git*

```bash
$sudo apt-get update
$sudo apt-get install gcc g++ build-essential cmake git libreadline6-dev wget
```


##### Step 1: *Download Wukong source code*

```bash
$git clone https://github.com/realstolz/wukong.git
```

Add the root path of Wukong (e.g., `/home/rchen/wukong`) to bash script (i.e., `~/.bashrc`).

```bash
# Wukong configuration
export WUKONG_ROOT=[/path/to/wukong]   
```

##### Step 2: *Install OpenMPI*

```bash
$cd  $WUKONG_ROOT/deps/
$tar zxvf openmpi-1.6.5.tar.gz
$mkdir openmpi-1.6.5-install
$cd openmpi-1.6.5/
$./configure --prefix=$WUKONG_ROOT/deps/openmpi-1.6.5-install
$make all
$make install
```


##### Step 3: *Install Boost*

```bash
$cd  $WUKONG_ROOT/deps/
$tar jxvf boost_1_58_0.tar.bz2  
$mkdir boost_1_58_0-install
$cd boost_1_58_0/
$./bootstrap.sh --prefix=../boost_1_58_0-install  
```

Add the following MPI configuration to `project-config.jam`

```bash
# MPI configuration
using mpi : $WUKONG_ROOT/deps/openmpi-1.6.5-install/bin/mpicc ;
```

```bash
$./b2 install  
```


##### Step 4: *Install Intel Threading Building Blocks (TBB)*

```bash
$cd $WUKONG_ROOT/deps/  
$tar zxvf tbb44_20151115oss_src.tgz  
$cd tbb44_20151115oss/
$make
```

Add below settings to bash script (i.e., `~/.bashrc`).

```bash
# Intel TBB configuration
source $WUKONG_ROOT/deps/tbb44_20151115oss/build/[version]/tbbvars.sh
```

For example: `$WUKONG_ROOT/deps/tbb44_20151115oss/build/linux_intel64_gcc_cc4.8_libc2.19_kernel3.14.27_release/tbbvars.sh`


##### Step 5: *Install ZeroMQ (http://zeromq.org/)*

```bash
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
```

Add below settings to bash script (i.e., `~/.bashrc`).

```bash
# ZeroMQ configuration
export CPATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/include:$CPATH
export LIBRARY_PATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/lib:$LD_LIBRARY_PATH
```


##### Step 6: *Install Portable Hardware Locality (hwloc)*

```bash
$cd $WUKONG_ROOT/deps/
$tar zxvf hwloc-1.11.7.tar.gz
$cd hwloc-1.11.7/
$./configure --prefix=$WUKONG_ROOT/deps/hwloc-1.11.7-install/
$make
$make install
```

Add below settings to bash script (i.e., `~/.bashrc`).

```bash
# hwloc configuration
export PATH=$WUKONG_ROOT/deps/hwloc-1.11.7-install/bin:$PATH
export CPATH=$WUKONG_ROOT/deps/hwloc-1.11.7-install/include:$CPATH
export LIBRARY_PATH=$WUKONG_ROOT/deps/hwloc-1.11.7-install/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$WUKONG_ROOT/deps/hwloc-1.11.7-install/lib:$LD_LIBRARY_PATH
```


##### Step 7(optional): *Install librdma*

```bash
$cd $WUKONG_ROOT/deps/
$tar zxvf librdma-1.0.0.tar.gz
$cd librdma-1.0.0/
$./configure --prefix=$WUKONG_ROOT/deps/librdma-1.0.0-install/
$make
$make install
```

Add below settings to bash script (i.e., `~/.bashrc`).

```bash
# librdma configuration
export CPATH=$WUKONG_ROOT/deps/librdma-1.0.0-install/include:$CPATH
export LIBRARY_PATH=$WUKONG_ROOT/deps/librdma-1.0.0-install/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$WUKONG_ROOT/deps/librdma-1.0.0-install/lib:$LD_LIBRARY_PATH
```


##### Step 8(optional): *Install HDFS support*

We assume that Hadoop/HDFS has been installed on your cluster. The ENV variable for Hadoop should be set correctly.

```bash
# Build hadoop.jar (assume that the Hadoop has been added on this machine) 
$cd $WUKONG_ROOT/deps/hadoop/
$./hadoop_deps.sh
```

Add below settings to bash script (i.e., `~/.bashrc`) according to the installation of Hadoop on your machine.

```bash
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
```

Enable the compile option in `CMakeLists.txt`  

```cmake
# Uncomment two lines below to enble HDFS support
add_definitions(-DHAS_HADOOP)
target_link_libraries(wukong hdfs)
```

>Note: if the `global_input_folder` start with `hdfs:`, then Wukong will read the files from HDFS.


### Copy Wukong dependencies to all machines.

1) Setup password-less SSH between the master node and all other machines.

Verify it is possible to ssh without password between any pairs of machines. These [instructions](http://www.linuxproblem.org/art_9.html) explain how to setup ssh without passswords.

Before proceeding, verify that this is setup correctly; check that the following connects to the remote machine without prompting for a password:

```bash
$ssh node1.some.domain
# from node0.some.domain
```

2) Edit `mpd.hosts` with the hostname (IP address) of all machines in your cluster. For example:

```bash
$cat ${WUKONG_ROOT}/scripts/mpd.hosts
node0.some.domain
node1.some.domain
node2.some.domain
```

3) Run the following commands to copy necessities to the rest machines.

```bash
$cd ${WUKONG_ROOT}/scripts
$./syncdeps.sh ../deps/dependencies mpd.hosts
```


<br>
<a name="run"></a>
## Building and running

### Step 1: Compile Wukong

1) Update `CMakeLists.txt`.

+ **Enable/disable RDMA Feature**: Currently, Wukong will enable RDMA feature by default, and suppose the driver has been well installed and configured. If you run RDMA on non-RDMA networks, you need manually comment `add_definitions(-DHAS_RDMA)`.

+ **Enable/disable HDFS support**: To support loading input dataset from HDFS, you need manually uncomment `add_definitions(-DHAS_HADOOP)` and `#target_link_libraries(wukong hdfs)` to enable it.

+ **Enable/disable the support to versatile queries**: To support versatile queries (e.g., ?S ?P ?O), you need manually uncomment `#add_definitions(-DVERSATILE)`. It should be noted that enabling this feature will use more main memory to store RDF graph.

+ **Use 32-bit or 64-bit ID**: The 32-bit ID is enough to support the dataset with more than 2 billion unified strings. If you want to support more large dataset (like LUBM-102400), you need manually uncomment `#add_definitions(-DDTYPE_64BIT)`. It will consume more memory and slightly increase the query latency.

2) Build Wukong.

```bash
$cd ${WUKONG_ROOT}/scripts
$./build.sh
```


### Step 2: Configure Wukong

1) Edit the Wukong's configuration file `config`, which is specified by `run.sh` (the first argument).

```bash
$cd $WUKONG_ROOT/scripts
$cat config
global_num_proxies          4
global_num_engines          16
global_input_folder         /path/to/input/id_lubm_2
global_load_minimal_index   1
global_data_port_base       5500
global_ctrl_port_base       9576
global_memstore_size_gb     20
global_rdma_buf_size_mb     128
global_rdma_rbf_size_mb     32
global_use_rdma             1
global_rdma_threshold       300
global_mt_threshold         16
global_enable_caching       0
global_enable_workstealing  0
global_silent               1
global_enable_planner       0
```

The main configuration items:  

* `global_num_proxies` and `global_num_engines`: set the number of proxy/engine threads
* `global_input_folder`: set the path to folder for input files
* `global_memstore_size_gb`: set the size (GB) of in-memory store for input data
* `global_rdma_buf_size_mb` and `global_rdma_rbf_size_mb`: set the size (MB) of in-memory data structures used by RDMA operations
* `global_use_rdma`: leverage RDMA operations to process queries or not
* `global_silent`: return back query results to the proxy or not
* `global_enable_planner`: enable standard SPARQL parser and auto query planner


> Note: disable `global_load_minimal_index` and `global_silent` if you'd like to print or dump query results.


2) Edit the file `core.bind` to manually control the thread binding with CPU cores. Each line corresponds to one NUMA node (CPU Socket) and each thread (identified by ID) will be orderly pinned to a core of the NUMA node. 

> Note: Wukong uses a contiguous thread ID for proxy and engine threads successively. 

For example, suppose Wukong has 4 proxies threads (ID from 0 to 3) and 16 engine threads (ID from 4 to 19) on each machine, then the following `core.bind` file will bind 2 proxy threads and 8 engine threads to each of first two NUMA nodes on all machines in your cluster. 

```bash
$cd ${WUKONG_ROOT}/scripts
$cat core.bind
# One node per line (NOTE: the empty line means to skip a node)
0  1  4  5  6  7  8  9 10 11
2  3 12 13 14 15 16 17 18 19
```


### Step 3: Copy all Wukong files to all machines

```bash
$cd ${WUKONG_ROOT}/scripts
$./sync.sh
```

> Note: whenever you rebuild Wukong or modify setting files in `$WUKONG_ROOT/scripts/`, you should run `sync.sh` again to sync with all machines.


### Step 4: Launch Wukong with a builtin local console

```bash
$cd ${WUKONG_ROOT}/scripts
# ./run.sh [#nodes] (e.g., 3 nodes)
$./run.sh 3
```



<br>
<a name="data"></a>
## Preparing RDF datasets

Use [LUBM](http://swat.cse.lehigh.edu/projects/lubm) (SWAT Projects - the Lehigh University Benchmark) as an example to introduce how to prepare RDF datasets for Wukong.

#### Step 1: Generate LUBM datasets with RAW format

1) Install LUBM data generator (UBA).

```bash
$cd ~;
$wget http://swat.cse.lehigh.edu/projects/lubm/uba1.7.zip
$wget http://swat.cse.lehigh.edu/projects/lubm/GeneratorLinuxFix.zip
$unzip uba1.7.zip -d ./uba1.7
$unzip GeneratorLinuxFix.zip -d .
$mv Generator.java uba1.7/src/edu/lehigh/swat/bench/uba/Generator.java
$javac uba1.7/src/edu/lehigh/swat/bench/uba/Generator.java
$cp uba1.7/src/edu/lehigh/swat/bench/uba/*.class uba1.7/classes/edu/lehigh/swat/bench/uba/
```

> Note: one patch is used to fix a bug about linux file path

2) Generate LUBM datasets (2 Universities) with RAW format.

```bash
$cd ~/uba1.7
$java -cp ./classes/ edu.lehigh.swat.bench.uba.Generator -univ 2 -onto http://swat.cse.lehigh.edu/onto/univ-bench.owl
Started...
/home/rchen/datasets/uba1.7/University0_0.owl generated
CLASS INSTANCE #: 1657, TOTAL SO FAR: 1657
PROPERTY INSTANCE #: 6896, TOTAL SO FAR: 6896
...
```


#### Step 2: Convert LUBM datasets to NT format

1) Download the rdfcat tool of Apache Jena.

```bash
$cd ~;
$wget http://archive.apache.org/dist/jena/binaries/apache-jena-2.7.4.zip
$unzip apache-jena-2.7.4.zip -d .
$export JENA_HOME=~/apache-jena-2.7.4/
```

2) Generate LUBM dataset (2 Universities) with NT format.

```bash
$cd ~/uba1.7
$find . -type f -name "University0_*.owl" -exec $JENA_HOME/bin/rdfcat -out N-TRIPLE -x {} >> uni0.nt \;
$find . -type f -name "University1_*.owl" -exec $JENA_HOME/bin/rdfcat -out N-TRIPLE -x {} >> uni1.nt \;
```

Each row in LUBM dataset with NT format (e.g., `uni0.nt`) consists of subject (S), predicate (P), object (O), and '.', like`<http://www.University97.edu> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> .`.


#### Step 3: Convert LUBM datasets to ID format

```bash
$cd ${WUKONG_ROOT}/datagen;
$g++ -std=c++11 generate_data.cpp -o generate_data
$mkdir nt_lubm_2
$mv ~/uba1.7/uni*.nt nt_lubm_2/
$./generate_data nt_lubm_2 id_lubm_2
Process No.1 input file: uni1.nt.
Process No.2 input file: uni0.nt.
#total_vertex = 58455
#normal_vertex = 58421
#index_vertex = 34
$ls id_lubm_2
id_uni0.nt  id_uni1.nt  str_index  str_normal
```

Each row in LUBM dataset with ID format (e.g., `id_uni0.nt`) consists of the 3 IDs (non-negative integer), like `132323  1  16`. `str_index` and `str_normal` store the mapping from string to ID for index (e.g., predicate) and normal (e.g., subject and object) entities respectively.

#### Step 4: Load LUBM datasets by Wukong 

Move dataset (e.g., `id_lubm_2`) to a distributed FS (e.g., NFS and HDFS), which can be accessed by all machines in your cluster, and update the `global_input_folder` in `config` file.

> Note: you can improve the loading time by enabling `str_normal_minimal` in `config` file, if you know which strings in `str_normal` will be used by queries in advance. You need create a `str_normal_minimal` file by the following command.

```bash
$cd id_lubm_2
$grep "<http://www.University0.edu>" str_normal >> str_normal_minimal
$grep "<http://www.Department0.University0.edu>" str_normal >> str_normal_minimal
```


