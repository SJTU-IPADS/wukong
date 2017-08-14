# Wukong Tutorials

## Table of Contents

- Install Wukong
- Prepare RDF data
- Query with Wukong

## Install Wukong (Take Ubuntu as an example)

The current version of Wukong was tested on Ubuntu Linux 64-bit 14.04. It requires a 64-bit operating system.

To install Wukong, you need to:

1. Clone the repository.
2. Go through the following steps.

### Step 1: Satisfy Dependencies

Make sure the following tools are installed:

- g++: Required for compiling Wukong.
- \*nix build tools (e.g., patch, make)

If not, run:

```bash
sudo apt-get update
sudo apt-get install gcc g++ build-essential cmake libreadline6-dev
```

**1. Add the root path of Wukong**

Add the root path of Wukong (e.g., `/home/rchen/wukong`) to bash script (i.e., `~/.bashrc`).

```bash
# Wukong configuration
export WUKONG_ROOT=[/path/to/wukong]
```

**2. Install Open MPI**

```bash
cd $WUKONG_ROOT/deps/
tar zxvf openmpi-1.6.5.tar.gz
mkdir openmpi-1.6.5-install
cd openmpi-1.6.5/
./configure --prefix=../openmpi-1.6.5-install
make all install
```

**3. Install Boost**

```bash
cd $WUKONG_ROOT/deps/
tar jxvf boost_1_58_0.tar.bz2
mkdir boost_1_58_0-install
cd boost_1_58_0/
./bootstrap.sh --prefix=../boost_1_58_0-install
```

Add the following MPI configuration to `$WUKONG_ROOT/deps/boost_1_58_0/project-config.jam`

```bash
# MPI configuration
using mpi : [/path/to/Wukong]/deps/openmpi-1.6.5-install/bin/mpicc;
```

finally, run:

```bash
./b2 install
```

**4. Install Intel TBB**

```bash
cd $WUKONG_ROOT/deps/
tar zxvf tbb44_20151115oss_src.tgz
cd tbb44_20151115oss/
make
```

Add below settings to bash script (i.e., `~/.bashrc`).

```bash
# Intel TBB configuration
source $WUKONG_ROOT/deps/tbb44_20151115oss/build/[version]/tbbvars.sh
```
For example:

```bash
source $WUKONG_ROOT/deps/tbb44_20151115oss/build/linux_intel64_gcc_cc4.8_libc2.19_kernel3.14.27_release/tbbvars.sh
```

**5. Install ZeroMQ**

```bash
cd $WUKONG_ROOT/deps/
tar zxvf zeromq-4.0.5.tar.gz
mkdir zeromq-4.0.5-install
cd zeromq-4.0.5/
./configure --prefix=$WUKONG_ROOT/deps/zeromq-4.0.5-install/
make
make install
cd ..
cp zmq.hpp  zeromq-4.0.5-install/include/
cp zhelpers.hpp  zeromq-4.0.5-install/include/
```

Add below settings to bash script (i.e., `~/.bashrc`).

```bash
# ZeroMQ configuration
export CPATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/include:$CPATH
export LIBRARY_PATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/lib:$LD_LIBRARY_PATH
```

**6. Install librdma (Optional)**

```bash
cd $WUKONG_ROOT/deps/
tar zxvf librdma-1.0.0.tar.gz
cd librdma-1.0.0/
./configure --prefix=$WUKONG_ROOT/deps/librdma-1.0.0-install/
make
make install
cd ..
```

Add below settings to bash script (i.e., `~/.bashrc`).

```bash
# librdma configuration
export CPATH=$WUKONG_ROOT/deps/librdma-1.0.0-install/include:$CPATH
export LIBRARY_PATH=$WUKONG_ROOT/deps/librdma-1.0.0-install/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$WUKONG_ROOT/deps/librdma-1.0.0-install/lib:$LD_LIBRARY_PATH
```

**7. Install HDFS Support(Optional)**

We assume that Hadoop/HDFS has been installed on your cluster. The ENV variable for Hadoop should be set correctly.

```bash
# Build hadoop.jar (assume that the Hadoop has been added on this machine)
cd $WUKONG_ROOT/deps/hadoop/
./hadoop_deps.sh
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

Enable the compile option in `$WUKONG_ROOT/CMakeLists.txt`

```bash
# Uncomment two lines below to enble HDFS support.
add_definitions(-DHAS_HADOOP)
target_link_libraries(wukong hdfs)
```

*NOTE: if the `global_input_folder` in `$WUKONG_ROOT/scripts/config` start with `hdfs:`, then Wukong will read the files from HDFS.*

### Step 2: Copy Wukong Dependencies to All Machines

Now let's copy the dependencies to all machines. Before that, you should check the file `$WUKONG_ROOT/deps/dependencies` and make sure it contains the following lines:

```bash
zeromq-4.0.5-install
tbb44_20151115oss
hadoop
librdma-1.0.0-install
openmpi-1.6.5-install
```

Then go through the following steps:

1. Setup password-less SSH between the master node and all other machines.
2. Create a file in `scripts` directory called `mpd.hosts` with the IP address of all machines.
    For example:
    ```bash
    $cat ${WUKONG_ROOT}/scripts/mpd.hosts
    10.0.0.100
    10.0.0.101
    10.0.0.102
    ```
3. Run the following commands to copy Wukong dependencies to the rest of the machines:
    ```bash
    cd ${WUKONG_ROOT}/scripts
    ./syncdeps.sh ../deps/dependencies mpd.hosts
    ```

### Step 3: Build Wukong

Before you build Wukong, check the following list:

1. Make sure the value of `CMAKE_CXX_COMPILER` in `$WUKONG_ROOT/CMakeLists.txt` is the path to your mpi. For example:
    ```bash
    set(CMAKE_CXX_COMPILER ${ROOT}/deps/openmpi-1.6.5-install/bin/mpic++)
    ```
2. Currently, Wukong will enable RDMA feature by default, and suppose the driver has been well installed and configured. If you want to disable RDMA, you need manually comment `add_definitions(-DHAS_RDMA)` in `$WUKONG_ROOT/CMakeLists.txt`.
3. To support versatile queries (e.g., ?S ?P ?O), you need manually uncomment `#add_definitions(-DVERSATILE)` in `$WUKONG_ROOT/CMakeLists.txt`. It should be noted that enabling this feature will use more main memory to store RDF graph.

Now it's time to build Wukong! Run:

```bash
cd ${WUKONG_ROOT}/scripts
./build.sh
```

After the building, synchronize all executable files (e.g., `build/wukong`) to all machines. Run:

```bash
cd ${WUKONG_ROOT}/scripts
./sync.sh
```

Congratulations, you've successfully installed Wukong!

*NOTE: Everytime you rebuild Wukong or modify configs in `$WUKONG_ROOT/scripts/config`, you should run sync.sh*

## Prepare RDF Data

Taking LUBM as an example, this tutorial introduces how to prepare RDF data for Wukong.

### 1 Install LUBM-generator

```bash
cd ~
wget http://swat.cse.lehigh.edu/projects/lubm/uba1.7.zip
wget http://swat.cse.lehigh.edu/projects/lubm/GeneratorLinuxFix.zip
unzip uba1.7.zip -d ./uba1.7
unzip GeneratorLinuxFix.zip -d .
cp Generator.java uba1.7/src/edu/lehigh/swat/bench/uba/Generator.java
cd uba1.7/src/
javac edu/lehigh/swat/bench/uba/Generator.java
cd ~
cp uba1.7/src/edu/lehigh/swat/bench/uba/*.class uba1.7/classes/edu/lehigh/swat/bench/uba/
```

### 2 Install Jena

```bash
cd ~
wget http://archive.apache.org/dist/jena/binaries/apache-jena-2.7.4.zip
unzip apache-jena-2.7.4.zip -d .
export JENA_HOME=~/apache-jena-2.7.4/
```

### 3 Generate LUBM-1 (.nt format)

```bash
cd uba1.7
java -cp ./classes/ edu.lehigh.swat.bench.uba.Generator -univ 1 -onto http://swat.cse.lehigh.edu/onto/univ-bench.owl
find . -type f -name "University*.owl" -exec $JENA_HOME/bin/rdfcat -out N-TRIPLE -x {} >> uni0.nt \;
# Now we get NT format of LUBM data
# each row consists of S P O and '.'
```

### 4 Generate LUBM (Wukong internal format)

```bash
cd ${WUKONG_ROOT}/datagen;
g++ -std=c++11 generate_data.cpp -o generate_data
mkdir lubm_raw_1
mv ~/uba1.7/uni0.nt lubm_raw_1/
./generate_data lubm_raw_1 id_lubm_1
```

Now you have got the RDF data stored in `id_lubm_1/`. Let's start using it in the next tutorial.

## Query with Wukong

### 1 Config

Edit the configs of Wukong in `$WUKONG_ROOT/scripts/config`.

```bash
global_num_engines			16
global_num_proxies			1
global_input_folder			hdfs://192.168.12.124:9000/wangn/rdfdata/id_lubm_40
global_load_minimal_index	1
global_data_port_base		5500
global_ctrl_port_base		9576
global_memstore_size_gb		20
global_rdma_buf_size_mb		128
global_rdma_rbf_size_mb		32
global_use_rdma				0
global_rdma_threshold		300
global_mt_threshold			16
global_enable_caching		0
global_enable_workstealing	0
global_silent 				0
global_enable_planner		1
```

Modify the `global_input_folder` to the path of your RDF data. If the path starts with "hdfs:", Wukong will load data from HDFS.

If you'd like to print or dump results, set `global_silent` and `global_load_minimal_index` to 0.

If you'd like to use rdma, set `global_use_rdma` to 1.

*NOTE: Don't forget to run sync.sh after editing config.*

### 2 Run

Wukong offers a builtin local console for testing:

```bash
cd ${WUKONG_ROOT}/scripts
./run.sh [num_of_nodes]
```

```bash
These are common Wukong commands:
    help           Display help infomation
    quit           Quit from console
    config <args>  Run commands on config
        -v          print current config
        -l <file>   load config items from <file>
        -s <string> set config items by <str> (format: item1=val1&item2=...)
    sparql         Run SPARQL queries
        -f <file>   a single query from the <file>
           -n <num>    run a single query <num> times
           -v <num>    print at most <num> lines of the result (default:10)
           -o <file>   write the result into the <file>
        -b <file>   a set of queries configured by the <file>
```

There are query examples in `$WUKONG_ROOT/scripts/sparql_query`. For example, input `sparql -f sparql_query/lubm_q2` to run the query `lubm_q2`.

Batch query is used to test throughput. There is an example in `$WUKONG_ROOT/scripts/batch`. Input `sparql -b batch/mix_config` to run.
