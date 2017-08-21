# Installing Wukong

## Dependencies

> Note: the current version of Wukong was tested on Ubuntu Linux 64-bit 14.04.
It requires a 64-bit operating system.

### Step 0: Install build tools and git.

```bash
$ sudo apt-get update
$ sudo apt-get install gcc g++ build-essential cmake git libreadline6-dev wget
```


### Step 1: Download Wukong

```bash
git clone https://github.com/realstolz/wukong.git wukong
```

Add the root path of Wukong (e.g., `/home/rchen/wukong`) to bash script (i.e., `~/.bashrc`).

```bash
# Wukong configuration
export WUKONG_ROOT=[/path/to/wukong]   
```

### Step 2: Install OpenMPI

```bash
$ cd  $WUKONG_ROOT/deps/
$ tar zxvf openmpi-1.6.5.tar.gz
$ mkdir openmpi-1.6.5-install
$ cd openmpi-1.6.5/
$ ./configure --prefix=$WUKONG_ROOT/deps/openmpi-1.6.5-install
$ make all
$ make install
```

### Step 3: Install Boost

```bash
$ cd  $WUKONG_ROOT/deps/
$ tar jxvf boost_1_58_0.tar.bz2  
$ mkdir boost_1_58_0-install
$ cd boost_1_58_0/
$ ./bootstrap.sh --prefix=../boost_1_58_0-install  
```

Add the following MPI configuration to `project-config.jam`

```bash
# MPI configuration
using mpi : $WUKONG_ROOT/deps/openmpi-1.6.5-install/bin/mpicc ;
```

```bash
$ ./b2 install  
```

### Step 4: Install Intel TBB

```bash
$ cd $WUKONG_ROOT/deps/  
$ tar zxvf tbb44_20151115oss_src.tgz  
$ cd tbb44_20151115oss/
$ make
```

Add below settings to bash script (i.e., `~/.bashrc`).

```bash
# Intel TBB configuration
source $WUKONG_ROOT/deps/tbb44_20151115oss/build/[version]/tbbvars.sh
```

For example: `$WUKONG_ROOT/deps/tbb44_20151115oss/build/linux_intel64_gcc_cc4.8_libc2.19_kernel3.14.27_release/tbbvars.sh`


### Step 5: Install ZeroMQ (http://zeromq.org/)

```bash
$ cd $WUKONG_ROOT/deps/
$ tar zxvf zeromq-4.0.5.tar.gz
$ mkdir zeromq-4.0.5-install
$ cd zeromq-4.0.5/
$ ./configure --prefix=$WUKONG_ROOT/deps/zeromq-4.0.5-install/
$ make
$ make install
$ cd ..
$ cp zmq.hpp  zeromq-4.0.5-install/include/
$ cp zhelpers.hpp  zeromq-4.0.5-install/include/
```

Add below settings to bash script (i.e., `~/.bashrc`).

```bash
# ZeroMQ configuration
export CPATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/include:$CPATH
export LIBRARY_PATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/lib:$LD_LIBRARY_PATH
```

### Step 6: Install Portable Hardware Locality (hwloc)

```bash
$ cd $WUKONG_ROOT/deps/
$ tar zxvf hwloc-1.11.7.tar.gz
$ cd hwloc-1.11.7/
$ ./autogen.sh
$ ./configure --prefix=$WUKONG_ROOT/deps/hwloc-1.11.7-install/
$ make
$ make install
```

Add below settings to bash script (i.e., `~/.bashrc`).

```bash
# libhwloc configuration
export PATH=$WUKONG_ROOT/deps/hwloc-1.11.7-install/bin:$PATH
export CPATH=$WUKONG_ROOT/deps/hwloc-1.11.7-install/include:$CPATH
export LIBRARY_PATH=$WUKONG_ROOT/deps/hwloc-1.11.7-install/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$WUKONG_ROOT/deps/hwloc-1.11.7-install/lib:$LD_LIBRARY_PATH
```

### Step 7(Optional): Install librdma

```bash
$ cd $WUKONG_ROOT/deps/
$ tar zxvf librdma-1.0.0.tar.gz
$ cd librdma-1.0.0/
$ ./configure --prefix=$WUKONG_ROOT/deps/librdma-1.0.0-install/
$ make
$ make install
```

Add below settings to bash script (i.e., `~/.bashrc`).

```bash
# librdma configuration
export CPATH=$WUKONG_ROOT/deps/librdma-1.0.0-install/include:$CPATH
export LIBRARY_PATH=$WUKONG_ROOT/deps/librdma-1.0.0-install/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$WUKONG_ROOT/deps/librdma-1.0.0-install/lib:$LD_LIBRARY_PATH
```

### Step 8(Optional): Install HDFS support

We assume that Hadoop/HDFS has been installed on your cluster. The ENV variable for Hadoop should be set correctly.

```bash
# Build hadoop.jar (assume that the Hadoop has been added on this machine) 
$ cd $WUKONG_ROOT/deps/hadoop/
$ ./hadoop_deps.sh
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



## Building and Running

### Step 1: Configuring Wukong

#### Enable/disable RDMA Feature

Currently, Wukong will enable RDMA feature by default, and suppose the driver has been well installed and configured. If you want to disable RDMA, you need manually comment `add_definitions(-DHAS_RDMA)` in `CMakeLists.txt`.


#### Enable/disable the support to versatile queries

To support versatile queries (e.g., ?S ?P ?O), you need manually uncomment `#add_definitions(-DVERSATILE)` in `CMakeLists.txt`. It should be noted that enabling this feature will use more main memory to store RDF graph.


### Step 2: Compiling Wukong

```bash
$ cd ${WUKONG_ROOT}/scripts
$ ./build.sh
```

### Step 3: Copy all dependencies to all machines

1) Setup password-less SSH between the master node and all other machines.

2) Create a file in `scripts` directory called `mpd.hosts` with the IP address of all machines.

For example:

```bash
$ cat ${WUKONG_ROOT}/scripts/mpd.hosts
10.0.0.100
10.0.0.101
10.0.0.102
```

3) Run the following commands to copy all dependencies to the rest of the machines:

```bash
$ cd ${WUKONG_ROOT}/scripts
$ ./syncdeps.sh ../deps/dependencies mpd.hosts
```

### Step 4: Copy all Wukong files (e.g., `build/wukong`) to all machines 

```bash
$ cd ${WUKONG_ROOT}/scripts
$ ./sync.sh
```

> Note: everytime you rebuild Wukong or modify configs in `$WUKONG_ROOT/scripts/config`, you should run sync.sh


### Step 5: Launch Wukong servers with a builtin local console

```bash
$ cd ${WUKONG_ROOT}/scripts
# ./run.sh [#nodes] (e.g., 3 nodes)
$ ./run.sh 3
```



## Prepare RDF Data (LUBM)

### Step 1: Install LUBM-generator (include a bug fix about linux paths)

```bash
$ cd ~;
$ wget http://swat.cse.lehigh.edu/projects/lubm/uba1.7.zip
$ wget http://swat.cse.lehigh.edu/projects/lubm/GeneratorLinuxFix.zip
$ unzip uba1.7.zip -d ./uba1.7
$ unzip GeneratorLinuxFix.zip -d .
$ cp Generator.java uba1.7/src/edu/lehigh/swat/bench/uba/Generator.java
$ cd uba1.7/src/
$ javac edu/lehigh/swat/bench/uba/Generator.java
$ cd ~
$ cp uba1.7/src/edu/lehigh/swat/bench/uba/*.class uba1.7/classes/edu/lehigh/swat/bench/uba/
```

#### Step 2: Install Jena (convert LUBM data format)

```bash
$ cd ~;
$ wget http://archive.apache.org/dist/jena/binaries/apache-jena-2.7.4.zip
$ unzip apache-jena-2.7.4.zip -d .
$ export JENA_HOME=~/apache-jena-2.7.4/
```

#### Step 3: Generate LUBM_1 dataset with NT-format

```bash
$ cd uba1.7
$ java -cp ./classes/ edu.lehigh.swat.bench.uba.Generator -univ 1 -onto http://swat.cse.lehigh.edu/onto/univ-bench.owl
$ find . -type f -name "University*.owl" -exec $JENA_HOME/bin/rdfcat -out N-TRIPLE -x {} >> uni0.nt \;
# Now we get NT format of LUBM data
# each row consists of S P O and '.'
```

#### Step 4: Generate LUBM_1 dataset with ID-format

```bash
$ cd ${WUKONG_ROOT}/datagen;
$ g++ -std=c++11 generate_data.cpp -o generate_data
$ mkdir lubm_raw_1
$ mv ~/uba1.7/uni0.nt lubm_raw_1/
$ ./generate_data lubm_raw_1 id_lubm_1
```
