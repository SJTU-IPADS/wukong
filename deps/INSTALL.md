## Step-by-step Instruction

### Table of Contents

* [OpenMPI v1.6.5](#openmpi)
* [Boost v1.67.0](#boost)
* [Intel TBB v4.4.2](#tbb)
* [ZeroMQ v4.0.5](#0MQ)
* [nanomsg v1.1.4](#nanomsg)
* [HWLOC v1.11.7](#hwloc)
* [Jemalloc v5.1.0](#jemalloc)
* [HDFS support](#hdfs)

<a name="openmpi"></a>
### OpenMPI v1.6.5

```bash
$cd  $WUKONG_ROOT/deps/
$tar zxvf openmpi-1.6.5.tar.gz
$mkdir openmpi-1.6.5-install
$cd openmpi-1.6.5/
$./configure --prefix=$WUKONG_ROOT/deps/openmpi-1.6.5-install
$make all
$make install
```


<a name="boost"></a>
### Boost v1.67.0

```bash
$cd  $WUKONG_ROOT/deps/
$tar jxvf boost_1_67_0.tar.gz
$mkdir boost_1_67_0-install
$cd boost_1_67_0/
$./bootstrap.sh --prefix=../boost_1_67_0-install
```

Add the following MPI configuration to `project-config.jam`

```bash
# MPI configuration
using mpi : $WUKONG_ROOT/deps/openmpi-1.6.5-install/bin/mpicc ;
```

```bash
$./b2 install
```


<a name="tbb"></a>
### Intel Threading Building Blocks (TBB) v4.4.2

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


<a name="0MQ"></a>
### ZeroMQ v4.0.5

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

> zmq.hpp is download from [C++ binding for 0MQ](https://github.com/zeromq/cppzmq/blob/master/zmq.hpp)
> zhelpers.hpp is download from [ØMQ - The Guide](https://github.com/booksbyus/zguide/blob/master/examples/C%2B%2B/zhelpers.hpp)

Add below settings to bash script (i.e., `~/.bashrc`).

```bash
# ZeroMQ configuration
export CPATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/include:$CPATH
export LIBRARY_PATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$WUKONG_ROOT/deps/zeromq-4.0.5-install/lib:$LD_LIBRARY_PATH
```

<a name="nanomsg"></a>
### nanomsg v1.1.4

```bash
$cd $WUKONG_ROOT/deps/
$tar zxvf 1.1.4.tar.gz
$mkdir nanomsg-1.1.4-install
$cd nanomsg-1.1.4
$mkdir build
$cd build
$cmake -DCMAKE_INSTALL_PREFIX=$WUKONG_ROOT/deps/nanomsg-1.1.4-install/ ..
$cmake --build . --target install
$cd $WUKONG_ROOT/deps/
$cp nn.hpp $WUKONG_ROOT/deps/nanomsg-1.1.4-install/include/nanomsg/
```

<a name="hwloc"></a>
### Portable Hardware Locality (hwloc) v1.11.7

```bash
$cd $WUKONG_ROOT/deps/
$mkdir hwloc-1.11.7-install
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


<a name="jemalloc"></a>
### Jemalloc v5.2.1

```bash
$cd $WUKONG_ROOT/deps/
$wget "https://github.com/jemalloc/jemalloc/releases/download/5.2.1/jemalloc-5.2.1.tar.bz2"
$mkdir jemalloc-5.2.1-install
$tar jxvf jemalloc-5.2.1.tar.bz2
$cd jemalloc-5.1.0/
$./configure --with-jemalloc-prefix=je --prefix=$WUKONG_ROOT/deps/jemalloc-5.2.1-install/
$make
$make install
```

Add below settings to bash script (i.e., `~/.bashrc`).

```bash
# hwloc configuration
export PATH=$WUKONG_ROOT/deps/jemalloc-5.1.0-install/bin:$PATH
export CPATH=$WUKONG_ROOT/deps/jemalloc-5.1.0-install/include:$CPATH
export LIBRARY_PATH=$WUKONG_ROOT/deps/jemalloc-5.1.0-install/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$WUKONG_ROOT/deps/jemalloc-5.1.0-install/lib:$LD_LIBRARY_PATH
```

<a name="hdfs"></a>
### HDFS support (optional)

We assume that Hadoop/HDFS has been installed on your cluster. The ENV variable for Hadoop should be set correctly.

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
export CLASSPATH=$WUKONG_ROOT/deps/hadoop/hadoop.jar:$CLASSPATH
```

Run the script `hadoop_deps.sh` to generate `hadoop.jar`

```bash
# Build hadoop.jar (assume that the Hadoop has been added on this machine)
$cd $WUKONG_ROOT/deps/hadoop/
$./hadoop_deps.sh
```

>Note: if the `global_input_folder` start with `hdfs:`, then Wukong will read the files from HDFS.
