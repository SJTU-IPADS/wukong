# Installing Wukong

## Table of Contents

* [Preparing and Downloading](#dl)
* [Satisfying Dependences](#deps)
* [Building and Running](#run)
* [Preparing RDF datasets](#data)

<br>
<a name="dl"></a>
## Preparing and Downloading

> Note: the current version of Wukong was tested on Ubuntu Linux 64-bit 14.04.
It requires a 64-bit operating system, and a few dependencies must be manually satisfied.

```bash
$sudo apt-get update
$sudo apt-get install gcc g++ build-essential cmake git libreadline6-dev wget
```

You can download Wukong directly from the Github Repository.

```bash
$git clone https://github.com/SJTU-IPADS/wukong.git
$cd wukong
```

<br>
<a name="deps"></a>
## Satisfying Dependences

#### Install Wukong dependecies on one of your cluster machines

Add the root path of Wukong (e.g., `/home/rchen/wukong`) to the bash script (i.e., `~/.bashrc`).

```bash
# Wukong configuration
export WUKONG_ROOT=[/path/to/wukong]   
```

We provide a shell script (i.e., `$WUKONG_ROOT/deps/deps.sh`) to download and build most of its required dependencies automatically within the local sub-directory (i.e., `$WUKONG_ROOT/deps/`). 

> Currently, we requires OpenMPI v1.6.5, Boost v1.58, Intel TBB v4.4.2, ZeroMQ v4.0.5, HWLOC v1.11.7, and LibRDMA v1.0.0 (optional).

```bash
$cd deps
$./deps.sh
```

> If you do not need RDMA feature, you could skip LibRDMA by running `./deps.sh no-rdma`

**Then add settings of tbb(step 4), zeromq(step 5), hwloc(step 6) and librdma(step 7) to bash script (i.e., `~/.bashrc`).**

If you want to do it manually, [deps/INSTALL.md](deps/INSTALL.md) provides step-by-step instruction.


#### Copy Wukong dependencies to all machines

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
## Building and Running

#### Compile Wukong

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


#### Configure Wukong

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


#### Copy all Wukong files to all machines

```bash
$cd ${WUKONG_ROOT}/scripts
$./sync.sh
```

> Note: whenever you rebuild Wukong or modify setting files in `$WUKONG_ROOT/scripts/`, you should run `sync.sh` again to sync with all machines.


#### Launch Wukong with a builtin local console

```bash
$cd ${WUKONG_ROOT}/scripts
# run.sh [#nodes] (e.g., 3 nodes)
$./run.sh 3
...
Input 'help' command to get more information
wukong>
```



<br>
<a name="data"></a>
## Preparing RDF datasets

Use [LUBM](http://swat.cse.lehigh.edu/projects/lubm) (SWAT Projects - the Lehigh University Benchmark) as an example to introduce how to prepare RDF datasets for Wukong.

##### Step 1: *Generate LUBM datasets with RAW format*

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


##### Step 2: *Convert LUBM datasets to NT format*

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


##### Step 3: *Convert LUBM datasets to ID format*

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

##### Step 4: *Load LUBM datasets by Wukong*

Move dataset (e.g., `id_lubm_2`) to a distributed FS (e.g., NFS and HDFS), which can be accessed by all machines in your cluster, and update the `global_input_folder` in `config` file.

> Note: you can improve the loading time by enabling `str_normal_minimal` in `config` file, if you know which strings in `str_normal` will be used by queries in advance. You need create a `str_normal_minimal` file by the following command.

```bash
$cd id_lubm_2
$grep "<http://www.University0.edu>" str_normal >> str_normal_minimal
$grep "<http://www.Department0.University0.edu>" str_normal >> str_normal_minimal
```