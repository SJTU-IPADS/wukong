# Wukong+G Tutorials

## Table of Contents

* [Build Wukong with GPU support](#build)
* [Configuring and running Wukong+G](#config)
* [Run SPARQL queries on Wukong+G](#run)
* [Caveat](#caveat)

<a  name="build"></a>

## Build Wukong with GPU support

To use Wukong with GPU support (Wukong+G), you need a [CUDA-enabled GPU](https://developer.nvidia.com/cuda-gpus) with [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) installed, and the GPU needs to support GPUDirect RDMA (at least Kepler-class). Edit the `.bashrc` in your home directory to add `/usr/local/cuda-8.0/bin` to `PATH ` and `/usr/local/cuda-8.0/include` to the `CPATH`.

```bash
export CUDA_HOME=/usr/local/cuda-8.0
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=${CUDA_HOME}/include:$CPATH
```

And edit the **CMakeLists.txt** in the `$WUKONG_ROOT` to enable the `USE_GPU` option. Then go to `$WUKONG_ROOT/scripts` and run `./build.sh` to build Wukong+G. Or you can just run `./build.sh -DUSE_GPU=ON` to build with GPU support. Currently, the `USE_GPU` option **conflicts with** `USE_JEMALLOC`, `USE_DYNAMIC_GSTORE` and  `USE_VERSATILE`.

Next, we need to install the [kernel module for GPUDirect RDMA](http://www.mellanox.com/page/products_dyn?product_family=295&mtag=gpudirect). Download it from the web and install it to **each machine** in the cluster.

```bash
tar xzf nvidia-peer-memory-1.0-3.tar.gz
cd nvidia-peer-memory-1.0
tar xzf nvidia-peer-memory_1.0.orig.tar.gz
cd nvidia-peer-memory-1.0
dpkg-buildpackage -us -uc
cd ..
sudo dpkg -i nvidia-peer-memory_1.0-3_all.deb
sudo dpkg -i nvidia-peer-memory-dkms_1.0-3_all.deb
```

You should manually load it after the installation, and make sure the module is loaded successfully.

```bash
sudo modprobe nv_peer_mem
lsmod | grep nv_peer_mem
```



> Note: You can use the `deviceQuery` utility provided by CUDA toolkit to query the capabilities of your GPU. You can find it in `/usr/local/cuda-8.0/samples/1_Utilities` in Ubuntu.



<a  name="config"></a>

## Configuring and running Wukong+G

1) Edit the `config` in `$WUKONG_ROOT/scripts` according to capabilities of your GPU .

```bash
$cd $WUKONG_ROOT/scripts
$vim config
```

The configuration items related to GPU support are:

* `global_num_gpus`: set the number of GPUs (currently we only support one GPU)
* `global_gpu_rdma_buf_size_mb`: set the size (MB) of buffer for one-sided RDMA operations
* `global_gpu_rbuf_size_mb`: set the size (MB) of result buffer in GPU memory for query processing
* `global_gpu_kvcache_size_gb`: set the size (GB) of key-value cache in GPU memory
* `global_gpu_key_blk_size_mb`: set the size (MB) of key block in key-value cache
* `global_gpu_value_blk_size_mb`: set the size (MB) of value block in key-value cache
* `global_enable_pipeline`: enable query execution overlaps with memory copy between CPU and GPU

2) Sync Wukong files to machines listed in `mpd.hosts`.

```bash
$cd ${WUKONG_ROOT}/scripts
$./sync.sh
sending incremental file list
...
```

3) Launch Wukong server on your cluster.

```bash
$cd ${WUKONG_ROOT}/scripts
$./run.sh 2
...
Input 'help' command to get more information
wukong>
```



<a name="run"></a>

## Run SPARQL queries on Wukong+G

Wukong+G adds a new argument `-g` to the `sparql` command, which tells the console to send the query to a GPU agent in the cluster. The GPU agent will use the GPU with device ID 0 to process queries.

1) Wukong+G commands.

```bash
wukong> help
These are common Wukong commands: :

help                display help infomation:

quit                quit from the console:

config <args>       run commands for configueration:
  -v                     print current config
  -l <fname>             load config items from <fname>
  -s <string>            set config items by <str> (e.g., item1=val1&item2=...)
  -h [ --help ]          help message about config

logger <args>       run commands for logger:
  -v                     print loglevel
  -s <level>             set loglevel to <level> (e.g., DEBUG=1, INFO=2,
                         WARNING=4, ERROR=5)
  -h [ --help ]          help message about logger

sparql <args>       run SPARQL queries in single or batch mode:
  -f <fname>             run a [single] SPARQL query from <fname>
  -m <factor> (=1)       set multi-threading <factor> for heavy query
                         processing
  -n <num> (=1)          repeat query processing <num> times
  -p <fname>             adopt user-defined query plan from <fname> for running
                         a single query
  -N <num> (=1)          do query optimization <num> times
  -v <lines> (=0)        print at most <lines> of results
  -o <fname>             output results into <fname>
  -g                     leverage GPU to accelerate heavy query processing
  -b <fname>             run a [batch] of SPARQL queries configured by <fname>
  -h [ --help ]          help message about sparql
  ...
```

2) run a single SPARQL query.

There are query examples in `$WUKONG_ROOT/scripts/sparql_query`. For example, enter `sparql -f sparql_query/lubm/basic/lubm_q7` to run the query `lubm_q7`.
Please note that if you didn't enable the planner, you should specify a query plan manually.


```bash
wukong> sparql -f sparql_query/lubm/basic/lubm_q7 -g
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 172 usec
INFO:     Optimization time: 507 usec
INFO:     Leverage GPU to accelerate query processing.
INFO:     (last) result size: 1763
INFO:     (average) latency: 1452 usec


wukong> sparql -f sparql_query/lubm/basic/lubm_q7 -p sparql_query/lubm/basic/osdi16_plan/lubm_q7.fmt -g
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 111 usec
INFO:     User-defined query plan is enabled
INFO:     Leverage GPU to accelerate query processing.
INFO:     (last) result size: 1763
INFO:     (average) latency: 1908 usec
```

Performance data can be found in `docs/performance` folder.

<a name="caveat"></a>

## Caveat
Although Wukong+G can notably speed up query processing, there is still plenty of room for improvement:

- We only tested Wukong+G in a RDMA-capable cluster with NVIDIA Tesla K40m and CUDA 8.0.
- Wukong+G assumes the predicate of triple patterns in a query is known, queries with unknown predicates cannot be handled by GPU engine.
- If a query produces huge intermediate result that exceeds the size of result buffer on GPU (``global_gpu_rbuf_size_mb``), Wukong+G cannot handle it (may crash or return wrong results).
- If the size of triples of a predicate cannot fit into the key-value cache on GPU, Wukong+G cannot handle it.
- If pipeline is enabled, there should be enough GPU memory to accommodate two triple patterns, the current pattern and the prefetched pattern.



