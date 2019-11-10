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
These are common Wukong commands: 
    help                display help infomation
    quit                quit from console
    config <args>       run commands on config
        -v                  print current config
        -l <file>           load config items from <file>
        -s <string>         set config items by <str> (format: item1=val1&item2=...)
    sparql <args>       run SPARQL queries
        -f <file> [<args>]  a single query from <file>
           -n <num>            run <num> times
           -v <num>            print at most <num> lines of results
           -o <file>           output results into <file>
           -g                  send the quer to GPU
```

2) run a single SPARQL query.

There are query examples in `$WUKONG_ROOT/scripts/query`. For example, input `sparql -f query/lubm_q7` to run the query `lubm_q7`.

```bash
wukong> sparql -f query/lubm/lubm_q7 -g -v 5
INFO:     The query will be sent to GPU
INFO:     (average) latency: 941 usec
INFO:     (last) result size: 73
INFO:     The first 5 rows of results:
1: <http://www.Department7.University1.edu/FullProfessor5>	<http://www.Department7.University1.edu/UndergraduateStudent42>	<http://www.Department7.University1.edu/Course8>
2: <http://www.Department7.University1.edu/FullProfessor5>	<http://www.Department7.University1.edu/UndergraduateStudent380>	<http://www.Department7.University1.edu/Course8>
3: <http://www.Department11.University1.edu/FullProfessor1>	<http://www.Department11.University1.edu/UndergraduateStudent134>	<http://www.Department11.University1.edu/Course2>
4: <http://www.Department2.University1.edu/FullProfessor4>	<http://www.Department2.University1.edu/UndergraduateStudent6>	<http://www.Department2.University1.edu/Course7>
5: <http://www.Department11.University0.edu/FullProfessor4>	<http://www.Department11.University0.edu/UndergraduateStudent265>	<http://www.Department11.University0.edu/Course7>
wukong>
```



<a name="caveat"></a>

## Caveat
Although Wukong+G can notably speed up query processing, there is still plenty of room for improvement:

- We only tested Wukong+G in a RDMA-capable cluster with NVIDIA Tesla K40m and CUDA 8.0.
- Wukong+G assumes the predicate of triple patterns in a query is known, queries with unknown predicates cannot be handled by GPU engine.
- If a query produces huge intermediate result that exceeds the size of result buffer on GPU (``global_gpu_rbuf_size_mb``), Wukong+G cannot handle it (may crash or return wrong results).
- If the size of triples of a predicate cannot fit into the key-value cache on GPU, Wukong+G cannot handle it.
- If pipeline is enabled, there should be enough GPU memory to accommodate two triple patterns, the current pattern and the prefetched pattern.



