# Wukong Tutorials

## Table of Contents

* [Deploying Wukong on a local cluster](#cluster)
* [Downloading LUBM sample dataset](#data)
* [Configuring and running Wukong](#run)
* [Processing SPARQL queries on Wukong](#query)
* [Dynamic data loading on Wukong](#load)
* [Graph storage integrity check on Wukong](#check)


<a name="cluster"></a>
## Deploying Wukong on a local cluster

Install Wukong's dependencies (e.g., OpenMPI), using instructions in the [INSTALL](./INSTALL.md#dep), on your master node (one of your cluster machines, e.g., `node0.some.domain`) and copy necessities to the rest machines.

> Note: suppose there are two machines in your cluster, namely `node0.some.domain` and `node1.some.domain`.


```bash
$cd ${WUKONG_ROOT}/scripts
$cat mpd.hosts
node0.some.domain
node1.some.domain
$./syncdeps.sh ../deps/dependencies mpd.hosts
```


<a name="data"></a>
## Downloading LUBM sample dataset

```bash
$ mkdir $WUKONG_ROOT/datasets
$ cd $WUKONG_ROOT/datasets
$ wget http://ipads.se.sjtu.edu.cn/wukong/id_lubm_2.tar.gz
$ tar zxvf id_lubm_2.tar.gz
$ ls id_lubm_2
id_uni0.nt  id_uni1.nt  str_index  str_normal  str_normal_minimal
```

Move dataset (e.g., `id_lubm_2`) to a distributed FS (e.g., `path/to/input/`)which can be accessed by all machines in your cluster.


<a name="run"></a>
## Configuring and running Wukong

1) Edit `config` and `core.bind`.

```bash
$cd $WUKONG_ROOT/scripts
$cat config
# general
global_num_proxies              1
global_num_engines              2
global_data_port_base           5500
global_ctrl_port_base           9576
global_mt_threshold             2
global_enable_workstealing      0
global_stealing_pattern         0
global_enable_planner           1
global_generate_statistics      1
global_enable_vattr             0
global_silent                   0

# kvstore
global_input_folder             path/to/input/id_lubm_2/
global_memstore_size_gb         40
# global_est_load_factor is used to calculate how many buckets one segment should be allocated.
global_est_load_factor          55

# RDMA
global_rdma_buf_size_mb         128
global_rdma_rbf_size_mb         32
global_use_rdma                 1
global_rdma_threshold           300
global_enable_caching           0

# GPU
global_num_gpus                 0
global_gpu_rdma_buf_size_mb     64
global_gpu_rbuf_size_mb         32
global_gpu_kvcache_size_gb      10
global_gpu_key_blk_size_mb      16
global_gpu_value_blk_size_mb    4
global_gpu_enable_pipeline      1
$
$cat core.bind
# One node per line (NOTE: the empty line means to skip a node)
0 1 2
```

The detail explanation of above `config` file can be found in [INSTALL](./INSTALL.md#run)

2) Sync Wukong files to all machines.

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


<a name="query"></a>
## Processing SPARQL queries on Wukong

1) Wukong commands.

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
  -o <fname>             output results info <fname>
  -g                     leverage GPU to accelerate heavy query processing
  -b <fname>             run a [batch] of SPARQL queries configured by <fname>
  -h [ --help ]          help message about sparql

sparql-emu <args>   emulate clients to continuously send SPARQL queries:
  -f <fname>             run queries generated from temples configured by
                         <fname>
  -p <fname>             adopt user-defined query plans from <fname> for
                         running queries
  -d <sec> (=10)         eval <sec> seconds (default: 10)
  -w <sec> (=5)          warmup <sec> seconds (default: 5)
  -n <num> (=20)         keep <num> queries being processed (default: 20)
  -h [ --help ]          help message about sparql-emu

load <args>         load RDF data into dynamic (in-memmory) graph store:
  -d <dname>             load data from directory <dname>
  -c                     check and skip duplicate RDF triples
  -h [ --help ]          help message about load

gsck <args>         check the integrity of (in-memmory) graph storage:
  -i                     check from index key/value pair to normal key/value
                         pair
  -n                     check from normal key/value pair to index key/value
                         pair
  -h [ --help ]          help message about gsck

load-stat           load statistics of SPARQL query optimizer:
  -f <fname>             load statistics from <fname> located at data folder
  -h [ --help ]          help message about load-stat

store-stat          store statistics of SPARQL query optimizer:
  -f <fname>             store statistics to <fname> located at data folder
  -h [ --help ]          help message about store-stat
```

2) run a single SPARQL query.

There are query examples in `$WUKONG_ROOT/scripts/sparql_query`. For example, input `sparql -f sparql_query/lubm/basic/lubm_q2` to run the query `lubm_q2`.

```bash
wukong> sparql -f sparql_query/lubm/basic/lubm_q7 -v 5
(average) latency: 3660 usec
(last) result size: 73
The first 5 rows of results: 
1:  <http://www.Department8.University1.edu/FullProfessor5> <http://www.Department8.University1.edu/UndergraduateStudent204>  <http://www.Department8.University1.edu/Course9>  
2:  <http://www.Department14.University1.edu/FullProfessor6>  <http://www.Department14.University1.edu/UndergraduateStudent141> <http://www.Department14.University1.edu/Course7> 
3:  <http://www.Department4.University0.edu/FullProfessor0> <http://www.Department4.University0.edu/UndergraduateStudent312>  <http://www.Department4.University0.edu/Course1>  
4:  <http://www.Department7.University1.edu/FullProfessor9> <http://www.Department7.University1.edu/UndergraduateStudent8>  <http://www.Department7.University1.edu/Course14> 
5:  <http://www.Department8.University1.edu/FullProfessor7> <http://www.Department8.University1.edu/UndergraduateStudent47> <http://www.Department8.University1.edu/Course13>
wukong>
wukong> sparql -f sparql_query/lubm/basic/lubm_q4 -n 1000
(average) latency: 199 usec
(last) result size: 10
wukong>
```


2) show and change the configuration of Wukong at runtime.

```bash
wukong> config -v
------ global configurations ------
the number of proxies: 1
the number of engines: 2
global_input_folder: path/to/input/id_lubm_2
global_memstore_size_gb: 2
global_est_load_factor: 55
global_data_port_base: 5500
global_ctrl_port_base: 9576
global_rdma_buf_size_mb: 0
global_rdma_rbf_size_mb: 0
global_use_rdma: 0
global_enable_caching: 0
global_enable_workstealing: 0
global_stealing_pattern: 0
global_rdma_threshold: 300
global_mt_threshold: 2
global_silent: 0
global_enable_planner: 1
global_generate_statistics: 1
global_enable_vattr: 0
global_num_gpus: 0
global_gpu_rdma_buf_size_mb: 0
global_gpu_rbuf_size_mb: 32
global_gpu_kvcache_size_gb: 10
global_gpu_key_blk_size_mb: 16
global_gpu_value_blk_size_mb: 4
global_gpu_enable_pipeline: 1
--
the number of servers: 2
the number of threads: 3
wukong>
wukong> config -s global_use_rdma=0
wukong> sparql -f sparql_query/lubm/basic/lubm_q4 -n 1000
(average) latency: 1128 usec
(last) result size: 10
```

<a name="load"></a>
## Dynamic data loading on Wukong

Make sure that you have enable dynamic data loading support with parameter `-USE_DYNAMIC_GSTORE=ON`.

1) Load new dataset from directory, the structure of directory is just the same as which used to initialize.

```bash
wukong> load -d /home/datanfs/nfs0/rdfdata/id_lubm_2/
INFO:     loading ID-mapping file: /home/datanfs/nfs0/rdfdata/id_lubm_2/str_index
INFO:     loading ID-mapping file: /home/datanfs/nfs0/rdfdata/id_lubm_2/str_normal
INFO:     loading ID-mapping file: /home/datanfs/nfs0/rdfdata/id_lubm_2/str_index
INFO:     loading ID-mapping file: /home/datanfs/nfs0/rdfdata/id_lubm_2/str_normal
INFO:     2 data files and 0 attribute files found in directory (/home/datanfs/nfs0/rdfdata/id_lubm_2/) at server 0
INFO:     2 data files and 0 attribute files found in directory (/home/datanfs/nfs0/rdfdata/id_lubm_2/) at server 1
INFO:     load 94802 triples from file /home/datanfs/nfs0/rdfdata/id_lubm_2/id_uni0.nt at server 0
INFO:     load 122091 triples from file /home/datanfs/nfs0/rdfdata/id_lubm_2/id_uni1.nt at server 0
INFO:     #0: 227ms for inserting into gstore
INFO:     load 110672 triples from file /home/datanfs/nfs0/rdfdata/id_lubm_2/id_uni0.nt at server 1
INFO:     load 145107 triples from file /home/datanfs/nfs0/rdfdata/id_lubm_2/id_uni1.nt at server 1
INFO:     #1: 316ms for inserting into gstore
INFO:     (average) latency: 660366 usec
```

2) Add -c option to check and skip duplicate triples in the dataset.

```bash
wukong> load -c -d path/tp/input/id_lubm_2/
INFO:     loading ID-mapping file: path/tp/input/id_lubm_2/str_index
INFO:     loading ID-mapping file: path/tp/input/id_lubm_2/str_normal
INFO:     loading ID-mapping file: path/tp/input/id_lubm_2/str_index
INFO:     loading ID-mapping file: path/tp/input/id_lubm_2/str_normal
INFO:     2 data files and 0 attribute files found in directory (/home/datanfs/nfs0/rdfdata/id_lubm_2/) at server 0
INFO:     2 data files and 0 attribute files found in directory (/home/datanfs/nfs0/rdfdata/id_lubm_2/) at server 1
INFO:     load 94802 triples from file path/tp/input/id_lubm_2/id_uni0.nt at server 0
INFO:     load 122091 triples from file path/tp/input/id_lubm_2/id_uni1.nt at server 0
INFO:     #0: 222ms for inserting into gstore
INFO:     load 110672 triples from file path/tp/input/id_lubm_2/id_uni0.nt at server 1
INFO:     load 145107 triples from file path/tp/input/id_lubm_2/id_uni1.nt at server 1
INFO:     #1: 784ms for inserting into gstore
INFO:     (average) latency: 1072962 usec
```

<a name="check"></a>
## Graph storage integrity check on Wukong
This command can help you make sure the correctness of current graph storage.

1) Check the storage integrity related with both index vertex and normal vertex

```bash
wukong> gsck
INFO:     Graph storage intergity check has started on server 0
INFO:     Graph storage intergity check has started on server 1
INFO:     Server#0 has checked 47 index vertices and 110115 normal vertices.
INFO:     Server#1 has checked 49 index vertices and 110013 normal vertices.
INFO:     (average) latency: 49943499 usec
```

2) Check the storage integrity related with index vertex.

```bash
wukong> gsck -i
INFO:     Graph storage intergity check has started on server 0
INFO:     Graph storage intergity check has started on server 1
INFO:     Server#0 has checked 47 index vertices and 0 normal vertices.
INFO:     Server#1 has checked 49 index vertices and 0 normal vertices.
INFO:     (average) latency: 18493196 usec
```

3) Check the storage integrity related with normal vertex.

```bash
wukong> gsck -n
INFO:     Graph storage intergity check has started on server 0
INFO:     Graph storage intergity check has started on server 1
INFO:     Server#0 has checked 0 index vertices and 110115 normal vertices.
INFO:     Server#1 has checked 0 index vertices and 110013 normal vertices.
INFO:     (average) latency: 36454664 usec
```
