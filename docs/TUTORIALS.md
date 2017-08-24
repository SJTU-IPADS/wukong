# Wukong Tutorials

## Table of Contents

* [Deploying Wukong on a local cluster](#cluster)
* [Downloading LUBM sample dataset](#data)
* [Configuring and running Wukong](#run)
* [Processing SPARQL queries on Wukong](#query)


<a name="cluster"></a>
## Deploying Wukong on a local cluster

Install Wukong's dependencies (e.g., OpenMPI), using instructions in the [INSTALL.md](./INSTALL.md#dep), on your master node (one of your cluster machines, e.g., `node0.some.domain`) and copy necessities to the rest machines.

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
$ wget http://ipads.se.sjtu.edu.cn/projects/wukong/id_lubm_2.tar.gz
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
global_num_proxies          1
global_num_engines          2
global_input_folder         path/to/input/id_lubm_2
global_load_minimal_index   0
global_data_port_base       5500
global_ctrl_port_base       9576
global_memstore_size_gb     20
global_rdma_buf_size_mb     128
global_rdma_rbf_size_mb     32
global_use_rdma             1
global_rdma_threshold       300
global_mt_threshold         2
global_enable_caching       0
global_enable_workstealing  0
global_silent               0
global_enable_planner       1
$cat core.bind
# One node per line (NOTE: the empty line means to skip a node)
0 1 2
```

The detail explanation of above `config` file can be found in [INSTALL.md](./INSTALL.md#run)

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
>
```


<a name="query"></a>
## Processing SPARQL queries

1) Wukong commands.

```bash
> help
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
        -b <file>           a set of queries configured by <file>
```

2) run a single SPARQL query.

There are query examples in `$WUKONG_ROOT/scripts/query`. For example, input `sparql -f query/lubm_q2` to run the query `lubm_q2`.

```bash
> sparql -f sparql_query/lubm_q7 -v 5
(average) latency: 3660 usec
(last) result size: 73
The first 5 rows of results: 
1:  <http://www.Department8.University1.edu/FullProfessor5> <http://www.Department8.University1.edu/UndergraduateStudent204>  <http://www.Department8.University1.edu/Course9>  
2:  <http://www.Department14.University1.edu/FullProfessor6>  <http://www.Department14.University1.edu/UndergraduateStudent141> <http://www.Department14.University1.edu/Course7> 
3:  <http://www.Department4.University0.edu/FullProfessor0> <http://www.Department4.University0.edu/UndergraduateStudent312>  <http://www.Department4.University0.edu/Course1>  
4:  <http://www.Department7.University1.edu/FullProfessor9> <http://www.Department7.University1.edu/UndergraduateStudent8>  <http://www.Department7.University1.edu/Course14> 
5:  <http://www.Department8.University1.edu/FullProfessor7> <http://www.Department8.University1.edu/UndergraduateStudent47> <http://www.Department8.University1.edu/Course13>
> sparql -f sparql_query/lubm_q4 -n 1000
(average) latency: 199 usec
(last) result size: 10
>
```


2) show and change the configuration of Wukong at runtime.

```bash
> config -v
------ global configurations ------
the number of engines: 2
the number of proxies: 1
global_input_folder: /home/datanfs/nfs0/rdfdata/id_lubm_2/
global_load_minimal_index: 0
global_data_port_base: 5700
global_ctrl_port_base: 9776
global_memstore_size_gb: 20
global_rdma_buf_size_mb: 128
global_rdma_rbf_size_mb: 64
global_use_rdma: 1
global_enable_caching: 0
global_enable_workstealing: 0
global_rdma_threshold: 300
global_mt_threshold: 2
global_silent: 0
global_enable_planner: 1
--
the number of servers: 2
the number of threads: 3
> config -s global_use_rdma=0
> sparql -f query/lubm_q4 -n 1000
(average) latency: 1128 usec
(last) result size: 10
```

