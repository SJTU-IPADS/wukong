# Wukong Commands

## Table of Contents
- [Getting help information in Wukong](#help)
- [Terminating current Wukong process](#quit)
- [Configuring Wukong](#config)
- [Configuring logger in Wukong](#logger)
- [Loading RDF data into dynamic (in-memory) graph store](#load)
- [Graph storage integrity check on Wukong](#gsck)
- [Loading statistics of SPARQL query optimizer](#load-stat)
- [Storing statistics of SPARQL query optimizer](#store-stat)
- [Running SPARQL queries in single or batch mode](#sparql)
- [Emulating clients to continuously send SPARQL queries](#sparql-emu)

<a name="help"></a>

## Getting help information in Wukong

The command `help` lists the usage of common Wukong commands.

```
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

<a name="quit"></a>

## Terminating current Wukong process

The command `quit` can help you terminate the current Wukong process.

```
wukong> quit
...
```

<a name="config"></a>

## Configuring Wukong

The command `config <args> ` can help you configure Wukong in runtime and print current configuration of Wukong.

1) Print configuration

Use command `config -v` to print current configuration of Wukong.

```
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
global_enable_planner: 0
global_generate_statistics: 0
global_enable_vattr: 1
global_num_gpus: 0
global_gpu_rdma_buf_size_mb: 0
global_gpu_rbuf_size_mb: 32
global_gpu_kvcache_size_gb: 10
global_gpu_key_blk_size_mb: 16
global_gpu_value_blk_size_mb: 4
global_gpu_enable_pipeline: 1
--
the number of servers: 1
the number of threads: 3
```

2) Configuring Wukong by file

Use command `config -l <fname>` to configure Wukong by loading configuration file `<fname>`.

```
wukong> config -l newConfigFile
```

3) Configure Wukong by string

Use command `config -s <str>` to configure Wukong by string `<str>`.

>Note: You can use `&` to set several configurations in one string (e.g., item1=val1&item2=...).

```
wukong> config -l global_mt_threshold=1
wukong> config -s global_mt_threshold=2&global_silent=1
```

4) Get help message about config

Use command `config -h` or `config --help` to print help message of command `config`.

```
wukong> config -h
config <args>       run commands for configueration:
  -v                    print current config
  -l <fname>            load config items from <fname>
  -s <string>           set config items by <str> (e.g., item1=val1&item2=...)
  -h [ --help ]         help message about config
```

<a name="logger"></a>

## Configuring logger in Wukong

The command `logger <args>` can help you configure log level and check current logger level.

1) Wukong log levels

Wukong provides 7 log levels and the current logger level configuration controls the message printing in Wukong.

log level | log level name | meaning
:-:|:-:|:-:
0|LOG_EVERYTHING|Log everything.
1|LOG_DEBUG|Debugging purposes only.
2|LOG_INFO|Used for providing general useful information.
3|LOG_EMPH|Outputs as LOG_INFO, but in LOG_WARNING colors. Useful foroutputting information you want to emphasize.
4|LOG_WARNING|Logs interesting conditions which are probably not fatal.
5|LOG_ERROR|Used for errors which are recoverable within the scope of the function.
6|LOG_FATAL|Used for fatal and probably irrecoverable conditions.
7|LOG_NONE|Log nothing.

2) Log level printing

Use command `logger -v` to print current configuration of log level.

```
wukong> logger -v
loglevel: 2 (INFO)
```

3) Log level configuring

Use command `logger -s <level>` to switch current log level to `<level>`.

```
wukong> logger -s 0
set loglevel to 0 (EVERYTHING)
wukong> logger -s 1
set loglevel to 1 (DEBUG)
wukong> logger -s 2
set loglevel to 2 (INFO)
wukong> logger -s 3
set loglevel to 3 (EMPH)
wukong> logger -s 4
set loglevel to 4 (WARNING)
wukong> logger -s 5
set loglevel to 5 (ERROR)
wukong> logger -s 6
set loglevel to 6 (FATAL)
wukong> logger -s 7
set loglevel to 7 (NONE)
```

4) Get help message about logger

```
wukong> logger -h
logger <args>       run commands for logger:
  -v                    print loglevel
  -s <level>            set loglevel to <level> (e.g., DEBUG=1, INFO=2,
                        WARNING=4, ERROR=5)
  -h [ --help ]         help message about logger
```

<a name="load"></a>

## Loading RDF data into dynamic (in-memory) graph store

The command `load` can help you load data into Wukong.

1) Use command `load -d <dname>` to load new dataset from directory `<dname>`, the structure of directory is just the same as which used to initialize.

```
wukong> load -d path/to/input/id_lubm_2/
INFO:     loading ID-mapping file: path/to/input/id_lubm_2/str_normal
INFO:     loading ID-mapping file: path/to/input/id_lubm_2/str_index
INFO:     2 data files and 0 attribute files found in directory (path/to/input/id_lubm_2/) at server 0
INFO:     load 205474 triples from file path/to/input/id_lubm_2/id_uni0.nt at server 0
INFO:     load 267198 triples from file path/to/input/id_lubm_2/id_uni1.nt at server 0
INFO:     #0: 72ms for inserting into gstore
INFO:     (average) latency: 106384 usec
```

2) Add `-c` option to check and skip duplicate triples in the dataset.

```
wukong> load -c -d path/to/input/id_lubm_2/
INFO:     loading ID-mapping file: path/to/input/id_lubm_2/str_normal
INFO:     loading ID-mapping file: path/to/input/id_lubm_2/str_index
INFO:     2 data files and 0 attribute files found in directory (path/to/input/id_lubm_2/) at server 0
INFO:     load 205474 triples from file path/to/input/id_lubm_2/id_uni0.nt at server 0
INFO:     load 267198 triples from file path/to/input/id_lubm_2/id_uni1.nt at server 0
INFO:     #0: 93ms for inserting into gstore
INFO:     (average) latency: 160743 usec
```

> Note: Make sure that you have enable dynamic data loading support with parameter `-USE_DYNAMIC_GSTORE=ON`. Otherwise you will get a error message like:

```
wukong> load
ERROR:    Can't load data into static graph store.
ERROR:    You can enable it by building Wukong with -DUSE_DYNAMIC_GSTORE=ON.
```

<a name="gsck"></a>

## Graph storage integrity check on Wukong

The command `gsck` can help you make sure the correctness of current graph storage.

1) Use command `gsck` to check the storage integrity related with both index vertex and normal vertex.

```
wukong> gsck
INFO:     Graph storage intergity check has started on server 0
INFO:     Server#0 already check 5%
...
INFO:     Server#0 already check 95%
INFO:     Server#0 has checked 49 index vertices and 220128 normal vertices.
INFO:     (average) latency: 3436217 usec
```

2) Add `-i` option to check the storage integrity related with index vertex.

```
wukong> gsck -i
INFO:     Graph storage intergity check has started on server 0
INFO:     Server#0 already check 5%
...
INFO:     Server#0 already check 95%
INFO:     Server#0 has checked 49 index vertices and 0 normal vertices.
INFO:     (average) latency: 3081938 usec
```

3) Add `-n` option to check the storage integrity related with normal vertex.

```
wukong> gsck -n
INFO:     Graph storage intergity check has started on server 0
INFO:     Server#0 already check 5%
...
INFO:     Server#0 already check 95%
INFO:     Server#0 has checked 0 index vertices and 220128 normal vertices.
INFO:     (average) latency: 2094172 usec
```

<a name="load-stat"></a>

## Loading statistics of SPARQL query optimizer

The command `load-stat -f <fname>` can help you load statistics of SPARQL query optimizer from file `<fname>` .

```
wukong> load-stat -f path/to/input/id_lubm_40/statfile
INFO:     1 ms for loading statistics at server 0
```

<a name="store-stat"></a>

## Storing statistics of SPARQL query optimizer

The command `load-stat -f <fname>` can help you store statistics of SPARQL query optimizer to file `<fname>`.

```
wukong> store-stat -f newStatfile
INFO:     store statistics to file newStatfile is finished.
```

<a name="sparql"></a>

## Run SPARQL queries in single or batch mode

The command `sparql <args>` can help you run SPARQL queries.
>Note: Wukong requires a query plan provided either by user or by Wukong's planner for every query .

### Running SPARQL query in single mode

1) Run query with user defined plan

Use command `sparql -f <fname> -p <pfname>` to run query in file `<fname>` with user pre-defined planning file `<pfname>`.

```
wukong> sparql -f sparql_query/lubm/basic/lubm_q7 -p sparql_query/lubm/basic/manual_plan/lubm_q7.fmt
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 215 usec
INFO:     User-defined query plan is enabled
INFO:     The query starts from an index vertex, you could use option -m to accelerate it.
INFO:     (last) result size: 73
INFO:     (average) latency: 7344 usec
```

2) Run query with plan generated by Wukong planner

Use command `sparql -f <fname>` to run query in file `<fname>` with plan generated by Wukong planner.

>Note: Make sure the configuration `global_enable_planner` is `1` (on) to enable Wukong planner.

```
wukong> sparql -f sparql_query/lubm/basic/lubm_q7
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 138 usec
INFO:     Optimization time: 2331 usec
INFO:     The query starts from an index vertex, you could use option -m to accelerate it.
INFO:     (last) result size: 73
INFO:     (average) latency: 5128 usec
```

### Run SPARQL query in batch mode

Use command `sparql -b <fname>` to run batch queries in file `<fname>`.

```
wukong> sparql -b sparql_query/lubm/batch/batch_q1
INFO:     Batch-mode start ...
Run the command: sparql -f sparql_query/lubm/basic/lubm_q1
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 131 usec
...
INFO:     Batch-mode end.
```

### Option for SPARQL query processing
1) Add `-m <factor>` option to set the multi-threading `<factor>`(threads number) for heavy query processing.

```
wukong> sparql -f sparql_query/lubm/basic/lubm_q2 -m 6
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 109 usec
INFO:     Optimization time: 20 usec
INFO:     (last) result size: 1889
INFO:     (average) latency: 2038 usec
```

2) Add `-n <num>` option to repeat query processing `<num>` times.

```
wukong> sparql -f sparql_query/lubm/basic/lubm_q2 -m 2 -n 10
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 28 usec
INFO:     Optimization time: 6 usec
INFO:     (last) result size: 1889
INFO:     (average) latency: 258 usec
```

3) Add `-N <num>` option to do query optimization `<num>` times.

```
wukong> sparql -f sparql_query/lubm/basic/lubm_q2 -m 2 -N 2
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 91 usec
INFO:     Optimization time: 14 usec
INFO:     (last) result size: 1889
INFO:     (average) latency: 2802 usec
```

4) Add `-v <lines>` option to print at most `<lines>` of results

```
wukong> sparql -f sparql_query/lubm/basic/lubm_q2 -m 2 -v 2
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 96 usec
INFO:     Optimization time: 19 usec
INFO:     (last) result size: 1889
INFO:     The first 2 rows of results:
1: <http://www.Department6.University1.edu/Course36>	"Course36"
2: <http://www.Department6.University1.edu/Course44>	"Course44"
INFO:     (average) latency: 1812 usec
```

5) Add `-o <fname>` option to output results info `<fname>`.

```
wukong> sparql -f sparql_query/lubm/basic/lubm_q2 -m 2 -o result_file
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 99 usec
INFO:     Optimization time: 19 usec
INFO:     (last) result size: 1889
INFO:     (average) latency: 1868 usec
```

6) Add `-g` option to leverage GPU to accelerate heavy query processing.

```
wukong> sparql -f sparql_query/lubm/basic/lubm_q1 -g
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 121 usec
INFO:     Optimization time: 907 usec
INFO:     Leverage GPU to accelerate query processing.
INFO:     (last) result size: 106
INFO:     (average) latency: 1355 usec
```

<a name="sparql-emu"></a>

## Emulating clients to continuously send SPARQL queries

The command `sparql-emu <args>` can help you emulate clients to continuously send SPARQL.

### Clients emulating
Use command `sparql-emu -f <fname>` to run queries generated from temples configured by file `<fname>`. 

```
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#GraduateCourse> has 43070 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#AssistantProfessor> has 7624 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> has 1000 candidates
INFO:     Throughput: 64.9971K queries/sec
INFO:     [1sec]
INFO:     ...
INFO:     [9sec]
INFO:     Throughput: 69.4413K queries/sec
INFO:     Throughput: 69.0693K queries/sec
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4	Q5	Q6
INFO:     1	64	64	121	72	65	63
...
INFO:     99	422	424	506	438	427	433
INFO:     100	6615	13317	1827	3303	6613	3315
INFO:     Throughput: 69.0326K queries/sec
```

The sparql-emu configuration file `<fname>` controls the sparql-emu by the structure showed below.
```
$lights      $heavies
$sparql-emu_query       $load_factor
...
$sparql-emu_query       $load_factor
```

1. The `$lights` denotes the number of light queries. The `$heavies` denotes the number of heavy queries.
2. Next, there are `$lights+$heavies` lines. Each line is a `$query-emu_query $load_factor` pattern that denotes the load factor `load_factor` of `$query-emu_query`.
3. Each `$query-emu_query` is a query-emu query file written in SPARQL. The only DIFFERENCE is that Wukong uses `%` to describe all the candidates of a kind of subject or object. For example, as the query-emu query file showed below, `%ub:AssistantProfessor` describe all the candidates that belong to type `ub:AssistantProfessor`(e.g., ub:AssistantProfessor1, ub:AssistantProfessor2...).

```
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#>

SELECT ?X WHERE {
 ?X  ub:publicationAuthor  %ub:AssistantProfessor  .
	?X  rdf:type  ub:Publication  .
}

```

### Option for clients emulating

1) Add `-p <fname>` option to adopt user-defined query plans from file `<fname>` for running queries.

```
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -p sparql_query/lubm/emulator/plan_config
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#GraduateCourse> has 43070 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#AssistantProfessor> has 7624 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> has 1000 candidates
INFO:     Throughput: 74.7658K queries/sec
INFO:     Throughput: 78.0418K queries/sec
INFO:     [1sec]
INFO:     ...
INFO:     [9sec]
INFO:     Throughput: 78.5904K queries/sec
INFO:     Throughput: 78.09K queries/sec
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4	Q5	Q6
...
INFO:     100	13431	13289	13444	13224	13433	1260
INFO:     Throughput: 78.4291K queries/sec
```

2) Add `-d <sec>` option to eval `<sec>` seconds on sparql-emu(default: 10).

```
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 6
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#GraduateCourse> has 43070 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#AssistantProfessor> has 7624 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> has 1000 candidates
INFO:     Throughput: 64.6384K queries/sec
INFO:     [1sec]
INFO:     ...
INFO:     [5sec]
INFO:     Throughput: 68.8448K queries/sec
INFO:     Throughput: 68.8662K queries/sec
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4	Q5	Q6
INFO:     1	64	65	123	73	65	64
INFO:     ...
INFO:     100	2550	2555	1963	2560	2568	1288
INFO:     Throughput: 68.9155K queries/sec
```

3) Add `-w <sec>` option to warmup `<sec>` seconds before emulating(default: 5).

```
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -w 1
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#GraduateCourse> has 43070 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#AssistantProfessor> has 7624 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> has 1000 candidates
INFO:     Throughput: 64.9577K queries/sec
INFO:     Throughput: 68.8247K queries/sec
INFO:     [1sec]
INFO:     ...
INFO:     [9sec]
INFO:     Throughput: 68.9033K queries/sec
INFO:     Throughput: 68.9042K queries/sec
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4	Q5	Q6
INFO:     1	64	64	121	72	65	61
INFO:     5	87	87	138	92	87	86
INFO:     10	102	102	149	107	102	102
INFO:     ...
INFO:     100	13373	13369	13374	13347	13327	4944
INFO:     Throughput: 69.0021K queries/sec
```

4) Add `-n <num>` option to keep `<num>` queries being processed during emulating(default: 20).

```
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -n 5
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#GraduateCourse> has 43070 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#AssistantProfessor> has 7624 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#Department> has 799 candidates
INFO:     Parsing a SPARQL template is done.
INFO:     <http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> has 1000 candidates
INFO:     Throughput: 46.5694K queries/sec
INFO:     Throughput: 49.713K queries/sec
INFO:     [1sec]
INFO:     ...
INFO:     [9sec]
INFO:     Throughput: 49.9632K queries/sec
INFO:     Throughput: 49.8509K queries/sec
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4	Q5	Q6
INFO:     1	46	47	105	58	48	45
INFO:     ...
INFO:     100	4388	1641	4403	1205	2161	1629
INFO:     Throughput: 49.7075K queries/sec
```