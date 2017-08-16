# Wukong Tutorials

## Table of Contents

- [Deploying Wukong on a Cluster](cluster)
- Prepare RDF data
- Query with Wukong

<a name="cluster"></a>
## Deploying Wukong on a Cluster

### Step 0: Install Wukong on one of your cluster nodes.

Install Wukong, using instructions in the [README.md](README.md), on your master node (one of your cluster nodes, e.g., `node0.some.domain`).



### Step 1: Copy all dependencies to all nodes

1) Update the file in the script directory called “mpd.hosts” with the names of all the cluster nodes participate in the computation.

For example:

```bash
cat $WUKONG_ROOT/scripts/mpd.hosts
node0.some.domain
node1.some.domain
...
node15.some.domain
```

2) Setup password-less SSH between the master node and all other machines.

Verify it is possible to ssh without password between any pairs of machines. These [instructions](http://www.linuxproblem.org/art_9.html) explain how to setup ssh without passswords.

Before proceeding, verify that this is setup correctly; check that the following connects to the remote machine without prompting for a password:

```bash
ssh node1.some.domain
# from node0.some.domain
```

3) Copy all dependences to the rest of the machines:

```bash
cd $WUKONG_ROOT/scripts
./syncdeps.sh ../deps/dependencies mpd.hosts
```



### Step 2: Prepare LUBM data

Taking LUBM as an example, this tutorial introduces how to prepare RDF data for Wukong.

1) Install LUBM-generator

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

2) Install Jena

```bash
cd ~
wget http://archive.apache.org/dist/jena/binaries/apache-jena-2.7.4.zip
unzip apache-jena-2.7.4.zip -d .
export JENA_HOME=~/apache-jena-2.7.4/
```

3) Generate LUBM-1 (.NT format)

```bash
cd uba1.7
java -cp ./classes/ edu.lehigh.swat.bench.uba.Generator -univ 1 -onto http://swat.cse.lehigh.edu/onto/univ-bench.owl
find . -type f -name "University*.owl" -exec $JENA_HOME/bin/rdfcat -out N-TRIPLE -x {} >> uni0.nt \;
# Now we get NT format of LUBM data
# each row consists of S P O and '.'
```

4) Generate ID format LUBM

```bash
cd ${WUKONG_ROOT}/datagen;
g++ -std=c++11 generate_data.cpp -o generate_data
mkdir lubm_raw_1
mv ~/uba1.7/uni0.nt lubm_raw_1/
./generate_data lubm_raw_1 id_lubm_1
```

Now you have got the RDF data stored in `id_lubm_1/`.



### Step 3: Run Wukong

1) Configure Wukong by `$WUKONG_ROOT/scripts/config`.

```bash
cat $WUKONG_ROOT/scripts/config
global_num_engines          16
global_num_proxies          1
global_input_folder         path/to/input/id_lubm_2
global_load_minimal_index   0
global_data_port_base       5500
global_ctrl_port_base       9576
global_memstore_size_gb     20
global_rdma_buf_size_mb     128
global_rdma_rbf_size_mb     32
global_use_rdma             0
global_rdma_threshold       300
global_mt_threshold         16
global_enable_caching       0
global_enable_workstealing  0
global_silent               0
global_enable_planner       0
```

Modify the `global_input_folder` to the path of your RDF data. If the path starts with "hdfs:", Wukong will load data from HDFS.

If you'd like to print or dump results, set `global_load_minimal_index` and `global_silent` to 0.

If you'd like to enable RDMA, set `global_use_rdma` to 1.


2) Sync Wukong files to all machines

```bash
cd ${WUKONG_ROOT}/scripts
./sync.sh
```


3) Launch Wukong server with a builtin local console

```bash
cd ${WUKONG_ROOT}/scripts
./run.sh 16
. . . 
>
```


### Step 4: Run SPARQL queries on Wukong

1) Wukong commands 

```bash
>help
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

2) run a single SPARQL query

There are query examples in `$WUKONG_ROOT/scripts/query`. For example, input `sparql -f query/lubm_q2` to run the query `lubm_q2`.

```bash
> sparql -f query/lubm_q2 -n 10 -v 10
xxx
>
```

3)
Batch query is used to test throughput. There is an example in `$WUKONG_ROOT/scripts/batch`. Input `sparql -b batch/mix_config` to run.

```bash
> sparql -f query/lubm_q1 -n 10 -v 10
xxx
>
```


