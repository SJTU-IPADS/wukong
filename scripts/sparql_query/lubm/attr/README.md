# Attribute Query Tutorial

###preparing for dataset 
you should RDF dataset that supports RDF attribute value.
For LUBM, we modify the dataset to add attribute triple .

**Modify  LUBM datasets to add attribute triple**

```
cd ${WUKONG_ROOT}/datagen
g++ -std=c++11 add_attribute.cpp -o add_attribute
mkdir nt_lubm_2
mv ~/uba1.7/uni*.nt nt_lubm_2/
./add_attribute nt_lubm_2 nt_lubm_2_attr
```
**Convert  Modified LUBM datasets to ID format**

```
./generate_data nt_lubm_2_attr  id_lubm_2
```


##  Config 
enable `global_enable_vattr`

```
global_num_proxies			4
global_num_engines			16
global_input_folder			/home/datanfs/nfs0/rdfdata/id_lubm_2/
global_data_port_base		5500
global_ctrl_port_base		9576
global_memstore_size_gb		20
global_rdma_buf_size_mb		128
global_rdma_rbf_size_mb		32
global_use_rdma				1
global_rdma_threshold		300
global_mt_threshold			8
global_enable_caching		0
global_enable_workstealing	0
global_silent 				1
global_enable_planner		0
global_generate_statistics  1
global_enable_vattr   		1
```
## Running the attribute query

```
wukong> sparql -f sparql_query/lubm/attr/lubm_attr_q2 -p sparql_query/lubm/attr -p /plan/lubm_attr_q2.fmt
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 2466 usec
INFO:     Query plan is successfully set
INFO:     The query starts from an index vertex, you could use option -m to accelerate it.
INFO:     (average) latency: 4338 usec
INFO:     (last) result size: 1889
```

with planner

```
wukong> sparql -f sparql_query/lubm/attr/lubm_attr_q2
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 67 usec
INFO:     No query plan is set
INFO:     Planning time: 513 usec
INFO:     The query starts from an index vertex, you could use option -m to accelerate it.
INFO:     (average) latency: 5131 usec
INFO:     (last) result size: 1889
```
