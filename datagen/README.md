# Data Tutorial

Wukong adopts an ID-Triples format to represent RDF. This tutorial use [LUBM](http://swat.cse.lehigh.edu/projects/lubm) (SWAT Projects - the Lehigh University Benchmark) as an example to introduce how to convert other RDF formats to the ID-Triples format. Tutorial to generate LUBM dataset can refer to [INSTALLs](../docs/INSTALL.md), step 1 of preparing RDF datasets.

## Table of Contents
* [Data pattern](#pattern)
* [Convert data](#convert)
* [Add attribute data](#attribute)

<a name="pattern"></a>

## Data Pattern

#### Normal triple pattern
Each row in LUBM dataset with N-Triples format (e.g., `uni0.nt`) consists of subject (S), predicate (P), object (O), and '.', like`<http://www.University97.edu> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> .`.

Each row in LUBM dataset with ID-Triples format (e.g., `id_uni0.nt`) consists of the 3 IDs (non-negative integer), like `132323  1  16`. `str_index` and `str_normal` store the mapping from string to ID for index (e.g., predicate) and normal (e.g., subject and object) entities respectively.

#### Attribute triple pattern
If there are `attribute triple pattern.`
Attribute triple pattern  with the qNT  format like that `eg:JeremyCarroll  eg:ageInYears  "40"^^xds:integer  .  `

Attribute triple pattern with the ID format(e.g.,attr_uni0.nt) consits of 4 IDs(subject_id, pred_id , type_id , obj_id_) like `132324 2 1 18`, the mapping of attribute predicate will be stored in str_attr_index.

<a name="convert"></a>

## Convert data

### Step 1: Convert RDF formats to N-Triples

Use the python tool (convert_rdf.py) to convert RDF format data to N-Triples format data.

```bash
$python convert_rdf.py -h
usage:
 python convert_rdf.py [Options] -i <input directory> -o <output directory> -s <data size> -p <input prefix> -w <output prefix>
Options:
  -r/--remove    Remove input files in input directory
```

Assume we want to convert LUBM dataset (2 Universities) to N-Triples format. Assume the directory of RDF/XML format files is `/wukongdata/lubm_rdf`. The output directory is `/wukongdata/nt_lubm`. The prefix of input file is `University` and prefix of output file is `uni`.

```bash
$python convert_rdf.py -i /wukongdata/lubm_rdf -o /wukongdata/nt_lubm -p University -w uni -s 2
...
Convert from RDF data to NT format data is done.
$ls /wukongdata/nt_lubm
uni0.nt  uni1.nt
```

Or, you can do it manually.

**Step1.1 Download Apache Jena**

We use a command-line tool called riot (provided by Apache Jena) to convert any other RDF formats to N-Triples.

> Note: You can also choose whatever tools you like as long as they can convert RDF data to the N-Triples format.

[Download Apache Jena](https://jena.apache.org/download/index.cgi) and uncompress it. Assume the path to Jena is `JENA_HOME`. It is not necessary to set the environment variable.

The code below is an example of this step.

```bash
$wget http://mirrors.tuna.tsinghua.edu.cn/apache/jena/binaries/apache-jena-3.17.0.tar.gz
$tar zxvf apache-jena-3.17.0.tar.gz
```

**Step 1.2 Convert to N-Triples**

Commands below shows the basic usage of riot, the RDF format converting tool. Read the official doc [here](http://jena.apache.org/documentation/io/#command-line-tools).

Assume we want to convert RDF/XML format files to N-Triples, and the file name is `SRC.owl`. The converted N-Triples format data will be stored in `OUT.nt`. `--syntax` is optional, riot can infer the source format.

```bash
$JENA_HOME/bin/riot --syntax=RDF/XML --output=N-Triples SRC.owl >> OUT.nt
```

For example, assume we want to convert LUBM dataset (2 Universities) to N-Triples format. Assume the path to the input LUBM dataset is `/wukongdata/lubm_rdf` and the path to the output N-Triples data is `/wukongdata/nt_lubm`.


```bash
$JENA_HOME/bin/riot --output=N-Triples /wukongdata/lubm_rdf/University0_*.owl >> /wukongdata/nt_lubm/uni0.nt
$JENA_HOME/bin/riot --output=N-Triples /wukongdata/lubm_rdf/University1_*.owl >> /wukongdata/nt_lubm/uni1.nt
```

### Step 2: Convert N-Triples to ID-Triples

Use python tool(convert_nt.py). The generate_data project can recover from failure automatically.
It will be tried automatically for times until convert of all the data completes.

Assume we want to convert LUBM dataset (2 Universities) from N-Triples format to ID Triples format.
Assume input directory is `/wukongdata/nt_lubm` and output directory is `/wukongdata/id_lubm`.
Since we do not add attribute data, so result files with `attr` are all empty.

```bash
$python convert_nt.py -i /wukongdata/nt_lubm -o /wukongdata/id_lubm
Process No.0 input file: uni1.nt.
Process No.1 input file: uni0.nt.
#total_vertex = 58418
#normal_vertex = 58386
#index_vertex = 32
#attr_vertex = 0
Convert from N-Triples to ID-Triples is done.
$ls /wukongdata/id_lubm
attr_uni0.nt  attr_uni1.nt  id_uni0.nt  id_uni1.nt  str_attr_index  str_index  str_normal
```

Or, you can do it manually.

**Step 2.1: Compile the code**

```
$g++ -std=c++11 -O2 generate_data.cpp -o generate_data -I ./
```

**Step 2.2: Convert**

Arguments of ./generate_data are the input directory(nt format directory) and the output directory (id format directory). There exists extra log files(file `log` and file `log_commit`) in output directory. File `log_commit` indicates the complishment of generate_data project. Remove it manually if necessary.

For example, assume we want to convert LUBM dataset (2 Universities) from N-Triples format to ID Triples format. Assume the path to the input data is `/wukongdata/nt_lubm` and the path to the output ID Triples data is `/wukongdata/id_lubm`.

```
$./generate_data /wukongdata/nt_lubm /wukongdata/id_lubm
Process No.0 input file: uni1.nt.
Process No.1 input file: uni0.nt.
#total_vertex = 58418
#normal_vertex = 58386
#index_vertex = 32
#attr_vertex = 0
$rm /wukongdata/id_lubm/log*
$ls /wukongdata/id_lubm
attr_uni0.nt  attr_uni1.nt  id_uni0.nt  id_uni1.nt  str_attr_index  str_index  str_normal
```
<a name="attribute"></a>

## Add attribute data

### Step 1: Compile the code

```
$g++ -std=c++11 add_attribute.cpp -o add_attribute
```

### Step 2: Add attribute data
Modify the LUBM dataset to add attribute data.

Arguments of ./add_attribute are the input directory(nt format directory) and the output directory(attribute triple directory).

Assume the path to the input data is `/wukongdata/nt_lubm` and the output path is `/wukongdata/nt_lubm_attr`.

```
$./add_attribute /wukongdata/nt_lubm /wukongdata/nt_lubm_attr
Process No.1 input file: uni1.nt.
Process No.2 input file: uni0.nt.
```
Then transform data from NT format to ID format, like [Convert data](#convert)
