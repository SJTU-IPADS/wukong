# Data Tutorial

Wukong adopts an ID-Triples format to represent RDF. This tutorial use [LUBM](http://swat.cse.lehigh.edu/projects/lubm) (SWAT Projects - the Lehigh University Benchmark) as an example to introduce how to convert other RDF formats to the ID-Triples format.

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

When there are many RDF/XML format files to convert, you can use our tool: convert.py.
Assume the directory of RDF/XML format files is rdf_dir. The output directory is nt_dir. The number of files to convert is 40.

```bash
$python convert.py -i rdf_dir -s 40 -o nt_dir
```

Or, you can do it manually.

**Step1.1 Download Apache Jena**

We use a command-line tool called riot (provided by Apache Jena) to convert any other RDF formats to N-Triples.

> Note: You can also choose whatever tools you like as long as they can convert RDF data to the N-Triples format.

[Download Apache Jena](https://jena.apache.org/download/index.cgi) and uncompress it. Assume the path to Jena is `JENA_HOME`. It is not necessary to set the environment variable.

The code below is an example of this step.

```bash
$cd ~;
$wget http://mirrors.tuna.tsinghua.edu.cn/apache/jena/binaries/apache-jena-3.13.1.tar.gz
$tar zxvf apache-jena-3.13.1.tar.gz
```

**Step 1.2 Convert to N-Triples**

Commands below shows the basic usage of riot, the RDF format converting tool. Read the official doc [here](http://jena.apache.org/documentation/io/#command-line-tools).

Assume we want to convert RDF/XML format files to N-Triples, and the file name is `SRC.owl`. The converted N-Triples format data will be stored in `OUT.nt`. `--syntax` is optional, riot can infer the source format.

```bash
$JENA_HOME/bin/riot --syntax=RDF/XML --output=N-Triples SRC.owl >> OUT.nt
```

For example, assume we want to convert LUBM dataset (2 Universities) to N-Triples format. Assume the path to the input LUBM dataset is `INPUT` and the path to the output N-Triples data is `OUTPUT`.


```bash
$JENA_HOME/bin/riot --output=N-Triples INPUT/University0_*.owl >> OUTPUT/uni0.nt
$JENA_HOME/bin/riot --output=N-Triples INPUT/University1_*.owl >> OUTPUT/uni1.nt
```

### Step 2: Convert N-Triples to ID-Triples

Use python tool, the convert project will be tried for times until all encoding completes.
It can recover from failure automatically. Suppose input directory is nt_dir and output directory is id_dir.

```bash
$python encode.py -i nt_dir -o id_dir.
```

Or, you can do it manually.

**Step 2.1: Compile the code**

```
$g++ -std=c++11 generate_data.cpp -o generate_data
```

**Step 2.2: Convert**

Arguments of ./generate_data are the input directory(nt format directory) and the output directory (id format directory). There exists extra log files in output directory. Remove it manually if necessary.

For example, assume we want to convert LUBM dataset (2 Universities) from N-Triples format to ID Triples format. Assume the path to the input data is `INPUT` and the path to the output ID Triples data is `OUTPUT`.

```
$./generate_data INPUT OUTPUT
Process No.1 input file: uni1.nt.
Process No.2 input file: uni0.nt.
#total_vertex = 58455
#normal_vertex = 58421
#index_vertex = 34
$cd OUTPUT
$rm log*
$ls
id_uni0.nt  id_uni1.nt  str_index  str_normal
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

Assume the path to the input data is `INPUT` and the output path is `OUTPUT`.

```
$./add_attribute INPUT OUTPUT
```
Then transform data from NT format to ID format, like [Convert data](#convert)
