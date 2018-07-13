# Data Tutorial
## Table of Contents
* [Transform data](#transfrom)
* [Add attribute data](#attribute)


<a name="transform"></a>
## Transform data
`step 1` : compile the code

```
$g++ -std=c++11 generate_data.cpp -o generate_data
```

`step 2` : transform the data format, from NT format to ID format.

arguments of ./generate_data are the input directory(nt format directory) and the output directory (id format directory) 

```
$./generate_data nt_lubm_2 id_lubm_2
Process No.1 input file: uni1.nt.
Process No.2 input file: uni0.nt.
#total_vertex = 58455
#normal_vertex = 58421
#index_vertex = 34
$ls id_lubm_2
id_uni0.nt  id_uni1.nt  str_index  str_normal
```
#### Normal triple pattern
Each row in LUBM dataset with NT format (e.g., `uni0.nt`) consists of subject (S), predicate (P), object (O), and '.', like`<http://www.University97.edu> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> .`.

Each row in LUBM dataset with ID format (e.g., `id_uni0.nt`) consists of the 3 IDs (non-negative integer), like `132323  1  16`. `str_index` and `str_normal` store the mapping from string to ID for index (e.g., predicate) and normal (e.g., subject and object) entities respectively.
#### Attribute triple pattern
If there are `attribute triple pattern.`
Attribute triple pattern  with the qNT  format like that `eg:JeremyCarroll  eg:ageInYears  "40"^^xds:integer  .  `

Attribute triple pattern with the ID format(e.g.,attr_uni0.nt) consits of 4 IDs(subject_id, pred_id , type_id , obj_id_) like `132324 2 1 18`, the mapping of attribute predicate will be stored in str_attr_index.



<a name="attribute"></a>
## Add attribute data
`step 1` : compile the code

```
$g++ -std=c++11 add_attribute.cpp -o add_attribute
```

`step 2` : 
 modify the LUBM dataset to add attribute data 

arguments of ./add_attribute are the input directory(nt format directory) and the output directory(attribute triple directory) 

```
$./add_attribute nt_lubm_2 nt_lubm_2_attr
```
and then transform data from NT format to ID format, like [Transform data](#transfrom)


