# This file lists reference output of example test cases.

## batch_q1

* command: sparql -b sparql_query/lubm/batch/batch_q1

```
INFO:     Batch-mode start ...
Run the command: sparql -f sparql_query/lubm/basic/lubm_q1
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 73 usec
INFO:     Optimization time: 324 usec
INFO:     The query starts from an index vertex, you could use option -m to accelerate it.
INFO:     (last) result size: 106
INFO:     (average) latency: 2987 usec

Run the command: sparql -f sparql_query/lubm/basic/lubm_q2 -v 5
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 23 usec
INFO:     Optimization time: 6 usec
INFO:     The query starts from an index vertex, you could use option -m to accelerate it.
INFO:     (last) result size: 43291
INFO:     The first 5 rows of results: 
1: <http://www.Department15.University3.edu/Course39>	"Course39"	
2: <http://www.Department15.University3.edu/Course3>	"Course3"	
3: <http://www.Department15.University3.edu/Course40>	"Course40"	
4: <http://www.Department15.University3.edu/Course28>	"Course28"	
5: <http://www.Department15.University3.edu/Course17>	"Course17"	
INFO:     (average) latency: 13249 usec

Run the command: sparql -f sparql_query/lubm/basic/lubm_q3 -m 8 -v 5
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 40 usec
INFO:     Optimization time: 192 usec
INFO:     (last) result size: 0
INFO:     The first 0 rows of results: 
INFO:     (average) latency: 7142 usec

Run the command: sparql -f sparql_query/lubm/basic/lubm_q4 -n 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 38 usec
INFO:     Optimization time: 10 usec
INFO:     (last) result size: 10
INFO:     (average) latency: 42 usec

INFO:     Batch-mode end.
```

## batch_q2

* command: sparql -b sparql_query/lubm/batch/batch_q2

```
INFO:     Batch-mode start ...
ERROR:    Failed to run the command: sparql sparql_query/lubm/basic/lubm_q1
ERROR:    only support single sparql query in batch mode (e.g., sparql -f ...)
ERROR:    Failed to run the command: sparql -b sparql_query/lubm/batch/batch_q1
ERROR:    only support single sparql query in batch mode (e.g., sparql -f ...)
Run the command: sparql -f sparql_query/lubm/basic/lubm_q3
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 76 usec
INFO:     Optimization time: 216 usec
INFO:     The query starts from an index vertex, you could use option -m to accelerate it.
INFO:     (last) result size: 0
INFO:     (average) latency: 3241 usec

INFO:     Batch-mode end.
```

## batch_q3

* command: sparql -b sparql_query/lubm/batch/batch_q3

```
INFO:     Batch-mode start ...
Run the command: sparql -f sparql_query/lubm/basic/lubm_q2 -n 3
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 33 usec
INFO:     Optimization time: 8 usec
INFO:     The query starts from an index vertex, you could use option -m to accelerate it.
INFO:     (last) result size: 43291
INFO:     (average) latency: 9283 usec

Run the command: sparql -f sparql_query/lubm/basic/lubm_q3 -n 4
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 45 usec
INFO:     Optimization time: 181 usec
INFO:     The query starts from an index vertex, you could use option -m to accelerate it.
INFO:     (last) result size: 0
INFO:     (average) latency: 1385 usec

Run the command: sparql -f sparql_query/lubm/basic/lubm_q4 -n 5
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 37 usec
INFO:     Optimization time: 8 usec
INFO:     (last) result size: 10
INFO:     (average) latency: 455 usec

Run the command: sparql -f sparql_query/lubm/basic/lubm_q5 -n 10
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 37 usec
INFO:     Optimization time: 5 usec
INFO:     (last) result size: 10
INFO:     (average) latency: 244 usec

INFO:     Batch-mode end.
```

## batch_q4

* command: sparql -b sparql_query/lubm/batch/batch_q4

```
INFO:     Batch-mode start ...
Run the command: sparql -f sparql_query/lubm/basic/no_such_file1
ERROR:    Query file not found: sparql_query/lubm/basic/no_such_file1
ERROR:    [0|0] Failed to run the command: sparql -f sparql_query/lubm/basic/no_such_file1 

Input 'help' command to get more information.

Run the command: sparql -f sparql_query/lubm/basic/no_such_file2
ERROR:    Query file not found: sparql_query/lubm/basic/no_such_file2
ERROR:    [0|0] Failed to run the command: sparql -f sparql_query/lubm/basic/no_such_file2 

Input 'help' command to get more information.

Run the command: sparql -f sparql_query/lubm/basic/no_such_file3
ERROR:    Query file not found: sparql_query/lubm/basic/no_such_file3
ERROR:    [0|0] Failed to run the command: sparql -f sparql_query/lubm/basic/no_such_file3 

Input 'help' command to get more information.

INFO:     Batch-mode end.
```