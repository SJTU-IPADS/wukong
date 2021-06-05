# T-sparql extension

#### Generate temporal RDF data

```bash
$cd ${WUKONG_ROOT}/datagen
$g++ -std=c++11 add_timestamp.cpp -o add_timestamp
$./add_timestamp /home/sl/wukong/datagen/id_lubm_3
```

At this time, the triples in the `id_lubm_2` directory will become quintuples of the following format:

```
205039  23      204607  1337335004      443247361
205039  23      204699  1544924311      1107554302
205041  1       21      87840508        1023763187
205041  5       131895  119772761       1484157313
205041  22      204527  1365067893      740859041
205041  14      205042  1039202204      1085642062
205041  15      131086  912734282       632043196
205041  23      204538  550812267       367607343
...
```

#### Run Wukong's temporal RDF mode

```bash
$cd ${WUKONG_ROOT}/scripts
$./build.sh -DTRDF_MODE=ON
$./run.sh 3
```

#### Run an temporal-extended SPARQL statement

```bash
$cat sparql_query/lubm/time/time1
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xs: <http://www.w3.org/2001/XMLSchema#>
PREFIX ub: <http://swat.cse.lehigh.edu/onto/univ-bench.owl#>

SELECT ?X ?Y ?s ?e FROM SNAPSHOT <2007-08-12T22:22:22> WHERE {
        [?s, ?e] ?X  ub:memberOf  ?Y  .
}
```

All query statements in the directory `${WUKONG_ROOT}/scripts/sparql_query/lubm/time/` are related to temporal RDF.

```bash
wukong> sparql -f sparql_query/lubm/time/time1 -v 5
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 94 usec
INFO:     Optimization time: 22 usec
INFO:     The query starts from an index vertex, you could use option -m to accelerate it.
INFO:     (last) result row num: 66565 , col num:2
INFO:     The first 5 rows of results:
1: <http://www.Department10.University1.edu/UndergraduateStudent2>      <http://www.Department10.University1.edu>      1990-01-11T15:57:19                                                  2009-09-18T14:00:31
2: <http://www.Department10.University1.edu/UndergraduateStudent5>      <http://www.Department10.University1.edu>      1974-05-26T23:20:56                                                  2011-03-15T22:37:17
3: <http://www.Department10.University1.edu/UndergraduateStudent9>      <http://www.Department10.University1.edu>      1971-09-16T18:01:56                                                  2010-01-14T21:34:56
4: <http://www.Department10.University1.edu/UndergraduateStudent10>     <http://www.Department10.University1.edu>      1983-01-23T06:48:03                                                  2014-04-03T11:43:57
5: <http://www.Department10.University1.edu/UndergraduateStudent13>     <http://www.Department10.University1.edu>      2006-04-16T14:41:31                                                  2007-08-21T01:14:41
INFO:     (average) latency: 16735 usec

wukong> sparql -f sparql_query/lubm/time/time2 -v 5
(last) result row num: 43291 , col num:2
1: <http://www.Department15.University3.edu/Course20>   "Course20"      2008-05-07T02:45:49     1973-05-20T09:40:16
2: <http://www.Department20.University3.edu/Course20>   "Course20"      1983-06-13T03:51:28     1982-05-12T22:03:32 
3: <http://www.Department8.University3.edu/Course20>    "Course20"      2006-06-01T23:33:38     1984-10-08T11:37:47
4: <http://www.Department10.University3.edu/Course20>   "Course20"      2011-09-21T17:04:24     1972-01-17T16:22:14
5: <http://www.Department11.University3.edu/Course20>   "Course20"      1991-01-16T22:55:30     1992-07-12T12:54:16

wukong> sparql -f sparql_query/lubm/time/time3 -v 5
(last) result row num: 906 , col num:2
1: <http://www.Department19.University3.edu/Lecturer4>  "Lecturer4"     "Lecturer4"     <http://www.Department19.University3.edu/Lecturer4>
2: <http://www.Department15.University36.edu/Lecturer4> "Lecturer4"     "Lecturer4"     <http://www.Department15.University36.edu/Lecturer4>
3: <http://www.Department20.University36.edu/Lecturer4> "Lecturer4"     "Lecturer4"     <http://www.Department20.University36.edu/Lecturer4>
4: <http://www.Department11.University17.edu/Lecturer4> "Lecturer4"     "Lecturer4"     <http://www.Department11.University17.edu/Lecturer4>
5: <http://www.Department10.University17.edu/Lecturer4> "Lecturer4"     "Lecturer4"     <http://www.Department10.University17.edu/Lecturer4>

wukong> sparql -f sparql_query/lubm/time/time4 -v 5
(last) result row num: 21135 , col num:2
1: <http://www.Department15.University3.edu/Lecturer4>  "Lecturer4"     "Lecturer4"     <http://www.Department15.University3.edu/Lecturer4>
2: <http://www.Department20.University3.edu/Lecturer4>  "Lecturer4"     "Lecturer4"     <http://www.Department20.University3.edu/Lecturer4>
3: <http://www.Department8.University3.edu/Lecturer4>   "Lecturer4"     "Lecturer4"     <http://www.Department8.University3.edu/Lecturer4>
4: <http://www.Department10.University3.edu/Lecturer4>  "Lecturer4"     "Lecturer4"     <http://www.Department10.University3.edu/Lecturer4>
5: <http://www.Department11.University3.edu/Lecturer4>  "Lecturer4"     "Lecturer4"     <http://www.Department11.University3.edu/Lecturer4>

wukong> sparql -f sparql_query/lubm/time/time5 -v 5
(last) result row num: 43291 , col num:2
1: <http://www.Department15.University3.edu/Course20>   "Course20"      2008-05-07T02:45:49     1973-05-20T09:40:16
2: <http://www.Department20.University3.edu/Course20>   "Course20"      1983-06-13T03:51:28     1982-05-12T22:03:32
3: <http://www.Department8.University3.edu/Course20>    "Course20"      2006-06-01T23:33:38     1984-10-08T11:37:47
4: <http://www.Department10.University3.edu/Course20>   "Course20"      2011-09-21T17:04:24     1972-01-17T16:22:14
5: <http://www.Department11.University3.edu/Course20>   "Course20"      1991-01-16T22:55:30     1992-07-12T12:54:16
```

