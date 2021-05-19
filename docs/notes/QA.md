# Question and Answer

## Table of Contents

* [Performance has a large range of error](#THP)

 
<a name="THP"></a>
## Q1: Enable the transparent hugepage (THP)
The results of SPARQL queries on my machine has a large difference to the results in docs/performance and/or research papers. How to tune the performance of SPARQL queries on Wukong?

```bash
wukong> sparql -f sparql_query/lubm/basic/lubm_q1 -m 16 -n 10
...
INFO:     (average) latency: 2113299 usec
INFO:     (last) result size: 2528
wukong> sparql -f sparql_query/lubm/basic/lubm_q1 -m 16 -n 10
...
INFO:     (average) latency: 5234887 usec
INFO:     (last) result size: 2528
wukong> sparql -f sparql_query/lubm/basic/lubm_q1 -m 16 -n 10
...
INFO:     (average) latency: 1793383 usec
INFO:     (last) result size: 2528
```
### Answer
Please check if transparent hugepages (THP) is enabled.
If it has been disabled (i.e., [never]), then you should enable THP before running Wukong. 

```bash
$ cat /sys/kernel/mm/transparent_hugepage/enabled
always madvise [never]
```
Transparent Hugepage Support can be enabled in system wide or only enabled inside `MADV_HUGEPAGE` regions (to avoid the risk of consuming more memory resources) or disabled entirely. This can be achieved with one of:

```bash
$ echo always > /sys/kernel/mm/transparent_hugepage/enabled
$ echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
$ echo never > /sys/kernel/mm/transparent_hugepage/enabled
```

Perhaps the command fails as following.

```bash
$ echo always > /sys/kernel/mm/transparent_hugepage/enabled
-bash: enabled: Permission denied
```

Then you can try one of these commands.

```bash
$ sudo bash -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"
$ sudo bash -c "echo madvise > /sys/kernel/mm/transparent_hugepage/enabled"
$ sudo bash -c "echo never > /sys/kernel/mm/transparent_hugepage/enabled"
```
