# Question and Answer

## Table of Contents

* [Performance has a large range of error](#performance_error)

 
<a name="performance_error"></a>
## Q1 
My evaluation result has a large range of error. The result is shown following in detail. Is there something wrong?

```bash
wukong> sparql -f query/lubm/lubm_q1 -m 16 -n 10
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 362 usec
INFO:     No query plan is set
INFO:     (average) latency: 2113299 usec
INFO:     (last) result size: 2528
wukong> sparql -f query/lubm/lubm_q1 -m 16 -n 10
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 159 usec
INFO:     No query plan is set
INFO:     (average) latency: 5234887 usec
INFO:     (last) result size: 2528
wukong> sparql -f query/lubm/lubm_q1 -m 16 -n 10
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 163 usec
INFO:     No query plan is set
INFO:     (average) latency: 1793383 usec
INFO:     (last) result size: 2528
```
### Answer
Check if transparent hugepages(THP) usage is enabled.
If enabled is [never] as follows, then enable THP before evaluation. 

```bash
$ cat /sys/kernel/mm/transparent_hugepage/enabled
always madvise [never]
```
Transparent Hugepage Support can be enabled system wide 
or only enabled inside MADV_HUGEPAGE regions (to avoid the risk of consuming more memory resources)
or disabled entirely. This can be achieved with one of:

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
