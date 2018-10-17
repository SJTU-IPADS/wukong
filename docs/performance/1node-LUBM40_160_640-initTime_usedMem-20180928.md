# Launch Performance on 1 Node

- Date: Sep. 26, 2018
- Author: Ning Wang, Yaozeng Zeng

## Table of Contents

- [Hardware Configuration](#hw)
- [Software Configuration](#sw)
- [Experiment Result](#res)

<a name="hw"></a>

## Hardware Configuration

### CPU

```
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                8
On-line CPU(s) list:   0-7
Thread(s) per core:    2
Core(s) per socket:    4
Socket(s):             1
NUMA node(s):          1
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 158
Model name:            Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz
Stepping:              9
CPU MHz:               800.027
CPU max MHz:           4200.0000
CPU min MHz:           800.0000
BogoMIPS:              7200.00
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              8192K
NUMA node0 CPU(s):     0-7
Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault invpcid_single pti retpoline intel_pt rsb_ctxsw tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx rdseed adx smap clflushopt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp
```

### Memory
```
              total        used        free      shared  buff/cache   available
Mem:          15911        1794       12787         648        1329       13082
Swap:         16253           4       16249
```


<a name="sw"></a>

## Software Configuration

### Wukong Version

- Version 1: 65a7310 (independent loader)
- Version 2:

### CMake Options

Item | Enabled
-- | --
RDMA | OFF
GPU | OFF
HDFS | OFF
Dynamic GStore | OFF
Versatile | OFF
64-bit ID | OFF

### Config

```
global_num_proxies          1
global_num_engines          16
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
global_enable_vattr         0
global_generate_statistics	0
```

<a name="res"></a>

## Experiment Result

Time is measured in milliseconds.
Memory is measured in MB.

### Version 1: 65a7310 (independent loader)

Dataset | str_server | load data to memory | aggregating triples | insert normal | insert index | total loading time(w/o str_server)
-- | -- | -- | -- | -- | --
lubm_40 | 1255 | 1312 | 467 | 1293 | 328 | 4755
lubm_160 | 5048 | 7352 | 1936 | 6321 | 1044 | 44600
lubm_640 | 21033 | 29389 | 8182 | 25117 | 3112 | 184828

### Version 2: 23ddee8 (refine the interface of get_edges/attr/index in dgraph and gstore. gstore models key/value to vertex/edge, and dgraph models triple/index/SPO to vertex/edge.)

Read memory information from sysinfo(inf.mem_unit*(inf.totalram + inf.totalswap - inf.freeram - inf.freeswap)).

Dataset | str_server | dgraph fields init | load raw data to memory | aggregating triples | gstore refresh | insert index | peak memory used | final memory used
-- | -- | -- | -- | -- | --
lubm_40 | 378.9 | 256.5 | 62.8 | 177.4 | 11651.5 | 33.9 | 12484.9 | 12538.0
lubm_160 | 1527.9 | 256.5 | 252.1 | 678.5 | 11538.7 | 215.7 | 14255.3 | 13945.0 
lubm_640 | 7593.1 | 256.5 | 2502.8 | 2019.4 | 8172.5 | 839.6 | 20545.6 | 19673.8

