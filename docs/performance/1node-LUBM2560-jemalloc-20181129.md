# Performance (single node, dynamic gstore, jemalloc, versatile)

###### Date: Nov. 29, 2018 

###### Author: Rong Chen


## Table of Contents

* [Hardware configuration](#hw)
* [Software configuration](#sw)
* [Dataset and workload](#dw)
* [Experimantal results (OPT-enabled)](#opt)

<br>
<a name="hw"></a>
## Hardware configuration
#### CPU
| N   | S x C x T  | Processor                        | 
| :-: | :--------: | :------------------------------- | 
| 1   | 2 x 10 x 2 | Intel Xeon E5-2650 v3 processors |

#### NIC
| N x P | Bandwidth | NIC                                        | 
| :---: | :-------: | :----------------------------------------- | 
| 1 x 2 | 56Gbps    | ConnectX-3 MCX353A IB NICs via PCIe 3.0 x8 |
| 1 x 1 | 10Gbps    | Intel X520 Ethernet NIC                    |

#### Switch
| N x P | Bandwidth | Switch                           | 
| :---: | :-------: | :------------------------------- | 
| 1 x / | 40Gbps    | Mellanox IS5025 IB Switch        |
| 1 x / | 10Gbps    | Force10 S4810P Ethernet Switch   |


<br>
<a name="sw"></a>
## Software configuration

##### Gitlab Version: @b67abe3

#### Configuration

```bash
$cd $WUKONG_ROOT/scripts
$cat config
global_num_proxies          4
global_num_engines          16
global_input_folder         /home/datanfs/nfs0/rdfdata/id_lubm_2560/
global_data_port_base       5700
global_ctrl_port_base       9776
global_memstore_size_gb     40
global_rdma_buf_size_mb     256
global_rdma_rbf_size_mb     128
global_use_rdma             1
global_rdma_threshold       300
global_mt_threshold         16
global_enable_caching       0
global_enable_workstealing  0
global_silent               1
global_enable_planner       1
global_generate_statistics  0
global_enable_vattr         0
$ 
$cat core.bind
# One node per line (NOTE: the empty line means to skip a node)
# One node per line (NOTE: the empty line means to skip a node)
0 1  4  5  6  7  8  9 10 11 
2 3 12 13 14 15 16 17 18 19 
```

#### Building and running command

```bash
$./build.sh -DUSE_RDMA=ON -DUSE_GPU=OFF -DUSE_HADOOP=OFF -DUSE_JEMALLOC=ON -DUSE_DYNAMIC_GSTORE=ON -DUSE_VERSATILE=ON -DUSE_DTYPE_64BIT=OFF
$./run.sh 1
```

<br>
<a name="dw"></a>
## Dataset and workload

**Dataset**: Leigh University Benchmark with 2,560 University (**LUBM-2560**)

**Queries**: `sparql_query/lubm/basic/lubm_{q1-12}`, `sparql_query/lubm/emulator/mix_config`


<br>
<a name="opt"></a>
## Experimantal results (optimizer-enable)

#### Summary

> Query folder: `sparql_query/lubm/basic`  

| Workload | OPT (us) | Latency (us) | #R (lines) | TH | Query   |
| :------: | -------: |------------: | ---------: | -: | :------ |
| Q1       | 285      | 207,018      | 2528       | 16 | lubm_q1 |
| Q2       |  14      |  63,909      | 2,765,067  | 16 | lubm_q2 |
| Q3       | 176      | 188,862      | 0          | 16 | lubm_q3 |
| Q4       |   7      |      24      | 10         |  1 | lubm_q4 |
| Q5       |   5      |      18      | 10         |  1 | lubm_q5 |
| Q6       |   8      |      76      | 125        |  1 | lubm_q6 |
| Q7       | 168      | 153,958      | 112,559    | 16 | lubm_q7 |
| Q8       |  10      |      97      | 8,569      |  1 | lubm_q8 |
| Q9       |   4      |      21      | 730        |  1 | lubm_q9 |

> Query folder: `sparql_query/lubm/emulator` 

| Workload | Thpt (q/s) | Configuration    | Config     |
| :------: | ---------: | :--------------- | :--------- |
| A1-A6    | 63.9418K   | -d 5 -w 1 -p 1   | mix_config |
| A1-A6    | 70.2683K   | -d 5 -w 1 -p 5   | mix_config |
| A1-A6    | 70.1843K   | -d 5 -w 1 -p 10  | mix_config |
| A1-A6    | 71.5456K   | -d 5 -w 1 -p 20  | mix_config |
| A1-A6    | 71.5581K   | -d 5 -w 1 -p 30  | mix_config |


#### Detail

```bash
wukong> config -v
------ global configurations ------
the number of proxies: 4
the number of engines: 16
global_input_folder: /home/datanfs/nfs0/rdfdata/id_lubm_2560/
global_data_port_base: 5700
global_ctrl_port_base: 9776
global_memstore_size_gb: 40
global_rdma_buf_size_mb: 256
global_rdma_rbf_size_mb: 128
global_use_rdma: 1
global_enable_caching: 0
global_enable_workstealing: 0
global_stealing_pattern: 0
global_rdma_threshold: 300
global_mt_threshold: 16
global_silent: 1
global_enable_planner: 1
global_generate_statistics: 0
global_enable_vattr: 0
--
the number of servers: 1
the number of threads: 20
wukong> 
wukong> sparql -f sparql_query/lubm/basic/lubm_q1 -m 16 -n 10 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 51 usec
INFO:     Optimization time: 112 usec
INFO:     (last) result size: 2528
INFO:     (average) latency: 220218 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q2 -m 16 -n 10 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 33 usec
INFO:     Optimization time: 1 usec
INFO:     (last) result size: 2765067
INFO:     (average) latency: 67675 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q3 -m 16 -n 10 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 50 usec
INFO:     Optimization time: 110 usec
INFO:     (last) result size: 0
INFO:     (average) latency: 198343 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q4 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 39 usec
INFO:     Optimization time: 1 usec
INFO:     (last) result size: 10
INFO:     (average) latency: 23 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q5 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 45 usec
INFO:     Optimization time: 0 usec
INFO:     (last) result size: 10
INFO:     (average) latency: 18 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q6 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 33 usec
INFO:     Optimization time: 1 usec
INFO:     (last) result size: 125
INFO:     (average) latency: 73 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q7 -m 16 -n 10 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 48 usec
INFO:     Optimization time: 21 usec
INFO:     (last) result size: 112559
INFO:     (average) latency: 160842 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q8 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 28 usec
INFO:     Optimization time: 0 usec
INFO:     (last) result size: 8569
INFO:     (average) latency: 96 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q9 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 23 usec
INFO:     Optimization time: 0 usec
INFO:     (last) result size: 730
INFO:     (average) latency: 20 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q10 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 32 usec
INFO:     Optimization time: 0 usec
INFO:     (last) result size: 5
INFO:     (average) latency: 20 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q11 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 24 usec
INFO:     Optimization time: 0 usec
INFO:     (last) result size: 1
INFO:     (average) latency: 17 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q12 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 31 usec
INFO:     Optimization time: 1 usec
INFO:     (last) result size: 3101
INFO:     (average) latency: 117 usec
wukong> 
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 1
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 20  21  140 39  24  178 
INFO:     5 27  28  151 46  31  201 
INFO:     10    33  34  159 51  37  213 
INFO:     15    36  37  165 53  39  224 
INFO:     20    38  39  170 55  41  234 
INFO:     25    40  41  175 57  43  245 
INFO:     30    41  42  179 59  45  254 
INFO:     35    43  44  184 61  46  262 
INFO:     40    44  45  189 62  48  270 
INFO:     45    46  47  194 64  49  276 
INFO:     50    47  48  199 65  50  283 
INFO:     55    48  49  205 67  52  290 
INFO:     60    50  51  212 68  53  298 
INFO:     65    51  52  219 70  54  306 
INFO:     70    52  54  229 71  56  316 
INFO:     75    54  55  239 73  58  330 
INFO:     80    56  57  251 75  59  345 
INFO:     85    58  59  266 78  62  363 
INFO:     90    61  62  288 81  65  379 
INFO:     95    65  66  328 86  69  398 
INFO:     96    66  67  339 88  70  403 
INFO:     97    68  69  353 89  71  409 
INFO:     98    70  71  376 91  73  417 
INFO:     99    73  75  408 96  76  425 
INFO:     100   3620    2728    4146    219 2138    2203    
INFO:     Throughput: 62.9786K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 5
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 62  63  201 80  64  250 
INFO:     5 111 111 232 123 113 287 
INFO:     10    126 126 249 135 128 308 
INFO:     15    135 136 260 143 137 323 
INFO:     20    142 143 270 149 143 335 
INFO:     25    148 148 280 154 149 347 
INFO:     30    153 153 289 159 154 357 
INFO:     35    158 158 297 164 159 366 
INFO:     40    162 163 305 168 163 376 
INFO:     45    166 167 314 173 168 386 
INFO:     50    171 172 322 177 172 395 
INFO:     55    175 176 332 182 177 408 
INFO:     60    180 181 344 187 181 419 
INFO:     65    185 186 356 192 186 433 
INFO:     70    190 191 369 198 192 447 
INFO:     75    196 197 384 205 198 461 
INFO:     80    203 205 403 215 205 479 
INFO:     85    213 215 428 228 216 499 
INFO:     90    229 233 464 257 234 523 
INFO:     95    292 297 525 321 294 564 
INFO:     96    317 321 545 344 317 579 
INFO:     97    350 354 573 377 352 608 
INFO:     98    395 399 621 424 396 652 
INFO:     99    465 467 701 492 463 740 
INFO:     100   1837    1811    1637    1536    1815    1121    
INFO:     Throughput: 70.5171K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 10
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 123 121 232 139 121 283 
INFO:     5 213 214 272 218 214 336 
INFO:     10    237 237 298 241 237 363 
INFO:     15    252 253 317 256 253 385 
INFO:     20    264 265 332 269 265 403 
INFO:     25    274 275 344 279 275 418 
INFO:     30    284 284 356 288 284 433 
INFO:     35    292 293 367 297 293 451 
INFO:     40    300 301 379 305 301 470 
INFO:     45    308 309 389 313 309 491 
INFO:     50    316 316 400 320 316 510 
INFO:     55    324 324 411 328 324 528 
INFO:     60    331 331 424 336 332 547 
INFO:     65    339 339 439 344 340 567 
INFO:     70    348 348 457 354 349 587 
INFO:     75    357 357 477 363 358 611 
INFO:     80    367 368 506 374 368 639 
INFO:     85    380 380 544 386 381 665 
INFO:     90    396 396 600 403 397 699 
INFO:     95    425 424 661 436 427 747 
INFO:     96    437 436 678 452 439 765 
INFO:     97    463 459 701 494 466 787 
INFO:     98    529 526 727 562 536 816 
INFO:     99    616 616 789 639 620 881 
INFO:     100   1994    2016    1602    1713    1864    1277    
INFO:     Throughput: 70.872K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 20
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 205 207 342 206 203 399 
INFO:     5 409 411 461 417 411 477 
INFO:     10    452 453 511 460 453 526 
INFO:     15    480 482 546 488 481 560 
INFO:     20    503 505 574 511 504 590 
INFO:     25    523 524 598 529 523 616 
INFO:     30    540 542 619 546 542 638 
INFO:     35    557 559 638 563 558 659 
INFO:     40    573 575 657 579 574 678 
INFO:     45    588 590 676 595 589 697 
INFO:     50    603 606 695 610 604 716 
INFO:     55    619 621 715 625 620 736 
INFO:     60    634 636 735 641 635 754 
INFO:     65    650 652 758 657 652 777 
INFO:     70    667 669 786 674 668 798 
INFO:     75    686 687 818 692 686 823 
INFO:     80    707 708 859 713 707 852 
INFO:     85    731 732 921 737 731 890 
INFO:     90    762 763 1044    769 762 947 
INFO:     95    811 814 1368    818 811 1247    
INFO:     96    828 832 1428    835 828 1315    
INFO:     97    850 855 1487    861 853 1393    
INFO:     98    899 911 1591    913 907 1502    
INFO:     99    1192    1234    1844    1242    1213    1682    
INFO:     100   4313    4183    3571    4860    3632    4389    
INFO:     Throughput: 70.3908K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 30
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 526 527 562 529 526 575 
INFO:     5 619 620 667 625 620 678 
INFO:     10    671 671 721 677 673 748 
INFO:     15    710 711 762 715 712 794 
INFO:     20    742 744 799 747 745 832 
INFO:     25    771 773 830 777 774 864 
INFO:     30    798 800 856 803 801 893 
INFO:     35    823 825 882 828 826 921 
INFO:     40    847 849 908 852 849 947 
INFO:     45    870 872 933 875 871 973 
INFO:     50    893 895 957 897 893 999 
INFO:     55    915 916 980 920 916 1023    
INFO:     60    937 938 1005    941 937 1047    
INFO:     65    959 960 1027    963 960 1072    
INFO:     70    982 983 1054    986 982 1098    
INFO:     75    1007    1007    1082    1010    1007    1127    
INFO:     80    1033    1034    1110    1037    1034    1157    
INFO:     85    1065    1064    1141    1069    1065    1192    
INFO:     90    1103    1103    1180    1109    1104    1233    
INFO:     95    1159    1160    1238    1166    1160    1295    
INFO:     96    1174    1175    1250    1183    1175    1309    
INFO:     97    1194    1195    1271    1203    1195    1328    
INFO:     98    1221    1220    1297    1230    1221    1359    
INFO:     99    1262    1260    1343    1273    1259    1401    
INFO:     100   2820    2631    2700    2658    3143    3000    
INFO:     Throughput: 70.5501K queries/sec
wukong> 
wukong> 
wukong> q 
```
