# Performance (single node, dynamic gstore, jemalloc, versatile)

###### Date: Nov. 12, 2018 

###### Author: Rong Chen


## Table of Contents

* [Hardware configuration](#hw)
* [Software configuration](#sw)
* [Dataset and workload](#dw)
* [Experimantal results (OPT-enabled)](#opt)
* [Experimantal results (OSDI16 Plan)](#osdi16)

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

##### Gitlab Version: @b33e3e7

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

**Queries**: `sparql_query/lubm/basic/lubm_{q1,q2,q3,q4,q5,q6,q7}`, `sparql_query/lubm/emulator/mix_config`


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
INFO:     ------ global configurations ------
INFO:     the number of proxies: 4
INFO:     the number of engines: 16
INFO:     global_input_folder: /home/datanfs/nfs0/rdfdata/id_lubm_2560/
INFO:     global_data_port_base: 5700
INFO:     global_ctrl_port_base: 9776
INFO:     global_memstore_size_gb: 40
INFO:     global_rdma_buf_size_mb: 256
INFO:     global_rdma_rbf_size_mb: 128
INFO:     global_use_rdma: 1
INFO:     global_enable_caching: 0
INFO:     global_enable_workstealing: 0
INFO:     global_stealing_pattern: 0
INFO:     global_rdma_threshold: 300
INFO:     global_mt_threshold: 16
INFO:     global_silent: 1
INFO:     global_enable_planner: 1
INFO:     global_generate_statistics: 0
INFO:     global_enable_vattr: 0
INFO:     global_num_gpus: 1
INFO:     global_gpu_rdma_buf_size_mb: 64
INFO:     global_gpu_rbuf_size_mb: 32
INFO:     global_gpu_kvcache_size_gb: 10
INFO:     global_gpu_key_blk_size_mb: 16
INFO:     global_gpu_value_blk_size_mb: 4
INFO:     global_gpu_enable_pipeline: 1
INFO:     --
INFO:     the number of servers: 1
INFO:     the number of threads: 20
wukong> 
wukong> sparql -f sparql_query/lubm/basic/lubm_q1 -m 16 -n 10
INFO:     Parsing time: 56 usec
INFO:     Planning time: 285 usec
INFO:     (average) latency: 207018 usec
INFO:     (last) result size: 2528
wukong> sparql -f sparql_query/lubm/basic/lubm_q2 -m 16 -n 10
INFO:     Parsing time: 40 usec
INFO:     Planning time: 14 usec
INFO:     (average) latency: 63909 usec
INFO:     (last) result size: 2765067
wukong> sparql -f sparql_query/lubm/basic/lubm_q3 -m 16 -n 10
INFO:     Parsing time: 52 usec
INFO:     Planning time: 176 usec
INFO:     (average) latency: 188862 usec
INFO:     (last) result size: 0
wukong> sparql -f sparql_query/lubm/basic/lubm_q4 -n 1000
INFO:     Parsing time: 38 usec
INFO:     Planning time: 7 usec
INFO:     (average) latency: 24 usec
INFO:     (last) result size: 10
wukong> sparql -f sparql_query/lubm/basic/lubm_q5 -n 1000
INFO:     Parsing time: 28 usec
INFO:     Planning time: 5 usec
INFO:     (average) latency: 18 usec
INFO:     (last) result size: 10
wukong> sparql -f sparql_query/lubm/basic/lubm_q6 -n 1000
INFO:     Parsing time: 34 usec
INFO:     Planning time: 8 usec
INFO:     (average) latency: 76 usec
INFO:     (last) result size: 125
wukong> sparql -f sparql_query/lubm/basic/lubm_q7 -m 16 -n 10
INFO:     Parsing time: 51 usec
INFO:     Planning time: 168 usec
INFO:     (average) latency: 153958 usec
INFO:     (last) result size: 112559
wukong> sparql -f sparql_query/lubm/basic/lubm_q8 -n 1000 
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 56 usec
INFO:     No query plan is set
INFO:     Planning time: 10 usec
INFO:     (average) latency: 99 usec
INFO:     (last) result size: 8569
wukong> sparql -f sparql_query/lubm/basic/lubm_q9 -n 1000
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 23 usec
INFO:     No query plan is set
INFO:     Planning time: 4 usec
INFO:     (average) latency: 21 usec
INFO:     (last) result size: 730
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 1
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 20  21  132 37  24  175 
INFO:     5 25  26  145 43  29  196 
INFO:     10    31  32  152 49  35  207 
INFO:     15    35  36  158 52  39  217 
INFO:     20    38  39  163 55  41  227 
INFO:     25    39  40  167 56  43  236 
INFO:     30    41  42  172 58  45  244 
INFO:     35    43  44  177 60  46  252 
INFO:     40    44  45  181 61  48  259 
INFO:     45    46  47  186 63  49  265 
INFO:     50    47  48  190 64  50  272 
INFO:     55    49  50  196 65  52  279 
INFO:     60    50  51  201 67  53  285 
INFO:     65    51  52  207 68  55  294 
INFO:     70    53  54  214 70  56  303 
INFO:     75    55  56  221 72  58  316 
INFO:     80    57  57  229 75  60  330 
INFO:     85    59  60  237 78  62  346 
INFO:     90    62  63  246 81  65  362 
INFO:     95    66  67  258 86  70  381 
INFO:     96    68  68  261 87  71  387 
INFO:     97    69  70  265 89  72  393 
INFO:     98    71  72  270 91  74  399 
INFO:     99    74  75  277 94  77  408 
INFO:     100   1850    1330    2181    234 1626    773 
INFO:     Throughput: 63.9418K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 5
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 48  49  206 65  50  268 
INFO:     5 104 104 244 118 107 309 
INFO:     10    122 123 261 134 125 331 
INFO:     15    133 134 275 143 135 346 
INFO:     20    141 141 285 150 143 360 
INFO:     25    147 148 295 156 149 372 
INFO:     30    153 153 303 161 154 383 
INFO:     35    158 158 312 166 159 395 
INFO:     40    162 163 320 170 164 405 
INFO:     45    167 167 329 175 168 416 
INFO:     50    171 172 338 180 173 427 
INFO:     55    176 177 347 184 178 439 
INFO:     60    180 181 357 189 182 452 
INFO:     65    186 186 368 194 187 467 
INFO:     70    191 192 380 201 193 483 
INFO:     75    197 198 392 208 200 502 
INFO:     80    205 205 406 217 207 519 
INFO:     85    214 215 423 232 218 540 
INFO:     90    232 233 449 262 238 566 
INFO:     95    299 302 497 328 308 614 
INFO:     96    329 328 517 355 336 630 
INFO:     97    366 366 548 393 372 661 
INFO:     98    413 412 599 441 419 718 
INFO:     99    486 482 675 503 488 797 
INFO:     100   1697    1678    1098    957 2674    1220    
INFO:     Throughput: 70.2683K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 10
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 166 164 230 175 168 281 
INFO:     5 217 216 264 221 218 326 
INFO:     10    239 238 293 243 240 353 
INFO:     15    254 254 311 258 254 373 
INFO:     20    265 266 326 270 266 390 
INFO:     25    276 276 339 280 277 406 
INFO:     30    285 286 350 290 286 421 
INFO:     35    294 294 361 298 295 437 
INFO:     40    302 303 372 307 303 454 
INFO:     45    310 311 382 314 311 473 
INFO:     50    318 319 392 323 320 494 
INFO:     55    326 327 404 331 328 514 
INFO:     60    335 335 418 339 336 532 
INFO:     65    343 344 433 348 345 549 
INFO:     70    353 353 449 358 354 569 
INFO:     75    363 363 469 368 365 595 
INFO:     80    375 375 490 379 376 630 
INFO:     85    389 389 516 393 390 662 
INFO:     90    405 406 564 410 407 698 
INFO:     95    433 433 643 437 434 749 
INFO:     96    442 443 661 449 443 763 
INFO:     97    458 459 686 471 460 783 
INFO:     98    506 511 714 530 510 808 
INFO:     99    600 607 767 622 610 867 
INFO:     100   1918    1976    2049    1805    1884    1390    
INFO:     Throughput: 70.1843K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 20
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 148 151 243 154 146 305 
INFO:     5 377 377 417 381 379 431 
INFO:     10    429 430 474 436 432 486 
INFO:     15    462 463 512 467 464 525 
INFO:     20    489 489 539 494 490 558 
INFO:     25    512 512 564 516 513 584 
INFO:     30    532 532 587 536 533 609 
INFO:     35    551 551 607 555 551 631 
INFO:     40    568 569 626 572 569 651 
INFO:     45    585 586 644 589 586 671 
INFO:     50    602 603 663 605 602 691 
INFO:     55    618 619 680 621 619 712 
INFO:     60    635 636 699 638 635 731 
INFO:     65    652 652 717 655 653 751 
INFO:     70    670 670 738 673 670 774 
INFO:     75    690 690 759 692 690 798 
INFO:     80    711 712 784 713 712 827 
INFO:     85    736 737 812 738 736 863 
INFO:     90    767 767 847 770 768 900 
INFO:     95    809 810 893 812 811 959 
INFO:     96    820 823 907 824 822 978 
INFO:     97    834 837 924 839 836 1014    
INFO:     98    853 855 948 860 854 1100    
INFO:     99    882 884 1020    895 884 1267    
INFO:     100   2613    2596    2086    2562    2613    2114    
INFO:     Throughput: 71.5456K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 30
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 203 207 271 213 204 367 
INFO:     5 549 550 592 555 550 607 
INFO:     10    628 629 678 634 628 692 
INFO:     15    677 677 729 681 679 745 
INFO:     20    715 717 771 720 719 792 
INFO:     25    751 751 806 755 753 828 
INFO:     30    782 781 839 786 784 865 
INFO:     35    811 811 869 814 813 896 
INFO:     40    838 839 897 841 840 925 
INFO:     45    864 865 924 867 866 952 
INFO:     50    889 889 949 893 890 981 
INFO:     55    913 913 975 918 914 1009    
INFO:     60    938 937 1000    943 938 1036    
INFO:     65    962 962 1026    968 964 1066    
INFO:     70    988 988 1053    993 989 1095    
INFO:     75    1015    1015    1081    1020    1017    1127    
INFO:     80    1045    1043    1113    1050    1046    1163    
INFO:     85    1079    1078    1150    1084    1080    1203    
INFO:     90    1122    1122    1195    1129    1124    1253    
INFO:     95    1185    1184    1260    1188    1187    1318    
INFO:     96    1203    1201    1277    1205    1204    1337    
INFO:     97    1223    1221    1298    1225    1224    1358    
INFO:     98    1248    1247    1327    1251    1250    1388    
INFO:     99    1286    1287    1366    1296    1289    1424    
INFO:     100   2742    2516    2591    2558    2761    2460    
INFO:     Throughput: 71.5581K queries/sec
wukong> 
wukong> 
wukong> q 
```

<br>
<a name="osdi16"></a>
## Experimantal results (OSDI16 Plan)

#### Summary

> Query folder: `sparql_query/lubm/basic`   
> Plan folder: `sparql_query/lubm/basic/osdi16_plan` (Q1-Q7) 
> and `sparql_query/lubm/basic/maunal_plan` (Q8-Q9)  

| Workload | Latency (us) | #R (lines) | TH | Query   | Plan (OSDI16) |
| :------: | -----------: | ---------: | -: | :------ | :------------ |
| Q1       | 387,611      | 2528       | 16 | lubm_q1 | lubm_q1.fmt   |
| Q2       |  64,163      | 2,765,067  | 16 | lubm_q2 | lubm_q2.fmt   |
| Q3       | 188,721      | 0          | 16 | lubm_q3 | lubm_q3.fmt   |
| Q4       |      24      | 10         |  1 | lubm_q4 | lubm_q4.fmt   |
| Q5       |      18      | 10         |  1 | lubm_q5 | lubm_q5.fmt   |
| Q6       |      76      | 125        |  1 | lubm_q6 | lubm_q6.fmt   |
| Q7       | 309,334      | 112,559    | 16 | lubm_q7 | lubm_q7.fmt   |
| Q8       |      97      | 8,569      |  1 | lubm_q8 | lubm_q8.fmt   |
| Q9       |      21      | 730        |  1 | lubm_q9 | lubm_q9.fmt   |


> Query folder: `sparql_query/lubm/emulator/`  
> Plan folder: `sparql_query/lubm/emulator/osdi16_plan`  

| Workload | Thpt (q/s) | Configuration    | Query      | Plan (OSDI16) |
| :------: | ---------: | :--------------- | :--------- | :------------ |
| A1-A6    | 64.7264K   | -d 5 -w 1 -p 1   | mix_config | plan_config   |
| A1-A6    | 69.6764K   | -d 5 -w 1 -p 5   | mix_config | plan_config   |
| A1-A6    | 70.2949K   | -d 5 -w 1 -p 10  | mix_config | plan_config   |
| A1-A6    | 71.3340K   | -d 5 -w 1 -p 20  | mix_config | plan_config   |
| A1-A6    | 69.8407K   | -d 5 -w 1 -p 30  | mix_config | plan_config   |

#### Detail

```bash
wukong> config -s global_enable_planner=0
wukong> config -v
INFO:     ------ global configurations ------
INFO:     the number of proxies: 4
INFO:     the number of engines: 16
INFO:     global_input_folder: /home/datanfs/nfs0/rdfdata/id_lubm_2560/
INFO:     global_data_port_base: 5700
INFO:     global_ctrl_port_base: 9776
INFO:     global_memstore_size_gb: 40
INFO:     global_rdma_buf_size_mb: 256
INFO:     global_rdma_rbf_size_mb: 128
INFO:     global_use_rdma: 1
INFO:     global_enable_caching: 0
INFO:     global_enable_workstealing: 0
INFO:     global_stealing_pattern: 0
INFO:     global_rdma_threshold: 300
INFO:     global_mt_threshold: 16
INFO:     global_silent: 1
INFO:     global_enable_planner: 0
INFO:     global_generate_statistics: 0
INFO:     global_enable_vattr: 0
INFO:     global_num_gpus: 1
INFO:     global_gpu_rdma_buf_size_mb: 64
INFO:     global_gpu_rbuf_size_mb: 32
INFO:     global_gpu_kvcache_size_gb: 10
INFO:     global_gpu_key_blk_size_mb: 16
INFO:     global_gpu_value_blk_size_mb: 4
INFO:     global_gpu_enable_pipeline: 1
INFO:     --
INFO:     the number of servers: 1
INFO:     the number of threads: 20
wukong> 
wukong> sparql -f sparql_query/lubm/basic/lubm_q1 -p sparql_query/lubm/basic/osdi16_plan/lubm_q1.fmt -m 16 -n 10
INFO:     Parsing time: 57 usec
INFO:     (average) latency: 387611 usec
INFO:     (last) result size: 2528
wukong> sparql -f sparql_query/lubm/basic/lubm_q2 -p sparql_query/lubm/basic/osdi16_plan/lubm_q2.fmt -m 16 -n 10
INFO:     Parsing time: 35 usec
INFO:     (average) latency: 64163 usec
INFO:     (last) result size: 2765067
wukong> sparql -f sparql_query/lubm/basic/lubm_q3 -p sparql_query/lubm/basic/osdi16_plan/lubm_q3.fmt -m 16 -n 10
INFO:     Parsing time: 56 usec
INFO:     (average) latency: 188721 usec
INFO:     (last) result size: 0
wukong> sparql -f sparql_query/lubm/basic/lubm_q4 -p sparql_query/lubm/basic/osdi16_plan/lubm_q4.fmt -n 1000
INFO:     Parsing time: 37 usec
INFO:     (average) latency: 24 usec
INFO:     (last) result size: 10
wukong> sparql -f sparql_query/lubm/basic/lubm_q5 -p sparql_query/lubm/basic/osdi16_plan/lubm_q5.fmt -n 1000
INFO:     Parsing time: 40 usec
INFO:     (average) latency: 18 usec
INFO:     (last) result size: 10
wukong> sparql -f sparql_query/lubm/basic/lubm_q6 -p sparql_query/lubm/basic/osdi16_plan/lubm_q6.fmt -n 1000
INFO:     Parsing time: 61 usec
INFO:     (average) latency: 76 usec
INFO:     (last) result size: 125
wukong> sparql -f sparql_query/lubm/basic/lubm_q7 -p sparql_query/lubm/basic/osdi16_plan/lubm_q7.fmt -m 16 -n 10
INFO:     Parsing time: 51 usec
INFO:     (average) latency: 309334 usec
INFO:     (last) result size: 112559
wukong> sparql -f sparql_query/lubm/basic/lubm_q8 -p sparql_query/lubm/basic/manual_plan/lubm_q8.fmt -n 1000
INFO:     Parsing time: 38 usec
INFO:     (average) latency: 97 usec
INFO:     (last) result size: 8569
wukong> sparql -f sparql_query/lubm/basic/lubm_q9 -p sparql_query/lubm/basic/manual_plan/lubm_q9.fmt -n 1000
INFO:     Parsing time: 33 usec
INFO:     (average) latency: 21 usec
INFO:     (last) result size: 730
wukong> 
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -p sparql_query/lubm/emulator/plan_config -d 5 -w 1 -n 1
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 20  21  135 36  23  171 
INFO:     5 26  27  147 44  30  196 
INFO:     10    33  34  153 50  36  207 
INFO:     15    36  37  159 53  40  217 
INFO:     20    39  40  164 55  42  227 
INFO:     25    40  42  169 57  44  237 
INFO:     30    42  43  173 59  46  246 
INFO:     35    44  45  178 60  47  253 
INFO:     40    45  46  183 62  49  259 
INFO:     45    47  48  187 63  50  266 
INFO:     50    48  49  191 65  52  273 
INFO:     55    50  51  197 66  53  280 
INFO:     60    51  52  202 68  54  287 
INFO:     65    52  53  208 69  56  295 
INFO:     70    54  55  215 71  57  304 
INFO:     75    56  57  222 73  59  316 
INFO:     80    57  59  229 76  61  330 
INFO:     85    60  61  238 78  63  346 
INFO:     90    63  64  246 82  67  363 
INFO:     95    67  68  257 87  71  382 
INFO:     96    68  69  260 88  72  387 
INFO:     97    70  71  264 90  74  391 
INFO:     98    72  73  268 92  76  397 
INFO:     99    75  76  278 95  79  406 
INFO:     100   393 1299    491 206 1668    2833    
INFO:     Throughput: 64.7264K queries/sec
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -p sparql_query/lubm/emulator/plan_config -d 5 -w 1 -n 5
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 77  82  201 99  82  249 
INFO:     5 117 117 229 128 119 283 
INFO:     10    131 131 244 140 133 303 
INFO:     15    139 140 256 147 141 317 
INFO:     20    146 147 265 154 148 329 
INFO:     25    152 152 274 159 153 340 
INFO:     30    157 157 283 164 158 349 
INFO:     35    162 162 291 169 163 359 
INFO:     40    166 167 299 174 168 370 
INFO:     45    171 171 307 179 173 380 
INFO:     50    176 176 315 184 177 392 
INFO:     55    181 181 324 189 183 403 
INFO:     60    186 186 333 195 188 415 
INFO:     65    191 192 343 201 193 428 
INFO:     70    198 198 354 207 200 441 
INFO:     75    204 205 366 215 206 456 
INFO:     80    212 212 379 225 214 473 
INFO:     85    222 222 394 239 225 491 
INFO:     90    239 239 416 265 242 518 
INFO:     95    289 290 455 320 296 560 
INFO:     96    310 312 474 340 317 573 
INFO:     97    338 340 497 365 344 591 
INFO:     98    379 378 532 409 387 629 
INFO:     99    443 439 588 473 451 693 
INFO:     100   2683    2515    1072    2517    2873    2903    
INFO:     Throughput: 69.6764K queries/sec
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -p sparql_query/lubm/emulator/plan_config -d 5 -w 1 -n 10
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 101 103 219 105 97  277 
INFO:     5 211 213 271 216 213 327 
INFO:     10    238 238 298 243 238 356 
INFO:     15    254 254 316 259 254 376 
INFO:     20    266 267 331 271 267 392 
INFO:     25    277 278 344 282 278 407 
INFO:     30    287 287 354 291 287 422 
INFO:     35    296 296 366 301 296 439 
INFO:     40    304 305 376 309 305 458 
INFO:     45    312 313 387 317 313 479 
INFO:     50    320 321 398 325 321 499 
INFO:     55    329 329 411 334 329 518 
INFO:     60    337 337 424 342 338 534 
INFO:     65    346 347 439 351 347 552 
INFO:     70    356 356 456 361 356 570 
INFO:     75    366 367 475 372 367 596 
INFO:     80    377 379 497 383 379 624 
INFO:     85    391 393 524 397 392 659 
INFO:     90    409 410 579 415 409 697 
INFO:     95    437 438 660 447 438 744 
INFO:     96    448 449 682 460 448 757 
INFO:     97    466 469 704 488 466 777 
INFO:     98    515 517 739 549 513 799 
INFO:     99    609 613 801 646 609 858 
INFO:     100   2184    2313    1989    1833    1805    2106    
INFO:     Throughput: 70.2949K queries/sec
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -p sparql_query/lubm/emulator/plan_config -d 5 -w 1 -n 20
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 187 187 283 197 190 359 
INFO:     5 373 374 431 381 376 456 
INFO:     10    426 428 483 434 430 505 
INFO:     15    459 460 516 466 462 541 
INFO:     20    484 485 544 492 487 572 
INFO:     25    506 506 567 512 508 598 
INFO:     30    525 525 589 531 527 620 
INFO:     35    542 542 608 549 545 638 
INFO:     40    559 559 627 565 562 658 
INFO:     45    575 575 645 581 578 677 
INFO:     50    592 591 663 597 594 696 
INFO:     55    608 608 679 613 610 715 
INFO:     60    624 624 697 630 626 736 
INFO:     65    641 641 716 647 643 758 
INFO:     70    659 659 737 665 660 780 
INFO:     75    679 678 760 685 680 806 
INFO:     80    701 701 785 707 702 837 
INFO:     85    727 727 813 734 728 872 
INFO:     90    759 758 848 767 759 914 
INFO:     95    805 803 900 813 804 981 
INFO:     96    818 816 915 826 818 1012    
INFO:     97    834 832 933 843 835 1052    
INFO:     98    855 855 971 866 857 1150    
INFO:     99    897 899 1086    919 901 1333    
INFO:     100   24503   24223   16713   21546   23838   23941   
INFO:     Throughput: 71.334K queries/sec
wukong>
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -p sparql_query/lubm/emulator/plan_config -d 5 -w 1 -n 30
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 545 546 594 547 547 601 
INFO:     5 633 633 686 638 635 710 
INFO:     10    684 686 744 689 687 772 
INFO:     15    722 722 783 727 724 812 
INFO:     20    751 752 815 758 754 848 
INFO:     25    778 779 843 785 781 878 
INFO:     30    802 804 869 810 805 907 
INFO:     35    826 827 893 833 828 932 
INFO:     40    848 849 919 855 850 958 
INFO:     45    870 871 942 876 872 981 
INFO:     50    892 893 967 898 894 1006    
INFO:     55    915 915 991 921 916 1032    
INFO:     60    939 939 1016    946 940 1058    
INFO:     65    964 963 1043    971 964 1085    
INFO:     70    990 990 1071    998 991 1114    
INFO:     75    1019    1019    1103    1026    1020    1145    
INFO:     80    1050    1050    1137    1057    1051    1181    
INFO:     85    1086    1086    1178    1094    1087    1219    
INFO:     90    1129    1129    1227    1138    1131    1268    
INFO:     95    1191    1192    1294    1200    1192    1336    
INFO:     96    1208    1208    1310    1218    1209    1356    
INFO:     97    1229    1229    1332    1237    1230    1377    
INFO:     98    1256    1257    1359    1265    1259    1404    
INFO:     99    1302    1301    1408    1307    1304    1460    
INFO:     100   4318    4331    4032    4077    4115    3858    
INFO:     Throughput: 69.8407K queries/sec
wukong> 
wukong> q 
```
