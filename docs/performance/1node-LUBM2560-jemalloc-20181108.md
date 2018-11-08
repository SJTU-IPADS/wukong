# Performance (single node, dynamic gstore, jemalloc, versatile)

###### Date: Nov. 8, 2018 

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

##### Gitlab Version: @4c60c2f

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

**Queries**: `query/lubm/lubm_{q1,q2,q3,q4,q5,q6,q7}`, `query/lubm/emulator/mix_config`


<br>
<a name="opt"></a>
## Experimantal results (optimizer-enable)

#### Summary

> query folder: `sparql_query/lubm/basic` 

| Workload | OPT (us) | Latency (us) | #R (lines) | TH | Query   |
| :------: | -------: |-----------: | ---------: | -: | :------ |
| Q1       | 285      | 207,018      | 2528       | 16 | lubm_q1 |
| Q2       |  14      |  63,909      | 2,765,067  | 16 | lubm_q2 |
| Q3       | 176      | 188,862      | 0          | 16 | lubm_q3 |
| Q4       |   7      |      24      | 10         |  1 | lubm_q4 |
| Q5       |   5      |      18      | 10         |  1 | lubm_q5 |
| Q6       |   8      |      76      | 125        |  1 | lubm_q6 |
| Q7       | 168      | 153,958      | 112,559    | 16 | lubm_q7 |

> query folder: `sparql_query/lubm/emulator`

| Workload | Thpt (q/s) | Configuration    | Config     |
| :------: | ---------: | :--------------- | :--------- |
| A1-A6    | 61.7696K   | -d 5 -w 1 -p 1   | mix_config |
| A1-A6    | 70.2683K   | -d 5 -w 1 -p 5   | mix_config |
| A1-A6    | 69.7533K   | -d 5 -w 1 -p 10  | mix_config |
| A1-A6    | 70.8965K   | -d 5 -w 1 -p 20  | mix_config |
| A1-A6    | 67.0482K   | -d 5 -w 1 -p 30  | mix_config |


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
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -p 1
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 20  21  153 40  24  200 
INFO:     5 26  27  166 47  30  223 
INFO:     10    31  33  175 51  35  236 
INFO:     15    35  36  181 54  39  248 
INFO:     20    37  38  187 57  41  260 
INFO:     25    39  40  193 59  43  270 
INFO:     30    40  42  198 60  44  281 
INFO:     35    42  43  203 62  46  289 
INFO:     40    44  45  208 63  47  298 
INFO:     45    45  46  213 65  49  306 
INFO:     50    46  47  219 66  50  313 
INFO:     55    48  49  226 68  52  321 
INFO:     60    49  50  233 69  53  329 
INFO:     65    51  52  241 71  55  339 
INFO:     70    52  53  250 73  56  351 
INFO:     75    54  55  261 75  58  367 
INFO:     80    56  57  273 77  60  385 
INFO:     85    58  59  284 80  62  403 
INFO:     90    61  62  300 84  65  423 
INFO:     95    65  66  331 89  69  445 
INFO:     96    66  68  341 90  71  450 
INFO:     97    68  69  353 92  72  457 
INFO:     98    70  71  372 94  74  463 
INFO:     99    73  75  405 99  77  472 
INFO:     100   2987    2668    1113    2821    7744    2308    
INFO:     Throughput: 61.7696K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -p 5
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
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -p 10
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 98  98  253 111 97  311 
INFO:     5 203 202 298 212 201 361 
INFO:     10    233 232 324 240 232 393 
INFO:     15    250 249 344 257 251 416 
INFO:     20    263 263 363 270 264 436 
INFO:     25    274 274 379 281 275 457 
INFO:     30    284 284 394 291 285 477 
INFO:     35    293 293 408 300 294 499 
INFO:     40    302 302 425 308 303 522 
INFO:     45    310 310 446 316 311 543 
INFO:     50    318 318 469 325 319 559 
INFO:     55    326 326 500 332 327 578 
INFO:     60    334 334 530 341 335 597 
INFO:     65    343 343 561 350 344 615 
INFO:     70    352 352 599 359 354 636 
INFO:     75    362 363 646 370 364 660 
INFO:     80    374 374 693 382 375 685 
INFO:     85    389 389 745 398 390 716 
INFO:     90    410 411 810 422 411 755 
INFO:     95    474 477 933 511 480 831 
INFO:     96    528 534 976 564 537 863 
INFO:     97    592 599 1021    628 598 931 
INFO:     98    664 672 1095    700 673 1042    
INFO:     99    776 793 1236    814 785 1174    
INFO:     100   2998    2984    2943    2558    2714    2760    
INFO:     Throughput: 69.7533K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -p 20
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 162 163 287 169 161 377 
INFO:     5 388 389 438 398 389 461 
INFO:     10    442 444 495 450 445 512 
INFO:     15    474 475 533 481 476 555 
INFO:     20    499 499 564 506 501 589 
INFO:     25    521 520 587 528 521 617 
INFO:     30    540 540 609 547 540 640 
INFO:     35    557 557 629 564 558 663 
INFO:     40    573 574 648 581 575 683 
INFO:     45    589 590 665 597 591 703 
INFO:     50    605 606 683 612 606 722 
INFO:     55    621 621 701 628 622 743 
INFO:     60    637 636 719 643 637 760 
INFO:     65    652 652 737 659 653 781 
INFO:     70    668 669 756 675 669 805 
INFO:     75    686 686 777 693 687 831 
INFO:     80    705 706 799 712 706 857 
INFO:     85    726 727 828 734 728 889 
INFO:     90    753 755 865 761 755 938 
INFO:     95    794 795 921 804 795 1097    
INFO:     96    807 807 942 817 807 1172    
INFO:     97    822 823 981 834 823 1246    
INFO:     98    845 846 1114    858 845 1315    
INFO:     99    892 894 1293    921 893 1401    
INFO:     100   11064   8970    2365    8162    10401   2198    
INFO:     Throughput: 70.8965K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -p 30
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 269 268 552 288 274 538 
INFO:     5 592 594 714 604 593 704 
INFO:     10    671 671 784 681 671 779 
INFO:     15    717 716 831 727 719 834 
INFO:     20    752 751 871 764 754 879 
INFO:     25    783 782 908 795 784 920 
INFO:     30    811 810 939 824 811 957 
INFO:     35    837 837 970 851 837 989 
INFO:     40    861 861 1000    876 861 1018    
INFO:     45    884 886 1033    898 886 1047    
INFO:     50    908 909 1065    922 910 1076    
INFO:     55    932 933 1095    946 934 1105    
INFO:     60    956 957 1130    971 958 1134    
INFO:     65    981 981 1167    995 983 1166    
INFO:     70    1008    1009    1216    1022    1010    1203    
INFO:     75    1037    1038    1286    1051    1039    1256    
INFO:     80    1070    1071    1404    1086    1071    1326    
INFO:     85    1110    1112    1630    1127    1112    1446    
INFO:     90    1168    1172    2029    1190    1170    1929    
INFO:     95    1325    1332    2418    1393    1338    2380    
INFO:     96    1468    1477    2521    1626    1490    2536    
INFO:     97    1812    1823    2713    1947    1833    2730    
INFO:     98    2120    2138    3089    2220    2137    3107    
INFO:     99    2608    2620    3760    2729    2624    3747    
INFO:     100   8084    7425    8049    9353    8808    8918    
INFO:     Throughput: 67.0482K queries/sec
wukong> 
wukong> 
wukong> q 
```

<br>
<a name="osdi16"></a>
## Experimantal results (OSDI16 Plan)

#### Summary

> query folder: `sparql_query/lubm/basic` and 
> plan folder: `sparql_query/lubm/basic/osdi16_plan`

| Workload | Latency (us) | #R (lines) | TH | Query   | Plan (OSDI16) |
| :------: | -----------: | ---------: | -: | :------ | :------------ |
| Q1       | 387,611      | 2528       | 16 | lubm_q1 | lubm_q1.fmt   |
| Q2       |  64,163      | 2,765,067  | 16 | lubm_q2 | lubm_q2.fmt   |
| Q3       | 188,721      | 0          | 16 | lubm_q3 | lubm_q3.fmt   |
| Q4       |      24      | 10         |  1 | lubm_q4 | lubm_q4.fmt   |
| Q5       |      18      | 10         |  1 | lubm_q5 | lubm_q5.fmt   |
| Q6       |      76      | 125        |  1 | lubm_q6 | lubm_q6.fmt   |
| Q7       | 309,334      | 112,559    | 16 | lubm_q7 | lubm_q7.fmt   |


> query folder: `sparql_query/lubm/emulator/` and
> plan folder: `sparql_query/lubm/emulator/`

| Workload | Thpt (q/s) | Configuration    | Query      | Plan (OSDI16) |
| :------: | ---------: | :--------------- | :--------- | :------------ |
| A1-A6    | 48.5415K   | -d 5 -w 1 -p 1   | mix_config | plan_config   |
| A1-A6    | 48.2640K   | -d 5 -w 1 -p 5   | mix_config | plan_config   |
| A1-A6    | 48.2382K   | -d 5 -w 1 -p 10  | mix_config | plan_config   |
| A1-A6    | 48.5595K   | -d 5 -w 1 -p 20  | mix_config | plan_config   |

> reference: execution without optimization time (using fixed query plan)

| Workload | Thpt (q/s) | Configuration    | File                           |
| :------: | ---------: | :--------------- | :----------------------------- |
| A1-A6    | 64.3845K   | -d 5 -w 1 -p 1   | query/lubm/emulator/mix_config |
| A1-A6    | 68.5610K   | -d 5 -w 1 -p 5   | query/lubm/emulator/mix_config |
| A1-A6    | 70.0947K   | -d 5 -w 1 -p 10  | query/lubm/emulator/mix_config |
| A1-A6    | 73.1285K   | -d 5 -w 1 -p 20  | query/lubm/emulator/mix_config |
| A1-A6    | 70.3801K   | -d 5 -w 1 -p 30  | query/lubm/emulator/mix_config |


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
wukong> 
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -F sparql_query/lubm/emulator/plan_config -d 5 -w 1 -p 1
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 21  22  137 38  25  172 
INFO:     5 30  31  149 49  33  199 
INFO:     10    36  38  157 54  40  211 
INFO:     15    40  41  163 56  43  221 
INFO:     20    42  43  167 59  45  231 
INFO:     25    44  45  172 61  48  242 
INFO:     30    46  47  177 63  49  250 
INFO:     35    48  49  182 65  51  257 
INFO:     40    49  50  186 66  53  263 
INFO:     45    51  52  191 67  54  270 
INFO:     50    52  53  195 69  56  276 
INFO:     55    54  55  200 70  57  283 
INFO:     60    55  56  206 72  59  290 
INFO:     65    57  58  213 74  60  299 
INFO:     70    58  59  220 76  62  310 
INFO:     75    60  61  227 78  64  322 
INFO:     80    63  64  234 81  66  339 
INFO:     85    65  66  242 84  69  355 
INFO:     90    69  70  251 88  73  371 
INFO:     95    73  74  264 93  77  393 
INFO:     96    74  76  267 94  78  398 
INFO:     97    76  77  271 96  80  402 
INFO:     98    78  79  276 98  82  408 
INFO:     99    81  83  284 101 86  417 
INFO:     100   5440    3645    423 313 1376    1782    
INFO:     Throughput: 48.5415K queries/sec
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -F sparql_query/lubm/emulator/plan_config -d 5 -w 1 -p 5
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 75  77  206 89  78  255 
INFO:     5 123 124 235 124 124 291 
INFO:     10    139 139 252 138 139 315 
INFO:     15    150 151 267 150 150 333 
INFO:     20    161 161 280 160 161 349 
INFO:     25    171 171 293 169 171 361 
INFO:     30    180 180 307 177 180 371 
INFO:     35    189 189 320 187 189 382 
INFO:     40    198 197 332 196 198 394 
INFO:     45    207 206 342 204 207 405 
INFO:     50    215 215 353 212 216 417 
INFO:     55    224 224 364 221 224 431 
INFO:     60    233 232 375 230 233 445 
INFO:     65    242 241 387 240 242 460 
INFO:     70    252 251 401 249 252 478 
INFO:     75    263 263 420 260 264 496 
INFO:     80    279 278 443 275 280 517 
INFO:     85    300 299 470 296 301 539 
INFO:     90    325 324 500 320 325 572 
INFO:     95    353 353 540 350 354 615 
INFO:     96    361 361 551 359 362 628 
INFO:     97    370 371 566 369 372 647 
INFO:     98    385 387 581 386 388 675 
INFO:     99    418 420 622 418 426 708 
INFO:     100   2955    2110    2970    1671    2863    2034    
INFO:     Throughput: 48.264K queries/sec
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -F sparql_query/lubm/emulator/plan_config -d 5 -w 1 -p 10
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 177 176 222 184 176 268 
INFO:     5 233 233 263 232 233 328 
INFO:     10    262 262 300 259 262 370 
INFO:     15    286 285 330 282 285 398 
INFO:     20    307 307 354 302 306 424 
INFO:     25    326 327 375 321 326 447 
INFO:     30    345 346 396 341 345 468 
INFO:     35    364 364 416 360 364 487 
INFO:     40    382 383 436 378 382 509 
INFO:     45    400 402 456 396 401 529 
INFO:     50    418 420 475 414 420 550 
INFO:     55    437 438 493 432 438 573 
INFO:     60    455 457 511 450 456 603 
INFO:     65    473 474 531 468 474 639 
INFO:     70    492 493 554 487 493 675 
INFO:     75    514 515 587 508 514 700 
INFO:     80    541 543 629 536 541 728 
INFO:     85    584 585 673 576 584 762 
INFO:     90    637 638 710 629 636 828 
INFO:     95    691 691 768 684 690 974 
INFO:     96    704 703 788 698 703 992 
INFO:     97    719 717 819 713 717 1014    
INFO:     98    737 734 893 731 735 1039    
INFO:     99    763 762 966 759 760 1085    
INFO:     100   2538    2028    1936    1946    2764    2793    
INFO:     Throughput: 48.2382K queries/sec
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -F sparql_query/lubm/emulator/plan_config -d 5 -w 1 -p 20
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 205 202 326 214 212 368 
INFO:     5 445 445 459 450 445 465 
INFO:     10    503 504 524 505 505 538 
INFO:     15    550 550 578 549 552 596 
INFO:     20    593 593 625 589 593 640 
INFO:     25    633 632 666 629 633 676 
INFO:     30    669 669 705 667 671 716 
INFO:     35    705 705 749 704 707 756 
INFO:     40    741 743 784 742 743 797 
INFO:     45    778 779 822 780 780 839 
INFO:     50    816 816 859 816 818 878 
INFO:     55    853 853 895 854 855 912 
INFO:     60    890 889 935 890 892 949 
INFO:     65    928 926 975 927 929 990 
INFO:     70    966 964 1011    965 967 1028    
INFO:     75    1009    1005    1054    1008    1010    1069    
INFO:     80    1063    1059    1110    1059    1063    1129    
INFO:     85    1145    1140    1191    1142    1146    1229    
INFO:     90    1256    1249    1299    1253    1254    1340    
INFO:     95    1368    1366    1413    1360    1367    1448    
INFO:     96    1394    1391    1440    1385    1391    1477    
INFO:     97    1421    1418    1465    1419    1419    1500    
INFO:     98    1453    1450    1497    1453    1451    1533    
INFO:     99    1496    1494    1546    1494    1497    1573    
INFO:     100   6201    6851    4507    3727    5776    4497    
INFO:     Throughput: 48.5595K queries/sec
wukong>
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 1
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 20  21  134 37  23  177 
INFO:     5 26  27  147 44  30  196 
INFO:     10    33  34  154 50  36  207 
INFO:     15    37  38  159 53  40  216 
INFO:     20    39  40  164 55  42  227 
INFO:     25    41  42  169 57  44  236 
INFO:     30    42  43  173 59  46  245 
INFO:     35    44  45  178 61  47  253 
INFO:     40    46  47  183 62  49  259 
INFO:     45    47  48  187 64  50  266 
INFO:     50    49  49  192 65  52  272 
INFO:     55    50  51  197 67  53  279 
INFO:     60    51  52  202 68  55  286 
INFO:     65    53  54  209 70  56  294 
INFO:     70    54  55  215 72  58  304 
INFO:     75    56  57  222 74  59  316 
INFO:     80    58  59  231 76  62  332 
INFO:     85    60  61  239 79  64  348 
INFO:     90    63  64  248 83  67  365 
INFO:     95    68  69  261 88  72  384 
INFO:     96    69  70  264 89  73  389 
INFO:     97    71  72  267 91  75  394 
INFO:     98    73  74  272 93  77  402 
INFO:     99    76  77  279 96  80  413 
INFO:     100   2613    1283    523 199 1734    1769    
INFO:     Throughput: 64.3845K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 5
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 90  91  205 103 93  260 
INFO:     5 119 120 230 130 121 288 
INFO:     10    132 133 246 141 134 307 
INFO:     15    141 142 257 149 143 321 
INFO:     20    148 148 268 155 149 334 
INFO:     25    153 153 277 160 155 345 
INFO:     30    158 159 285 166 160 355 
INFO:     35    163 164 292 170 165 366 
INFO:     40    168 168 301 175 170 375 
INFO:     45    173 173 308 181 174 385 
INFO:     50    178 178 317 186 180 395 
INFO:     55    183 184 326 191 185 407 
INFO:     60    189 189 336 198 191 420 
INFO:     65    195 196 346 205 197 433 
INFO:     70    202 202 357 212 204 447 
INFO:     75    209 210 371 220 211 461 
INFO:     80    218 218 384 229 219 478 
INFO:     85    228 228 400 243 230 497 
INFO:     90    244 244 421 266 247 522 
INFO:     95    292 293 460 325 298 563 
INFO:     96    313 314 476 345 318 576 
INFO:     97    340 340 501 369 346 590 
INFO:     98    381 380 540 411 385 622 
INFO:     99    446 444 597 472 452 689 
INFO:     100   1907    1939    1695    1825    2202    1931    
INFO:     Throughput: 68.561K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 10
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 79  78  199 94  80  263 
INFO:     5 196 197 265 203 196 324 
INFO:     10    227 228 293 234 228 356 
INFO:     15    245 246 312 251 246 376 
INFO:     20    259 260 328 265 260 394 
INFO:     25    271 271 341 276 272 409 
INFO:     30    281 282 353 286 282 426 
INFO:     35    290 291 364 296 291 443 
INFO:     40    299 300 376 306 300 462 
INFO:     45    308 309 387 315 309 481 
INFO:     50    317 318 398 323 318 499 
INFO:     55    326 326 412 333 327 518 
INFO:     60    335 335 426 342 337 535 
INFO:     65    345 345 441 352 346 553 
INFO:     70    356 356 459 363 357 573 
INFO:     75    367 368 479 375 369 596 
INFO:     80    380 381 500 388 382 628 
INFO:     85    396 396 526 404 397 665 
INFO:     90    416 415 574 424 416 703 
INFO:     95    446 447 661 458 447 758 
INFO:     96    458 458 682 472 458 772 
INFO:     97    477 477 710 504 478 785 
INFO:     98    524 524 752 569 524 809 
INFO:     99    623 626 816 666 619 866 
INFO:     100   35095   35116   12258   34441   34618   24338   
INFO:     Throughput: 70.0947K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 20
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 130 130 211 141 132 279 
INFO:     5 194 194 281 199 195 394 
INFO:     10    258 257 405 253 260 455 
INFO:     15    415 417 472 416 417 502 
INFO:     20    458 459 514 462 460 541 
INFO:     25    488 489 545 492 490 577 
INFO:     30    513 514 572 517 514 605 
INFO:     35    535 535 594 539 537 628 
INFO:     40    555 555 615 559 556 649 
INFO:     45    574 574 636 578 575 671 
INFO:     50    592 592 655 596 593 690 
INFO:     55    610 609 673 613 610 712 
INFO:     60    627 627 692 631 628 733 
INFO:     65    645 645 714 649 646 756 
INFO:     70    664 664 735 668 664 781 
INFO:     75    684 684 758 688 684 808 
INFO:     80    706 707 785 711 707 843 
INFO:     85    733 732 814 738 732 878 
INFO:     90    764 764 851 771 765 921 
INFO:     95    811 811 898 820 812 981 
INFO:     96    825 824 914 832 825 1002    
INFO:     97    841 840 933 849 841 1035    
INFO:     98    863 862 962 874 863 1098    
INFO:     99    904 901 1031    919 903 1337    
INFO:     100   16703   16690   12764   12815   16700   12772   
INFO:     Throughput: 73.1285K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 30
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 238 237 300 241 234 374 
INFO:     5 539 546 598 534 539 613 
INFO:     10    642 643 702 647 643 716 
INFO:     15    692 692 757 696 693 776 
INFO:     20    730 729 795 734 731 819 
INFO:     25    762 761 827 767 762 855 
INFO:     30    790 789 857 795 791 887 
INFO:     35    817 816 882 823 817 917 
INFO:     40    842 841 907 849 842 943 
INFO:     45    866 865 933 874 866 971 
INFO:     50    891 889 958 899 891 998 
INFO:     55    915 913 986 923 915 1026    
INFO:     60    940 938 1013    947 941 1054    
INFO:     65    966 964 1040    974 966 1082    
INFO:     70    993 992 1070    1002    994 1112    
INFO:     75    1022    1021    1101    1032    1024    1148    
INFO:     80    1054    1053    1134    1064    1056    1184    
INFO:     85    1091    1091    1174    1101    1093    1227    
INFO:     90    1139    1136    1224    1149    1140    1279    
INFO:     95    1205    1205    1290    1217    1208    1350    
INFO:     96    1224    1224    1308    1236    1226    1367    
INFO:     97    1246    1246    1333    1258    1247    1389    
INFO:     98    1274    1274    1367    1290    1276    1421    
INFO:     99    1326    1325    1417    1341    1323    1487    
INFO:     100   14838   13499   12833   11718   14853   8980    
INFO:     Throughput: 70.3801K queries/sec
wukong> 
wukong> q 
```
