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

##### Gitlab Version: @840998e

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

112 220218  2528
1   67675   2765067
110 198343  0
1   23  10
0   18  10
1   73  125
21  160842  112559
0   96  8569
0   20  730
0   19  5
0   17  1
1   117 3101

> Query folder: `sparql_query/lubm/basic`  

| Workload | OPT (us) | Latency (us) | #R (lines) | TH | Query    |
| :------: | -------: |------------: | ---------: | -: | :------- |
| Q1       | 112      | 211,036      | 2528       | 16 | lubm_q1  |
| Q2       |   1      |  64,207      | 2,765,067  | 16 | lubm_q2  |
| Q3       | 110      | 188,177      | 0          | 16 | lubm_q3  |
| Q4       |   1      |      23      | 10         |  1 | lubm_q4  |
| Q5       |   0      |      18      | 10         |  1 | lubm_q5  |
| Q6       |   1      |      72      | 125        |  1 | lubm_q6  |
| Q7       |  21      | 152,771      | 112,559    | 16 | lubm_q7  |
| Q8       |   0      |      98      | 8,569      |  1 | lubm_q8  |
| Q9       |   0      |      20      | 730        |  1 | lubm_q9  |
| Q10      |   0      |      19      | 5          |  1 | lubm_q10 |
| Q11      |   0      |      17      | 1          |  1 | lubm_q11 |
| Q12      |   1      |     119      | 3,101      |  1 | lubm_q12 |

> Query folder: `sparql_query/lubm/emulator` 

| Workload | Thpt (q/s) | Configuration    | Config     |
| :------: | ---------: | :--------------- | :--------- |
| A1-A6    | 63.4399K   | -d 5 -w 1 -p 1   | mix_config |
| A1-A6    | 70.1163K   | -d 5 -w 1 -p 5   | mix_config |
| A1-A6    | 71.2101K   | -d 5 -w 1 -p 10  | mix_config |
| A1-A6    | 69.7149K   | -d 5 -w 1 -p 20  | mix_config |
| A1-A6    | 69.8308K   | -d 5 -w 1 -p 30  | mix_config |


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
global_rdma_buf_size_mb: 128
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
INFO:     Parsing time: 50 usec
INFO:     Optimization time: 112 usec
INFO:     (last) result size: 2528
INFO:     (average) latency: 211036 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q2 -m 16 -n 10 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 32 usec
INFO:     Optimization time: 1 usec
INFO:     (last) result size: 2765067
INFO:     (average) latency: 64207 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q3 -m 16 -n 10 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 52 usec
INFO:     Optimization time: 110 usec
INFO:     (last) result size: 0
INFO:     (average) latency: 188177 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q4 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 45 usec
INFO:     Optimization time: 1 usec
INFO:     (last) result size: 10
INFO:     (average) latency: 23 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q5 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 29 usec
INFO:     Optimization time: 0 usec
INFO:     (last) result size: 10
INFO:     (average) latency: 18 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q6 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 44 usec
INFO:     Optimization time: 1 usec
INFO:     (last) result size: 125
INFO:     (average) latency: 72 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q7 -m 16 -n 10 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 49 usec
INFO:     Optimization time: 21 usec
INFO:     (last) result size: 112559
INFO:     (average) latency: 152771 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q8 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 31 usec
INFO:     Optimization time: 0 usec
INFO:     (last) result size: 8569
INFO:     (average) latency: 98 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q9 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 23 usec
INFO:     Optimization time: 0 usec
INFO:     (last) result size: 730
INFO:     (average) latency: 20 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q10 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 28 usec
INFO:     Optimization time: 0 usec
INFO:     (last) result size: 5
INFO:     (average) latency: 19 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q11 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 25 usec
INFO:     Optimization time: 0 usec
INFO:     (last) result size: 1
INFO:     (average) latency: 17 usec
wukong> sparql -f sparql_query/lubm/basic/lubm_q12 -n 1000 -N 100
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 32 usec
INFO:     Optimization time: 1 usec
INFO:     (last) result size: 3101
INFO:     (average) latency: 119 usec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 1
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 20  21  131 36  23  172 
INFO:     5 25  26  144 42  28  194 
INFO:     10    31  32  152 48  34  206 
INFO:     15    35  36  157 52  39  216 
INFO:     20    38  39  162 55  41  226 
INFO:     25    40  41  166 57  43  235 
INFO:     30    42  43  171 59  45  243 
INFO:     35    43  44  175 60  47  251 
INFO:     40    45  46  180 62  48  258 
INFO:     45    47  48  185 63  50  264 
INFO:     50    48  49  189 65  51  271 
INFO:     55    49  50  195 66  53  277 
INFO:     60    51  52  200 68  54  285 
INFO:     65    52  53  206 69  55  293 
INFO:     70    54  55  213 71  57  302 
INFO:     75    55  56  220 73  59  315 
INFO:     80    57  58  227 75  61  330 
INFO:     85    60  60  236 78  63  347 
INFO:     90    62  63  245 81  66  364 
INFO:     95    67  67  257 86  70  383 
INFO:     96    68  69  260 87  72  387 
INFO:     97    69  70  264 89  73  393 
INFO:     98    71  72  268 91  75  401 
INFO:     99    74  75  275 94  78  410 
INFO:     100   686 1321    1233    120 232 2207    
INFO:     Throughput: 63.4399K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 5
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 39  40  160 53  41  220 
INFO:     5 103 104 214 115 106 272 
INFO:     10    124 125 234 133 126 293 
INFO:     15    135 136 247 142 137 308 
INFO:     20    142 143 258 149 144 321 
INFO:     25    149 149 268 155 150 333 
INFO:     30    154 155 277 160 156 342 
INFO:     35    159 160 285 165 161 352 
INFO:     40    164 165 293 170 165 362 
INFO:     45    169 169 301 174 170 372 
INFO:     50    173 174 309 180 175 383 
INFO:     55    178 179 318 185 180 394 
INFO:     60    184 184 327 190 185 406 
INFO:     65    189 190 337 197 191 420 
INFO:     70    195 196 349 203 197 434 
INFO:     75    202 203 361 210 204 449 
INFO:     80    210 211 375 219 212 466 
INFO:     85    220 221 391 231 222 485 
INFO:     90    234 235 412 251 237 508 
INFO:     95    279 283 451 307 288 547 
INFO:     96    300 304 463 325 309 562 
INFO:     97    330 331 486 351 338 586 
INFO:     98    370 373 527 395 375 613 
INFO:     99    431 436 607 450 436 682 
INFO:     100   4673    3486    3270    3541    4723    1936    
INFO:     Throughput: 70.1163K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 10
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 57  55  174 64  58  233 
INFO:     5 120 120 241 124 123 299 
INFO:     10    219 218 273 223 220 334 
INFO:     15    242 241 297 246 243 359 
INFO:     20    257 257 314 261 258 378 
INFO:     25    270 269 330 274 271 395 
INFO:     30    281 281 343 285 282 411 
INFO:     35    291 291 355 295 292 427 
INFO:     40    300 300 366 304 301 444 
INFO:     45    308 308 376 313 310 466 
INFO:     50    317 317 388 322 318 487 
INFO:     55    326 326 400 330 327 508 
INFO:     60    334 335 413 339 336 525 
INFO:     65    344 344 428 349 345 546 
INFO:     70    354 354 446 359 356 563 
INFO:     75    365 365 465 370 367 586 
INFO:     80    378 378 488 383 379 620 
INFO:     85    393 393 513 398 394 658 
INFO:     90    411 411 558 416 413 698 
INFO:     95    438 439 653 445 440 749 
INFO:     96    448 448 671 456 449 766 
INFO:     97    462 464 694 475 464 788 
INFO:     98    500 505 725 531 505 811 
INFO:     99    601 599 788 632 601 863 
INFO:     100   1883    1980    1932    1684    2017    1786    
INFO:     Throughput: 71.2101K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 20
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 190 192 299 207 198 376 
INFO:     5 421 421 465 426 423 475 
INFO:     10    463 463 508 469 465 525 
INFO:     15    491 491 540 496 492 558 
INFO:     20    513 512 565 517 513 587 
INFO:     25    532 532 585 535 532 611 
INFO:     30    549 549 603 552 550 634 
INFO:     35    565 565 622 568 566 654 
INFO:     40    580 581 638 584 582 674 
INFO:     45    595 596 656 600 597 692 
INFO:     50    610 611 672 615 611 708 
INFO:     55    625 626 688 630 626 726 
INFO:     60    640 641 703 646 642 743 
INFO:     65    656 657 721 661 658 761 
INFO:     70    673 675 741 678 676 782 
INFO:     75    692 693 762 697 694 804 
INFO:     80    712 714 785 718 715 831 
INFO:     85    737 738 811 744 739 861 
INFO:     90    767 768 846 773 769 899 
INFO:     95    808 809 891 814 809 953 
INFO:     96    819 821 902 826 821 972 
INFO:     97    833 834 919 838 834 1000    
INFO:     98    850 852 944 856 852 1085    
INFO:     99    881 882 1011    886 883 1276    
INFO:     100   2567    2463    2357    2608    2877    2400    
INFO:     Throughput: 69.7149K queries/sec
wukong> 
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config -d 5 -w 1 -n 30
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 261 260 325 263 262 374 
INFO:     5 605 603 650 605 600 663 
INFO:     10    678 676 728 681 676 743 
INFO:     15    719 717 773 724 718 795 
INFO:     20    752 750 807 757 752 834 
INFO:     25    781 778 836 785 779 869 
INFO:     30    807 804 866 811 806 899 
INFO:     35    831 828 893 835 829 925 
INFO:     40    854 851 916 857 852 948 
INFO:     45    876 873 940 880 874 974 
INFO:     50    898 896 963 902 897 999 
INFO:     55    920 918 984 924 920 1022    
INFO:     60    943 942 1008    947 943 1046    
INFO:     65    967 965 1033    972 966 1072    
INFO:     70    993 991 1060    998 992 1102    
INFO:     75    1020    1018    1088    1025    1020    1135    
INFO:     80    1050    1049    1123    1055    1050    1170    
INFO:     85    1085    1084    1161    1090    1085    1206    
INFO:     90    1128    1128    1204    1132    1129    1252    
INFO:     95    1187    1186    1264    1195    1189    1321    
INFO:     96    1203    1203    1280    1210    1205    1340    
INFO:     97    1222    1221    1300    1231    1224    1363    
INFO:     98    1247    1246    1327    1255    1249    1389    
INFO:     99    1287    1284    1361    1291    1288    1420    
INFO:     100   2569    2854    2559    2363    2551    2790    
INFO:     Throughput: 69.8308K queries/sec
wukong> 
wukong> 
wukong> q 
```
