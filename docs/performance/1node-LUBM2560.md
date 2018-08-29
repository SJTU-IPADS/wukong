# Performance 

###### Date: Aug. 29, 2018 

###### Author: Rong Chen


## Table of Contents

* [Hardware configuration](#hw)
* [Software configuration](#sw)
* [Dataset and workload](#dw)
* [Experimantal results (RDMA-enabled)](#res)
* [Experimantal results (w/ zeromq)](#res2)

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

##### Gitlab Version: @4bf49086

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
global_enable_planner       0
global_generate_statistics  1
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
$./build.sh -DUSE_RDMA=ON -DUSE_GPU=OFF -DUSE_HADOOP=OFF -DUSE_JEMALLOC=OFF -DUSE_DYNAMIC_GSTORE=OFF -DUSE_VERSATILE=OFF -DUSE_DTYPE_64BIT=OFF
$./run.sh 1
```

<br>
<a name="dw"></a>
## Dataset and workload

**Dataset**: Leigh University Benchmark with 2,560 University (**LUBM-2560**)

**Queries**: `query/lubm/lubm_{q1,q2,q3,q4,q5,q6,q7}`, `query/lubm/emulator/mix_config`


<br>
<a name="res"></a>
## Experimantal results (RDMA-enabled)

#### Summary

| Workload | Latency (us) | #Results (lines)  | TH | File               |
| :------: | -----------: | ----------------: | -: | :----------------- |
| Q1       | 457,185      | 2528              | 16 | query/lubm/lubm_q1 |
| Q2       |  75,772      | 2,765,067         | 16 | query/lubm/lubm_q2 |
| Q3       | 207,471      | 0                 | 16 | query/lubm/lubm_q3 |
| Q4       |      28      | 10                |  1 | query/lubm/lubm_q4 |
| Q5       |      20      | 10                |  1 | query/lubm/lubm_q5 |
| Q6       |      87      | 125               |  1 | query/lubm/lubm_q6 |
| Q7       | 352,150      | 112,559           | 16 | query/lubm/lubm_q7 |

| Workload | Thpt (q/s) | Configuration   | File                           |
| :------: | ---------: | :-------------- | :----------------------------- |
| A1-A6    | 51.6123K   | -d 5 -w 1 -p 1  | query/lubm/emulator/mix_config |
| A1-A6    | 56.0161K   | -d 5 -w 1 -p 5  | query/lubm/emulator/mix_config |
| A1-A6    | 56.6080K   | -d 5 -w 1 -p 10  | query/lubm/emulator/mix_config |
| A1-A6    | 56.3037K   | -d 5 -w 1 -p 20  | query/lubm/emulator/mix_config |
| A1-A6    | 56.8284K   | -d 5 -w 1 -p 30  | query/lubm/emulator/mix_config |

#### Detail

```bash
wukong> sparql -f query/lubm/lubm_q1 -m 16 -n 10
INFO:     Parsing time: 80 usec
INFO:     (average) latency: 457185 usec
INFO:     (last) result size: 2528
wukong> sparql -f query/lubm/lubm_q2 -m 16 -n 10
INFO:     Parsing time: 46 usec
INFO:     (average) latency: 75772 usec
INFO:     (last) result size: 2765067
wukong> sparql -f query/lubm/lubm_q3 -m 16 -n 10
INFO:     Parsing time: 68 usec
INFO:     (average) latency: 207471 usec
INFO:     (last) result size: 0
wukong> sparql -f query/lubm/lubm_q4 -n 1000
INFO:     Parsing time: 57 usec
INFO:     (average) latency: 28 usec
INFO:     (last) result size: 10
wukong> sparql -f query/lubm/lubm_q5 -n 1000
INFO:     Parsing time: 59 usec
INFO:     (average) latency: 20 usec
INFO:     (last) result size: 10
wukong> sparql -f query/lubm/lubm_q6 -n 1000
INFO:     Parsing time: 46 usec
INFO:     (average) latency: 86 usec
INFO:     (last) result size: 125
wukong> sparql -f query/lubm/lubm_q7 -m 16 -n 10
INFO:     Parsing time: 66 usec
INFO:     (average) latency: 352150 usec
INFO:     (last) result size: 112559
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 1
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 25  26  164 49  29  209 
INFO:     5 35  36  184 61  39  239 
INFO:     10    43  44  195 67  47  256 
INFO:     15    47  48  202 71  51  269 
INFO:     20    49  50  208 74  54  279 
INFO:     25    51  53  213 77  56  288 
INFO:     30    53  55  218 79  58  296 
INFO:     35    55  56  222 81  60  304 
INFO:     40    57  58  227 83  61  312 
INFO:     45    58  60  231 84  63  319 
INFO:     50    60  61  236 86  65  327 
INFO:     55    62  63  241 88  66  336 
INFO:     60    63  65  246 89  68  344 
INFO:     65    65  66  252 91  70  353 
INFO:     70    67  68  258 93  71  364 
INFO:     75    69  70  266 95  73  375 
INFO:     80    71  73  275 97  76  389 
INFO:     85    74  75  286 100 78  403 
INFO:     90    78  79  303 103 82  420 
INFO:     95    83  85  337 109 88  451 
INFO:     96    85  86  347 111 89  463 
INFO:     97    87  88  358 113 91  482 
INFO:     98    89  91  372 116 94  508 
INFO:     99    93  95  408 126 98  554 
INFO:     100   192 258 658 443 303 997 
INFO:     Throughput: 51.6123K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 5
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 72  75  248 101 77  309 
INFO:     5 114 116 282 142 120 350 
INFO:     10    136 138 302 162 141 377 
INFO:     15    150 152 317 174 154 396 
INFO:     20    160 162 329 183 164 411 
INFO:     25    169 170 342 192 173 424 
INFO:     30    177 178 355 200 180 440 
INFO:     35    184 185 366 207 187 453 
INFO:     40    190 192 379 215 194 465 
INFO:     45    197 199 394 223 201 478 
INFO:     50    204 205 412 232 207 491 
INFO:     55    211 213 436 241 214 507 
INFO:     60    218 220 466 252 222 524 
INFO:     65    227 229 509 264 231 546 
INFO:     70    237 239 559 279 242 566 
INFO:     75    249 251 617 297 254 596 
INFO:     80    263 266 690 320 271 636 
INFO:     85    285 291 793 351 295 710 
INFO:     90    329 337 955 399 339 913 
INFO:     95    441 447 1272    509 444 1413    
INFO:     96    489 494 1371    557 490 1621    
INFO:     97    559 561 1491    631 560 1910    
INFO:     98    697 687 1718    753 687 2216    
INFO:     99    1059    1024    2099    1126    1044    2706    
INFO:     100   5740    5260    5014    4652    5562    6148    
INFO:     Throughput: 56.0161K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 10
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 182 183 292 204 185 355 
INFO:     5 253 253 342 268 255 410 
INFO:     10    284 285 374 298 286 445 
INFO:     15    305 306 394 319 306 470 
INFO:     20    320 322 413 335 322 491 
INFO:     25    334 335 428 348 336 510 
INFO:     30    346 347 443 360 348 530 
INFO:     35    357 358 458 371 359 552 
INFO:     40    368 369 472 382 370 575 
INFO:     45    379 380 487 393 381 600 
INFO:     50    390 391 505 403 392 623 
INFO:     55    400 402 523 414 403 644 
INFO:     60    412 413 543 426 414 665 
INFO:     65    423 425 565 438 426 687 
INFO:     70    436 438 592 452 439 711 
INFO:     75    450 452 620 467 454 742 
INFO:     80    467 468 657 484 470 778 
INFO:     85    487 488 712 507 490 821 
INFO:     90    514 517 788 537 518 875 
INFO:     95    572 575 875 607 576 953 
INFO:     96    601 607 904 644 606 978 
INFO:     97    658 665 939 711 666 1015    
INFO:     98    751 759 998 802 756 1070    
INFO:     99    912 917 1122    965 916 1252    
INFO:     100   5119    4602    5037    3032    5121    4835    
INFO:     Throughput: 56.608K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 20
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 297 293 473 344 303 519 
INFO:     5 470 473 584 498 476 611 
INFO:     10    534 536 645 559 540 669 
INFO:     15    574 576 685 596 579 709 
INFO:     20    605 607 717 626 610 747 
INFO:     25    632 633 745 652 636 778 
INFO:     30    655 656 773 677 659 806 
INFO:     35    677 678 797 697 681 832 
INFO:     40    698 699 821 719 702 857 
INFO:     45    718 719 847 739 722 882 
INFO:     50    738 740 873 759 742 907 
INFO:     55    758 760 901 780 763 933 
INFO:     60    779 781 931 801 784 964 
INFO:     65    801 802 969 824 806 997 
INFO:     70    824 826 1015    849 829 1039    
INFO:     75    851 851 1085    875 856 1087    
INFO:     80    881 882 1179    908 888 1177    
INFO:     85    921 922 1370    952 928 1358    
INFO:     90    977 982 1689    1015    989 1747    
INFO:     95    1128    1144    2090    1271    1165    2468    
INFO:     96    1284    1328    2244    1548    1370    2665    
INFO:     97    1623    1648    2525    1793    1678    2918    
INFO:     98    1989    2022    2918    2230    2049    3180    
INFO:     99    2749    2757    3408    2881    2790    3861    
INFO:     100   9922    10052   10264   9356    10171   10208   
INFO:     Throughput: 56.3037K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 30
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 495 490 653 544 498 676 
INFO:     5 714 709 811 733 716 846 
INFO:     10    792 793 895 812 797 929 
INFO:     15    846 847 945 866 850 986 
INFO:     20    890 890 985 907 893 1033    
INFO:     25    928 928 1024    946 931 1075    
INFO:     30    962 961 1060    980 964 1110    
INFO:     35    995 993 1093    1013    996 1143    
INFO:     40    1025    1024    1128    1043    1026    1175    
INFO:     45    1056    1054    1161    1071    1056    1208    
INFO:     50    1086    1084    1193    1102    1087    1243    
INFO:     55    1116    1115    1225    1133    1118    1279    
INFO:     60    1149    1146    1262    1166    1150    1316    
INFO:     65    1182    1178    1298    1199    1183    1352    
INFO:     70    1217    1214    1343    1233    1218    1397    
INFO:     75    1255    1253    1386    1273    1256    1441    
INFO:     80    1300    1298    1437    1316    1301    1500    
INFO:     85    1354    1351    1505    1370    1355    1570    
INFO:     90    1426    1427    1595    1443    1431    1676    
INFO:     95    1564    1561    1825    1584    1566    2432    
INFO:     96    1625    1619    2186    1642    1621    2781    
INFO:     97    1766    1742    2709    1762    1741    3176    
INFO:     98    2628    2553    3205    2592    2565    3786    
INFO:     99    3328    3321    4101    3346    3288    5364    
INFO:     100   13695   12979   14062   13044   12940   12014   
INFO:     Throughput: 56.8284K queries/sec
wukong> q 
```


<br>
<a name="res2"></a>
## Experimantal results (w/ zeromq)

#### Summary

| Workload | Latency (us) | #Results (lines)  | TH | File               |
| :------: | -----------: | ----------------: | -: | :----------------- |
| Q1       | 466,544      | 2528              | 16 | query/lubm/lubm_q1 |
| Q2       |  78,764      | 2,765,067         | 16 | query/lubm/lubm_q2 |
| Q3       | 224,585      | 0                 | 16 | query/lubm/lubm_q3 |
| Q4       |      63      | 10                |  1 | query/lubm/lubm_q4 |
| Q5       |      56      | 10                |  1 | query/lubm/lubm_q5 |
| Q6       |     122      | 125               |  1 | query/lubm/lubm_q6 |
| Q7       | 353,381      | 112,559           | 16 | query/lubm/lubm_q7 |

| Workload | Thpt (q/s) | Configuration   | File                           |
| :------: | ---------: | :-------------- | :----------------------------- |
| A1-A6    | 23.2710K   | -d 5 -w 1 -p 1  | query/lubm/emulator/mix_config |
| A1-A6    | 40.6267K   | -d 5 -w 1 -p 5  | query/lubm/emulator/mix_config |
| A1-A6    | 44.2526K   | -d 5 -w 1 -p 10  | query/lubm/emulator/mix_config |
| A1-A6    | 48.0790K   | -d 5 -w 1 -p 20  | query/lubm/emulator/mix_config |
| A1-A6    | 47.4467K   | -d 5 -w 1 -p 30  | query/lubm/emulator/mix_config |

#### Detail

```bash
wukong> config -v
wukong> config -s global_use_rdma=0
wukong> 
wukong> sparql -f query/lubm/lubm_q1 -m 16 -n 10
WARNING:  RDMA is not enabled, skip corun optimization!
INFO:     Parsing time: 87 usec
INFO:     (average) latency: 466544 usec
INFO:     (last) result size: 2528
wukong> sparql -f query/lubm/lubm_q2 -m 16 -n 10
INFO:     Parsing time: 56 usec
INFO:     (average) latency: 78764 usec
INFO:     (last) result size: 2765067
wukong> sparql -f query/lubm/lubm_q3 -m 16 -n 10
INFO:     Parsing time: 65 usec
INFO:     (average) latency: 224585 usec
INFO:     (last) result size: 0
wukong> sparql -f query/lubm/lubm_q4 -n 1000
INFO:     Parsing time: 60 usec
INFO:     (average) latency: 63 usec
INFO:     (last) result size: 10
wukong> sparql -f query/lubm/lubm_q5 -n 1000
INFO:     Parsing time: 44 usec
INFO:     (average) latency: 56 usec
INFO:     (last) result size: 10
wukong> sparql -f query/lubm/lubm_q6 -n 1000
INFO:     Parsing time: 48 usec
INFO:     (average) latency: 122 usec
INFO:     (last) result size: 125
wukong> sparql -f query/lubm/lubm_q7 -m 16 -n 10
INFO:     Parsing time: 59 usec
INFO:     (average) latency: 353381 usec
INFO:     (last) result size: 112559
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 1
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 81  83  227 106 86  282 
INFO:     5 93  94  247 116 97  309 
INFO:     10    99  101 259 124 103 327 
INFO:     15    104 105 268 129 108 344 
INFO:     20    108 109 276 134 112 355 
INFO:     25    111 113 284 139 116 366 
INFO:     30    115 116 291 143 120 377 
INFO:     35    118 120 298 148 123 389 
INFO:     40    122 124 305 152 127 399 
INFO:     45    125 127 313 156 131 410 
INFO:     50    129 131 323 160 135 423 
INFO:     55    133 135 334 165 139 437 
INFO:     60    138 140 354 170 144 453 
INFO:     65    142 144 484 176 148 477 
INFO:     70    147 149 686 182 153 511 
INFO:     75    152 154 797 189 158 566 
INFO:     80    157 160 919 202 165 960 
INFO:     85    164 167 1058    251 172 1236    
INFO:     90    171 176 1228    279 182 1517    
INFO:     95    185 196 1471    346 225 1939    
INFO:     96    190 211 1535    361 237 2083    
INFO:     97    201 229 1624    376 248 2242    
INFO:     98    227 242 1734    426 260 2402    
INFO:     99    249 258 1912    466 283 2633    
INFO:     100   2200    2303    2858    1710    2236    3813    
INFO:     Throughput: 23.271K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 5
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 232 230 398 259 234 461 
INFO:     5 280 281 450 308 285 528 
INFO:     10    305 307 484 336 311 563 
INFO:     15    322 325 507 356 329 586 
INFO:     20    336 339 528 371 343 605 
INFO:     25    348 351 545 383 355 623 
INFO:     30    359 362 560 395 366 640 
INFO:     35    369 372 577 405 376 658 
INFO:     40    379 381 593 414 386 672 
INFO:     45    388 391 612 424 395 685 
INFO:     50    397 399 637 434 404 701 
INFO:     55    406 408 662 443 413 716 
INFO:     60    415 417 706 453 422 734 
INFO:     65    424 426 767 464 431 754 
INFO:     70    434 436 842 475 441 779 
INFO:     75    445 447 921 489 452 815 
INFO:     80    457 460 1006    506 466 868 
INFO:     85    473 476 1141    528 482 991 
INFO:     90    497 500 1321    564 507 1351    
INFO:     95    554 557 1618    665 567 1986    
INFO:     96    585 588 1751    705 601 2192    
INFO:     97    653 650 1879    761 662 2475    
INFO:     98    799 789 2092    859 792 2752    
INFO:     99    1179    1176    2473    1248    1176    3568    
INFO:     100   5589    6571    5799    5519    4742    7034    
INFO:     Throughput: 40.6267K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 10
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 328 334 511 363 335 601 
INFO:     5 429 431 628 467 435 716 
INFO:     10    481 483 692 519 487 774 
INFO:     15    517 519 731 556 523 815 
INFO:     20    544 547 766 585 552 844 
INFO:     25    568 571 794 610 576 874 
INFO:     30    590 593 820 632 598 901 
INFO:     35    610 613 846 653 618 928 
INFO:     40    630 633 874 672 638 952 
INFO:     45    649 651 901 692 657 975 
INFO:     50    668 670 930 711 677 1003    
INFO:     55    686 689 962 730 696 1031    
INFO:     60    706 708 1004    751 716 1064    
INFO:     65    727 729 1051    772 737 1097    
INFO:     70    749 752 1118    797 759 1143    
INFO:     75    775 777 1221    824 785 1204    
INFO:     80    805 807 1361    858 814 1325    
INFO:     85    844 847 1523    900 852 1537    
INFO:     90    901 905 1725    964 909 1945    
INFO:     95    1061    1064    2079    1145    1057    2706    
INFO:     96    1171    1164    2212    1276    1161    2828    
INFO:     97    1371    1352    2474    1522    1366    2986    
INFO:     98    1797    1737    2831    1954    1780    3138    
INFO:     99    2594    2575    3222    2689    2567    3579    
INFO:     100   6044    6340    6834    4758    6748    6770    
INFO:     Throughput: 44.2526K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 20
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 433 439 612 476 442 681 
INFO:     5 598 603 782 636 606 878 
INFO:     10    702 709 896 739 713 1002    
INFO:     15    787 796 999 832 799 1102    
INFO:     20    861 872 1085    906 872 1173    
INFO:     25    926 937 1152    973 937 1245    
INFO:     30    984 996 1222    1037    998 1310    
INFO:     35    1040    1052    1280    1095    1055    1375    
INFO:     40    1096    1106    1341    1149    1109    1427    
INFO:     45    1149    1159    1399    1206    1163    1482    
INFO:     50    1202    1212    1452    1258    1216    1531    
INFO:     55    1256    1266    1502    1311    1270    1576    
INFO:     60    1312    1321    1560    1366    1325    1623    
INFO:     65    1369    1377    1610    1420    1381    1666    
INFO:     70    1427    1433    1665    1477    1437    1718    
INFO:     75    1490    1493    1712    1533    1497    1771    
INFO:     80    1555    1558    1775    1592    1563    1830    
INFO:     85    1626    1630    1852    1667    1635    1909    
INFO:     90    1718    1722    1959    1755    1727    2020    
INFO:     95    1870    1881    2167    1908    1886    2263    
INFO:     96    1933    1949    2257    1981    1949    2376    
INFO:     97    2037    2053    2397    2105    2062    2573    
INFO:     98    2298    2356    2744    2420    2344    3099    
INFO:     99    3097    3157    3480    3180    3116    3636    
INFO:     100   8750    8516    8844    8497    8321    8887    
INFO:     Throughput: 48.079K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 30
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 574 576 780 618 573 827 
INFO:     5 806 810 980 834 808 1065    
INFO:     10    955 959 1163    980 956 1264    
INFO:     15    1084    1085    1320    1109    1084    1430    
INFO:     20    1200    1199    1468    1223    1197    1583    
INFO:     25    1304    1307    1602    1346    1303    1713    
INFO:     30    1401    1405    1724    1460    1402    1824    
INFO:     35    1496    1499    1831    1560    1500    1939    
INFO:     40    1592    1594    1923    1658    1595    2030    
INFO:     45    1682    1685    2011    1752    1690    2113    
INFO:     50    1769    1775    2094    1838    1778    2186    
INFO:     55    1854    1858    2171    1915    1865    2260    
INFO:     60    1939    1942    2245    1993    1950    2327    
INFO:     65    2020    2024    2317    2069    2032    2392    
INFO:     70    2098    2105    2399    2148    2112    2475    
INFO:     75    2180    2191    2490    2231    2194    2583    
INFO:     80    2271    2281    2617    2315    2283    2745    
INFO:     85    2384    2394    2843    2427    2394    3064    
INFO:     90    2592    2601    3331    2648    2602    3674    
INFO:     95    3405    3442    4099    3531    3443    4316    
INFO:     96    3629    3684    4292    3723    3665    4526    
INFO:     97    3856    3904    4481    3977    3892    4804    
INFO:     98    4166    4180    4840    4265    4173    5137    
INFO:     99    4716    4693    5495    4750    4705    6079    
INFO:     100   13273   11994   10249   11707   13365   13078   
INFO:     Throughput: 47.4467K queries/sec
wukong> q 
```
