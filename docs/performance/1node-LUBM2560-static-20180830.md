# Performance (single node, static gstore)

###### Date: Aug. 30, 2018 

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

##### Gitlab Version: @d1cfe572

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
| Q1       | 415,420      | 2528              | 16 | query/lubm/lubm_q1 |
| Q2       |  65,635      | 2,765,067         | 16 | query/lubm/lubm_q2 |
| Q3       | 192,233      | 0                 | 16 | query/lubm/lubm_q3 |
| Q4       |      24      | 10                |  1 | query/lubm/lubm_q4 |
| Q5       |      17      | 10                |  1 | query/lubm/lubm_q5 |
| Q6       |      80      | 125               |  1 | query/lubm/lubm_q6 |
| Q7       | 322,475      | 112,559           | 16 | query/lubm/lubm_q7 |

| Workload | Thpt (q/s) | Configuration   | File                           |
| :------: | ---------: | :-------------- | :----------------------------- |
| A1-A6    | 65.7079K   | -d 5 -w 1 -p 1  | query/lubm/emulator/mix_config |
| A1-A6    | 71.1509K   | -d 5 -w 1 -p 5  | query/lubm/emulator/mix_config |
| A1-A6    | 70.6853K   | -d 5 -w 1 -p 10  | query/lubm/emulator/mix_config |
| A1-A6    | 71.4191K   | -d 5 -w 1 -p 20  | query/lubm/emulator/mix_config |
| A1-A6    | 70.8027K   | -d 5 -w 1 -p 30  | query/lubm/emulator/mix_config |

#### Detail

```bash
wukong> sparql -f query/lubm/lubm_q1 -m 16 -n 10
INFO:     Parsing time: 54 usec
INFO:     (average) latency: 415420 usec
INFO:     (last) result size: 2528
wukong> sparql -f query/lubm/lubm_q2 -m 16 -n 10
INFO:     Parsing time: 39 usec
INFO:     (average) latency: 65635 usec
INFO:     (last) result size: 2765067
wukong> sparql -f query/lubm/lubm_q3 -m 16 -n 10
INFO:     Parsing time: 54 usec
INFO:     (average) latency: 192233 usec
INFO:     (last) result size: 0
wukong> sparql -f query/lubm/lubm_q4 -n 1000
INFO:     Parsing time: 52 usec
INFO:     (average) latency: 24 usec
INFO:     (last) result size: 10
wukong> sparql -f query/lubm/lubm_q5 -n 1000
INFO:     Parsing time: 35 usec
INFO:     (average) latency: 17 usec
INFO:     (last) result size: 10
wukong> sparql -f query/lubm/lubm_q6 -n 1000
INFO:     Parsing time: 38 usec
INFO:     (average) latency: 80 usec
INFO:     (last) result size: 125
wukong> sparql -f query/lubm/lubm_q7 -m 16 -n 10
INFO:     Parsing time: 78 usec
INFO:     (average) latency: 322475 usec
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
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 1
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    19    20    138    38    22    177    
INFO:     5    25    26    150    46    29    201    
INFO:     10    31    32    157    51    35    213    
INFO:     15    35    36    163    54    38    223    
INFO:     20    37    38    168    56    41    234    
INFO:     25    39    40    173    58    42    243    
INFO:     30    41    42    178    60    44    251    
INFO:     35    42    43    182    62    46    259    
INFO:     40    44    45    186    63    47    266    
INFO:     45    45    46    191    65    49    272    
INFO:     50    47    48    196    66    50    279    
INFO:     55    48    49    201    68    52    286    
INFO:     60    50    51    206    69    53    292    
INFO:     65    51    52    211    71    54    301    
INFO:     70    53    54    218    73    56    310    
INFO:     75    54    55    225    75    58    321    
INFO:     80    56    57    233    77    60    336    
INFO:     85    59    59    241    80    62    352    
INFO:     90    61    62    250    84    65    369    
INFO:     95    66    67    262    89    70    388    
INFO:     96    67    68    265    90    71    392    
INFO:     97    69    70    268    92    73    397    
INFO:     98    70    71    273    94    75    404    
INFO:     99    73    74    279    97    78    413    
INFO:     100    375    238    584    215    2196    2146    
INFO:     Throughput: 65.7079K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 5
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    41    42    180    59    43    242    
INFO:     5    99    100    225    115    103    284    
INFO:     10    123    124    244    135    125    306    
INFO:     15    134    135    256    144    136    322    
INFO:     20    142    143    266    151    144    334    
INFO:     25    148    149    276    157    150    345    
INFO:     30    154    155    285    163    155    357    
INFO:     35    159    160    294    167    160    365    
INFO:     40    164    164    302    172    165    375    
INFO:     45    168    169    310    178    170    385    
INFO:     50    173    174    318    183    174    395    
INFO:     55    178    179    327    188    179    405    
INFO:     60    183    184    337    194    184    417    
INFO:     65    189    189    346    199    190    430    
INFO:     70    194    195    356    206    196    443    
INFO:     75    201    202    368    214    202    459    
INFO:     80    208    209    380    225    210    476    
INFO:     85    219    220    395    240    221    496    
INFO:     90    235    237    418    266    239    522    
INFO:     95    289    290    458    321    295    569    
INFO:     96    312    312    471    337    315    581    
INFO:     97    343    341    497    365    344    600    
INFO:     98    384    384    536    402    386    637    
INFO:     99    448    447    602    465    450    707    
INFO:     100    1579    1594    1120    1501    1627    1576    
INFO:     Throughput: 71.1509K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 10
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    112    110    232    116    111    282    
INFO:     5    214    213    276    221    215    333    
INFO:     10    238    239    303    245    240    365    
INFO:     15    253    255    321    261    255    386    
INFO:     20    266    267    336    273    267    404    
INFO:     25    276    277    348    283    278    419    
INFO:     30    286    287    359    292    287    436    
INFO:     35    295    295    370    301    296    453    
INFO:     40    303    304    380    309    304    473    
INFO:     45    310    311    391    317    312    495    
INFO:     50    318    319    401    325    320    514    
INFO:     55    326    327    413    333    328    532    
INFO:     60    334    335    428    342    336    550    
INFO:     65    343    344    443    350    344    568    
INFO:     70    352    353    461    359    353    590    
INFO:     75    361    363    480    369    363    613    
INFO:     80    373    374    502    380    374    642    
INFO:     85    386    387    535    394    387    669    
INFO:     90    402    403    591    412    404    702    
INFO:     95    431    433    658    443    434    751    
INFO:     96    444    444    675    460    446    763    
INFO:     97    468    471    699    493    474    783    
INFO:     98    530    533    732    562    536    806    
INFO:     99    621    625    799    645    621    865    
INFO:     100    2008    2024    1844    1090    2030    1373    
INFO:     Throughput: 70.6853K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 20
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    161    159    252    169    161    340    
INFO:     5    393    390    431    401    392    463    
INFO:     10    443    443    490    452    444    516    
INFO:     15    475    475    525    481    476    551    
INFO:     20    499    499    553    505    500    583    
INFO:     25    521    520    577    527    521    607    
INFO:     30    539    539    597    546    540    630    
INFO:     35    557    557    616    564    557    650    
INFO:     40    573    572    635    580    573    668    
INFO:     45    588    588    653    596    588    685    
INFO:     50    603    603    671    611    603    704    
INFO:     55    618    618    687    625    619    723    
INFO:     60    633    633    705    640    633    741    
INFO:     65    649    649    722    656    649    761    
INFO:     70    665    665    739    671    665    781    
INFO:     75    683    683    761    689    682    804    
INFO:     80    702    702    784    708    701    831    
INFO:     85    723    724    811    730    723    862    
INFO:     90    751    752    843    758    751    901    
INFO:     95    792    792    890    797    793    967    
INFO:     96    804    804    903    810    804    991    
INFO:     97    820    820    922    828    819    1038    
INFO:     98    839    842    955    850    839    1144    
INFO:     99    874    881    1109    891    875    1300    
INFO:     100    3187    3263    2389    2201    3248    2658    
INFO:     Throughput: 71.4191K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 30
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    396    387    505    360    374    555    
INFO:     5    611    612    671    620    610    684    
INFO:     10    669    670    735    679    669    749    
INFO:     15    710    710    778    718    709    795    
INFO:     20    742    743    808    752    743    831    
INFO:     25    771    771    838    779    771    864    
INFO:     30    796    796    866    804    797    894    
INFO:     35    820    821    890    829    821    920    
INFO:     40    843    844    913    851    845    946    
INFO:     45    866    866    937    874    868    972    
INFO:     50    888    888    961    896    890    998    
INFO:     55    910    910    984    918    912    1024    
INFO:     60    933    933    1009    942    935    1048    
INFO:     65    956    956    1033    965    958    1074    
INFO:     70    981    981    1058    990    983    1102    
INFO:     75    1008    1008    1084    1018    1009    1130    
INFO:     80    1038    1039    1117    1049    1039    1164    
INFO:     85    1071    1073    1152    1082    1073    1200    
INFO:     90    1112    1114    1193    1124    1115    1248    
INFO:     95    1172    1174    1257    1183    1175    1313    
INFO:     96    1188    1191    1275    1200    1191    1331    
INFO:     97    1208    1212    1296    1219    1211    1348    
INFO:     98    1233    1237    1323    1243    1235    1375    
INFO:     99    1274    1277    1366    1283    1276    1425    
INFO:     100    2814    2827    2812    2694    2844    2161    
INFO:     Throughput: 70.8027K queries/sec
wukong> q 
```


<br>
<a name="res2"></a>
## Experimantal results (w/ zeromq)

#### Summary

| Workload | Latency (us) | #Results (lines)  | TH | File               |
| :------: | -----------: | ----------------: | -: | :----------------- |
| Q1       | 416,150      | 2528              | 16 | query/lubm/lubm_q1 |
| Q2       |  66,010      | 2,765,067         | 16 | query/lubm/lubm_q2 |
| Q3       | 192,333      | 0                 | 16 | query/lubm/lubm_q3 |
| Q4       |      58      | 10                |  1 | query/lubm/lubm_q4 |
| Q5       |      51      | 10                |  1 | query/lubm/lubm_q5 |
| Q6       |     123      | 125               |  1 | query/lubm/lubm_q6 |
| Q7       | 322,577      | 112,559           | 16 | query/lubm/lubm_q7 |

| Workload | Thpt (q/s) | Configuration   | File                           |
| :------: | ---------: | :-------------- | :----------------------------- |
| A1-A6    | 24.8135K   | -d 5 -w 1 -p 1  | query/lubm/emulator/mix_config |
| A1-A6    | 43.7478K   | -d 5 -w 1 -p 5  | query/lubm/emulator/mix_config |
| A1-A6    | 53.3552K   | -d 5 -w 1 -p 10  | query/lubm/emulator/mix_config |
| A1-A6    | 60.7887K   | -d 5 -w 1 -p 20  | query/lubm/emulator/mix_config |
| A1-A6    | 63.8647K   | -d 5 -w 1 -p 30  | query/lubm/emulator/mix_config |
| A1-A6    | 66.6167K   | -d 5 -w 1 -p 50  | query/lubm/emulator/mix_config |
| A1-A6    | 67.3030K   | -d 5 -w 1 -p 80  | query/lubm/emulator/mix_config |
| A1-A6    | 67.9672K   | -d 5 -w 1 -p 100  | query/lubm/emulator/mix_config |
| A1-A6    | 68.5924K   | -d 5 -w 1 -p 200  | query/lubm/emulator/mix_config |
| A1-A6    | 69.0850K   | -d 5 -w 1 -p 500  | query/lubm/emulator/mix_config |
| A1-A6    | 69.0864K   | -d 5 -w 1 -p 1000  | query/lubm/emulator/mix_config |


#### Detail

```bash
wukong> config -v
wukong> config -s global_use_rdma=0
wukong> 
wukong> sparql -f query/lubm/lubm_q1 -m 16 -n 10
WARNING:  RDMA is not enabled, skip corun optimization!
INFO:     Parsing time: 60 usec
INFO:     (average) latency: 416150 usec
INFO:     (last) result size: 2528
wukong> sparql -f query/lubm/lubm_q2 -m 16 -n 10
INFO:     Parsing time: 44 usec
INFO:     (average) latency: 66010 usec
INFO:     (last) result size: 2765067
wukong> sparql -f query/lubm/lubm_q3 -m 16 -n 10
INFO:     Parsing time: 56 usec
INFO:     (average) latency: 192333 usec
INFO:     (last) result size: 0
wukong> sparql -f query/lubm/lubm_q4 -n 1000
INFO:     Parsing time: 41 usec
INFO:     (average) latency: 58 usec
INFO:     (last) result size: 10
wukong> sparql -f query/lubm/lubm_q5 -n 1000
INFO:     Parsing time: 38 usec
INFO:     (average) latency: 51 usec
INFO:     (last) result size: 10
wukong> sparql -f query/lubm/lubm_q6 -n 1000
INFO:     Parsing time: 37 usec
INFO:     (average) latency: 123 usec
INFO:     (last) result size: 125
wukong> sparql -f query/lubm/lubm_q7 -m 16 -n 10
INFO:     Parsing time: 76 usec
INFO:     (average) latency: 322577 usec
INFO:     (last) result size: 112559
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 1
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P Q1  Q2  Q3  Q4  Q5  Q6
INFO:     1 80  81  207 99  83  263 
INFO:     5 92  92  226 110 95  289 
INFO:     10    98  99  238 119 101 313 
INFO:     15    104 104 248 127 107 327 
INFO:     20    108 109 259 133 112 341 
INFO:     25    113 114 269 138 117 352 
INFO:     30    118 119 278 143 122 363 
INFO:     35    122 124 286 148 127 373 
INFO:     40    127 129 295 152 131 383 
INFO:     45    132 133 304 157 136 392 
INFO:     50    137 138 312 161 141 401 
INFO:     55    142 143 319 166 146 412 
INFO:     60    147 148 327 171 151 423 
INFO:     65    152 154 334 176 156 437 
INFO:     70    158 160 343 182 162 452 
INFO:     75    165 166 351 189 169 466 
INFO:     80    172 173 360 197 176 483 
INFO:     85    179 181 371 205 184 500 
INFO:     90    188 190 384 214 193 519 
INFO:     95    200 201 406 227 204 543 
INFO:     96    203 204 412 230 207 550 
INFO:     97    207 208 418 234 211 560 
INFO:     98    213 213 430 239 216 571 
INFO:     99    220 220 444 248 224 593 
INFO:     100   767 1995    549 778 1451    1936    
INFO:     Throughput: 24.8135K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 5
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    149    152    302    172    150    379    
INFO:     5    242    245    379    264    246    455    
INFO:     10    275    278    417    298    280    496    
INFO:     15    297    300    440    320    302    524    
INFO:     20    315    317    458    336    320    545    
INFO:     25    330    332    476    351    334    566    
INFO:     30    344    346    492    364    348    585    
INFO:     35    358    360    506    378    362    606    
INFO:     40    372    374    522    391    376    623    
INFO:     45    387    388    537    406    390    640    
INFO:     50    401    402    554    418    404    656    
INFO:     55    413    415    569    429    416    671    
INFO:     60    424    426    586    439    426    688    
INFO:     65    435    436    601    450    437    705    
INFO:     70    445    446    618    460    447    722    
INFO:     75    456    457    634    470    458    741    
INFO:     80    467    468    653    481    469    764    
INFO:     85    480    481    673    494    482    794    
INFO:     90    496    497    698    513    498    827    
INFO:     95    527    527    737    546    528    876    
INFO:     96    538    538    749    560    540    891    
INFO:     97    559    559    768    588    561    905    
INFO:     98    616    616    800    643    614    930    
INFO:     99    705    705    879    713    699    1005    
INFO:     100    12534    12439    12379    3300    12504    5551    
INFO:     Throughput: 43.7478K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 10
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    230    226    405    251    235    465    
INFO:     5    363    363    516    386    369    592    
INFO:     10    426    427    587    448    431    657    
INFO:     15    467    468    626    489    472    690    
INFO:     20    501    502    654    521    506    718    
INFO:     25    530    531    674    551    534    743    
INFO:     30    555    557    693    576    559    767    
INFO:     35    577    578    709    597    581    786    
INFO:     40    596    598    726    616    601    810    
INFO:     45    615    615    742    634    619    833    
INFO:     50    631    632    758    650    635    859    
INFO:     55    647    647    775    665    651    885    
INFO:     60    662    662    793    680    666    911    
INFO:     65    677    677    813    694    681    935    
INFO:     70    693    693    834    709    696    963    
INFO:     75    709    709    859    725    712    991    
INFO:     80    728    729    884    745    732    1023    
INFO:     85    752    754    919    770    756    1061    
INFO:     90    787    788    964    804    790    1104    
INFO:     95    852    852    1038    870    855    1175    
INFO:     96    874    873    1062    892    878    1196    
INFO:     97    905    901    1101    921    908    1233    
INFO:     98    944    944    1148    966    954    1277    
INFO:     99    1032    1029    1223    1054    1036    1358    
INFO:     100    4681    4682    4478    4680    4665    4405    
INFO:     Throughput: 53.3552K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 20
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    364    366    545    397    371    623    
INFO:     5    541    542    701    564    544    791    
INFO:     10    632    633    800    655    634    890    
INFO:     15    695    698    874    720    697    969    
INFO:     20    747    749    930    769    747    1032    
INFO:     25    791    792    981    814    791    1081    
INFO:     30    830    831    1023    853    831    1133    
INFO:     35    867    868    1069    893    868    1178    
INFO:     40    904    905    1113    931    906    1220    
INFO:     45    941    942    1154    968    943    1257    
INFO:     50    979    981    1191    1009    981    1292    
INFO:     55    1018    1021    1228    1046    1023    1324    
INFO:     60    1060    1066    1265    1088    1066    1355    
INFO:     65    1107    1111    1303    1134    1112    1386    
INFO:     70    1155    1158    1338    1181    1160    1421    
INFO:     75    1206    1209    1375    1232    1210    1458    
INFO:     80    1258    1258    1413    1280    1261    1499    
INFO:     85    1313    1313    1460    1332    1315    1543    
INFO:     90    1376    1374    1513    1394    1377    1598    
INFO:     95    1464    1463    1594    1483    1467    1688    
INFO:     96    1488    1488    1615    1509    1493    1719    
INFO:     97    1519    1519    1648    1539    1524    1748    
INFO:     98    1565    1563    1698    1587    1568    1789    
INFO:     99    1643    1641    1805    1663    1640    1877    
INFO:     100    5256    5221    5269    5206    5210    3381    
INFO:     Throughput: 60.7887K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 30
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    434    437    578    458    437    646    
INFO:     5    653    651    795    682    660    871    
INFO:     10    792    789    935    818    795    1030    
INFO:     15    895    897    1048    927    901    1150    
INFO:     20    982    983    1144    1016    986    1247    
INFO:     25    1053    1056    1222    1084    1058    1328    
INFO:     30    1118    1120    1296    1151    1125    1403    
INFO:     35    1179    1181    1364    1214    1187    1471    
INFO:     40    1239    1242    1435    1273    1246    1531    
INFO:     45    1299    1302    1494    1330    1305    1589    
INFO:     50    1357    1360    1555    1388    1362    1644    
INFO:     55    1416    1419    1615    1449    1421    1707    
INFO:     60    1477    1479    1673    1508    1481    1769    
INFO:     65    1541    1541    1733    1573    1546    1821    
INFO:     70    1607    1608    1792    1639    1612    1881    
INFO:     75    1676    1677    1853    1704    1680    1936    
INFO:     80    1745    1747    1912    1770    1750    1994    
INFO:     85    1822    1824    1976    1847    1824    2059    
INFO:     90    1907    1909    2053    1930    1909    2139    
INFO:     95    2025    2023    2159    2043    2024    2248    
INFO:     96    2059    2056    2191    2073    2058    2280    
INFO:     97    2100    2099    2237    2115    2099    2335    
INFO:     98    2155    2154    2298    2164    2153    2399    
INFO:     99    2244    2251    2403    2260    2243    2519    
INFO:     100    5155    5153    3306    5163    5146    5123    
INFO:     Throughput: 63.8647K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 50
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    586    614    758    621    607    862    
INFO:     5    943    943    1097    981    941    1151    
INFO:     10    1174    1179    1319    1212    1177    1366    
INFO:     15    1356    1359    1502    1392    1357    1547    
INFO:     20    1509    1514    1670    1547    1509    1718    
INFO:     25    1638    1641    1816    1679    1637    1882    
INFO:     30    1749    1752    1933    1788    1749    2022    
INFO:     35    1852    1854    2045    1894    1853    2130    
INFO:     40    1950    1950    2144    1995    1950    2234    
INFO:     45    2042    2041    2234    2086    2043    2320    
INFO:     50    2130    2131    2319    2170    2130    2400    
INFO:     55    2216    2218    2400    2257    2215    2481    
INFO:     60    2299    2300    2478    2337    2297    2559    
INFO:     65    2383    2383    2554    2418    2379    2636    
INFO:     70    2469    2469    2639    2501    2467    2719    
INFO:     75    2562    2564    2726    2591    2560    2805    
INFO:     80    2664    2667    2835    2688    2663    2919    
INFO:     85    2779    2783    2953    2801    2782    3033    
INFO:     90    2922    2921    3096    2937    2923    3178    
INFO:     95    3118    3115    3277    3128    3122    3383    
INFO:     96    3172    3171    3322    3185    3177    3432    
INFO:     97    3240    3241    3390    3249    3244    3495    
INFO:     98    3330    3331    3484    3337    3333    3586    
INFO:     99    3486    3493    3680    3482    3495    3794    
INFO:     100    12125    12337    11959    11914    12324    12245    
INFO:     Throughput: 66.6167K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 80
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    1145    1160    1276    1185    1153    1357    
INFO:     5    1504    1517    1668    1525    1520    1741    
INFO:     10    1828    1843    1970    1843    1851    2064    
INFO:     15    2107    2123    2266    2135    2127    2343    
INFO:     20    2356    2369    2519    2379    2367    2598    
INFO:     25    2572    2581    2756    2596    2582    2828    
INFO:     30    2759    2763    2939    2778    2768    3025    
INFO:     35    2922    2926    3111    2937    2928    3222    
INFO:     40    3067    3074    3264    3078    3074    3371    
INFO:     45    3201    3208    3399    3215    3211    3497    
INFO:     50    3328    3332    3524    3338    3335    3616    
INFO:     55    3451    3452    3641    3466    3457    3728    
INFO:     60    3568    3571    3752    3586    3573    3834    
INFO:     65    3681    3686    3862    3708    3687    3932    
INFO:     70    3798    3803    3964    3825    3803    4033    
INFO:     75    3920    3924    4079    3948    3925    4144    
INFO:     80    4050    4058    4202    4074    4058    4274    
INFO:     85    4195    4206    4363    4215    4203    4433    
INFO:     90    4385    4397    4567    4401    4392    4631    
INFO:     95    4654    4657    4825    4663    4656    4908    
INFO:     96    4730    4732    4901    4735    4729    4977    
INFO:     97    4822    4821    4991    4820    4821    5075    
INFO:     98    4934    4931    5115    4916    4936    5202    
INFO:     99    5115    5109    5333    5080    5119    5401    
INFO:     100    8285    8049    7118    7098    8342    7082    
INFO:     Throughput: 67.303K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 100
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    1464    1468    1574    1486    1465    1673    
INFO:     5    1880    1875    2026    1890    1869    2113    
INFO:     10    2222    2217    2350    2220    2223    2445    
INFO:     15    2566    2565    2715    2554    2571    2840    
INFO:     20    2893    2896    3041    2872    2908    3180    
INFO:     25    3171    3172    3348    3174    3187    3470    
INFO:     30    3399    3409    3592    3412    3418    3750    
INFO:     35    3609    3622    3816    3642    3626    3978    
INFO:     40    3809    3816    4003    3834    3822    4166    
INFO:     45    3996    4000    4186    4022    3999    4325    
INFO:     50    4165    4169    4346    4186    4162    4472    
INFO:     55    4318    4323    4502    4338    4317    4612    
INFO:     60    4467    4470    4647    4488    4468    4748    
INFO:     65    4612    4609    4779    4630    4608    4865    
INFO:     70    4756    4754    4911    4772    4750    4983    
INFO:     75    4898    4895    5047    4920    4896    5125    
INFO:     80    5054    5051    5192    5066    5053    5287    
INFO:     85    5234    5232    5368    5238    5230    5465    
INFO:     90    5441    5440    5578    5439    5438    5667    
INFO:     95    5745    5741    5889    5749    5739    5969    
INFO:     96    5834    5830    5978    5841    5835    6047    
INFO:     97    5949    5936    6072    5951    5939    6156    
INFO:     98    6090    6075    6242    6091    6085    6321    
INFO:     99    6325    6324    6453    6294    6313    6558    
INFO:     100    9494    9507    9248    9421    9514    9303    
INFO:     Throughput: 67.9672K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 200
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    2831    2824    2963    2820    2821    3048    
INFO:     5    3696    3670    3928    3683    3691    4054    
INFO:     10    4463    4452    4695    4460    4470    4741    
INFO:     15    5240    5219    5460    5228    5243    5455    
INFO:     20    5914    5905    6156    5925    5920    6184    
INFO:     25    6460    6438    6678    6475    6470    6774    
INFO:     30    6901    6896    7147    6924    6914    7292    
INFO:     35    7320    7311    7594    7342    7326    7727    
INFO:     40    7693    7697    7975    7732    7705    8091    
INFO:     45    8043    8043    8290    8081    8047    8402    
INFO:     50    8336    8334    8568    8357    8335    8685    
INFO:     55    8592    8588    8814    8608    8596    8930    
INFO:     60    8817    8816    9022    8843    8831    9143    
INFO:     65    9039    9040    9236    9067    9053    9328    
INFO:     70    9255    9256    9450    9274    9266    9521    
INFO:     75    9469    9474    9665    9503    9489    9716    
INFO:     80    9705    9703    9866    9730    9715    9939    
INFO:     85    9970    9958    10134    9989    9973    10190    
INFO:     90    10313    10305    10478    10330    10310    10535    
INFO:     95    10807    10807    10964    10852    10822    11023    
INFO:     96    10953    10950    11106    10976    10960    11152    
INFO:     97    11117    11116    11262    11139    11119    11296    
INFO:     98    11313    11311    11442    11333    11307    11540    
INFO:     99    11633    11612    11761    11648    11609    11883    
INFO:     100    13887    13986    13218    16089    16920    13946    
INFO:     Throughput: 68.5924K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 500
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    7369    7363    7520    7433    7357    7535    
INFO:     5    9689    9678    9793    9849    9689    9789    
INFO:     10    11817    11826    11999    11960    11877    11986    
INFO:     15    13617    13650    13814    13744    13641    13648    
INFO:     20    14940    14982    15196    15076    14963    15069    
INFO:     25    15927    15955    16171    15998    15940    16101    
INFO:     30    16781    16807    17007    16855    16783    16934    
INFO:     35    17548    17589    17815    17596    17573    17787    
INFO:     40    18308    18341    18547    18355    18342    18630    
INFO:     45    19082    19099    19323    19138    19098    19376    
INFO:     50    19792    19808    20013    19849    19815    20056    
INFO:     55    20431    20458    20605    20481    20463    20630    
INFO:     60    21008    21035    21187    21074    21043    21267    
INFO:     65    21526    21555    21680    21589    21556    21738    
INFO:     70    22002    22031    22134    22049    22022    22227    
INFO:     75    22475    22505    22584    22534    22499    22642    
INFO:     80    22940    22960    23050    22953    22953    23096    
INFO:     85    23424    23452    23554    23429    23442    23577    
INFO:     90    24018    24061    24150    24042    24030    24162    
INFO:     95    24845    24863    24934    24815    24837    25028    
INFO:     96    25060    25095    25161    25051    25069    25302    
INFO:     97    25327    25372    25475    25346    25348    25499    
INFO:     98    25660    25699    25823    25637    25684    25853    
INFO:     99    26274    26311    26365    26277    26312    26545    
INFO:     100    29583    29536    29574    29548    29481    28607    
INFO:     Throughput: 69.085K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 1000
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    13967    14011    14069    14165    14036    13870    
INFO:     5    19875    19946    20197    20016    19912    20128    
INFO:     10    23607    23623    23921    23822    23607    23895    
INFO:     15    25988    25970    26369    26146    25919    26334    
INFO:     20    28212    28099    28528    28180    28157    28476    
INFO:     25    30088    29947    30412    30063    29999    30433    
INFO:     30    31715    31637    32167    31752    31613    32140    
INFO:     35    33147    33103    33607    33176    33046    33546    
INFO:     40    34613    34615    35111    34660    34551    34837    
INFO:     45    36136    36119    36622    36124    36060    36363    
INFO:     50    37809    37794    38249    37740    37740    37995    
INFO:     55    39509    39485    39887    39592    39444    39686    
INFO:     60    41101    41106    41474    41209    41065    41331    
INFO:     65    42501    42519    42785    42597    42488    42800    
INFO:     70    43564    43577    43815    43632    43558    43832    
INFO:     75    44492    44490    44707    44518    44485    44754    
INFO:     80    45449    45441    45612    45434    45436    45714    
INFO:     85    46385    46366    46602    46335    46376    46595    
INFO:     90    47346    47344    47533    47286    47317    47538    
INFO:     95    48718    48696    48919    48637    48678    49029    
INFO:     96    49185    49180    49400    49157    49108    49473    
INFO:     97    49835    49852    50066    49843    49795    50153    
INFO:     98    50671    50726    50847    50672    50662    50939    
INFO:     99    51815    51769    51908    51856    51735    52028    
INFO:     100    56372    56337    55992    56164    56358    55684    
INFO:     Throughput: 69.0864K queries/sec
wukong> q 
```
