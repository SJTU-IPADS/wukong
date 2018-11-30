# Performance (single node, dynamic gstore, jemalloc)

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

##### Gitlab Version: @9c60ecc

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
$./build.sh -DUSE_RDMA=ON -DUSE_GPU=OFF -DUSE_HADOOP=OFF -DUSE_JEMALLOC=ON -DUSE_DYNAMIC_GSTORE=ON -DUSE_VERSATILE=OFF -DUSE_DTYPE_64BIT=OFF
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
| Q1       | 423,522      | 2528              | 16 | query/lubm/lubm_q1 |
| Q2       |  66,840      | 2,765,067         | 16 | query/lubm/lubm_q2 |
| Q3       | 195,773      | 0                 | 16 | query/lubm/lubm_q3 |
| Q4       |      24      | 10                |  1 | query/lubm/lubm_q4 |
| Q5       |      18      | 10                |  1 | query/lubm/lubm_q5 |
| Q6       |      79      | 125               |  1 | query/lubm/lubm_q6 |
| Q7       | 322,197      | 112,559           | 16 | query/lubm/lubm_q7 |

| Workload | Thpt (q/s) | Configuration   | File                           |
| :------: | ---------: | :-------------- | :----------------------------- |
| A1-A6    | 63.0946K   | -d 5 -w 1 -p 1  | query/lubm/emulator/mix_config |
| A1-A6    | 70.8256K   | -d 5 -w 1 -p 5  | query/lubm/emulator/mix_config |
| A1-A6    | 68.7502K   | -d 5 -w 1 -p 10  | query/lubm/emulator/mix_config |
| A1-A6    | 71.4652K   | -d 5 -w 1 -p 20  | query/lubm/emulator/mix_config |
| A1-A6    | 70.7068K   | -d 5 -w 1 -p 30  | query/lubm/emulator/mix_config |

#### Detail

```bash
wukong> sparql -f query/lubm/lubm_q1 -m 16 -n 10
INFO:     Parsing time: 51 usec
INFO:     (average) latency: 423522 usec
INFO:     (last) result size: 2528
wukong> sparql -f query/lubm/lubm_q2 -m 16 -n 10
INFO:     Parsing time: 38 usec
INFO:     (average) latency: 66840 usec
INFO:     (last) result size: 2765067
wukong> sparql -f query/lubm/lubm_q3 -m 16 -n 10
INFO:     Parsing time: 50 usec
INFO:     (average) latency: 195773 usec
INFO:     (last) result size: 0
wukong> sparql -f query/lubm/lubm_q4 -n 1000
INFO:     Parsing time: 40 usec
INFO:     (average) latency: 24 usec
INFO:     (last) result size: 10
wukong> sparql -f query/lubm/lubm_q5 -n 1000
INFO:     Parsing time: 30 usec
INFO:     (average) latency: 18 usec
INFO:     (last) result size: 10
wukong> sparql -f query/lubm/lubm_q6 -n 1000
INFO:     Parsing time: 38 usec
INFO:     (average) latency: 79 usec
INFO:     (last) result size: 125
wukong> sparql -f query/lubm/lubm_q7 -m 16 -n 10
INFO:     Parsing time: 49 usec
INFO:     (average) latency: 322197 usec
INFO:     (last) result size: 112559
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 1
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    19    20    126    37    22    170    
INFO:     5    24    25    140    45    28    199    
INFO:     10    28    29    147    50    32    211    
INFO:     15    32    33    153    53    36    221    
INFO:     20    35    36    157    55    39    231    
INFO:     25    37    39    162    57    41    241    
INFO:     30    39    40    166    59    43    249    
INFO:     35    41    42    170    61    45    256    
INFO:     40    43    44    175    63    46    264    
INFO:     45    44    45    179    64    48    270    
INFO:     50    46    47    184    66    49    277    
INFO:     55    47    48    190    67    51    284    
INFO:     60    49    50    197    69    52    293    
INFO:     65    50    51    208    71    54    302    
INFO:     70    52    53    226    73    56    314    
INFO:     75    54    55    266    76    57    327    
INFO:     80    56    57    313    79    60    344    
INFO:     85    58    60    375    83    62    365    
INFO:     90    62    63    474    89    66    398    
INFO:     95    67    69    650    140    72    725    
INFO:     96    68    71    712    146    74    839    
INFO:     97    71    73    795    152    77    1002    
INFO:     98    74    79    898    162    87    1249    
INFO:     99    106    120    1042    217    129    1670    
INFO:     100    2601    1310    1933    482    2743    5291    
INFO:     Throughput: 63.0946K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 5
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    42    42    157    59    44    226    
INFO:     5    102    102    207    118    104    277    
INFO:     10    125    126    228    136    127    300    
INFO:     15    136    137    241    146    138    315    
INFO:     20    144    144    252    153    145    326    
INFO:     25    150    151    261    159    151    336    
INFO:     30    156    156    270    164    157    346    
INFO:     35    161    161    278    169    162    356    
INFO:     40    165    166    286    174    167    366    
INFO:     45    170    171    294    179    171    375    
INFO:     50    175    175    302    184    176    384    
INFO:     55    180    180    310    189    181    395    
INFO:     60    185    185    319    195    186    406    
INFO:     65    190    191    328    201    191    418    
INFO:     70    196    197    339    207    197    432    
INFO:     75    203    203    352    215    204    446    
INFO:     80    210    211    367    225    212    462    
INFO:     85    220    221    386    240    222    478    
INFO:     90    236    238    412    265    240    502    
INFO:     95    285    287    454    317    290    541    
INFO:     96    305    308    472    334    308    552    
INFO:     97    332    333    494    358    334    571    
INFO:     98    372    372    530    393    375    604    
INFO:     99    436    436    586    454    435    671    
INFO:     100    1937    2439    2318    1202    2058    1256    
INFO:     Throughput: 70.8256K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 10
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    126    129    229    150    131    293    
INFO:     5    202    204    274    217    206    342    
INFO:     10    232    234    298    243    234    372    
INFO:     15    250    251    317    259    251    392    
INFO:     20    263    265    333    272    265    410    
INFO:     25    274    275    346    283    276    427    
INFO:     30    284    285    358    293    286    444    
INFO:     35    293    294    370    302    295    464    
INFO:     40    302    303    382    310    303    483    
INFO:     45    310    311    394    318    312    505    
INFO:     50    319    319    408    327    320    528    
INFO:     55    327    327    424    334    328    550    
INFO:     60    335    336    446    343    337    572    
INFO:     65    344    345    472    352    345    596    
INFO:     70    354    354    511    363    355    624    
INFO:     75    364    365    568    374    366    658    
INFO:     80    377    377    640    388    378    699    
INFO:     85    392    392    721    404    394    753    
INFO:     90    414    414    831    430    415    867    
INFO:     95    486    481    1059    539    483    1209    
INFO:     96    550    544    1130    599    547    1356    
INFO:     97    629    624    1228    677    628    1529    
INFO:     98    740    739    1384    799    740    1781    
INFO:     99    1023    1013    1704    1081    1026    2227    
INFO:     100    5232    4582    4491    4079    5253    5283    
INFO:     Throughput: 68.7502K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 20
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    117    114    225    123    115    287    
INFO:     5    398    396    438    403    398    458    
INFO:     10    446    446    491    452    448    513    
INFO:     15    477    476    522    482    477    552    
INFO:     20    500    500    551    506    501    582    
INFO:     25    521    521    576    527    522    607    
INFO:     30    540    539    596    546    541    628    
INFO:     35    557    556    614    563    558    648    
INFO:     40    573    573    632    579    574    667    
INFO:     45    589    588    649    595    590    687    
INFO:     50    605    604    666    611    606    705    
INFO:     55    621    619    682    627    621    724    
INFO:     60    636    636    699    643    637    743    
INFO:     65    653    652    716    660    653    763    
INFO:     70    670    670    737    677    671    785    
INFO:     75    688    688    761    695    689    810    
INFO:     80    708    709    787    716    709    836    
INFO:     85    732    733    816    739    732    865    
INFO:     90    761    761    849    767    760    905    
INFO:     95    802    802    898    809    802    966    
INFO:     96    814    813    917    820    814    995    
INFO:     97    828    828    940    833    829    1048    
INFO:     98    848    848    977    853    849    1174    
INFO:     99    885    885    1163    897    887    1344    
INFO:     100    3009    3123    2432    2992    2436    2859    
INFO:     Throughput: 71.4652K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 30
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    275    276    339    289    280    405    
INFO:     5    603    601    651    612    604    661    
INFO:     10    667    667    718    675    669    740    
INFO:     15    710    710    762    717    711    788    
INFO:     20    742    743    799    749    744    827    
INFO:     25    771    771    830    778    773    861    
INFO:     30    797    796    857    804    799    891    
INFO:     35    822    820    881    829    823    919    
INFO:     40    845    843    906    851    846    943    
INFO:     45    868    866    929    874    868    967    
INFO:     50    891    888    952    898    891    993    
INFO:     55    913    911    976    920    913    1017    
INFO:     60    936    935    1001    943    936    1041    
INFO:     65    960    959    1024    967    960    1068    
INFO:     70    984    983    1050    992    985    1097    
INFO:     75    1010    1011    1079    1019    1011    1125    
INFO:     80    1041    1040    1111    1047    1041    1159    
INFO:     85    1074    1074    1144    1081    1074    1195    
INFO:     90    1115    1115    1189    1123    1117    1241    
INFO:     95    1177    1177    1249    1185    1178    1304    
INFO:     96    1193    1193    1265    1199    1195    1323    
INFO:     97    1213    1212    1288    1221    1214    1343    
INFO:     98    1239    1238    1316    1247    1239    1369    
INFO:     99    1280    1278    1354    1294    1283    1416    
INFO:     100    2922    2912    2868    2930    2914    2782    
INFO:     Throughput: 70.7068K queries/sec
wukong> q 
```


<br>
<a name="res2"></a>
## Experimantal results (w/ zeromq)

#### Summary

| Workload | Latency (us) | #Results (lines)  | TH | File               |
| :------: | -----------: | ----------------: | -: | :----------------- |
| Q1       | 424,014      | 2528              | 16 | query/lubm/lubm_q1 |
| Q2       |  67,247      | 2,765,067         | 16 | query/lubm/lubm_q2 |
| Q3       | 196,179      | 0                 | 16 | query/lubm/lubm_q3 |
| Q4       |      60      | 10                |  1 | query/lubm/lubm_q4 |
| Q5       |      51      | 10                |  1 | query/lubm/lubm_q5 |
| Q6       |     119      | 125               |  1 | query/lubm/lubm_q6 |
| Q7       | 323,367      | 112,559           | 16 | query/lubm/lubm_q7 |

| Workload | Thpt (q/s) | Configuration   | File                           |
| :------: | ---------: | :-------------- | :----------------------------- |
| A1-A6    | 24.0383K   | -d 5 -w 1 -p 1  | query/lubm/emulator/mix_config |
| A1-A6    | 41.5931K   | -d 5 -w 1 -p 5  | query/lubm/emulator/mix_config |
| A1-A6    | 53.5088K   | -d 5 -w 1 -p 10  | query/lubm/emulator/mix_config |
| A1-A6    | 60.0385K   | -d 5 -w 1 -p 20  | query/lubm/emulator/mix_config |
| A1-A6    | 63.2644K   | -d 5 -w 1 -p 30  | query/lubm/emulator/mix_config |
| A1-A6    | 65.1721K   | -d 5 -w 1 -p 50  | query/lubm/emulator/mix_config |
| A1-A6    | 67.0173K   | -d 5 -w 1 -p 80  | query/lubm/emulator/mix_config |
| A1-A6    | 67.2771K   | -d 5 -w 1 -p 100  | query/lubm/emulator/mix_config |
| A1-A6    | 68.0728K   | -d 5 -w 1 -p 200  | query/lubm/emulator/mix_config |
| A1-A6    | 68.8246K   | -d 5 -w 1 -p 500  | query/lubm/emulator/mix_config |
| A1-A6    | 68.9566K   | -d 5 -w 1 -p 1000  | query/lubm/emulator/mix_config |


#### Detail

```bash
wukong> config -v
wukong> config -s global_use_rdma=0
wukong> 
wukong> sparql -f query/lubm/lubm_q1 -m 16 -n 10
WARNING:  RDMA is not enabled, skip corun optimization!
INFO:     Parsing time: 65 usec
INFO:     (average) latency: 424014 usec
INFO:     (last) result size: 2528
wukong> sparql -f query/lubm/lubm_q2 -m 16 -n 10
INFO:     Parsing time: 40 usec
INFO:     (average) latency: 67247 usec
INFO:     (last) result size: 2765067
wukong> sparql -f query/lubm/lubm_q3 -m 16 -n 10
INFO:     Parsing time: 55 usec
INFO:     (average) latency: 196179 usec
INFO:     (last) result size: 0
wukong> sparql -f query/lubm/lubm_q4 -n 1000
INFO:     Parsing time: 51 usec
INFO:     (average) latency: 60 usec
INFO:     (last) result size: 10
wukong> sparql -f query/lubm/lubm_q5 -n 1000
INFO:     Parsing time: 30 usec
INFO:     (average) latency: 51 usec
INFO:     (last) result size: 10
wukong> sparql -f query/lubm/lubm_q6 -n 1000
INFO:     Parsing time: 38 usec
INFO:     (average) latency: 119 usec
INFO:     (last) result size: 125
wukong> sparql -f query/lubm/lubm_q7 -m 16 -n 10
INFO:     Parsing time: 58 usec
INFO:     (average) latency: 323367 usec
INFO:     (last) result size: 112559
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 1
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    81    82    199    100    84    261    
INFO:     5    92    93    219    112    95    290    
INFO:     10    98    99    230    119    101    309    
INFO:     15    102    104    239    126    106    323    
INFO:     20    107    108    246    132    111    336    
INFO:     25    110    112    253    137    115    348    
INFO:     30    114    116    262    142    119    360    
INFO:     35    119    120    270    146    124    369    
INFO:     40    123    125    279    150    128    379    
INFO:     45    127    129    288    154    132    390    
INFO:     50    132    134    299    159    137    401    
INFO:     55    137    138    309    163    141    414    
INFO:     60    141    143    320    168    146    427    
INFO:     65    146    148    332    173    151    443    
INFO:     70    152    154    350    178    157    460    
INFO:     75    158    160    372    185    163    481    
INFO:     80    165    167    401    193    170    505    
INFO:     85    174    176    429    203    180    534    
INFO:     90    185    188    473    216    191    587    
INFO:     95    200    203    533    239    207    705    
INFO:     96    204    207    556    247    211    731    
INFO:     97    209    212    577    255    216    788    
INFO:     98    215    219    617    267    224    880    
INFO:     99    225    230    668    302    237    1258    
INFO:     100    1342    1324    1440    4954    4863    4271    
INFO:     Throughput: 24.0383K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 5
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    167    154    314    178    164    410    
INFO:     5    267    268    398    289    270    479    
INFO:     10    302    303    432    326    306    516    
INFO:     15    325    327    453    350    329    546    
INFO:     20    345    346    469    368    348    568    
INFO:     25    362    364    484    384    366    587    
INFO:     30    377    379    500    398    381    606    
INFO:     35    390    391    514    409    393    624    
INFO:     40    401    402    530    419    404    640    
INFO:     45    411    412    546    428    414    655    
INFO:     50    419    420    563    436    422    670    
INFO:     55    428    428    579    444    430    684    
INFO:     60    436    437    594    452    438    701    
INFO:     65    444    445    611    459    446    717    
INFO:     70    452    452    627    468    454    738    
INFO:     75    460    461    647    477    462    758    
INFO:     80    470    471    669    487    472    783    
INFO:     85    481    482    695    499    483    814    
INFO:     90    496    497    735    516    498    848    
INFO:     95    523    523    799    548    524    906    
INFO:     96    532    533    821    559    534    932    
INFO:     97    546    547    851    581    548    966    
INFO:     98    572    572    885    617    575    1037    
INFO:     99    658    648    953    703    658    1192    
INFO:     100    3322    3646    2028    3815    2927    3838    
INFO:     Throughput: 41.5931K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 10
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    216    221    389    253    222    440    
INFO:     5    371    373    512    397    377    592    
INFO:     10    425    428    574    452    431    652    
INFO:     15    463    466    612    491    468    683    
INFO:     20    494    497    640    521    500    712    
INFO:     25    520    523    663    548    526    735    
INFO:     30    544    547    682    571    550    758    
INFO:     35    566    568    698    591    571    778    
INFO:     40    586    588    712    611    591    799    
INFO:     45    603    605    728    627    608    820    
INFO:     50    620    622    743    643    624    842    
INFO:     55    635    637    760    657    639    865    
INFO:     60    649    651    778    671    654    888    
INFO:     65    663    665    796    684    668    913    
INFO:     70    678    680    816    698    683    941    
INFO:     75    693    694    841    712    698    969    
INFO:     80    710    711    873    729    714    1004    
INFO:     85    730    731    914    749    734    1044    
INFO:     90    756    757    973    776    761    1093    
INFO:     95    804    805    1061    828    810    1175    
INFO:     96    821    822    1090    848    828    1201    
INFO:     97    847    849    1119    875    853    1238    
INFO:     98    893    894    1172    928    898    1307    
INFO:     99    1009    1005    1287    1036    1009    1443    
INFO:     100    14193    14165    3695    3897    14155    4250    
INFO:     Throughput: 53.5088K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 20
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    323    328    484    349    329    555    
INFO:     5    517    519    676    548    521    755    
INFO:     10    622    623    778    648    624    885    
INFO:     15    691    692    859    717    693    969    
INFO:     20    746    746    917    770    749    1034    
INFO:     25    792    791    974    815    795    1091    
INFO:     30    835    833    1027    861    837    1144    
INFO:     35    874    874    1077    901    877    1191    
INFO:     40    914    913    1121    942    917    1234    
INFO:     45    954    952    1163    985    957    1273    
INFO:     50    995    992    1205    1024    998    1308    
INFO:     55    1037    1035    1245    1069    1041    1340    
INFO:     60    1081    1079    1282    1116    1085    1369    
INFO:     65    1128    1127    1320    1159    1131    1401    
INFO:     70    1177    1174    1355    1206    1180    1433    
INFO:     75    1226    1223    1393    1251    1228    1466    
INFO:     80    1274    1272    1433    1295    1277    1506    
INFO:     85    1326    1323    1478    1345    1327    1551    
INFO:     90    1386    1385    1534    1403    1389    1609    
INFO:     95    1470    1470    1620    1485    1475    1715    
INFO:     96    1494    1496    1644    1510    1498    1752    
INFO:     97    1525    1529    1684    1542    1530    1810    
INFO:     98    1575    1574    1748    1588    1575    1890    
INFO:     99    1666    1664    1891    1683    1675    2035    
INFO:     100    3937    3930    3857    3917    3937    3870    
INFO:     Throughput: 60.0385K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 30
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    483    482    601    501    485    710    
INFO:     5    676    679    808    702    681    886    
INFO:     10    808    810    941    830    811    1030    
INFO:     15    910    908    1041    930    910    1140    
INFO:     20    993    993    1121    1013    994    1232    
INFO:     25    1064    1063    1203    1083    1064    1310    
INFO:     30    1128    1128    1273    1151    1128    1380    
INFO:     35    1190    1188    1340    1212    1189    1444    
INFO:     40    1245    1246    1408    1269    1248    1510    
INFO:     45    1302    1304    1467    1327    1304    1573    
INFO:     50    1360    1362    1524    1386    1362    1631    
INFO:     55    1419    1421    1584    1442    1421    1689    
INFO:     60    1477    1482    1643    1500    1480    1744    
INFO:     65    1539    1543    1700    1561    1542    1796    
INFO:     70    1603    1608    1756    1624    1605    1851    
INFO:     75    1670    1674    1815    1691    1672    1909    
INFO:     80    1737    1742    1874    1757    1741    1968    
INFO:     85    1809    1814    1942    1828    1815    2028    
INFO:     90    1892    1897    2022    1915    1898    2110    
INFO:     95    2011    2015    2138    2031    2017    2220    
INFO:     96    2046    2047    2170    2069    2050    2254    
INFO:     97    2089    2092    2207    2114    2092    2293    
INFO:     98    2146    2152    2270    2180    2151    2353    
INFO:     99    2247    2256    2385    2286    2256    2461    
INFO:     100    18526    18517    18456    18463    18529    7392    
INFO:     Throughput: 63.2644K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 50
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    764    767    869    792    769    1011    
INFO:     5    1041    1046    1153    1069    1053    1237    
INFO:     10    1237    1231    1352    1256    1246    1438    
INFO:     15    1414    1408    1533    1433    1425    1630    
INFO:     20    1563    1563    1701    1592    1573    1821    
INFO:     25    1696    1693    1861    1723    1707    1990    
INFO:     30    1813    1811    1992    1844    1821    2127    
INFO:     35    1922    1921    2110    1953    1929    2248    
INFO:     40    2025    2024    2217    2056    2032    2345    
INFO:     45    2121    2122    2320    2152    2128    2433    
INFO:     50    2215    2214    2414    2243    2221    2516    
INFO:     55    2307    2306    2502    2332    2313    2597    
INFO:     60    2397    2397    2583    2419    2402    2681    
INFO:     65    2486    2486    2676    2508    2491    2771    
INFO:     70    2574    2579    2766    2602    2586    2850    
INFO:     75    2666    2670    2849    2693    2677    2929    
INFO:     80    2766    2768    2938    2793    2775    3020    
INFO:     85    2873    2875    3034    2899    2880    3125    
INFO:     90    2997    2998    3159    3020    3002    3241    
INFO:     95    3166    3166    3326    3190    3173    3421    
INFO:     96    3217    3211    3382    3238    3223    3471    
INFO:     97    3273    3271    3452    3295    3281    3537    
INFO:     98    3359    3356    3542    3363    3368    3636    
INFO:     99    3526    3520    3710    3536    3534    3884    
INFO:     100    7428    7374    6285    6956    7400    7330    
INFO:     Throughput: 65.1721K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 80
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    1163    1164    1287    1179    1164    1321    
INFO:     5    1515    1516    1655    1524    1519    1733    
INFO:     10    1860    1852    1971    1874    1852    2066    
INFO:     15    2138    2133    2243    2150    2140    2334    
INFO:     20    2386    2379    2503    2406    2386    2601    
INFO:     25    2607    2599    2727    2632    2607    2870    
INFO:     30    2792    2784    2940    2813    2795    3065    
INFO:     35    2958    2952    3116    2986    2964    3247    
INFO:     40    3110    3109    3280    3132    3115    3423    
INFO:     45    3255    3255    3419    3274    3258    3559    
INFO:     50    3388    3390    3545    3410    3393    3678    
INFO:     55    3514    3514    3664    3536    3518    3795    
INFO:     60    3636    3634    3780    3656    3637    3902    
INFO:     65    3753    3749    3892    3773    3754    4002    
INFO:     70    3868    3866    3998    3887    3871    4105    
INFO:     75    3987    3984    4114    4002    3993    4217    
INFO:     80    4114    4113    4242    4135    4121    4343    
INFO:     85    4262    4262    4388    4287    4269    4486    
INFO:     90    4440    4435    4579    4464    4446    4667    
INFO:     95    4707    4703    4860    4727    4707    4910    
INFO:     96    4776    4776    4939    4792    4782    4981    
INFO:     97    4860    4859    5014    4873    4866    5059    
INFO:     98    4970    4974    5137    4989    4981    5156    
INFO:     99    5143    5153    5313    5164    5153    5299    
INFO:     100    7528    7506    7433    7356    7565    7322    
INFO:     Throughput: 67.0173K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 100
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    1397    1410    1505    1438    1401    1610    
INFO:     5    1806    1823    1969    1845    1810    2036    
INFO:     10    2210    2227    2374    2237    2213    2395    
INFO:     15    2549    2556    2727    2573    2547    2715    
INFO:     20    2871    2876    3064    2897    2873    3020    
INFO:     25    3173    3176    3376    3182    3177    3378    
INFO:     30    3427    3435    3646    3445    3434    3669    
INFO:     35    3656    3661    3898    3683    3661    3947    
INFO:     40    3863    3872    4112    3887    3870    4184    
INFO:     45    4050    4064    4297    4078    4062    4366    
INFO:     50    4221    4234    4443    4246    4233    4524    
INFO:     55    4385    4397    4594    4412    4396    4693    
INFO:     60    4538    4551    4731    4563    4547    4833    
INFO:     65    4689    4701    4878    4718    4700    4968    
INFO:     70    4839    4848    5014    4862    4847    5103    
INFO:     75    4986    4994    5150    5009    4991    5234    
INFO:     80    5135    5148    5301    5160    5144    5400    
INFO:     85    5306    5318    5462    5319    5314    5553    
INFO:     90    5513    5521    5675    5529    5521    5762    
INFO:     95    5801    5810    5968    5829    5812    6069    
INFO:     96    5882    5895    6056    5920    5895    6177    
INFO:     97    5980    5997    6166    6028    6002    6267    
INFO:     98    6119    6133    6301    6169    6148    6383    
INFO:     99    6329    6334    6490    6366    6355    6627    
INFO:     100    9550    9638    8514    8478    9563    8455    
INFO:     Throughput: 67.2771K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 200
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    2925    2928    3069    2957    2946    3152    
INFO:     5    3671    3694    3855    3714    3689    3952    
INFO:     10    4384    4383    4491    4402    4383    4625    
INFO:     15    4946    4954    5065    4999    4949    5187    
INFO:     20    5580    5572    5737    5622    5578    5823    
INFO:     25    6114    6113    6264    6157    6121    6346    
INFO:     30    6547    6560    6743    6596    6553    6822    
INFO:     35    6944    6954    7141    7011    6951    7267    
INFO:     40    7341    7349    7563    7429    7342    7717    
INFO:     45    7717    7728    7940    7814    7725    8106    
INFO:     50    8066    8081    8295    8170    8082    8472    
INFO:     55    8395    8413    8615    8490    8417    8767    
INFO:     60    8671    8690    8873    8757    8695    9012    
INFO:     65    8927    8946    9112    9003    8943    9252    
INFO:     70    9172    9186    9339    9233    9181    9466    
INFO:     75    9410    9423    9578    9464    9416    9683    
INFO:     80    9657    9671    9827    9707    9667    9906    
INFO:     85    9936    9950    10103    9973    9942    10181    
INFO:     90    10289    10286    10454    10332    10295    10523    
INFO:     95    10835    10824    11010    10886    10835    11039    
INFO:     96    10988    10978    11144    11033    10992    11164    
INFO:     97    11164    11162    11320    11225    11179    11383    
INFO:     98    11395    11391    11524    11437    11412    11577    
INFO:     99    11791    11775    11934    11850    11810    12002    
INFO:     100    17074    17111    16811    17082    17011    16475    
INFO:     Throughput: 68.0728K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 500
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    7762    7803    7898    7836    7751    7880    
INFO:     5    9910    9971    10071    10028    9956    10116    
INFO:     10    11421    11476    11532    11514    11530    11577    
INFO:     15    12819    12894    12975    12929    12953    13071    
INFO:     20    13926    13960    14109    14007    14021    14186    
INFO:     25    14933    14943    15106    15059    14996    15191    
INFO:     30    15769    15765    15975    15862    15795    16049    
INFO:     35    16556    16546    16740    16637    16583    16845    
INFO:     40    17303    17280    17540    17395    17320    17615    
INFO:     45    18046    18035    18297    18118    18071    18363    
INFO:     50    18770    18781    18997    18863    18802    19110    
INFO:     55    19544    19576    19751    19604    19560    19853    
INFO:     60    20362    20397    20539    20412    20367    20649    
INFO:     65    21143    21181    21300    21189    21148    21391    
INFO:     70    21839    21857    22007    21882    21858    22122    
INFO:     75    22440    22451    22558    22535    22440    22695    
INFO:     80    22978    23001    23139    23077    22973    23230    
INFO:     85    23543    23562    23735    23636    23548    23787    
INFO:     90    24230    24252    24410    24318    24237    24417    
INFO:     95    25186    25210    25403    25241    25191    25382    
INFO:     96    25439    25471    25725    25532    25453    25684    
INFO:     97    25817    25849    26117    25915    25847    26013    
INFO:     98    26330    26348    26545    26376    26338    26601    
INFO:     99    27023    27063    27302    27150    27039    27397    
INFO:     100    30173    30118    30132    30165    30079    29955    
INFO:     Throughput: 68.8246K queries/sec
wukong> 
wukong> sparql-emu -f query/lubm/emulator/mix_config -d 5 -w 1 -p 1000
INFO:     Per-query CDF graph
INFO:     CDF Res: 
INFO:     P    Q1    Q2    Q3    Q4    Q5    Q6
INFO:     1    12706    12899    13478    12787    12783    13841    
INFO:     5    17308    17369    17931    17325    17390    17404    
INFO:     10    21651    21734    22197    21656    21720    21790    
INFO:     15    24759    24833    25035    24685    24891    24964    
INFO:     20    27703    27828    27980    27498    27867    27861    
INFO:     25    30006    30136    30096    29914    30103    30247    
INFO:     30    31784    31911    31888    31763    31884    32102    
INFO:     35    33362    33502    33634    33461    33468    33769    
INFO:     40    34746    34855    34929    34839    34809    35143    
INFO:     45    36068    36184    36184    36264    36158    36508    
INFO:     50    37418    37500    37501    37583    37486    37785    
INFO:     55    38629    38730    38782    38819    38733    39076    
INFO:     60    39895    40002    40050    40080    40007    40403    
INFO:     65    41138    41270    41322    41328    41258    41645    
INFO:     70    42344    42434    42495    42491    42445    42723    
INFO:     75    43437    43519    43639    43595    43534    43839    
INFO:     80    44555    44613    44745    44667    44615    44846    
INFO:     85    45553    45650    45829    45668    45673    45965    
INFO:     90    46757    46835    46978    46866    46865    47158    
INFO:     95    48501    48527    48708    48562    48552    48873    
INFO:     96    48964    48969    49157    48999    49028    49369    
INFO:     97    49586    49587    49818    49538    49634    49884    
INFO:     98    50318    50276    50599    50227    50327    50648    
INFO:     99    51962    51803    52322    51766    51834    52395    
INFO:     100    58420    58490    58231    58121    58436    58438    
INFO:     Throughput: 68.9566K queries/sec
wukong> q 
```
