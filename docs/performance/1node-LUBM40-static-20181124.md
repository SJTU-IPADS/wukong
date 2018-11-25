# Performance (single node, static gstore)

###### Date: Nov. 24, 2018

###### Author: Siyuan Wang

## Table of Contents

* [Hardware configuration](#hw)
* [Software configuration](#sw)
* [Dataset and workload](#dw)
* [Experimantal results (RDMA-enabled)](#res)

<br>
<a name="hw"></a>
## Hardware configuration
#### CPU
| N   | S x C x T  | Processor                        |
| :-: | :--------: | :------------------------------- |
| 1   | 2 x 12 x 2 | Intel Xeon E5-2650 v4 processors |

#### NIC
| N x P | Bandwidth | NIC                                        |
| :---: | :-------: | :----------------------------------------- |
| 1 x 2 | 56Gbps    | ConnectX-3 MCX353A IB NICs via PCIe 3.0 x8 |

#### Switch
| N x P | Bandwidth | Switch                           |
| :---: | :-------: | :------------------------------- |
| 1 x / | 40Gbps    | Mellanox IS5025 IB Switch        |


<br>
<a name="sw"></a>
## Software configuration

##### Gitlab Version: @8191323

#### Configuration

```bash
$cd $WUKONG_ROOT/scripts
$cat config
#general
global_num_proxies              2
global_num_engines              16
global_input_folder             /mnt/nfs/rdfdata/id_lubm_40/
global_data_port_base           5500
global_ctrl_port_base           9576
global_memstore_size_gb         10
global_mt_threshold             16
global_enable_workstealing      0
global_stealing_pattern         0
global_enable_planner           0
global_generate_statistics      0
global_enable_vattr             0
global_silent                   1

# RDMA
global_rdma_buf_size_mb         128
global_rdma_rbf_size_mb         64
global_use_rdma                 1
global_rdma_threshold           300
global_enable_caching           0

# GPU
global_num_gpus                 1
global_gpu_rdma_buf_size_mb     64
global_gpu_rbuf_size_mb         100
global_gpu_kvcache_size_gb      10
global_gpu_key_blk_size_mb      16
global_gpu_value_blk_size_mb    8
global_gpu_enable_pipeline      1
$
$cat core.bind
# One node per line (NOTE: the empty line means to skip a node)
0 2 4 6 8 10 12 14 16 18 20 22
1 3 5 7 9 11 13 15 17 19 21 23
```

#### Building and running command

### Wukong
```bash
$./build.sh -DUSE_RDMA=ON -DUSE_GPU=OFF -DUSE_HADOOP=OFF -DUSE_JEMALLOC=OFF -DUSE_DYNAMIC_GSTORE=OFF -DUSE_VERSATILE=OFF -DUSE_DTYPE_64BIT=OFF
$./run.sh 1
```
### Wukong-GPU
```bash
$./build.sh -DUSE_RDMA=ON -DUSE_GPU=ON -DUSE_HADOOP=OFF -DUSE_JEMALLOC=OFF -DUSE_DYNAMIC_GSTORE=OFF -DUSE_VERSATILE=OFF -DUSE_DTYPE_64BIT=OFF
$./run.sh 1
```

<br>
<a name="dw"></a>
## Dataset and workload

**Dataset**: Leigh University Benchmark with 40 University (**LUBM-40**)

**Queries**: `sparql_query/lubm/basic/lubm_{q1,q2,q3,q4,q5,q6,q7}`, `sparql_query/lubm/emulator/mix_config_heavy`
**Plans**: `sparql_query/lubm/basic/osdi16_plan/lubm_{q1,q2,q3,q4,q5,q6,q7}.fmt`


<br>
<a name="res"></a>
## Experimantal results of CPU-only Wukong (RDMA-enabled)
#### Summary

| Workload | Latency (us) | #Results (lines)  | TH | File               |
| :------: | -----------: | ----------------: | -: | :----------------- |
| Q1       | 4,317        | 106               | 16 | query/lubm/lubm_q1 |
| Q2       | 578          | 43,291            | 16 | query/lubm/lubm_q2 |
| Q3       | 1,092        | 0                 | 16 | query/lubm/lubm_q3 |
| Q4       | 36           | 10                |  1 | query/lubm/lubm_q4 |
| Q5       | 29           | 10                |  1 | query/lubm/lubm_q5 |
| Q6       | 78           | 125               |  1 | query/lubm/lubm_q6 |
| Q7       | 2,399        | 1,763             | 16 | query/lubm/lubm_q7 |

| Workload | Thpt (q/s) | Configuration   | File                           |
| :------: | ---------: | :-------------- | :----------------------------- |
| Q1-Q3,Q7 | 352.4      | -d 10 -w 5 -p 1 | query/lubm/emulator/mix_config_heavy |
| Q1-Q3,Q7 | 395.8      | -d 10 -w 5 -p 5  | query/lubm/emulator/mix_config_heavy |
| Q1-Q3,Q7 | 394.0      | -d 10 -w 5 -p 10 | query/lubm/emulator/mix_config_heavy |
| Q1-Q3,Q7 | 394.4      | -d 10 -w 5 -p 15  | query/lubm/emulator/mix_config_heavy |
| Q1-Q3,Q7 | 403.2      | -d 10 -w 5 -p 20  | query/lubm/emulator/mix_config_heavy |

#### Detail

```bash
wukong> sparql -f sparql_query/lubm/basic/lubm_q1 -p sparql_query/lubm/basic/osdi16_plan/lubm_q1.fmt -m 16 -n 50
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 126 usec
INFO:     User-defined query plan is enabled
INFO:     (last) result size: 106
INFO:     (average) latency: 5674 usec
wukong>
wukong> sparql -f sparql_query/lubm/basic/lubm_q2 -p sparql_query/lubm/basic/osdi16_plan/lubm_q2.fmt -m 16 -n 50
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 112 usec
INFO:     User-defined query plan is enabled
INFO:     (last) result size: 43291
INFO:     (average) latency: 1068 usec
wukong>
wukong> sparql -f sparql_query/lubm/basic/lubm_q3 -p sparql_query/lubm/basic/osdi16_plan/lubm_q3.fmt -m 16 -n 50
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 169 usec
INFO:     User-defined query plan is enabled
INFO:     (last) result size: 0
INFO:     (average) latency: 1433 usec
wukong>
wukong> sparql -f sparql_query/lubm/basic/lubm_q4 -p sparql_query/lubm/basic/osdi16_plan/lubm_q4.fmt -n 20000
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 148 usec
INFO:     User-defined query plan is enabled
INFO:     (last) result size: 10
INFO:     (average) latency: 38 usec
wukong>
wukong> sparql -f sparql_query/lubm/basic/lubm_q5 -p sparql_query/lubm/basic/osdi16_plan/lubm_q5.fmt -n 20000
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 115 usec
INFO:     User-defined query plan is enabled
INFO:     (last) result size: 10
INFO:     (average) latency: 30 usec
wukong>
wukong> sparql -f sparql_query/lubm/basic/lubm_q6 -p sparql_query/lubm/basic/osdi16_plan/lubm_q6.fmt -n 20000
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 134 usec
INFO:     User-defined query plan is enabled
INFO:     (last) result size: 125
INFO:     (average) latency: 79 usec
wukong>
wukong> sparql -f sparql_query/lubm/basic/lubm_q7 -p sparql_query/lubm/basic/osdi16_plan/lubm_q7.fmt -m 16 -n 50
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 120 usec
INFO:     User-defined query plan is enabled
INFO:     (last) result size: 1763
INFO:     (average) latency: 4317 usec
wukong>
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config_heavy -p sparql_query/lubm/emulator/plan_config_heavy -d 10 -w 5 -n 1
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4
INFO:     1	4525	749	1292	2446
INFO:     5	5354	961	1960	3323
INFO:     10	5618	1194	2288	3668
INFO:     15	5844	1400	2514	3908
INFO:     20	6111	1669	2735	4139
INFO:     25	6391	1927	2940	4423
INFO:     30	6621	2286	3124	4674
INFO:     35	6886	2694	3357	4872
INFO:     40	7187	3006	3588	5144
INFO:     45	7392	3200	3941	5355
INFO:     50	7658	3496	4199	5562
INFO:     55	7926	3870	4491	5890
INFO:     60	8243	4137	4825	6306
INFO:     65	8485	4465	5155	6650
INFO:     70	8792	4958	5646	7133
INFO:     75	9178	5603	6037	7525
INFO:     80	9650	6038	6518	8007
INFO:     85	10088	6456	7136	8633
INFO:     90	10718	7081	7659	9331
INFO:     95	11911	8435	9018	10630
INFO:     96	12265	8663	9663	10852
INFO:     97	12662	8875	10287	11275
INFO:     98	13111	9518	10939	12264
INFO:     99	14002	10912	12016	13294
INFO:     100	17895	14575	13018	14821
INFO:     Throughput: 0.3524K queries/sec
wukong>
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config_heavy -p sparql_query/lubm/emulator/plan_config_heavy -d 10 -w 5 -n 5
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4
INFO:     1	14551	8701	10712	11192
INFO:     5	17033	12617	13287	13954
INFO:     10	18481	13913	14980	15611
INFO:     15	19566	14817	16281	16967
INFO:     20	20457	15842	17079	18090
INFO:     25	21189	16577	17853	19087
INFO:     30	22161	17561	18589	20041
INFO:     35	23074	18391	19640	20911
INFO:     40	24096	19441	20626	21746
INFO:     45	25005	20434	21388	22799
INFO:     50	26187	21696	22571	23860
INFO:     55	27238	22619	23617	24780
INFO:     60	28224	23516	24659	25641
INFO:     65	29718	24877	25978	26775
INFO:     70	31106	25953	27061	28349
INFO:     75	32824	27428	28460	29922
INFO:     80	35337	29443	29954	32070
INFO:     85	38787	32012	32119	34601
INFO:     90	42567	36571	36112	38291
INFO:     95	47760	43113	41257	44223
INFO:     96	49079	43937	43019	45440
INFO:     97	51699	45601	45115	46564
INFO:     98	53532	47912	48444	48743
INFO:     99	56846	51846	54285	51338
INFO:     100	65646	58133	61145	73824
INFO:     Throughput: 0.3958K queries/sec
wukong>
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config_heavy -p sparql_query/lubm/emulator/plan_config_heavy -d 10 -w 5 -n 10
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4
INFO:     1	31441	27068	28252	29513
INFO:     5	34798	30605	32533	33234
INFO:     10	37662	33580	34777	35829
INFO:     15	39477	35346	36272	37641
INFO:     20	41089	37134	37940	39114
INFO:     25	42403	38382	39423	40508
INFO:     30	43770	39430	40713	42224
INFO:     35	45052	40668	41765	43184
INFO:     40	46304	42160	43182	44660
INFO:     45	47708	43340	44679	46193
INFO:     50	48905	44532	45849	47283
INFO:     55	50067	45567	47584	48570
INFO:     60	51578	46999	48977	49913
INFO:     65	53372	48731	50884	51713
INFO:     70	55250	50630	52753	53521
INFO:     75	57635	52875	55183	56245
INFO:     80	60848	56141	57387	59849
INFO:     85	66073	61009	62225	64372
INFO:     90	74445	66929	69906	72219
INFO:     95	93295	83921	94006	93205
INFO:     96	99661	93708	100240	100467
INFO:     97	105896	101175	105820	105843
INFO:     98	113501	110341	117453	113261
INFO:     99	121172	122053	121322	121530
INFO:     100	146673	145744	134670	145864
INFO:     Throughput: 0.394K queries/sec
wukong>
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config_heavy -p sparql_query/lubm/emulator/plan_config_heavy -d 10 -w 5 -n 15
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4
INFO:     1	43875	37937	39652	42054
INFO:     5	50181	46375	46832	49268
INFO:     10	54042	49383	50857	52732
INFO:     15	56494	52166	52994	55107
INFO:     20	58492	54294	55007	56936
INFO:     25	60664	57092	56894	58518
INFO:     30	62052	58605	58625	60104
INFO:     35	63335	60251	60292	61285
INFO:     40	65093	61584	61937	63402
INFO:     45	66616	63510	63916	65221
INFO:     50	68679	65502	66373	67497
INFO:     55	71033	67955	68301	69557
INFO:     60	74055	70493	71250	72378
INFO:     65	77016	73782	74457	76007
INFO:     70	81148	76498	78039	81122
INFO:     75	86404	82456	83482	85926
INFO:     80	96053	87767	90573	92699
INFO:     85	107333	96914	99486	102337
INFO:     90	120017	109325	110291	117101
INFO:     95	147279	137036	135489	149249
INFO:     96	156292	146831	145859	154727
INFO:     97	162849	152705	155177	159301
INFO:     98	170761	162731	165442	165693
INFO:     99	178417	179169	172360	177990
INFO:     100	185121	197350	199574	204654
INFO:     Throughput: 0.3944K queries/sec
wukong>
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config_heavy -p sparql_query/lubm/emulator/plan_config_heavy -d 10 -w 5 -n 20
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4
INFO:     1	58466	53898	58862	60133
INFO:     5	66546	64264	65651	66671
INFO:     10	70136	67663	68728	69382
INFO:     15	73321	70555	71013	71975
INFO:     20	75700	72659	73642	74561
INFO:     25	77566	74144	75255	76354
INFO:     30	79557	75501	77121	77734
INFO:     35	81394	77298	79087	79645
INFO:     40	83033	79301	81046	81939
INFO:     45	85310	80990	83478	84297
INFO:     50	87264	82989	85644	86609
INFO:     55	90390	85966	88078	89109
INFO:     60	93381	88660	91760	91692
INFO:     65	97913	91695	96081	97093
INFO:     70	104515	96945	101468	102360
INFO:     75	114409	102451	109925	112387
INFO:     80	124619	112425	127838	122485
INFO:     85	141262	126032	143889	135662
INFO:     90	169532	150591	171793	155806
INFO:     95	206771	193016	199769	191805
INFO:     96	209261	200202	203768	203512
INFO:     97	213891	205298	207737	209864
INFO:     98	218275	212905	212354	215191
INFO:     99	225578	221873	220975	224024
INFO:     100	245369	251469	240815	245238
INFO:     Throughput: 0.4032K queries/sec
```

<br>
<a name="res2"></a>
## Experimantal results of Wukong-GPU (RDMA-enabled)

#### Summary

| Workload | Latency (us) | #Results (lines)  | TH | File               |
| :------: | -----------: | ----------------: | -: | :----------------- |
| Q1       | 3,509        | 106               | 1 | query/lubm/lubm_q1 |
| Q2       | 3,473        | 43,291            | 1 | query/lubm/lubm_q2 |
| Q3       | 2,094        | 0                 | 1 | query/lubm/lubm_q3 |
| Q4       | 42           | 10                | 1 | query/lubm/lubm_q4 |
| Q5       | 32           | 10                | 1 | query/lubm/lubm_q5 |
| Q6       | 98           | 125               | 1 | query/lubm/lubm_q6 |
| Q7       | 2,317        | 1,763             | 1 | query/lubm/lubm_q7 |

| Workload | Thpt (q/s) | Configuration   | File                                 |
| :------: | ---------: | :-------------- | :----------------------------------- |
| Q1-Q3,Q7 | 399.4      | -d 10 -w 5 -p 1  | query/lubm/emulator/mix_config_heavy |
| Q1-Q3,Q7 | 401.0      | -d 10 -w 5 -p 5  | query/lubm/emulator/mix_config_heavy |
| Q1-Q3,Q7 | 399.3      | -d 10 -w 5 -p 10 | query/lubm/emulator/mix_config_heavy |
| Q1-Q3,Q7 | 398.4      | -d 10 -w 5 -p 15 | query/lubm/emulator/mix_config_heavy |
| Q1-Q3,Q7 | 395.4      | -d 10 -w 5 -p 20 | query/lubm/emulator/mix_config_heavy |

#### Detail

```bash
wukong> sparql -f sparql_query/lubm/basic/lubm_q1 -p sparql_query/lubm/basic/osdi16_plan/lubm_q1.fmt -g -n 50
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 102 usec
INFO:     User-defined query plan is enabled
INFO:     Leverage GPU to accelerate query processing.
INFO:     (last) result size: 106
INFO:     (average) latency: 3509 usec
wukong>
wukong> sparql -f sparql_query/lubm/basic/lubm_q2 -p sparql_query/lubm/basic/osdi16_plan/lubm_q2.fmt -g -n 50
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 89 usec
INFO:     User-defined query plan is enabled
INFO:     Leverage GPU to accelerate query processing.
INFO:     (last) result size: 43291
INFO:     (average) latency: 3473 usec
wukong>
wukong> sparql -f sparql_query/lubm/basic/lubm_q3 -p sparql_query/lubm/basic/osdi16_plan/lubm_q3.fmt -g -n 50
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 201 usec
INFO:     User-defined query plan is enabled
INFO:     Leverage GPU to accelerate query processing.
INFO:     (last) result size: 0
INFO:     (average) latency: 2094 usec
wukong>
wukong> sparql -f sparql_query/lubm/basic/lubm_q4 -p sparql_query/lubm/basic/osdi16_plan/lubm_q4.fmt -n 20000
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 171 usec
INFO:     User-defined query plan is enabled
INFO:     (last) result size: 10
INFO:     (average) latency: 42 usec
wukong>
wukong> sparql -f sparql_query/lubm/basic/lubm_q5 -p sparql_query/lubm/basic/osdi16_plan/lubm_q5.fmt -n 20000
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 103 usec
INFO:     User-defined query plan is enabled
INFO:     (last) result size: 10
INFO:     (average) latency: 32 usec
wukong>
wukong> sparql -f sparql_query/lubm/basic/lubm_q6 -p sparql_query/lubm/basic/osdi16_plan/lubm_q6.fmt -n 20000
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 145 usec
INFO:     User-defined query plan is enabled
INFO:     (last) result size: 125
INFO:     (average) latency: 98 usec
wukong>
wukong> sparql -f sparql_query/lubm/basic/lubm_q7 -p sparql_query/lubm/basic/osdi16_plan/lubm_q7.fmt -g -n 50
INFO:     Parsing a SPARQL query is done.
INFO:     Parsing time: 185 usec
INFO:     User-defined query plan is enabled
INFO:     Leverage GPU to accelerate query processing.
INFO:     (last) result size: 1763
INFO:     (average) latency: 2317 usec
wukong>
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config_heavy -p sparql_query/lubm/emulator/plan_config_heavy -d 10 -w 5 -n 1
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4
INFO:     1	4551	4202	3119	3330
INFO:     5	4586	4334	3149	3384
INFO:     10	4728	4447	3313	3445
INFO:     15	4771	4506	3363	3514
INFO:     20	5529	5180	4112	4332
INFO:     25	5561	5244	4131	4352
INFO:     30	5572	5300	4141	4365
INFO:     35	5584	5340	4156	4386
INFO:     40	5688	5373	4295	4407
INFO:     45	5726	5410	4314	4422
INFO:     50	5741	5441	4325	4437
INFO:     55	5751	5468	4333	4465
INFO:     60	5759	5497	4344	4511
INFO:     65	5769	5517	4353	4526
INFO:     70	5784	5539	4372	4544
INFO:     75	5808	5584	4540	4699
INFO:     80	5992	5738	4612	4825
INFO:     85	6982	6643	5558	5757
INFO:     90	6995	6699	5570	5768
INFO:     95	7006	6768	5581	5782
INFO:     96	7009	6781	5586	5785
INFO:     97	7011	6792	5590	5789
INFO:     98	7014	6802	5593	5794
INFO:     99	7020	6816	5598	5801
INFO:     100	7034	6838	5624	5835
INFO:     Throughput: 0.3994K queries/sec
wukong>
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config_heavy -p sparql_query/lubm/emulator/plan_config_heavy -d 10 -w 5 -n 5
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4
INFO:     1	21998	22049	20626	20881
INFO:     5	23051	22793	21630	21732
INFO:     10	23445	23244	21922	22083
INFO:     15	24148	23920	22571	22831
INFO:     20	24412	24113	22891	23083
INFO:     25	24558	24291	23070	23248
INFO:     30	24698	24477	23221	23397
INFO:     35	24910	24671	23384	23600
INFO:     40	25497	25217	23930	24234
INFO:     45	25662	25403	24221	24386
INFO:     50	25755	25549	24334	24542
INFO:     55	25866	25659	24470	24626
INFO:     60	25998	25809	24583	24717
INFO:     65	26144	26000	25020	24976
INFO:     70	26792	26596	25526	25631
INFO:     75	27017	26752	25699	25811
INFO:     80	27177	26909	25791	25938
INFO:     85	27361	27133	26081	26177
INFO:     90	28281	28025	26896	27030
INFO:     95	28701	28382	27208	27347
INFO:     96	29322	28495	27361	27589
INFO:     97	29576	29128	28129	28340
INFO:     98	29788	29352	28337	28503
INFO:     99	30036	29718	28784	28671
INFO:     100	31177	32009	31010	29787
INFO:     Throughput: 0.400998K queries/sec
wukong>
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config_heavy -p sparql_query/lubm/emulator/plan_config_heavy -d 10 -w 5 -n 10
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4
INFO:     1	    45295	44497	42839	43572
INFO:     5	    46940	46520	45103	45870
INFO:     10	47700	47428	46216	46549
INFO:     15	48310	47895	46824	47150
INFO:     20	48781	48515	47403	47606
INFO:     25	48982	48785	47715	47886
INFO:     30	49220	49171	47999	48262
INFO:     35	49756	49671	48560	48773
INFO:     40	50149	49956	48816	49029
INFO:     45	50335	50185	49040	49218
INFO:     50	50562	50526	49261	49450
INFO:     55	51007	51031	49732	49973
INFO:     60	51333	51283	50053	50212
INFO:     65	51543	51504	50250	50417
INFO:     70	51788	52017	50479	50574
INFO:     75	52435	52473	50979	51189
INFO:     80	52799	52771	51323	51571
INFO:     85	53673	53475	51694	51798
INFO:     90	54299	54075	52641	52747
INFO:     95	55465	55158	53832	54068
INFO:     96	55575	55264	54085	54290
INFO:     97	56463	55465	54199	54371
INFO:     98	56794	56072	55057	54921
INFO:     99	57061	56640	55506	55633
INFO:     100	59401	59030	57838	57994
INFO:     Throughput: 0.399378K queries/sec
wukong>
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config_heavy -p sparql_query/lubm/emulator/plan_config_heavy -d 10 -w 5 -n 15
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4
INFO:     1	    68386	68559	67553	67175
INFO:     5	    70699	70445	69643	69807
INFO:     10	71958	71444	70662	70865
INFO:     15	72718	72119	71273	71774
INFO:     20	73278	72915	71816	72198
INFO:     25	73655	73218	72248	72625
INFO:     30	74158	73718	72853	73089
INFO:     35	74554	74194	73245	73422
INFO:     40	74862	74483	73706	73782
INFO:     45	75254	74795	74151	74291
INFO:     50	75732	75293	74507	74744
INFO:     55	76066	75718	74771	75017
INFO:     60	76411	75985	75228	75485
INFO:     65	77040	76562	75739	75942
INFO:     70	77326	76924	76041	76297
INFO:     75	77678	77283	76425	76863
INFO:     80	78313	77865	77021	77290
INFO:     85	78600	78384	77403	77705
INFO:     90	79628	79144	78278	78553
INFO:     95	80819	79969	79487	79735
INFO:     96	81059	80578	79860	79960
INFO:     97	81378	80974	80289	80426
INFO:     98	82239	81962	80958	81037
INFO:     99	83223	82724	81849	82041
INFO:     100	84599	85721	83318	83016
INFO:     Throughput: 0.3984K queries/sec
wukong>
wukong> sparql-emu -f sparql_query/lubm/emulator/mix_config_heavy -p sparql_query/lubm/emulator/plan_config_heavy -d 10 -w 5 -n 20
INFO:     Per-query CDF graph
INFO:     CDF Res:
INFO:     P	Q1	Q2	Q3	Q4
INFO:     1	    91410	61766	90117	90055
INFO:     5	    94938	94496	93667	93555
INFO:     10	96316	95984	95070	94891
INFO:     15	97357	96792	95966	95915
INFO:     20	98258	97641	96618	96624
INFO:     25	99100	98531	97518	97318
INFO:     30	99730	99253	98141	97998
INFO:     35	100347	99982	98850	98555
INFO:     40	100767	100339	99289	99184
INFO:     45	101413	100945	99880	99594
INFO:     50	101840	101420	100393	100195
INFO:     55	102215	101851	100791	100715
INFO:     60	102733	102446	101424	101373
INFO:     65	103113	102849	101933	101722
INFO:     70	103671	103375	102545	102204
INFO:     75	104305	104031	102952	102850
INFO:     80	104934	104809	103441	103441
INFO:     85	105671	105518	104237	104322
INFO:     90	106463	106459	105394	105329
INFO:     95	107529	107474	106827	106301
INFO:     96	107746	107863	107027	106797
INFO:     97	108249	108051	107349	107020
INFO:     98	108790	108530	107926	107527
INFO:     99	109309	109368	108616	108314
INFO:     100	112618	113362	110142	111024
INFO:     Throughput: 0.3954K queries/sec



