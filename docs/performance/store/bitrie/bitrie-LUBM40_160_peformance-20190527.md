# BiTrie Performance

- Date: May. 27, 2019
- Author: Yaozeng Zeng

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
CPU(s):                40
On-line CPU(s) list:   0-39
Thread(s) per core:    2
Core(s) per socket:    10
Socket(s):             2
NUMA node(s):          2
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 63
Model name:            Intel(R) Xeon(R) CPU E5-2650 v3 @ 2.30GHz
Stepping:              2
CPU MHz:               1200.000
CPU max MHz:           2301.0000
CPU min MHz:           1200.0000
BogoMIPS:              4600.65
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              25600K
NUMA node0 CPU(s):     0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38
NUMA node1 CPU(s):     1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39
Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm epb tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm xsaveopt cqm_llc cqm_occup_llc dtherm ida arat pln pts
```

<a name="sw"></a>

## Software Configuration

### Wukong Version

- Version 1: 2ec4de (port BiTrie into StringServer)

### Config

```
bucket number   59
associativity   8
```

<a name="res"></a>

## Experiment Result

Baseline is boost::unordered_map.

### Version 1: 2ec4de (port BiTrie into StringServer)

Get memory information from mallinfo(diff between uordblks and hblkhd).

Dataset | memory | baseline memory | init time | baseline init time | STR2ID time | STR2ID compared w/ baseline | ID2STR time | ID2STR compared w/ baseline |
 :------: | ----: |-------: | ---: | ----: | :----- | :-: | -----: |----: |
lubm_40   | 121mb |  377mb  | 4.6s |  7.7s | 0.54us | 55% | 0.34us | 147% |
lubm_160  | 477mb | 1524mb  | 18s  | 32.2s | 0.64us | 64% | 0.37us | 155% |
