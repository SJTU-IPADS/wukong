# Evaluate  latency of network

### Introduction
This is a micro-benchmark to evaluate the network latency between two servers.

it will calculate the time of one round-trip  between two servers.
In `mpd.hosts` file, you can config the ip address of servers.  It only support two servers now .like this:

```
$cat mpd.hosts
10.0.0.100
10.0.0.103
```

### Usage
* build  and sync the file

```
$./build.sh
$./sync.sh
```
* run the function

```
$./run.sh -n 100 -s 1000
```

command like this.

```
$./run.sh -h
network test::
  -h [ --help ]                help message about network test
  -n [ --num ] <num> (=100)    run <num> times
  -s [ --size ] <size> (=1000) set sending message <size>
```


