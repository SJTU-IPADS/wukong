#!/bin/bash

${WUKONG_ROOT}/deps/openmpi-1.6.5-install/bin/mpiexec -x CLASSPATH -x LD_LIBRARY_PATH -hostfile mpd.hosts -n 2 ./build/network mpd.hosts $@ 
