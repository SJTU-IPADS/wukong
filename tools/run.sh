#!/bin/sh

# NOTE: the hostfile of of mpiexec must match that of wukong (i.e., mpd.hosts)
/usr/bin/mpiexec -x LD_LIBRARY_PATH -hostfile mpd.hosts -n $1 ../build/wukong config mpd.hosts