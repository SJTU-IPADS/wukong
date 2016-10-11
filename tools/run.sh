#!/bin/sh
#
# Configure wukong at the file 'config'
# List all nodes at the file 'mpd.host'
# Share the input RDF data among all nodes through DFS (e.g., NFS or HDFS)

# Standalone Mode (w/o mpi):
# ../build/wukong config

# Distributed Mode (w/ mpi):
# NOTE: the hostfile of of mpiexec must match that of wukong (i.e., mpd.hosts)
/usr/bin/mpiexec -x CLASSPATH -x LD_LIBRARY_PATH -hostfile mpd.hosts -n $1 ../build/wukong config mpd.hosts
