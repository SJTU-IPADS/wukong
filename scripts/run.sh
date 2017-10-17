#!/bin/sh
#
# 1. List all nodes at the file 'mpd.host'
# 2. Share the input RDF data among all nodes through DFS (e.g., NFS or HDFS)
# 3. Configure wukong at the file 'config'
# 4. Edit the file 'core.bind' to control the thread binding.

# NOTE: the hostfile of mpiexec must match that of wukong (i.e., mpd.hosts)
${WUKONG_ROOT}/deps/openmpi-1.6.5-install/bin/mpiexec -x CLASSPATH -x LD_LIBRARY_PATH -hostfile mpd.hosts -n $1 ../build/wukong config mpd.hosts -b core.bind
