#!/bin/bash
# $1 machine number
# $2 config_file
if [ -d $folderpath ]; then
	/usr/bin/mpiexec -x LD_LIBRARY_PATH -hostfile mpd.hosts -n $1 ../graph-store-distributed/test_graph_distributed.out $2
else
	echo  $folderpath "not exist"
fi

