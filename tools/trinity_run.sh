#!/bin/bash
# $1 machine number
# $2 config_file
if [ -d $folderpath ]; then
	/usr/bin/mpiexec -hostfile mpd.hosts -n $1 ../graph-store-distributed/trinity_simulate_test.out $2
else
	echo  $folderpath "not exist"
fi

