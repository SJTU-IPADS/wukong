#!/bin/bash
# $1 machine number
# $2 config_file
if [ -d $folderpath ]; then
	mpiexec -hostfile mpd.hosts -n $1 ../graph-store-distributed/test_traverser_keeppath.out $2
else
	echo  $folderpath "not exist"
fi


#mpiexec -hostfile mpd.hosts -n $1 ../graph-store-distributed/test_traverser_keeppath.out /home/sjx/nfs/LUBM/id_univ$2/