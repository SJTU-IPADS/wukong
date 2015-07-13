#!/bin/bash
# $1 machine number
# $2 input files
folderpath="/home/sjx/nfs/LUBM/id_univ$2/"
if [ -d $folderpath ]; then
	mpiexec -hostfile mpd.hosts -n $1 ../graph-store-distributed/test_traverser_keeppath.out $folderpath
else
	echo  $folderpath "not exist"
fi


#mpiexec -hostfile mpd.hosts -n $1 ../graph-store-distributed/test_traverser_keeppath.out /home/sjx/nfs/LUBM/id_univ$2/