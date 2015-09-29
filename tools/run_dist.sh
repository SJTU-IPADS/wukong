#!/bin/bash
# $1 machine number
# $2 input files
# $3 batch_factor
folderpath="/home/datanfs/nfs0/LUBM/id_univ$2/"
if [ -d $folderpath ]; then
	mpiexec -hostfile mpd.hosts -n $1 ../graph-store-distributed/test_traverser_keeppath.out $folderpath $3
else
	echo  $folderpath "not exist"
fi


#mpiexec -hostfile mpd.hosts -n $1 ../graph-store-distributed/test_traverser_keeppath.out /home/sjx/nfs/LUBM/id_univ$2/