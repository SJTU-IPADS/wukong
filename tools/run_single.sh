#!/bin/bash
# $1 machine number
# $2 input files
folderpath="/home/sjx/nfs/LUBM/id_univ$2/"
if [ -d $folderpath ]; then
	mpirun -np $1 ../graph-store-distributed/test_traverser_keeppath.out $folderpath
else
	echo  $folderpath "not exist"
fi
