#!/bin/bash

mkdir -p lubm
cd lubm
java -cp ../uba/uba1.7/classes/ edu.lehigh.swat.bench.uba.Generator -univ $1 -onto http://swat.cse.lehigh.edu/onto/univ-bench.owl
cd ..

let "max=$1-1"
worker=6
let "max_worker=$worker-1"
for i in `seq 0 $max_worker`
do
	./generate_worker.sh $i $worker $max &
done
