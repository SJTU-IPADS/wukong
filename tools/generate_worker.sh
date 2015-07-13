#!/bin/bash
#$1 worker_id
#$2 interval
#$3 max

cd lubm

for i in `seq $1 $2 $3`
do
	echo "Doing uni $i"
	echo "Doing uni $i" >> generate_log
	mkdir uni$i
	mv University${i}_* uni$i
	cd uni$i
	find . -type f -name "University*.owl" -exec rdfcat -out N-TRIPLE -x {} >> uni$i.nt \;
	cat uni$i.nt | grep -v http://www.w3.org/2002/07/owl#Ont | grep -v http://www.w3.org/2002/07/owl#imports > university-data-$i.nt.tmp
	rm uni$i.nt
	mv university-data-$i.nt.tmp uni$i.nt
	mv uni$i.nt ..
	cd ..
	rm -rf uni$i
	echo "End uni $i"
	echo "End uni $i" >> generate_log
done
echo "worker $1 finish all jobs"
echo "worker $1 finish all jobs" >> generate_log

cd ..