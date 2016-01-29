#!/bin/bash
#cat mpd.hosts | while read machine
for machine in $(cat mpd.hosts)
do
	ssh ${machine} killall test_graph_distributed.out
done
