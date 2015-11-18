#!/bin/bash
cat mpd.hosts | while read machine
do
	rsync -rtuv /home/sjx/online-graph/graph-query/ ${machine}:/home/sjx/online-graph/graph-query/
done