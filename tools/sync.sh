#!/bin/bash
cat mpd.hosts | while read machine
do
	rsync -rtuv /home/sjx/graph-store/graph-store/ ${machine}:/home/sjx/graph-store/graph-store/
done