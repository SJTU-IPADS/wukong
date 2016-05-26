#!/bin/bash
root_dir=/home/sjx/graph-query/


cat mpd.hosts | while read machine
do
	#Don't copy things like Makefile, *.o, etc.
	rsync -rtuv --exclude=deps/* --include=build/wukong* --exclude=build/* --exclude=.git   ${root_dir} ${machine}:${root_dir}
done
