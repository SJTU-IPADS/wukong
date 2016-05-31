#!/bin/bash
root_dir=${WUKONG_ROOT}/


cat mpd.hosts | while read machine
do
	#Don't copy things like Makefile, *.o, etc.
	rsync -rtuv --exclude=deps/* --include=build/wukong* --exclude=build/* --exclude=.git $r{oot_dir} ${machine}:${root_dir}
done
