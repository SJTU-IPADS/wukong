#!/bin/bash
root_dir=${WUKONG_ROOT}/

if [ "$root_dir" = "/" ] ;
then
	echo  "PLEASE set WUKONG_ROOT"
	exit 0
fi

cat mpd.hosts | while read machine
do
	#Don't copy things like Makefile, *.o, etc.
	rsync -rtuv --exclude=deps/* --include=build/wukong* --exclude=build/* --exclude=.git   ${root_dir} ${machine}:${root_dir}
done
