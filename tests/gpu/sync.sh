#!/bin/bash
#e.g. ./sync.sh

root=${WUKONG_ROOT}/test/

if [ "$root" = "/" ] ;
then
	echo  "PLEASE set WUKONG_ROOT"
	exit 0
fi

cat mpd.hosts | while read machine
do
	#Don't copy things like Makefile, CMakeFiles, etc in build directory.
	rsync -rtuvl --include=gpu/build/gputest* --exclude=gpu/build/* --exclude=.git   ${root} ${machine}:${root}
done
