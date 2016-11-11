#!/bin/bash
#e.g. ./sync.sh

root=${WUKONG_ROOT}/

if [ "$root" = "/" ] ;
then
	echo  "PLEASE set WUKONG_ROOT"
	exit 0
fi

cat mpd.hosts | while read machine
do
	#Don't copy things like Makefile, CMakeFiles, etc in build directory.
	rsync -rtuvl --include=build/wukong* --exclude=build/* --exclude=.git --exclude=deps/* --exclude=src/* --exclude=include/* --exclude=utils/* --exclude=generate/* --exclude=test/*  ${root} ${machine}:${root}
done
