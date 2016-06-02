#!/bin/bash
#e.g. ./sync.sh

root=${WUKONG_ROOT}/

cat mpd.hosts | while read machine
do
	#Don't copy things like Makefile, CMakeFiles, etc in build directory.
	rsync -rtuvl --include=build/wukong* --exclude=build/* --exclude=.git --exclude=deps/* --exclude=src/* --exclude=include/* --exclude=utils/* --exclude=generate/* --exclude=test/*  ${root} ${machine}:${root}
done
