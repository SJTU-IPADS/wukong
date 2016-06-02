#!/bin/bash
# e.g., ./syncdeps.sh ../deps/dependencies mpd.hosts

root=${WUKONG_ROOT}

deps=$1
machines=$2

cat $deps | while read dep
do
cat $machines | while read machine
do
	rsync -a --rsync-path="mkdir -p ${root}/deps/${dep}/ && rsync" -rtuvl ${root}/deps/${dep}/ ${machine}:${root}/deps/${dep}/
done
done
