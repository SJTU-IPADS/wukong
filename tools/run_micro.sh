#!/bin/bash

# $1 NUM_SENDER;
# $2 NUM_RECVER;
# $3 batch_factor;
# $4 extra_work;

# mpiexec -hostfile mpd.hosts -n 4 ../micro/micro.out 4 1 100 0

mpiexec -hostfile mpd.hosts -n 2 ../micro/verb.out 1 1 1 0