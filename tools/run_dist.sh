#!/bin/bash

mpiexec -hostfile mpd.hosts -n $1 ../graph-store-distributed/test_traverser_keeppath.out /home/sjx/nfs/LUBM/id_univ6/