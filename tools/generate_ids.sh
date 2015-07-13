#!/bin/bash
mkdir id_univ$1
cd id_univ$1
../index_server.out ../lubm
mv ../id_univ$1 ~/nfs/LUBM/id_univ$1
cd ..