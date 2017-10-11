#!/bin/bash

# vars
librdma="librdma-1.0.0"
tbb="tbb44_20151115oss"
openmpi="openmpi-1.6.5"
hwloc="hwloc-1.11.7"
boost="boost_1_58_0"
zeromq="zeromq-4.0.5"

# mpi
install_mpi(){
  echo "Installing ${openmpi}..."
  if [ ! -d "${openmpi}-install" ]
  then
    mkdir "${openmpi}-install"
    if [ ! -d "${openmpi}" ]
    then
      if [ ! -f "${openmpi}.tar.gz" ]
      then
        wget "https://www.open-mpi.org/software/ompi/v1.6/downloads/${openmpi}.tar.gz"
      fi
      tar zxf "${openmpi}.tar.gz" && cd "$WUKONG_ROOT/deps/${openmpi}"
      ./configure --prefix="$WUKONG_ROOT/deps/${openmpi}-install/"
      make all
      make install
      cd ..
    fi
  else
    echo "found ${openmpi}."
  fi
}

# boost
install_boost(){
  echo "Installing ${boost}..."
  if [ ! -d "${boost}-install" ]
  then
    mkdir "${boost}-install"
    if [ ! -d "${boost}" ]
    then
      if [ ! -f "${boost}.tar.gz" ]
      then
        wget "http://sourceforge.net/projects/boost/files/boost/1.58.0/${boost}.tar.gz"
      fi
      tar zxf "${boost}.tar.gz" && cd "$WUKONG_ROOT/deps/${boost}"
      ./bootstrap.sh --prefix="../${boost}-install/"
      echo 'using mpi : $WUKONG_ROOT/deps/openmpi-1.6.5-install/bin/mpicc ;' >> project-config.jam
      ./b2 install
      cd ..
    fi
  else
    echo "found ${boost}."
  fi
}

# tbb
install_tbb(){
  echo "Installing ${tbb}..."
  if [ ! -d "${tbb}" ]
  then
    if [ ! -f "${tbb}_src.tgz" ]
    then
      wget "https://www.threadingbuildingblocks.org/sites/default/files/software_releases/source/${tbb}_src.tgz"
    fi
    tar zxf "${tbb}_src.tgz" && cd "$WUKONG_ROOT/deps/${tbb}"
    make
    cd ..
  else
    echo "found ${tbb}."
  fi
}

# zeromq
install_zeromq(){
  echo "Installing ${zeromq}..."
  if [ ! -d "${zeromq}-install" ]
  then
    mkdir "${zeromq}-install"
    if [ ! -d "${zeromq}" ]
    then
      if [ ! -f "${zeromq}.tar.gz" ]
      then
        wget "https://archive.org/download/zeromq_4.0.5/${zeromq}.tar.gz"
      fi
      tar zxf "${zeromq}.tar.gz" && cd "$WUKONG_ROOT/deps/${zeromq}"
      ./configure --prefix="$WUKONG_ROOT/deps/${zeromq}-install/"
      make
      make install
      cd ..
      cp zmq.hpp "${zeromq}-install/include"
      cp zhelpers.hpp "${zeromq}-install/include"
    fi
  else
    echo "found ${zeromq}."
  fi
}

# hwloc
install_hwloc(){
  echo "Installing ${hwloc}..."
  if [ ! -d "${hwloc}-install" ]
  then
    mkdir "${hwloc}-install"
    if [ ! -d "${hwloc}" ]
    then
      if [ ! -f "${hwloc}.tar.gz" ]
      then
        wget "https://www.open-mpi.org/software/hwloc/v1.11/downloads/${hwloc}.tar.gz"
      fi
      tar zxf "${hwloc}.tar.gz" && cd "$WUKONG_ROOT/deps/${hwloc}"
      ./configure --prefix="$WUKONG_ROOT/deps/${hwloc}-install/"
      make
      make install
      cd ..
    fi
  else
    echo "found ${hwloc}."
  fi
}

# librdma
install_librdma(){
  echo "Installing ${librdma}..."
  if [ ! -d "${librdma}-install" ]
  then
    mkdir "${librdma}-install"
    if [ ! -d "${librdma}" ]
    then
      if [ ! -f "${librdma}.tar.gz" ]
      then
        wget "http://ipads.se.sjtu.edu.cn/wukong/${librdma}.tar.gz"
      fi
      tar zxf "${librdma}.tar.gz" && cd "$WUKONG_ROOT/deps/${librdma}"
      ./configure --prefix="$WUKONG_ROOT/deps/${librdma}-install/"
      make
      make install
      cd ..
    fi
  else
    echo "found librdma."
  fi
}

# handle options
install_mpi
install_boost
install_tbb
install_zeromq
install_hwloc
if [ "$#" == "1" ] && [ "$1" == "no-rdma" ] ; then
  echo 'librdma will not be installed.'
else
  install_librdma
fi
