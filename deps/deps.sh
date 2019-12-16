#!/bin/bash
trap - ERR
# vars
openmpi="openmpi-1.6.5"
boost="boost_1_67_0"
tbb="tbb44_20151115oss"
zeromq="zeromq-4.0.5"
nanomsg="nanomsg-1.1.4"
hwloc="hwloc-1.11.7"
jemalloc="jemalloc-5.2.1"

write_log(){
    divider='-----------------------------'
    echo $divider >> install_deps.log
    date '+%Y-%m-%d %H:%M:%S' >> install_deps.log
    echo "on installing $1:" >> install_deps.log
}


install_mpi(){
    trap "return" ERR
    echo "Installing ${openmpi}..."
    write_log "${openmpi}"
    cd "$WUKONG_ROOT/deps"
    if [ ! -d "${openmpi}-install" ]; then
        mkdir "${openmpi}-install"
        if [ ! -d "${openmpi}" ]; then
            if [ ! -f "${openmpi}.tar.gz" ]; then
                wget "https://www.open-mpi.org/software/ompi/v1.6/downloads/${openmpi}.tar.gz"
            fi
            tar zxf "${openmpi}.tar.gz"
        fi
        cd "$WUKONG_ROOT/deps/${openmpi}"
        trap - ERR
        ./configure --prefix="$WUKONG_ROOT/deps/${openmpi}-install/" 2>>install_deps.log
        make all 2>>install_deps.log
        make install 2>>install_deps.log
    else
        trap - ERR
        echo "found ${openmpi}."
    fi
}

install_boost(){
    trap "return" ERR
    echo "Installing ${boost}..."
    write_log "${boost}"
    cd "$WUKONG_ROOT/deps"
    if [ ! -d "${boost}-install" ]; then
        mkdir "${boost}-install"
        if [ ! -d "${boost}" ]; then
            if [ ! -f "${boost}.tar.gz" ]; then
                wget "https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.tar.gz"
            fi
            tar zxf "${boost}.tar.gz"
        fi
        cd "$WUKONG_ROOT/deps/${boost}"
        trap - ERR
        ./bootstrap.sh --prefix="$WUKONG_ROOT/deps/${boost}-install" 2>>install_deps.log
        echo "using mpi : \$WUKONG_ROOT/deps/${openmpi}-install/bin/mpicc ;" >> project-config.jam
        ./b2 install 2>>install_deps.log
    else
        trap - ERR
        echo "found ${boost}."
    fi
}

install_tbb(){
    trap "return" ERR
    echo "Installing ${tbb}..."
    write_log "${tbb}"
    cd "$WUKONG_ROOT/deps"
    if [ ! -d "${tbb}" ]; then
        if [ ! -f "${tbb}_src.tgz" ]; then
            wget "https://www.threadingbuildingblocks.org/sites/default/files/software_releases/source/${tbb}_src.tgz"
        fi
        tar zxf "${tbb}_src.tgz"
        cd "$WUKONG_ROOT/deps/${tbb}"
        trap - ERR
        make 2>>install_deps.log
    else
        trap - ERR
        echo "found ${tbb}."
    fi
    cd "$WUKONG_ROOT/deps/${tbb}/build"
    tbb_prev="source \$WUKONG_ROOT/deps/${tbb}/build/"
    tbb_ver=`ls | grep _release`
    if [ ! $TBBROOT ]; then
        echo -e "\n#Intel TBB configuration" >> ~/.bashrc
        echo ${tbb_prev}${tbb_ver}"/tbbvars.sh" >> ~/.bashrc
        source ~/.bashrc
    fi
}

install_nanomsg(){
    trap "return" ERR
    echo "Installing ${nanomsg}..."
    write_log "${nanomsg}"
    cd "$WUKONG_ROOT/deps"
    if [ ! -d "${nanomsg}-install" ]; then
        mkdir "${nanomsg}-install"
        if [ ! -d "${nanomsg}" ]; then
            if [ ! -f "1.1.4.tar.gz" ]; then
                wget "https://github.com/nanomsg/nanomsg/archive/1.1.4.tar.gz"
            fi
            tar zxf "1.1.4.tar.gz"
        fi
        cd "$WUKONG_ROOT/deps/${nanomsg}"
        mkdir build
        cd build
        trap - ERR
        cmake -DCMAKE_INSTALL_PREFIX=${WUKONG_ROOT}/deps/${nanomsg}-install/ ..
        cmake --build . --target install
        cd "$WUKONG_ROOT/deps"
        cp nn.hpp "$WUKONG_ROOT/deps/${nanomsg}-install/include/nanomsg/"
    else
        trap - ERR
        echo "found ${nanomsg}."
    fi
    if [ $( echo "${CPATH}" | grep "${nanomsg}-install" | wc -l ) -eq 0 ]; then
        echo '# nanomsg configuration' >> ~/.bashrc
        echo "export CPATH=\$WUKONG_ROOT/deps/${nanomsg}-install/include:\$CPATH" >> ~/.bashrc
        echo "export LIBRARY_PATH=\$WUKONG_ROOT/deps/${nanomsg}-install/lib:\$LIBRARY_PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$WUKONG_ROOT/deps/${nanomsg}-install/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
        source ~/.bashrc
    fi
}

install_zeromq(){
    trap "return" ERR
    echo "Installing ${zeromq}..."
    write_log "${zeromq}"
    cd "$WUKONG_ROOT/deps"
    if [ ! -d "${zeromq}-install" ]; then
        mkdir "${zeromq}-install"
        if [ ! -d "${zeromq}" ]; then
            if [ ! -f "${zeromq}.tar.gz" ]; then
                wget "https://github.com/zeromq/zeromq4-x/releases/download/v4.0.5/${zeromq}.tar.gz"
            fi
            tar zxf "${zeromq}.tar.gz"
        fi
        cd "$WUKONG_ROOT/deps/${zeromq}"
        trap - ERR
        ./configure --prefix="$WUKONG_ROOT/deps/${zeromq}-install/"  2>>install_deps.log
        make 2>>install_deps.log
        make install 2>>install_deps.log
        cd "$WUKONG_ROOT/deps"
        cp zmq.hpp "${zeromq}-install/include"
        cp zhelpers.hpp "${zeromq}-install/include"
    else
        trap - ERR
        echo "found ${zeromq}."
    fi
    if [ $( echo "${CPATH}" | grep "${zeromq}-install" | wc -l ) -eq 0 ]; then
        echo '# ZeroMQ configuration' >> ~/.bashrc
        echo "export CPATH=\$WUKONG_ROOT/deps/${zeromq}-install/include:\$CPATH" >> ~/.bashrc
        echo "export LIBRARY_PATH=\$WUKONG_ROOT/deps/${zeromq}-install/lib:\$LIBRARY_PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$WUKONG_ROOT/deps/${zeromq}-install/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
        source ~/.bashrc
    fi
}

install_hwloc(){
    trap "return" ERR
    echo "Installing ${hwloc}..."
    write_log "${hwloc}"
    cd "$WUKONG_ROOT/deps"
    if [ ! -d "${hwloc}-install" ]; then
        mkdir "${hwloc}-install"
        if [ ! -d "${hwloc}" ]; then
            if [ ! -f "${hwloc}.tar.gz" ]; then
                wget "https://www.open-mpi.org/software/hwloc/v1.11/downloads/${hwloc}.tar.gz"
            fi
            tar zxf "${hwloc}.tar.gz"
        fi
        cd "$WUKONG_ROOT/deps/${hwloc}"
        trap - ERR
        ./configure --prefix="$WUKONG_ROOT/deps/${hwloc}-install/" 2>>install_deps.log
        make 2>>install_deps.log
        make install 2>>install_deps.log
    else
        trap - ERR
        echo "found ${hwloc}."
    fi
    if [ $( echo "${PATH}" | grep "${hwloc}-install" | wc -l ) -eq 0 ]; then
        echo '# hwloc configuration' >> ~/.bashrc
        echo "export PATH=\$WUKONG_ROOT/deps/${hwloc}-install/bin:\$PATH" >> ~/.bashrc
        echo "export CPATH=\$WUKONG_ROOT/deps/${hwloc}-install/include:\$CPATH" >> ~/.bashrc
        echo "export LIBRARY_PATH=\$WUKONG_ROOT/deps/${hwloc}-install/lib:\$LIBRARY_PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$WUKONG_ROOT/deps/${hwloc}-install/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
        source ~/.bashrc
    fi
}

install_jemalloc(){
    trap "return" ERR
    echo "Installing ${jemalloc}..."
    write_log "${jemalloc}"
    cd "$WUKONG_ROOT/deps"
    if [ ! -d "${jemalloc}-install" ]; then
        mkdir "${jemalloc}-install"
        if [ ! -d "${jemalloc}" ]; then
            if [ ! -f "${jemalloc}.tar.gz" ]; then
                wget "https://github.com/jemalloc/jemalloc/releases/download/5.2.1/${jemalloc}.tar.bz2"
            fi
            tar jxf "${jemalloc}.tar.bz2"
        fi
        cd "$WUKONG_ROOT/deps/${jemalloc}"
        trap - ERR
        ./configure --with-jemalloc-prefix=je --prefix="$WUKONG_ROOT/deps/${jemalloc}-install/" 2>>install_deps.log
# Option --with-jemalloc-prefix=je configures prefix which renames jemalloc's API.
# This helps to use glibc's malloc API distinctly from jemalloc's versions.
# To disable jemalloc's new/delete API overload, add --disable-cxx as follows.
#        ./configure --disable-cxx --with-jemalloc-prefix=je --prefix="$WUKONG_ROOT/deps/${jemalloc}-install/" 2>>install_deps.log
        make 2>>install_deps.log
        make install 2>>install_deps.log
    else
        trap - ERR
        echo "found ${jemalloc}."
    fi
    if [ $( echo "${PATH}" | grep "${jemalloc}-install" | wc -l ) -eq 0 ]; then
        echo '# jemalloc configuration' >> ~/.bashrc
        echo "export PATH=\$WUKONG_ROOT/deps/${jemalloc}-install/bin:\$PATH" >> ~/.bashrc
        echo "export CPATH=\$WUKONG_ROOT/deps/${jemalloc}-install/include:\$CPATH" >> ~/.bashrc
        echo "export LIBRARY_PATH=\$WUKONG_ROOT/deps/${jemalloc}-install/lib:\$LIBRARY_PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$WUKONG_ROOT/deps/${jemalloc}-install/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
        source ~/.bashrc
    fi
}

del_mpi(){
    echo 'removing mpi...'
    rm -rf "$WUKONG_ROOT/deps/${openmpi}-install" "$WUKONG_ROOT/deps/${openmpi}"
}

del_boost(){
    echo 'removing boost...'
    rm -rf "$WUKONG_ROOT/deps/${boost}-install" "$WUKONG_ROOT/deps/${boost}"
}

del_tbb(){
    echo 'removing tbb...'
    rm -rf "$WUKONG_ROOT/deps/${tbb}"
    sed -i '/\(Intel TBB configuration\)\|\(tbbvars\.sh\)/d' ~/.bashrc
    unset TBBROOT
}

del_zeromq(){
    echo 'removing zeromq...'
    rm -rf "$WUKONG_ROOT/deps/${zeromq}-install" "$WUKONG_ROOT/deps/${zeromq}"
    sed -i '/\(ZeroMQ configuration\)\|\(CPATH.*zeromq\)\|\(LIBRARY_PATH.*zeromq\)\|\(LD_LIBRARY_PATH.*zeromq\)/d' ~/.bashrc

    pattern=":$WUKONG_ROOT/deps/zeromq[^:]*"
    NEW_CPATH=`echo $CPATH | sed 's%'"$pattern"'%%g'`
    NEW_LIBRARY_PATH=`echo $LIBRARY_PATH | sed 's%'"$pattern"'%%g'`
    NEW_LD_LIBRARY_PATH=`echo $LD_LIBRARY_PATH | sed 's%'"$pattern"'%%g'`
    export CPATH=$NEW_CPATH
    export LIBRARY_PATH=$NEW_LIBRARY_PATH
    export LD_LIBRARY_PATH=$NEW_LD_LIBRARY_PATH
}

del_nanomsg(){
    echo 'removing nanomsg...'
    rm -rf "$WUKONG_ROOT/deps/${nanomsg}-install" "$WUKONG_ROOT/deps/${nanomsg}"
    sed -i '/\(nanomsg configuration\)\|\(CPATH.*nanomsg\)\|\(LIBRARY_PATH.*nanomsg\)\|\(LD_LIBRARY_PATH.*nanomsg\)/d' ~/.bashrc

    pattern=":$WUKONG_ROOT/deps/nanomsg[^:]*"
    NEW_CPATH=`echo $CPATH | sed 's%'"$pattern"'%%g'`
    NEW_LIBRARY_PATH=`echo $LIBRARY_PATH | sed 's%'"$pattern"'%%g'`
    NEW_LD_LIBRARY_PATH=`echo $LD_LIBRARY_PATH | sed 's%'"$pattern"'%%g'`
    export CPATH=$NEW_CPATH
    export LIBRARY_PATH=$NEW_LIBRARY_PATH
    export LD_LIBRARY_PATH=$NEW_LD_LIBRARY_PATH
}

del_hwloc(){
    echo 'removing hwloc...'
    rm -rf "$WUKONG_ROOT/deps/${hwloc}-install" "$WUKONG_ROOT/deps/${hwloc}"
    sed -i '/\(hwloc configuration\)\|\(PATH.*hwloc\)\|\(CPATH.*hwloc\)\|\(LIBRARY_PATH.*hwloc\)\|\(LD_LIBRARY_PATH.*hwloc\)/d' ~/.bashrc

    pattern=":$WUKONG_ROOT/deps/hwloc[^:]*"
    NEW_PATH=`echo $PATH | sed 's%'"$pattern"'%%g'`
    NEW_CPATH=`echo $CPATH | sed 's%'"$pattern"'%%g'`
    NEW_LIBRARY_PATH=`echo $LIBRARY_PATH | sed 's%'"$pattern"'%%g'`
    NEW_LD_LIBRARY_PATH=`echo $LD_LIBRARY_PATH | sed 's%'"$pattern"'%%g'`
    export PATH=$NEW_PATH
    export CPATH=$NEW_CPATH
    export LIBRARY_PATH=$NEW_LIBRARY_PATH
    export LD_LIBRARY_PATH=$NEW_LD_LIBRARY_PATH
}

del_jemalloc(){
    echo 'removing jemalloc...'
    rm -rf "$WUKONG_ROOT/deps/${jemalloc}-install" "$WUKONG_ROOT/deps/${jemalloc}"
    sed -i '/\(jemalloc configuration\)\|\(PATH.*jemalloc\)\|\(CPATH.*jemalloc\)\|\(LIBRARY_PATH.*jemalloc\)\|\(LD_LIBRARY_PATH.*jemalloc\)/d' ~/.bashrc

    pattern=":$WUKONG_ROOT/deps/jemalloc[^:]*"
    NEW_PATH=`echo $PATH | sed 's%'"$pattern"'%%g'`
    NEW_CPATH=`echo $CPATH | sed 's%'"$pattern"'%%g'`
    NEW_LIBRARY_PATH=`echo $LIBRARY_PATH | sed 's%'"$pattern"'%%g'`
    NEW_LD_LIBRARY_PATH=`echo $LD_LIBRARY_PATH | sed 's%'"$pattern"'%%g'`
    export PATH=$NEW_PATH
    export CPATH=$NEW_CPATH
    export LIBRARY_PATH=$NEW_LIBRARY_PATH
    export LD_LIBRARY_PATH=$NEW_LD_LIBRARY_PATH
}

clean_deps(){
    echo 'compressed packages will not be removed.'
    if [[ "$#" == "1" || "$2" == "all" ]]; then
        echo 'cleaning all dependencies...'
        del_mpi
        del_boost
        del_tbb
        del_zeromq
        #del_nanomsg
        del_hwloc
        del_jemalloc
    else
        for ((i=2;i<=$#;i++)); do
            item=${!i}
            case "$item" in
                "mpi") del_mpi ;;
                "boost") del_boost ;;
                "tbb") del_tbb ;;
                "zeromq") del_zeromq ;;
                "nanomsg") del_nanomsg ;;
                "hwloc") del_hwloc ;;
                "jemalloc") del_jemalloc ;;
                *) echo "cannot clean $item" ;;
            esac
        done
    fi
}

install_deps(){
    if [[ "$#" == "1" || "$2" == "all" ]]; then
        install_all_deps
    else
        for ((i=2;i<=$#;i++)); do
            item=${!i}
            case "$item" in
                "mpi") install_mpi ;;
                "boost") install_boost ;;
                "tbb") install_tbb ;;
                "zeromq") install_zeromq;;
                "nanomsg") install_nanomsg ;;
                "hwloc") install_hwloc ;;
                "jemalloc") install_jemalloc ;;
                *) echo "cannot install $item" ;;
            esac
        done
    fi
}

install_all_deps(){
    install_mpi
    install_boost
    install_tbb
    install_zeromq
    #install_nanomsg
    install_hwloc
    install_jemalloc
}

# handle options
if [ $WUKONG_ROOT ]; then
    if [ "$1" == "clean" ]; then
        clean_deps "$@"
    elif [ "$1" == "install" ]; then
        install_deps "$@"
    else
        install_all_deps
    fi
    cd "$WUKONG_ROOT/deps"
    source ~/.bashrc
else
    echo 'Please set WUKONG_ROOT first.'
fi
trap - ERR
