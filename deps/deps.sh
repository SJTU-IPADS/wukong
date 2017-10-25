#!/bin/bash
trap "trap - ERR; return" ERR
# vars
openmpi="openmpi-1.6.5"
boost="boost_1_58_0"
tbb="tbb44_20151115oss"
zeromq="zeromq-4.0.5"
hwloc="hwloc-1.11.7"
librdma="librdma-1.0.0"

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
                wget "http://sourceforge.net/projects/boost/files/boost/1.58.0/${boost}.tar.gz"
            fi
            tar zxf "${boost}.tar.gz"
        fi
        cd "$WUKONG_ROOT/deps/${boost}"
        trap - ERR
        ./bootstrap.sh --prefix="$WUKONG_ROOT/deps/${boost}-install/" 2>>install_deps.log
        echo "using mpi : \$WUKONG_ROOT/deps/${openmpi}-install/bin/mpicc ;" >> project-config.jam
        ./b2 install 2>>install_deps.log
    else
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

install_zeromq(){
    trap "return" ERR
    echo "Installing ${zeromq}..."
    write_log "${zeromq}"
    cd "$WUKONG_ROOT/deps"
    if [ ! -d "${zeromq}-install" ]; then
        mkdir "${zeromq}-install"
        if [ ! -d "${zeromq}" ]; then
            if [ ! -f "${zeromq}.tar.gz" ]; then
                wget "https://archive.org/download/zeromq_4.0.5/${zeromq}.tar.gz"
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

install_librdma(){
    trap "return" ERR
    echo "Installing ${librdma}..."
    write_log "${librdma}"
    cd "$WUKONG_ROOT/deps"
    if [ ! -d "${librdma}-install" ]; then
        mkdir "${librdma}-install"
        if [ ! -d "${librdma}" ]; then
            if [ ! -f "${librdma}.tar.gz" ]; then
                wget "http://ipads.se.sjtu.edu.cn/wukong/${librdma}.tar.gz"
            fi
            tar zxf "${librdma}.tar.gz"
        fi
        cd "$WUKONG_ROOT/deps/${librdma}"
        trap - ERR
        ./configure --prefix="$WUKONG_ROOT/deps/${librdma}-install/" 2>>install_deps.log
        make 2>>install_deps.log
        make install 2>>install_deps.log
    else
        echo "found librdma."
    fi
    if [ $( echo "${CPATH}" | grep "${librdma}-install" | wc -l ) -eq 0 ]; then
        echo '# librdma configuration' >> ~/.bashrc
        echo "export CPATH=\$WUKONG_ROOT/deps/${librdma}-install/include:\$CPATH" >> ~/.bashrc
        echo "export LIBRARY_PATH=\$WUKONG_ROOT/deps/${librdma}-install/lib:\$LIBRARY_PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$WUKONG_ROOT/deps/${librdma}-install/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
        source ~/.bashrc
    fi
}

del_mpi(){
    trap "return" ERR
	echo 'removing mpi...'
	rm -rf "$WUKONG_ROOT/deps/${openmpi}-install" "$WUKONG_ROOT/deps/${openmpi}"
}

del_boost(){
    trap "return" ERR
	echo 'removing boost...'
	rm -rf "$WUKONG_ROOT/deps/${boost}-install" "$WUKONG_ROOT/deps/${boost}"
}

del_tbb(){
    trap "return" ERR
	echo 'removing tbb...'
	rm -rf "$WUKONG_ROOT/deps/${tbb}"
	sed -i '/\(Intel TBB configuration\)\|\(tbbvars\.sh\)/d' ~/.bashrc
	unset TBBROOT
}

del_zeromq(){
    trap "return" ERR
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

del_hwloc(){
    trap "return" ERR
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

del_librdma(){
    trap "return" ERR
	echo 'removing librdma...'
	rm -rf "$WUKONG_ROOT/deps/${librdma}-install" "$WUKONG_ROOT/deps/${librdma}"
	sed -i '/\(librdma configuration\)\|\(CPATH.*librdma\)\|\(LIBRARY_PATH.*librdma\)\|\(LD_LIBRARY_PATH.*librdma\)/d' ~/.bashrc

	pattern=":$WUKONG_ROOT/deps/librdma[^:]*"
	NEW_CPATH=`echo $CPATH | sed 's%'"$pattern"'%%g'`
	NEW_LIBRARY_PATH=`echo $LIBRARY_PATH | sed 's%'"$pattern"'%%g'`
	NEW_LD_LIBRARY_PATH=`echo $LD_LIBRARY_PATH | sed 's%'"$pattern"'%%g'`
	export CPATH=$NEW_CPATH
	export LIBRARY_PATH=$NEW_LIBRARY_PATH
	export LD_LIBRARY_PATH=$NEW_LD_LIBRARY_PATH
}

clean_deps(){
    trap "return" ERR
    echo 'compressed packages will not be removed.'
	if [[ "$#" == "1" || "$2" == "all" ]]; then
		echo 'cleaning all dependencies...'
		del_mpi
		del_boost
		del_tbb
		del_zeromq
		del_hwloc
		del_librdma
	else
		for ((i=2;i<=$#;i++)); do
			item=${!i}
			case "$item" in
				"mpi") del_mpi ;;
				"boost") del_boost ;;
				"tbb") del_tbb ;;
				"zeromq") del_zeromq ;;
				"hwloc") del_hwloc ;;
				"librdma") del_librdma ;;
				*) echo "cannot clean $item" ;;
			esac
		done
	fi
}

# handle options
if [ $WUKONG_ROOT ]; then
    if [ "$1" == "clean" ]; then
        clean_deps "$@"
    else
        install_mpi
        install_boost
        install_tbb
        install_zeromq
        install_hwloc
        if [ "$1" == "no-rdma" ]; then
            echo 'librdma will not be installed.'
        else
            install_librdma
        fi
    fi
    cd "$WUKONG_ROOT/deps"
    source ~/.bashrc
else
    echo 'Please set WUKONG_ROOT first.'
fi
trap - ERR
