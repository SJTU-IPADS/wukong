FROM ubuntu:16.04
ENV DEBIAN_FRONTEND=noninteractive

ENV MOFED_VERSION 4.9-2.2.4.0
ENV OS_VERSION ubuntu16.04
ENV PLATFORM x86_64
ENV OFED https://www.mellanox.com/downloads/ofed/MLNX_OFED-${MOFED_VERSION}/MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}.tgz
ENV OFED_FILE MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}.tgz
ENV OFED_PATH MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_VERSION}-${PLATFORM}

RUN cd /root \
    && mkdir wukong \
    && cd wukong \
    && apt-get -y update \
    && apt-get -y install apt-utils \
    && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        gcc g++ libreadline6-dev libbz2-dev python-dev rsync \
            build-essential cmake tcsh tcl tk \
	        make git curl vim wget ca-certificates \
		        iputils-ping net-tools ethtool \
			        perl lsb-release python-libxml2 \
				        iproute2 pciutils libnl-route-3-200 \
					        kmod libnuma1 lsof openssh-server \
						        swig libelf1 automake libglib2.0-0 \
							        autoconf graphviz chrpath flex libnl-3-200 m4 \
								        debhelper autotools-dev gfortran libltdl-dev pkg-config bison dpatch && \
									        rm -rf /rm -rf /var/lib/apt/lists/*
RUN wget --quiet ${OFED} && \
    tar -xvf ${OFED_FILE} && \
        ${OFED_PATH}/mlnxofedinstall --user-space-only --without-fw-update -q && \
	    cd .. && \
	        rm -rf ${MOFED_DIR} && \
		    rm -rf *.tgz
RUN cat /etc/ssh/ssh_config | grep -vE '(Port|StrictHostKeyChecking)' > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    echo "    Port 50001" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config && \
    cat /etc/ssh/sshd_config | grep -vE '(Port|PermitRootLogin)' > /etc/ssh/sshd_config.new && \
    echo "Port 50001" >> /etc/ssh/sshd_config.new && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config.new && \
    mv /etc/ssh/sshd_config.new /etc/ssh/sshd_config
    

ADD . /root/wukong/
RUN cd /root/wukong/deps \
    && echo "export WUKONG_ROOT=/root/wukong" >> ~/.bashrc \
    && export WUKONG_ROOT=/root/wukong \
    && /bin/bash -c "source deps.sh"
