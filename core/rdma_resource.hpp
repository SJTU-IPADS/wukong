/*
 * Copyright (c) 2016 Shanghai Jiao Tong University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://ipads.se.sjtu.edu.cn/projects/wukong.html
 *
 */

#pragma once

#include "tcp_adaptor.hpp"

#pragma GCC diagnostic warning "-fpermissive"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <inttypes.h>
#include <endian.h>
#include <byteswap.h>
#include <getopt.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <vector>
#include <pthread.h>

#include "timer.hpp"

#include "rdmaio.h"
using namespace rdmaio;

#ifdef HAS_RDMA

class RDMA {
    class RDMA_Device {
    public:
        RdmaCtrl* ctrl = NULL;
        RDMA_Device(int num_nodes, int num_threads, int node_id,
                    char *mem, uint64_t mem_sz, string ipfn)
            {

            // record IPs of ndoes
            vector<string> ipset;
            ifstream ipfile(ipfn);
            string ip;
            while (ipfile >> ip)
                ipset.push_back(ip);

            //initialization of new librdma
            //node_id, ipset, port, thread_id-no use, enable single memory region 
            ctrl = new RdmaCtrl(node_id, ipset, 19344, true);
            ctrl->open_device();
            ctrl->set_connect_mr(mem, mem_sz);
            ctrl->register_connect_mr();//single
            ctrl->start_server();
            for(uint j = 0; j < num_threads; ++j){
               for(uint i = 0;i < num_nodes;++i) {
                   ctrl->create_rc_qp(j,i,0,1);
               }
            }

        }

        // 0 on success, -1 otherwise
        int RdmaRead(int dst_tid, int dst_nid, char *local, uint64_t size, uint64_t off) {
            Qp* qp = ctrl->get_rc_qp(dst_tid,dst_nid);
            qp->rc_post_send(IBV_WR_RDMA_READ,local,size,off,IBV_SEND_SIGNALED);
            qp->poll_completion();
            return 0;
            // return rdmaOp(dst_tid, dst_nid, local, size, off, IBV_WR_RDMA_READ);
        }

        int RdmaWrite(int dst_tid, int dst_nid, char *local, uint64_t size, uint64_t off) {
            Qp* qp = ctrl->get_rc_qp(dst_tid,dst_nid);
            qp->rc_post_send(IBV_WR_RDMA_WRITE,local,size,off,IBV_SEND_SIGNALED);
            qp->poll_completion();
            return 0;
            // return rdmaOp(dst_tid, dst_nid, local, size, off, IBV_WR_RDMA_WRITE);
        }

    }; // end of class RdmaResource

public:
    RDMA_Device *dev = NULL;

    RDMA() { }

    ~RDMA() { if (dev != NULL) delete dev; }

    void init_dev(int num_nodes, int num_threads, int node_id,
                  char *mem, uint64_t mem_sz, string ipfn) {
        dev = new RDMA_Device(num_nodes, num_threads, node_id, mem, mem_sz, ipfn);
    }

    inline static bool has_rdma() { return true; }

    static RDMA &get_rdma() {
        static RDMA rdma;
        return rdma;
    }
}; // end of clase RDMA

void RDMA_init(int num_nodes, int num_threads, int node_id, char *mem, uint64_t mem_sz, string ipfn) {
    uint64_t t = timer::get_usec();

    RDMA &rdma = RDMA::get_rdma();

    // init RDMA device
    rdma.init_dev(num_nodes, num_threads, node_id, mem, mem_sz, ipfn);

    t = timer::get_usec() - t;
    cout << "INFO: initializing RMDA done (" << t / 1000  << " ms)" << endl;
}

#else

class RDMA {
    class RDMA_Device {
    public:
        RDMA_Device(int num_nodes, int num_threads, int node_id,
                    string fname, char *mem, uint64_t mem_sz) {
            cout << "This system is compiled without RDMA support." << endl;
            assert(false);
        }

        void servicing() {
            cout << "This system is compiled without RDMA support." << endl;
            assert(false);
        }

        void connect() {
            cout << "This system is compiled without RDMA support." << endl;
            assert(false);
        }

        string ip_of(int sid) {
            cout << "This system is compiled without RDMA support." << endl;
            assert(false);
            return string();
        }

        int RdmaRead(int dst_tid, int dst_nid, char *local,
                     uint64_t size, uint64_t remote_offset) {
            cout << "This system is compiled without RDMA support." << endl;
            assert(false);
            return 0;
        }

        int RdmaWrite(int dst_tid, int dst_nid, char *local,
                      uint64_t size, uint64_t remote_offset) {
            cout << "This system is compiled without RDMA support." << endl;
            assert(false);
            return 0;
        }

        int RdmaCmpSwap(int dst_tid, int dst_nid, char *local,
                        uint64_t compare, uint64_t swap,
                        uint64_t size, uint64_t off) {
            cout << "This system is compiled without RDMA support." << endl;
            assert(false);
            return 0;
        }
    }; // end of class RdmaResource

public:
    RDMA_Device *dev = NULL;

    RDMA() {
        std::cout << "This system is compiled without RDMA support."
                  << std::endl;
    }

    ~RDMA() { }

    void init_dev(int num_nodes, int num_threads, int node_id,
                  char *mem, uint64_t mem_sz, string ipfn) {
        std::cout << "This system is compiled without RDMA support."
                  << std::endl;
    }

    inline static bool has_rdma() { return false; }

    static RDMA &get_rdma() {
        static RDMA rdma;
        return rdma;
    }

};

void RDMA_init(int num_nodes, int num_threads, int node_id,
               char *mem, uint64_t mem_sz, string ipfn) {
    std::cout << "This system is compiled without RDMA support."
              << std::endl;
}

#endif
