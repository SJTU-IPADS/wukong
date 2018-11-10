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
 *      http://ipads.se.sjtu.edu.cn/projects/wukong
 *
 */

#pragma once

#pragma GCC diagnostic warning "-fpermissive"

#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <vector>

using namespace std;

#include "global.hpp"

// utils
#include "timer.hpp"
#include "assertion.hpp"

#ifdef HAS_RDMA

// rdma_lib
#include "rdmaio.hpp"

using namespace rdmaio;

class RDMA {

public:
    enum MemType { CPU, GPU };

    struct MemoryRegion {
        MemType type;
        char *addr;
        uint64_t sz;
        void *mem;
    };

    class RDMA_Device {
        static const uint64_t RDMA_CTRL_PORT = 19344;
    public:
        RdmaCtrl* ctrl = NULL;

        // currently we only support one cpu and one gpu mr in mrs!
        RDMA_Device(int nnodes, int nthds, int nid, vector<RDMA::MemoryRegion> &mrs, string ipfn) {
            // record IPs of ndoes
            vector<string> ipset;
            ifstream ipfile(ipfn);
            string ip;

            // get first nnodes IPs
            for (int i = 0; i < nnodes; i++) {
                ipfile >> ip;
                ipset.push_back(ip);
            }

            // init device and create QPs
            ctrl = new RdmaCtrl(nid, ipset, RDMA_CTRL_PORT, true); // enable single context
            ctrl->query_devinfo();
            ctrl->open_device();
            for (auto mr : mrs) {
                switch (mr.type) {
                case RDMA::MemType::CPU:
                    ctrl->set_connect_mr(mr.addr, mr.sz);
                    ctrl->register_connect_mr();
                    break;
                case RDMA::MemType::GPU:
#ifdef USE_GPU
                    ctrl->set_connect_mr_gpu(mr.addr, mr.sz);
                    ctrl->register_connect_mr_gpu();
                    break;
#else
                    logstream(LOG_ERROR) << "Build wukong w/o GPU support." << LOG_endl;
                    ASSERT(false);
#endif
                default:
                    logstream(LOG_ERROR) << "Unkown memory region." << LOG_endl;
                }
            }

            ctrl->start_server();
            for (uint j = 0; j < nthds; ++j) {
                for (uint i = 0; i < nnodes; ++i) {
                    // FIXME: statically use 1 device and 1 port
                    //
                    // devID: [0, #devs), portID: (0, #ports]
                    // 0: always choose the 1st (RDMA) device
                    // 1: always choose the 1st (RDMA) port
                    Qp *qp = ctrl->create_rc_qp(j, i, 0, 1);
                    ASSERT(qp != NULL);
                }
            }

            // connect all QPs
            while (1) {
                int connected = 0;
                for (uint j = 0; j < nthds; ++j) {
                    for (uint i = 0; i < nnodes; ++i) {
                        Qp *qp = ctrl->get_rc_qp(j, i);

                        if (qp->inited_) // has connected
                            connected ++;
                        else if (qp->connect_rc())
                            connected ++;
                    }
                }

                if (connected == nthds * nnodes) break; // done
            }
        }

#ifdef USE_GPU
        // (sync) GPUDirect RDMA Write (w/ completion)
        int GPURdmaWrite(int tid, int nid, char *local_gpu,
                         uint64_t sz, uint64_t off, bool to_gpu = false) {
            Qp* qp = ctrl->get_rc_qp(tid, nid);

            int flags = IBV_SEND_SIGNALED;
            qp->rc_post_send_gpu(IBV_WR_RDMA_WRITE, local_gpu, sz, off, flags, to_gpu);
            qp->poll_completion();
            return 0;
        }
#endif

        // (sync) RDMA Read (w/ completion)
        int RdmaRead(int tid, int nid, char *local, uint64_t sz, uint64_t off) {
            Qp* qp = ctrl->get_rc_qp(tid, nid);

            // sweep remaining completion events (due to selective RDMA writes)
            if (!qp->first_send())
                qp->poll_completion();

            qp->rc_post_send(IBV_WR_RDMA_READ, local, sz, off, IBV_SEND_SIGNALED);
            qp->poll_completion();
            return 0;
        }

        // (sync) RDMA Write (w/ completion)
        int RdmaWrite(int tid, int nid, char *local, uint64_t sz, uint64_t off) {
            Qp* qp = ctrl->get_rc_qp(tid, nid);

            int flags = IBV_SEND_SIGNALED;
            qp->rc_post_send(IBV_WR_RDMA_WRITE, local, sz, off, flags);
            qp->poll_completion();
            return 0;
        }

        // (blind) RDMA Write (w/o completion)
        int RdmaWriteNonSignal(int tid, int nid, char *local, uint64_t sz, uint64_t off) {
            Qp* qp = ctrl->get_rc_qp(tid, nid);
            int flags = 0;
            qp->rc_post_send(IBV_WR_RDMA_WRITE, local, sz, off, flags);
            return 0;
        }

        // (adaptive) RDMA Write (w/o completion)
        int RdmaWriteSelective(int tid, int nid, char *local, uint64_t sz, uint64_t off) {
            Qp* qp = ctrl->get_rc_qp(tid, nid);

            int flags = (qp->first_send() ? IBV_SEND_SIGNALED : 0);
            qp->rc_post_send(IBV_WR_RDMA_WRITE, local, sz, off, flags);
            if (qp->need_poll())  // sweep all completion (batch)
                qp->poll_completion();
            return 0;
        }
    };

    RDMA_Device *dev = NULL;

    RDMA() { }

    ~RDMA() { if (dev != NULL) delete dev; }

    void init_dev(int nnodes, int nthds, int nid, vector<RDMA::MemoryRegion> &mrs, string ipfn) {
        dev = new RDMA_Device(nnodes, nthds, nid, mrs, ipfn);
    }

    inline static bool has_rdma() { return true; }

    static RDMA &get_rdma() {
        static RDMA rdma;
        return rdma;
    }
};

void RDMA_init(int nnodes, int nthds, int nid, vector<RDMA::MemoryRegion> &mrs, string ipfn) {
    uint64_t t = timer::get_usec();

    // init RDMA device
    RDMA &rdma = RDMA::get_rdma();
    rdma.init_dev(nnodes, nthds, nid, mrs, ipfn);

    t = timer::get_usec() - t;
    logstream(LOG_INFO) << "initializing RMDA done (" << t / 1000  << " ms)" << LOG_endl;
}

#else

class RDMA {

public:
    enum MemType { CPU, GPU };

    struct MemoryRegion {
        MemType type;
        char *addr;
        uint64_t sz;
        void *mem;
    };

    class RDMA_Device {
    public:
        RDMA_Device(int nnodes, int nthds, int nid, vector<RDMA::MemoryRegion> &mrs, string fname) {
            logstream(LOG_INFO) << "This system is compiled without RDMA support." << LOG_endl;
            ASSERT(false);
        }

        int RdmaRead(int tid, int nid, char *local, uint64_t sz, uint64_t off) {
            logstream(LOG_INFO) << "This system is compiled without RDMA support." << LOG_endl;
            ASSERT(false);
            return 0;
        }

        int RdmaWrite(int tid, int nid, char *local, uint64_t sz, uint64_t off) {
            logstream(LOG_INFO) << "This system is compiled without RDMA support." << LOG_endl;
            ASSERT(false);
            return 0;
        }

        int RdmaWriteNonSignal(int tid, int nid, char *local, uint64_t sz, uint64_t off) {
            logstream(LOG_INFO) << "This system is compiled without RDMA support." << LOG_endl;
            ASSERT(false);
            return 0;
        }

        int RdmaWriteSelective(int tid, int nid, char *local, uint64_t sz, uint64_t off) {
            logstream(LOG_INFO) << "This system is compiled without RDMA support." << LOG_endl;
            ASSERT(false);
            return 0;
        }
    };

    RDMA_Device *dev = NULL;

    RDMA() { }

    ~RDMA() { }

    void init_dev(int nnodes, int nthds, int nid, vector<RDMA::MemoryRegion> &mrs, string ipfn) {
        dev = new RDMA_Device(nnodes, nthds, nid, mrs, ipfn);
    }

    inline static bool has_rdma() { return false; }

    static RDMA &get_rdma() {
        static RDMA rdma;
        return rdma;
    }
};

void RDMA_init(int nnodes, int nthds, int nid, vector<RDMA::MemoryRegion> &mrs, string ipfn) {
    logstream(LOG_INFO) << "This system is compiled without RDMA support." << LOG_endl;
}

#endif // end of HAS_RDMA
