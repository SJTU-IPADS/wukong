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

//#pragma GCC diagnostic warning ignored "-fpermissive"

#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <vector>

#include "core/common/global.hpp"

// utils
#include "utils/timer.hpp"
#include "utils/assertion.hpp"

#ifdef HAS_RDMA

// rdma_lib
#include "rib/core/lib.hh"

using namespace rdmaio;
using namespace rdmaio::rmem;
using namespace rdmaio::qp;

// #define FLAGS_use_nic_idx 0
// #define FLAGS_reg_nic_name 73
// #define FLAGS_reg_cpu_mem_name 73
// #define FLAGS_reg_gpu_mem_name 74

namespace wukong {

class RDMA {

public:
    enum MemType { CPU, GPU };

    struct MemoryRegion {
        MemType type;
        char *addr;
        uint64_t sz;
        void *mem;
    };

    struct Connection {
        ConnectManager *cm;
        Arc<RC> qp;
        uint64_t key;
        rmem::RegAttr remoteMr;
    };

    class RDMA_Device {
    public:
        static const int FLAGS_use_nic_idx = 0;
        static const int FLAGS_reg_nic_name = 73;
        static const int FLAGS_reg_cpu_mem_name = 73;
        static const int FLAGS_reg_gpu_mem_name = 74;

        int sid;
        RCtrl *rctrl;
        Arc<RNic> rnic;
        std::vector<std::vector<Connection>> connections;

        Arc<RegHandler> local_cpu_mem;
    #ifdef USE_GPU
        Arc<RegHandler> local_gpu_mem;
    #endif

         // currently we only support one cpu and one gpu mr in mrs!
        RDMA_Device(int nnodes, int nthds, int nid, std::vector<RDMA::MemoryRegion> &mrs, std::string ipfn) {
            // record IPs of nnodes
            this->sid = nid;

            std::vector<std::string> ipset;
            std::ifstream ipfile(ipfn);
            std::string ip;

            // get nnodes IPs
            for (int i = 0; i < nnodes; i++) {
                ipfile >> ip;
                ipset.push_back(ip);
            }

            rctrl = new RCtrl(Global::rdma_ctrl_port_base);

            // open the NIC
            rnic = RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();

            // register the nic with name 0 to the rctrl
            RDMA_ASSERT(rctrl->opened_nics.reg(FLAGS_reg_nic_name, rnic));

            for (auto mr : mrs) {
                switch (mr.type) {
                case RDMA::MemType::CPU: {
                    this->local_cpu_mem = RegisterMem(FLAGS_reg_cpu_mem_name, mr.addr, mr.sz);
                    break;
                }
                case RDMA::MemType::GPU: {
                #ifdef USE_GPU
                    this->local_gpu_mem = RegisterMem(FLAGS_reg_gpu_mem_name, mr.addr, mr.sz);
                    break;
                #else
                    logstream(LOG_ERROR) << "Build wukong w/o GPU support." << LOG_endl;
                    ASSERT(false);
                #endif
                }
                default:
                    logstream(LOG_ERROR) << "Unkown memory region." << LOG_endl;
                }
            }

            rctrl->start_daemon();

            connections.resize(nthds);
            for(auto& vec : connections) {
                vec.resize(nnodes);
            }

            for (uint j = 0; j < nthds; ++j) {
                for (uint i = 0; i < nnodes; ++i) {
                    CreateConnection(ipset[i], j, i);
                }
            }
        }

        void CreateConnection(std::string ip, int tid, int nid) {
            std::string addr = ip + ":" + std::to_string(Global::rdma_ctrl_port_base);
            auto qp = RC::create(rnic, QPConfig()).value();
            ConnectManager *cm = new ConnectManager(addr);
            if (cm->wait_ready(1000000, 2) == IOCode::Timeout)
                RDMA_ASSERT(false) << "cm connect to server timeout";

            // global-unique qp name
            std::string qp_name = "client-qp"+ std::to_string(this->sid) + ":" 
                                    + std::to_string(tid) + ":" +std::to_string(nid);
            auto qp_res = cm->cc_rc(qp_name, qp, FLAGS_reg_nic_name, QPConfig());
            RDMA_ASSERT(qp_res == IOCode::Ok) << std::get<0>(qp_res.desc);
            auto key = std::get<1>(qp_res.desc);

            auto fetch_res = cm->fetch_remote_mr(FLAGS_reg_cpu_mem_name);
            RDMA_ASSERT(fetch_res == IOCode::Ok) << std::get<0>(fetch_res.desc);
            rmem::RegAttr remote_attr = std::get<1>(fetch_res.desc);

            connections[tid][nid].cm = cm;
            connections[tid][nid].qp = qp;
            connections[tid][nid].key = key;
            connections[tid][nid].remoteMr = remote_attr;
        }

        Arc<RegHandler> RegisterMem(int reg_mem_name, char* addr, uint64_t size) {
            auto result = rctrl->registered_mrs.create_then_reg(
                        reg_mem_name, Arc<RMem>(new RMem(size, 
                        [&](u64 sz) { return addr; }, [](RMem::raw_ptr_t p) {})),
                        rctrl->opened_nics.query(FLAGS_reg_nic_name).value());
            RDMA_ASSERT(result);
            return result.value().first;
        }

        // (sync) RDMA Read (w/ completion)
        void RdmaRead(int tid, int nid, char *local, uint32_t sz, uint64_t off) {
            auto& connection = connections[tid][nid];

            auto res_s = connection.qp->send_normal(
                   {.op = IBV_WR_RDMA_READ,
                    .flags = IBV_SEND_SIGNALED,
                    .len = sz,
                    .wr_id = 0},
                   {.local_addr = reinterpret_cast<RMem::raw_ptr_t>(local),
                    .remote_addr = off,
                    .imm_data = 0},
                    local_cpu_mem->get_reg_attr().value(),
                    connection.remoteMr);
            RDMA_ASSERT(res_s == IOCode::Ok);
            auto res_p = connection.qp->wait_one_comp();
            RDMA_ASSERT(res_p == IOCode::Ok);
        }

        // (sync) RDMA Write (w/ completion)
        void RdmaWrite(int tid, int nid, char *local, uint32_t sz, uint64_t off) {
            auto& connection = connections[tid][nid];

            auto res_s = connection.qp->send_normal(
                   {.op = IBV_WR_RDMA_WRITE,
                    .flags = IBV_SEND_SIGNALED,
                    .len = sz,
                    .wr_id = 0},
                   {.local_addr = reinterpret_cast<RMem::raw_ptr_t>(local),
                    .remote_addr = off,
                    .imm_data = 0},
                    local_cpu_mem->get_reg_attr().value(),
                    connection.remoteMr);
            RDMA_ASSERT(res_s == IOCode::Ok);
            auto res_p = connection.qp->wait_one_comp();
            RDMA_ASSERT(res_p == IOCode::Ok);
        }

        ~RDMA_Device() { if (rctrl != NULL) delete rctrl; }

#ifdef USE_GPU
        // (sync) GPUDirect RDMA Write (w/ completion)
        void GPURdmaWrite(int tid, int nid, char *local_gpu,
                         uint32_t sz, uint64_t off, bool to_gpu = false) {
            auto& connection = connections[tid][nid];

            auto res_s = connection.qp->send_normal(
                   {.op = IBV_WR_RDMA_WRITE,
                    .flags = IBV_SEND_SIGNALED,
                    .len = sz,
                    .wr_id = 0},
                   {.local_addr = reinterpret_cast<RMem::raw_ptr_t>(local_gpu),
                    .remote_addr = off,
                    .imm_data = 0},
                    local_gpu_mem->get_reg_attr().value(),
                    connection.remoteMr);
            RDMA_ASSERT(res_s == IOCode::Ok);
            auto res_p = connection.qp->wait_one_comp();
            RDMA_ASSERT(res_p == IOCode::Ok);
        }
#endif

        // (blind) RDMA Write (w/o completion)
        int RdmaWriteNonSignal(int tid, int nid, char *local, uint64_t sz, uint64_t off) {
            // TODO
            return 0;
        }

        // (adaptive) RDMA Write (w/o completion)
        int RdmaWriteSelective(int tid, int nid, char *local, uint64_t sz, uint64_t off) {
            // TODO
            return 0;
        }
    };

    RDMA_Device *dev = NULL;

    RDMA() { }

    ~RDMA() { if (dev != NULL) delete dev; }

    void init_dev(int nnodes, int nthds, int nid, std::vector<RDMA::MemoryRegion> &mrs, std::string ipfn) {
        dev = new RDMA_Device(nnodes, nthds, nid, mrs, ipfn);
    }

    inline static bool has_rdma() { return true; }

    static RDMA &get_rdma() {
        static RDMA rdma;
        return rdma;
    }
};

void RDMA_init(int nnodes, int nthds, int nid, std::vector<RDMA::MemoryRegion> &mrs, std::string ipfn) {
    uint64_t t = timer::get_usec();

    // init RDMA device
    RDMA &rdma = RDMA::get_rdma();
    rdma.init_dev(nnodes, nthds, nid, mrs, ipfn);

    t = timer::get_usec() - t;
    logstream(LOG_INFO) << "[RDMA] initializing RDMA done (" << t / 1000  << " ms)" << LOG_endl;
}

} // namespace wukong

#else

namespace wukong {

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
        RDMA_Device(int nnodes, int nthds, int nid, std::vector<RDMA::MemoryRegion> &mrs, std::string fname) {
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

    void init_dev(int nnodes, int nthds, int nid, std::vector<RDMA::MemoryRegion> &mrs, std::string ipfn) {
        dev = new RDMA_Device(nnodes, nthds, nid, mrs, ipfn);
    }

    inline static bool has_rdma() { return false; }

    static RDMA &get_rdma() {
        static RDMA rdma;
        return rdma;
    }
};

void RDMA_init(int nnodes, int nthds, int nid, std::vector<RDMA::MemoryRegion> &mrs, std::string ipfn) {
    logstream(LOG_INFO) << "This system is compiled without RDMA support." << LOG_endl;
}

} // namespace wukong

#endif // end of HAS_RDMA
