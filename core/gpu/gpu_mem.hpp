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

#ifdef USE_GPU

#include "rdma.hpp"
#include "type.hpp"

// utils
#include "unit.hpp"
#include "gpu.hpp"
#include "global.hpp"

class GPUMem {
private:
    int devid;
    int num_servers;
    int num_agents; // #gpu_engine

    // The Wukong's (device) GPU memory layout: kvcache | result buffer | RDMA buffer | heap
    char *mem_gpu;
    uint64_t mem_gpu_sz;

    // key-value cache
    char *kvc;
    uint64_t kvc_sz;
    uint64_t kvc_off;

    // (running) result buffer
    // A dual-buffer (in-buf and out-buf), used to store (last and current) history table.
    char *irbuf;
    uint64_t irbuf_off;
    char *orbuf;
    uint64_t orbuf_off;
    uint64_t rbuf_sz;
    bool rbuf_reversed = false;

    // RDMA buffer (#threads)
    char *buf;
    uint64_t buf_off;
    uint64_t buf_sz; // per thread

public:
    GPUMem(int devid, int num_servers, int num_agents)
        : devid(devid), num_servers(num_servers), num_agents(num_agents) {

        kvc_sz = GiB2B(Global::gpu_kvcache_size_gb);

        rbuf_sz = MiB2B(Global::gpu_rbuf_size_mb);

        // only used by RDMA device
        if (RDMA::get_rdma().has_rdma())
            buf_sz = MiB2B(Global::gpu_rdma_buf_size_mb);
        else
            buf_sz = 0;

        mem_gpu_sz = kvc_sz + rbuf_sz * 2 + buf_sz * num_agents;

        // allocate memory and zeroing
        CUDA_ASSERT(cudaSetDevice(devid));
        CUDA_ASSERT(cudaMalloc(&mem_gpu, mem_gpu_sz));
        CUDA_ASSERT(cudaMemset(mem_gpu, 0, mem_gpu_sz));

        // kvcache
        kvc_off = 0;
        kvc = mem_gpu + kvc_off;

        // result (dual) buffer
        irbuf_off = kvc_off + kvc_sz;
        irbuf = mem_gpu + irbuf_off;
        orbuf_off = irbuf_off + rbuf_sz;
        orbuf = mem_gpu + orbuf_off;

        // RDMA buffer
        buf_off = orbuf_off + rbuf_sz;
        buf = mem_gpu + buf_off;

        logstream(LOG_INFO) << "GPUMem: devid: " << devid
                            << ", num_servers: " << num_servers
                            << ", num_agents: " << num_agents
                            << LOG_endl;
    }

    ~GPUMem() { CUDA_ASSERT(cudaFree(mem_gpu)); }

    inline char *address() { return mem_gpu; }
    inline uint64_t size() { return mem_gpu_sz; }

    // gpu cache
    inline char *kvcache() { return kvc; }
    inline uint64_t kvcache_offset() { return kvc_off; }
    inline uint64_t kvcache_size() { return kvc_sz; }

    // result buffer
    inline void reverse_rbuf() { rbuf_reversed = !rbuf_reversed; }

    inline char *res_inbuf() { return (rbuf_reversed ? orbuf : irbuf); }
    inline uint64_t res_inbuf_offset() { return (rbuf_reversed ? orbuf_off : irbuf_off); }

    inline char *res_outbuf() { return (rbuf_reversed ? irbuf : orbuf); }
    inline uint64_t res_outbuf_offset() { return (rbuf_reversed ? irbuf_off : orbuf_off); }

    inline uint64_t res_buf_size() { return rbuf_sz; }

    // RDMA buffer layout: header | type | body
    inline char *rdma_buf_hdr(int tid) { return buf + buf_sz * (tid % num_agents); }
    inline char *rdma_buf_body(int tid) { return rdma_buf_hdr(tid) + sizeof(uint64_t); }
    /* inline char *rdma_buf_type(int tid) { return rdma_buf_hdr(tid) + sizeof(uint64_t); }
     * inline char *rdma_buf_body(int tid) { return rdma_buf_hdr(tid) + 2 * sizeof(uint64_t); } */
    inline uint64_t rdma_buf_hdr_offset(int tid) { return buf_off + buf_sz * (tid % num_agents); }
    inline uint64_t rdma_buf_type_offset(int tid) { return buf_off + buf_sz * (tid % num_agents) + sizeof(uint64_t); }
    inline uint64_t rdma_buf_body_offset(int tid) { return buf_off + buf_sz * (tid % num_agents) + 2 * sizeof(uint64_t); }
    inline uint64_t rdma_buf_body_size() { return buf_sz - 2 * sizeof(uint64_t); }

};

#endif
