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

#include <unordered_map>

#include "core/common/rdma.hpp"
#include "core/common/type.hpp"

// utils
#include "core/common/global.hpp"
#include "gpu_utils.hpp"
#include "utils/unit.hpp"

#define MAX_GPU_CHANNELS 8

namespace wukong {
class GPUMem {
public:
    struct rbuf_t {
        friend class GPUMem;

    private:
        char* irbuf;
        char* orbuf;
        bool reversed;
        bool used;

    public:
        sid_t* get_inbuf() const {
            return reinterpret_cast<sid_t*>(reversed ? orbuf : irbuf);
        }

        sid_t* get_outbuf() const {
            return reinterpret_cast<sid_t*>(reversed ? irbuf : orbuf);
        }

        inline void reverse() { reversed = !reversed; }
    };

private:
    int devid;
    int num_servers;
    int num_agents;

    // The Wukong's (device) GPU memory layout: kvcache | result buffer | RDMA buffer | heap
    char* mem_gpu;
    uint64_t mem_gpu_sz;

    // key-value cache
    char* kvc;
    uint64_t kvc_sz;
    uint64_t kvc_off;

    // (running) result buffer
    // A dual-buffer (in-buf and out-buf), used to store (last and current) history table.
    // uint64_t irbuf_off;
    // uint64_t orbuf_off;
    // bool rbuf_reversed = false;
    uint64_t rbuf_sz;
    uint64_t rbuf_off;

    // RDMA buffer (#threads)
    char* buf;
    uint64_t buf_off;
    uint64_t buf_sz;  // per thread

    // record gpu rbuf allocated to each query
    std::unordered_map<int, rbuf_t*> rbuf_alloc_map;

    rbuf_t* rbufs;

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

        mem_gpu_sz = kvc_sz + buf_sz * num_agents + rbuf_sz * 2 * MAX_GPU_CHANNELS;

        // allocate memory and zeroing
        CUDA_ASSERT(cudaSetDevice(devid));
        CUDA_ASSERT(cudaMalloc(&mem_gpu, mem_gpu_sz));
        CUDA_ASSERT(cudaMemset(mem_gpu, 0, mem_gpu_sz));

        // kvcache
        kvc_off = 0;
        kvc = mem_gpu + kvc_off;

        // RDMA buffer
        buf_off = kvc_off + kvc_sz;
        buf = mem_gpu + buf_off;

        rbuf_off = buf_off + buf_sz * num_agents;
        char* rbuf_ptr = mem_gpu + rbuf_off;

        rbufs = new rbuf_t[MAX_GPU_CHANNELS];

        for (int i = 0; i < MAX_GPU_CHANNELS; ++i) {
            rbufs[i].irbuf = rbuf_ptr + i * 2 * rbuf_sz;
            rbufs[i].orbuf = rbufs[i].irbuf + rbuf_sz;
            rbufs[i].reversed = false;
            rbufs[i].used = false;
        }

        logstream(LOG_INFO) << "[GPUMem] devid: " << devid
                            << ", num_servers: " << num_servers
                            << ", num_agents: " << num_agents
                            << LOG_endl;
    }

    ~GPUMem() { CUDA_ASSERT(cudaFree(mem_gpu)); }

    inline char* address() { return mem_gpu; }
    inline uint64_t size() { return mem_gpu_sz; }

    // gpu cache
    inline char* kvcache() { return kvc; }
    inline uint64_t kvcache_offset() { return kvc_off; }
    inline uint64_t kvcache_size() { return kvc_sz; }

    rbuf_t* alloc_rbuf(int qid) {
        auto it = rbuf_alloc_map.find(qid);
        if (it != rbuf_alloc_map.end()) {
            return it->second;
        }

        rbuf_t* rbuf_ptr = nullptr;
    retry:
        for (int i = 0; i < Global::num_gpu_channels; ++i) {
            if (!rbufs[i].used) {
                rbuf_ptr = &rbufs[i];
                break;
            }
        }

        if (!rbuf_ptr) {
            timer::cpu_relax(20);
            goto retry;
        }

        rbuf_ptr->used = true;
        rbuf_alloc_map[qid] = rbuf_ptr;

        ASSERT(rbuf_alloc_map.size() <= Global::num_gpu_channels);
        return rbuf_ptr;
    }

    rbuf_t* get_allocated_rbuf(int qid) {
        auto it = rbuf_alloc_map.find(qid);
        if (it == rbuf_alloc_map.end()) {
            logstream(LOG_ERROR) << "[GPUMem] get_allocated_rbuf(): qid " << qid << " was not assigned a rbuf!" << LOG_endl;
            ASSERT(false);
        } else {
            return it->second;
        }
    }

    void free_rbuf(int qid) {
        ASSERT(rbuf_alloc_map.empty() == false);

        auto it = rbuf_alloc_map.find(qid);
        if (it == rbuf_alloc_map.end()) {
            logstream(LOG_ERROR) << "[GPUMem] free_rbuf(): qid " << qid << " was not assigned a rbuf!" << LOG_endl;
            ASSERT(false);
        }

        ASSERT(it->second->used == true);
        it->second->used = false;
        rbuf_alloc_map.erase(it);
    }

    inline uint64_t res_buf_size() { return rbuf_sz; }

    // RDMA buffer layout: header | type | body
    inline char* rdma_buf_hdr(int tid) { return buf + buf_sz * (tid % num_agents); }
    inline char* rdma_buf_body(int tid) { return rdma_buf_hdr(tid) + sizeof(uint64_t); }
    inline uint64_t rdma_buf_hdr_offset(int tid) { return buf_off + buf_sz * (tid % num_agents); }
    inline uint64_t rdma_buf_type_offset(int tid) { return buf_off + buf_sz * (tid % num_agents) + sizeof(uint64_t); }
    inline uint64_t rdma_buf_body_offset(int tid) { return buf_off + buf_sz * (tid % num_agents) + 2 * sizeof(uint64_t); }
    inline uint64_t rdma_buf_body_size() { return buf_sz - 2 * sizeof(uint64_t); }
};

}  // namespace wukong

#endif
