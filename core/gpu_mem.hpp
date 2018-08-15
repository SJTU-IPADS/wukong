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

#include "global.hpp"
#include "rdma.hpp"
#include "unit.hpp"
#include "gpu_utils.hpp"
#include "type.hpp"
#include "gstore.hpp"

class GPUMem {
private:
    int devid;
    int num_servers;
    int num_agents; // #gpu_engine

    // The GPU memory layout: key/value cache | history in/out buffer(intermediate results) | rdma buffer | heap
    char *mem_gpu;
    uint64_t mem_gpu_sz;

    // cache on gpu
    char *kvc;
    uint64_t kvc_sz;
    uint64_t kvc_off;

    // history inbuf and outbuf, used to store (old and updated) history.
    char *inbuf;
    uint64_t inbuf_off;
    char *outbuf;
    uint64_t outbuf_off;
    uint64_t history_buf_sz;

    // rdma buffer
    char *buf; // #threads
    uint64_t buf_sz; // buffer size of single thread
    uint64_t buf_off;

public:
    GPUMem(int devid, int num_servers, int num_agents)
    :devid(devid), num_servers(num_servers), num_agents(num_agents) {
        kvc_sz = GiB2B(global_gpu_kvcache_size_gb);
        history_buf_sz = global_gpu_max_element * sizeof(sid_t);
        if (RDMA::get_rdma().has_rdma()) {
            // only used by RDMA device
            buf_sz = MiB2B(global_gpu_rdma_buf_size_mb);
        } else {
            buf_sz = 0;
        }
        mem_gpu_sz = kvc_sz + history_buf_sz * 2 + buf_sz * num_agents;

        CUDA_ASSERT(cudaSetDevice(devid));
        CUDA_ASSERT(cudaMalloc(&mem_gpu, mem_gpu_sz));
        CUDA_ASSERT(cudaMemset(mem_gpu, 0, mem_gpu_sz));

        kvc_off = 0;
        kvc = mem_gpu + kvc_off;

        inbuf_off = kvc_off + kvc_sz;
        inbuf = mem_gpu + inbuf_off;

        outbuf_off = inbuf_off + history_buf_sz;
        outbuf = mem_gpu + outbuf_off;

        buf_off = outbuf_off + history_buf_sz;
        buf = mem_gpu + buf_off;

        logstream(LOG_INFO) << "GPUMem: devid: " << devid << ", num_servers: " << num_servers << ", num_agents: " << num_agents << LOG_endl;
    }

    ~GPUMem() { CUDA_ASSERT(cudaFree(mem_gpu)); }

    inline char *address() { return mem_gpu; }
    inline uint64_t size() { return mem_gpu_sz; }

    // gpu cache
    inline char *kvcache() { return kvc; }
    inline uint64_t kvcache_size() { return kvc_sz; }
    inline uint64_t kvcache_offset() { return kvc_off; }

    // history_inbuf
    inline char *history_inbuf() { return inbuf; }
    inline uint64_t history_inbuf_size() { return history_buf_sz; }
    inline uint64_t history_inbuf_offset() { return inbuf_off; }

    // history_outbuf
    inline char *history_outbuf() { return outbuf; }
    inline uint64_t history_outbuf_size() { return history_buf_sz; }
    inline uint64_t history_outbuf_offset() { return outbuf_off; }

    // rdma buffer layout: header | type | body
    inline char *rdma_buf_hdr(int tid) { return buf + buf_sz * (tid % num_agents); }
    inline char *rdma_buf_type(int tid) { return buf + buf_sz * (tid % num_agents) + sizeof(uint64_t); }
    inline char *rdma_buf_body(int tid) { return buf + buf_sz * (tid % num_agents) + 2 * sizeof(uint64_t); }
    inline uint64_t rdma_buf_body_size() { return buf_sz - 2 * sizeof(uint64_t); }
    inline uint64_t rdma_buf_hdr_offset(int tid) { return buf_off + buf_sz * (tid % num_agents); }
    inline uint64_t rdma_buf_type_offset(int tid) { return buf_off + buf_sz * (tid % num_agents) + sizeof(uint64_t); }
    inline uint64_t rdma_buf_body_offset(int tid) { return buf_off + buf_sz * (tid % num_agents) + 2 * sizeof(uint64_t); }
};

#endif
