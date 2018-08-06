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

#include "unit.hpp"
#include "gpu_utils.hpp"
#include "type.hpp"

class GPUMem {
private:
    int devid;
    int num_servers;
    int num_agents; // #gpu_engine

    // The GPU memory layout: key/value cache | history in/out buffer(intermediate results) | rdma buffer | heap
    char *mem_gpu;
    uint64_t mem_gpu_sz;

    // cache on gpu
    char *kvs;
    uint64_t kvs_sz;
    uint64_t kvs_off;

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
        kvs_sz = GiB2B(global_gpu_kvcache_size_gb);
        history_buf_sz = global_gpu_max_element * sizeof(sid_t);
        if (RDMA::get_rdma().has_rdma()) {
            // only used by RDMA device
            buf_sz = MiB2B(global_gpu_rdma_buf_size_mb);
        } else {
            buf_sz = 0;
        }
        mem_gpu_sz = kvs_sz + history_buf_sz * 2 + buf_sz * num_agents;

        CUDA_ASSERT(cudaSetDevice(devid));
        CUDA_ASSERT(cudaMalloc(&mem_gpu, mem_gpu_sz));
        CUDA_ASSERT(cudaMemset(mem_gpu, 0, mem_gpu_sz));

        kvs_off = 0;
        kvs = mem_gpu + kvs_off;

        inbuf_off = kvs_off + kvs_sz;
        inbuf = mem_gpu + inbuf_off;

        outbuf_off = inbuf_off + history_buf_sz;
        outbuf = mem_gpu + outbuf_off;

        buf_off = outbuf_off + history_buf_sz;
        buf = mem_gpu + buf_off;

        logstream(LOG_INFO) << "GPUMem: devid: " << devid << ", num_servers: " << num_servers << ", num_agents: " << num_agents << LOG_endl;
    }

    ~GPUMem() { CUDA_ASSERT(cudaFree(mem_gpu)); }

    inline char *memory() { return mem_gpu; }
    inline uint64_t memory_size() { return mem_gpu_sz; }

    // kvstore
    inline char *kvstore() { return kvs; }
    inline uint64_t kvstore_size() { return kvs_sz; }
    inline uint64_t kvstore_offset() { return kvs_off; }

    // history_inbuf
    inline char *history_inbuf() { return inbuf; }
    inline uint64_t history_inbuf_size() { return history_buf_sz; }
    inline uint64_t history_inbuf_offset() { return inbuf_off; }

    // history_outbuf
    inline char *history_outbuf() { return outbuf; }
    inline uint64_t history_outbuf_size() { return history_buf_sz; }
    inline uint64_t history_outbuf_offset() { return outbuf_off; }

    // buffer
    inline char *buffer(int tid) { return buf + buf_sz * (tid % num_agents); }
    inline uint64_t buffer_size() { return buf_sz; }
    inline uint64_t buffer_offset(int tid) { return buf_off + buf_sz * (tid % num_agents); }

};
#endif
