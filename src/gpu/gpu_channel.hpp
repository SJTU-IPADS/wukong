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

#ifdef USE_GPU

#include "core/sparql/query.hpp"
#include "gpu_utils.hpp"

namespace wukong {

// The channel is an abstraction for execution resource for triple patterns
// which are processing on GPU. It contains temporary buffers and pattern parameters.
class GPUChannel {
private:
    struct Occupier {
        bool valid = false;
        int pattern_step = 0;
        // SPARQLQuery::PatternType pattern_type;
        SPARQLQuery *job = nullptr;

        Occupier() { }
    };

    // each channel associated with a CUDA Stream
    cudaStream_t stream;
    cudaEvent_t pipe_finish_ev;

    // Siyuan: result buffer不与Channel绑定，result buffer由GPUMem提供，Channel
    // 只在执行一条pattern的时候使用，而不是贯穿一个query所有的pattern执行。
public:

    uint32_t id;
    bool taken; // whether this channel is taken
    GPUErrorCode error_code = GPUErrorCode::NORMAL;
    void* error_info = nullptr;
    Occupier occupier;
    // GPU resources
    GPUEngineParam para;

    GPUChannel() { }

    void init(uint32_t id, vertex_t *vertices_d, edge_t *edges_d, uint64_t nkey_blks,
                   uint64_t nvalue_blks, uint64_t nbuckets_kblk, uint64_t nentries_vblk) {

        this->id = id;
        para.init(vertices_d, edges_d, nkey_blks, nvalue_blks, nbuckets_kblk, nentries_vblk);

        CUDA_ASSERT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_ASSERT(cudaEventCreateWithFlags(&pipe_finish_ev, cudaEventDisableTiming));

        reset();
    }

    inline cudaStream_t get_stream() {
        return stream;
    }

    inline cudaEvent_t finish_event() {
        return pipe_finish_ev;
    }

    void add_finish_event() {
        cudaEventRecord(pipe_finish_ev, stream);
    }

    cudaError_t poll_finish_event() {
        return cudaEventQuery(pipe_finish_ev);
    }

    cudaError_t flush() {
        ASSERT(taken);
        ASSERT(occupier.valid);
        return cudaEventSynchronize(pipe_finish_ev);
    }

    void set_occupier(SPARQLQuery *req_ptr) {
        this->taken = true;
        occupier.job = req_ptr;
        occupier.valid = true;
        occupier.pattern_step = req_ptr->pattern_step;
    }

    inline void reset() {
        taken = false;
        occupier.valid = false;
        occupier.pattern_step = 0;
    }
};

}  // namespace wukong

#endif  // USE_GPU
