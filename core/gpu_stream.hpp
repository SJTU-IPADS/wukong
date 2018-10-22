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

#include <cuda_runtime.h>

class GPUStreamPool {

private:
    vector<cudaStream_t> streams;
    uint64_t rr_cnt;
    int num_streams;
    cudaStream_t split_stream;


public:
    GPUStreamPool(int num_streams) : num_streams(num_streams) {
        rr_cnt = 0;
        streams.reserve(num_streams);
        for (int i = 0; i < num_streams; ++i)
            cudaStreamCreate(&streams[i]);
        cudaStreamCreate(&split_stream);
    }

    ~GPUStreamPool() {
        for (auto s : streams)
            cudaStreamDestroy(s);
        cudaStreamDestroy(split_stream);
    }

    cudaStream_t get_stream() {
        // select Stream in a round-robin way
        int idx = (rr_cnt++) % num_streams;
        return streams[idx];
    }

    cudaStream_t get_stream(int pid) {
        assert(pid > 0);
        return streams[pid % num_streams];
    }

    cudaStream_t get_split_query_stream() { return split_stream; }

};

#endif
