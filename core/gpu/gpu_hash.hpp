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

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cmath>

#include "store/meta.hpp"
#include "store/vertex.hpp"

// utils
#include "global.hpp"
#include "gpu.hpp"
#include "unit.hpp"

struct GPUMem;

struct GPUEngineParam {
    struct {
        ssid_t start_vid = 0;
        ssid_t pid = 0;
        int dir = -1;
        ssid_t end_vid = 0;
        int col_num;
        int row_num = -1;
        uint64_t segment_edge_start = 0;

        int var2col_start = -1;
        int var2col_end = -1;

    } query;

    struct {
        ikey_t *d_key_list = nullptr;
        vertex_t *vertex_gaddr = nullptr;
        edge_t *edge_gaddr = nullptr;

        uint64_t *d_slot_id_list = nullptr;
        uint64_t *d_vertex_mapping = nullptr;
        uint64_t *d_edge_mapping = nullptr;

        uint64_t vertex_blk_sz;
        uint64_t edge_blk_sz;

        int *d_prefix_sum_list = nullptr;
        int *d_edge_size_list = nullptr;

        uint64_t *d_offset_list = nullptr;
        uint64_t *d_edge_off_list = nullptr;

        sid_t *d_in_rbuf;
        sid_t *d_out_rbuf;

        rdf_seg_meta_t *d_segment_meta;
    } gpu;

    GPUEngineParam(vertex_t *d_vertices, edge_t *d_edges, uint64_t nkey_blks,
                   uint64_t nvalue_blks, uint64_t nbuckets, uint64_t nentries) {
        gpu.vertex_gaddr = d_vertices;
        gpu.edge_gaddr = d_edges;
        gpu.vertex_blk_sz = nbuckets;
        gpu.edge_blk_sz = nentries;

        CUDA_ASSERT(cudaMalloc( (void **)&gpu.d_key_list, MiB2B(Global::gpu_rbuf_size_mb) ));

        CUDA_ASSERT(cudaMalloc( (void **)&gpu.d_slot_id_list, MiB2B(Global::gpu_rbuf_size_mb) ));
        CUDA_ASSERT(cudaMalloc( (void **)&gpu.d_prefix_sum_list, MiB2B(Global::gpu_rbuf_size_mb) ));
        CUDA_ASSERT(cudaMalloc( (void **)&gpu.d_edge_size_list, MiB2B(Global::gpu_rbuf_size_mb) ));
        CUDA_ASSERT(cudaMalloc( (void **)&gpu.d_offset_list, MiB2B(Global::gpu_rbuf_size_mb) ));
        CUDA_ASSERT(cudaMalloc( (void **)&gpu.d_edge_off_list, MiB2B(Global::gpu_rbuf_size_mb) ));

        CUDA_ASSERT(cudaMalloc( (void **)&gpu.d_vertex_mapping, sizeof(uint64_t) * nkey_blks));
        CUDA_ASSERT(cudaMalloc( (void **)&gpu.d_edge_mapping, sizeof(uint64_t) * nvalue_blks));
        CUDA_ASSERT(cudaMalloc( (void **)&gpu.d_segment_meta, sizeof(rdf_seg_meta_t) ));
    }


    void load_segment_mappings(const std::vector<uint64_t> &vertex_mapping,
                               const std::vector<uint64_t> &edge_mapping,
                               const rdf_seg_meta_t &seg, cudaStream_t stream = 0) {

        CUDA_ASSERT(cudaMemcpyAsync(gpu.d_vertex_mapping,
                                    &(vertex_mapping[0]),
                                    sizeof(uint64_t) * seg.num_key_blks,
                                    cudaMemcpyHostToDevice,
                                    stream));

        CUDA_ASSERT(cudaMemcpyAsync(gpu.d_edge_mapping,
                                    &(edge_mapping[0]),
                                    sizeof(uint64_t) * seg.num_value_blks,
                                    cudaMemcpyHostToDevice,
                                    stream));
    }

    void load_segment_meta(rdf_seg_meta_t seg_meta, cudaStream_t stream = 0) {
        CUDA_ASSERT(cudaMemcpyAsync(gpu.d_segment_meta,
                                    &seg_meta,
                                    sizeof(seg_meta),
                                    cudaMemcpyHostToDevice,
                                    stream));
    }

    void set_result_bufs(char *d_in_rbuf, char *d_out_rbuf) {
        gpu.d_in_rbuf = (sid_t *) d_in_rbuf;
        gpu.d_out_rbuf = (sid_t *) d_out_rbuf;
    }

};

void gpu_shuffle_result_buf(GPUEngineParam &param, int num_servers, std::vector<int> &buf_sizes,
                            std::vector<int> &buf_heads, cudaStream_t stream = 0);
void gpu_split_result_buf(GPUEngineParam &param, int num_servers, cudaStream_t stream = 0);
void gpu_calc_prefix_sum(GPUEngineParam &param, cudaStream_t stream = 0);

void gpu_generate_key_list_k2u(GPUEngineParam &param, cudaStream_t stream = 0);
void gpu_get_slot_id_list(GPUEngineParam &param, cudaStream_t stream = 0);

void gpu_get_edge_list(GPUEngineParam &param, cudaStream_t stream = 0);
void gpu_get_edge_list_k2k(GPUEngineParam &param, cudaStream_t stream = 0);
void gpu_get_edge_list_k2c(GPUEngineParam &param, cudaStream_t stream = 0);

int gpu_update_result_buf_i2u(GPUEngineParam &param, cudaStream_t stream = 0);
int gpu_update_result_buf_k2k(GPUEngineParam &param, cudaStream_t stream = 0);
int gpu_update_result_buf_k2u(GPUEngineParam &param, cudaStream_t stream = 0);
int gpu_update_result_buf_k2c(GPUEngineParam &param, cudaStream_t stream = 0);


#endif // end of USE_GPU
