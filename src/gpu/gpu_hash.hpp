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
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cmath>
#include <map>
#include <vector>

#include "core/common/global.hpp"
#include "core/store/segment_meta.hpp"
#include "core/store/vertex.hpp"

// utils
#include "gpu_utils.hpp"
#include "utils/assertion.hpp"
#include "utils/unit.hpp"

namespace wukong {

struct GPUMem;

#define WK_CLINE 64

struct pattern_info_t {
    int start_vid;
    int start_var2col;

    int end_vid;
    int end_var2col;

    int dir;
    int pid;
    int col_num = 0;
    int row_num = 0;

    int rbuf_start = 0;  // start offset in (sid_t*)rbuf for the pattern
    int rbuf_start_row = 0;
    int rbuf_max_row = 0;  // max number of rows for the pattern

    // pointer to corresponding segment on gpu
    rdf_seg_meta_t* segment_meta_dptr = nullptr;

    // pointer to mapping table buffer on gpu
    uint64_t* key_mapping_dptr = nullptr;
    uint64_t* value_mapping_dptr = nullptr;
} __attribute__((aligned(WK_CLINE)));

struct updated_pattern_info_t {
    size_t rbuf_start = 0;
    size_t rbuf_start_row = 0;
    size_t rbuf_max_row = 0;
    int col_num = 0;
} __attribute__((aligned(WK_CLINE)));

struct GPUEngineParam {
    uint64_t cache_nkey_blks = 0;
    uint64_t cache_nvalue_blks = 0;

    // pattern info of single pattern
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

    // data structures used on GPU
    struct {
        ikey_t* d_key_list = nullptr;
        vertex_t* vertex_gaddr = nullptr;
        edge_t* edge_gaddr = nullptr;

        uint64_t* d_slot_id_list = nullptr;
        uint64_t* d_vertex_mapping = nullptr;
        uint64_t* d_edge_mapping = nullptr;

        uint64_t vertex_blk_sz;
        uint64_t edge_blk_sz;

        int* d_prefix_sum_list = nullptr;
        int* d_edge_size_list = nullptr;

        uint64_t* d_offset_list = nullptr;
        uint64_t* d_edge_off_list = nullptr;

        sid_t* d_in_rbuf;
        sid_t* d_out_rbuf;

        // mapping table buffer on gpu
        // can only contain #key_blocks and #value_blocks mapping entries
        uint64_t* d_kblk_mapping_table = nullptr;
        size_t total_key_blks = 0;
        std::map<segid_t, uint64_t*> kblk_mapping_cache;

        uint64_t* d_vblk_mapping_table = nullptr;
        size_t total_value_blks = 0;
        std::map<segid_t, uint64_t*> vblk_mapping_cache;

        // array of segment metadata
        rdf_seg_meta_t* segment_metas_d;

        // array of all patterns in the combined query
        pattern_info_t* pattern_infos_d;
        updated_pattern_info_t* updated_patt_infos_d;
        int num_patterns = 0;
    } gpu;

    GPUEngineParam() {}

    GPUEngineParam(vertex_t* d_vertices, edge_t* d_edges, uint64_t nkey_blks,
                   uint64_t nvalue_blks, uint64_t nbuckets, uint64_t nentries) {
        init(d_vertices, d_edges, nkey_blks, nvalue_blks, nbuckets, nentries);
    }

    void init(vertex_t* d_vertices, edge_t* d_edges, uint64_t nkey_blks,
              uint64_t nvalue_blks, uint64_t nbuckets, uint64_t nentries) {
        cache_nkey_blks = nkey_blks;
        cache_nvalue_blks = nvalue_blks;

        gpu.vertex_gaddr = d_vertices;
        gpu.edge_gaddr = d_edges;
        gpu.vertex_blk_sz = nbuckets;
        gpu.edge_blk_sz = nentries;

        // NOTE: these temporary buffers can be allocated before execute one pattern dynamically
        CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&gpu.d_key_list), MiB2B(Global::gpu_rbuf_size_mb)));

        CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&gpu.d_slot_id_list), MiB2B(Global::gpu_rbuf_size_mb)));
        CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&gpu.d_prefix_sum_list), MiB2B(Global::gpu_rbuf_size_mb)));
        CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&gpu.d_edge_size_list), MiB2B(Global::gpu_rbuf_size_mb)));
        CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&gpu.d_offset_list), MiB2B(Global::gpu_rbuf_size_mb)));
        CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&gpu.d_edge_off_list), MiB2B(Global::gpu_rbuf_size_mb)));

        // NOTE: these buffers has long life cycle, thus should be allocated statically
        CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&gpu.d_kblk_mapping_table), sizeof(uint64_t) * nkey_blks));
        CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&gpu.d_vblk_mapping_table), sizeof(uint64_t) * nvalue_blks));

        CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&gpu.segment_metas_d), sizeof(rdf_seg_meta_t) * Global::pattern_combine_window));
        CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&gpu.pattern_infos_d), sizeof(pattern_info_t) * Global::pattern_combine_window));
        CUDA_ASSERT(cudaMalloc(reinterpret_cast<void**>(&gpu.updated_patt_infos_d), sizeof(updated_pattern_info_t) * Global::pattern_combine_window));
    }

    void reset() {
        gpu.total_key_blks = 0;
        gpu.total_value_blks = 0;
        gpu.num_patterns = 0;
        gpu.kblk_mapping_cache.clear();
        gpu.vblk_mapping_cache.clear();
    }

    void load_segment_mappings(const std::vector<uint64_t>& vertex_mapping,
                               const std::vector<uint64_t>& edge_mapping,
                               const rdf_seg_meta_t& seg, cudaStream_t stream = 0) {
        CUDA_ASSERT(cudaMemcpyAsync(gpu.d_kblk_mapping_table,
                                    &(vertex_mapping[0]),
                                    sizeof(uint64_t) * seg.num_key_blks,
                                    cudaMemcpyHostToDevice,
                                    stream));

        CUDA_ASSERT(cudaMemcpyAsync(gpu.d_vblk_mapping_table,
                                    &(edge_mapping[0]),
                                    sizeof(uint64_t) * seg.num_value_blks,
                                    cudaMemcpyHostToDevice,
                                    stream));
    }

    uint64_t* load_key_mappings(const std::vector<uint64_t>& key_mapping,
                                const segid_t segid,
                                const rdf_seg_meta_t& seg,
                                cudaStream_t stream = 0) {
        if (gpu.kblk_mapping_cache.find(segid) != gpu.kblk_mapping_cache.end()) {
            return gpu.kblk_mapping_cache[segid];
        } else {
            uint64_t* dptr = gpu.d_kblk_mapping_table + gpu.total_key_blks;
            CUDA_ASSERT(cudaMemcpyAsync(dptr,
                                        &(key_mapping[0]),
                                        sizeof(uint64_t) * seg.num_key_blks,
                                        cudaMemcpyHostToDevice,
                                        stream));

            gpu.total_key_blks += seg.num_key_blks;
            ASSERT(gpu.total_key_blks <= cache_nkey_blks);
            gpu.kblk_mapping_cache[segid] = dptr;
            return dptr;
        }
    }

    uint64_t* load_value_mappings(const std::vector<uint64_t>& value_mapping,
                                  const segid_t segid,
                                  const rdf_seg_meta_t& seg,
                                  cudaStream_t stream = 0) {
        if (gpu.vblk_mapping_cache.find(segid) != gpu.vblk_mapping_cache.end()) {
            return gpu.vblk_mapping_cache[segid];
        } else {
            uint64_t* dptr = gpu.d_vblk_mapping_table + gpu.total_value_blks;
            CUDA_ASSERT(cudaMemcpyAsync(dptr,
                                        &(value_mapping[0]),
                                        sizeof(uint64_t) * seg.num_value_blks,
                                        cudaMemcpyHostToDevice,
                                        stream));

            gpu.total_value_blks += seg.num_value_blks;
            ASSERT(gpu.total_value_blks <= cache_nvalue_blks);
            gpu.vblk_mapping_cache[segid] = dptr;
            return dptr;
        }
    }

    void load_segment_meta(const rdf_seg_meta_t& seg_meta,
                           cudaStream_t stream = 0) {
        CUDA_ASSERT(cudaMemcpyAsync(gpu.segment_metas_d,
                                    &seg_meta,
                                    sizeof(rdf_seg_meta_t),
                                    cudaMemcpyHostToDevice,
                                    stream));
    }

    void load_pattern_meta(rdf_seg_meta_t& seg_meta,
                           pattern_info_t& pattern,
                           cudaStream_t stream = 0) {
        rdf_seg_meta_t* segmeta_dpr = gpu.segment_metas_d + gpu.num_patterns;
        CUDA_ASSERT(cudaMemcpyAsync(segmeta_dpr,
                                    &seg_meta,
                                    sizeof(rdf_seg_meta_t),
                                    cudaMemcpyHostToDevice,
                                    stream));

        pattern.segment_meta_dptr = segmeta_dpr;

        // finally copy pattern_into to gpu
        CUDA_ASSERT(cudaMemcpyAsync(gpu.pattern_infos_d + gpu.num_patterns,
                                    &pattern,
                                    sizeof(pattern_info_t),
                                    cudaMemcpyHostToDevice,
                                    stream));

        // increase counter for pattern_info
        ++gpu.num_patterns;
        ASSERT(gpu.num_patterns <= Global::pattern_combine_window);
    }

    void load_pattern_metas(std::vector<rdf_seg_meta_t>& seg_metas,
                            std::vector<pattern_info_t>& patterns,
                            cudaStream_t stream = 0) {
        ASSERT_EQ(seg_metas.size(), patterns.size());
        CUDA_ASSERT(cudaMemcpyAsync(gpu.segment_metas_d,
                                    seg_metas.data(),
                                    sizeof(rdf_seg_meta_t) * seg_metas.size(),
                                    cudaMemcpyHostToDevice,
                                    stream));

        for (int i = 0; i < patterns.size(); i++) {
            patterns[i].segment_meta_dptr = gpu.segment_metas_d + i;
        }

        // finally copy pattern_into to gpu
        CUDA_ASSERT(cudaMemcpyAsync(gpu.pattern_infos_d,
                                    patterns.data(),
                                    sizeof(pattern_info_t) * patterns.size(),
                                    cudaMemcpyHostToDevice,
                                    stream));

        // increase counter for pattern_info
        gpu.num_patterns += patterns.size();
        ASSERT(gpu.num_patterns <= Global::pattern_combine_window);
    }

    void set_result_bufs(sid_t* d_in_rbuf, sid_t* d_out_rbuf) {
        gpu.d_in_rbuf = d_in_rbuf;
        gpu.d_out_rbuf = d_out_rbuf;
    }
} __attribute__((aligned(WK_CLINE)));

void gpu_split_giant_query(GPUEngineParam& param, int row_num, int col_num, int num_jobs, int query_size,
                           std::vector<int>& buf_offs, cudaStream_t stream = 0);
void gpu_shuffle_result_buf(GPUEngineParam& param, int num_servers, std::vector<int>& buf_sizes,
                            std::vector<int>& buf_heads, cudaStream_t stream = 0);
void gpu_split_result_buf(GPUEngineParam& param, int num_servers, cudaStream_t stream = 0);
void gpu_calc_prefix_sum(GPUEngineParam& param, cudaStream_t stream = 0);

void gpu_generate_key_list(GPUEngineParam& param, cudaStream_t stream = 0);
void gpu_get_slot_id_list(GPUEngineParam& param, cudaStream_t stream = 0);

void gpu_get_edge_list(GPUEngineParam& param, cudaStream_t stream = 0);
void gpu_get_edge_list_k2k(GPUEngineParam& param, cudaStream_t stream = 0);
void gpu_get_edge_list_k2c(GPUEngineParam& param, cudaStream_t stream = 0);

int gpu_update_result_buf_i2u(GPUEngineParam& param, cudaStream_t stream = 0);
void gpu_update_result_buf_k2k(GPUEngineParam& param, cudaStream_t stream = 0);
void gpu_update_result_buf_k2u(GPUEngineParam& param, cudaStream_t stream = 0);
void gpu_update_result_buf_k2c(GPUEngineParam& param, cudaStream_t stream = 0);

/* multi-query combining */
void gpu_generate_key_list_combined(GPUEngineParam& param, cudaStream_t stream = 0);
void gpu_get_slot_id_list_combined(GPUEngineParam& param, cudaStream_t stream = 0);
void gpu_get_edge_list_combined(GPUEngineParam& param, cudaStream_t stream = 0);
void gpu_get_edge_list_k2c_combined(GPUEngineParam& param, cudaStream_t stream = 0);
void gpu_get_edge_list_k2k_combined(GPUEngineParam& param, cudaStream_t stream = 0);

void gpu_update_result_buf_k2u_combined(GPUEngineParam& param, cudaStream_t stream = 0);
void gpu_update_result_buf_k2c_combined(GPUEngineParam& param, cudaStream_t stream = 0);

}  // namespace wukong

#endif  // end of USE_GPU
