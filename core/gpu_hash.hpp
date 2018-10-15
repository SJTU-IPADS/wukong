#pragma once

#ifdef USE_GPU

// #include <cuda_runtime.h>
// #include <vector>

// #include <stdio.h>
// #include <thrust/functional.h>
#include <thrust/device_ptr.h>
// Siyuan: useless device_vector for Wukong+G
// #include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cmath>

#include "gpu_utils.hpp"
#include "unit.hpp"
#include "rdf_meta.hpp"
#include "vertex.hpp"

struct GPUMem;
extern int global_gpu_rbuf_size_mb;

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
        vertex_t *d_vertex_list = nullptr;
        vertex_t* d_vertex_addr = nullptr;
        edge_t *d_edge_addr = nullptr;

        uint64_t* d_slot_id_list = nullptr;

        uint64_t* d_vertex_mapping = nullptr;
        uint64_t* d_edge_mapping = nullptr;

        uint64_t vertex_block_sz;
        uint64_t edge_block_sz;

        int *d_prefix_sum_list = nullptr;
        int *d_edge_size_list = nullptr;

        uint64_t *d_offset_list = nullptr;
        uint64_t *d_edge_off_list = nullptr;
        uint64_t *d_bucket_id_list = nullptr;

        sid_t* d_in_rbuf;
        sid_t* d_out_rbuf;

        rdf_segment_meta_t *d_segment_meta;
    } gpu;

    GPUEngineParam(vertex_t *d_vertices, edge_t *d_edges, uint64_t nkey_blks, uint64_t nvalue_blks) {
        gpu.d_vertex_addr = d_vertices;
        gpu.d_edge_addr = d_edges;

        CUDA_ASSERT(cudaMalloc( (void**)&gpu.d_key_list, MiB2B(global_gpu_rbuf_size_mb) ));
        // for debug
        CUDA_ASSERT(cudaMalloc( (void**)&gpu.d_vertex_list, MiB2B(global_gpu_rbuf_size_mb) ));
        CUDA_ASSERT(cudaMemset((void*)gpu.d_vertex_list, 0, MiB2B(global_gpu_rbuf_size_mb)));
        CUDA_ASSERT(cudaMalloc( (void**)&gpu.d_bucket_id_list, MiB2B(global_gpu_rbuf_size_mb) ));
        // end for debug


        CUDA_ASSERT(cudaMalloc( (void**)&gpu.d_slot_id_list, MiB2B(global_gpu_rbuf_size_mb) ));
        CUDA_ASSERT(cudaMalloc( (void**)&gpu.d_prefix_sum_list, MiB2B(global_gpu_rbuf_size_mb) ));
        CUDA_ASSERT(cudaMalloc( (void**)&gpu.d_edge_size_list, MiB2B(global_gpu_rbuf_size_mb) ));
        CUDA_ASSERT(cudaMalloc( (void**)&gpu.d_offset_list, MiB2B(global_gpu_rbuf_size_mb) ));
        CUDA_ASSERT(cudaMalloc( (void**)&gpu.d_edge_off_list, MiB2B(global_gpu_rbuf_size_mb) ));

        CUDA_ASSERT(cudaMalloc( (void**)&gpu.d_vertex_mapping, sizeof(uint64_t) * nkey_blks));
        CUDA_ASSERT(cudaMalloc( (void**)&gpu.d_edge_mapping, sizeof(uint64_t) * nvalue_blks));
        CUDA_ASSERT(cudaMalloc( (void**)&gpu.d_segment_meta, sizeof(rdf_segment_meta_t) ));
    }


    void load_segment_mappings(const std::vector<uint64_t>& vertex_mapping,
            const std::vector<uint64_t>& edge_mapping, const rdf_segment_meta_t &seg, cudaStream_t stream = 0) {

        CUDA_ASSERT(cudaMemcpy(gpu.d_vertex_mapping,
                          &(vertex_mapping[0]),
                          sizeof(uint64_t) * seg.num_key_blocks(),
                          cudaMemcpyHostToDevice));
                          // stream));

        CUDA_ASSERT(cudaMemcpy(gpu.d_edge_mapping,
                          &(edge_mapping[0]),
                          sizeof(uint64_t) * seg.num_value_blocks(),
                          cudaMemcpyHostToDevice));
                          // stream));
    }

    void load_segment_meta(rdf_segment_meta_t seg_meta, cudaStream_t stream = 0) {
        CUDA_ASSERT(cudaMemcpy(gpu.d_segment_meta,
                          &seg_meta,
                          sizeof(seg_meta),
                          cudaMemcpyHostToDevice));
                          // stream));
    }

    void set_result_bufs(char *d_in_rbuf, char *d_out_rbuf) {
        gpu.d_in_rbuf = (sid_t*) d_in_rbuf;
        gpu.d_out_rbuf = (sid_t*) d_out_rbuf;
    }

    void set_cache_param(uint64_t block_num_buckets, uint64_t block_num_edges ) {
        gpu.vertex_block_sz = block_num_buckets;
        gpu.edge_block_sz = block_num_edges;
    }

};

void gpu_shuffle_result_buf(GPUEngineParam& param, int num_servers, std::vector<int>& buf_sizes,
        std::vector<int>& buf_heads, cudaStream_t stream = 0);
void gpu_split_result_buf(GPUEngineParam &param, int num_servers, cudaStream_t stream = 0);
void gpu_calc_prefix_sum(GPUEngineParam& param, cudaStream_t stream = 0);

void gpu_generate_key_list_k2u(GPUEngineParam &param, cudaStream_t stream = 0);
void gpu_get_slot_id_list(GPUEngineParam &param, cudaStream_t stream = 0);

void gpu_get_edge_list(GPUEngineParam &param, cudaStream_t stream = 0);
void gpu_get_edge_list_k2k(GPUEngineParam &param, cudaStream_t stream = 0);
void gpu_get_edge_list_k2c(GPUEngineParam &param, cudaStream_t stream = 0);

int gpu_update_result_buf_i2u(GPUEngineParam& param, cudaStream_t stream = 0);
int gpu_update_result_buf_k2k(GPUEngineParam& param, cudaStream_t stream = 0);
int gpu_update_result_buf_k2u(GPUEngineParam& param, cudaStream_t stream = 0);
int gpu_update_result_buf_k2c(GPUEngineParam& param, cudaStream_t stream = 0);
int gpu_update_result_buf_i2u(GPUEngineParam& param, cudaStream_t stream);



#endif // end of USE_GPU
