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
//

struct ikey_t;
struct vertex_t;
struct edge_t;
struct GPUMem;
// struct rdf_segment_meta_t;
extern int global_gpu_rbuf_size_mb;

struct GPUEngineParam {
    struct {
        ssid_t start_vid = 0;
        ssid_t pid = 0;
        int dir = -1;
        int end_vid = 0;
        int col_num;
        int row_num = -1;
        uint64_t segment_edge_start = 0;
    } query;

    struct {
        // cudaStream_t stream;  // end of generate_key_list_i2u

        // device_vector<ikey_t> key_dv;
        ikey_t *d_key_list = nullptr;
        vertex_t* d_vertex_addr = nullptr;
        edge_t *d_edge_addr = nullptr;

        // device_vector<uint64_t> slots_dv;
        uint64_t* d_slot_id_list = nullptr;
        // pred_meta_t* d_pred_metas = nullptr;

        // device_vector<uint64_t> vertex_mapping_dv;
        // uint64_t*  d_vertex_headers = nullptr;
        uint64_t* d_vertex_mapping = nullptr;
        // device_vector<uint64_t> edge_mapping_dv;
        // uint64_t* d_edge_headers = nullptr;
        uint64_t* d_edge_mapping = nullptr;

        // Siyuan: 这2个找gcache拿
        uint64_t vertex_block_sz;
        uint64_t edge_block_sz;

        // cudaStream_t stream_id; // end of get_slot_id_list(common)
        // uint64_t *slot_id_list;
        // void *d_vertex_addr;
        // device_vector<int> index_dv;
        int *d_prefix_sum_list = nullptr;
        // device_vector<int> index_dv_mirror;
        int *d_edge_size_list = nullptr;

        // device_vector<uint64_t> offset_dv;
        uint64_t *d_offset_list = nullptr;
        // device_vector<uint64_t> edge_off_dv;
        uint64_t *d_edge_off_list = nullptr;
        // device_ptr<sid_t*> in_rbuf_dp;
        // device_ptr<sid_t*> out_rbuf_dp;

        int* d_in_rbuf;
        int* d_out_rbuf;

        rdf_segment_meta_t *d_segment_meta;
    } gpu;

    GPUEngineParam(vertex_t *d_vertices, edge_t *d_edges, uint64_t nkey_blks, uint64_t nvalue_blks) {
        // TODO allocate GPU memory for devices
        gpu.d_vertex_addr = d_vertices;
        gpu.d_edge_addr = d_edges;

        CUDA_ASSERT(cudaMalloc( (void**)&gpu.d_in_rbuf, MiB2B(global_gpu_rbuf_size_mb) ));
        CUDA_ASSERT(cudaMalloc( (void**)&gpu.d_out_rbuf, MiB2B(global_gpu_rbuf_size_mb) ));

        CUDA_ASSERT(cudaMalloc( (void**)&gpu.d_key_list, MiB2B(global_gpu_rbuf_size_mb) ));
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

        printf("load_segment_mappings: segment: #key_blks=%d, #value_blks=%d\n", seg.num_key_blocks(), seg.num_value_blocks());


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
        gpu.d_in_rbuf = (int*) d_in_rbuf;
        gpu.d_out_rbuf = (int*) d_out_rbuf;
    }

    void set_cache_param(uint64_t block_num_buckets, uint64_t block_num_edges ) {
        gpu.vertex_block_sz = block_num_buckets;
        gpu.edge_block_sz = block_num_edges;
    }

};


void gpu_lookup_hashtable_k2u(GPUEngineParam& param, cudaStream_t stream = 0);
void gpu_shuffle_result_buf(GPUEngineParam& param, std::vector<int>& buf_sizes, cudaStream_t stream = 0);
void gpu_split_result_buf(GPUEngineParam &param, int num_servers, cudaStream_t stream = 0);
void gpu_calc_prefix_sum(GPUEngineParam& param, cudaStream_t stream = 0);
void gpu_get_slot_id_list(GPUEngineParam &param, cudaStream_t stream = 0);


void gpu_get_edge_list(GPUEngineParam &param, cudaStream_t stream = 0);
void gpu_get_edge_list_k2k(GPUEngineParam &param, cudaStream_t stream = 0);
void gpu_get_edge_list_k2c(GPUEngineParam &param, cudaStream_t stream = 0);
int gpu_update_result_buf_i2u(GPUEngineParam& param, cudaStream_t stream = 0);
int gpu_update_result_buf_k2k(GPUEngineParam& param, cudaStream_t stream = 0);
int gpu_update_result_buf_k2u(GPUEngineParam& param, cudaStream_t stream = 0);
int gpu_update_result_buf_i2u(GPUEngineParam& param, cudaStream_t stream);



/*****  Following is old interfaces *****/

/* 
 * TODO: Do we need these two functions?
 * void generate_key_list_i2u(int *result_table,
 *                        int index_vertex,
 *                        int direction,
 *                        void *key_list,
 *                        int query_size,
 *                        cudaStream_t stream_id=0);
 * 
 * void generate_key_list_k2u(int *result_table,
 *                        int start,
 *                        int direction,
 *                        int predict,
 *                        int col_num,
 *                        void *key_list,
 *                        int query_size,
 *                        cudaStream_t stream_id=0);
 *  */





// void get_slot_id_list(void* d_vertex_addr,
                 // void* d_key_list,
                 // uint64_t* d_slot_id_list,
                 // // pred_meta_t* pred_metas,
                 // uint64_t* vertex_headers,
                 // uint64_t pred_vertex_shard_size,
                 // int query_size,
                 // cudaStream_t stream_id=0);

// void get_edge_list(uint64_t *slot_id_list,
                    // void *d_vertex_addr,
                    // int *index_list,
                    // int *index_list_mirror,
                    // uint64_t *ptr_list,
                    // uint64_t pred_orin_edge_start,
                    // uint64_t* edge_headers,
                    // uint64_t pred_edge_shard_size,
                    // int query_size,
                    // cudaStream_t stream_id=0);

// void get_edge_list_k2k(uint64_t *slot_id_list,
                    // void *d_vertex_addr,
                    // int *index_list,
                    // int *index_list_mirror,
                    // uint64_t *ptr_list,
                    // int query_size,
                    // void *edge_addr,
                    // int *result_table,
                    // int col_num,
                    // int end,
                    // uint64_t pred_orin_edge_start,
                    // uint64_t* edge_headers,
                    // uint64_t pred_edge_shard_size,
                    // cudaStream_t stream_id=0);

// void get_edge_list_k2c(uint64_t *slot_id_list,
                    // void *d_vertex_addr,
                    // int *index_list,
                    // int *index_list_mirror,
                    // uint64_t *ptr_list,
                    // int query_size,
                    // void *edge_addr,
                    // int end,
                    // uint64_t pred_orin_edge_start,
                    // uint64_t* edge_headers,
                    // uint64_t pred_edge_shard_size,
                    // cudaStream_t stream_id=0);

// int update_result_table_i2u(int *result_table,
                                  // int *updated_result_table,
                                  // int *index_list,
                                  // uint64_t *ptr_list,
                                  // void *edge_addr,
                                  // uint64_t* edge_headers,
                                  // uint64_t pred_edge_shard_size,
                                  // cudaStream_t stream_id=0
                                  // );


// int update_result_table_k2u(int *result_table,
                                  // int *updated_result_table,
                                  // int *index_list,
                                  // uint64_t *ptr_list,
                                  // int column_num,
                                  // void *edge_addr,
                                  // uint64_t* edge_headers,
                                  // uint64_t pred_edge_shard_size,
                                  // int query_size,
                                  // cudaStream_t stream_id=0);

// int update_result_table_k2k(int *result_table,
                                  // int *updated_result_table,
                                  // int *index_list,
                                  // uint64_t *ptr_list,
                                  // int column_num,
                                  // void *edge_addr,
                                  // int end,
                                  // int query_size,
                                  // cudaStream_t stream_id=0);

// void calc_prefix_sum(int* d_out_arr,
                     // int* d_in_arr,
                     // int query_size,
                     // cudaStream_t stream_id=0);

// void update_result_table_sub(int *result_table,
                                  // int *updated_result_table,
                                  // int *mapping_list,
                                  // int *server_id_list,
                                  // int *prefix_sum_list,
                                  // int column_num,
                                  // int num_sub_request,
                                  // int query_size,
                                  // cudaStream_t stream_id=0);

// void calc_dispatched_position(int *d_result_table,
                              // int *d_mapping_list,
                              // int *d_server_id_list,
                              // int *d_server_sum_list,
                              // int *gpu_sub_table_size_list,
                              // int start,
                              // int column_num,
                              // int num_sub_request,
                              // int query_size,
                              // cudaStream_t stream_id=0);


#endif // end of USE_GPU
