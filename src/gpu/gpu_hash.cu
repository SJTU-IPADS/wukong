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

#include <thrust/execution_policy.h>
#include "gpu_hash.hpp"

namespace wukong {

#define ASSOCIATIVITY 8
#define SLOT_ID_ERROR ((uint64_t)(-1))
#define __inline__ inline __attribute__((always_inline))

/*********************************************
 *                                           *
 *                Utilities                  *
 *                                           *
 *********************************************/
__device__ static uint64_t myhash(ikey_t lkey) {
    uint64_t r = 0;
    r += lkey.vid;
    r <<= NBITS_IDX;
    r += lkey.pid;
    r <<= NBITS_DIR;
    r += lkey.dir;

    uint64_t key = r;
    key = (~key) + (key << 21);  // key = (key << 21) - key - 1;
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);  // key * 265
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);  // key * 21
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}

__device__ __inline__ uint64_t offset2pos(uint64_t offset, uint64_t* head_blks, uint64_t blk_size) {
    return head_blks[offset / blk_size] + offset % blk_size;
}

/*********************************************
 *                                           *
 *                Query functions            *
 *                                           *
 *********************************************/

__global__ void d_generate_key_list_i2u(sid_t* result_table,
                                        int index_vertex,
                                        int direction,
                                        ikey_t* key_list,
                                        int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        ikey_t r = ikey_t(0, index_vertex, direction);
        key_list[index] = r;
    }
}

void generate_key_list_i2u(sid_t* result_table,
                           int index_vertex,
                           int direction,
                           void* key_list,
                           int query_size,
                           cudaStream_t stream_id) {
    d_generate_key_list_i2u<<<WUKONG_GET_BLOCKS(query_size), WUKONG_CUDA_NUM_THREADS, 0, stream_id>>>(
        result_table,
        index_vertex,
        direction,
        reinterpret_cast<ikey_t*>(key_list),
        query_size);
}

__global__ void k_generate_key_list(sid_t* result_table,
                                    ikey_t* key_list,
                                    int var2col_start,
                                    int direction,
                                    int predict,
                                    int col_num,
                                    int row_num) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < row_num) {
        assert(col_num >= 0 && var2col_start >= 0);
        int prev_id = result_table[index * col_num + var2col_start];
        assert(prev_id > 0);
        ikey_t r = ikey_t(prev_id, predict, direction);
        key_list[index] = r;
    }
}

void gpu_generate_key_list(GPUEngineParam& param, cudaStream_t stream) {
    assert(param.query.row_num > 0);
    assert(param.query.var2col_start >= 0);

    k_generate_key_list<<<WUKONG_GET_BLOCKS(param.query.row_num),
                          WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.d_in_rbuf,
        param.gpu.d_key_list,
        param.query.var2col_start,
        param.query.dir,
        param.query.pid,
        param.query.col_num,
        param.query.row_num);

    CUDA_ASSERT(cudaPeekAtLastError());
}

__device__ __inline__ pattern_info_t* tid2pattern_info(int tid, pattern_info_t* pattrns, int num_patterns) {
    for (auto i = 0; i < num_patterns; ++i) {
        if (tid < pattrns[i].rbuf_max_row)
            return pattrns + i;
    }
    return nullptr;
}

__device__ void d_generate_key_list_k2u(int index,
                                        sid_t* result_table,
                                        ikey_t* key_list,
                                        int var2col,
                                        int direction,
                                        int predict,
                                        int col_num,
                                        int row_num) {
    int prev_id = result_table[index * col_num + var2col];
    assert(prev_id > 0);
    ikey_t r = ikey_t(prev_id, predict, direction);
    key_list[index] = r;
}

__global__ void k_get_slot_id_list(vertex_t* vertex_gaddr,
                                   ikey_t* d_key_list,
                                   uint64_t* d_slot_id_list,
                                   ikey_t empty_key,
                                   rdf_seg_meta_t* seg_meta,
                                   uint64_t* vertex_headers,
                                   uint64_t vertex_blk_sz,
                                   int query_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < query_size) {
        ikey_t key = d_key_list[index];
        uint64_t bucket_id = offset2pos(myhash(key) % seg_meta->num_buckets, vertex_headers, vertex_blk_sz);
        while (true) {
            for (uint64_t i = 0; i < ASSOCIATIVITY; i++) {
                uint64_t slot_id = bucket_id * ASSOCIATIVITY + i;
                if (i < ASSOCIATIVITY - 1) {
                    // data part
                    if (vertex_gaddr[slot_id].key == key) {
                        // we found it
                        d_slot_id_list[index] = slot_id;
                        return;
                    }
                } else {
                    if (!(vertex_gaddr[slot_id].key == empty_key)) {
                        // next pointer
                        // uint64_t next_bucket_id = vertex_gaddr[slot_id].key.vid-pred_metas[key.pid].indrct_hdr_start+pred_metas[key.pid].partition_sz;
                        uint64_t next_bucket_id = vertex_gaddr[slot_id].key.vid - seg_meta->ext_bucket_list[0].start + seg_meta->num_buckets;
                        bucket_id = offset2pos(next_bucket_id, vertex_headers, vertex_blk_sz);
                        break;
                    } else {
                        d_slot_id_list[index] = SLOT_ID_ERROR;
                        return;
                    }
                }
            }
        }
    }
}

// done
void gpu_get_slot_id_list(GPUEngineParam& param, cudaStream_t stream) {
    assert(param.query.row_num > 0);
    assert(param.query.var2col_start >= 0);

    ikey_t empty_key = ikey_t();

    k_get_slot_id_list<<<WUKONG_GET_BLOCKS(param.query.row_num),
                         WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.vertex_gaddr,
        param.gpu.d_key_list,
        param.gpu.d_slot_id_list,
        empty_key,
        param.gpu.segment_metas_d,
        param.gpu.d_kblk_mapping_table,
        param.gpu.vertex_blk_sz,
        param.query.row_num);

    CUDA_ASSERT(cudaPeekAtLastError());
}

__global__ void k_get_edge_list(uint64_t* slot_id_list,
                                vertex_t* vertex_gaddr,
                                edge_t* edge_gaddr,
                                int* index_list,
                                int* index_list_mirror,
                                uint64_t* off_list,
                                uint64_t seg_edge_start,
                                uint64_t* edge_headers,
                                uint64_t edge_blk_sz,
                                int query_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < query_size) {
        uint64_t id = slot_id_list[index];
        if (id != SLOT_ID_ERROR) {
            iptr_t r = vertex_gaddr[id].ptr;
            index_list_mirror[index] = r.size;
            off_list[index] = r.off - seg_edge_start;
        } else {
            index_list_mirror[index] = 0;
            off_list[index] = 0;
        }
    }
}

// done (k2u)
void gpu_get_edge_list(GPUEngineParam& param, cudaStream_t stream) {
    assert(param.query.row_num > 0);

    k_get_edge_list<<<WUKONG_GET_BLOCKS(param.query.row_num),
                      WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.d_slot_id_list,
        param.gpu.vertex_gaddr,
        param.gpu.edge_gaddr,
        param.gpu.d_prefix_sum_list,
        param.gpu.d_edge_size_list,
        param.gpu.d_offset_list,
        param.query.segment_edge_start,
        param.gpu.d_vblk_mapping_table,
        param.gpu.edge_blk_sz,
        param.query.row_num);

    CUDA_ASSERT(cudaPeekAtLastError());
}

__global__ void k_get_edge_list_k2c(
    uint64_t* slot_id_list,
    vertex_t* vertex_gaddr,
    int* edge_size_list,
    uint64_t* offset_list,
    edge_t* edge_gaddr,
    int64_t end,
    uint64_t seg_edge_start,
    uint64_t* edge_headers,
    uint64_t edge_blk_sz,
    int query_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < query_size) {
        uint64_t id = slot_id_list[index];
        if (id == SLOT_ID_ERROR) {
            edge_size_list[index] = 0;
            offset_list[index] = 0;
            return;
        }

        iptr_t r = vertex_gaddr[id].ptr;
        edge_size_list[index] = 0;
        offset_list[index] = r.off - seg_edge_start;

        for (int k = 0; k < r.size; k++) {
            uint64_t ptr = offset2pos(r.off - seg_edge_start + k,
                                      edge_headers,
                                      edge_blk_sz);
            if (edge_gaddr[ptr].val == end) {
                edge_size_list[index] = 1;
                break;
            }
        }
    }
}

void gpu_get_edge_list_k2c(GPUEngineParam& param, cudaStream_t stream) {
    assert(param.query.row_num > 0);

    k_get_edge_list_k2c<<<WUKONG_GET_BLOCKS(param.query.row_num),
                          WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.d_slot_id_list,
        param.gpu.vertex_gaddr,
        param.gpu.d_edge_size_list,
        param.gpu.d_offset_list,
        param.gpu.edge_gaddr,
        param.query.end_vid,
        param.query.segment_edge_start,
        param.gpu.d_vblk_mapping_table,
        param.gpu.edge_blk_sz,
        param.query.row_num);

    CUDA_ASSERT(cudaPeekAtLastError());
}

__global__ void k_get_edge_list_k2k(uint64_t* slot_id_list,
                                    vertex_t* vertex_gaddr,
                                    int* index_list,
                                    int* index_list_mirror,
                                    uint64_t* offset_list,
                                    int query_size,
                                    edge_t* edge_gaddr,
                                    sid_t* result_table,
                                    int col_num,
                                    int var2col_end,
                                    uint64_t seg_edge_start,
                                    uint64_t* edge_headers,
                                    uint64_t edge_blk_sz) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < query_size) {
        uint64_t id = slot_id_list[index];
        if (id == SLOT_ID_ERROR) {
            index_list_mirror[index] = 0;
            offset_list[index] = 0;
            return;
        }

        iptr_t r = vertex_gaddr[id].ptr;
        sid_t end_id = result_table[index * col_num + var2col_end];
        offset_list[index] = r.off - seg_edge_start;
        index_list_mirror[index] = 0;
        for (int k = 0; k < r.size; k++) {
            uint64_t ptr = offset2pos(r.off - seg_edge_start + k,
                                      edge_headers,
                                      edge_blk_sz);

            if (edge_gaddr[ptr].val == end_id) {
                index_list_mirror[index] = 1;
                break;
            }
        }
    }
}

void gpu_get_edge_list_k2k(GPUEngineParam& param, cudaStream_t stream) {
    assert(param.query.row_num > 0);
    assert(param.query.var2col_end >= 0);

    k_get_edge_list_k2k<<<WUKONG_GET_BLOCKS(param.query.row_num),
                          WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.d_slot_id_list,
        param.gpu.vertex_gaddr,
        param.gpu.d_prefix_sum_list,
        param.gpu.d_edge_size_list,
        param.gpu.d_offset_list,
        param.query.row_num,
        param.gpu.edge_gaddr,
        param.gpu.d_in_rbuf,
        param.query.col_num,
        param.query.var2col_end,
        param.query.segment_edge_start,
        param.gpu.d_vblk_mapping_table,
        param.gpu.edge_blk_sz);

    CUDA_ASSERT(cudaPeekAtLastError());
}

__device__ void d_get_edge_list_k2c(int index,
                                    uint64_t* slot_id_list,
                                    vertex_t* vertex_gaddr,
                                    int* index_list,
                                    int* index_list_mirror,
                                    uint64_t* ptr_list,
                                    edge_t* edge_gaddr,
                                    int64_t end,
                                    uint64_t seg_edge_start,
                                    uint64_t* edge_headers,
                                    uint64_t edge_blk_sz,
                                    int query_size) {
    uint64_t id = slot_id_list[index];
    iptr_t r = vertex_gaddr[id].ptr;

    index_list_mirror[index] = 0;
    ptr_list[index] = r.off - seg_edge_start;
    for (int k = 0; k < r.size; k++) {
        uint64_t ptr = offset2pos(r.off - seg_edge_start + k,
                                  edge_headers,
                                  edge_blk_sz);
        if (edge_gaddr[ptr].val == end) {
            index_list_mirror[index] = 1;
            break;
        }
    }
}

__global__ void k_update_result_buf_i2u(sid_t* result_table,
                                        sid_t* updated_result_table,
                                        int* index_list,
                                        uint64_t* ptr_list,
                                        edge_t* edge_gaddr,
                                        uint64_t* edge_headers,
                                        uint64_t edge_blk_sz) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int edge_num = 0;
    edge_num = index_list[0];

    if (index < edge_num) {
        uint64_t ptr = offset2pos(ptr_list[0] + index,
                                  edge_headers,
                                  edge_blk_sz);
        updated_result_table[index] = edge_gaddr[ptr].val;
    }
}

int gpu_update_result_buf_i2u(GPUEngineParam& param, cudaStream_t stream) {
    int table_size = 0;
    CUDA_ASSERT(cudaMemcpyAsync(&table_size,
                                param.gpu.d_prefix_sum_list,
                                sizeof(int),
                                cudaMemcpyDeviceToHost, stream));

    k_update_result_buf_i2u<<<WUKONG_GET_BLOCKS(param.query.row_num),
                              WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.d_in_rbuf,
        param.gpu.d_out_rbuf,
        param.gpu.d_prefix_sum_list,
        param.gpu.d_offset_list,
        param.gpu.edge_gaddr,
        param.gpu.d_vblk_mapping_table,
        param.gpu.edge_blk_sz);

    CUDA_ASSERT(cudaStreamSynchronize(stream));
    return table_size;
}

__global__ void k_update_result_buf_k2k(sid_t* result_table,
                                        sid_t* updated_result_table,
                                        int* index_list,
                                        int column_num,
                                        int query_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < query_size) {
        int edge_num = 0, start = 0;
        if (index == 0) {
            edge_num = index_list[index];
            start = 0;
        } else {
            edge_num = index_list[index] - index_list[index - 1];
            start = column_num * index_list[index - 1];
        }
        sid_t buff[20];
        for (int c = 0; c < column_num; c++) {
            buff[c] = result_table[column_num * index + c];
        }
        for (int k = 0; k < edge_num; k++) {
            for (int c = 0; c < column_num; c++) {
                updated_result_table[start + c] = buff[c];
            }
        }
    }
}

// done
void gpu_update_result_buf_k2k(GPUEngineParam& param, cudaStream_t stream) {
    k_update_result_buf_k2k<<<WUKONG_GET_BLOCKS(param.query.row_num),
                              WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.d_in_rbuf,
        param.gpu.d_out_rbuf,
        param.gpu.d_prefix_sum_list,
        param.query.col_num,
        param.query.row_num);
}

void gpu_update_result_buf_k2c(GPUEngineParam& param, cudaStream_t stream) {
    gpu_update_result_buf_k2k(param, stream);
}

void gpu_calc_prefix_sum(GPUEngineParam& param,
                         cudaStream_t stream) {
    thrust::device_ptr<int> d_in_ptr(param.gpu.d_edge_size_list);
    thrust::device_ptr<int> d_out_ptr(param.gpu.d_prefix_sum_list);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), d_in_ptr, d_in_ptr + param.query.row_num, d_out_ptr);
}

// Calculate destination server for each records in the result buffer
__global__ void k_hash_tuples_to_server(sid_t* result_table,
                                        int* server_id_list,
                                        int var2col,
                                        int col_num,
                                        int num_sub_request,
                                        int query_size) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx < query_size) {
        server_id_list[row_idx] = result_table[row_idx * col_num + var2col] % num_sub_request;
    }
}

void hash_dispatched_server_id(sid_t* result_table,
                               int* server_id_list,
                               int var2col,
                               int col_num,
                               int num_sub_request,
                               int query_size,
                               cudaStream_t stream) {
    k_hash_tuples_to_server<<<WUKONG_GET_BLOCKS(query_size),
                              WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        result_table,
        server_id_list,
        var2col,
        col_num,
        num_sub_request,
        query_size);
}

__global__ void k_history_dispatch(sid_t* result_table,
                                   int* position_list,
                                   int* server_id_list,
                                   int* server_sum_list,
                                   int start,
                                   int col_num,
                                   int num_sub_request,
                                   int query_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < query_size) {
        int server_id = server_id_list[index];
        position_list[index] = atomicAdd(&server_sum_list[server_id], 1);
    }
}

void history_dispatch(sid_t* result_table,
                      int* position_list,
                      int* server_id_list,
                      int* server_sum_list,
                      int start,
                      int col_num,
                      int num_sub_request,
                      int query_size,
                      cudaStream_t stream) {
    k_history_dispatch<<<WUKONG_GET_BLOCKS(query_size),
                         WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        result_table,
        position_list,
        server_id_list,
        server_sum_list,
        start,
        col_num,
        num_sub_request,
        query_size);
}

__global__ void k_query_split(int* d_row_sum_list,
                              int* d_buf_offset_list,
                              int row_num,
                              int col_num,
                              int sub_query_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < row_num - 1) {
        int sub_row_num = d_row_sum_list[index];
        int sub_row_num_next = d_row_sum_list[index + 1];
        int query_id = sub_row_num / sub_query_size;
        int query_id_next = sub_row_num_next / sub_query_size;
        if (query_id != query_id_next) {
            d_buf_offset_list[query_id_next] = (index + 1) * col_num;
        }
    }
}

void query_split(int* d_row_sum_list,
                 int* d_buf_offset_list,
                 int row_num,
                 int col_num,
                 int query_size,
                 int num_jobs,
                 cudaStream_t stream) {
    int sub_query_size = query_size / num_jobs;
    k_query_split<<<WUKONG_GET_BLOCKS(row_num),
                    WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        d_row_sum_list,
        d_buf_offset_list,
        row_num,
        col_num,
        sub_query_size);
}

// Put the partitioned result buffer to output result buf (via different offset)
__global__ void k_split_result_buf(sid_t* d_in_result_buf,
                                   sid_t* d_out_result_buf,
                                   int* d_position_list,
                                   int* server_id_list,
                                   int* server_sum_list,
                                   int column_num,
                                   int num_sub_request,
                                   int query_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < query_size) {
        int dst_sid = server_id_list[index];
        int mapped_index = server_sum_list[dst_sid] + d_position_list[index];
        for (int c = 0; c < column_num; c++) {
            d_out_result_buf[column_num * mapped_index + c] = d_in_result_buf[column_num * index + c];
        }
    }
}

void gpu_split_result_buf(GPUEngineParam& param, int num_servers, cudaStream_t stream) {
    // borrow other buffers for temporary use
    int* d_position_list = reinterpret_cast<int*>(param.gpu.d_slot_id_list);
    int* d_server_id_list = param.gpu.d_prefix_sum_list;
    int* d_server_sum_list = param.gpu.d_edge_size_list;

    k_split_result_buf<<<WUKONG_GET_BLOCKS(param.query.row_num),
                         WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.d_in_rbuf,
        param.gpu.d_out_rbuf,
        d_position_list,
        d_server_id_list,
        d_server_sum_list,
        param.query.col_num,
        num_servers,
        param.query.row_num);
}

void gpu_shuffle_result_buf(GPUEngineParam& param, int num_jobs, std::vector<int>& buf_sizes,
                            std::vector<int>& buf_heads, cudaStream_t stream) {
    // borrow other buffers for temporary use
    int* d_position_list = reinterpret_cast<int*>(param.gpu.d_slot_id_list);
    int* d_server_id_list = param.gpu.d_prefix_sum_list;
    int* d_server_sum_list = param.gpu.d_edge_size_list;

    CUDA_ASSERT(cudaMemsetAsync(d_server_sum_list, 0, num_jobs * sizeof(int), stream));

    // calculate destination server for each record
    hash_dispatched_server_id(param.gpu.d_in_rbuf,
                              d_server_id_list,
                              param.query.var2col_start,
                              param.query.col_num,
                              num_jobs,
                              param.query.row_num,
                              stream);

    history_dispatch(param.gpu.d_in_rbuf,
                     d_position_list,
                     d_server_id_list,
                     d_server_sum_list,
                     param.query.var2col_start,
                     param.query.col_num,
                     num_jobs,
                     param.query.row_num,
                     stream);

    CUDA_ASSERT(cudaMemcpyAsync(&buf_sizes[0],
                                d_server_sum_list,
                                sizeof(int) * num_jobs,
                                cudaMemcpyDeviceToHost,
                                stream));

    CUDA_STREAM_SYNC(stream);

    // calculate exclusive prefix sum for d_server_sum_list
    thrust::device_ptr<int> dptr(d_server_sum_list);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), dptr, dptr + num_jobs, dptr);

    CUDA_STREAM_SYNC(stream);

    CUDA_ASSERT(cudaMemcpyAsync(&buf_heads[0],
                                d_server_sum_list,
                                sizeof(int) * num_jobs,
                                cudaMemcpyDeviceToHost,
                                stream));
}

void gpu_split_giant_query(GPUEngineParam& param, int row_num, int col_num,
                           int num_jobs, int query_size, std::vector<int>& buf_offs, cudaStream_t stream) {
    // borrow other buffers for temporary use
    int* d_row_sum_list = param.gpu.d_prefix_sum_list;
    int* d_buf_offset_list = reinterpret_cast<int*>(param.gpu.d_offset_list);
    CUDA_ASSERT(cudaMemsetAsync(d_buf_offset_list, -1, num_jobs * sizeof(int), stream));

    // split query using row num
    query_split(d_row_sum_list,
                d_buf_offset_list,
                row_num,
                col_num,
                query_size,
                num_jobs,
                stream);

    CUDA_ASSERT(cudaMemcpyAsync(&buf_offs[0],
                                d_buf_offset_list,
                                sizeof(int) * num_jobs,
                                cudaMemcpyDeviceToHost,
                                stream));

    CUDA_STREAM_SYNC(stream);
    buf_offs[0] = 0;
}

__global__ void k_update_result_table_k2u(sid_t* result_table,
                                          sid_t* updated_result_table,
                                          int* prefix_sum_list,
                                          uint64_t* off_list,
                                          edge_t* edge_gaddr,
                                          uint64_t* edge_headers,
                                          uint64_t edge_blk_sz,
                                          int col_num,
                                          int query_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < query_size) {
        int edge_num = 0, start = 0;
        if (index == 0) {
            edge_num = prefix_sum_list[index];
            start = 0;
        } else {
            edge_num = prefix_sum_list[index] - prefix_sum_list[index - 1];
            start = (col_num + 1) * prefix_sum_list[index - 1];
        }

        sid_t buff[20];
        for (int c = 0; c < col_num; c++) {
            buff[c] = result_table[col_num * index + c];
        }

        for (int k = 0; k < edge_num; k++) {
            // put original columns to table
            for (int c = 0; c < col_num; c++) {
                updated_result_table[start + k * (col_num + 1) + c] = buff[c];
            }
            // put the new column to table
            uint64_t ptr = offset2pos(off_list[index] + k, edge_headers, edge_blk_sz);
            updated_result_table[start + k * (col_num + 1) + col_num] = edge_gaddr[ptr].val;
        }
    }
}

void gpu_update_result_buf_k2u(GPUEngineParam& param, cudaStream_t stream) {
    k_update_result_table_k2u<<<WUKONG_GET_BLOCKS(param.query.row_num),
                                WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.d_in_rbuf,
        param.gpu.d_out_rbuf,
        param.gpu.d_prefix_sum_list,
        param.gpu.d_offset_list,
        param.gpu.edge_gaddr,
        param.gpu.d_vblk_mapping_table,
        param.gpu.edge_blk_sz,
        param.query.col_num,
        param.query.row_num);
}

/*********************************
 * Kernels for combined patterns
 *********************************/

__global__ void k_generate_key_list_combined(sid_t* result_table,
                                             ikey_t* key_list,
                                             pattern_info_t* patterns,
                                             int num_patterns,
                                             int total_rows) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < total_rows) {
        pattern_info_t* pattern = tid2pattern_info(index, patterns, num_patterns);
        assert(pattern != nullptr);

        sid_t* table = (result_table + pattern->rbuf_start);
        int idx = index - pattern->rbuf_start_row;
        assert(idx >= 0);
        int prev_id = table[idx * pattern->col_num + pattern->start_var2col];
        ikey_t r = ikey_t(prev_id, pattern->pid, pattern->dir);
        key_list[index] = r;
    }
}

void gpu_generate_key_list_combined(GPUEngineParam& param, cudaStream_t stream) {
    assert(param.query.row_num > 0);

    k_generate_key_list_combined<<<WUKONG_GET_BLOCKS(param.query.row_num),
                                   WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.d_in_rbuf,
        param.gpu.d_key_list,
        param.gpu.pattern_infos_d,
        param.gpu.num_patterns,
        param.query.row_num);
}

__global__ void k_get_slot_id_list_combined(vertex_t* vertices_d,
                                            ikey_t* key_list_d,
                                            uint64_t* slot_id_list_d,
                                            ikey_t empty_key,
                                            uint64_t vertex_blk_size,
                                            pattern_info_t* patterns_d,
                                            int num_patterns,
                                            int total_rows) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < total_rows) {
        ikey_t key = key_list_d[index];
        // if we have multiple segments, how to access the corresponding segment
        // according to the key
        pattern_info_t* pattern = tid2pattern_info(index, patterns_d, num_patterns);
        assert(pattern != nullptr);

        rdf_seg_meta_t* segment_meta = pattern->segment_meta_dptr;
        assert(segment_meta != nullptr);
        uint64_t bucket_id = offset2pos(myhash(key) % segment_meta->num_buckets,
                                        pattern->key_mapping_dptr, vertex_blk_size);

        while (true) {
            for (uint64_t i = 0; i < ASSOCIATIVITY; i++) {
                uint64_t slot_id = bucket_id * ASSOCIATIVITY + i;
                if (i < ASSOCIATIVITY - 1) {
                    // data part
                    if (vertices_d[slot_id].key == key) {
                        // we found it
                        slot_id_list_d[index] = slot_id;
                        return;
                    }
                } else {
                    if (!(vertices_d[slot_id].key == empty_key)) {
                        // next pointer
                        // uint64_t next_bucket_id = vertices_d[slot_id].key.vid-pred_metas[key.pid].indrct_hdr_start+pred_metas[key.pid].partition_sz;
                        uint64_t next_bucket_id = vertices_d[slot_id].key.vid - segment_meta->ext_bucket_list[0].start + segment_meta->num_buckets;
                        bucket_id = offset2pos(next_bucket_id, pattern->key_mapping_dptr, vertex_blk_size);
                        break;
                    } else {
                        slot_id_list_d[index] = SLOT_ID_ERROR;
                        return;
                    }
                }
            }
        }
    }
}

void gpu_get_slot_id_list_combined(GPUEngineParam& param, cudaStream_t stream) {
    assert(param.query.row_num > 0);

    ikey_t empty_key = ikey_t();

    k_get_slot_id_list_combined<<<WUKONG_GET_BLOCKS(param.query.row_num),
                                  WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.vertex_gaddr,
        param.gpu.d_key_list,
        param.gpu.d_slot_id_list,
        empty_key,
        param.gpu.vertex_blk_sz,
        param.gpu.pattern_infos_d,
        param.gpu.num_patterns,
        param.query.row_num);
}

__global__ void k_get_edge_list_combined(uint64_t* slot_id_list,
                                         vertex_t* vertex_d,
                                         int* edge_size_list,
                                         uint64_t* offset_list,
                                         pattern_info_t* patterns,
                                         int num_patterns,
                                         int row_num) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < row_num) {
        pattern_info_t* pattern_ptr = tid2pattern_info(index, patterns, num_patterns);
        assert(pattern_ptr != nullptr);

        uint64_t slot_id = slot_id_list[index];
        if (slot_id == SLOT_ID_ERROR) {
            edge_size_list[index] = 0;
            offset_list[index] = 0;
            return;
        }

        iptr_t ptr = vertex_d[slot_id].ptr;
        edge_size_list[index] = ptr.size;
        offset_list[index] = ptr.off - pattern_ptr->segment_meta_dptr->edge_start;
    }
}

void gpu_get_edge_list_combined(GPUEngineParam& param, cudaStream_t stream) {
    assert(param.query.row_num > 0);

    k_get_edge_list_combined<<<WUKONG_GET_BLOCKS(param.query.row_num),
                               WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.d_slot_id_list,
        param.gpu.vertex_gaddr,
        param.gpu.d_edge_size_list,
        param.gpu.d_offset_list,
        param.gpu.pattern_infos_d,
        param.gpu.num_patterns,
        param.query.row_num);
}

__global__ void k_get_edge_list_k2k_combined(sid_t* result_table,
                                             uint64_t* slot_id_list,
                                             vertex_t* vertex_d,
                                             edge_t* edge_d,
                                             uint64_t edge_blk_size,
                                             int* edge_size_list,
                                             uint64_t* offset_list,
                                             pattern_info_t* patterns,
                                             int num_patterns,
                                             int row_num) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < row_num) {
        pattern_info_t* pattern = tid2pattern_info(index, patterns, num_patterns);
        assert(pattern != nullptr);
        rdf_seg_meta_t* segment_meta_dptr = pattern->segment_meta_dptr;

        uint64_t slot_id = slot_id_list[index];
        if (slot_id == SLOT_ID_ERROR) {
            edge_size_list[index] = 0;
            offset_list[index] = 0;
            return;
        }

        iptr_t r = vertex_d[slot_id].ptr;
        edge_size_list[index] = 0;

        sid_t* table = (result_table + pattern->rbuf_start);
        int row_idx = index - pattern->rbuf_start_row;
        sid_t end_id = table[row_idx * pattern->col_num + pattern->end_var2col];
        offset_list[index] = r.off - segment_meta_dptr->edge_start;

        for (int k = 0; k < r.size; k++) {
            uint64_t ptr = offset2pos((r.off - segment_meta_dptr->edge_start + k),
                                      pattern->value_mapping_dptr, edge_blk_size);

            if (edge_d[ptr].val == end_id) {
                edge_size_list[index] = 1;
                break;
            }
        }
    }
}

void gpu_get_edge_list_k2k_combined(GPUEngineParam& param, cudaStream_t stream) {
    k_get_edge_list_k2k_combined<<<WUKONG_GET_BLOCKS(param.query.row_num),
                                   WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.d_in_rbuf,
        param.gpu.d_slot_id_list,
        param.gpu.vertex_gaddr,
        param.gpu.edge_gaddr,
        param.gpu.edge_blk_sz,
        param.gpu.d_edge_size_list,
        param.gpu.d_offset_list,
        param.gpu.pattern_infos_d,
        param.gpu.num_patterns,
        param.query.row_num);
}

__global__ void k_get_edge_list_k2c_combined(uint64_t* slot_id_list,
                                             vertex_t* vertex_d,
                                             edge_t* edge_d,
                                             uint64_t edge_blk_size,
                                             int* edge_size_list,
                                             uint64_t* offset_list,
                                             pattern_info_t* patterns,
                                             int num_patterns,
                                             int row_num) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < row_num) {
        pattern_info_t* pattern_ptr = tid2pattern_info(index, patterns, num_patterns);
        assert(pattern_ptr != nullptr);
        rdf_seg_meta_t* segment_meta_dptr = pattern_ptr->segment_meta_dptr;

        uint64_t slot_id = slot_id_list[index];
        if (slot_id == SLOT_ID_ERROR) {
            edge_size_list[index] = 0;
            offset_list[index] = 0;
            return;
        }

        iptr_t r = vertex_d[slot_id].ptr;
        edge_size_list[index] = 0;
        offset_list[index] = r.off - segment_meta_dptr->edge_start;

        for (int k = 0; k < r.size; k++) {
            uint64_t pos = offset2pos((r.off - segment_meta_dptr->edge_start + k),
                                      pattern_ptr->value_mapping_dptr, edge_blk_size);
            if (edge_d[pos].val == pattern_ptr->end_vid) {
                edge_size_list[index] = 1;
                break;
            }
        }
    }
}

void gpu_get_edge_list_k2c_combined(GPUEngineParam& param, cudaStream_t stream) {
    k_get_edge_list_k2c_combined<<<WUKONG_GET_BLOCKS(param.query.row_num),
                                   WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.d_slot_id_list,
        param.gpu.vertex_gaddr,
        param.gpu.edge_gaddr,
        param.gpu.edge_blk_sz,
        param.gpu.d_edge_size_list,
        param.gpu.d_offset_list,
        param.gpu.pattern_infos_d,
        param.gpu.num_patterns,
        param.query.row_num);
}

__global__ void k_update_result_buf_k2k_combined(sid_t* result_table,
                                                 sid_t* updated_result_table,
                                                 int* prefix_sum_list,
                                                 pattern_info_t* patterns,
                                                 updated_pattern_info_t* updated_patterns,
                                                 int num_patterns,
                                                 int row_num) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < row_num) {
        pattern_info_t* pattern = tid2pattern_info(index, patterns, num_patterns);
        updated_pattern_info_t* updated_patt = updated_patterns + (pattern - patterns);

        assert(pattern != nullptr);

        int edge_num = 0, start_in_newbuf = 0;
        size_t base_psum = 0;
        int col_num = pattern->col_num;

        if (index == 0) {
            edge_num = prefix_sum_list[index];
            start_in_newbuf = 0;
        } else {
            edge_num = prefix_sum_list[index] - prefix_sum_list[index - 1];
            base_psum = prefix_sum_list[pattern->rbuf_start_row - 1];
            start_in_newbuf = updated_patt->rbuf_start + (prefix_sum_list[index - 1] - base_psum) * col_num;
        }

        sid_t row_buf[20];
        sid_t* table = (result_table + pattern->rbuf_start);
        int row_idx = index - pattern->rbuf_start_row;
        for (int c = 0; c < col_num; c++) {
            row_buf[c] = table[col_num * row_idx + c];
        }

        for (int k = 0; k < edge_num; ++k) {
            for (int c = 0; c < col_num; ++c) {
                updated_result_table[start_in_newbuf + c] = row_buf[c];
            }
        }
    }
}

void gpu_update_result_buf_k2k_combined(GPUEngineParam& param, cudaStream_t stream) {
    k_update_result_buf_k2k_combined<<<WUKONG_GET_BLOCKS(param.query.row_num),
                                       WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.d_in_rbuf,
        param.gpu.d_out_rbuf,
        param.gpu.d_prefix_sum_list,
        param.gpu.pattern_infos_d,
        param.gpu.updated_patt_infos_d,
        param.gpu.num_patterns,
        param.query.row_num);
}

void gpu_update_result_buf_k2c_combined(GPUEngineParam& param, cudaStream_t stream) {
    gpu_update_result_buf_k2k_combined(param, stream);
}

__global__ void k_update_result_buf_k2u_combined(sid_t* result_table,
                                                 sid_t* updated_result_table,
                                                 int* prefix_sum_list,
                                                 uint64_t* offset_list,
                                                 edge_t* edge_gaddr,
                                                 pattern_info_t* patterns,
                                                 updated_pattern_info_t* updated_patterns,
                                                 int num_patterns,
                                                 uint64_t edge_blk_sz,
                                                 int row_num) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < row_num) {
        pattern_info_t* pattern = tid2pattern_info(index, patterns, num_patterns);
        updated_pattern_info_t* updated_patt = updated_patterns + (pattern - patterns);

        assert(updated_patt != nullptr);

        size_t edge_num = 0, start_in_newbuf = 0;
        size_t base_psum = 0;
        int col_num = pattern->col_num;

        if (index == 0) {
            edge_num = prefix_sum_list[index];
            start_in_newbuf = 0;
        } else {
            edge_num = prefix_sum_list[index] - prefix_sum_list[index - 1];
            base_psum = prefix_sum_list[pattern->rbuf_start_row - 1];

            start_in_newbuf = updated_patt->rbuf_start + (prefix_sum_list[index - 1] - base_psum) * (col_num + 1);
        }

        sid_t row_buf[20];
        sid_t* table = (result_table + pattern->rbuf_start);
        int row_idx = index - pattern->rbuf_start_row;
        for (int c = 0; c < col_num; c++) {
            row_buf[c] = table[col_num * row_idx + c];
        }

        for (int k = 0; k < edge_num; k++) {
            // put original columns to table
            for (int c = 0; c < col_num; c++) {
                updated_result_table[start_in_newbuf + k * (col_num + 1) + c] = row_buf[c];
            }
            // put the new column to table
            uint64_t pos = offset2pos(offset_list[index] + k, pattern->value_mapping_dptr, edge_blk_sz);
            updated_result_table[start_in_newbuf + k * (col_num + 1) + col_num] = edge_gaddr[pos].val;
        }
    }
}

void gpu_update_result_buf_k2u_combined(GPUEngineParam& param, cudaStream_t stream) {
    k_update_result_buf_k2u_combined<<<WUKONG_GET_BLOCKS(param.query.row_num),
                                       WUKONG_CUDA_NUM_THREADS, 0, stream>>>(
        param.gpu.d_in_rbuf,
        param.gpu.d_out_rbuf,
        param.gpu.d_prefix_sum_list,
        param.gpu.d_offset_list,
        param.gpu.edge_gaddr,
        param.gpu.pattern_infos_d,
        param.gpu.updated_patt_infos_d,
        param.gpu.num_patterns,
        param.gpu.edge_blk_sz,
        param.query.row_num);
}

}  // namespace wukong
