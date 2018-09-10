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

#include <vector>

#include "assertion.hpp"
#include "type.hpp"
#include "query.hpp"
#include "gpu_hash.hpp"
#include "gpu_utils.hpp"

using namespace std;

class GpuEngineImpl {
private:
    GPUMem *gmem;
    GPUCache *gcache;
    GPUEngineParam engine_param;


    void reverse_result_buf() {
        gmem->reverse_rbuf();
    }

public:
    GpuEngineImpl() {
        // init engine_param
    }

    ~GpuEngineImpl();


    vector<sid_t> index_to_unknown(SPARQLQuery &req, sid_t tpid, dir_t d) {
        ASSERT_MSG(false, "not implemented")
        return nullptr;
    }

    void known_to_unknown(SPARQLQuery &req, sid_t start, sid_t pid, dir_t d, vector<int> &new_table) {
        cudaStream_t stream = 0;//streamPool->get_stream(predict);

        engine_param.query_state = {
            .start_vid = start,
            .pid = pid,
            .dir = d,
            .col_num = req.result.col_num,
            .row_num = req.get_row_num()
            // .pid_edge_segment_start = xxx
        };

        if(!req.is_first_handler()) {
            d_result_table = (int*)req.gpu_history_ptr;
        }
        else {
            // Siyuan: 如果执行的是第一条pattern，那么在进入handler之前
            // 就应该已经把history拷贝到GPU上了，不该进入这个分支
            ASSERT(false);
        }

        ASSERT(gmem->res_inbuf() != gmem->res_outbuf());
        ASSERT(nullptr != gmem->res_inbuf());

        // before processing the query, we should ensure the data of required predicates is loaded.
        if (!gcache->check_pred_exist(predict))
            gcache->load_predicate(predict, predict, req, stream, false);

        segid_t seg_to_use(0, pid, d);
        gcache->load_segment(seg_to_use, seg_to_use, req.pattern_group, stream, false);

        // preload next
        if (global_gpu_enable_pipeline) {
            while(!req.preds.empty() && gcache->check_pred_exist(req.preds[0])) {
                req.preds.erase(req.preds.begin());
            }
            if(!req.preds.empty()) {
                gcache->load_predicate(req.preds[0], predict, req, streamPool->get_stream(req.preds[0]), true);
            }
        }
        vector<uint64_t> vertex_headers = gcache->get_seg_vertex_headers(predict);
        vector<uint64_t> edge_headers = gcache->get_seg_edge_headers(predict);
        uint64_t pred_vertex_shard_size = gcache->shard_bucket_num;
        uint64_t pred_edge_shard_size = gcache->shard_entry_num;

        // setup GPU parameters
        engine_param.gpu = {
            .key_mapping_dv = vertex_headers,
            .value_mapping_dv = edge_headers,
            .in_rbuf_dp = thrust::device_pointer_cast((sid_t*) gmem->res_inbuf()),
            .out_rbuf_dp = thrust::device_pointer_cast((sid_t*) gmem->res_outbuf())
        };

        gpu_lookup_hashtable_k2u(engine_param, stream);

        gpu_calc_prefix_sum(engine_param.gpu.index_dv,
                        engine_param.gpu.index_dv_mirror,
                        engine_param.query_state.row_num,
                        stream);

        int table_size = gpu_update_result_buf_k2u(engine_param, stream);

        CUDA_STREAM_SYNC(stream);

        auto &dptr = engine_param.gpu.out_rbuf_dp;

        if (req.pattern_step + 1 == req.pattern_group.size()) {
            new_table.reserve(table_size);
            thrust::copy(dptr, dptr + table_size, new_table.begin());

            // TODO: shall we have to clear result buf?
            // req.clear_result_buf();

        } else {
            auto raw_ptr = thrust::raw_pointer_cast(dptr);
            req.set_result_buf(static_cast<char*>(raw_ptr), table_size);

            // TODO: when to reverse in_rbuf & out_rbuf
            /* if (req.gpu_state.origin_result_buf_dp != nullptr) {
             *     d_updated_result_table = (int*)req.gpu_state.origin_result_buf_dp;
             *     req.gpu_origin_buffer_head = nullptr;
             * }
             * else {
             *     d_updated_result_table = d_result_table;
             * } */
            reverse_result_buf();
        }

    }

    vector<sid_t> known_to_known(const SPARQLQuery &req, sid_t start, sid_t pid,
            sid_t end, dir_t d) {
    
        return nullptr;
    }

    vector<sid_t> known_to_const(const SPARQLQuery &req, sid_t start, sid_t pid,
            sid_t end, dir_t d) {
    
        return nullptr;
    }

    vector<sid_t> known_to_unknown(const SPARQLQuery &req, sid_t vid, sid_t pid, dir_t d) {
    
        return nullptr;
    }

    void generate_sub_query(const SPARQLQuery &req, sid_t start, int num_jobs,
            const vector<int*>& buf_dps, const vector<int> buf_sizes) {


    }

};





#endif  // USE_GPU
