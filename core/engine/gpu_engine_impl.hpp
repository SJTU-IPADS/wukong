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

#include "global.hpp"
#include "assertion.hpp"
#include "query.hpp"
#include "gpu_hash.hpp"
#include "gpu_utils.hpp"
#include "gpu_cache.hpp"
#include "gpu_stream.hpp"
#include "logger2.hpp"

using namespace std;

class GPUEngineImpl final {
private:
    GPUMem *gmem;
    GPUCache *gcache;
    GPUStreamPool *stream_pool;
    GPUEngineParam engine_param;

    void reverse_result_buf() {
        gmem->reverse_rbuf();
    }

    segid_t pattern_to_segid(const SPARQLQuery& req, int pattern_id) {
        SPARQLQuery::Pattern patt = req.pattern_group.patterns[pattern_id];
        segid_t segid(0, patt.predicate, patt.direction);
        // rdf_segment_meta_t seg = gcache->get_segment_meta(segid);
        return segid;
    }

    vector<segid_t> pattgrp_to_segids(const SPARQLQuery::PatternGroup& pattgrp) {
        vector<segid_t> segids;
        for (const auto &p : pattgrp.patterns) {
            segid_t segid(0, p.predicate, p.direction);
            segids.push_back(segid);
        }
        return segids;
    }


public:
    GPUEngineImpl(GPUCache *gcache, GPUMem *gmem, GPUStreamPool *stream_pool)
        : gcache(gcache), gmem(gmem), stream_pool(stream_pool),
          engine_param(gcache->dev_vertex_addr(), gcache->dev_edge_addr(),
                  gcache->num_key_blocks(), gcache->num_value_blocks()) {
        // init engine_param
    }

    ~GPUEngineImpl() { }


    char* load_result_buf(const SPARQLQuery::Result &r) {
        logstream(LOG_EMPH) << "load_result_buf: table_size=> " << r.result_table.size() << LOG_endl;
        CUDA_ASSERT( cudaMemcpy((void**)gmem->res_inbuf(),
                    &r.result_table[0],
                    sizeof(r.result_table[0]) * r.result_table.size(),
                    cudaMemcpyHostToDevice) );

        return gmem->res_inbuf();
    }


    // TODO
    vector<sid_t> index_to_unknown(SPARQLQuery &req, sid_t tpid, dir_t d) {
        ASSERT_MSG(false, "not implemented");
        return vector<sid_t>();
    }

    void known_to_unknown(SPARQLQuery &req, ssid_t start, ssid_t pid, dir_t d, vector<sid_t> &new_table) {
        cudaStream_t stream = stream_pool->get_stream(pid);
        segid_t current_seg = pattern_to_segid(req, req.pattern_step);
        rdf_segment_meta_t seg_meta = gcache->get_segment_meta(current_seg);

        engine_param.query.start_vid = start;
        engine_param.query.pid = pid;
        engine_param.query.dir = d;
        engine_param.query.col_num = req.result.get_col_num(),
        engine_param.query.row_num = req.result.get_row_num(),
        engine_param.query.segment_edge_start = seg_meta.edge_start;

        // not the first pattern
        // if (req.pattern_step != 0) {
            // d_result_table = (int*)req.gpu_history_ptr;
        // } else {
            // ASSERT(false);
        // }

        ASSERT(gmem->res_inbuf() != gmem->res_outbuf());
        ASSERT(nullptr != gmem->res_inbuf());


        // before processing the query, we should ensure the data of required predicates is loaded.
        vector<segid_t> required_segs = pattgrp_to_segids(req.pattern_group);

        if (!gcache->seg_in_cache(current_seg))
            gcache->load_segment(current_seg, required_segs, stream);


        // prefetch segment of next pattern
        if (global_gpu_enable_pipeline) {
            auto next_seg = pattern_to_segid(req, req.pattern_step + 1);
            auto stream2 = stream_pool->get_stream(next_seg.pid);

            if (!gcache->seg_in_cache(next_seg)) {
                gcache->prefetch_segment(next_seg, current_seg, required_segs, stream2);
            }
        }

        vector<uint64_t> vertex_mapping = gcache->get_vertex_mapping(current_seg);
        vector<uint64_t> edge_mapping = gcache->get_edge_mapping(current_seg);
        // TODO 把segment的metadata 拷贝到GPU memory上

        // uint64_t pred_vertex_shard_size = gcache->shard_bucket_num;
        // uint64_t pred_edge_shard_size = gcache->shard_entry_num;

        // setup GPU engine parameters
        rdf_segment_meta_t segment = gcache->get_segment_meta(current_seg);
        engine_param.load_segment_mappings(vertex_mapping, edge_mapping, segment);
        engine_param.load_segment_meta(seg_meta);
        engine_param.set_result_bufs(gmem->res_inbuf(), gmem->res_outbuf());
        engine_param.set_cache_param(global_block_num_buckets, global_block_num_edges);
        logstream(LOG_EMPH) << "cache: #buckets: " << global_block_num_buckets << ", #edges: " <<  global_block_num_edges << LOG_endl;

        // CUDA_DEVICE_SYNC;

        // process query on GPU
        gpu_lookup_hashtable_k2u(engine_param);
        logstream(LOG_EMPH) << "GPU_lookup_hash_k2u done. row_num=>" << engine_param.query.row_num << LOG_endl;

        gpu_calc_prefix_sum(engine_param);
        logstream(LOG_EMPH) << "GPU_calc_prefix_sum done. row_num=>" << engine_param.query.row_num << LOG_endl;

        int table_size = gpu_update_result_buf_k2u(engine_param);
        logstream(LOG_EMPH) << "GPU_update_result_buf_k2u done. table_size=" << table_size << ", col_num=" << engine_param.query.col_num << LOG_endl;

        // CUDA_STREAM_SYNC(stream);

        // CUDA_DEVICE_SYNC;

        // copy the result on GPU to CPU if we come to the last pattern
        if (req.pattern_step + 1 == req.pattern_group.patterns.size()) {
            new_table.resize(table_size);
            thrust::device_ptr<int> dptr(engine_param.gpu.d_out_rbuf);
            thrust::copy(dptr, dptr + table_size, new_table.begin());
            logstream(LOG_EMPH) << "new_table.size()=" << new_table.size() << LOG_endl;

            // TODO: Do we need to clear result buf?
            // req.clear_result_buf();

        } else {
            req.set_result_buf((char*)engine_param.gpu.d_out_rbuf, table_size);

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

    // TODO
    vector<sid_t> known_to_known(const SPARQLQuery &req, sid_t start, sid_t pid,
            sid_t end, dir_t d) {
    
        ASSERT_MSG(false, "not implemented");
        return vector<sid_t>();
    }

    // TODO
    vector<sid_t> known_to_const(const SPARQLQuery &req, sid_t start, sid_t pid,
            sid_t end, dir_t d) {
    
        ASSERT_MSG(false, "not implemented");
        return vector<sid_t>();
    }

    // TODO
    void generate_sub_query(const SPARQLQuery &req, sid_t start, int num_jobs,
            vector<int*>& buf_dps, vector<int>& buf_sizes) {
        int query_size = req.result.get_row_num();
        cudaStream_t stream = 0;

        ASSERT(req.pattern_step > 0);

        if (!req.pattern_step == 0) {
        } else {
            ASSERT(false);
        }

        // calc_dispatched_position
        gpu_shuffle_result_buf(engine_param, buf_sizes, stream);

        // update_result_table_sub
        gpu_split_result_buf(engine_param, num_jobs, stream);

        for (int i = 0; i < num_jobs; ++i) {
            // buf_sizes[i] *= req.result.get_col_num();
            buf_dps[i] = (int*) gmem->res_outbuf() + buf_sizes[i];
        }

        // TODO: Do we need to reverse rbuf here?
        reverse_result_buf();
    }

};





#endif  // USE_GPU
