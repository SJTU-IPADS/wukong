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
#include <utility>

#include "global.hpp"
#include "assertion.hpp"
#include "query.hpp"

// gpu
#include "gpu_hash.hpp"
#include "gpu_cache.hpp"
#include "gpu_stream.hpp"

// utils
#include "gpu.hpp"
#include "logger2.hpp"

using namespace std;

class GPUEngineCuda final {
private:
    int sid;
    GPUMem *gmem;
    GPUCache *gcache;
    GPUStreamPool *stream_pool;
    GPUEngineParam param;

    // Since the result of current pattern will be the input of next pattern,
    // we need to reverse the pointers of result buffer to avoid memory copy.
    void reverse_result_buf() {
        gmem->reverse_rbuf();
    }

    segid_t pattern_to_segid(const SPARQLQuery &req, int pattern_id) {
        SPARQLQuery::Pattern patt = req.pattern_group.patterns[pattern_id];
        segid_t segid(0, patt.predicate, patt.direction);
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

    bool has_next_pattern(const SPARQLQuery &req) {
        return req.pattern_step + 1 < req.pattern_group.patterns.size();
    }


public:
    GPUEngineCuda(int sid, GPUCache *gcache, GPUMem *gmem, GPUStreamPool *stream_pool)
        : sid(sid), gcache(gcache), gmem(gmem), stream_pool(stream_pool),
          param(gcache->get_vertex_gaddr(), gcache->get_edge_gaddr(),
                gcache->get_num_key_blks(), gcache->get_num_value_blks(),
                gcache->get_nbuckets_kblk(), gcache->get_nentries_vblk()) {
    }

    ~GPUEngineCuda() { }

    char *load_result_buf(const SPARQLQuery::Result &r) {
        CUDA_ASSERT( cudaMemcpy((void**)gmem->res_inbuf(),
                                &r.result_table[0],
                                sizeof(r.result_table[0]) * r.result_table.size(),
                                cudaMemcpyHostToDevice) );

        return gmem->res_inbuf();
    }

    char *load_result_buf(const char *rbuf, uint64_t size) {
        CUDA_ASSERT( cudaMemcpy((void**)gmem->res_inbuf(),
                                rbuf,
                                size,
                                cudaMemcpyHostToDevice) );

        return gmem->res_inbuf();
    }

    vector<sid_t> index_to_unknown(SPARQLQuery &req, sid_t tpid, dir_t d) {
        ASSERT_MSG(false, "not implemented");
        return vector<sid_t>();
    }

    void known_to_unknown(SPARQLQuery &req, ssid_t start, ssid_t pid, dir_t d, vector<sid_t> &new_table) {
        cudaStream_t stream = stream_pool->get_stream(pid);
        segid_t current_seg = pattern_to_segid(req, req.pattern_step);
        rdf_seg_meta_t seg_meta = gcache->get_segment_meta(current_seg);

        logstream(LOG_DEBUG) << "known_to_unknown: segment: #buckets: " << seg_meta.num_buckets
                             << ", #edges: " << seg_meta.num_edges << "." << LOG_endl;


        param.query.start_vid = start;
        param.query.pid = pid;
        param.query.dir = d;
        param.query.col_num = req.result.get_col_num();
        param.query.row_num = req.result.gpu.get_row_num();
        param.query.segment_edge_start = seg_meta.edge_start;
        param.query.var2col_start = req.result.var2col(start);

        logstream(LOG_DEBUG) << "known_to_unknown: #ext_buckets: "
                             << seg_meta.get_ext_bucket_list_size() << LOG_endl;

        ASSERT(gmem->res_inbuf() != gmem->res_outbuf());
        ASSERT(nullptr != gmem->res_inbuf());

        // before processing the query, we should ensure the data of required predicates is loaded.
        vector<segid_t> required_segs = pattgrp_to_segids(req.pattern_group);

        if (!gcache->seg_in_cache(current_seg))
            gcache->load_segment(current_seg, required_segs, stream);


        // prefetch segment of next pattern
        if (Global::gpu_enable_pipeline && has_next_pattern(req)) {
            auto next_seg = pattern_to_segid(req, req.pattern_step + 1);
            auto stream2 = stream_pool->get_stream(next_seg.pid);

            if (!gcache->seg_in_cache(next_seg)) {
                gcache->prefetch_segment(next_seg, current_seg, required_segs, stream2);
            }
        }

        vector<uint64_t> vertex_mapping = gcache->get_vertex_mapping(current_seg);
        vector<uint64_t> edge_mapping = gcache->get_edge_mapping(current_seg);

        // load mapping tables and metadata of segment to GPU memory
        param.load_segment_mappings(vertex_mapping, edge_mapping, seg_meta, stream);
        param.load_segment_meta(seg_meta, stream);

        // setup result buffers on GPU
        if (req.pattern_step == 0) {
            param.set_result_bufs(gmem->res_inbuf(), gmem->res_outbuf());
        } else {    // for sub-query
            param.set_result_bufs(req.result.gpu.rbuf(), gmem->res_outbuf());
        }

        CUDA_STREAM_SYNC(stream);

        gpu_generate_key_list_k2u(param, stream);

        gpu_get_slot_id_list(param, stream);

        gpu_get_edge_list(param, stream);

        gpu_calc_prefix_sum(param, stream);

        int num_elems = gpu_update_result_buf_k2u(param);

#ifdef GPU_DEBUG
        logstream(LOG_INFO) << "#" << sid
                            << " GPU_update_result_buf_k2u done. #elems: " << num_elems
                            << ", #cols: " << param.query.col_num
                            << LOG_endl;
#endif

        ASSERT(WUKONG_GPU_RBUF_SIZE(num_elems) < gmem->res_buf_size());
        req.result.gpu.set_rbuf((char*)param.gpu.d_out_rbuf, num_elems);

        // copy the result on GPU to CPU if we come to the last pattern
        if (!has_next_pattern(req)) {
            new_table.resize(num_elems);
            thrust::device_ptr<int> dptr(param.gpu.d_out_rbuf);
            thrust::copy(dptr, dptr + num_elems, new_table.begin());
        } else {
            reverse_result_buf();
        }

    }

    void known_to_known(SPARQLQuery &req, ssid_t start, sid_t pid,
                        ssid_t end, dir_t d, vector<sid_t> &new_table) {

        cudaStream_t stream = stream_pool->get_stream(pid);
        segid_t current_seg = pattern_to_segid(req, req.pattern_step);
        rdf_seg_meta_t seg_meta = gcache->get_segment_meta(current_seg);

        logstream(LOG_DEBUG) << "known_to_known: segment: #buckets: " << seg_meta.num_buckets
                             << ", #edges: " << seg_meta.num_edges << "." << LOG_endl;


        param.query.start_vid = start;
        param.query.pid = pid;
        param.query.dir = d;
        param.query.end_vid = end;
        param.query.col_num = req.result.get_col_num();
        param.query.row_num = req.result.gpu.get_row_num();
        param.query.segment_edge_start = seg_meta.edge_start;
        param.query.var2col_start = req.result.var2col(start);
        param.query.var2col_end = req.result.var2col(end);

        ASSERT(gmem->res_inbuf() != gmem->res_outbuf());
        ASSERT(nullptr != gmem->res_inbuf());

        // before processing the query, we should ensure the data of required predicates is loaded.
        vector<segid_t> required_segs = pattgrp_to_segids(req.pattern_group);

        if (!gcache->seg_in_cache(current_seg))
            gcache->load_segment(current_seg, required_segs, stream);


        // preload next predicate
        if (Global::gpu_enable_pipeline && has_next_pattern(req)) {
            auto next_seg = pattern_to_segid(req, req.pattern_step + 1);
            auto stream2 = stream_pool->get_stream(next_seg.pid);

            if (!gcache->seg_in_cache(next_seg)) {
                gcache->prefetch_segment(next_seg, current_seg, required_segs, stream2);
            }
        }

        vector<uint64_t> vertex_mapping = gcache->get_vertex_mapping(current_seg);
        vector<uint64_t> edge_mapping = gcache->get_edge_mapping(current_seg);

        // copy metadata of segment to GPU memory
        param.load_segment_mappings(vertex_mapping, edge_mapping, seg_meta);
        param.load_segment_meta(seg_meta);

        if (req.pattern_step == 0) {
            param.set_result_bufs(gmem->res_inbuf(), gmem->res_outbuf());
        } else {
            param.set_result_bufs(req.result.gpu.rbuf(), gmem->res_outbuf());
        }

        gpu_generate_key_list_k2u(param, stream);

        gpu_get_slot_id_list(param, stream);

        gpu_get_edge_list_k2k(param, stream);

        gpu_calc_prefix_sum(param, stream);

        int num_elems = gpu_update_result_buf_k2k(param);

#ifdef GPU_DEBUG
        logstream(LOG_INFO) << "#" << sid
                            << " GPU_update_result_buf_k2k done. #elems: " << num_elems
                            << LOG_endl;
#endif

        ASSERT(WUKONG_GPU_RBUF_SIZE(num_elems) < gmem->res_buf_size());
        req.result.gpu.set_rbuf((char*)param.gpu.d_out_rbuf, num_elems);

        // copy the result on GPU to CPU if we come to the last pattern
        if (!has_next_pattern(req)) {
            new_table.resize(num_elems);
            thrust::device_ptr<int> dptr(param.gpu.d_out_rbuf);
            thrust::copy(dptr, dptr + num_elems, new_table.begin());
        } else {
            reverse_result_buf();
        }

    }

    void known_to_const(SPARQLQuery &req, ssid_t start, ssid_t pid,
                        ssid_t end, dir_t d, vector<sid_t> &new_table) {
        cudaStream_t stream = stream_pool->get_stream(pid);
        segid_t current_seg = pattern_to_segid(req, req.pattern_step);
        rdf_seg_meta_t seg_meta = gcache->get_segment_meta(current_seg);

        logstream(LOG_DEBUG) << "known_to_const: segment: #buckets: " << seg_meta.num_buckets
                             << ", #edges: " << seg_meta.num_edges << "." << LOG_endl;

        param.query.pid = pid;
        param.query.dir = d;
        param.query.end_vid = end;
        param.query.col_num = req.result.get_col_num();
        param.query.row_num = req.result.gpu.get_row_num();
        param.query.segment_edge_start = seg_meta.edge_start;
        param.query.var2col_start = req.result.var2col(start);

        ASSERT(gmem->res_inbuf() != gmem->res_outbuf());
        ASSERT(nullptr != gmem->res_inbuf());


        // before processing the query, we should ensure the data of required predicates is loaded.
        vector<segid_t> required_segs = pattgrp_to_segids(req.pattern_group);

        if (!gcache->seg_in_cache(current_seg))
            gcache->load_segment(current_seg, required_segs, stream);


        // preload next predicate
        if (Global::gpu_enable_pipeline && has_next_pattern(req)) {
            auto next_seg = pattern_to_segid(req, req.pattern_step + 1);
            auto stream2 = stream_pool->get_stream(next_seg.pid);

            if (!gcache->seg_in_cache(next_seg)) {
                gcache->prefetch_segment(next_seg, current_seg, required_segs, stream2);
            }
        }

        vector<uint64_t> vertex_mapping = gcache->get_vertex_mapping(current_seg);
        vector<uint64_t> edge_mapping = gcache->get_edge_mapping(current_seg);

        // copy metadata of segment to GPU memory
        param.load_segment_mappings(vertex_mapping, edge_mapping, seg_meta);
        param.load_segment_meta(seg_meta);
        // setup GPU engine parameters
        if (req.pattern_step == 0) {
            param.set_result_bufs(gmem->res_inbuf(), gmem->res_outbuf());
        } else {
            param.set_result_bufs(req.result.gpu.rbuf(), gmem->res_outbuf());
        }

        gpu_generate_key_list_k2u(param, stream);

        gpu_get_slot_id_list(param, stream);

        gpu_get_edge_list_k2c(param, stream);

        gpu_calc_prefix_sum(param, stream);

        int num_elems = gpu_update_result_buf_k2c(param);

#ifdef GPU_DEBUG
        logstream(LOG_INFO) << "#" << sid
                            << " GPU_update_result_buf_k2c done. #elems: " << num_elems
                            << LOG_endl;
#endif

        ASSERT(WUKONG_GPU_RBUF_SIZE(num_elems) < gmem->res_buf_size());
        req.result.gpu.set_rbuf((char*)param.gpu.d_out_rbuf, num_elems);

        // copy the result on GPU to CPU if we come to the last pattern
        if (!has_next_pattern(req)) {
            new_table.resize(num_elems);
            thrust::device_ptr<int> dptr(param.gpu.d_out_rbuf);
            thrust::copy(dptr, dptr + num_elems, new_table.begin());
        } else {
            reverse_result_buf();
        }

    }

    void generate_sub_query(const SPARQLQuery &req, sid_t start, int num_jobs,
                            vector<sid_t*>& buf_ptrs, vector<int>& buf_sizes) {
        cudaStream_t stream = stream_pool->get_split_query_stream();

        ASSERT(req.pattern_step > 0);

        param.query.start_vid = start;
        param.query.col_num = req.result.get_col_num();
        param.query.row_num = req.result.gpu.get_row_num();
        param.query.var2col_start = req.result.var2col(start);

        ASSERT(param.query.row_num > 0);

        if (req.pattern_step == 0) {
            param.set_result_bufs(gmem->res_inbuf(), gmem->res_outbuf());
        } else {
            param.set_result_bufs(req.result.gpu.rbuf(), gmem->res_outbuf());
        }

        vector<int> buf_heads(num_jobs);

        // shuffle records in result buffer according to server ID
        gpu_shuffle_result_buf(param, num_jobs, buf_sizes, buf_heads, stream);

        // split the result buffer of req
        gpu_split_result_buf(param, num_jobs, stream);

        CUDA_STREAM_SYNC(stream);

        for (int i = 0; i < num_jobs; ++i) {
            buf_sizes[i] *= req.result.get_col_num();
            buf_heads[i] *= req.result.get_col_num();
            buf_ptrs[i] = ((sid_t*) gmem->res_outbuf()) + buf_heads[i];

#ifdef GPU_DEBUG
            logstream(LOG_INFO) << "#" << sid << " i=" << i
                                << " sub_table_size=" << buf_sizes[i]
                                << ", sub_table_head=" << buf_heads[i]
                                << LOG_endl;
#endif
        }

        reverse_result_buf();
    }

};

#endif  // USE_GPU
