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

#include "core/common/global.hpp"
#include "utils/assertion.hpp"
#include "core/sparql/query.hpp"

// gpu
#include "gpu_hash.hpp"
#include "gpu_cache.hpp"
#include "gpu_stream.hpp"
#include "gpu_channel.hpp"
#include "gpu_utils.hpp"

// utils
#include "utils/logger2.hpp"

namespace wukong {

class GPUEngineCuda final {
private:
    int sid;
    GPUMem *gmem;
    GPUCache *gcache;


    /* [Deprecated]
    *  Since the result of current pattern will be the input of next pattern,
    *  we need to reverse the pointers of result buffer to avoid memory copy.
    */
    void reverse_result_buf() {
        gmem->reverse_rbuf();
    }

    /* @return a index segment id */
    segid_t pattern_to_segid(const SPARQLQuery &req, int pattern_id) {
        SPARQLQuery::Pattern patt = req.pattern_group.patterns[pattern_id];
        segid_t segid(0, patt.predicate, patt.direction);
        return segid;
    }

    std::vector<segid_t> pattgrp_to_segids(const SPARQLQuery::PatternGroup& pattgrp) {
        std::vector<segid_t> segids;
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
    GPUEngineCuda(int sid, GPUCache *gcache, GPUMem *gmem)
        : sid(sid), gcache(gcache), gmem(gmem) {
    }

    ~GPUEngineCuda() { }

    /* [Deprecated] */
    sid_t *get_res_inbuf() const {
        return gmem->res_inbuf();
    }

    /* [Deprecated] */
    sid_t *get_res_outbuf() const {
        return gmem->res_outbuf();
    }

    /* [Deprecated] */
    size_t load_result_buf(const char *rbuf, size_t size) {
        CUDA_ASSERT( cudaMemcpy((void**)gmem->res_inbuf(),
                                rbuf,
                                size,
                                cudaMemcpyHostToDevice) );

        return size;
    }

    /* @return: number of vertex loaded
    *  load result_table into GPUM[rbuf_d+dst_offset]  
    */
    size_t load_result_buf(size_t dst_offset, const std::vector<sid_t> &result_table, sid_t *rbuf_d) {
        size_t nvids = result_table.size();
        CUDA_ASSERT( cudaMemcpy((void**)(rbuf_d + dst_offset),
                                &result_table[0],
                                nvids * WUKONG_VID_SIZE,
                                cudaMemcpyHostToDevice) );

        return nvids;
    }

    /* @return: number of vertex loaded
    *  load result_table into GPUM[rbuf_d+dst_offset] (specify a CUDA stream)
    */
    size_t load_result_buf(size_t dst_offset, const std::vector<sid_t> &result_table, sid_t *rbuf_d, cudaStream_t stream) {
        size_t nvids = result_table.size();
        CUDA_ASSERT( cudaMemcpyAsync((void**)(rbuf_d + dst_offset),
                                &result_table[0],
                                nvids * WUKONG_VID_SIZE,
                                cudaMemcpyHostToDevice, stream) );

        return nvids;
    }

    std::vector<sid_t> index_to_unknown(SPARQLQuery &req, sid_t tpid, dir_t d) {
        ASSERT_MSG(false, "not implemented");
        return std::vector<sid_t>();
    }

    /* [For combined query]
    *  To calculate result buffer meta data(rbuf_info) for all jobs
    */
    size_t calculate_outbuf_ptrs(CombinedSPARQLQuery &combined, size_t total_row_num, int *prefix_sum_list_dp) {
        // copy prefix sum list to cpu
        std::vector<int> prefix_sums;
        prefix_sums.resize(total_row_num);
        thrust::device_ptr<int> dptr(prefix_sum_list_dp);
        thrust::copy(dptr, dptr + total_row_num, prefix_sums.begin());

        uint32_t prev_sum = 0, row_cnts = 0;
        size_t accum_table_sz = 0;

        for (auto &e : combined.get_jobs()) {
            row_cnts += e.rbuf_info.row_num;

            //[TODO] Not an elegant solution!
            int row_num;
            if(row_cnts == 0){
                //assert(false);
                row_num = 0;
            }else{
                row_num = prefix_sums[row_cnts - 1] - prev_sum;
            }

            e.rbuf_info.start_off = accum_table_sz;
            e.rbuf_info.row_num = row_num;

#ifdef PATTERN_COMBINE_DEBUG
            std::string str;
            switch (combined.type) {
                case SPARQLQuery::PatternType::K2U:
                    str = "k2u_combined: ";
                    break;
                case SPARQLQuery::PatternType::K2K:
                    str = "k2k_combined: ";
                    break;
                case SPARQLQuery::PatternType::K2C:
                    str = "k2c_combined: ";
                    break;
                default:
                    break;
            }

            logstream(LOG_EMPH) << str << "[calc_outbuf_ptrs] row_num=" << row_num << LOG_endl;
#endif

            switch (combined.type) {
                case SPARQLQuery::PatternType::K2U: {
                    size_t dim = row_num * (e.req_ptr->result.col_num + 1);
                    e.rbuf_info.size = dim;
                    accum_table_sz += dim;
                }
                    break;

                case SPARQLQuery::PatternType::K2C:
                case SPARQLQuery::PatternType::K2K: {
                    size_t dim = row_num * e.req_ptr->result.col_num;
                    e.rbuf_info.size = dim;
                    accum_table_sz += dim;
                }
                    break;

                default:
                    ASSERT(false);
            }
            assert(prev_sum == (prefix_sums[row_cnts - 1] - row_num));
            prev_sum = prefix_sums[row_cnts - 1];
        }

        return accum_table_sz;
    }

    void setup_pattern_infos(CombinedSPARQLQuery &combined, GPUChannel &channel) {
        GPUEngineParam &param = channel.para;
        cudaStream_t stream = channel.get_stream();
        pattern_info_t prev_patt;
        size_t total_row = 0;
        std::vector<segid_t> required_segs = combined.get_required_segids();

        uint64_t start = timer::get_usec();

        std::vector<rdf_seg_meta_t> seg_metas;
        std::vector<pattern_info_t> patterns;
        auto cur = combined.get_jobs().begin(), prev = cur;
        for ( ; cur != combined.get_jobs().end(); prev = cur, ++cur) {
            auto &patt = cur->req_ptr->get_pattern();

            segid_t segid(0, patt.predicate, patt.direction);
            rdf_seg_meta_t segmeta = gcache->get_segment_meta(segid);

            pattern_info_t patt_info;
            patt_info.start_vid = patt.subject;
            patt_info.end_vid = patt.object;
            patt_info.dir = patt.direction;
            patt_info.pid = patt.predicate;

            if (patt.subject < 0)
                patt_info.start_var2col = cur->req_ptr->result.var2col(patt.subject);
            if (patt.object < 0)
                patt_info.end_var2col = cur->req_ptr->result.var2col(patt.object);
            patt_info.col_num = cur->req_ptr->result.col_num;
            patt_info.row_num = cur->rbuf_info.row_num;
            //check consistency
            assert(patt_info.row_num == cur->req_ptr->result.get_row_num());

            // set offset to gpu result buffer
            patt_info.rbuf_start = cur->rbuf_info.start_off;

            patt_info.rbuf_start_row = prev_patt.rbuf_max_row;
            patt_info.rbuf_max_row = total_row + patt_info.row_num;

            // load blocks of those segments into gpu
            if (!gcache->seg_in_cache(segid))
                gcache->load_segment(segid, required_segs, stream);

            // load key/value block mapping into gpu
            std::vector<uint64_t> key_mapping = gcache->create_key_mapping(segid);
            std::vector<uint64_t> value_mapping = gcache->create_value_mapping(segid);

            ASSERT(key_mapping.size() == segmeta.num_key_blks);
            ASSERT(value_mapping.size() == segmeta.num_value_blks);

            patt_info.key_mapping_dptr = param.load_key_mappings(key_mapping, segid, segmeta, stream);
            patt_info.value_mapping_dptr = param.load_value_mappings(value_mapping, segid, segmeta, stream);

            //param.load_pattern_meta(segmeta, patt_info, stream);
            seg_metas.push_back(segmeta);
            patterns.push_back(patt_info);

#ifdef PATTERN_COMBINE_DEBUG
            logstream(LOG_INFO) << "pattern#" << ": step: " << cur->req_ptr->pattern_step
                                << ", row_num: " << patt_info.row_num
                                << ", col_num: " << patt_info.col_num
                                << ", [rbuf_start: " << patt_info.rbuf_start
                                << ", start_row: " << patt_info.rbuf_start_row
                                << ", max_row: " << patt_info.rbuf_max_row << "]" << LOG_endl;
#endif

            total_row += patt_info.row_num;
            prev_patt = patt_info;
        }
        // load metadata of triple pattern to gpu
        param.load_pattern_metas(seg_metas, patterns, stream);
        param.query.row_num = total_row;

        logstream(LOG_INFO) << "Setup pattern info time:" << timer::get_usec() - start << LOG_endl;
    }

    void setup_updated_pattern_infos(CombinedSPARQLQuery &combined, GPUChannel &channel) {
        cudaStream_t stream = channel.get_stream();
        GPUEngineParam &param = channel.para;

        std::vector<updated_pattern_info_t> updated_patt_infos;
        updated_pattern_info_t updated_info, prev_updated_info;

        size_t total_row = 0;
        auto cur = combined.get_jobs().begin(), prev = cur;
        for (; cur != combined.get_jobs().end(); prev = cur, ++cur) {
            updated_info.rbuf_start = cur->rbuf_info.start_off;
            updated_info.rbuf_start_row = prev_updated_info.rbuf_max_row;
            updated_info.rbuf_max_row = total_row + cur->rbuf_info.row_num;
            if (combined.type == SPARQLQuery::PatternType::K2U)
                updated_info.col_num = cur->req_ptr->result.col_num + 1;
            else
                updated_info.col_num = cur->req_ptr->result.col_num;

            updated_patt_infos.push_back(updated_info);

#ifdef PATTERN_COMBINE_DEBUG
            logstream(LOG_INFO) << "updated_pattern#"
                                << ", row_num: " << (updated_info.rbuf_max_row - updated_info.rbuf_start_row)
                                << ", [rbuf_start: " << updated_info.rbuf_start
                                << ", start_row: " << updated_info.rbuf_start_row
                                << ", max_row: " << updated_info.rbuf_max_row << "]" << LOG_endl;
#endif

            total_row += cur->rbuf_info.row_num;
            prev_updated_info = updated_info;
        }

        CUDA_ASSERT(cudaMemcpyAsync(param.gpu.updated_patt_infos_d,
                    &updated_patt_infos[0], updated_patt_infos.size() * sizeof(updated_pattern_info_t),
                    cudaMemcpyHostToDevice, stream));
    }

    void known_to_unknown_combined(SPARQLQuery &req, std::vector<sid_t> &new_table, GPUChannel &channel) {
        CombinedSPARQLQuery &combined = static_cast<CombinedSPARQLQuery &>(req);
        cudaStream_t stream = channel.get_stream();
        GPUMem::rbuf_t *rbuf_ptr = gmem->get_allocated_rbuf(req.qid);
        GPUEngineParam &param = channel.para;

        // ignore states in the result member
        ASSERT(combined.result.row_num == 0);
        ASSERT(combined.get_jobs().size() <= Global::pattern_combine_window);

        // setup result buffers on GPU
        param.set_result_bufs(rbuf_ptr->get_inbuf(), rbuf_ptr->get_outbuf());

        setup_pattern_infos(combined, channel);

        // (we don't do prefetching in combined mode)
        // launch gpu kernels for known_to_unknown

        gpu_generate_key_list_combined(param, stream);

        gpu_get_slot_id_list_combined(param, stream);

        gpu_get_edge_list_combined(param, stream);

        gpu_calc_prefix_sum(param, stream);

        CUDA_STREAM_SYNC(stream);

        // calculate medium jobs offset to outbuf
        size_t table_size = calculate_outbuf_ptrs(combined, param.query.row_num, param.gpu.d_prefix_sum_list);

        setup_updated_pattern_infos(combined, channel);

        if(WUKONG_GPU_RBUF_SIZE(table_size) <= MiB2B(Global::gpu_rbuf_size_mb)){
            gpu_update_result_buf_k2u_combined(param, stream);
            // use the output rbuf to set the gpu rbuf of combined job
            combined.result.gpu.set_rbuf(param.gpu.d_out_rbuf, table_size);

            combined.pattern_step = combined.pattern_step + 1;
            rbuf_ptr->reverse();
        }else{
            channel.error_code = GPUErrorCode::GIANT_TOTAL_RESULT_TABLE;
            channel.error_info = (void*) new int(table_size);
        }
    }

 void known_to_unknown(SPARQLQuery &req, ssid_t start, ssid_t pid, dir_t d, std::vector<sid_t> &new_table, GPUChannel &channel) {
        cudaStream_t stream = channel.get_stream();
        GPUMem::rbuf_t *rbuf_ptr = gmem->get_allocated_rbuf(req.qid);

        segid_t current_seg = pattern_to_segid(req, req.pattern_step);
        rdf_seg_meta_t seg_meta = gcache->get_segment_meta(current_seg);

        logstream(LOG_DEBUG) << "known_to_unknown: segment: #buckets: " << seg_meta.num_buckets
                             << ", #edges: " << seg_meta.num_edges << "." << LOG_endl;

        GPUEngineParam &param = channel.para;
        param.query.start_vid = start;
        param.query.pid = pid;
        param.query.dir = d;
        param.query.col_num = req.result.get_col_num();
        param.query.row_num = req.result.get_row_num();
        param.query.segment_edge_start = seg_meta.edge_start;
        param.query.var2col_start = req.result.var2col(start);

        logstream(LOG_DEBUG) << "known_to_unknown: #ext_buckets: "
                             << seg_meta.get_ext_bucket_list_size() << LOG_endl;

        ASSERT(rbuf_ptr->get_inbuf() != rbuf_ptr->get_outbuf());

        // before processing the query, we should ensure the data of required predicates is loaded.
        std::vector<segid_t> required_segs = pattgrp_to_segids(req.pattern_group);

        if (!gcache->seg_in_cache(current_seg))
            gcache->load_segment(current_seg, required_segs, stream);


        // prefetch segment of next pattern
        if (Global::gpu_enable_pipeline && has_next_pattern(req)) {
            auto next_seg = pattern_to_segid(req, req.pattern_step + 1);
            auto stream2 = GPUStreamPool::get_pool().get_stream(next_seg.pid);

            if (!gcache->seg_in_cache(next_seg)) {
                gcache->prefetch_segment(next_seg, current_seg, required_segs, stream2);
            }
        }

        std::vector<uint64_t> vertex_mapping = gcache->create_key_mapping(current_seg);
        std::vector<uint64_t> edge_mapping = gcache->create_value_mapping(current_seg);

        // load mapping tables and metadata of segment to GPU memory
        param.load_segment_mappings(vertex_mapping, edge_mapping, seg_meta, stream);
        param.load_segment_meta(seg_meta, stream);

        // setup result buffers on GPU
        param.set_result_bufs(rbuf_ptr->get_inbuf(), rbuf_ptr->get_outbuf());

        gpu_generate_key_list(param, stream);

        gpu_get_slot_id_list(param, stream);

        gpu_get_edge_list(param, stream);

        gpu_calc_prefix_sum(param, stream);

        /* Check for giant result table */
        std::vector<int> row_nums(param.query.row_num);
        int new_row_num = 0;
        CUDA_ASSERT( cudaMemcpyAsync(&new_row_num, param.gpu.d_prefix_sum_list + param.query.row_num - 1, sizeof(int), cudaMemcpyDeviceToHost, stream));
        /* [FIXME] Now for error check, a synchronization here is necessary.
         * But performance is reduced seriously, an async-callback solution is better. 
         */
        CUDA_ASSERT( cudaMemcpyAsync(&row_nums[0], param.gpu.d_prefix_sum_list, sizeof(int)*param.query.row_num, cudaMemcpyDeviceToHost, stream));
        CUDA_STREAM_SYNC(stream);
        int table_size = new_row_num * (param.query.col_num + 1);

        if(WUKONG_GPU_RBUF_SIZE(table_size) <= MiB2B(Global::gpu_rbuf_size_mb)){
            gpu_update_result_buf_k2u(param, stream);
            rbuf_ptr->reverse();

            logstream(LOG_DEBUG) << "#" << sid
                            //<< " GPU_update_result_buf_k2u done. #elems: " << num_elems
                            << ", #cols: " << param.query.col_num
                            << LOG_endl;
        }else{
            channel.error_code = GPUErrorCode::GIANT_TOTAL_RESULT_TABLE;
            channel.error_info = (void*) new int(new_row_num);
        }
    }

    subquery_list_t split_giant_query(SPARQLQuery &req, GPUChannel &channel){
        cudaStream_t stream = channel.get_stream();
        GPUEngineParam &param = channel.para;
        GPUMem::rbuf_t *rbuf_ptr = gmem->get_allocated_rbuf(req.qid);

        int new_row_num = *((int*)channel.error_info);
        int table_size = new_row_num * (param.query.col_num + 1);

        int num_jobs = WUKONG_GPU_RBUF_SIZE(table_size) / MiB2B(Global::gpu_rbuf_size_mb);
        // prevent useless split and get stuck in infinite loop
        if(num_jobs == 1) num_jobs++;
        std::vector<int> buf_offs(num_jobs);
        // split the result buffer of req
        gpu_split_giant_query(param, param.query.row_num, param.query.col_num, num_jobs, new_row_num, buf_offs, stream);
        //delete uncoverd query offset, reset job number
        std::remove(buf_offs.begin(), buf_offs.end(), -1);
        num_jobs = buf_offs.size();
        //if this assertion failed, there is a power-law vertex in query result , which means we can't continue this query
        if(num_jobs < 1) assert(false);

        subquery_list_t sub_reqs(num_jobs);
        for (int i = 0; i < num_jobs; ++i) {
            SPARQLQuery *req_ptr = new SPARQLQuery;
            req_ptr->pqid = req.qid;
            req_ptr->pg_type = req.pg_type == SPARQLQuery::PGType::UNION ?
                                SPARQLQuery::PGType::BASIC : req.pg_type;
            req_ptr->pattern_group = req.pattern_group;
            req_ptr->pattern_step = req.pattern_step;
            req_ptr->corun_step = req.corun_step;
            req_ptr->fetch_step = req.fetch_step;
            req_ptr->local_var = req.local_var;
            req_ptr->priority = req.priority + 1;

            req_ptr->job_type = SPARQLQuery::SubJobType::SPLIT_JOB;
            req_ptr->dev_type = SPARQLQuery::DeviceType::GPU;

            req_ptr->result.set_col_num(req.result.col_num);
            req_ptr->result.attr_col_num = req.result.attr_col_num;
            req_ptr->result.blind = req.result.blind;
            req_ptr->result.v2c_map  = req.result.v2c_map;
            req_ptr->result.nvars  = req.result.nvars;
            req_ptr->result.set_device(SPARQLQuery::DeviceType::GPU);

            int buf_size = (i==(num_jobs-1)) ? (param.query.row_num * param.query.col_num-buf_offs[i])
                                                                               : (buf_offs[i+1] - buf_offs[i]);
            sid_t* buf_ptr = ((sid_t*) rbuf_ptr->get_inbuf()) + buf_offs[i];
    
            req_ptr->result.gpu.set_rbuf(buf_ptr, buf_size);
            // if gpu history table is empty, return reply in advance
            if (req_ptr->result.gpu.empty()) {
                delete req_ptr;
                req_ptr = nullptr;
            }
            sub_reqs[i] = req_ptr;
        }
        return sub_reqs;
    }

    // TODO
    void known_to_known_combined(SPARQLQuery &req, std::vector<sid_t> &new_table, GPUChannel &channel) {
        CombinedSPARQLQuery &combined = static_cast<CombinedSPARQLQuery &>(req);
        cudaStream_t stream = channel.get_stream();
        GPUMem::rbuf_t *rbuf_ptr = gmem->get_allocated_rbuf(req.qid);
        GPUEngineParam &param = channel.para;

        // ignore states in the result member
        ASSERT(combined.result.row_num == 0);
        ASSERT(combined.get_jobs().size() <= Global::pattern_combine_window);

        // setup result buffers on GPU
        param.set_result_bufs(rbuf_ptr->get_inbuf(), rbuf_ptr->get_outbuf());

        setup_pattern_infos(combined, channel);

        // (we don't do prefetching in combined mode)
        // launch gpu kernels for known_to_unknown

        gpu_generate_key_list_combined(param, stream);

        gpu_get_slot_id_list_combined(param, stream);

        gpu_get_edge_list_k2k_combined(param, stream);

        gpu_calc_prefix_sum(param, stream);

        // calculate medium jobs offset to outbuf
        size_t table_size = calculate_outbuf_ptrs(combined, param.query.row_num, param.gpu.d_prefix_sum_list);

        setup_updated_pattern_infos(combined, channel);

        gpu_update_result_buf_k2c_combined(param, stream);

        ASSERT(WUKONG_GPU_RBUF_SIZE(table_size) < gmem->res_buf_size());

        // use the output rbuf to set the gpu rbuf of combined job
        combined.result.gpu.set_rbuf(param.gpu.d_out_rbuf, table_size);

        ++combined.pattern_step;
        rbuf_ptr->reverse();

        // CUDA_STREAM_SYNC(stream);
    }

    void known_to_known(SPARQLQuery &req, ssid_t start, sid_t pid,
                        ssid_t end, dir_t d, std::vector<sid_t> &new_table, GPUChannel &channel) {
        cudaStream_t stream = channel.get_stream();
        GPUMem::rbuf_t *rbuf_ptr = gmem->get_allocated_rbuf(req.qid);

        segid_t current_seg = pattern_to_segid(req, req.pattern_step);
        rdf_seg_meta_t seg_meta = gcache->get_segment_meta(current_seg);

        logstream(LOG_DEBUG) << "known_to_known: segment: #buckets: " << seg_meta.num_buckets
                             << ", #edges: " << seg_meta.num_edges << "." << LOG_endl;

        GPUEngineParam &param = channel.para;
        param.query.start_vid = start;
        param.query.pid = pid;
        param.query.dir = d;
        param.query.end_vid = end;
        param.query.col_num = req.result.get_col_num();
        param.query.row_num = req.result.get_row_num();
        param.query.segment_edge_start = seg_meta.edge_start;
        param.query.var2col_start = req.result.var2col(start);
        param.query.var2col_end = req.result.var2col(end);

        ASSERT(rbuf_ptr->get_inbuf() != rbuf_ptr->get_outbuf());

        // before processing the query, we should ensure the data of required predicates is loaded.
        std::vector<segid_t> required_segs = pattgrp_to_segids(req.pattern_group);

        if (!gcache->seg_in_cache(current_seg))
            gcache->load_segment(current_seg, required_segs, stream);


        // preload next predicate
        if (Global::gpu_enable_pipeline && has_next_pattern(req)) {
            auto next_seg = pattern_to_segid(req, req.pattern_step + 1);
            auto stream2 = GPUStreamPool::get_pool().get_stream(next_seg.pid);

            if (!gcache->seg_in_cache(next_seg)) {
                gcache->prefetch_segment(next_seg, current_seg, required_segs, stream2);
            }
        }

        std::vector<uint64_t> vertex_mapping = gcache->create_key_mapping(current_seg);
        std::vector<uint64_t> edge_mapping = gcache->create_value_mapping(current_seg);

        // copy metadata of segment to GPU memory
        param.load_segment_mappings(vertex_mapping, edge_mapping, seg_meta);
        param.load_segment_meta(seg_meta);

        // setup result buffers on GPU
        param.set_result_bufs(rbuf_ptr->get_inbuf(), rbuf_ptr->get_outbuf());

        gpu_generate_key_list(param, stream);

        gpu_get_slot_id_list(param, stream);

        gpu_get_edge_list_k2k(param, stream);

        gpu_calc_prefix_sum(param, stream);

        gpu_update_result_buf_k2k(param, stream);

        rbuf_ptr->reverse();

        #ifdef GPU_DEBUG
        logstream(LOG_INFO) << "#" << sid
                            << " GPU_update_result_buf_k2k done."
                            << LOG_endl;
        #endif
    }

    void known_to_const_combined(SPARQLQuery &req, std::vector<sid_t> &new_table, GPUChannel &channel) {
        CombinedSPARQLQuery &combined = static_cast<CombinedSPARQLQuery &>(req);
        cudaStream_t stream = channel.get_stream();
        GPUMem::rbuf_t *rbuf_ptr = gmem->get_allocated_rbuf(req.qid);
        GPUEngineParam &param = channel.para;

        // ignore states in the result member
        ASSERT(combined.result.row_num == 0);
        ASSERT(combined.get_jobs().size() <= Global::pattern_combine_window);

        // setup result buffers on GPU
        param.set_result_bufs(rbuf_ptr->get_inbuf(), rbuf_ptr->get_outbuf());

        // TODO: replace stream with Channel abstraction

        setup_pattern_infos(combined, channel);

        // (we don't do prefetching in combined mode)
        // launch gpu kernels for known_to_unknown

        gpu_generate_key_list_combined(param, stream);

        gpu_get_slot_id_list_combined(param, stream);

        gpu_get_edge_list_k2c_combined(param, stream);

        gpu_calc_prefix_sum(param, stream);

        CUDA_STREAM_SYNC(stream);

        // calculate medium jobs offset to outbuf
        size_t table_size = calculate_outbuf_ptrs(combined, param.query.row_num, param.gpu.d_prefix_sum_list);

        setup_updated_pattern_infos(combined, channel);

        gpu_update_result_buf_k2c_combined(param, stream);

        ASSERT(WUKONG_GPU_RBUF_SIZE(table_size) < gmem->res_buf_size());

        // use the output rbuf to set the gpu rbuf of combined job
        combined.result.gpu.set_rbuf(param.gpu.d_out_rbuf, table_size);

        ++combined.pattern_step;
        rbuf_ptr->reverse();

        // CUDA_STREAM_SYNC(stream);
    }

    void known_to_const(SPARQLQuery &req, ssid_t start, ssid_t pid,
                        ssid_t end, dir_t d, std::vector<sid_t> &new_table, GPUChannel &channel) {
        cudaStream_t stream = channel.get_stream();
        GPUMem::rbuf_t *rbuf_ptr = gmem->get_allocated_rbuf(req.qid);
        segid_t current_seg = pattern_to_segid(req, req.pattern_step);
        rdf_seg_meta_t seg_meta = gcache->get_segment_meta(current_seg);

        logstream(LOG_DEBUG) << "known_to_const: segment: #buckets: " << seg_meta.num_buckets
                             << ", #edges: " << seg_meta.num_edges << "." << LOG_endl;

        GPUEngineParam &param = channel.para;
        param.query.pid = pid;
        param.query.dir = d;
        param.query.end_vid = end;
        param.query.col_num = req.result.get_col_num();
        param.query.row_num = req.result.get_row_num();
        param.query.segment_edge_start = seg_meta.edge_start;
        param.query.var2col_start = req.result.var2col(start);

        ASSERT(rbuf_ptr->get_inbuf() != rbuf_ptr->get_outbuf());

        // before processing the query, we should ensure the data of required predicates is loaded.
        std::vector<segid_t> required_segs = pattgrp_to_segids(req.pattern_group);

        if (!gcache->seg_in_cache(current_seg))
            gcache->load_segment(current_seg, required_segs, stream);


        // preload next predicate
        if (Global::gpu_enable_pipeline && has_next_pattern(req)) {
            auto next_seg = pattern_to_segid(req, req.pattern_step + 1);
            auto stream2 = GPUStreamPool::get_pool().get_stream(next_seg.pid);

            if (!gcache->seg_in_cache(next_seg)) {
                gcache->prefetch_segment(next_seg, current_seg, required_segs, stream2);
            }
        }

        std::vector<uint64_t> vertex_mapping = gcache->create_key_mapping(current_seg);
        std::vector<uint64_t> edge_mapping = gcache->create_value_mapping(current_seg);

        // copy metadata of segment to GPU memory
        param.load_segment_mappings(vertex_mapping, edge_mapping, seg_meta, stream);
        param.load_segment_meta(seg_meta, stream);
        // setup GPU engine parameters
        param.set_result_bufs(rbuf_ptr->get_inbuf(), rbuf_ptr->get_outbuf());

        gpu_generate_key_list(param, stream);

        gpu_get_slot_id_list(param, stream);

        gpu_get_edge_list_k2c(param, stream);

        gpu_calc_prefix_sum(param, stream);

        // calculate medium jobs offset to outbuf
        // int num_elems = gpu_update_result_buf_k2c(param);
        gpu_update_result_buf_k2c(param, stream);

#ifdef GPU_DEBUG
        logstream(LOG_INFO) << "#" << sid
                            << " GPU_update_result_buf_k2c done. #elems: " << num_elems
                            << LOG_endl;
#endif
        rbuf_ptr->reverse();
    }

    void generate_sub_query(SPARQLQuery &req, sid_t start, int num_jobs,
                            std::vector<sid_t*>& buf_ptrs, std::vector<int>& buf_sizes, 
                            GPUChannel &channel, int rbuf_key) {
        cudaStream_t stream = channel.get_stream(); //GPUStreamPool::get_pool().get_split_query_stream();
        GPUEngineParam &param = channel.para;
        GPUMem::rbuf_t *rbuf_ptr = gmem->get_allocated_rbuf(rbuf_key);

        ASSERT(req.pattern_step > 0);

        param.query.start_vid = start;
        param.query.col_num = req.result.get_col_num();
        param.query.row_num = req.result.get_row_num();
        param.query.var2col_start = req.result.var2col(start);

        ASSERT(param.query.row_num > 0);
        
        //param.set_result_bufs(rbuf_ptr->get_inbuf(), rbuf_ptr->get_outbuf());
        //we need to adapt to both combined query and single query
        param.set_result_bufs(req.result.gpu.rbuf(), rbuf_ptr->get_outbuf());

        std::vector<int> buf_heads(num_jobs);

        // shuffle records in result buffer according to server ID
        gpu_shuffle_result_buf(param, num_jobs, buf_sizes, buf_heads, stream);

        // split the result buffer of req
        gpu_split_result_buf(param, num_jobs, stream);

        CUDA_STREAM_SYNC(stream);

        for (int i = 0; i < num_jobs; ++i) {
            buf_sizes[i] *= req.result.get_col_num();
            buf_heads[i] *= req.result.get_col_num();
            buf_ptrs[i] = ((sid_t*) rbuf_ptr->get_outbuf()) + buf_heads[i];

#ifdef GPU_DEBUG
            logstream(LOG_INFO) << "#" << sid << " i=" << i
                                << " sub_table_size=" << buf_sizes[i]
                                << ", sub_table_head=" << buf_heads[i]
                                << LOG_endl;
#endif
        }

        // We should not reverse result buffer
        //rbuf_ptr->reverse();
    }

};

}  // namespace wukong

#endif  // USE_GPU
