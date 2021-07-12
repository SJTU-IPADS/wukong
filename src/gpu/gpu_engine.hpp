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

#include "core/common/type.hpp"
#include "core/store/dgraph.hpp"
#include "core/sparql/query.hpp"

// gpu
#include "gpu_engine_cuda.hpp"
#include "gpu_channel.hpp"
#include "gpu_mem.hpp"

#include "core/engine/rmap.hpp"

#include "utils/assertion.hpp"
#include "utils/math.hpp"
#include "utils/timer.hpp"

#include <tbb/concurrent_queue.h>

namespace wukong {

class GPUEngine {
private:
    int sid;    // server id
    int tid;    // thread id

    DGraph *graph = nullptr;
    GPUMem *gmem = nullptr;

    RMap rmap; // a map of replies for pending (fork-join) queries
    pthread_spinlock_t rmap_lock;

    GPUEngineCuda backend;

    tbb::concurrent_queue<SPARQLQuery> sub_queries;

public:
    void index_to_unknown(SPARQLQuery &req) {
        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ssid_t tpid = pattern.subject;
        ssid_t id01 = pattern.predicate;
        dir_t d     = pattern.direction;
        ssid_t var  = pattern.object;
        SPARQLQuery::Result &res = req.result;

        ASSERT_MSG((res.var2col(var) == NO_RESULT), "GPUEngine doesn't support index_to_known");
        ASSERT(id01 == PREDICATE_ID || id01 == TYPE_ID); // predicate or type index
        ASSERT(res.get_col_num() == 0);

        std::vector<sid_t> updated_result_table;

        uint64_t sz = 0;
        edge_t *edges = graph->get_index(tid, tpid, d, sz);
        int start = req.mt_tid % req.mt_factor;
        int length = sz / req.mt_factor;

        // every thread takes a part of consecutive edges
        for (uint64_t k = start * length; k < (start + 1) * length; k++)
            updated_result_table.push_back(edges[k].val);

        // fixup the last participant
        if (start == req.mt_factor - 1)
            for (uint64_t k = (start + 1) * length; k < sz; k++)
                updated_result_table.push_back(edges[k].val);

        // update result and metadata
        res.result_table.swap(updated_result_table);
        res.set_col_num(1);
        res.add_var2col(var, 0);
        res.update_nrows();

        req.pattern_step++;
        req.local_var = var;
    }

    void const_to_unknown(SPARQLQuery &req) {
        logstream(LOG_INFO) << "#" << sid << " [CONST_TO_UNKNOWN] " << LOG_endl;
        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ssid_t start = pattern.subject;
        ssid_t pid   = pattern.predicate;
        dir_t d      = pattern.direction;
        ssid_t end   = pattern.object;
        std::vector<sid_t> updated_result_table;
        SPARQLQuery::Result &res = req.result;

        ASSERT(res.get_col_num() == 0);

        uint64_t sz = 0;
        edge_t *vids = graph->get_triples(tid, start, pid, d, sz);
        for (uint64_t k = 0; k < sz; k++)
            updated_result_table.push_back(vids[k].val);

        // update result and metadata
        res.result_table.swap(updated_result_table);
        res.add_var2col(end, res.get_col_num());
        res.set_col_num(res.get_col_num() + 1);
        res.update_nrows();

        req.pattern_step++;
    }

    void known_to_unknown(SPARQLQuery &req, GPUChannel &channel) {

        std::vector<sid_t> updated_result_table;
        SPARQLQuery::Result &res = req.result;

#ifdef WUKONG_GPU_DEBUG
        logstream(LOG_INFO) << "#" << sid << " [begin] " << ((req.combined) ? ("COMBINED") : "")
                            << " GPU known_to_unknown: table_size=" << res.gpu.table_size()
                            << ", row_num=" << res.get_row_num() << ", step=" << req.pattern_step << LOG_endl;
#endif

        if (!res.gpu.empty()) {
            if (req.combined) {
                backend.known_to_unknown_combined(req, updated_result_table, channel);
            } else {
                SPARQLQuery::Pattern &pattern = req.get_pattern();
                ssid_t start = pattern.subject;
                ssid_t pid   = pattern.predicate;
                dir_t d      = pattern.direction;
                backend.known_to_unknown(req, start, pid, d, updated_result_table, channel);
            }
        }

    }

    /// ?Y P ?X . (?Y and ?X are KNOWN)
    /// e.g.,
    ///
    /// 1) Use [?Y]+P1 to retrieve all of neighbors
    /// 2) Match [?Y]'s X within above neighbors
    void known_to_known(SPARQLQuery &req, GPUChannel &channel) {
        std::vector<sid_t> updated_result_table;
        SPARQLQuery::Result &res = req.result;

#ifdef WUKONG_GPU_DEBUG
        logstream(LOG_INFO) << "#" << sid << " [begin] " << ((req.combined) ? ("COMBINED") : "")
                            << " GPU known_to_known: table_size=" << res.gpu.table_size()
                            << ", row_num=" << res.get_row_num() << ", step=" << req.pattern_step << LOG_endl;
#endif

        if (!res.gpu.empty()) {
            if (req.combined) {
                backend.known_to_known_combined(req, updated_result_table, channel);
            } else {
                SPARQLQuery::Pattern &pattern = req.get_pattern();
                ssid_t start = pattern.subject;
                ssid_t pid   = pattern.predicate;
                dir_t d      = pattern.direction;
                ssid_t end   = pattern.object;
                backend.known_to_known(req, start, pid, end, d, updated_result_table, channel);
            }
        }
    }

    /// ?X P C . (?X is KNOWN)
    /// e.g.,
    ///
    /// 1) Use [?X]+P1 to retrieve all of neighbors
    /// 2) Match E1 within above neighbors
    void known_to_const(SPARQLQuery &req, GPUChannel &channel) {

        std::vector<sid_t> updated_result_table;
        SPARQLQuery::Result &res = req.result;

#ifdef WUKONG_GPU_DEBUG
        logstream(LOG_INFO) << "#" << sid << " [begin] " << ((req.combined) ? ("COMBINED") : "")
                            << " GPU known_to_const: table_size=" << res.gpu.table_size()
                            << ", row_num=" << res.get_row_num() << ", step=" << req.pattern_step << LOG_endl;
#endif

        if (!res.gpu.empty()) {
            if (req.combined) {
                backend.known_to_const_combined(req, updated_result_table, channel);
            } else {
                SPARQLQuery::Pattern &pattern = req.get_pattern();
                ssid_t start = pattern.subject;
                ssid_t pid   = pattern.predicate;
                dir_t d      = pattern.direction;
                ssid_t end   = pattern.object;
                backend.known_to_const(req, start, pid, end, d, updated_result_table, channel);
            }
        }

    }

    // GPUEngine(int sid, int tid, GPUMem *gmem, GPUCache *gcache, GPUStreamPool *stream_pool, DGraph *graph)
    //     : sid(sid), tid(tid), gmem(gmem), backend(sid, gcache, gmem, stream_pool), graph(graph) {

    // }
    GPUEngine(int sid, int tid, GPUMem *gmem, GPUCache *gcache, DGraph *graph)
        : sid(sid), tid(tid), gmem(gmem), backend(sid, gcache, gmem), graph(graph) {

    }

    ~GPUEngine() { }

    // check whether intermediate result (rbuf) is loaded into GPU
    // TODO: think about carefully how to check whether
    // the result buffer is ready (for combined and single)
    bool result_buf_ready(SPARQLQuery &req) {
        if (req.combined) {
            CombinedSPARQLQuery &combined = static_cast<CombinedSPARQLQuery&>(req);
            for (auto const& job : combined.get_jobs()) {
                if (!job.rbuf_info.loaded)
                    return false;
            }
            return true;
        }

        if (req.result.gpu.valid() && req.result.gpu.table_size() > 0)
            return true;
        // if (req.pattern_step > 0 && req.result.result_table.empty())
            // return true;
        if (req.result.gpu.valid() && req.result.result_table.empty())
            return true;

        return false;
    }

    void free_result_buf(int qid) {
        gmem->free_rbuf(qid);
    }

    void load_result_buf(SPARQLQuery &req, GPUChannel &channel) {
        size_t off = 0, table_size = 0;

        // allocate a result buffer
        ASSERT(gmem != nullptr);
        GPUMem::rbuf_t *rbuf_ptr = gmem->alloc_rbuf(req.qid);

        if (req.combined) {
            CombinedSPARQLQuery &combined = static_cast<CombinedSPARQLQuery&>(req);

            auto it = combined.get_jobs().begin();

            if (combined.staged) {
                off += backend.load_result_buf(0, combined.staged_table, rbuf_ptr->get_inbuf(), channel.get_stream());
                size_t total = 0;
                for (; it != combined.get_jobs().end(); ++it) {
                    rbuf_info_t& rbuf_info = it->rbuf_info;
                    ASSERT(rbuf_info.loaded == false);
                    rbuf_info.start_off = total;
                    rbuf_info.loaded = true;
                    //[QUESTION] why comment this line?
                    // it->req_ptr->result.gpu.set_rbuf(rbuf_ptr.get_inbuf() + rbuf_info.start_off, rbuf_info.size);
                    total += it->rbuf_info.size;
                    //[TODO] If we use non-blocking channel, the following assertion may be triggered!!
                    if (total >= off){
                        assert(false);
                        break;
                    }
                }

                table_size += total;
                combined.staged = false;

                // clear staged table
                std::vector<sid_t> empty_vec;
                combined.staged_table.swap(empty_vec);

            } else {
                off = combined.rbuf_offset();
            }

            // only copy result_table of patterns which are not loaded
            for (; it != combined.get_jobs().end(); ++it) {
                if (it->rbuf_info.loaded)
                    continue;

                SPARQLQuery::Result &res = it->req_ptr->result;
                size_t nvids = backend.load_result_buf(off, res.result_table, rbuf_ptr->get_inbuf(), channel.get_stream());
                res.gpu.set_rbuf(rbuf_ptr->get_inbuf() + off, nvids);
                res.set_device(SPARQLQuery::DeviceType::GPU);

                rbuf_info_t& rbuf_info = it->rbuf_info;
                rbuf_info.start_off = off;
                rbuf_info.size = nvids;
                rbuf_info.row_num = it->req_ptr->result.get_row_num();
                ASSERT(rbuf_info.row_num > 0);
                rbuf_info.loaded = true;

                off += nvids;
                table_size += nvids;
            }

        } else {
            table_size = req.result.result_table.size();
            backend.load_result_buf(off, req.result.result_table, rbuf_ptr->get_inbuf(), channel.get_stream());
        }

        req.result.gpu.set_rbuf(rbuf_ptr->get_inbuf(), table_size);
        req.result.set_device(SPARQLQuery::DeviceType::GPU);
    }

    /* Deprecated */
    void load_result_buf(SPARQLQuery &req, const std::string &rbuf_str) {
        auto nbytes = backend.load_result_buf(rbuf_str.data(), rbuf_str.size());
        req.result.gpu.set_rbuf(backend.get_res_inbuf(), nbytes / WUKONG_VID_SIZE);
    }

    bool execute_one_pattern(SPARQLQuery &req, GPUChannel &channel) {
        ASSERT(result_buf_ready(req));
        ASSERT(req.combined || !req.done(SPARQLQuery::SQState::SQ_PATTERN));

        // TODO: implement remaining handlers for combined job
        if (req.combined) {
            CombinedSPARQLQuery &combined = static_cast<CombinedSPARQLQuery &>(req);
            logstream(LOG_INFO) << "Execute a combined query, job size:" << 
            combined.combined_job_size() << LOG_endl;

            switch(combined.type) {
                case SPARQLQuery::PatternType::K2U:
                known_to_unknown(req, channel);
                break;

                case SPARQLQuery::PatternType::K2C:
                known_to_const(req, channel);
                break;

                case SPARQLQuery::PatternType::K2K:
                known_to_known(req, channel);
                break;

                default:
                ASSERT(false);
                break;
            }

            return true;
        }

        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ssid_t start     = pattern.subject;
        ssid_t predicate = pattern.predicate;
        dir_t direction  = pattern.direction;
        ssid_t end       = pattern.object;

        // triple pattern with UNKNOWN predicate/attribute
        if (req.result.var_stat(predicate) != CONST_VAR) {
            logstream(LOG_ERROR) << "Unsupported variable at predicate." << LOG_endl;
            logstream(LOG_ERROR) << "Please add definition VERSATILE in CMakeLists.txt." << LOG_endl;
            ASSERT(false);
        }

        // triple pattern with attribute
        if (Global::enable_vattr && req.get_pattern(req.pattern_step).pred_type != (char)SID_t) {
            ASSERT("GPUEngine doesn't support attribute");
        }

        // triple pattern with KNOWN predicate
        switch (const_pair(req.result.var_stat(start),
                           req.result.var_stat(end))) {

        // start from CONST
        case const_pair(CONST_VAR, CONST_VAR):
            logstream(LOG_ERROR) << "Unsupported triple pattern [CONST|KNOWN|CONST]" << LOG_endl;
            ASSERT(false);
        case const_pair(CONST_VAR, KNOWN_VAR):
            logstream(LOG_ERROR) << "Unsupported triple pattern [CONST|KNOWN|KNOWN]" << LOG_endl;
            ASSERT(false);
            break;
        case const_pair(CONST_VAR, UNKNOWN_VAR):
            logstream(LOG_ERROR) << "Unsupported triple pattern [CONST|KNOWN|KNOWN]" << LOG_endl;
            ASSERT(false);
            break;

        // start from KNOWN
        case const_pair(KNOWN_VAR, CONST_VAR):
            known_to_const(req, channel);
            break;
        case const_pair(KNOWN_VAR, KNOWN_VAR):
            known_to_known(req, channel);
            break;
        case const_pair(KNOWN_VAR, UNKNOWN_VAR):
            known_to_unknown(req, channel);
            break;

        // start from UNKNOWN (incorrect query plan)
        case const_pair(UNKNOWN_VAR, CONST_VAR):
        case const_pair(UNKNOWN_VAR, KNOWN_VAR):
        case const_pair(UNKNOWN_VAR, UNKNOWN_VAR):
            logstream(LOG_ERROR) << "Unsupported triple pattern [UNKNOWN|KNOWN|??]" << LOG_endl;
            ASSERT(false);

        default:
            logstream(LOG_ERROR) << "Unsupported triple pattern with known predicate "
                                 << "(" << req.result.var_stat(start)
                                 << "|" << req.result.var_stat(end)
                                 << ")" << LOG_endl;
            ASSERT(false);
        }

        return true;
    }

    std::vector<SPARQLQuery*> generate_sub_query(SPARQLQuery &req, GPUChannel &channel, int rbuf_key) {
        ASSERT(Global::num_servers > 1);
        SPARQLQuery::Pattern &pattern = req.get_pattern();
        sid_t start = pattern.subject;

        // generate sub requests for all servers
        std::vector<SPARQLQuery*> sub_reqs(Global::num_servers);
        for (int i = 0; i < Global::num_servers; i++) {
            SPARQLQuery *req_ptr = new SPARQLQuery;
            req_ptr->pqid = req.qid;
            req_ptr->pg_type = req.pg_type == SPARQLQuery::PGType::UNION ?
                                  SPARQLQuery::PGType::BASIC : req.pg_type;
            req_ptr->pattern_group = req.pattern_group;
            req_ptr->pattern_step = req.pattern_step;
            req_ptr->corun_step = req.corun_step;
            req_ptr->fetch_step = req.fetch_step;
            req_ptr->local_var = start;
            req_ptr->priority = req.priority + 1;

            req_ptr->job_type = SPARQLQuery::SubJobType::SPLIT_JOB;
            req_ptr->dev_type = SPARQLQuery::DeviceType::GPU;

            req_ptr->result.set_col_num(req.result.col_num);
            req_ptr->result.attr_col_num = req.result.attr_col_num;
            req_ptr->result.blind = req.result.blind;
            req_ptr->result.v2c_map  = req.result.v2c_map;
            req_ptr->result.nvars  = req.result.nvars;

            req_ptr->result.set_device(SPARQLQuery::DeviceType::GPU);
            sub_reqs[i] = req_ptr;
        }

        ASSERT(req.pg_type != SPARQLQuery::PGType::OPTIONAL);

        std::vector<sid_t*> buf_ptrs(Global::num_servers);
        std::vector<int> buf_sizes(Global::num_servers);

        backend.generate_sub_query(req, start, Global::num_servers, buf_ptrs, buf_sizes, channel, rbuf_key);

        logstream(LOG_DEBUG) << "#" << sid << " generate_sub_query for req#" << req.qid << ", parent: " << req.pqid
                             << ", step: " << req.pattern_step << ", col num: " << req.result.get_col_num()
                             << ", row num: " << req.result.get_row_num() << LOG_endl;

        for (int i = 0; i < Global::num_servers; ++i) {
            SPARQLQuery *req_ptr = sub_reqs[i];
            req_ptr->result.gpu.set_rbuf(buf_ptrs[i], buf_sizes[i]);

            logstream(LOG_DEBUG) << "#" << sid << " a sub_query #" << req_ptr->qid << " is generated, parent: " << req_ptr->pqid
                             << ", step: " << req_ptr->pattern_step << ", col num: " << req_ptr->result.get_col_num()
                             << ", row num: " << req_ptr->result.get_row_num() << LOG_endl;

            // if gpu history table is empty, return reply in advance
            if (req_ptr->result.gpu.empty()) {
                // req_ptr->job_type = SPARQLQuery::SubJobType::FULL_JOB;
                // req_ptr->result.gpu.clear();
                delete req_ptr;
                sub_reqs[i] = nullptr;
            }
        }

        return sub_reqs;
    }

    subquery_list_t split_giant_query(SPARQLQuery &req, GPUChannel &channel){
        return backend.split_giant_query(req, channel);
    }

};

}  // namespace wukong

#endif  // USE_GPU
