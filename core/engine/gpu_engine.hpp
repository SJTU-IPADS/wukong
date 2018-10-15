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

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <vector>

#include "config.hpp"
#include "type.hpp"
#include "comm/adaptor.hpp"
#include "dgraph.hpp"
#include "query.hpp"
#include "assertion.hpp"
#include "math.hpp"
#include "timer.hpp"
#include "rmap.hpp"

#include "gpu_engine_impl.hpp"

using namespace std;

class GPUEngine {
private:
    int sid;    // server id
    int tid;    // thread id

    DGraph *graph;
    Adaptor *adaptor;
    Messenger *msgr;
    RMap rmap; // a map of replies for pending (fork-join) queries
    pthread_spinlock_t rmap_lock;

    GPUEngineImpl impl;

    tbb::concurrent_queue<SPARQLQuery> sub_queries;

    void index_to_unknown(SPARQLQuery &req) {
        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ssid_t tpid = pattern.subject;
        ssid_t id01 = pattern.predicate;
        dir_t d     = pattern.direction;
        ssid_t var  = pattern.object;
        SPARQLQuery::Result &res = req.result;

        ASSERT(id01 == PREDICATE_ID || id01 == TYPE_ID); // predicate or type index
        ASSERT(res.get_col_num() == 0);

        vector<sid_t> updated_result_table;

        uint64_t sz = 0;
        edge_t *edges = graph->get_index(tid, tpid, d, sz);
        int start = req.tid % req.mt_factor;
        int length = sz / req.mt_factor;

        // every thread takes a part of consecutive edges
        for (uint64_t k = start * length; k < (start + 1) * length; k++)
            updated_result_table.push_back(edges[k].val);

        // fixup the last participant
        if (start == req.mt_factor - 1)
            for (uint64_t k = (start + 1) * length; k < sz; k++)
                updated_result_table.push_back(edges[k].val);

        res.result_table.swap(updated_result_table);
        res.set_col_num(1);
        res.add_var2col(var, 0);
        req.pattern_step++;
        req.local_var = var;
    }

    // TODO
    void const_to_unknown(SPARQLQuery &req) {
        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ssid_t start = pattern.subject;
        ssid_t pid   = pattern.predicate;
        dir_t d      = pattern.direction;
        ssid_t end   = pattern.object;
        std::vector<sid_t> updated_result_table;
        SPARQLQuery::Result &res = req.result;

        ASSERT(res.get_col_num() == 0);

        uint64_t sz = 0;
        edge_t *edges = graph->get_triples(tid, start, d, pid, sz);
        for (uint64_t k = 0; k < sz; k++)
            updated_result_table.push_back(edges[k].val);

        res.result_table.swap(updated_result_table);
        res.add_var2col(end, res.get_col_num());
        res.set_col_num(res.get_col_num() + 1);
        req.pattern_step++;
    }

    void known_to_unknown(SPARQLQuery &req) {
        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ssid_t start = pattern.subject;
        ssid_t pid   = pattern.predicate;
        dir_t d      = pattern.direction;
        ssid_t end   = pattern.object;
        SPARQLQuery::Result &res = req.result;

        std::vector<sid_t> updated_result_table;

        logstream(LOG_DEBUG) << "#" << sid << " [begin] known_to_unknown: row_num=" << res.get_row_num() << ", step=" << req.pattern_step << LOG_endl;
        if (req.result.get_row_num() != 0) {
            ASSERT(nullptr != req.result.gpu.result_buf_dp);
            impl.known_to_unknown(req, start, pid, d, updated_result_table);
        }

        res.result_table.swap(updated_result_table);
        res.add_var2col(end, res.get_col_num());
        res.set_col_num(res.get_col_num() + 1);
        req.pattern_step++;
        // logstream(LOG_DEBUG) << "[end] known_to_unknown: GPU row_num=" << res.get_row_num() <<
            // ", col_num=" << res.get_col_num() << LOG_endl;

        logstream(LOG_DEBUG) << "#" << sid << "[end] GPU known_to_unknown: table_size=" << res.gpu.result_buf_nelems
            << ", row_num=" << res.get_row_num() << ", step=" << req.pattern_step << LOG_endl;
    }

    /// ?Y P ?X . (?Y and ?X are KNOWN)
    /// e.g.,
    ///
    /// 1) Use [?Y]+P1 to retrieve all of neighbors
    /// 2) Match [?Y]'s X within above neighbors
    void known_to_known(SPARQLQuery &req) {
        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ssid_t start = pattern.subject;
        ssid_t pid   = pattern.predicate;
        dir_t d      = pattern.direction;
        ssid_t end   = pattern.object;
        SPARQLQuery::Result &res = req.result;

        std::vector<sid_t> updated_result_table;

        logstream(LOG_DEBUG) << "#" << sid << " [begin] known_to_known: row_num=" << res.get_row_num() << ", step=" << req.pattern_step << LOG_endl;
        if (req.result.get_row_num() != 0) {
            ASSERT(nullptr != req.result.gpu.result_buf_dp);
            impl.known_to_known(req, start, pid, end, d, updated_result_table);
        }

        res.result_table.swap(updated_result_table);
        req.pattern_step++;
        logstream(LOG_DEBUG) << "#" << sid << "[end] GPU known_to_known: table_size=" << res.gpu.result_buf_nelems
            << ", row_num=" << res.get_row_num() << ", step=" << req.pattern_step << LOG_endl;
    }

    /// ?X P C . (?X is KNOWN)
    /// e.g.,
    ///
    /// 1) Use [?X]+P1 to retrieve all of neighbors
    /// 2) Match E1 within above neighbors
    void known_to_const(SPARQLQuery &req) {

        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ssid_t start = pattern.subject;
        ssid_t pid   = pattern.predicate;
        dir_t d      = pattern.direction;
        ssid_t end   = pattern.object;
        SPARQLQuery::Result &res = req.result;

        std::vector<sid_t> updated_result_table;

        logstream(LOG_DEBUG) << "#" << sid << " [begin] known_to_const: row_num=" << res.get_row_num() << ", step=" << req.pattern_step << LOG_endl;
        if (req.result.get_row_num() != 0) {
            ASSERT(nullptr != req.result.gpu.result_buf_dp);
            impl.known_to_const(req, start, pid, end, d, updated_result_table);
        }

        res.result_table.swap(updated_result_table);
        req.pattern_step++;
        // logstream(LOG_DEBUG) << "[end] known_to_const: row_num=" << res.get_row_num() << ", col_num=" << res.get_col_num() << LOG_endl;
        logstream(LOG_DEBUG) << "#" << sid << "[end] GPU known_to_const: table_size=" << res.gpu.result_buf_nelems
            << ", row_num=" << res.get_row_num() << ", step=" << req.pattern_step << LOG_endl;
    }

    // fork-join or in-place execution
    bool need_fork_join(SPARQLQuery &req) {
        // always need NOT fork-join when executing on single machine
        if (global_num_servers == 1) return false;

        // always need fork-join mode w/o RDMA
        if (!global_use_rdma) return true;

        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ASSERT(req.result.variable_type(pattern.subject) == known_var);
        //ssid_t start = pattern.subject;
        //return ((req.local_var != start)
        //        && (req.result.get_row_num() >= global_rdma_threshold));
        // GPUEngine only supports fork-join mode now
        sid_t start = req.get_pattern().subject;

        return ((req.local_var != start)
               && (req.result.get_row_num() >= 0));
    }


public:
    GPUEngine(int sid, int tid, GPUMem *gmem, GPUCache *gcache, GPUStreamPool *stream_pool, DGraph *graph)
        : sid(sid), tid(tid), impl(sid, gcache, gmem, stream_pool), graph(graph) {

    }

    ~GPUEngine() { }

    bool result_buf_ready(const SPARQLQuery &req) {
        if (req.result.result_table.empty()) {
            return true;
        } else {
            return req.result.gpu.result_buf_dp != nullptr;
        }
    }

    void load_result_buf(SPARQLQuery &req) {
        char *rbuf = impl.load_result_buf(req.result);
        req.result.set_gpu_result_buf(rbuf, req.result.result_table.size());
    }

    char* load_result_buf(SPARQLQuery &req, const string &rbuf_str) {
        char *rbuf = impl.load_result_buf(rbuf_str.c_str(), rbuf_str.size());
        req.result.set_gpu_result_buf(rbuf, rbuf_str.size() / WUKONG_GPU_ELEM_SIZE);
    }

    bool execute_one_pattern(SPARQLQuery &req) {
        ASSERT(result_buf_ready(req));
        ASSERT(!req.done(SPARQLQuery::SQState::SQ_PATTERN));

        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ssid_t start     = pattern.subject;
        ssid_t predicate = pattern.predicate;
        dir_t direction  = pattern.direction;
        ssid_t end       = pattern.object;

        if (req.pattern_step == 0 && req.start_from_index()) {
            if (req.result.var2col(end) != NO_RESULT)
                ASSERT("GPUEngine doesn't support index_to_known");
            else
                index_to_unknown(req);
            return true;
        }

        // triple pattern with UNKNOWN predicate/attribute
        if (req.result.variable_type(predicate) != const_var) {
            logstream(LOG_ERROR) << "Unsupported variable at predicate." << LOG_endl;
            logstream(LOG_ERROR) << "Please add definition VERSATILE in CMakeLists.txt." << LOG_endl;
            ASSERT(false);
        }

        // triple pattern with attribute
        if (global_enable_vattr && req.get_pattern(req.pattern_step).pred_type > 0) {
            ASSERT("GPUEngine doesn't support attr");
        }

        // triple pattern with KNOWN predicate
        switch (const_pair(req.result.variable_type(start),
                           req.result.variable_type(end))) {

        // start from CONST
        case const_pair(const_var, const_var):
            logstream(LOG_ERROR) << "Unsupported triple pattern [CONST|KNOWN|CONST]" << LOG_endl;
            ASSERT(false);
        case const_pair(const_var, known_var):
            logstream(LOG_ERROR) << "Unsupported triple pattern [CONST|KNOWN|KNOWN]" << LOG_endl;
            ASSERT(false);
            break;
        case const_pair(const_var, unknown_var):
            const_to_unknown(req);
            break;

        // start from KNOWN
        case const_pair(known_var, const_var):
            known_to_const(req);
            break;
        case const_pair(known_var, known_var):
            known_to_known(req);
            break;
        case const_pair(known_var, unknown_var):
            known_to_unknown(req);
            break;

        // start from UNKNOWN (incorrect query plan)
        case const_pair(unknown_var, const_var):
        case const_pair(unknown_var, known_var):
        case const_pair(unknown_var, unknown_var):
            logstream(LOG_ERROR) << "Unsupported triple pattern [UNKNOWN|KNOWN|??]" << LOG_endl;
            ASSERT(false);

        default:
            logstream(LOG_ERROR) << "Unsupported triple pattern with known predicate "
                                 << "(" << req.result.variable_type(start)
                                 << "|" << req.result.variable_type(end)
                                 << ")" << LOG_endl;
            ASSERT(false);
        }

        return true;
    }

    // TODO
    vector<SPARQLQuery> generate_sub_query(SPARQLQuery &req) {
        SPARQLQuery::Pattern &pattern = req.get_pattern();
        sid_t start = pattern.subject;

        // generate sub requests for all servers
        vector<SPARQLQuery> sub_reqs(global_num_servers);
        for (int i = 0; i < global_num_servers; i++) {
            sub_reqs[i].pid = req.id;
            sub_reqs[i].pg_type = req.pg_type == SPARQLQuery::PGType::UNION ?
                                  SPARQLQuery::PGType::BASIC : req.pg_type;
            sub_reqs[i].pattern_group = req.pattern_group;
            sub_reqs[i].pattern_step = req.pattern_step;
            sub_reqs[i].corun_step = req.corun_step;
            sub_reqs[i].fetch_step = req.fetch_step;
            sub_reqs[i].local_var = start;
            sub_reqs[i].priority = req.priority + 1;

            sub_reqs[i].job_type = SPARQLQuery::SubJobType::SPLIT_JOB;
            sub_reqs[i].result.dev_type = sub_reqs[i].dev_type = SPARQLQuery::DeviceType::GPU;

            sub_reqs[i].result.col_num = req.result.col_num;
            sub_reqs[i].result.attr_col_num = req.result.attr_col_num;
            sub_reqs[i].result.blind = req.result.blind;
            sub_reqs[i].result.v2c_map  = req.result.v2c_map;
            sub_reqs[i].result.nvars  = req.result.nvars;
        }

        ASSERT(req.pg_type != SPARQLQuery::PGType::OPTIONAL);

        if (global_num_servers == 1) {
            sub_reqs[0].result.set_gpu_result_buf(req.result.gpu.result_buf_dp, req.result.gpu.result_buf_nelems);
        } else {
            std::vector<sid_t*> buf_ptrs(global_num_servers);
            std::vector<int> buf_sizes(global_num_servers);

            impl.generate_sub_query(req, start, global_num_servers, buf_ptrs, buf_sizes);

            logstream(LOG_DEBUG) << "#" << sid << " generate_sub_query for req#" << req.id << ", parent: " << req.pid
                << ", step: " << req.pattern_step << LOG_endl;

            for (int i = 0; i < global_num_servers; ++i) {
                SPARQLQuery &r = sub_reqs[i];
                r.result.set_gpu_result_buf((char*)buf_ptrs[i], buf_sizes[i]);

                // if gpu history table is empty, set it to FULL_QUERY, which
                // will be sent by native RDMA
                if (r.result.gpu_result_buf_empty()) {
                    r.job_type = SPARQLQuery::SubJobType::FULL_JOB;
                    r.result.clear_gpu_result_buf();
                }
            }

        }

        return sub_reqs;
    }

};

#endif  // USE_GPU
