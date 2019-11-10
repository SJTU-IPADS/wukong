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

#include "type.hpp"
#include "dgraph.hpp"
#include "query.hpp"

// gpu
#include "gpu_engine_cuda.hpp"

#include "comm/adaptor.hpp"
#include "engine/rmap.hpp"

#include "assertion.hpp"
#include "math.hpp"
#include "timer.hpp"


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

    GPUEngineCuda backend;

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

    void known_to_unknown(SPARQLQuery &req) {
        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ssid_t start = pattern.subject;
        ssid_t pid   = pattern.predicate;
        dir_t d      = pattern.direction;
        ssid_t end   = pattern.object;
        SPARQLQuery::Result &res = req.result;

        std::vector<sid_t> updated_result_table;

        logstream(LOG_DEBUG) << "#" << sid
                             << " [begin] known_to_unknown: row_num=" << res.gpu.get_row_num()
                             << ", step=" << req.pattern_step << LOG_endl;
        if (!res.gpu.is_rbuf_empty()) {
            backend.known_to_unknown(req, start, pid, d, updated_result_table);
        }

        // update result and metadata
        res.result_table.swap(updated_result_table);
        res.add_var2col(end, res.get_col_num());
        res.set_col_num(res.get_col_num() + 1);
        res.update_nrows();

        req.pattern_step++;
        logstream(LOG_DEBUG) << "#" << sid
                             << "[end] GPU known_to_unknown: table_size=" << res.gpu.rbuf_num_elems()
                             << ", row_num=" << res.gpu.get_row_num() << ", step=" << req.pattern_step
                             << LOG_endl;
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

        logstream(LOG_DEBUG) << "#" << sid
                             << " [begin] known_to_known: row_num=" << res.gpu.get_row_num()
                             << ", step=" << req.pattern_step
                             << LOG_endl;
        if (!res.gpu.is_rbuf_empty()) {
            backend.known_to_known(req, start, pid, end, d, updated_result_table);
        }

        // update result and metadata
        res.result_table.swap(updated_result_table);
        res.update_nrows();

        req.pattern_step++;
        logstream(LOG_DEBUG) << "#" << sid << "[end] GPU known_to_known: table_size=" << res.gpu.rbuf_num_elems()
                             << ", row_num=" << res.gpu.get_row_num() << ", step=" << req.pattern_step
                             << LOG_endl;
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

        logstream(LOG_DEBUG) << "#" << sid << " [begin] known_to_const: row_num=" << res.gpu.get_row_num()
                             << ", step=" << req.pattern_step << LOG_endl;
        if (!res.gpu.is_rbuf_empty()) {
            backend.known_to_const(req, start, pid, end, d, updated_result_table);
        }

        // update result and metadata
        res.result_table.swap(updated_result_table);
        res.update_nrows();

        req.pattern_step++;
        logstream(LOG_DEBUG) << "#" << sid << "[end] GPU known_to_const: table_size=" << res.gpu.rbuf_num_elems()
                             << ", row_num=" << res.gpu.get_row_num() << ", step=" << req.pattern_step << LOG_endl;
    }

    // when need to access neighbors of a remote vertex, we need to fork the query
    bool need_fork_join(SPARQLQuery &req) {
        // always need NOT fork-join when executing on single machine
        if (Global::num_servers == 1) return false;

        // always need fork-join mode w/o RDMA
        if (!Global::use_rdma) return true;

        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ASSERT(req.result.var_stat(pattern.subject) == KNOWN_VAR);
        sid_t start = req.get_pattern().subject;

        // GPUEngine only supports fork-join mode now
        return ((req.local_var != start)
                && (req.result.gpu.get_row_num() >= 0));
    }


public:
    GPUEngine(int sid, int tid, GPUMem *gmem, GPUCache *gcache, GPUStreamPool *stream_pool, DGraph *graph)
        : sid(sid), tid(tid), backend(sid, gcache, gmem, stream_pool), graph(graph) {

    }

    ~GPUEngine() { }

    // check whether intermediate result (rbuf) is loaded into GPU
    bool result_buf_ready(const SPARQLQuery &req) {
        if (req.pattern_step == 0)
            return true;
        else
            return req.result.gpu.is_rbuf_valid();
    }

    void load_result_buf(SPARQLQuery &req) {
        char *rbuf = backend.load_result_buf(req.result);
        req.result.gpu.set_rbuf(rbuf, req.result.result_table.size());
    }

    void load_result_buf(SPARQLQuery &req, const string &rbuf_str) {
        char *rbuf = backend.load_result_buf(rbuf_str.c_str(), rbuf_str.size());
        req.result.gpu.set_rbuf(rbuf, rbuf_str.size() / WUKONG_GPU_ELEM_SIZE);
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
            const_to_unknown(req);
            break;

        // start from KNOWN
        case const_pair(KNOWN_VAR, CONST_VAR):
            known_to_const(req);
            break;
        case const_pair(KNOWN_VAR, KNOWN_VAR):
            known_to_known(req);
            break;
        case const_pair(KNOWN_VAR, UNKNOWN_VAR):
            known_to_unknown(req);
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

    vector<SPARQLQuery> generate_sub_query(SPARQLQuery &req) {
        ASSERT(Global::num_servers > 1);

        SPARQLQuery::Pattern &pattern = req.get_pattern();
        sid_t start = pattern.subject;

        // generate sub requests for all servers
        vector<SPARQLQuery> sub_reqs(Global::num_servers);
        for (int i = 0; i < Global::num_servers; i++) {
            sub_reqs[i].pqid = req.qid;
            sub_reqs[i].pg_type = req.pg_type == SPARQLQuery::PGType::UNION ?
                                  SPARQLQuery::PGType::BASIC : req.pg_type;
            sub_reqs[i].pattern_group = req.pattern_group;
            sub_reqs[i].pattern_step = req.pattern_step;
            sub_reqs[i].corun_step = req.corun_step;
            sub_reqs[i].fetch_step = req.fetch_step;
            sub_reqs[i].local_var = start;
            sub_reqs[i].priority = req.priority + 1;

            sub_reqs[i].job_type = SPARQLQuery::SubJobType::SPLIT_JOB;
            sub_reqs[i].dev_type = SPARQLQuery::DeviceType::GPU;

            sub_reqs[i].result.set_col_num(req.result.col_num);
            sub_reqs[i].result.attr_col_num = req.result.attr_col_num;
            sub_reqs[i].result.blind = req.result.blind;
            sub_reqs[i].result.v2c_map  = req.result.v2c_map;
            sub_reqs[i].result.nvars  = req.result.nvars;
        }

        ASSERT(req.pg_type != SPARQLQuery::PGType::OPTIONAL);

        std::vector<sid_t*> buf_ptrs(Global::num_servers);
        std::vector<int> buf_sizes(Global::num_servers);

        backend.generate_sub_query(req, start, Global::num_servers, buf_ptrs, buf_sizes);

        logstream(LOG_DEBUG) << "#" << sid << " generate_sub_query for req#" << req.qid
                             << ", parent: " << req.pqid
                             << ", step: " << req.pattern_step << LOG_endl;

        for (int i = 0; i < Global::num_servers; ++i) {
            SPARQLQuery &r = sub_reqs[i];
            r.result.gpu.set_rbuf((char*)buf_ptrs[i], buf_sizes[i]);

            // if gpu history table is empty, set it to FULL_QUERY, which
            // will be sent by native RDMA
            if (r.result.gpu.is_rbuf_empty()) {
                r.job_type = SPARQLQuery::SubJobType::FULL_JOB;
                r.result.gpu.clear_rbuf();
            }
        }

        return sub_reqs;
    }

};

#endif  // USE_GPU
