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
#include <algorithm>//sort

#include "config.hpp"
#include "type.hpp"
#include "coder.hpp"
#include "adaptor.hpp"
#include "dgraph.hpp"
#include "query.hpp"
#include "assertion.hpp"
#include "mymath.hpp"
#include "timer.hpp"
#include "rmap.hpp"

using namespace std;

#define BUSY_POLLING_THRESHOLD 10000000 // busy polling task queue 10s
#define MIN_SNOOZE_TIME 10 // MIX snooze time
#define MAX_SNOOZE_TIME 80 // MAX snooze time
#define GPU_AGENT_TID_START (global_num_proxies + global_num_engines)


// a vector of pointers of all local GPU agents
class GPU_Agent;
std::vector<GPU_Agent *> gpu_agents;

/**
 * An agent thread will assist the GPU with query handling
 */
class GPU_Agent {
private:
    class Message {
    public:
        int sid;
        int tid;
        Bundle bundle;

        Message(int sid, int tid, Bundle &bundle)
            : sid(sid), tid(tid), bundle(bundle) { }
    };

    pthread_spinlock_t recv_lock;
    std::vector<SPARQLQuery> msg_fast_path;
    std::vector<SPARQLQuery> runqueue;

    RMap rmap; // a map of replies for pending (fork-join) queries
    pthread_spinlock_t rmap_lock;

    vector<Message> pending_msgs;

    inline void sweep_msgs() {
        if (!pending_msgs.size()) return;

        logstream(LOG_INFO) << "#" << tid << " "
                            << pending_msgs.size() << " pending msgs on engine." << LOG_endl;
        for (vector<Message>::iterator it = pending_msgs.begin(); it != pending_msgs.end();)
            if (adaptor->send(it->sid, it->tid, it->bundle))
                it = pending_msgs.erase(it);
            else
                ++it;
    }

    bool send_request(Bundle &bundle, int dst_sid, int dst_tid) {
        if (adaptor->send(dst_sid, dst_tid, bundle))
            return true;

        // failed to send, then stash the msg to avoid deadlock
        pending_msgs.push_back(Message(dst_sid, dst_tid, bundle));
        return false;
    }

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
        edge_t *edges = graph->get_index_edges_local(tid, tpid, d, &sz);
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
        req.local_var = -1;
    }

    /// A query whose parent's PGType is UNION may call this pattern
    void const_to_known(SPARQLQuery &req) {
        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ssid_t start = pattern.subject;
        ssid_t pid   = pattern.predicate;
        dir_t d      = pattern.direction;
        ssid_t end   = pattern.object;
        std::vector<sid_t> updated_result_table;
        SPARQLQuery::Result &res = req.result;
        int col = res.var2col(end);

        ASSERT(col != NO_RESULT);

        uint64_t sz = 0;
        edge_t *edges = graph->get_edges_global(tid, start, d, pid, &sz);

        boost::unordered_set<sid_t> unique_set;
        for (uint64_t k = 0; k < sz; k++)
            unique_set.insert(edges[k].val);

        for (uint64_t i = 0; i < res.get_row_num(); i++) {
            // matched
            if (unique_set.find(res.get_row_col(i, col)) != unique_set.end())
                res.append_row_to(i, updated_result_table);
        }
        res.result_table.swap(updated_result_table);
        req.pattern_step++;
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
        edge_t *edges = graph->get_edges_global(tid, start, d, pid, &sz);
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
        updated_result_table.reserve(res.result_table.size());

        // simple dedup for consecutive same vertices
        sid_t cached = BLANK_ID;
        edge_t *edges = NULL;
        uint64_t sz = 0;
        for (int i = 0; i < res.get_row_num(); i++) {
            sid_t cur = res.get_row_col(i, res.var2col(start));
            if (req.pg_type == SPARQLQuery::PGType::OPTIONAL &&
                    (!res.optional_matched_rows[i] || cur == BLANK_ID)) {
                res.append_row_to(i, updated_result_table);
                updated_result_table.push_back(BLANK_ID);
                continue;
            }
            if (cur != cached) {  // a new vertex
                cached = cur;
                edges = graph->get_edges_global(tid, cur, d, pid, &sz);
            }

            for (uint64_t k = 0; k < sz; k++) {
                res.append_row_to(i, updated_result_table);
                updated_result_table.push_back(edges[k].val);
            }
        }
        res.result_table.swap(updated_result_table);
        res.add_var2col(end, res.get_col_num());
        res.set_col_num(res.get_col_num() + 1);
        req.pattern_step++;
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

        vector<sid_t> updated_result_table;

        // simple dedup for consecutive same vertices
        sid_t cached = BLANK_ID;
        edge_t *edges = NULL;
        uint64_t sz = 0;
        for (int i = 0; i < res.get_row_num(); i++) {
            sid_t cur = res.get_row_col(i, res.var2col(start));
            if (cur != cached) {  // a new vertex
                cached = cur;
                edges = graph->get_edges_global(tid, cur, d, pid, &sz);
            }

            sid_t known = res.get_row_col(i, res.var2col(end));
            for (uint64_t k = 0; k < sz; k++) {
                if (edges[k].val == known) {
                    // append a matched intermediate result
                    res.append_row_to(i, updated_result_table);
                    break;
                }
            }
        }

        res.result_table.swap(updated_result_table);
        req.pattern_step++;
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

        vector<sid_t> updated_result_table;

        // simple dedup for consecutive same vertices
        sid_t cached = BLANK_ID;
        edge_t *edges = NULL;
        uint64_t sz = 0;
        bool exist = false;
        for (int i = 0; i < res.get_row_num(); i++) {
            sid_t cur = res.get_row_col(i, res.var2col(start));
            if (cur != cached) {  // a new vertex
                exist = false;
                cached = cur;
                edges = graph->get_edges_global(tid, cur, d, pid, &sz);

                for (uint64_t k = 0; k < sz; k++) {
                    if (edges[k].val == end) {
                        // append a matched intermediate result
                        exist = true;
                        res.append_row_to(i, updated_result_table);
                        break;
                    }
                }
            } else {
                // the matching result can also be reused
                res.append_row_to(i, updated_result_table);
            }

        }
        res.result_table.swap(updated_result_table);
        req.pattern_step++;
    }

    vector<SPARQLQuery> generate_sub_query(SPARQLQuery &req) {
        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ssid_t start = pattern.subject;

        // generate sub requests for all servers
        vector<SPARQLQuery> sub_reqs(global_num_servers);
        for (int i = 0; i < global_num_servers; i++) {
            sub_reqs[i].pid = req.id;
            sub_reqs[i].pg_type = req.pg_type = req.pg_type;
            sub_reqs[i].pattern_group = req.pattern_group;
            sub_reqs[i].pattern_step = req.pattern_step;
            sub_reqs[i].corun_step = req.corun_step;
            sub_reqs[i].fetch_step = req.fetch_step;
            sub_reqs[i].local_var = start;
            sub_reqs[i].priority = req.priority + 1;

            sub_reqs[i].result.col_num = req.result.col_num;
            sub_reqs[i].result.attr_col_num = req.result.attr_col_num;
            sub_reqs[i].result.blind = req.result.blind;
            sub_reqs[i].result.v2c_map  = req.result.v2c_map;
            sub_reqs[i].result.nvars  = req.result.nvars;
        }

        ASSERT(req.pg_type != SPARQLQuery::PGType::OPTIONAL);

        // group intermediate results to servers
        for (int i = 0; i < req.result.get_row_num(); i++) {
            int dst_sid = mymath::hash_mod(req.result.get_row_col(i, req.result.var2col(start)),
                                           global_num_servers);
            req.result.append_row_to(i, sub_reqs[dst_sid].result.result_table);
            req.result.append_attr_row_to(i, sub_reqs[dst_sid].result.attr_res_table);
        }

        return sub_reqs;
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
        // GPU_Agent only supports fork-join mode now
        return (req.result.get_row_num() >= 0); // FIXME: not consider dedup
    }


    bool execute_one_pattern(SPARQLQuery &req) {
        ASSERT(!req.done(SPARQLQuery::SQState::SQ_PATTERN));

        logstream(LOG_DEBUG) << "[" << sid << "-" << tid << "]"
                             << " step=" << req.pattern_step << LOG_endl;

        SPARQLQuery::Pattern &pattern = req.get_pattern();
        ssid_t start     = pattern.subject;
        ssid_t predicate = pattern.predicate;
        dir_t direction  = pattern.direction;
        ssid_t end       = pattern.object;

        if (req.pattern_step == 0 && req.start_from_index()) {
            if (req.result.var2col(end) != NO_RESULT)
                ASSERT("GPU_AGENT doesn't support index_to_known");
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
            ASSERT("GPU_AGENT doesn't support attr");
        }

        // triple pattern with KNOWN predicate
        switch (const_pair(req.result.variable_type(start),
                           req.result.variable_type(end))) {

        // start from CONST
        case const_pair(const_var, const_var):
            logstream(LOG_ERROR) << "Unsupported triple pattern [CONST|KNOWN|CONST]" << LOG_endl;
            ASSERT(false);
        case const_pair(const_var, known_var):
            const_to_known(req);
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

    bool execute_patterns(SPARQLQuery &r) {
        logstream(LOG_DEBUG) << "[" << sid << "-" << tid << "]"
                             << " id=" << r.id << " pid=" << r.pid << LOG_endl;

        if (r.pattern_step == 0
                && r.pattern_group.parallel == false
                && r.start_from_index()
                && (global_num_servers * r.mt_factor > 1)) {
            // The mt_factor can be set on proxy side before sending to engine,
            // but must smaller than global_mt_threshold (Default: mt_factor == 1)
            // Normally, we will NOT let global_mt_threshold == #gpu, which will cause HANG
            int sub_reqs_size = global_num_servers * r.mt_factor;
            rmap.put_parent_request(r, sub_reqs_size);
            SPARQLQuery sub_query = r;
            for (int i = 0; i < global_num_servers; i++) {
                for (int j = 0; j < r.mt_factor; j++) {
                    //SPARQLQuery sub_query;
                    sub_query.id = -1;
                    sub_query.pid = r.id;
                    // start from the next engine thread
                    int dst_tid = (tid + j + 1 - GPU_AGENT_TID_START) % global_num_gpus
                                  + GPU_AGENT_TID_START;
                    sub_query.tid = j;
                    sub_query.mt_factor = 1;
                    sub_query.pattern_group.parallel = true;

                    Bundle bundle(sub_query);
                    send_request(bundle, i, dst_tid);
                }
            }

            return false;
        }

        do {
            execute_one_pattern(r);

            // co-run optimization
            if (r.corun_enabled && (r.pattern_step == r.corun_step)) {
                ASSERT("GPU_Agent doesn't support CORUN");
            }

            if (r.done(SPARQLQuery::SQState::SQ_PATTERN)) {
                // only send back row_num in blind mode
                r.result.row_num = r.result.get_row_num();
                return true;
            }

            if (need_fork_join(r)) {
                vector<SPARQLQuery> sub_reqs = generate_sub_query(r);
                rmap.put_parent_request(r, sub_reqs.size());
                for (int i = 0; i < sub_reqs.size(); i++) {
                    if (i != sid) {
                        Bundle bundle(sub_reqs[i]);
                        send_request(bundle, i, tid);
                    } else {
                        pthread_spin_lock(&recv_lock);
                        msg_fast_path.push_back(sub_reqs[i]);
                        pthread_spin_unlock(&recv_lock);
                    }
                }
                return false;
            }
        } while (true);
    }

    void execute_sparql_query(SPARQLQuery &r, GPU_Agent *engine) {
        // encode the lineage of the query (server & thread)
        if (r.id == -1) r.id = coder.get_and_inc_qid();

        if (r.state == SPARQLQuery::SQState::SQ_REPLY) {
            pthread_spin_lock(&engine->rmap_lock);
            engine->rmap.put_reply(r);

            if (!engine->rmap.is_ready(r.pid)) {
                pthread_spin_unlock(&engine->rmap_lock);
                return; // not ready (waiting for the rest)
            }

            // all sub-queries have done, continue to execute
            r = engine->rmap.get_merged_reply(r.pid);
            pthread_spin_unlock(&engine->rmap_lock);
        }

        // 1. Pattern
        if (r.has_pattern() && !r.done(SPARQLQuery::SQState::SQ_PATTERN)) {
            r.state = SPARQLQuery::SQState::SQ_PATTERN;
            if (!execute_patterns(r)) return;
        }

        // 6. Reply
        r.shrink_query();
        r.state = SPARQLQuery::SQState::SQ_REPLY;
        Bundle bundle(r);
        send_request(bundle, coder.sid_of(r.pid), coder.tid_of(r.pid));
    }

    void execute(Bundle &bundle, GPU_Agent *engine) {
        if (bundle.type == SPARQL_QUERY) {
            SPARQLQuery r = bundle.get_sparql_query();
            execute_sparql_query(r, engine);
        }
    }

public:
    const static uint64_t TIMEOUT_THRESHOLD = 10000; // 10 msec

    int sid;    // server id
    int tid;    // thread id

    String_Server *str_server;
    DGraph *graph;
    Adaptor *adaptor;

    Coder coder;

    bool at_work; // whether engine is at work or not
    uint64_t last_time; // busy or not (work-oblige)

    GPU_Agent(int sid, int tid, String_Server * str_server, DGraph * graph, Adaptor * adaptor)
        : sid(sid), tid(tid), str_server(str_server), graph(graph), adaptor(adaptor),
          coder(sid, tid), last_time(timer::get_usec()) {
        pthread_spin_init(&recv_lock, 0);
        pthread_spin_init(&rmap_lock, 0);
    }

    void run() {
        // NOTE: the 'tid' of engine is not start from 0,
        // which can not be used by gpu_agents[] directly
        int own_id = tid - GPU_AGENT_TID_START;

        uint64_t snooze_interval = MIN_SNOOZE_TIME;

        // reset snooze
        auto reset_snooze = [&snooze_interval](bool & at_work, uint64_t &last_time) {
            at_work = true; // keep calm (no snooze)
            last_time = timer::get_usec();
            snooze_interval = MIN_SNOOZE_TIME;
        };

        while (true) {
            at_work = false;

            // check and send pending messages first
            sweep_msgs();

            // fast path (priority)
            SPARQLQuery request; // FIXME: only sparql query use fast-path now
            pthread_spin_lock(&recv_lock);
            if (msg_fast_path.size() > 0) {
                request = msg_fast_path[0];
                msg_fast_path.erase(msg_fast_path.begin());
                at_work = true;
            }
            pthread_spin_unlock(&recv_lock);

            if (at_work) {
                reset_snooze(at_work, last_time);
                execute_sparql_query(request, gpu_agents[own_id]);
                continue; // exhaust all queries
            }

            // normal path: own runqueue
            Bundle bundle;
            while (adaptor->tryrecv(bundle)) {
                if (bundle.type == SPARQL_QUERY) {
                    // to be fair, engine will handle sub-queries priority
                    // instead of processing a new query.
                    SPARQLQuery req = bundle.get_sparql_query();
                    if (req.priority != 0) {
                        reset_snooze(at_work, last_time);
                        execute_sparql_query(req, gpu_agents[own_id]);
                        break;
                    }

                    runqueue.push_back(req);
                } else {
                    // FIXME: Jump a queue!
                    reset_snooze(at_work, last_time);
                    execute(bundle, gpu_agents[own_id]);
                    break;
                }
            }

            if (!at_work && runqueue.size() > 0) {
                // get new task
                SPARQLQuery req = runqueue[0];
                runqueue.erase(runqueue.begin());

                reset_snooze(at_work, last_time);
                execute_sparql_query(req, gpu_agents[own_id]);
            }

        }
    }
};

#endif  // USE_GPU
