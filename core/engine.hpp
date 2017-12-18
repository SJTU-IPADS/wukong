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

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <stdlib.h> //qsort

#include "config.hpp"
#include "type.hpp"
#include "coder.hpp"
#include "adaptor.hpp"
#include "dgraph.hpp"
#include "query.hpp"

#include "mymath.hpp"
#include "timer.hpp"

using namespace std;

// The map is used to colloect the replies of sub-queries in fork-join execution
class Reply_Map {
private:

    struct Item {
        int count;
        request_or_reply parent_request;
        request_or_reply merged_reply;
    };

    boost::unordered_map<int, Item> internal_item_map;

public:
    void put_parent_request(request_or_reply &r, int cnt) {
        Item data;
        data.count = cnt;
        data.parent_request = r;
        internal_item_map[r.id] = data;
    }

    void put_reply(request_or_reply &r) {
        int pid = r.pid;
        Item &data = internal_item_map[pid];

        vector<sid_t> &result_table = data.merged_reply.result_table;
        data.count--;
        data.merged_reply.step = r.step;
        data.merged_reply.col_num = r.col_num;
        data.merged_reply.blind = r.blind;
        data.merged_reply.row_num += r.row_num;

        int new_size = result_table.size() + r.result_table.size();
        result_table.reserve(new_size);
        result_table.insert( result_table.end(), r.result_table.begin(), r.result_table.end());
    }

    bool is_ready(int pid) {
        return internal_item_map[pid].count == 0;
    }

    request_or_reply get_merged_reply(int pid) {
        request_or_reply r = internal_item_map[pid].parent_request;
        request_or_reply &merged_reply = internal_item_map[pid].merged_reply;

        r.step = merged_reply.step;
        r.col_num = merged_reply.col_num;
        r.blind = merged_reply.blind;
        r.row_num = merged_reply.row_num;

        r.result_table.swap(merged_reply.result_table);
        internal_item_map.erase(pid);
        return r;
    }
};



typedef pair<int64_t, int64_t> v_pair;

int64_t hash_pair(const v_pair &x) {
    int64_t r = x.first;
    r = r << 32;
    r += x.second;
    return hash<int64_t>()(r);
}


// a vector of pointers of all local engines
class Engine;
std::vector<Engine *> engines;


class Engine {
private:
    class Message {
    public:
        int sid;
        int tid;
        request_or_reply r;

        Message(int sid, int tid, request_or_reply &r)
            : sid(sid), tid(tid), r(r) { }
    };

    pthread_spinlock_t recv_lock;
    std::vector<request_or_reply> msg_fast_path;

    Reply_Map rmap; // a map of replies for pending (fork-join) queries
    pthread_spinlock_t rmap_lock;

    vector<Message> pending_msgs;

    inline void sweep_msgs() {
        if (!pending_msgs.size()) return;

        for (vector<Message>::iterator it = pending_msgs.begin(); it != pending_msgs.end(); )
            if (adaptor->send(it->sid, it->tid, it->r))
                it = pending_msgs.erase(it);
            else
                ++it;
    }

    bool send_request(int sid, int tid, request_or_reply &r) {
        if (adaptor->send(sid, tid, r))
            return true;

        // failed to send, then stash the msg to void deadlock
        pending_msgs.push_back(Message(sid, tid, r));
        return false;
    }

    // all of these means const predicate
    void const_to_unknown(request_or_reply &req) {
        ssid_t start = req.cmd_chains[req.step * 4];
        ssid_t pid   = req.cmd_chains[req.step * 4 + 1];
        dir_t d      = (dir_t)req.cmd_chains[req.step * 4 + 2];
        ssid_t end   = req.cmd_chains[req.step * 4 + 3];
        std::vector<sid_t> updated_result_table;

        // the query plan is wrong
        assert(req.get_col_num() == 0);

        uint64_t sz = 0;
        edge_t *res = graph->get_edges_global(tid, start, d, pid, &sz);
        for (uint64_t k = 0; k < sz; k++)
            updated_result_table.push_back(res[k].val);

        req.result_table.swap(updated_result_table);
        req.add_var2col(end, 0);
        req.set_col_num(1);
        req.step++;
    }

    void const_to_known(request_or_reply &req) { assert(false); } /// TODO

    void known_to_unknown(request_or_reply &req) {
        ssid_t start = req.cmd_chains[req.step * 4];
        ssid_t pid   = req.cmd_chains[req.step * 4 + 1];
        dir_t d      = (dir_t)req.cmd_chains[req.step * 4 + 2];
        ssid_t end   = req.cmd_chains[req.step * 4 + 3];
        std::vector<sid_t> updated_result_table;

        // the query plan is wrong
        //assert(req.get_col_num() == req.var2col(end));

        updated_result_table.reserve(req.result_table.size());
        for (int i = 0; i < req.get_row_num(); i++) {
            sid_t prev_id = req.get_row_col(i, req.var2col(start));
            uint64_t sz = 0;
            edge_t *res = graph->get_edges_global(tid, prev_id, d, pid, &sz);
            for (uint64_t k = 0; k < sz; k++) {
                req.append_row_to(i, updated_result_table);
                updated_result_table.push_back(res[k].val);
            }
        }

        req.add_var2col(end, req.get_col_num());
        req.set_col_num(req.get_col_num() + 1);
        req.result_table.swap(updated_result_table);
        req.step++;
    }

    void known_to_known(request_or_reply &req) {
        ssid_t start = req.cmd_chains[req.step * 4];
        ssid_t pid   = req.cmd_chains[req.step * 4 + 1];
        dir_t d     = (dir_t)req.cmd_chains[req.step * 4 + 2];
        ssid_t end   = req.cmd_chains[req.step * 4 + 3];
        vector<sid_t> updated_result_table;

        for (int i = 0; i < req.get_row_num(); i++) {
            sid_t prev_id = req.get_row_col(i, req.var2col(start));
            uint64_t sz = 0;
            edge_t *res = graph->get_edges_global(tid, prev_id, d, pid, &sz);
            sid_t end2 = req.get_row_col(i, req.var2col(end));
            for (uint64_t k = 0; k < sz; k++) {
                if (res[k].val == end2) {
                    req.append_row_to(i, updated_result_table);
                    break;
                }
            }
        }

        req.result_table.swap(updated_result_table);
        req.step++;
    }

    void known_to_const(request_or_reply &req) {
        ssid_t start = req.cmd_chains[req.step * 4];
        ssid_t pid   = req.cmd_chains[req.step * 4 + 1];
        dir_t d     = (dir_t)req.cmd_chains[req.step * 4 + 2];
        ssid_t end   = req.cmd_chains[req.step * 4 + 3];
        vector<sid_t> updated_result_table;

        for (int i = 0; i < req.get_row_num(); i++) {
            sid_t prev_id = req.get_row_col(i, req.var2col(start));
            uint64_t sz = 0;
            edge_t *res = graph->get_edges_global(tid, prev_id, d, pid, &sz);
            for (uint64_t k = 0; k < sz; k++) {
                if (res[k].val == end) {
                    req.append_row_to(i, updated_result_table);
                    break;
                }
            }
        }

        req.result_table.swap(updated_result_table);
        req.step++;
    }

    void index_to_unknown(request_or_reply &req) {
        ssid_t idx = req.cmd_chains[req.step * 4];
        ssid_t nothing = req.cmd_chains[req.step * 4 + 1];
        dir_t d = (dir_t)req.cmd_chains[req.step * 4 + 2];
        ssid_t var = req.cmd_chains[req.step * 4 + 3];
        vector<sid_t> updated_result_table;

        // the query plan is wrong
        assert(req.get_col_num() == 0);

        uint64_t sz = 0;
        edge_t *res = graph->get_index_edges_local(tid, idx, d, &sz);
        int start = req.tid;
        for (uint64_t k = start; k < sz; k += global_mt_threshold)
            updated_result_table.push_back(res[k].val);

        req.result_table.swap(updated_result_table);
        req.set_col_num(1);
        req.add_var2col(var, 0);
        req.step++;
        req.local_var = -1;
    }

    void const_unknown_unknown(request_or_reply &req) {
        ssid_t start = req.cmd_chains[req.step * 4];
        ssid_t pid   = req.cmd_chains[req.step * 4 + 1];
        dir_t d      = (dir_t)req.cmd_chains[req.step * 4 + 2];
        ssid_t end   = req.cmd_chains[req.step * 4 + 3];
        vector<sid_t> updated_result_table;

        // the query plan is wrong
        assert(req.get_col_num() == 0);

        uint64_t npids = 0;
        edge_t *pids = graph->get_edges_global(tid, start, d, PREDICATE_ID, &npids);

        // use a local buffer to store "known" predicates
        edge_t *tpids = (edge_t *)malloc(npids * sizeof(edge_t));
        memcpy((char *)tpids, (char *)pids, npids * sizeof(edge_t));

        for (uint64_t p = 0; p < npids; p++) {
            uint64_t sz = 0;
            edge_t *res = graph->get_edges_global(tid, start, d, tpids[p].val, &sz);
            for (uint64_t k = 0; k < sz; k++) {
                updated_result_table.push_back(tpids[p].val);
                updated_result_table.push_back(res[k].val);
            }
        }

        free(tpids);

        req.result_table.swap(updated_result_table);
        req.add_var2col(pid, 0);
        req.add_var2col(end, 1);
        req.set_col_num(2);
        req.step++;
    }

    void known_unknown_unknown(request_or_reply &req) {
        ssid_t start = req.cmd_chains[req.step * 4];
        ssid_t pid   = req.cmd_chains[req.step * 4 + 1];
        dir_t d     = (dir_t)req.cmd_chains[req.step * 4 + 2];
        ssid_t end   = req.cmd_chains[req.step * 4 + 3];
        vector<sid_t> updated_result_table;

        for (int i = 0; i < req.get_row_num(); i++) {
            sid_t prev_id = req.get_row_col(i, req.var2col(start));

            uint64_t npids = 0;
            edge_t *pids = graph->get_edges_global(tid, prev_id, d, PREDICATE_ID, &npids);

            // use a local buffer to store "known" predicates
            edge_t *tpids = (edge_t *)malloc(npids * sizeof(edge_t));
            memcpy((char *)tpids, (char *)pids, npids * sizeof(edge_t));

            for (uint64_t p = 0; p < npids; p++) {
                uint64_t sz = 0;
                edge_t *res = graph->get_edges_global(tid, prev_id, d, tpids[p].val, &sz);
                for (uint64_t k = 0; k < sz; k++) {
                    req.append_row_to(i, updated_result_table);
                    updated_result_table.push_back(tpids[p].val);
                    updated_result_table.push_back(res[k].val);
                }
            }

            free(tpids);
        }

        req.add_var2col(pid, req.get_col_num());
        req.add_var2col(end, req.get_col_num() + 1);
        req.set_col_num(req.get_col_num() + 2);
        req.result_table.swap(updated_result_table);
        req.step++;
    }

    void known_unknown_const(request_or_reply &req) {
        ssid_t start = req.cmd_chains[req.step * 4];
        ssid_t pid   = req.cmd_chains[req.step * 4 + 1];
        dir_t d     = (dir_t)req.cmd_chains[req.step * 4 + 2];
        ssid_t end   = req.cmd_chains[req.step * 4 + 3];
        vector<sid_t> updated_result_table;

        for (int i = 0; i < req.get_row_num(); i++) {
            sid_t prev_id = req.get_row_col(i, req.var2col(start));
            uint64_t npids = 0;
            edge_t *pids = graph->get_edges_global(tid, prev_id, d, PREDICATE_ID, &npids);

            // use a local buffer to store "known" predicates
            edge_t *tpids = (edge_t *)malloc(npids * sizeof(edge_t));
            memcpy((char *)tpids, (char *)pids, npids * sizeof(edge_t));


            for (uint64_t p = 0; p < npids; p++) {
                uint64_t sz = 0;
                edge_t *res = graph->get_edges_global(tid, prev_id, d, tpids[p].val, &sz);
                for (uint64_t k = 0; k < sz; k++) {
                    if (res[k].val == end) {
                        req.append_row_to(i, updated_result_table);
                        updated_result_table.push_back(tpids[p].val);
                        break;
                    }
                }
            }

            free(tpids);
        }

        req.add_var2col(pid, req.get_col_num());
        req.set_col_num(req.get_col_num() + 1);
        req.result_table.swap(updated_result_table);
        req.step++;
    }

    vector<request_or_reply> generate_sub_query(request_or_reply &req) {
        ssid_t start = req.cmd_chains[req.step * 4];
        ssid_t end   = req.cmd_chains[req.step * 4 + 3];

        // generate sub requests for all servers
        vector<request_or_reply> sub_reqs(global_num_servers);
        for (int i = 0; i < global_num_servers; i++) {
            sub_reqs[i].pid = req.id;
            sub_reqs[i].cmd_chains = req.cmd_chains;
            sub_reqs[i].step = req.step;
            sub_reqs[i].col_num = req.col_num;
            sub_reqs[i].blind = req.blind;
            sub_reqs[i].local_var = start;
            sub_reqs[i].v2c_map  = req.v2c_map;
            sub_reqs[i].nvars  = req.nvars;
        }

        // group intermediate results to servers
        for (int i = 0; i < req.get_row_num(); i++) {
            int sid = mymath::hash_mod(req.get_row_col(i, req.var2col(start)),
                                       global_num_servers);
            req.append_row_to(i, sub_reqs[sid].result_table);
        }

        return sub_reqs;
    }

    // fork-join or in-place execution
    bool need_fork_join(request_or_reply &req) {
        // always need fork-join mode w/o RDMA
        if (!global_use_rdma) return true;

        ssid_t start = req.cmd_chains[req.step * 4];
        return ((req.local_var != start)
                && (req.get_row_num() >= global_rdma_threshold));
    }

    void do_corun(request_or_reply &req) {
        int corun_step = req.step + 1;
        int fetch_step = req.cmd_chains[req.step * 4 + 3];

        // step.1 remove dup;
        uint64_t t0 = timer::get_usec();

        boost::unordered_set<sid_t> unique_set;
        ssid_t vid = req.cmd_chains[corun_step * 4];
        assert(vid < 0);
        int col_idx = req.var2col(vid);
        for (int i = 0; i < req.get_row_num(); i++)
            unique_set.insert(req.get_row_col(i, col_idx));

        // step.2 generate cmd_chain for sub-reqs
        vector<ssid_t> sub_chain;
        vector<int> pvars_map; // from new_id to col_idx of id

        boost::unordered_map<sid_t, sid_t> sub_pvars;
        for (int i = corun_step * 4; i < fetch_step * 4; i++) {
            ssid_t id = req.cmd_chains[i];

            if (id < 0) { // remap pattern variable
                if (sub_pvars.find(id) == sub_pvars.end()) {
                    sid_t new_id = - (sub_pvars.size() + 1); // starts from -1
                    sub_pvars[id] = new_id;
                    pvars_map.push_back(req.var2col(id));
                }

                sub_chain.push_back(sub_pvars[id]);
            } else {
                sub_chain.push_back(id);
            }
        }

        // step.3 make sub-req
        request_or_reply sub_req;

        // query
        sub_req.cmd_chains = sub_chain;
        sub_req.nvars = pvars_map.size();

        // result
        boost::unordered_set<sid_t>::iterator iter;
        for (iter = unique_set.begin(); iter != unique_set.end(); iter++)
            sub_req.result_table.push_back(*iter);
        sub_req.col_num = 1;
        sub_req.add_var2col(sub_pvars[vid], 0);

        sub_req.blind = false; // must take back results
        uint64_t t1 = timer::get_usec(); // time to generate the sub-request

        // step.4 execute sub-req
        while (true) {
            execute_one_step(sub_req);
            if (sub_req.is_finished())
                break;
        }
        uint64_t t2 = timer::get_usec(); // time to run the sub-request

        uint64_t t3, t4;
        vector<sid_t> updated_result_table;

        if (sub_req.get_col_num() > 2) { // qsort
            mytuple::qsort_tuple(sub_req.get_col_num(), sub_req.result_table);

            t3 = timer::get_usec();
            vector<sid_t> tmp_vec;
            tmp_vec.resize(sub_req.get_col_num());
            for (int i = 0; i < req.get_row_num(); i++) {
                for (int c = 0; c < pvars_map.size(); c++)
                    tmp_vec[c] = req.get_row_col(i, pvars_map[c]);

                if (mytuple::binary_search_tuple(sub_req.get_col_num(),
                                                 sub_req.result_table, tmp_vec))
                    req.append_row_to(i, updated_result_table);
            }
            t4 = timer::get_usec();
        } else { // hash join
            boost::unordered_set<v_pair> remote_set;
            for (int i = 0; i < sub_req.get_row_num(); i++)
                remote_set.insert(v_pair(sub_req.get_row_col(i, 0),
                                         sub_req.get_row_col(i, 1)));

            t3 = timer::get_usec();
            vector<sid_t> tmp_vec;
            tmp_vec.resize(sub_req.get_col_num());
            for (int i = 0; i < req.get_row_num(); i++) {
                for (int c = 0; c < pvars_map.size(); c++)
                    tmp_vec[c] = req.get_row_col(i, pvars_map[c]);

                v_pair target = v_pair(tmp_vec[0], tmp_vec[1]);
                if (remote_set.find(target) != remote_set.end())
                    req.append_row_to(i, updated_result_table);
            }
            t4 = timer::get_usec();
        }

        // debug
        if (sid == 0 && tid == 0) {
            cout << "prepare " << (t1 - t0) << " us" << endl;
            cout << "execute sub-request " << (t2 - t1) << " us" << endl;
            cout << "sort " << (t3 - t2) << " us" << endl;
            cout << "lookup " << (t4 - t3) << " us" << endl;
        }

        req.result_table.swap(updated_result_table);
        req.step = fetch_step;
    }

    bool execute_one_step(request_or_reply &req) {
        if (req.is_finished()) {
            return false;
        }
        if (req.step == 0 && req.start_from_index()) {
            index_to_unknown(req);
            return true;
        }
        ssid_t start = req.cmd_chains[req.step * 4];
        ssid_t predicate = req.cmd_chains[req.step * 4 + 1];
        dir_t direction = (dir_t)req.cmd_chains[req.step * 4 + 2];
        ssid_t end = req.cmd_chains[req.step * 4 + 3];

        if (predicate < 0) {
#ifdef VERSATILE
            switch (var_pair(req.variable_type(start),
                             req.variable_type(end))) {
            case var_pair(const_var, unknown_var):
                const_unknown_unknown(req);
                break;
            case var_pair(known_var, unknown_var):
                known_unknown_unknown(req);
                break;
            default :
                assert(false);
                break;
            }
            return true;
#else
            cout << "ERROR: unsupport variable at predicate." << endl;
            cout << "Please add definition VERSATILE in CMakeLists.txt." << endl;
            assert(false);
#endif
        }

        // known_predicate
        switch (var_pair(req.variable_type(start), req.variable_type(end))) {
        // start from const_var
        case var_pair(const_var, const_var):
            cout << "ERROR: unsupported triple pattern (from const_var to const_var)" << endl;
            assert(false);
        case var_pair(const_var, unknown_var):
            const_to_unknown(req);
            break;
        case var_pair(const_var, known_var):
            cout << "ERROR: unsupported triple pattern (from const_var to known_var)" << endl;
            assert(false);

        // start from known_var
        case var_pair(known_var, const_var):
            known_to_const(req);
            break;
        case var_pair(known_var, known_var):
            known_to_known(req);
            break;
        case var_pair(known_var, unknown_var):
            known_to_unknown(req);
            break;

        // start from unknown_var
        case var_pair(unknown_var, const_var):
        case var_pair(unknown_var, known_var):
        case var_pair(unknown_var, unknown_var):
            cout << "ERROR: unsupported triple pattern (from unknown_var)" << endl;
            assert(false);

        default :
            assert(false);
        }

        return true;
    }

    void execute_request(request_or_reply &r) {
        r.id = coder.get_and_inc_qid();
        while (true) {
            uint64_t t1 = timer::get_usec();
            execute_one_step(r);
            t1 = timer::get_usec() - t1;

            // co-run optimization
            if (!r.is_finished() && (r.cmd_chains[r.step * 4 + 2] == CORUN))
                do_corun(r);

            if (r.is_finished()) {
                r.row_num = r.get_row_num();
                if (r.blind)
                    r.clear_data(); // avoid take back the results

                send_request(coder.sid_of(r.pid), coder.tid_of(r.pid), r);
                return;
            }

            if (need_fork_join(r)) {
                vector<request_or_reply> sub_reqs = generate_sub_query(r);
                rmap.put_parent_request(r, sub_reqs.size());
                for (int i = 0; i < sub_reqs.size(); i++) {
                    if (i != sid) {
                        send_request(i, tid, sub_reqs[i]);
                    } else {
                        pthread_spin_lock(&recv_lock);
                        msg_fast_path.push_back(sub_reqs[i]);
                        pthread_spin_unlock(&recv_lock);
                    }
                }
                return;
            }
        }
        return;
    }

    void execute_reply(request_or_reply &r, Engine *engine) {
        pthread_spin_lock(&engine->rmap_lock);
        engine->rmap.put_reply(r);
        if (engine->rmap.is_ready(r.pid)) {
            request_or_reply reply = engine->rmap.get_merged_reply(r.pid);
            pthread_spin_unlock(&engine->rmap_lock);

            send_request(coder.sid_of(reply.pid), coder.tid_of(reply.pid), reply);
        } else {
            pthread_spin_unlock(&engine->rmap_lock);
        }
    }

#if DYNAMIC_GSTORE
    void execute_insert(request_or_reply &req) {
        int insert_ret = 0;
        string fname = req.get_insert_fname();
        ifstream input(fname.c_str());
        if (input.good()) {
            graph -> static_insert(input);
            insert_ret = 1;
        }
        req.set_insert_ret(insert_ret);
        adaptor->send(coder.sid_of(req.pid), coder.tid_of(req.pid), req);
        return;
    }
#endif

    void execute(request_or_reply &r, Engine *engine) {
        if (r.r_type == query_req) {
            if (r.is_request())
                execute_request(r);
            else
                execute_reply(r, engine);
#if DYNAMIC_GSTORE
        } else {
            execute_insert(r);
#endif         
        }
    }

public:
    const static uint64_t TIMEOUT_THRESHOLD = 10000; // 10 msec

    int sid;    // server id
    int tid;    // thread id

    DGraph *graph;
    Adaptor *adaptor;

    Coder coder;

    uint64_t last_time; // busy or not (work-oblige)

    Engine(int sid, int tid,DGraph *graph, Adaptor *adaptor)
        : sid(sid), tid(tid), graph(graph), adaptor(adaptor),
          coder(sid, tid), last_time(0) {
        pthread_spin_init(&recv_lock, 0);
        pthread_spin_init(&rmap_lock, 0);
    }

    void run() {
        // NOTE: the 'tid' of engine is not start from 0,
        // which can not be used by engines[] directly
        int own_id = tid - global_num_proxies;
        // TODO: replace pair to ring
        int nbr_id = (global_num_engines - 1) - own_id;

        int send_wait_cnt = 0;
        while (true) {
            request_or_reply r;
            bool success;

            // fast path
            last_time = timer::get_usec();
            success = false;

            pthread_spin_lock(&recv_lock);
            if (msg_fast_path.size() > 0) {
                r = msg_fast_path.back();
                msg_fast_path.pop_back();
                success = true;
            }
            pthread_spin_unlock(&recv_lock);

            if (success) {
                execute(r, engines[own_id]);
                continue; // fast-path priority
            }

            // check and send pending messages
            sweep_msgs();

            // normal path
            last_time = timer::get_usec();

            // own queue
            success = false;
            pthread_spin_lock(&recv_lock);
            success = adaptor->tryrecv(r);
            if (success && r.start_from_index()) {
                msg_fast_path.push_back(r);
                success = false;
            }
            pthread_spin_unlock(&recv_lock);

            if (success) execute(r, engines[own_id]);

            // work-oblige is disabled
            if (!global_enable_workstealing) continue;

            // neighbor queue
            last_time = timer::get_usec();
            if (last_time < engines[nbr_id]->last_time + TIMEOUT_THRESHOLD)
                continue; // neighboring worker is self-sufficient

            success = false;
            pthread_spin_lock(&engines[nbr_id]->recv_lock);
            success = engines[nbr_id]->adaptor->tryrecv(r);
            if (success && r.start_from_index()) {
                engines[nbr_id]->msg_fast_path.push_back(r);
                success = false;
            }
            pthread_spin_unlock(&engines[nbr_id]->recv_lock);

            if (success) execute(r, engines[nbr_id]);
        }
    }

};
