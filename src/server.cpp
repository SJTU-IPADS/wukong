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
 *      http://ipads.se.sjtu.edu.cn/projects/wukong.html
 *
 */

#include "server.h"
#include <stdlib.h> //qsort


server::server(distributed_graph& _g, thread_cfg* _cfg): g(_g), cfg(_cfg)
{
    last_time = -1;
    pthread_spin_init(&recv_lock, 0);
    pthread_spin_init(&wqueue_lock, 0);
}

void
server::const_to_unknown(request_or_reply& req)
{
    int64_t start       = req.cmd_chains[req.step * 4];
    int64_t predicate   = req.cmd_chains[req.step * 4 + 1];
    int64_t direction   = req.cmd_chains[req.step * 4 + 2];
    int64_t end         = req.cmd_chains[req.step * 4 + 3];
    vector<int64_t> updated_result_table;

    if (!((req.column_num() == 0) && (req.column_num() == req.var2column(end)))) {
        //it means the query plan is wrong
        assert(false);
    }
    int edge_num = 0;
    edge *edge_ptr;
    edge_ptr = g.get_edges_global(cfg->wid, start, direction, predicate, &edge_num);
    for (int k = 0; k < edge_num; k++) {
        updated_result_table.push_back(edge_ptr[k].val);
    }

    req.result_table.swap(updated_result_table);
    req.set_column_num(1);
    req.step++;
}

void
server::const_to_known(request_or_reply& req)
{
    //TODO
}

void
server::known_to_unknown(request_or_reply& req)
{
    int64_t start       = req.cmd_chains[req.step * 4];
    int64_t predict     = req.cmd_chains[req.step * 4 + 1];
    int64_t direction   = req.cmd_chains[req.step * 4 + 2];
    int64_t end         = req.cmd_chains[req.step * 4 + 3];
    vector<int64_t> updated_result_table;
    updated_result_table.reserve(req.result_table.size());
    if (req.column_num() != req.var2column(end) ) {
        //it means the query plan is wrong
        assert(false);
    }

    for (int i = 0; i < req.row_num(); i++) {
        int64_t prev_id = req.get_row_column(i, req.var2column(start));
        int edge_num = 0;
        edge *edge_ptr;
        edge_ptr = g.get_edges_global(cfg->wid, prev_id, direction, predict, &edge_num);

        for (int k = 0; k < edge_num; k++) {
            req.append_row_to(i, updated_result_table);
            updated_result_table.push_back(edge_ptr[k].val);
        }
    }
    req.set_column_num(req.column_num() + 1);
    req.result_table.swap(updated_result_table);
    req.step++;
}

void
server::known_to_known(request_or_reply &req)
{
    int64_t start       = req.cmd_chains[req.step * 4];
    int64_t predict     = req.cmd_chains[req.step * 4 + 1];
    int64_t direction   = req.cmd_chains[req.step * 4 + 2];
    int64_t end         = req.cmd_chains[req.step * 4 + 3];
    vector<int64_t> updated_result_table;

    for (int i = 0; i < req.row_num(); i++) {
        int64_t prev_id = req.get_row_column(i, req.var2column(start));
        int edge_num = 0;
        edge *edge_ptr;
        edge_ptr = g.get_edges_global(cfg->wid, prev_id, direction, predict, &edge_num);
        int64_t end_id = req.get_row_column(i, req.var2column(end));
        for (int k = 0; k < edge_num; k++) {
            if (edge_ptr[k].val == end_id) {
                req.append_row_to(i, updated_result_table);
                break;
            }
        }
    }
    req.result_table.swap(updated_result_table);
    req.step++;
}

void
server::known_to_const(request_or_reply &req)
{
    int64_t start       = req.cmd_chains[req.step * 4];
    int64_t predict     = req.cmd_chains[req.step * 4 + 1];
    int64_t direction   = req.cmd_chains[req.step * 4 + 2];
    int64_t end         = req.cmd_chains[req.step * 4 + 3];
    vector<int64_t> updated_result_table;

    for (int i = 0; i < req.row_num(); i++) {
        int64_t prev_id = req.get_row_column(i, req.var2column(start));
        int edge_num = 0;
        edge *edge_ptr;
        edge_ptr = g.get_edges_global(cfg->wid, prev_id, direction, predict, &edge_num);
        for (int k = 0; k < edge_num; k++) {
            if (edge_ptr[k].val == end) {
                req.append_row_to(i, updated_result_table);
                break;
            }
        }
    }
    req.result_table.swap(updated_result_table);
    req.step++;
}

void
server::index_to_unknown(request_or_reply &req)
{
    int64_t index_vertex = req.cmd_chains[req.step * 4];
    int64_t nothing      = req.cmd_chains[req.step * 4 + 1];
    int64_t direction    = req.cmd_chains[req.step * 4 + 2];
    int64_t var          = req.cmd_chains[req.step * 4 + 3];
    vector<int64_t> updated_result_table;

    if (!(req.column_num() == 0 && req.column_num() == req.var2column(var))) {
        //it means the query plan is wrong
        cout << "ncols: " << req.column_num() << "\t"
             << "var: " << var << "\t"
             << "var2col: " << req.var2column(var)
             << endl;
        assert(false);
    }

    int edge_num = 0;
    edge *edge_ptr;
    edge_ptr = g.local_storage.get_index_edges_local(cfg->wid, index_vertex, direction, &edge_num);
    int64_t start_id = req.mt_current_thread;
    for (int k = start_id; k < edge_num; k += req.mt_total_thread) {
        updated_result_table.push_back(edge_ptr[k].val);
    }

    req.result_table.swap(updated_result_table);
    req.set_column_num(1);
    req.step++;
    req.local_var = -1;
}

void
server::const_unknown_unknown(request_or_reply &req)
{
    int64_t start       = req.cmd_chains[req.step * 4];
    int64_t predict     = req.cmd_chains[req.step * 4 + 1];
    int64_t direction   = req.cmd_chains[req.step * 4 + 2];
    int64_t end         = req.cmd_chains[req.step * 4 + 3];
    vector<int64_t> updated_result_table;

    if (req.column_num() != 0 ) {
        //it means the query plan is wrong
        assert(false);
    }
    int npredict = 0;
    edge *predict_ptr = g.get_edges_global(cfg->wid, start, direction, 0, &npredict);
    // foreach possible predict
    for (int p = 0; p < npredict; p++) {
        int edge_num = 0;
        edge *edge_ptr;
        edge_ptr = g.get_edges_global(cfg->wid, start, direction, predict_ptr[p].val, &edge_num);
        for (int k = 0; k < edge_num; k++) {
            updated_result_table.push_back(predict_ptr[p].val);
            updated_result_table.push_back(edge_ptr[k].val);
        }
    }
    req.result_table.swap(updated_result_table);
    req.set_column_num(2);
    req.step++;
}

void
server::known_unknown_unknown(request_or_reply& req)
{
    int64_t start = req.cmd_chains[req.step * 4];
    int64_t predict = req.cmd_chains[req.step * 4 + 1];
    int64_t direction = req.cmd_chains[req.step * 4 + 2];
    int64_t end = req.cmd_chains[req.step * 4 + 3];
    vector<int64_t> updated_result_table;

    // foreach vertex
    for (int i = 0; i < req.row_num(); i++) {
        int64_t prev_id = req.get_row_column(i, req.var2column(start));
        int npredict = 0;
        edge *predict_ptr = g.get_edges_global(cfg->wid, prev_id, direction, 0, &npredict);
        // foreach possible predict
        for (int p = 0; p < npredict; p++) {
            int edge_num = 0;
            edge *edge_ptr;
            edge_ptr = g.get_edges_global(cfg->wid, prev_id, direction, predict_ptr[p].val, &edge_num);
            for (int k = 0; k < edge_num; k++) {
                req.append_row_to(i, updated_result_table);
                updated_result_table.push_back(predict_ptr[p].val);
                updated_result_table.push_back(edge_ptr[k].val);
            }
        }
    }

    req.set_column_num(req.column_num() + 2);
    req.result_table.swap(updated_result_table);
    req.step++;
}

void
server::known_unknown_const(request_or_reply& req)
{
    int64_t start = req.cmd_chains[req.step * 4];
    int64_t predict = req.cmd_chains[req.step * 4 + 1];
    int64_t direction = req.cmd_chains[req.step * 4 + 2];
    int64_t end = req.cmd_chains[req.step * 4 + 3];
    vector<int64_t> updated_result_table;

    // foreach vertex
    for (int i = 0; i < req.row_num(); i++) {
        int64_t prev_id = req.get_row_column(i, req.var2column(start));
        int npredict = 0;
        edge *predict_ptr = g.get_edges_global(cfg->wid, prev_id, direction, 0, &npredict);
        // foreach possible predict
        for (int p = 0; p < npredict; p++) {
            int edge_num = 0;
            edge *edge_ptr;
            edge_ptr = g.get_edges_global(cfg->wid, prev_id, direction, predict_ptr[p].val, &edge_num);
            for (int k = 0; k < edge_num; k++) {
                if (edge_ptr[k].val == end) {
                    req.append_row_to(i, updated_result_table);
                    updated_result_table.push_back(predict_ptr[p].val);
                    break;
                }
            }
        }
    }

    req.set_column_num(req.column_num() + 1);
    req.result_table.swap(updated_result_table);
    req.step++;
}

typedef std::pair<int, int> v_pair;

size_t
hash_pair(const v_pair &x)
{
    size_t r = x.first;
    r = r << 32;
    r += x.second;
    return hash<size_t>()(r);
}

void
server::do_corun(request_or_reply& req)
{
    // step.1 remove dup;
    uint64_t t0 = timer::get_usec();

    boost::unordered_set<int64_t> remove_dup_set;
    int64_t dup_var = req.cmd_chains[req.step * 4 + 4];
    assert(dup_var < 0);
    for (int i = 0; i < req.row_num(); i++) {
        remove_dup_set.insert(req.get_row_column(i, req.var2column(dup_var)));
    }

    // step.2 generate cmd_chain for sub-req
    vector<int64_t> sub_chain;
    boost::unordered_map<int64_t, int64_t> var_mapping;
    vector<int> reverse_mapping;
    int fetch_step = req.cmd_chains[req.step * 4 + 3];
    for (int i = req.step * 4 + 4; i < fetch_step * 4; i++) {
        if (req.cmd_chains[i] < 0 && ( var_mapping.find(req.cmd_chains[i]) == var_mapping.end()) ) {
            int64_t new_id = -1 - var_mapping.size();
            var_mapping[req.cmd_chains[i]] = new_id;
            reverse_mapping.push_back(req.var2column(req.cmd_chains[i]));
        }
        if (req.cmd_chains[i] < 0) {
            sub_chain.push_back(var_mapping[req.cmd_chains[i]]);
        } else {
            sub_chain.push_back(req.cmd_chains[i]);
        }
    }

    // step.3 make sub-req
    request_or_reply sub_req;
    {
        boost::unordered_set<int64_t>::iterator iter;
        for (iter = remove_dup_set.begin(); iter != remove_dup_set.end(); iter++) {
            sub_req.result_table.push_back(*iter);
        }
        sub_req.cmd_chains = sub_chain;
        sub_req.silent = false;
        sub_req.col_num = 1;
    }

    uint64_t t1 = timer::get_usec();
    // step.4 execute sub-req
    while (true) {
        execute_one_step(sub_req);
        if (sub_req.is_finished()) {
            break;
        }
    }
    uint64_t t2 = timer::get_usec();

    uint64_t t3, t4;
    vector<int64_t> updated_result_table;

    if (sub_req.column_num() > 2) {
        //if(true){ // always use qsort
        mytuple::qsort_tuple(sub_req.column_num(), sub_req.result_table);
        vector<int64_t> tmp_vec;
        tmp_vec.resize(sub_req.column_num());
        t3 = timer::get_usec();
        for (int i = 0; i < req.row_num(); i++) {
            for (int c = 0; c < reverse_mapping.size(); c++) {
                tmp_vec[c] = req.get_row_column(i, reverse_mapping[c]);
            }
            if (mytuple::binary_search_tuple(sub_req.column_num(), sub_req.result_table, tmp_vec)) {
                req.append_row_to(i, updated_result_table);
            }
        }
        t4 = timer::get_usec();
    } else { // hash join
        boost::unordered_set<v_pair> remote_set;
        for (int i = 0; i < sub_req.row_num(); i++) {
            remote_set.insert(v_pair(sub_req.get_row_column(i, 0), sub_req.get_row_column(i, 1)));
        }
        vector<int64_t> tmp_vec;
        tmp_vec.resize(sub_req.column_num());
        t3 = timer::get_usec();
        for (int i = 0; i < req.row_num(); i++) {
            for (int c = 0; c < reverse_mapping.size(); c++) {
                tmp_vec[c] = req.get_row_column(i, reverse_mapping[c]);
            }
            v_pair target = v_pair(tmp_vec[0], tmp_vec[1]);
            if (remote_set.find(target) != remote_set.end()) {
                req.append_row_to(i, updated_result_table);
            }
        }
        t4 = timer::get_usec();
    }
    if (cfg->sid == 0 && cfg->wid == global_nfewkrs) {
        cout << "prepare " << (t1 - t0) << " us" << endl;
        cout << "execute sub-request " << (t2 - t1) << " us" << endl;
        cout << "sort " << (t3 - t2) << " us" << endl;
        cout << "lookup " << (t4 - t3) << " us" << endl;
    }

    req.result_table.swap(updated_result_table);
    req.step = fetch_step;
}

bool
server::execute_one_step(request_or_reply& req)
{
    if (req.is_finished()) {
        return false;
    }
    if (req.step == 0 && req.use_index_vertex()) {
        index_to_unknown(req);
        return true;
    }
    int64_t start       = req.cmd_chains[req.step * 4];
    int64_t predict     = req.cmd_chains[req.step * 4 + 1];
    int64_t direction   = req.cmd_chains[req.step * 4 + 2];
    int64_t end         = req.cmd_chains[req.step * 4 + 3];

    if (predict < 0) {
        switch (var_pair(req.variable_type(start), req.variable_type(end))) {
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
    }

    // known_predict
    switch (var_pair(req.variable_type(start), req.variable_type(end))) {
    ///start from const_var
    case var_pair(const_var, const_var):
        cout << "error:const_var->const_var" << endl;
        assert(false);
        break;
    case var_pair(const_var, unknown_var):
        const_to_unknown(req);
        break;
    case var_pair(const_var, known_var):
        cout << "error:const_var->known_var" << endl;
        assert(false);
        break;

    ///start from known_var
    case var_pair(known_var, const_var):
        known_to_const(req);
        break;
    case var_pair(known_var, known_var):
        known_to_known(req);
        break;
    case var_pair(known_var, unknown_var):
        known_to_unknown(req);
        break;

    ///start from unknown_var
    case var_pair(unknown_var, const_var):
    case var_pair(unknown_var, known_var):
    case var_pair(unknown_var, unknown_var):
        cout << "error:unknown_var->" << endl;
        assert(false);
    default :
        cout << "default" << endl;
        break;
    }
    return true;
}

vector<request_or_reply>
server::generate_sub_query(request_or_reply& req)
{
    int64_t start = req.cmd_chains[req.step * 4];
    int64_t end = req.cmd_chains[req.step * 4 + 3];

    vector<request_or_reply> sub_reqs;
    int num_sub_request = global_nsrvs;
    sub_reqs.resize(num_sub_request);
    for (int i = 0; i < sub_reqs.size(); i++) {
        sub_reqs[i].parent_id = req.id;
        sub_reqs[i].cmd_chains = req.cmd_chains;
        sub_reqs[i].step = req.step;
        sub_reqs[i].col_num = req.col_num;
        sub_reqs[i].silent = req.silent;
        sub_reqs[i].local_var = start;
    }
    for (int i = 0; i < req.row_num(); i++) {
        int machine = mymath::hash_mod(req.get_row_column(i, req.var2column(start)), num_sub_request);
        req.append_row_to(i, sub_reqs[machine].result_table);
    }
    return sub_reqs;
}

vector<request_or_reply>
server::generate_mt_sub_requests(request_or_reply& req)
{
    int64_t start = req.cmd_chains[req.step * 4];
    int64_t end = req.cmd_chains[req.step * 4 + 3];
    int nthread = max(1, min(global_multithread_factor, global_nbewkrs));

    vector<request_or_reply> sub_reqs;
    int num_sub_request = global_nsrvs * nthread ;
    sub_reqs.resize(num_sub_request );
    for (int i = 0; i < sub_reqs.size(); i++) {
        sub_reqs[i].parent_id = req.id;
        sub_reqs[i].cmd_chains = req.cmd_chains;
        sub_reqs[i].step = req.step;
        sub_reqs[i].col_num = req.col_num;
        sub_reqs[i].silent = req.silent;
        sub_reqs[i].local_var = start;
    }
    for (int i = 0; i < req.row_num(); i++) {
        // id = wid*global_nsrvs + m_id
        //so  m_id = id % global_nsrvs
        //    wid = id / global_nsrvs
        int id = mymath::hash_mod(req.get_row_column(i, req.var2column(start)), num_sub_request);
        req.append_row_to(i, sub_reqs[id].result_table);
    }
    return sub_reqs;
}

bool
server::need_fork_join(request_or_reply &r)
{
    int64_t start = r.cmd_chains[r.step * 4];

    // fork-join or in-place execution
    return ((r.local_var != start) && (r.row_num() >= global_rdma_threshold));
}

void
server::execute(request_or_reply &req)
{
    uint64_t t1, t2;

    while (true) {
        t1 = timer::get_usec();
        execute_one_step(req);
        t2 = timer::get_usec();

        if (cfg->sid == 0 && cfg->wid == global_nfewkrs) { // debug
            //cout<<"step#" << req.step << ": " << t2-t1 << " us" <<endl;
        }

        // co-run execution
        if (!req.is_finished() && (req.cmd_chains[req.step * 4 + 2] == CORUN)) {
            t1 = timer::get_usec();
            do_corun(req);
            t2 = timer::get_usec();
            if ((cfg->sid == 0) && (cfg->wid == global_nfewkrs)) {
                //cout << "corun: " << t2-t1 <<" us" << endl;
            }
        }

        if (req.is_finished()) {
            req.silent_row_num = req.row_num();
            if (req.silent)
                req.clear_data();

            SendR(cfg, cfg->sid_of(req.parent_id), cfg->wid_of(req.parent_id), req);
            return;
        }

        if (need_fork_join(req)) {
            vector<request_or_reply> sub_rs = generate_sub_query(req);
            wqueue.put_parent_request(req, sub_rs.size());
            for (int i = 0; i < sub_rs.size(); i++) {
                if (i != cfg->sid) {
                    SendR(cfg, i, cfg->wid, sub_rs[i]);
                } else {
                    pthread_spin_lock(&recv_lock);
                    msg_fast_path.push_back(sub_rs[i]);
                    pthread_spin_unlock(&recv_lock);
                }
            }
            return;
        }
    }
    return;
}

void
server::run(void)
{
    int own_id = cfg->wid - global_nfewkrs;
    int possible_array[2] = {own_id , global_nbewkrs - 1 - own_id};
    uint64_t try_count = 0;

    while (true) {
        last_time = timer::get_usec();
        request_or_reply r;
        int recvid;

        // step 1: pool msg
        while (true) {
            // fast path (first)
            bool get_from_fast_path = false;

            pthread_spin_lock(&recv_lock);
            if (msg_fast_path.size() > 0) {
                r = msg_fast_path.back();
                msg_fast_path.pop_back();
                get_from_fast_path = true;
            }
            pthread_spin_unlock(&recv_lock);

            if (get_from_fast_path)
                break;

            // slow path
            int size = global_enable_workstealing ? 2 : 1;
            recvid = possible_array[try_count % size];
            try_count++;
            if (recvid == own_id) {
                // tryrecv
            } else {
                uint64_t last_of_other = s_array[recvid]->last_time;
                if (last_time > last_of_other && (last_time - last_of_other) > 10000 ) {
                    // tryrecv
                } else {
                    continue;
                }
            }

            bool success;
            pthread_spin_lock(&s_array[recvid]->recv_lock);
            success = TryRecvR(s_array[recvid]->cfg, r);
            if (success && recvid != own_id && r.use_index_vertex()) {
                s_array[recvid]->msg_fast_path.push_back(r);
                success = false;
            }
            pthread_spin_unlock(&s_array[recvid]->recv_lock);
            if (success) {
                break;
            }
        }

        // step 2: handle the msg
        if (r.is_request()) { // request
            r.id = cfg->get_and_inc_qid();
            int before = r.row_num();
            execute(r);
        } else { // reply
            pthread_spin_lock(&s_array[recvid]->wqueue_lock);
            s_array[recvid]->wqueue.put_reply(r);
            if (s_array[recvid]->wqueue.is_ready(r.parent_id)) {
                request_or_reply reply = s_array[recvid]->wqueue.get_merged_reply(r.parent_id);
                pthread_spin_unlock(&s_array[recvid]->wqueue_lock);
                SendR(cfg, cfg->sid_of(reply.parent_id), cfg->wid_of(reply.parent_id), reply);
            }
            pthread_spin_unlock(&s_array[recvid]->wqueue_lock);
        }
    }
}
