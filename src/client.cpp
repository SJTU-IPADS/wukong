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

#include "client.h"

client::client(thread_cfg *_cfg, string_server *_str_server):
    cfg(_cfg), str_server(_str_server), parser(_str_server) { }

void
client::setpid(request_or_reply &req)
{
    req.parent_id = cfg->get_and_inc_qid();
}

void
client::send(request_or_reply &req)
{
    if (req.parent_id == -1)
        setpid(req);

    if (req.use_index_vertex() && global_enable_index_partition) {
        int nthread = max(1, min(global_multithread_factor, global_nbewkrs));
        for (int i = 0; i < global_nsrvs; i++) {
            for (int j = 0; j < nthread; j++) {
                req.mt_total_thread = nthread;
                req.mt_current_thread = j;
                SendR(cfg, i, global_nfewkrs + j, req);
            }
        }
        return ;
    }
    req.first_target = mymath::hash_mod(req.cmd_chains[0], global_nsrvs);

    // one-to-one mapping
    //int server_per_client = global_nbewkrs / global_nfewkrs;
    //int mid = global_nfewkrs + server_per_client * cfg->wid + cfg->get_random() % server_per_client;

    // random
    int tid = global_nfewkrs + cfg->get_random() % global_nbewkrs;
    SendR(cfg, req.first_target, tid, req);
}

request_or_reply
client::recv(void)
{
    request_or_reply r = RecvR(cfg);
    if (r.use_index_vertex() && global_enable_index_partition ) {
        int nthread = max(1, min(global_multithread_factor, global_nbewkrs));
        for (int count = 0; count < global_nsrvs * nthread - 1 ; count++) {
            request_or_reply r2 = RecvR(cfg);
            r.silent_row_num += r2.silent_row_num;
            int new_size = r.result_table.size() + r2.result_table.size();
            r.result_table.reserve(new_size);
            r.result_table.insert( r.result_table.end(), r2.result_table.begin(), r2.result_table.end());
        }
    }

    return r;
}

void
client::print_result(request_or_reply &reply, int row_to_print)
{
    for (int i = 0; i < row_to_print; i++) {
        cout << i + 1 << ":  ";
        for (int c = 0; c < reply.column_num(); c++) {
            int id = reply.get_row_column(i, c);
            if (str_server->id2str.find(id) == str_server->id2str.end()) {
                cout << "NULL  ";
            } else {
                cout << str_server->id2str[reply.get_row_column(i, c)] << "  ";
            }
        }
        cout << endl;
    }
}
