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
client::send(request_or_reply &req)
{
    assert(req.pid != -1);

    if (req.start_from_index()) {
        int nthread = max(1, min(global_mt_threshold, global_nbewkrs));
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
    if (r.start_from_index()) {
        int nthread = max(1, min(global_mt_threshold, global_nbewkrs));
        for (int count = 0; count < global_nsrvs * nthread - 1 ; count++) {
            request_or_reply r2 = RecvR(cfg);
            r.row_num += r2.row_num;
            int new_size = r.result_table.size() + r2.result_table.size();
            r.result_table.reserve(new_size);
            r.result_table.insert(r.result_table.end(), r2.result_table.begin(), r2.result_table.end());
        }
    }

    return r;
}

void
client::print_result(request_or_reply &r, int row2print)
{
    cout << "The first " << row2print << " rows of results: " << endl;
    for (int i = 0; i < row2print; i++) {
        cout << i + 1 << ":  ";
        for (int c = 0; c < r.get_col_num(); c++) {
            int id = r.get_row_col(i, c);
            if (str_server->id2str.find(id) == str_server->id2str.end())
                cout << id << "\t";
            else
                cout << str_server->id2str[r.get_row_col(i, c)] << "  ";
        }
        cout << endl;
    }
}
