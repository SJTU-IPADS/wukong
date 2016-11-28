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

#include "wait_queue.h"

bool
wait_queue::is_ready(int pid)
{
    return internal_item_map[pid].count == 0;
}

request_or_reply
wait_queue::get_merged_reply(int pid)
{
    request_or_reply r = internal_item_map[pid].parent_request;
    request_or_reply& merged_reply = internal_item_map[pid].merged_reply;
    r.step = merged_reply.step;
    r.col_num = merged_reply.col_num;
    r.silent = merged_reply.silent;
    r.silent_row_num = merged_reply.silent_row_num;
    r.result_table.swap(merged_reply.result_table);
    internal_item_map.erase(pid);
    return r;
}

void
wait_queue::put_parent_request(request_or_reply &req, int count)
{
    item data;
    data.count = count;
    data.parent_request = req;
    internal_item_map[req.id] = data;
}

void
wait_queue::put_reply(request_or_reply &reply)
{
    int pid = reply.pid;
    item& data = internal_item_map[pid];
    vector<int64_t>& result_table = data.merged_reply.result_table;
    data.count--;
    data.merged_reply.step = reply.step;
    data.merged_reply.col_num = reply.col_num;
    data.merged_reply.silent = reply.silent;
    data.merged_reply.silent_row_num += reply.silent_row_num;
    int new_size = result_table.size() + reply.result_table.size();
    result_table.reserve(new_size);
    result_table.insert( result_table.end(), reply.result_table.begin(), reply.result_table.end());
};
