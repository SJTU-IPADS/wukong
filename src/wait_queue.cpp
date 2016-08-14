#include "wait_queue.h"

bool
wait_queue::is_ready(int parent_id)
{
    return internal_item_map[parent_id].count == 0;
}

request_or_reply
wait_queue::get_merged_reply(int parent_id)
{
    request_or_reply r = internal_item_map[parent_id].parent_request;
    request_or_reply& merged_reply = internal_item_map[parent_id].merged_reply;
    r.step = merged_reply.step;
    r.col_num = merged_reply.col_num;
    r.silent = merged_reply.silent;
    r.silent_row_num = merged_reply.silent_row_num;
    r.result_table.swap(merged_reply.result_table);
    internal_item_map.erase(parent_id);
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
    int parent_id = reply.parent_id;
    item& data = internal_item_map[parent_id];
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
