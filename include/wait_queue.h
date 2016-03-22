#pragma once
#include <vector>
#include <pthread.h>
#include <boost/unordered_map.hpp>

#include "global_cfg.h"
#include "query_basic_types.h"

struct item{
	int count;
    request_or_reply parent_request;
    request_or_reply merged_reply;
};
class wait_queue{
    boost::unordered_map<int,item> internal_item_map;
public:
    void put_parent_request(request_or_reply& req,int count);
	void put_reply(request_or_reply& reply);
	bool is_ready(int parent_id);
	request_or_reply get_merged_reply(int parent_id);
};
