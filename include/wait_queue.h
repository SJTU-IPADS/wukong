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

#pragma once

#include <vector>
#include <pthread.h>
#include <boost/unordered_map.hpp>

#include "global_cfg.h"
#include "query_basic_types.h"

struct item {
	int count;
	request_or_reply parent_request;
	request_or_reply merged_reply;
};

class wait_queue {
	boost::unordered_map<int, item> internal_item_map;
public:
	void put_parent_request(request_or_reply &r, int count);
	void put_reply(request_or_reply &r);
	bool is_ready(int pid);
	request_or_reply get_merged_reply(int pid);
};
