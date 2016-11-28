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

#include "query_basic_types.h"
#include "global_cfg.h"
#include "thread_cfg.h"
#include "message_wrap.h"
#include "sparql_parser.h"
#include "string_server.h"

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>


class client {
public:
	thread_cfg *cfg;

	string_server *str_server;

	sparql_parser parser;

	client(thread_cfg *_cfg, string_server *str_server);

	void setpid(request_or_reply &r) { r.pid = cfg->get_and_inc_qid(); }

	void send(request_or_reply &r);
	request_or_reply recv(void);

	void print_result(request_or_reply &r, int row2print);
};
