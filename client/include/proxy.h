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

#include <zhelpers.hpp>
#include <zmq.hpp>
#include <unordered_map>
#include <queue>

struct Task {

};

class Proxy {
public:
	zmq::context_t context;
	zmq::socket_t router;
	std::unordered_map<string, Task> task_map;

	Proxy(int _port): context(1), router(zmq::socket_t(context, ZMQ_ROUTER)) {
		s_set_id(router);
		sprintf(address, "tcp://*:%d", _port);
		router.bind(address);
	}

	void run() {
		while (1) {
			zmq::pollitem_t items[] = {{front_end, 0, ZMQ_POLLIN, 0}};
			zmq::poll(&items[0], 1, task_map.size() ? 0 : -1);
			if (items[0].revents & ZMQ_POLLIN) {
				std::string identity = s_recv(front_end);
				s_recv(front_end);
				std::string infor = s_recv(front_end);

			}
		}
	}
};