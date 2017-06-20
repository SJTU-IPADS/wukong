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

#include <zmq.hpp>
#include <zhelpers.hpp>
#include <pthread.h>
#include <iostream>
#include <sstream>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/unordered_map.hpp>
#include <tbb/concurrent_hash_map.h>

#include "cs_basic_type.hpp"
#include "proxy.hpp"

using namespace std;


class Monitor {
public:
	Proxy *proxy;
	int port;

	zmq::context_t context;
	zmq::socket_t *router;

	boost::lockfree::spsc_queue<CS_Request,
	      boost::lockfree::capacity<1024>> queue;
	pthread_spinlock_t send_lock;
	tbb::concurrent_hash_map<int, string> id_table; // from id to cid

	Monitor(Proxy *_proxy, int _port = 5450): proxy(_proxy), port(_port) {
		pthread_spin_init(&send_lock, 0);

		router = new zmq::socket_t(context, ZMQ_ROUTER);
		s_set_id(*router);
		char address[30] = "";
		sprintf(address, "tcp://*:%d", port + proxy->tid);
		cout << "port " << port + proxy->tid << endl;
		router->bind(address);
	}

	~Monitor() { delete router; }

	string recv(void) {
		string cid = s_recv(*router);
		s_recv(*router);
		string s = s_recv(*router);
		// queue.push(make_pair(cid, s));
		return s;
	}

	void send(std::string cid, std::string s) {
		s_sendmore(*router, cid);
		s_sendmore(*router, "");
		s_send(*router, s);
	}


	CS_Request recv_req(void) {
		string cid = s_recv(*router);
		s_recv(*router);
		string s = s_recv(*router);

		stringstream ss;
		ss << s;
		boost::archive::binary_iarchive ia(ss);

		CS_Request creq;
		ia >> creq;
		creq.cid = cid;
		return creq;
	}

	void send_rep(CS_Reply &crep) {
		pthread_spin_lock(&send_lock);

		stringstream ss;
		boost::archive::binary_oarchive oa(ss);
		oa << crep;

		s_sendmore(*router, crep.cid);
		s_sendmore(*router, "");
		s_send(*router, ss.str());

		pthread_spin_unlock(&send_lock);
	}

	void push(CS_Request creq) {
		/// TODO: check queue full
		queue.push(creq);
	}

	CS_Request pop(void) {
		while (queue.empty()) s_sleep(1000);

		CS_Request creq = queue.front();
		queue.pop();
		return creq;
	}

	void insert_cid(int id, string cid) {
		tbb::concurrent_hash_map<int, string>::accessor a;
		id_table.insert(a, id);
		a->second = cid;
	}

	string get_cid(int id) {
		tbb::concurrent_hash_map<int, string>::const_accessor a;
		if (id_table.find(a, id))
			return (string)(a->second);
		else
			return "";
	}

	int remove_cid(int id) {
		return id_table.erase(id);
	}
};

void *recv_cmd(void *ptr) {
	cout << "star to receive commands from clients" << endl;

	Monitor *monitor = (Monitor *)ptr;
	while (true) {
		cout << "wait to new recv" << endl;
		CS_Request creq = monitor->recv_req();
		monitor->push(creq);
		cout << "recv a new request" << endl;
	}
}

void *send_cmd(void *ptr) {
	cout << "start to send commands to clients" << endl;

	Monitor *monitor = (Monitor *)ptr;
	while (true) {
		request_or_reply r = monitor->proxy->recv_reply();
		cout << "(last) result size: " << r.row_num << endl;
		if (!global_silent && !r.blind)
			monitor->proxy->print_result(r, min(r.row_num, global_max_print_row));

		CS_Reply crep;
		crep.ncol = r.col_num;
		crep.result_table = r.result_table;
		crep.cid = monitor->get_cid(r.pid);
		monitor->send_rep(crep);
		monitor->remove_cid(r.pid);
	}
}

void run_monitor(Proxy *proxy, int port) {
	Monitor *monitor = new Monitor(proxy, port);
	pthread_t tid[2];
	pthread_create(&(tid[0]), NULL, recv_cmd, (void *)monitor);
	pthread_create(&(tid[1]), NULL, send_cmd, (void *)monitor);

	while (true) {
		CS_Request creq = monitor->pop();
		string fname = creq.content;
		cout << fname << endl;
		request_or_reply r;

		ifstream ifs(fname);
		if (!ifs) {
			cout << "Query file not found: " << fname << endl;
			continue;
		}

		bool ok = proxy->parser.parse(ifs, r);
		if (!ok) {
			cout << "ERROR: SPARQL query parse error" << endl;
			CS_Reply crep;
			crep.type = "error";
			crep.content = "bad file";
			crep.cid = creq.cid;
			monitor->send_rep(crep);
			continue;
		}

		proxy->setpid(r);
		proxy->send_request(r);
		monitor->insert_cid(r.pid, creq.cid);
	}

	for (int i = 0; i < 2; i++) {
		int rc = pthread_join(tid[i], NULL);
		if (rc) {
			printf("ERROR: return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
	}
}

