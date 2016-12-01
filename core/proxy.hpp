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

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

#include "query_basic_types.hpp"
#include "global_cfg.hpp"
#include "thread_cfg.hpp"
#include "message_wrap.hpp"
#include "parser.hpp"
#include "string_server.hpp"
#include "logger.hpp"

#include "mymath.hpp"
#include "timer.hpp"

int batch_factor = 20; // discard later

class Proxy {
private:

	void fill_request(request_template &tpl, request_or_reply &r) {
		for (int i = 0; i < tpl.ptypes_pos.size(); i++) {
			vector<int64_t> candidates = tpl.ptypes_grp[i];
			r.cmd_chains[tpl.ptypes_pos[i]] =
			    candidates[cfg->get_random() % (candidates.size())];
		}
	}

	void fill_template(request_template &req_template) {
		req_template.ptypes_grp.resize(req_template.ptypes_str.size());
		for (int i = 0; i < req_template.ptypes_str.size(); i++) {
			string type = req_template.ptypes_str[i]; // the Types of random-constant

			request_or_reply type_request, type_reply;

			// a TYPE query to collect constants with the certain type
			if (!parser.add_type_pattern(type, type_request)) {
				cout << "ERROR: failed to add a special type pattern (type: "
				     << type << ")." << endl;
				assert(false);
			}

			// do a TYPE query to collect all of candidates for a certain type
			setpid(type_request);
			send(type_request);
			type_reply = recv();

			vector<int64_t> candidates(type_reply.result_table);
			// There is no candidate with the Type for a random-constant in the template
			// TODO: it should report empty for all queries of the template
			assert(candidates.size() > 0);

			req_template.ptypes_grp[i] = candidates;

			cout << type << " has " << req_template.ptypes_grp[i].size() << " candidates" << endl;
		}
	}

public:
	thread_cfg *cfg;

	String_Server *str_server;

	Parser parser;

	Proxy(thread_cfg *_cfg, String_Server *_str_server):
		cfg(_cfg), str_server(_str_server), parser(_str_server) { }

	void setpid(request_or_reply &r) { r.pid = cfg->get_and_inc_qid(); }

	void send(request_or_reply &req) {
		assert(req.pid != -1);

		if (req.start_from_index()) {
			for (int i = 0; i < global_nsrvs; i++) {
				for (int j = 0; j < global_mt_threshold; j++) {
					req.tid = j;
					SendR(cfg, i, global_num_proxies + j, req);
				}
			}
			return ;
		}
		req.first_target = mymath::hash_mod(req.cmd_chains[0], global_nsrvs);

		/* use one-to-one mapping if there are multiple frontend workers */
		//int ratio = global_num_engines / global_num_proxies;
		//int mid = global_num_proxies + ratio * cfg->wid + cfg->get_random() % ratio;

		// random
		int tid = global_num_proxies + cfg->get_random() % global_num_engines;
		SendR(cfg, req.first_target, tid, req);
	}

	request_or_reply recv(void) {
		request_or_reply r = RecvR(cfg);
		if (r.start_from_index()) {
			for (int count = 0; count < global_nsrvs * global_mt_threshold - 1 ; count++) {
				request_or_reply r2 = RecvR(cfg);
				r.row_num += r2.row_num;
				int new_size = r.result_table.size() + r2.result_table.size();
				r.result_table.reserve(new_size);
				r.result_table.insert(r.result_table.end(), r2.result_table.begin(), r2.result_table.end());
			}
		}
		return r;
	}

	void print_result(request_or_reply &r, int row2print) {
		cout << "The first " << row2print << " rows of results: " << endl;
		for (int i = 0; i < row2print; i++) {
			cout << i + 1 << ":  ";
			for (int c = 0; c < r.get_col_num(); c++) {
				int id = r.get_row_col(i, c);
				/*
				 * Must load the entire ID mapping files (incl. normal and index),
				 * If you want to print the query results with strings.
				 */
				if (str_server->id2str.find(id) == str_server->id2str.end())
					cout << id << "\t";
				else
					cout << str_server->id2str[r.get_row_col(i, c)] << "  ";
			}
			cout << endl;
		}
	}

	void run_single_query(istream &is, int cnt) {
		request_or_reply request, reply;

		if (!parser.parse(is, request)) {
			cout << "ERROR: parse failed! ("
			     << parser.strerror << ")" << endl;
			return;
		}

		uint64_t t = timer::get_usec();
		for (int i = 0; i < cnt; i++) {
			setpid(request);
			send(request);
			reply = recv();
		}
		t = timer::get_usec() - t;

		cout << "(last) result size: " << reply.row_num << endl;
		cout << "(average) latency: " << (t / cnt) << " usec" << endl;

		if (!global_silent)
			print_result(reply, min(reply.row_num, global_max_print_row));
	}

	void nonblocking_run_batch_query(istream &is, Logger& logger) {
		int total_query_type;
		int total_request;
		int sleep_round = 1;

		is >> total_query_type >> total_request >> sleep_round;

		vector<int > loads;
		vector<request_template > tpls;
		//vector<request_or_reply > vec_req;

		tpls.resize(total_query_type);
		//vec_req.resize(total_query_type);
		for (int i = 0; i < total_query_type; i++) {
			string fname;
			is >> fname;
			ifstream ifs(fname);
			if (!ifs) {
				cout << "Query file not found: " << fname << endl;
				return ;
			}

			int load;
			is >> load;
			assert(load > 0);
			loads.push_back(load);

			bool success = parser.parse_template(ifs, tpls[i]);
			if (!success) {
				cout << "sparql parse error" << endl;
				return ;
			}

			fill_template(tpls[i]);
			//vec_req[i].cmd_chains = tpls[i].cmd_chains;
			//vec_req[i].silent = global_silent;
		}

		int send_request = 0, recv_request = 0;
		while (recv_request != total_request) {
			for (int t = 0; t < 10; t++) {
				if (send_request < total_request) {
					int idx = mymath::get_distribution(cfg->get_random(), loads);

					//request_or_reply r;
					//r.cmd_chains = tpls[idx].cmd_chains;
					request_or_reply r(tpls[idx].cmd_chains);
					fill_request(tpls[idx], r);

					setpid(r);
					logger.start_record(r.pid, idx);
					send(r);

					send_request++;
				}
			}

			for (int i = 0; i < sleep_round; i++) {
				timer::cpu_relax(100);
				request_or_reply reply;
				bool success = TryRecvR(cfg, reply);
				while (success) {
					recv_request++;
					logger.end_record(reply.pid);
					success = TryRecvR(cfg, reply);
				}
			}
		}
	}

	// discard later
	void run_batch_query(istream &is, Logger& logger) {
		int total_query_type;
		int total_request;
		int sleep_round = 1;

		is >> total_query_type >> total_request >> sleep_round;

		vector<int> loads;
		vector<request_template> vec_template;
		vector<request_or_reply> vec_req;

		vec_template.resize(total_query_type);
		vec_req.resize(total_query_type);
		for (int i = 0; i < total_query_type; i++) {
			string fname;
			is >> fname;
			ifstream ifs(fname);
			if (!ifs) {
				cout << "Query file not found: " << fname << endl;
				return ;
			}

			int load;
			is >> load;
			assert(load > 0);
			loads.push_back(load);

			bool success = parser.parse_template(ifs, vec_template[i]);
			if (!success) {
				cout << "SPARQL parse error" << endl;
				return ;
			}

			fill_template(vec_template[i]);
			vec_req[i].cmd_chains = vec_template[i].cmd_chains;
			vec_req[i].silent = global_silent;
		}
		uint64_t start_time = timer::get_usec();
		for (int i = 0; i < batch_factor; i++) {
			int idx = mymath::get_distribution(cfg->get_random(), loads);
			fill_request(vec_template[idx], vec_req[idx]);
			setpid(vec_req[idx]);
			logger.start_record(vec_req[idx].pid, idx);
			send(vec_req[idx]);
		}
		for (int i = 0; i < total_request; i++) {
			request_or_reply reply = recv();
			logger.end_record(reply.pid);
			int idx = mymath::get_distribution(cfg->get_random(), loads);
			fill_request(vec_template[idx], vec_req[idx]);
			setpid(vec_req[idx]);
			logger.start_record(vec_req[idx].pid, idx);
			send(vec_req[idx]);
		}
		for (int i = 0; i < batch_factor; i++) {
			request_or_reply reply = recv();
			logger.end_record(reply.pid);
		}
		uint64_t end_time = timer::get_usec();
		cout << 1000.0 * (total_request + batch_factor) / (end_time - start_time) << " Kops" << endl;
	}

};
