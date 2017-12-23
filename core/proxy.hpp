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
 *      http://ipads.se.sjtu.edu.cn/projects/wukong
 *
 */

#pragma once

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>

#include "config.hpp"
#include "coder.hpp"
#include "query.hpp"
#include "adaptor.hpp"
#include "parser.hpp"
#include "planner.hpp"
#include "data_statistic.hpp"
#include "string_server.hpp"
#include "logger.hpp"

#include "mymath.hpp"
#include "timer.hpp"

using namespace std;

#define PARALLEL_FACTOR 20


// a vector of pointers of all local proxies
class Proxy;
std::vector<Proxy *> proxies;

class Proxy {

private:
	class Message {
	public:
		int sid;
		int tid;
		request_or_reply r;

		Message(int sid, int tid, request_or_reply &r)
			: sid(sid), tid(tid), r(r) { }
	};

	vector<Message> pending_msgs;

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
			type_request.blind = false; // must take back the results
			send_request(type_request);
			type_reply = recv_reply();

			vector<sid_t> candidates(type_reply.result_table);
			// There is no candidate with the Type for a random-constant in the template
			// TODO: it should report empty for all queries of the template
			assert(candidates.size() > 0);

			req_template.ptypes_grp[i] = candidates;

			/* cout << "[INFO] " << type << " has "
			     << req_template.ptypes_grp[i].size() << " candidates" << endl; */
		}
	}

	inline bool send(request_or_reply &r, int dst_sid, int dst_tid) {
		if (adaptor->send(dst_sid, dst_tid, r))
			return true;

		pending_msgs.push_back(Message(dst_sid, dst_tid, r));
		return false;
	}

	inline bool send(request_or_reply &r, int dst_sid) {
		// NOTE: the partitioned mapping has better tail latency in batch mode
		int range = global_num_engines / global_num_proxies;
		// FIXME: BUG if global_num_engines < global_num_proxies
		assert(range > 0);

		int base = global_num_proxies + (range * tid);
		// randomly choose engine without preferred one
		int dst_eid = coder.get_random() % range;

		// If the preferred engine is busy, try the rest engines with round robin
		for (int i = 0; i < range; i++)
			if (adaptor->send(dst_sid, base + (dst_eid + i) % range, r))
				return true;

		pending_msgs.push_back(Message(dst_sid, (base + dst_eid), r));
		return false;
	}

	inline void sweep_msgs() {
		if (!pending_msgs.size()) return;

		for (vector<Message>::iterator it = pending_msgs.begin(); it != pending_msgs.end();) {
			if (adaptor->send(it->sid, it->tid, it->r))
				it = pending_msgs.erase(it);
			else
				++it;
		}
	}

public:
	int sid;    // server id
	int tid;    // thread id

	String_Server *str_server;
	Adaptor *adaptor;

	Coder coder;
	Parser parser;
	Planner planner;
	data_statistic *statistic;


	Proxy(int sid, int tid, String_Server *str_server, Adaptor *adaptor, data_statistic *statistic)
		: sid(sid), tid(tid), str_server(str_server), adaptor(adaptor),
		  coder(sid, tid), parser(str_server), statistic(statistic) { }

	void setpid(request_or_reply &r) { r.pid = coder.get_and_inc_qid(); }

	void send_request(request_or_reply &r) {
		assert(r.pid != -1);

		// submit the request to all engines (parallel)
		if (r.start_from_index()) {
			for (int i = 0; i < global_num_servers; i++) {
				for (int j = 0; j < global_mt_threshold; j++) {
					r.tid = j; // specified engine
					send(r, i, global_num_proxies + j);
				}
			}
			return;
		}

		// submit the request to a certain server
		int start_sid = mymath::hash_mod(r.cmd_chains[0], global_num_servers);
		send(r, start_sid);
	}

	request_or_reply recv_reply(void) {
		request_or_reply r = adaptor->recv();
		if (r.start_from_index()) {
			for (int count = 0; count < global_num_servers * global_mt_threshold - 1 ; count++) {
				request_or_reply r2 = adaptor->recv();
				r.row_num += r2.row_num;
				int new_size = r.result_table.size() + r2.result_table.size();
				r.result_table.reserve(new_size);
				r.result_table.insert(r.result_table.end(), r2.result_table.begin(), r2.result_table.end());

				int new_attr_size = r.attr_res_table.size() + r2.attr_res_table.size();
				r.attr_res_table.reserve(new_attr_size);
				r.attr_res_table.insert(r.attr_res_table.end(), r2.attr_res_table.begin(), r2.attr_res_table.end());
			}
		}
		return r;
	}

	bool tryrecv_reply(request_or_reply &r) {
		bool success = adaptor->tryrecv(r);
		if (success && r.start_from_index()) {
			// TODO: avoid parallel submit for try recieve mode
			cout << "Unsupport try recieve parallel query now!" << endl;
			assert(false);
		}

		return success;
	}

	void print_result(request_or_reply &r, int row2print) {
		cout << "The first " << row2print << " rows of results: " << endl;
		for (int i = 0; i < row2print; i++) {
			cout << i + 1 << ":  ";
			for (int c = 0; c < r.get_col_num(); c++) {
				sid_t id = r.get_row_col(i, c);
				// WARNING: If you want to print the query results with strings,
				// must load the entire ID mapping files (i.e., global_load_minimal_index=false).
				//
				// TODO: good format
				if (str_server->exist(id))
					cout << str_server->id2str[id] << "\t";
				else
					cout << id << "\t";
			}
			for (int c = 0; c < r.get_attr_col_num(); c++) {
				attr_t  tmp = r.get_attr_row_col(i, c);
				cout << tmp << "\t";
			}
			cout << endl;
		}
	}

	void dump_result(request_or_reply &r, string ofname) {
		if (boost::starts_with(ofname, "hdfs:")) {
			wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
			wukong::hdfs::fstream file(hdfs, ofname, true);

			// FIXME: row_num vs. get_col_num()
			for (int i = 0; i < r.row_num; i++) {
				file << i + 1 << ": ";
				for (int c = 0; c < r.get_col_num(); c++) {
					sid_t id = r.get_row_col(i, c);
					// WARNING: If you want to print the query results with strings,
					// must load the entire ID mapping files (i.e., global_load_minimal_index=false).
					if (str_server->exist(id))
						file << str_server->id2str[id] << "\t";
					else
						file << id << "\t";
				}
				file << endl;
			}
			file.close();
		} else {
			ofstream file(ofname, std::ios::out);
			if (!file.good()) {
				cout << "Can't open/create output file: " << ofname << endl;
				return;
			}

			// FIXME: row_num vs. get_col_num()
			for (int i = 0; i < r.row_num; i++) {
				file << i + 1 << ": ";
				for (int c = 0; c < r.get_col_num(); c++) {
					sid_t id = r.get_row_col(i, c);
					// WARNING: If you want to print the query results with strings,
					// must load the entire ID mapping files (i.e., global_load_minimal_index=false).
					if (str_server->exist(id))
						file << str_server->id2str[id] << "\t";
					else
						file << id << "\t";
				}
				file << endl;
			}
			file.close();
		}
	}

	int run_single_query(istream &is, int cnt,
	                     request_or_reply &reply, Logger &logger) {
		request_or_reply request;
		uint64_t t_parse1 = timer::get_usec();
		if (!parser.parse(is, request)) {
			cout << "ERROR: Parsing failed! ("
			     << parser.strerror << ")" << endl;
			is.clear();
			is.seekg(0);
			return -2; // parsing failed
		}
		uint64_t t_parse2 = timer::get_usec();

		if (global_enable_planner) {
			// planner
			uint64_t t_plan1 = timer::get_usec();
			bool exec = planner.generate_plan(request, statistic);
			uint64_t t_plan2 = timer::get_usec();
			cout << "parsing time : " << t_parse2 - t_parse1 << " usec" << endl;
			cout << "planning time : " << t_plan2 - t_plan1 << " usec" << endl;
			if (exec == false) { // for empty result
				cout << "(last) result size: 0" << endl;
				return -1; // planning failed
			}
		}

		logger.init();
		for (int i = 0; i < cnt; i++) {
			setpid(request);
			// only take back results of the last request if not silent
			request.blind = i < (cnt - 1) ? true : global_silent;
			send_request(request);
			reply = recv_reply();
		}
		logger.finish();
		return 0;
	}

	void run_batch_query(istream &is, Logger &logger) {
		uint64_t duration  = 10000000;  // 10sec

		int ntypes;
		int nqueries;
		int try_rounds = 1;

		is >> ntypes >> try_rounds;

		vector<request_template> tpls(ntypes);
		vector<int> loads(ntypes);

		for (int i = 0; i < ntypes; i++) {
			string fname;
			is >> fname;
			ifstream ifs(fname);
			if (!ifs) {
				cout << "[ERROR] Query file not found: " << fname << endl;
				return ;
			}

			int load;
			is >> load;
			assert(load > 0);
			loads[i] = load;

			bool success = parser.parse_template(ifs, tpls[i]);
			if (!success) {
				cout << "[ERROR] Template parsing failed!" << endl;
				return ;
			}
			fill_template(tpls[i]);
		}

		logger.init();
		uint64_t send_cnt = 0, recv_cnt = 0;

		bool done = false;
		uint64_t init = timer::get_usec();
		while (done) {
			// send requests
			for (int t = 0; t < PARALLEL_FACTOR; t++) {
				sweep_msgs(); // sweep pending msgs first

				int idx = mymath::get_distribution(coder.get_random(), loads);
				request_or_reply request = tpls[idx].instantiate(coder.get_random());
				if (global_enable_planner)
					planner.generate_plan(request, statistic);
				setpid(request);
				request.blind = true; // always not take back results in batch mode

				logger.start_record(request.pid, idx);
				send_request(request);

				send_cnt++;
			}

			// recieve replies (best of effort)
			for (int i = 0; i < try_rounds; i++) {
				request_or_reply r;
				while (bool success = tryrecv_reply(r)) {
					recv_cnt++;
					logger.end_record(r.pid);
				}
			}

			logger.print_timely_thpt(recv_cnt, sid, tid);
			done = (timer::get_usec() - init) > duration;
		}

		while (recv_cnt < send_cnt) {
			request_or_reply r;
			if (tryrecv_reply(r)) {
				recv_cnt ++;
				logger.end_record(r.pid);
			}

			logger.print_timely_thpt(recv_cnt, sid, tid);
		}
	}
};
