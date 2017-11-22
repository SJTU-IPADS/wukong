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


#define PARALLEL_FACTOR 20


// a vector of pointers of all local proxies
class Proxy;
std::vector<Proxy *> proxies;

class Proxy {

private:
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

			cout << type << " has " << req_template.ptypes_grp[i].size() << " candidates" << endl;
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
					r.tid = j;
					adaptor->send(i, global_num_proxies + j, r);
				}
			}
			return ;
		}

		// submit the request to a certain engine
		int start_sid = mymath::hash_mod(r.cmd_chains[0], global_num_servers);

		// random assign request to range partitioned engines
		// NOTE: the partitioned mapping has better tail latency in batch mode
		int ratio = global_num_engines / global_num_proxies;
		// TODO: BUG if global_num_engines < global_num_proxies
		assert(ratio > 0);
		int start_tid = global_num_proxies + (ratio * tid) + (coder.get_random() % ratio);

		adaptor->send(start_sid, start_tid, r);
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
				int id = r.get_row_col(i, c);
				// WARNING: If you want to print the query results with strings,
				// must load the entire ID mapping files (i.e., global_load_minimal_index=false).
				//
				// TODO: good format
				if (str_server->id2str.find(id) == str_server->id2str.end())
					cout << id << "\t";
				else
					cout << str_server->id2str[r.get_row_col(i, c)] << "\t";
			}
			cout << endl;
		}
	}

	void dump_result(request_or_reply &r, string ofname) {
		if (boost::starts_with(ofname, "hdfs:")) {
			wukong::hdfs &hdfs = wukong::hdfs::get_hdfs();
			wukong::hdfs::fstream file(hdfs, ofname, true);
			for (int i = 0; i < r.row_num; i++) {
				file << i + 1 << ": ";
				for (int c = 0; c < r.get_col_num(); c++) {
					int id = r.get_row_col(i, c);
					// WARNING: If you want to print the query results with strings,
					// must load the entire ID mapping files (i.e., global_load_minimal_index=false).
					if (str_server->id2str.find(id) == str_server->id2str.end())
						file << id << "\t";
					else
						file << str_server->id2str[r.get_row_col(i, c)] << "\t";
				}
				file << endl;
			}
			file.close();
		} else {
			ofstream ofs(ofname, std::ios::out);
			if (!ofs.good()) {
				cout << "Can't open/create output file: " << ofname << endl;
			} else {
				for (int i = 0; i < r.row_num; i++) {
					ofs << i + 1 << ": ";
					for (int c = 0; c < r.get_col_num(); c++) {
						int id = r.get_row_col(i, c);
						// WARNING: If you want to print the query results with strings,
						// must load the entire ID mapping files (i.e., global_load_minimal_index=false).
						if (str_server->id2str.find(id) == str_server->id2str.end())
							ofs << id << "\t";
						else
							ofs << str_server->id2str[r.get_row_col(i, c)] << "\t";
					}
					ofs << endl;
				}
			}
			ofs.close();
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

	void nonblocking_run_batch_query(istream &is, Logger &logger) {
		int ntypes;
		int nqueries;
		int try_round = 1;

		is >> ntypes >> nqueries >> try_round;

		vector<request_template> tpls(ntypes);
		vector<int> loads(ntypes);

		for (int i = 0; i < ntypes; i++) {
			string fname;
			is >> fname;
			ifstream ifs(fname);
			if (!ifs) {
				cout << "ERROR: Query file not found: " << fname << endl;
				return ;
			}

			int load;
			is >> load;
			assert(load > 0);
			loads[i] = load;

			bool success = parser.parse_template(ifs, tpls[i]);
			if (!success) {
				cout << "ERROR: Template parsing failed!" << endl;
				return ;
			}
			fill_template(tpls[i]);
		}

		logger.init();
		int send_cnt = 0, recv_cnt = 0, flying_cnt = 0;
		while (recv_cnt < nqueries) {
			for (int t = 0; t < PARALLEL_FACTOR; t++) {
				if (send_cnt < nqueries) {
					int idx = mymath::get_distribution(coder.get_random(), loads);
					request_or_reply request = tpls[idx].instantiate(coder.get_random());
					if (global_enable_planner)
						planner.generate_plan(request, statistic);
					setpid(request);
					request.blind = true; // always not take back results in batch mode
					logger.start_record(request.pid, idx);
					send_request(request);
					send_cnt ++;
				}
			}

			// wait a piece of time and try several times
			for (int i = 0; i < try_round; i++) {
				timer::cpu_relax(100);

				// try to recieve the replies (best of effort)
				request_or_reply r;
				bool success = tryrecv_reply(r);
				while (success) {
					recv_cnt ++;
					logger.end_record(r.pid);

					success = tryrecv_reply(r);
				}
			}
		}
		logger.finish();
	}

	void run_batch_query(istream &is, Logger &logger) {
		int ntypes;
		int nqueries;
		int try_round = 1; // dummy

		is >> ntypes >> nqueries >> try_round;

		vector<int> loads(ntypes);
		vector<request_template> tpls(ntypes);

		// prepare various temples
		for (int i = 0; i < ntypes; i++) {
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
			loads[i] = load;

			bool success = parser.parse_template(ifs, tpls[i]);
			if (!success) {
				cout << "sparql parse error" << endl;
				return ;
			}
			fill_template(tpls[i]);
		}

		logger.init();
		// send PARALLEL_FACTOR queries and keep PARALLEL_FACTOR flying queries
		for (int i = 0; i < PARALLEL_FACTOR; i++) {
			int idx = mymath::get_distribution(coder.get_random(), loads);
			request_or_reply r = tpls[idx].instantiate(coder.get_random());

			setpid(r);
			r.blind = true;  // avoid send back results by default
			logger.start_record(r.pid, idx);
			send_request(r);
		}

		// recv one query, and then send another query
		for (int i = 0; i < nqueries - PARALLEL_FACTOR; i++) {
			// recv one query
			request_or_reply r2 = recv_reply();
			logger.end_record(r2.pid);

			// send another query
			int idx = mymath::get_distribution(coder.get_random(), loads);
			request_or_reply r = tpls[idx].instantiate(coder.get_random());

			setpid(r);
			r.blind = true;  // avoid send back results by default
			logger.start_record(r.pid, idx);
			send_request(r);
		}

		// recv the rest queries
		for (int i = 0; i < PARALLEL_FACTOR; i++) {
			request_or_reply r = recv_reply();
			logger.end_record(r.pid);
		}

		logger.finish();
		logger.print_thpt();
	}

};
