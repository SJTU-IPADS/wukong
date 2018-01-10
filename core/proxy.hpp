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
#include <unistd.h>

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


// a vector of pointers of all local proxies
class Proxy;
std::vector<Proxy *> proxies;

class Proxy {

private:
	class Message {
	public:
		int sid;
		int tid;
		Bundle bundle;

		Message(int sid, int tid, Bundle &bundle)
			: sid(sid), tid(tid), bundle(bundle) { }
	};

	vector<Message> pending_msgs;

	void fill_template(request_template &req_template) {
		req_template.ptypes_grp.resize(req_template.ptypes_str.size());
		for (int i = 0; i < req_template.ptypes_str.size(); i++) {
			string type = req_template.ptypes_str[i]; // the Types of random-constant

			SPARQLQuery type_request, type_reply;

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

			cout << "[INFO] " << type << " has "
			     << req_template.ptypes_grp[i].size() << " candidates" << endl;
		}
	}

	inline bool send(Bundle &bundle, int dst_sid, int dst_tid) {
		if (adaptor->send(dst_sid, dst_tid, bundle))
			return true;

		pending_msgs.push_back(Message(dst_sid, dst_tid, bundle));
		return false;
	}

	inline bool send(Bundle &bundle, int dst_sid) {
		// NOTE: the partitioned mapping has better tail latency in batch mode
		int range = global_num_engines / global_num_proxies;
		// FIXME: BUG if global_num_engines < global_num_proxies
		assert(range > 0);

		int base = global_num_proxies + (range * tid);
		// randomly choose engine without preferred one
		int dst_eid = coder.get_random() % range;

		// If the preferred engine is busy, try the rest engines with round robin
		for (int i = 0; i < range; i++)
			if (adaptor->send(dst_sid, base + (dst_eid + i) % range, bundle))
				return true;

		pending_msgs.push_back(Message(dst_sid, (base + dst_eid), bundle));
		return false;
	}

	inline void sweep_msgs() {
		if (!pending_msgs.size()) return;

		cout << "[INFO]#" << tid << " " << pending_msgs.size()
		     << " pending msgs on proxy." << endl;
		for (vector<Message>::iterator it = pending_msgs.begin();
		        it != pending_msgs.end();) {
			if (adaptor->send(it->sid, it->tid, it->bundle))
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


	Proxy(int sid, int tid, String_Server *str_server,
	      Adaptor *adaptor, data_statistic *statistic)
		: sid(sid), tid(tid), str_server(str_server), adaptor(adaptor),
		  coder(sid, tid), parser(str_server), statistic(statistic) { }

	void setpid(SPARQLQuery &r) { r.pid = coder.get_and_inc_qid(); }

	void setpid(RDFLoad &r) { r.pid = coder.get_and_inc_qid(); }

	void send_request(SPARQLQuery &r) {
		assert(r.pid != -1);

		// submit the request to all engines (parallel)
		if (r.start_from_index()) {
			for (int i = 0; i < global_num_servers; i++) {
				for (int j = 0; j < global_mt_threshold; j++) {
					r.tid = j; // specified engine
					Bundle bundle(r);
					send(bundle, i, global_num_proxies + j);
				}
			}
			return;
		}

		// submit the request to a certain server
		int start_sid = mymath::hash_mod(r.cmd_chains[0], global_num_servers);
		Bundle bundle(r);
		send(bundle, start_sid);
	}

	SPARQLQuery recv_reply(void) {
		Bundle bundle = adaptor->recv();
		assert(bundle.type == SPARQL_QUERY);
		SPARQLQuery r = bundle.get_sparql_query();
		if (r.start_from_index()) {
			for (int count = 0; count < global_num_servers * global_mt_threshold - 1; count++) {
				Bundle bundle2 = adaptor->recv();
				assert(bundle2.type == SPARQL_QUERY);
				SPARQLQuery r2 = bundle2.get_sparql_query();
				r.row_num += r2.row_num;
				int new_size = r.result_table.size() + r2.result_table.size();
				r.result_table.reserve(new_size);
				r.result_table.insert(r.result_table.end(),
				                      r2.result_table.begin(),
				                      r2.result_table.end());

				int new_attr_size = r.attr_res_table.size() + r2.attr_res_table.size();
				r.attr_res_table.reserve(new_attr_size);
				r.attr_res_table.insert(r.attr_res_table.end(),
				                        r2.attr_res_table.begin(),
				                        r2.attr_res_table.end());
			}
		}
		return r;
	}

	bool tryrecv_reply(SPARQLQuery &r) {
		Bundle bundle;
		bool success = adaptor->tryrecv(bundle);
		if (success) {
			assert(bundle.type == SPARQL_QUERY);
			r = bundle.get_sparql_query();

			if (r.start_from_index()) {
				cout << "Unsupport try recieve parallel query now!" << endl;
				assert(false);
			}
		}

		return success;
	}

	int run_single_query(istream &is, int cnt,
	                     SPARQLQuery &reply, Logger &logger) {
		SPARQLQuery request;
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
				return -3; // planning failed
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
		return 0; // success
	}

	int run_batch_query(istream &is, int d, int w, int s, int p, Logger &logger) {
		uint64_t duration = SEC(d);
		uint64_t warmup = SEC(w);
		uint64_t sleep = USEC(s);
		int parallel_factor = p, send_rate = p;
		int try_rounds = 5;

		int ntypes;
		is >> ntypes;
		if (ntypes <= 0) {
			cout << "[ERROR] invalid #query_types! (" << ntypes << " < 0)" << endl;
			return -2; // parsing failed
		}

		vector<request_template> tpls(ntypes);
		vector<int> loads(ntypes);

		for (int i = 0; i < ntypes; i++) {
			string fname;
			is >> fname;
			ifstream ifs(fname);
			if (!ifs) {
				cout << "[ERROR] Query file not found: " << fname << endl;
				return -1; // file not found
			}

			int load;
			is >> load;
			assert(load > 0);
			loads[i] = load;

			bool success = parser.parse_template(ifs, tpls[i]);
			if (!success) {
				cout << "[ERROR] Template parsing failed!" << endl;
				return -2; // parsing failed
			}

			fill_template(tpls[i]);
		}

		logger.init();

		bool timing = false;
		uint64_t send_cnt = 0, recv_cnt = 0, flying_cnt = 0;
		uint64_t init = timer::get_usec();
		while ((timer::get_usec() - init) < duration) {
			// send requests
			for (int t = 0; t < send_rate; t++) {
				sweep_msgs(); // sweep pending msgs first

				int idx = mymath::get_distribution(coder.get_random(), loads);
				SPARQLQuery request = tpls[idx].instantiate(coder.get_random());
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
				usleep(sleep / try_rounds);

				SPARQLQuery r;
				while (bool success = tryrecv_reply(r)) {
					recv_cnt++;
					logger.end_record(r.pid);
				}
			}

			logger.print_timely_thpt(recv_cnt, sid, tid);

			// skip warmup
			if (!timing && (timer::get_usec() - init) > warmup) {
				logger.start_thpt(recv_cnt);
				timing = true;
			}

                        flying_cnt = send_cnt - recv_cnt;

                        if (flying_cnt > parallel_factor) {  // send too many queries
				send_rate = send_rate - (int)(flying_cnt - parallel_factor);
                                send_rate = send_rate > 0 ? send_rate : 1; // at least send 1 query
                                continue;
                        }
                        if (send_rate < parallel_factor) {
				send_rate = send_rate + (parallel_factor - flying_cnt);
                                send_rate = send_rate < parallel_factor ?
                                    send_rate : parallel_factor; // at most send parallel_factor queries
                        }
		}
		logger.end_thpt(recv_cnt);

		// recieve all replies to calculate the tail latency
		while (recv_cnt < send_cnt) {
			sweep_msgs();	// sweep pending msgs first

			SPARQLQuery r;
			while (tryrecv_reply(r)) {
				recv_cnt ++;
				logger.end_record(r.pid);
			}

			logger.print_timely_thpt(recv_cnt, sid, tid);
		}

		logger.finish();

		return 0; // success
	}

#if DYNAMIC_GSTORE
	int dynamic_load_data(string &dname, RDFLoad &reply, Logger &logger) {
		RDFLoad request;
		request.type = DYNAMIC_LOAD;
		request.load_dname = dname;

		logger.init();

		setpid(request);
		for (int i = 0; i < global_num_servers; i++) {
			Bundle bundle(request);
			send(bundle, i);
		}

		int ret = 0;
		for (int i = 0; i < global_num_servers; i++) {
			Bundle bundle = adaptor->recv();
			assert(bundle.type == DYNAMIC_LOAD);

			reply = bundle.get_rdf_load();
			if (reply.load_ret < 0)
				ret = reply.load_ret;
		}

		logger.finish();
		return ret;
	}
#endif
};
