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

#include "builtin_console.h"

static void
client_barrier(struct thread_cfg *cfg)
{
	static int _curr = 0;
	static __thread int _next = 1;

	// inter-server barrier
	if (cfg->wid == 0)
		MPI_Barrier(MPI_COMM_WORLD);

	// intra-server barrier
	__sync_fetch_and_add(&_curr, 1);
	while (_curr < _next)
		usleep(1); // wait
	_next += global_nfewkrs; // next barrier
}

void
run_single_query(client *clnt, istream &is, int cnt)
{
	request_or_reply request, reply;

	if (!clnt->parser.parse(is, request)) {
		cout << "ERROR: parse SPARQL query failed!" << endl;
		return;
	}

	request.silent = global_silent;
	uint64_t t = timer::get_usec();
	for (int i = 0; i < cnt; i++) {
		clnt->send(request);
		reply = clnt->recv();
	}
	t = timer::get_usec() - t;

	cout << "(last) result size: " << reply.silent_row_num << endl;
	cout << "(average) latency: " << (t / cnt) << " usec" << endl;

	int row_to_print = min((uint64_t)reply.row_num(),
	                       (uint64_t)global_max_print_row);
	if (row_to_print > 0)
		clnt->print_result(reply, row_to_print);
}


static void
translate_req_template(client *clnt, request_template &req_template)
{
	req_template.ptypes_grp.resize(req_template.ptypes_str.size());
	for (int i = 0; i < req_template.ptypes_str.size(); i++) {
		string type = req_template.ptypes_str[i];

		request_or_reply type_request, type_reply;

		// a TYPE query to collect constants with the certain type
		if (!clnt->parser.add_type_pattern(type, type_request)) {
			cout << "ERROR: failed to add a special type pattern (type: "
			     << type << ")." << endl;
			assert(false);
		}

		// do TYPE query
		clnt->send(type_request);
		type_reply = clnt->recv();

		vector<int64_t> *ptr = new vector<int64_t>(type_reply.result_table);
		req_template.ptypes_grp[i] = ptr;

		cout << type << " has " << ptr->size() << " objects" << endl;
	}
}

static void
instantiate_request(client *clnt, request_template &req_template, request_or_reply &r)
{
	for (int i = 0; i < req_template.ptypes_pos.size(); i++) {
		int pos = req_template.ptypes_pos[i];
		vector<int64_t> *vecptr = req_template.ptypes_grp[i];
		if (vecptr == NULL || vecptr->size() == 0) {
			assert(false);
		}
		r.cmd_chains[pos] = (*vecptr)[clnt->cfg->get_random() % (vecptr->size())];
	}
}

void
batch_execute(client* clnt, istream &is, batch_logger& logger)
{
	int total_query_type;
	int total_request;
	int sleep_round = 1;

	is >> total_query_type >> total_request >> sleep_round;

	vector<int > loads;
	vector<request_template > vec_template;
	vector<request_or_reply > vec_req;
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
		loads.push_back(load);
		bool success = clnt->parser.parse_template(ifs, vec_template[i]);
		translate_req_template(clnt, vec_template[i]);
		vec_req[i].cmd_chains = vec_template[i].cmd_chains;
		if (!success) {
			cout << "SPARQL parse error" << endl;
			return ;
		}
		vec_req[i].silent = global_silent;
	}
	uint64_t start_time = timer::get_usec();
	for (int i = 0; i < global_batch_factor; i++) {
		int idx = mymath::get_distribution(clnt->cfg->get_random(), loads);
		instantiate_request(clnt, vec_template[idx], vec_req[idx]);
		clnt->setpid(vec_req[idx]);
		logger.start_record(vec_req[idx].parent_id, idx);
		clnt->send(vec_req[idx]);
	}
	for (int i = 0; i < total_request; i++) {
		request_or_reply reply = clnt->recv();
		logger.end_record(reply.parent_id);
		int idx = mymath::get_distribution(clnt->cfg->get_random(), loads);
		instantiate_request(clnt, vec_template[idx], vec_req[idx]);
		clnt->setpid(vec_req[idx]);
		logger.start_record(vec_req[idx].parent_id, idx);
		clnt->send(vec_req[idx]);
	}
	for (int i = 0; i < global_batch_factor; i++) {
		request_or_reply reply = clnt->recv();
		logger.end_record(reply.parent_id);
	}
	uint64_t end_time = timer::get_usec();
	cout << 1000.0 * (total_request + global_batch_factor) / (end_time - start_time) << " Kops" << endl;
}


void
nonblocking_execute(client* clnt, istream &is, batch_logger& logger)
{
	int total_query_type;
	int total_request;
	int sleep_round = 1;

	is >> total_query_type >> total_request >> sleep_round;

	vector<int > loads;
	vector<request_template > vec_template;
	vector<request_or_reply > vec_req;

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
		loads.push_back(load);
		bool success = clnt->parser.parse_template(ifs, vec_template[i]);
		translate_req_template(clnt, vec_template[i]);
		vec_req[i].cmd_chains = vec_template[i].cmd_chains;
		if (!success) {
			cout << "sparql parse error" << endl;
			return ;
		}
		vec_req[i].silent = global_silent;
	}

	int send_request = 0;
	int recv_request = 0;
	while (recv_request != total_request) {
		for (int t = 0; t < 10; t++) {
			if (send_request < total_request) {
				send_request++;
				int idx = mymath::get_distribution(clnt->cfg->get_random(), loads);
				instantiate_request(clnt, vec_template[idx], vec_req[idx]);
				clnt->setpid(vec_req[idx]);
				logger.start_record(vec_req[idx].parent_id, idx);
				clnt->send(vec_req[idx]);
			}
		}

		for (int i = 0; i < sleep_round; i++) {
			timer::cpu_relax(100);
			request_or_reply reply;
			bool success = TryRecvR(clnt->cfg, reply);
			while (success) {
				recv_request++;
				logger.end_record(reply.parent_id);
				success = TryRecvR(clnt->cfg, reply);
			}
		}
	}
}

void
print_help(void)
{
	cout << "These are common Wukong commands: " << endl;
	cout << "    help         Display help infomation" << endl;
	cout << "    quit         Quit from client" << endl;
	cout << "    reconfig     Reload config file" << endl;
	cout << "    sparql       Run SPARQL queries" << endl;
	cout << "        -f <file>   a single query from the <file>" << endl;
	cout << "        -n <num>    run a single query <num> times" << endl;
	cout << "        -b <file>   a set of queries configured by the <file>" << endl;
	cout << "        -s <string> a single query from input string" << endl;
}

#define IS_MASTER(_cfg) ((_cfg)->sid == 0 && (_cfg)->wid == 0)
#define PRINT_ID(_cfg) (cout << "[" << (_cfg)->sid << "-" << (_cfg)->wid << "]$ ")

/**
 * The Wukong's builtin client
 */
void
builtin_console(client *clnt)
{
	struct thread_cfg *cfg = clnt->cfg;

	// the master client worker (i.e., sid == 0 and wid == 0)
	client_barrier(cfg);
	if (IS_MASTER(cfg))
		cout << endl
		     << "Input \'help\'' command to get more information"
		     << endl
		     << endl;

	while (true) {
		client_barrier(cfg);

next:
		string cmd;
		if (IS_MASTER(cfg)) {
			cout << "> ";
			std::getline(std::cin, cmd);

			// trim input
			size_t pos = cmd.find_first_not_of(" \t"); // trim blanks from head
			if (pos == string::npos) goto next;
			cmd.erase(0, pos);

			pos = cmd.find_last_not_of(" \t");  // trim blanks from tail
			cmd.erase(pos + 1, cmd.length() - (pos + 1));

			if (cmd == "help") {
				print_help();
				goto next;
			}

			// send commands to all client workers
			for (int i = 0; i < global_nsrvs; i++) {
				for (int j = 0; j < global_nfewkrs; j++) {
					if (i == 0 && j == 0)
						continue ;
					cfg->node->Send(i, j, cmd);
				}
			}
		} else {
			// recieve commands
			cmd = cfg->node->Recv();
		}

		if (cmd == "quit" || cmd == "q") {
			if (cfg->wid == 0)
				exit(0); // each server exits once
		} else if (cmd == "reconfig") {
			if (cfg->wid == 0)
				reload_cfg(); // each server reload config file
		} else {
			std::stringstream cmd_ss(cmd);
			std::string token;

			// get keyword of command
			cmd_ss >> token;

			// handle SPARQL queries
			if (token == "sparql") {
				string fname, bfname, query;
				int cnt = 1;
				bool f_enable = false, b_enable = false, q_enable = false;

				// parse parameters
				while (cmd_ss >> token) {
					if (token == "-f") {
						cmd_ss >> fname;
						f_enable = true;
					} else if (token == "-n") {
						cmd_ss >> cnt;
					} else if (token == "-b") {
						cmd_ss >> bfname;
						b_enable = true;
					} else if (token == "-s") {
						string start;
						cmd_ss >> start;
						query = cmd.substr(cmd.find(start));
						q_enable = true;
						break ;
					} else {
						if (IS_MASTER(cfg)) {
							cout << "Unknown option: " << token << endl;
							print_help();
						}
						goto next;
					}
				}

				if (f_enable) {
					// use the master client to run a single query
					if (IS_MASTER(cfg)) {
						ifstream ifs(fname);
						if (!ifs) {
							cout << "Query file not found: " << fname << endl;
							continue ;
						}
						run_single_query(clnt, ifs, cnt);
					}
				}

				if (b_enable) {
					batch_logger logger;

					// dedicate the master frontend worker to run a single query
					// and others to run a set of queries if '-f' is enabled
					if (!f_enable || !IS_MASTER(cfg)) {
						ifstream ifs(bfname);
						if (!ifs) {
							PRINT_ID(cfg);
							cout << "Configure file not found: " << bfname << endl;
							continue ;
						}

						logger.init();
						nonblocking_execute(clnt, ifs, logger);
						//batch_execute(clnt,filename,logger);
						logger.finish();
					}

					client_barrier(clnt->cfg);
					// print a statistic of runtime for the batch processing
					if (IS_MASTER(cfg)) {
						// collect logs from other clients
						for (int i = 0; i < global_nsrvs * global_nfewkrs - 1; i++) {
							batch_logger log = RecvObject<batch_logger>(clnt->cfg);
							logger.merge(log);
						}
						logger.print();
					} else {
						// send logs to the master client
						SendObject<batch_logger>(clnt->cfg, 0, 0, logger);
					}

				}

				if (q_enable) {
					// TODO: SPARQL string
					if (IS_MASTER(cfg)) {
						// TODO
						cout << "Query: " << query << endl;
						cout << "The option '-s' is unsupported now!" << endl;
					}
				}
			} else {
				if (IS_MASTER(cfg)) {
					cout << "Unknown command: " << token << endl;
					print_help();
				}
			}
		}
	}
}
