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

#include <iostream>
#include <string>
#include <boost/unordered_map.hpp>
#include <set>

#include "utils.h"
#include "global_cfg.hpp"
#include "thread_cfg.hpp"
#include "proxy.hpp"
#include "logger.hpp"

using namespace std;

static void
console_barrier(struct thread_cfg *cfg)
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
	_next += global_num_proxies; // next barrier
}

void
print_help(void)
{
	cout << "These are common Wukong commands: " << endl;
	cout << "    help         Display help infomation" << endl;
	cout << "    quit         Quit from console" << endl;
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
 * The Wukong's console is co-located with the main proxy (the 1st proxy thread on the 1st server)
 * and provide a simple interactive cmdline to tester
 */
void
run_console(Proxy *proxy)
{
	struct thread_cfg *cfg = proxy->cfg;

	// the main proxy thread (i.e., sid == 0 and wid == 0)
	console_barrier(cfg);
	if (IS_MASTER(cfg))
		cout << endl
		     << "Input \'help\'' command to get more information"
		     << endl
		     << endl;

	while (true) {
		console_barrier(cfg);

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

			// send commands to all proxy threads
			for (int i = 0; i < global_nsrvs; i++) {
				for (int j = 0; j < global_num_proxies; j++) {
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
					// use the main proxy thread to run a single query
					if (IS_MASTER(cfg)) {
						ifstream ifs(fname);
						if (!ifs) {
							cout << "Query file not found: " << fname << endl;
							continue ;
						}
						proxy->run_single_query(ifs, cnt);
					}
				}

				if (b_enable) {
					Logger logger;

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
						proxy->nonblocking_run_batch_query(ifs, logger);
						//proxy->run_batch_query(ifs,logger);
						logger.finish();
					}

					console_barrier(cfg);
					// print a statistic of runtime for the batch processing
					if (IS_MASTER(cfg)) {
						// collect logs from other proxy threads
						for (int i = 0; i < global_nsrvs * global_num_proxies - 1; i++) {
							Logger log = RecvObject<Logger>(cfg);
							logger.merge(log);
						}
						logger.print();
					} else {
						// send logs to the main proxy
						SendObject<Logger>(cfg, 0, 0, logger);
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