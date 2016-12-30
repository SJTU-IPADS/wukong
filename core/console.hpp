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

#include "config.hpp"
#include "proxy.hpp"
#include "logger.hpp"

using namespace std;

static void console_barrier(int tid)
{
	static int _curr = 0;
	static __thread int _next = 1;

	// inter-server barrier
	if (tid == 0)
		MPI_Barrier(MPI_COMM_WORLD);

	// intra-server barrier
	__sync_fetch_and_add(&_curr, 1);
	while (_curr < _next)
		usleep(1); // wait
	_next += global_num_proxies; // next barrier
}

void print_help(void)
{
	cout << "These are common Wukong commands: " << endl;
	cout << "    help           Display help infomation" << endl;
	cout << "    quit           Quit from console" << endl;
	cout << "    reload-config  Reload config file" << endl;
	cout << "    show-config    Show current config" << endl;
	cout << "    sparql         Run SPARQL queries" << endl;
	cout << "        -f <file>   a single query from the <file>" << endl;
	cout << "        -n <num>    run a single query <num> times" << endl;
	cout << "        -b <file>   a set of queries configured by the <file>" << endl;
	cout << "        -s <string> a single query from input string" << endl;
}

// the master proxy is the 1st proxy of the 1st server (i.e., sid == 0 and tid == 0)
#define IS_MASTER(_p) ((_p)->sid == 0 && (_p)->tid == 0)
#define PRINT_ID(_p) (cout << "[" << (_p)->sid << "-" << (_p)->tid << "]$ ")

/**
 * The Wukong's console is co-located with the main proxy (the 1st proxy thread on the 1st server)
 * and provide a simple interactive cmdline to tester
 */
void run_console(Proxy *proxy)
{
	console_barrier(proxy->tid);
	if (IS_MASTER(proxy))
		cout << endl
		     << "Input \'help\'' command to get more information"
		     << endl
		     << endl;

	while (true) {
		console_barrier(proxy->tid);
next:
		string cmd;
		if (IS_MASTER(proxy)) {
			cout << "> ";
			std::getline(std::cin, cmd);

			// trim input
			size_t pos = cmd.find_first_not_of(" \t"); // trim blanks from head
			if (pos == string::npos) goto next;
			cmd.erase(0, pos);

			pos = cmd.find_last_not_of(" \t");  // trim blanks from tail
			cmd.erase(pos + 1, cmd.length() - (pos + 1));

			// only process on the master console
			if (cmd == "help") {
				print_help();
				goto next;
			} else if (cmd == "show-config") {
				show_config();
				goto next;
			}

			// send commands to all proxy threads
			for (int i = 0; i < global_num_servers; i++) {
				for (int j = 0; j < global_num_proxies; j++) {
					if (i == 0 && j == 0)
						continue ;
					proxy->adaptor->send_object<string>(i, j, cmd);
				}
			}
		} else {
			// recieve commands
			cmd = proxy->adaptor->recv_object<string>();
		}

		// process on all consoles
		if (cmd == "quit" || cmd == "q") {
			if (proxy->tid == 0)
				exit(0); // each server exits once by the 1st proxy thread
		} else if (cmd == "reload-config") {
			if (proxy->tid == 0)
				reload_config(); // each server reload config file once by the 1st proxy
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
						if (IS_MASTER(proxy)) {
							cout << "Unknown option: " << token << endl;
							print_help();
						}
						goto next;
					}
				}

				if (f_enable) {
					// use the main proxy thread to run a single query
					if (IS_MASTER(proxy)) {
						ifstream ifs(fname);
						if (!ifs) {
							cout << "Query file not found: " << fname << endl;
							continue ;
						}
						Logger logger;
						proxy->run_single_query(ifs, cnt, logger);
						logger.print_latency(cnt);
					}
				}

				if (b_enable) {
					Logger logger;

					// dedicate the master frontend worker to run a single query
					// and others to run a set of queries if '-f' is enabled
					if (!f_enable || !IS_MASTER(proxy)) {
						ifstream ifs(bfname);
						if (!ifs) {
							PRINT_ID(proxy);
							cout << "Configure file not found: " << bfname << endl;
							continue ;
						}
						proxy->nonblocking_run_batch_query(ifs, logger);
						//proxy->run_batch_query(ifs, logger);
					}

					console_barrier(proxy->tid);

					// print a statistic of runtime for the batch processing on all servers
					if (IS_MASTER(proxy)) {
						for (int i = 0; i < global_num_servers * global_num_proxies - 1; i++) {
							Logger other = proxy->adaptor->recv_object<Logger>();
							logger.merge(other);
						}
						logger.print_rdf();
						logger.print_thpt();
					} else {
						// send logs to the master proxy
						proxy->adaptor->send_object<Logger>(0, 0, logger);
					}
				}

				if (q_enable) {
					// TODO: SPARQL string
					if (IS_MASTER(proxy)) {
						// TODO
						cout << "Query: " << query << endl;
						cout << "The option '-s' is unsupported now!" << endl;
					}
				}
			} else {
				if (IS_MASTER(proxy)) {
					cout << "Unknown command: " << token << endl;
					print_help();
				}
			}
		}
	}
}