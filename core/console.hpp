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

#include <iostream>
#include <string>
#include <set>
#include <boost/unordered_map.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "config.hpp"
#include "proxy.hpp"
#include "logger.hpp"
#include "timer.hpp"
using namespace std;

// communicate between proxy threads
TCP_Adaptor *con_adaptor;

bool enable_command = false;
string command = "";

template<typename T>
static void console_send(int sid, int tid, T &r) {
	std::stringstream ss;
	boost::archive::binary_oarchive oa(ss);
	oa << r;
	con_adaptor->send(sid, tid, ss.str());
}

template<typename T>
static T console_recv(int tid) {
	std::string str;
	str = con_adaptor->recv(tid);

	std::stringstream ss;
	ss << str;

	boost::archive::binary_iarchive ia(ss);
	T r;
	ia >> r;
	return r;
}

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
	cout << "    help                display help infomation" << endl;
	cout << "    quit                quit from console" << endl;
	cout << "    config <args>       run commands on config" << endl;
	cout << "        -v                  print current config" << endl;
	cout << "        -l <file>           load config items from <file>" << endl;
	cout << "        -s <string>         set config items by <str> (format: item1=val1&item2=...)" << endl;
	cout << "    sparql <args>       run SPARQL queries" << endl;
	cout << "        -f <file> [<args>]  a single query from <file>" << endl;
	cout << "           -n <num>            run <num> times" << endl;
	cout << "           -v <num>            print at most <num> lines of results" << endl;
	cout << "           -o <file>           output results into <file>" << endl;
	cout << "        -b <file>           a set of queries configured by <file> (batch-mode)" << endl;
}

// the master proxy is the 1st proxy of the 1st server (i.e., sid == 0 and tid == 0)
#define IS_MASTER(_p) ((_p)->sid == 0 && (_p)->tid == 0)
#define PRINT_ID(_p) (cout << "[" << (_p)->sid << "-" << (_p)->tid << "]$ ")

static void file2str(string fname, string &str)
{
	ifstream file(fname.c_str());
	if (!file) {
		cout << "ERROR: " << fname << " does not exist." << endl;
		return;
	}

	string line;
	while (std::getline(file, line))
		str += line + " ";
}

static void args2str(string &str)
{
	size_t found = str.find_first_of("=&");
	while (found != string::npos) {
		str[found] = ' ';
		found = str.find_first_of("=&", found + 1);
	}
}

/**
 * The Wukong's console is co-located with the main proxy (the 1st proxy thread on the 1st server)
 * and provide a simple interactive cmdline to tester
 */
void run_console(Proxy *proxy)
{
	console_barrier(proxy->tid);
	if (IS_MASTER(proxy))
		cout << endl
		     << "Input \'help\' command to get more information"
		     << endl
		     << endl;

	string cmd = "";
	while (true) {
		console_barrier(proxy->tid);
next:
		if (IS_MASTER(proxy)) {
            // direct-run command
			if (enable_command) {
                // if it had run the command then excute quit
                if (cmd == command) {
                   cmd = "quit";
                } else {
                    cmd = command;
                }
            } else {
                cout << "wukong> ";
			    std::getline(std::cin, cmd);
            }
			
            // trim input
			size_t pos = cmd.find_first_not_of(" \t"); // trim blanks from head
			if (pos == string::npos) goto next;
			cmd.erase(0, pos);

			pos = cmd.find_last_not_of(" \t");  // trim blanks from tail
			cmd.erase(pos + 1, cmd.length() - (pos + 1));

			// only run <cmd> on the master console
			if (cmd == "help") {
				print_help();
				goto next;
			}

			// send <cmd> to all consoles
			for (int i = 0; i < global_num_servers; i++) {
				for (int j = 0; j < global_num_proxies; j++) {
					if (i == 0 && j == 0) continue ;
					console_send<string>(i, j, cmd);
				}
			}
		} else {
			// recieve <cmd>
			cmd = console_recv<string>(proxy->tid);
		}

		// run <cmd> on all consoles
		if (cmd == "quit" || cmd == "q") {
			if (proxy->tid == 0)
				exit(0); // each server exits once by the 1st console
		} else {
			std::stringstream cmd_ss(cmd);
			std::string token;

			// get keyword of command
			cmd_ss >> token;
			if (token == "config") {
				if (proxy->tid != 0) continue;

				string fname, str;
				bool v_enable = false, l_enable = false, s_enable = false;

				// parse parameters
				while (cmd_ss >> token) {
					if (token == "-v") {
						v_enable = true;
					} else if (token == "-l") {
						l_enable = true;
						if (!(cmd_ss >> fname)) goto failed;
					} else if (token == "-s") {
						s_enable = true;
						if (!(cmd_ss >> str)) goto failed;
					} else {
						goto failed;
					}
				}

				if (v_enable) { // -v
					if (IS_MASTER(proxy))
						print_config();
				} else if (l_enable || s_enable) { // -l <file> or -s <str>
					if (IS_MASTER(proxy)) {
						if (l_enable) // -l
							file2str(fname, str);
						else if (s_enable) // -s
							args2str(str);

						// send <str> to all consoles
						for (int i = 1; i < global_num_servers; i++)
							console_send<string>(i, 0, str);
					} else {
						// recieve <str>
						str = console_recv<string>(proxy->tid);
					}

					if (!str.empty()) {
						reload_config(str);
					} else {
						if (IS_MASTER(proxy))
							cout << "Failed to load config file: " << fname << endl;
					}
				} else {
					goto failed;
				}
			} else if (token == "sparql") { // handle SPARQL queries
				string fname, bfname, ofname;
				int cnt = 1, nlines = 0;
				bool f_enable = false, b_enable = false, o_enable = false;

				// parse parameters
				while (cmd_ss >> token) {
					if (token == "-f") {
						cmd_ss >> fname;
						f_enable = true;
					} else if (token == "-n") {
						cmd_ss >> cnt;
					} else if (token == "-v") {
						cmd_ss >> nlines;
					} else if (token == "-o") {
						cmd_ss >> ofname;
						o_enable = true;
					} else if (token == "-b") {
						cmd_ss >> bfname;
						b_enable = true;
					} else {
						goto failed;
					}
				}

				if (!f_enable && !b_enable) goto failed; // meaningless args for SPARQL queries

				if (f_enable) { // -f <file>
					// use the main proxy thread to run a single query
					if (IS_MASTER(proxy)) {
						ifstream ifs(fname);
						if (!ifs.good()) {
							cout << "Query file not found: " << fname << endl;
							continue ;
						}

						if (global_silent) {
							if (nlines > 0) {
								cout << "Can't print results (-v) with global_silent." << endl;
								continue;
							}

							if (o_enable) {
								cout << "Can't output results (-o) with global_silent." << endl;
								continue;
							}
						}

						request_or_reply reply;
						Logger logger;
						int ret = proxy->run_single_query(ifs, cnt, reply, logger);
						if (ret != 0) {
							cout << "Failed to run the query (ERROR: " << ret << ")!" << endl;
							continue;
						}

						// print or dump results
						logger.print_latency(cnt);
						cout << "(last) result size: " << reply.row_num << endl;
						if (!global_silent && !reply.blind) {
							if (global_load_minimal_index)
								cout << "WARNING: Can't print/output results in string format\n"
								     << "         with global_load_minimal_index enabled." << endl;

							if (nlines > 0)
								proxy->print_result(reply, min(reply.row_num, nlines));

							if (o_enable)
								proxy->dump_result(reply, ofname);
						}
					}
				}

				if (b_enable) { // -b <config>
					Logger logger;

					// dedicate the master frontend worker to run a single query
					// and others to run a set of queries if '-f' is enabled
					if (!f_enable || !IS_MASTER(proxy)) {
						// Currently, batch-mode is not supported by our SPARQL parser and planner
						// since queries in batch-mode use non-standard SPARQL grammer.
						if (global_enable_planner) {
							cout << "Can't run queries in batch mode with global_enable_planner." << endl;
							continue;
						}

						ifstream ifs(bfname);
						if (!ifs.good()) {
							PRINT_ID(proxy);
							cout << "Configure file not found: " << bfname << endl;
							continue;
						}

						proxy->nonblocking_run_batch_query(ifs, logger);
						//proxy->run_batch_query(ifs, logger);
					}

					// FIXME: maybe hang in here if the input file misses in some machines
					//        or inconsistent global variables (global_enable_planner)
					console_barrier(proxy->tid);

					// print a statistic of runtime for the batch processing on all servers
					if (IS_MASTER(proxy)) {
						for (int i = 0; i < global_num_servers * global_num_proxies - 1; i++) {
							Logger other = console_recv<Logger>(proxy->tid);
							logger.merge(other);
						}
						logger.print_rdf();
						logger.print_thpt();
					} else {
						// send logs to the master proxy
						console_send<Logger>(0, 0, logger);
					}
				}
			} else {
failed:
				if (IS_MASTER(proxy)) {
					cout << "Failed to run the command: " << cmd << endl;
					print_help();
				}
			}
		}
	}
}
