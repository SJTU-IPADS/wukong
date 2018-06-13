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

using namespace std;

// communicate between proxy threads
TCP_Adaptor *con_adaptor;

bool enable_oneshot = false;
string oneshot_cmd = "";

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
	cout << "    sparql <args>       run a single SPARQL query" << endl;
	cout << "        -f <file> [<args>]  run a single query from <file>" << endl;
	cout << "           -m <factor>         set multi-threading factor <factor> for heavy queries" << endl;
	cout << "           -n <num>            run <num> times" << endl;
	cout << "           -v <num>            print at most <num> lines of results" << endl;
	cout << "           -o <file>           output results into <file>" << endl;
	cout << "        -b <file>           run a batch of queries configured by <file>" << endl;
	cout << "    sparql-emu <args>   emulate clients to continuously send SPARQL queries" << endl;
	cout << "        -f <file> [<args>]  run queries generated from temples configured by <file>" << endl;
	cout << "           -d <sec>            eval <sec> seconds (default: 10)" << endl;
	cout << "           -w <sec>            warmup <sec> seconds (default: 5)" << endl;
	cout << "           -p <num>            send <num> queries in parallel (default: 20)" << endl;
	cout << "    load-stat           load statistics from a file" << endl;
	cout << "        -f <file>           load statistics from <file> located at data folder" << endl;
	cout << "    store-stat           store statistics to a file" << endl;
	cout << "        -f <file>           store statistics to <file> located at data folder" << endl;
	cout << "    load <args>         load linked data into dynamic (in-memmory) graph store" << endl;
	cout << "        -d <dname>          load data from directory <dname>" << endl;
	cout << "    gsck <args>         check the graph storage integrity" << endl;
	cout << "        -i                  check from index key/value pair to normal key/value pair" << endl;
	cout << "        -n                  check from normal key/value pair to index key/value pair" << endl;
	cout << "        -a                  check all the above" << endl;
}

// the master proxy is the 1st proxy of the 1st server (i.e., sid == 0 and tid == 0)
#define IS_MASTER(_p) ((_p)->sid == 0 && (_p)->tid == 0)
#define PRINT_ID(_p) (cout << "[" << (_p)->sid << "-" << (_p)->tid << "]$ ")

static void file2str(string fname, string &str)
{
	ifstream file(fname.c_str());
	if (!file) {
		logstream(LOG_ERROR) << fname << " does not exist." << LOG_endl;
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


// usage: sparql -f <file> [<args>]
// args_ss:
//   -n <num>
//   -v <num>
//   -o <fname>
bool run_sparql_cmd(Proxy *proxy, std::stringstream &args_ss, string &fname)
{
	std::string token;
	string ofname;
	int cnt = 1, nlines = 0, mt_factor = 1;
	bool o_enable = false;

	// parse parameters
	while (args_ss >> token) {
		if (token == "-m") {
			args_ss >> mt_factor;
		} else if (token == "-n") {
			args_ss >> cnt;
		} else if (token == "-v") {
			args_ss >> nlines;
		} else if (token == "-o") {
			args_ss >> ofname;
			o_enable = true;
		} else {
			return false;
		}
	}

	// read a SPARQL query from a file
	ifstream ifs(fname);
	if (!ifs.good()) {
		logstream(LOG_ERROR) << "Query file not found: " << fname << LOG_endl;
		return false;
	}

	if (global_silent) { // not retrieve the query results
		if (nlines > 0) {
			logstream(LOG_ERROR) << "Can't print results (-v) with global_silent." << LOG_endl;
			return false;
		}

		if (o_enable) {
			logstream(LOG_ERROR) << "Can't output results (-o) with global_silent." << LOG_endl;
			return false;
		}
	}

	SPARQLQuery reply;
	SPARQLQuery::Result &result = reply.result;
	Logger logger;
	int ret = proxy->run_single_query(ifs, mt_factor, cnt, reply, logger);
	if (ret != 0) {
		logstream(LOG_ERROR) << "Failed to run the query (ERRNO: " << ret << ")!" << LOG_endl;
		return false;
	}

	logger.print_latency(cnt);
	logstream(LOG_INFO) << "(last) result size: " << result.row_num << LOG_endl;

	// print or dump results
	if (!global_silent && !result.blind && (nlines > 0 || o_enable)) {
		if (nlines > 0)
			result.print_result(min(result.get_row_num(), nlines), proxy->str_server);

		if (o_enable)
			result.dump_result(ofname, result.get_row_num(), proxy->str_server);
	}

	return true;
}


// usage: sparql-emu -f <file> [<args>]
// args_ss:
//   -d <sec>
//   -w <sec>
//   -p <num>
bool run_sparql_emu(Proxy *proxy, std::stringstream &args_ss, string &fname)
{
	std::string token;
	int duration = 10, warmup = 5, parallel_factor = 20;

	// parse parameters
	while (args_ss >> token) {
		if (token == "-d") {
			args_ss >> duration;
		} else if (token == "-w") {
			args_ss >> warmup;
		} else if (token == "-p") {
			args_ss >> parallel_factor;
		} else {
			return false;
		}
	}

	// read config file of a SPARQL emulator
	ifstream ifs(fname);
	if (!ifs.good()) {
		logstream(LOG_ERROR) << "Configure file not found: " << fname << LOG_endl;
		return false;
	}

	if (duration <= 0 || warmup < 0 || parallel_factor <= 0) {
		logstream(LOG_ERROR) << "invalid parameters for SPARQL emulator! "
		                     << "(duration=" << duration << ", warmup=" << warmup
		                     << ", parallel_factor=" << parallel_factor << ")" << LOG_endl;
		return false;
	}

	if (duration <= warmup) {
		logstream(LOG_INFO) << "Duration time (" << duration
		                    << "sec) is less than warmup time ("
		                    << warmup << "sec)." << LOG_endl;
		return false;
	}

	Logger logger;
	proxy->run_query_emu(ifs, duration, warmup, parallel_factor, logger);

	// FIXME: maybe hang in here if the input file misses in some machines
	//        or inconsistent global variables (e.g., global_enable_planner)
	console_barrier(proxy->tid);

	// print a performance statistic for running emulator on all servers
	if (IS_MASTER(proxy)) {
		for (int i = 0; i < global_num_servers * global_num_proxies - 1; i++) {
			Logger other = console_recv<Logger>(proxy->tid);
			logger.merge(other);
		}
		logger.aggregate();
		logger.print_cdf();
		logger.print_thpt();
	} else {
		// send logs to the master proxy
		console_send<Logger>(0, 0, logger);
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

	bool once = true;
	while (true) {
		console_barrier(proxy->tid);
next:
		string cmd;
		if (IS_MASTER(proxy)) {
			if (enable_oneshot) {
				// one-shot command mode: run the command once
				if (once) {
					logstream(LOG_INFO) << "Run one-shot command: " << oneshot_cmd << LOG_endl;
					cmd = oneshot_cmd;

					once = false;
				} else {
					logstream(LOG_INFO) << "Done" << LOG_endl;
					cmd = "quit";
				}
			} else {
				// interactive mode: print a prompt and retrieve the command
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
							logstream(LOG_ERROR) << "Failed to load config file: " << fname << LOG_endl;
					}
				} else {
					goto failed;
				}
			} else if (token == "sparql") { // handle SPARQL queries
				// use the main proxy thread to run a single query
				if (!IS_MASTER(proxy)) continue;

				// parse command
				string fname;
				bool f_enable = false, b_enable = false;
				while (cmd_ss >> token) {
					if (token == "-f") {
						cmd_ss >> fname;
						f_enable = true;
						break;
					} else if (token == "-b") {
						cmd_ss >> fname;
						b_enable = true;
						break;
					} else {
						goto failed;
					}
				}

				if (!(f_enable ^ b_enable)) goto failed; // invalid cmd

				// [single mode]
				//  usage: sparql -f <file> [<args>]
				//  args:
				//    -n <num>
				//    -v <num>
				//    -o <fname>
				if (f_enable) {
					if (!run_sparql_cmd(proxy, cmd_ss, fname))
						goto failed;
				}

				// [batch mode]
				//  usage: sparql -b <fname>
				//  file-format:
				//    sparql -f <fname> [<args>]
				//    sparql -f <fname> [<args>]
				//    ...
				if (b_enable) {
					logstream(LOG_ERROR) << "Unsupported command now!" << LOG_endl;
				}
			} else if (token == "sparql-emu") { // run a SPARQL emulator on each proxy

				// parse command
				string fname;
				bool f_enable = false;
				while (cmd_ss >> token) {
					if (token == "-f") {
						cmd_ss >> fname;
						f_enable = true;
						break;
					} else {
						goto failed;
					}
				}

				if (!f_enable) goto failed; // invalid cmd

				// [emulator]
				//  usage: sparql-emu -f <file> [<args>]
				//  args:
				//    -d <sec>
				//    -w <sec>
				//    -p <num>
				if (!run_sparql_emu(proxy, cmd_ss, fname))
					goto failed;
			} else if (token == "load") {
				// use the main proxy thread to run gstore checker
				if (!IS_MASTER(proxy)) continue;

#if DYNAMIC_GSTORE
				string dname;
				bool d_enable = false;
				bool c_enable = false;

				while (cmd_ss >> token) {
					if (token == "-c") {
						c_enable = true;
					} else if (token == "-d") {
						cmd_ss >> dname;
						d_enable = true;
					} else {
						goto failed;
					}
				}

				if (!d_enable) goto failed;  // invalid cmd

				// force a "/" at the end of dname.
				if (dname[dname.length() - 1] != '/')
					dname = dname + "/";

				Logger logger;
				RDFLoad reply;
				int ret = proxy->dynamic_load_data(dname, reply, logger, c_enable);
				if (ret != 0) {
					logstream(LOG_ERROR) << "Failed to load dynamic data from directory " << dname
					                     << " (ERRNO: " << ret << ")!" << LOG_endl;
					continue;
				}
				logger.print_latency();
#else
				logstream(LOG_ERROR) << "Can't load linked data into static graph-store." << LOG_endl;
				logstream(LOG_ERROR) << "You can enable it by building Wukong with -DUSE_DYNAMIC_GSTORE=ON." << LOG_endl;
#endif
			} else if (token == "load-stat") {
				// use the main proxy thread to load statistics
				if (!IS_MASTER(proxy)) continue;

				// parse command
				string fname;
				bool f_enable = false;
				while (cmd_ss >> token) {
					if (token == "-f") {
						cmd_ss >> fname;
						f_enable = true;
					} else {
						goto failed;
					}
				}

				// if fname is not given, try the dataset name by default
				if (!f_enable) {
					vector<string> strs;
					boost::split(strs, global_input_folder, boost::is_any_of("/"));
					fname = strs[strs.size() - 2] + ".statfile";
				}

				proxy->statistic->load_stat_from_file(fname);
			} else if (token == "store-stat") {
				// use the main proxy thread to save statistics
				if (!IS_MASTER(proxy)) continue;

				// parse command
				string fname;
				bool f_enable = false;
				while (cmd_ss >> token) {
					if (token == "-f") {
						cmd_ss >> fname;
						f_enable = true;
					} else {
						goto failed;
					}
				}

				// if fname is not given, use the dataset name by default
				if (!f_enable) {
					vector<string> strs;
					boost::split(strs, global_input_folder, boost::is_any_of("/"));
					fname = strs[strs.size() - 2] + ".statfile";
				}

				proxy->statistic->store_stat_to_file(fname);
			} else if (token == "gsck") {
				// use the main proxy thread to run gstore checker
				if (!IS_MASTER(proxy)) continue;

				bool i_enable = false;
				bool n_enable = false;

				while (cmd_ss >> token) {
					if (token == "-i") {
						i_enable = true;
					} else if (token == "-n") {
						n_enable = true;
					} else if (token == "-a") {
						i_enable = true;
						n_enable = true;
					} else {
						goto failed;
					}
				}

				Logger logger;
				GStoreCheck reply;
				int ret = proxy->gstore_check(reply, logger, i_enable, n_enable);
				if (ret != 0) {
					logstream(LOG_ERROR) << "Some error found in gstore "
					                     << " (ERRNO: " << ret << ")!" << LOG_endl;
					continue;
				}
				logger.print_latency();
			} else {
failed:
				if (IS_MASTER(proxy)) {
					logstream(LOG_ERROR) << "Failed to run the command: " << cmd << LOG_endl;
					print_help();
				}
			}
		}
	}
}