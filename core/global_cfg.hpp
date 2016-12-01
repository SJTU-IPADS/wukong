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

#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

using namespace std;

int global_nsrvs = 1;    // the number of servers
int global_nthrs = 2;    // the number of threads per server (incl. proxy and engine)

int global_num_engines = 1;    // the number of engines
int global_num_proxies = 1;    // the number of proxies

string global_input_folder;
bool global_load_minimal_index = true;

int global_eth_port_base = 5500;
int global_rdma_port_base = 9576;

int global_total_memory_gb = 20;
int global_perslot_msg_mb = 256;
int global_perslot_rdma_mb = 128;
int global_hash_header_million = 1000;

bool global_use_rdma = true;
bool global_use_rbf = true;		// ring-buffer (by RDMA WRITE)
bool global_use_loc_cache = false;
int global_enable_workstealing = false;

int global_mt_threshold = 16;
int global_rdma_threshold = 300;

int global_max_print_row = 10;
bool global_silent = true;

/* set by command line */
string cfg_fname;
string host_fname;

/**
 * dump current global setting
 */
void
dump_cfg(void)
{
	cout << "------ global configurations ------" << endl;

	// setting by config file
	cout << "global_eth_port_base: " 		<< global_eth_port_base				<< endl;
	cout << "global_rdma_port_base: " 		<< global_rdma_port_base			<< endl;
	cout << "global_use_rbf: " 				<< global_use_rbf 					<< endl;
	cout << "global_use_rdma: " 			<< global_use_rdma					<< endl;
	cout << "global_rdma_threshold: " 		<< global_rdma_threshold			<< endl;
	cout << "the number of engines: "		<< global_num_engines 				<< endl;
	cout << "the number of proxies: "		<< global_num_proxies				<< endl;
	cout << "global_mt_threshold: " 		<< global_mt_threshold  			<< endl;
	cout << "global_input_folder: " 		<< global_input_folder				<< endl;
	cout << "global_use_loc_cache: " 		<< global_use_loc_cache				<< endl;
	cout << "global_load_minimal_index: " 	<< global_load_minimal_index 		<< endl;
	cout << "global_silent: " 				<< global_silent					<< endl;
	cout << "global_max_print_row: " 		<< global_max_print_row				<< endl;
	cout << "global_total_memory_gb: " 		<< global_total_memory_gb			<< endl;
	cout << "global_perslot_msg_mb: " 		<< global_perslot_msg_mb   			<< endl;
	cout << "global_perslot_rdma_mb: " 		<< global_perslot_rdma_mb			<< endl;
	cout << "global_hash_header_million: " 	<< global_hash_header_million		<< endl;
	cout << "global_enable_workstealing: " 	<< global_enable_workstealing		<< endl;
	cout << "--" << endl;

	// compute from other cfg settings
	cout << "the number of servers: " 		<< global_nsrvs				<< endl;
	cout << "the number of threads: " 		<< global_nthrs				<< endl;
}

/**
 * re-configure Wukong
 */
void
reload_cfg(void)
{
	// TODO: it should ensure that there is no outstanding queries.

	ifstream file(cfg_fname.c_str());
	if (!file) {
		cout << "ERROR: the configure file "
		     << cfg_fname
		     << " does not exist."
		     << endl;
		exit(0);
	}

	map<string, string> config_map;
	string row;
	string val;
	// while(file>>row>>val){
	// 	config_map[row]=val;
	// }
	string line;
	while (std::getline(file, line)) {
		istringstream iss(line);
		iss >> row >> val;
		config_map[row] = val;
	}

	global_use_loc_cache = atoi(config_map["global_use_loc_cache"].c_str());
	global_enable_workstealing = atoi(config_map["global_enable_workstealing"].c_str());
	global_mt_threshold = atoi(config_map["global_mt_threshold"].c_str());
	global_rdma_threshold = atoi(config_map["global_rdma_threshold"].c_str());
	global_max_print_row = atoi(config_map["global_max_print_row"].c_str());
	global_silent = atoi(config_map["global_silent"].c_str());

	// limited the number of engines
	global_mt_threshold = max(1, min(global_mt_threshold, global_num_engines));
	return;
}

void
load_cfg(int nsrvs)
{
	ifstream file(cfg_fname.c_str());
	if (!file) {
		cout << "ERROR: the configure file "
		     << cfg_fname
		     << " does not exist."
		     << endl;
		exit(0);
	}

	string line, row, val;
	map<string, string> config_map;
	while (std::getline(file, line)) {
		istringstream iss(line);
		iss >> row >> val;
		config_map[row] = val;
	}

	global_num_engines = atoi(config_map["global_num_engines"].c_str());
	global_num_proxies = atoi(config_map["global_num_proxies"].c_str());
	global_input_folder = config_map["global_input_folder"];
	global_load_minimal_index = atoi(config_map["global_load_minimal_index"].c_str());
	global_eth_port_base = atoi(config_map["global_eth_port_base"].c_str());
	global_rdma_port_base = atoi(config_map["global_rdma_port_base"].c_str());

	global_total_memory_gb = atoi(config_map["global_total_memory_gb"].c_str());
	global_perslot_msg_mb = atoi(config_map["global_perslot_msg_mb"].c_str());
	global_perslot_rdma_mb = atoi(config_map["global_perslot_rdma_mb"].c_str());
	global_hash_header_million = atoi(config_map["global_hash_header_million"].c_str());

	global_use_rdma = atoi(config_map["global_use_rdma"].c_str());
	global_use_rbf = atoi(config_map["global_use_rbf"].c_str());
	global_use_loc_cache = atoi(config_map["global_use_loc_cache"].c_str());
	global_enable_workstealing = atoi(config_map["global_enable_workstealing"].c_str());

	global_rdma_threshold = atoi(config_map["global_rdma_threshold"].c_str());
	global_mt_threshold = atoi(config_map["global_mt_threshold"].c_str());
	global_max_print_row = atoi(config_map["global_max_print_row"].c_str());
	global_silent = atoi(config_map["global_silent"].c_str());

	global_nsrvs = nsrvs;
	global_nthrs = global_num_engines + global_num_proxies;

	// limited the number of engines
	global_mt_threshold = max(1, min(global_mt_threshold, global_num_engines));

	// make sure to check that the global_input_folder is non-empty.
	if (global_input_folder.length() == 0) {
		cout << "ERROR: the directory path of RDF data can not be empty!"
		     << "You should set \"global_input_folder\" in config file." << endl;
		exit(-1);
	}

	// force a "/" at the end of global_input_folder.
	if (global_input_folder[global_input_folder.length() - 1] != '/')
		global_input_folder = global_input_folder + "/";

	// debug dump
	if (!global_silent) dump_cfg();

	return;
}
