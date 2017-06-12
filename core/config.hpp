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

#include <assert.h>
#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

using namespace std;

int global_num_servers = 1;    // the number of servers
int global_num_threads = 2;    // the number of threads per server (incl. proxy and engine)

int global_num_engines = 1;    // the number of engines
int global_num_proxies = 1;    // the number of proxies

string global_input_folder;
bool global_load_minimal_index = true;

int global_data_port_base = 5500;
int global_ctrl_port_base = 9576;

int global_memstore_size_gb = 20;
int global_rdma_buf_size_mb = 128;
int global_rdma_rbf_size_mb = 32;
int global_num_keys_million = 1000;

bool global_use_rdma = true;
bool global_enable_caching = true;
int global_enable_workstealing = false;

int global_mt_threshold = 16;
int global_rdma_threshold = 300;

int global_max_print_row = 10;
bool global_silent = true;

// for planner
bool global_enable_planner = true;

/* set by command line */
string cfg_fname;
string host_fname;

/**
 * list current global setting
 */
void show_config(void)
{
	cout << "------ global configurations ------" << endl;

	// setting by config file
	cout << "the number of engines: "		<< global_num_engines 			<< endl;
	cout << "the number of proxies: "		<< global_num_proxies			<< endl;
	cout << "global_input_folder: " 		<< global_input_folder			<< endl;
	cout << "global_load_minimal_index: " 	<< global_load_minimal_index 	<< endl;
	cout << "global_data_port_base: " 		<< global_data_port_base		<< endl;
	cout << "global_ctrl_port_base: " 		<< global_ctrl_port_base		<< endl;
	cout << "global_memstore_size_gb: " 	<< global_memstore_size_gb		<< endl;
	cout << "global_rdma_rbf_size_mb: " 	<< global_rdma_rbf_size_mb   	<< endl;
	cout << "global_rdma_buf_size_mb: " 	<< global_rdma_buf_size_mb		<< endl;
	cout << "global_num_keys_million: " 	<< global_num_keys_million		<< endl;
	cout << "global_use_rdma: " 			<< global_use_rdma				<< endl;
	cout << "global_enable_caching: " 		<< global_enable_caching		<< endl;
	cout << "global_enable_workstealing: " 	<< global_enable_workstealing	<< endl;
	cout << "global_rdma_threshold: " 		<< global_rdma_threshold		<< endl;
	cout << "global_mt_threshold: " 		<< global_mt_threshold  		<< endl;
	cout << "global_max_print_row: " 		<< global_max_print_row			<< endl;
	cout << "global_silent: " 				<< global_silent				<< endl;
	cout << "global_enable_planner: " 		<< global_enable_planner 		<< endl;

	cout << "--" << endl;

	// compute from other settings
	cout << "the number of servers: " 		<< global_num_servers			<< endl;
	cout << "the number of threads: " 		<< global_num_threads			<< endl;
}

/**
 * re-configure Wukong
 */
void reload_config(void)
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

	map<string, string> configs;
	string line, row, val;
	while (std::getline(file, line)) {
		istringstream iss(line);
		iss >> row >> val;
		configs[row] = val;
	}

	for (auto const &entry : configs) {
		if (entry.first == "global_enable_caching")
			global_enable_caching = atoi(entry.second.c_str());
		else if (entry.first == "global_enable_workstealing")
			global_enable_workstealing = atoi(entry.second.c_str());
		else if (entry.first == "global_rdma_threshold")
			global_rdma_threshold = atoi(entry.second.c_str());
		else if (entry.first == "global_mt_threshold")
			global_mt_threshold = atoi(entry.second.c_str());
		else if (entry.first == "global_max_print_row")
			global_max_print_row = atoi(entry.second.c_str());
		else if (entry.first == "global_silent")
			global_silent = atoi(entry.second.c_str());
		else if (entry.first == "global_enable_planner")
			global_enable_planner = atoi(entry.second.c_str());
		else
			cout << "WARNING: unsupported re-configuration item! ("
			     << entry.first << ")" << endl;
	}
	assert(global_mt_threshold > 0);

	// limited the number of engines
	global_mt_threshold = max(1, min(global_mt_threshold, global_num_engines));
	return;
}

void load_config(int num_servers)
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
	map<string, string> configs;
	while (std::getline(file, line)) {
		istringstream iss(line);
		iss >> row >> val;
		configs[row] = val;
	}

	for (auto const &entry : configs) {
		if (entry.first == "global_num_engines")
			global_num_engines = atoi(entry.second.c_str());
		else if (entry.first == "global_num_proxies")
			global_num_proxies = atoi(entry.second.c_str());
		else if (entry.first == "global_input_folder")
			global_input_folder = entry.second;
		else if (entry.first == "global_load_minimal_index")
			global_load_minimal_index = atoi(entry.second.c_str());
		else if (entry.first == "global_data_port_base")
			global_data_port_base = atoi(entry.second.c_str());
		else if (entry.first == "global_ctrl_port_base")
			global_ctrl_port_base = atoi(entry.second.c_str());
		else if (entry.first == "global_memstore_size_gb")
			global_memstore_size_gb = atoi(entry.second.c_str());
		else if (entry.first == "global_rdma_rbf_size_mb")
			global_rdma_rbf_size_mb = atoi(entry.second.c_str());
		else if (entry.first == "global_rdma_buf_size_mb")
			global_rdma_buf_size_mb = atoi(entry.second.c_str());
		else if (entry.first == "global_num_keys_million")
			global_num_keys_million = atoi(entry.second.c_str());
		else if (entry.first == "global_use_rdma")
			global_use_rdma = atoi(entry.second.c_str());
		else if (entry.first == "global_enable_caching")
			global_enable_caching = atoi(entry.second.c_str());
		else if (entry.first == "global_enable_workstealing")
			global_enable_workstealing = atoi(entry.second.c_str());
		else if (entry.first == "global_rdma_threshold")
			global_rdma_threshold = atoi(entry.second.c_str());
		else if (entry.first == "global_mt_threshold")
			global_mt_threshold = atoi(entry.second.c_str());
		else if (entry.first == "global_max_print_row")
			global_max_print_row = atoi(entry.second.c_str());
		else if (entry.first == "global_silent")
			global_silent = atoi(entry.second.c_str());
		else if (entry.first == "global_enable_planner")
			global_enable_planner = atoi(entry.second.c_str());
		else
			cout << "WARNING: unsupported configuration item! ("
			     << entry.first << ")" << endl;
	}

	global_num_servers = num_servers;
	global_num_threads = global_num_engines + global_num_proxies;

	assert(num_servers > 0);
	assert(global_num_engines > 0);
	assert(global_num_proxies > 0);
	assert(global_memstore_size_gb > 0);
	assert(global_rdma_rbf_size_mb > 0);
	assert(global_rdma_buf_size_mb > 0);
	assert(global_mt_threshold > 0);

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

	return;
}
