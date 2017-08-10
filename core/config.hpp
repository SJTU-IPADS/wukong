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

#include "rdma.hpp"

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
int global_rdma_buf_size_mb = 64;
int global_rdma_rbf_size_mb = 16;
//int global_num_keys_million = 1000;

bool global_use_rdma = true;
bool global_enable_caching = true;
int global_enable_workstealing = false;

int global_mt_threshold = 16;
int global_rdma_threshold = 300;

bool global_silent = true;  // don't take back results by default

// for planner
bool global_enable_planner = true;


static bool set_immutable_config(string cfg_name, string value)
{

	if (cfg_name == "global_num_engines") {
		global_num_engines = atoi(value.c_str());
		assert(global_num_engines > 0);
	} else if (cfg_name == "global_num_proxies") {
		global_num_proxies = atoi(value.c_str());
		assert(global_num_proxies > 0);
	}
	else if (cfg_name == "global_input_folder") {
		global_input_folder = value;

		// make sure to check that the global_input_folder is non-empty.
		if (global_input_folder.length() == 0) {
			cout << "ERROR: the directory path of RDF data can not be empty!"
			     << "You should set \"global_input_folder\" in config file." << endl;
			exit(-1);
		}

		// force a "/" at the end of global_input_folder.
		if (global_input_folder[global_input_folder.length() - 1] != '/')
			global_input_folder = global_input_folder + "/";
	} else if (cfg_name == "global_load_minimal_index") {
		global_load_minimal_index = atoi(value.c_str());
	} else if (cfg_name == "global_data_port_base") {
		global_data_port_base = atoi(value.c_str());
		assert(global_data_port_base > 0);
	} else if (cfg_name == "global_ctrl_port_base") {
		global_ctrl_port_base = atoi(value.c_str());
		assert(global_ctrl_port_base > 0);
	} else if (cfg_name == "global_memstore_size_gb") {
		global_memstore_size_gb = atoi(value.c_str());
		assert(global_memstore_size_gb > 0);
	} else if (cfg_name == "global_rdma_buf_size_mb") {
		global_rdma_buf_size_mb = atoi(value.c_str());
		assert(global_rdma_buf_size_mb > 0);
	} else if (cfg_name == "global_rdma_rbf_size_mb") {
		global_rdma_rbf_size_mb = atoi(value.c_str());
		assert(global_rdma_rbf_size_mb > 0);
	} else {
		return false;
	}

	return true;
}

static bool set_mutable_config(string cfg_name, string value)
{
	if (cfg_name == "global_use_rdma") {
		if (atoi(value.c_str())) {
			if (!RDMA::get_rdma().has_rdma()) {
				cout << "ERROR: can't enable RDMA due to building Wukong w/o RDMA support!\n"
				     << "HINT: please disable global_use_rdma in config file." << endl;
				global_use_rdma = false; // disable RDMA if no RDMA device
				return true;
			}

			global_use_rdma = true;
		} else {
			global_use_rdma = false;
		}
	} else if (cfg_name == "global_rdma_threshold") {
		global_rdma_threshold = atoi(value.c_str());
	} else if (cfg_name == "global_mt_threshold") {
		global_mt_threshold = atoi(value.c_str());
		assert(global_mt_threshold > 0);
	} else if (cfg_name == "global_enable_caching") {
		global_enable_caching = atoi(value.c_str());
	} else if (cfg_name == "global_enable_workstealing") {
		global_enable_workstealing = atoi(value.c_str());
	} else if (cfg_name == "global_silent") {
		global_silent = atoi(value.c_str());
	} else if (cfg_name == "global_enable_planner") {
		global_enable_planner = atoi(value.c_str());
	} else {
		return false;
	}

	return true;
}

static void str2items(string str, map<string, string> &items)
{
	istringstream iss(str);
	string row, val;
	while (iss >> row >> val)
		items[row] = val;
}

static void file2items(string fname, map<string, string> &items)
{
	ifstream file(fname.c_str());
	if (!file) {
		cout << "ERROR: " << fname << " does not exist." << endl;
		exit(0);
	}

	string line, row, val;
	while (std::getline(file, line)) {
		istringstream iss(line);
		iss >> row >> val;
		items[row] = val;
	}
}

/**
 * reload config
 */
void reload_config(string str)
{
	// TODO: it should ensure that there is no outstanding queries.

	// load config file
	map<string, string> items;
	str2items(str, items);

	for (auto const &entry : items) {
		set_mutable_config(entry.first, entry.second);
	}

	// limited the number of engines
	global_mt_threshold = max(1, min(global_mt_threshold, global_num_engines));

	return;
}

/**
 * load config
 */
void load_config(string fname, int num_servers)
{
	global_num_servers = num_servers;
	assert(num_servers > 0);

	// load config file
	map<string, string> items;
	file2items(fname, items);

	for (auto const &entry : items) {
		if (!(set_immutable_config(entry.first, entry.second)
		        || set_mutable_config(entry.first, entry.second))) {
			cout << "WARNING: unsupported configuration item! ("
			     << entry.first << ")" << endl;
		}
	}

	// set the total number of threads
	global_num_threads = global_num_engines + global_num_proxies;

	// limited the number of engines
	global_mt_threshold = max(1, min(global_mt_threshold, global_num_engines));

	return;
}

/**
 * print current config
 */
void print_config(void)
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
	cout << "global_rdma_buf_size_mb: " 	<< global_rdma_buf_size_mb		<< endl;
	cout << "global_rdma_rbf_size_mb: " 	<< global_rdma_rbf_size_mb   	<< endl;
	cout << "global_use_rdma: " 			<< global_use_rdma				<< endl;
	cout << "global_enable_caching: " 		<< global_enable_caching		<< endl;
	cout << "global_enable_workstealing: " 	<< global_enable_workstealing	<< endl;
	cout << "global_rdma_threshold: " 		<< global_rdma_threshold		<< endl;
	cout << "global_mt_threshold: " 		<< global_mt_threshold  		<< endl;
	cout << "global_silent: " 				<< global_silent				<< endl;
	cout << "global_enable_planner: " 		<< global_enable_planner 		<< endl;

	cout << "--" << endl;

	// compute from other settings
	cout << "the number of servers: " 		<< global_num_servers			<< endl;
	cout << "the number of threads: " 		<< global_num_threads			<< endl;
}
