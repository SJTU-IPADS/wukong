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

#include "global_cfg.h"


/* non-configurable global variables */
int global_rdftype_id;	// reserved ID for rdf:type
int global_nsrvs;
int global_nthrs;


/* configurable global variables */
int global_eth_port_base;
int global_rdma_port_base;
bool global_use_rbf;
bool global_use_rdma;
int global_nbewkrs;
int global_nfewkrs;

std::string global_input_folder;
bool global_load_minimal_index;
int global_max_print_row;
int global_total_memory_gb;
int global_perslot_msg_mb;
int global_perslot_rdma_mb;
int global_hash_header_million;
int global_enable_workstealing;
int global_verbose;

/* shared by client and server */
int global_batch_factor;
bool global_use_loc_cache;
bool global_silent;
int global_mt_threshold;
int global_rdma_threshold;

/* TODO: split the config file and related code
		 into two parts: client and server */

std::string cfg_fname;
std::string host_fname;

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
	cout << "the number of backend workers: "	<< global_nbewkrs 				<< endl;
	cout << "the number of frontend workers: "	<< global_nfewkrs				<< endl;
	cout << "global_batch_factor: " 		<< global_batch_factor				<< endl;
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
	cout << "global_verbose: " 				<< global_verbose					<< endl;
	cout << "--" << endl;

	// compute from other cfg settings
	cout << "global_rdftype_id: " 			<< global_rdftype_id				<< endl;
	cout << "the number of servers: " 		<< global_nsrvs				<< endl;
	cout << "the number of threads: " 		<< global_nthrs				<< endl;
}

/**
 * reconfig client
 */
void
reload_cfg(void)
{
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

	global_batch_factor = atoi(config_map["global_batch_factor"].c_str());
	global_use_loc_cache = atoi(config_map["global_use_loc_cache"].c_str());
	global_silent = atoi(config_map["global_silent"].c_str());
	global_mt_threshold = atoi(config_map["global_mt_threshold"].c_str());
	global_rdma_threshold = atoi(config_map["global_rdma_threshold"].c_str());

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

	global_eth_port_base = atoi(config_map["global_eth_port_base"].c_str());
	global_rdma_port_base = atoi(config_map["global_rdma_port_base"].c_str());
	global_use_rbf = atoi(config_map["global_use_rbf"].c_str());
	global_use_rdma = atoi(config_map["global_use_rdma"].c_str());
	global_rdma_threshold = atoi(config_map["global_rdma_threshold"].c_str());
	global_nbewkrs = atoi(config_map["global_num_backends"].c_str());
	global_nfewkrs = atoi(config_map["global_num_frontends"].c_str());
	global_batch_factor = atoi(config_map["global_batch_factor"].c_str());
	global_mt_threshold = atoi(config_map["global_mt_threshold"].c_str());
	global_input_folder = config_map["global_input_folder"];
	global_use_loc_cache = atoi(config_map["global_use_loc_cache"].c_str());
	global_load_minimal_index = atoi(config_map["global_load_minimal_index"].c_str());
	global_silent = atoi(config_map["global_silent"].c_str());
	global_max_print_row = atoi(config_map["global_max_print_row"].c_str());
	global_total_memory_gb = atoi(config_map["global_total_memory_gb"].c_str());
	global_perslot_msg_mb = atoi(config_map["global_perslot_msg_mb"].c_str());
	global_perslot_rdma_mb = atoi(config_map["global_perslot_rdma_mb"].c_str());
	global_hash_header_million = atoi(config_map["global_hash_header_million"].c_str());
	global_enable_workstealing = atoi(config_map["global_enable_workstealing"].c_str());
	global_verbose = atoi(config_map["global_verbose"].c_str());

	// reserve ID 1 to rdf:type
	global_rdftype_id = 1;

	global_nsrvs = nsrvs;
	global_nthrs = global_nbewkrs + global_nfewkrs;


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
	if (global_verbose) dump_cfg();
	return;
}
