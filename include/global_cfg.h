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
#include <stdlib.h> //atoi
#include <sstream>

using namespace std;

/* non-configurable global variables */
extern int global_rdftype_id;	// only a global variable, but non-configurable
extern int global_nsrvs;		// the number of servers
extern int global_nthrs;		// the number of threads per server (incl. proxy and engine)

/* configurable global variables */
extern int global_num_engines;		// the number of engines
extern int global_num_proxies;		// the number of proxies

extern string global_input_folder;
extern bool global_load_minimal_index;
extern int global_max_print_row;
extern int global_total_memory_gb;
extern int global_perslot_msg_mb;
extern int global_perslot_rdma_mb;
extern int global_hash_header_million;

extern int global_eth_port_base;
extern int global_rdma_port_base;

extern bool global_use_rdma;
extern bool global_use_rbf;		// ring-buffer (by RDMA WRITE)
extern int global_enable_workstealing;
extern int global_verbose;

/* shared by proxy and engine */
extern int global_batch_factor;
extern bool global_use_loc_cache;
extern bool global_silent;
extern int global_mt_threshold;
extern int global_rdma_threshold;

/* set by command line */
extern std::string cfg_fname;
extern std::string host_fname;

void dump_cfg(void);
void reload_cfg(void);
void load_cfg(int nsrvs);
