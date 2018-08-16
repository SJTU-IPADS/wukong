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

#include <string>

using namespace std;

int global_num_servers = 1;    // the number of servers
int global_num_threads = 2;    // the number of threads per server (incl. proxy and engine)

int global_num_proxies = 1;    // the number of proxies
int global_num_engines = 1;    // the number of engines

string global_input_folder;

int global_data_port_base = 5500;
int global_ctrl_port_base = 9576;

int global_memstore_size_gb = 20;
int global_rdma_buf_size_mb = 64;
int global_rdma_rbf_size_mb = 16;

bool global_use_rdma = true;
int global_rdma_threshold = 300;

int global_mt_threshold = 16;

bool global_enable_caching = true;
bool global_enable_workstealing = false;

bool global_silent = true;  // don't take back results by default

bool global_enable_planner = true;  // for planner
bool global_generate_statistics = true;

bool global_enable_vattr = false;  // for attr

// GPU support
int global_num_gpus = 1;
int global_gpu_kvcache_size_gb = 10;	// key-value cache
int global_gpu_rbuf_size_mb =  32;		// result (dual) buffer
int global_gpu_rdma_buf_size_mb = 64;	// RDMA buffer
int global_gpu_key_blk_size_mb = 16;
int global_gpu_value_blk_size_mb = 4;
