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

class Global {
public:
    // another choice
    // e.g., static int &num_threads() { static int _num_threads = 2; return _num_threads; }

    static int num_servers __attribute__((weak));
    static int num_threads __attribute__((weak));

    static int num_proxies __attribute__((weak));
    static int num_engines __attribute__((weak));

    static string input_folder __attribute__((weak));

    static int data_port_base __attribute__((weak));
    static int ctrl_port_base __attribute__((weak));

    static int rdma_buf_size_mb __attribute__((weak));
    static int rdma_rbf_size_mb __attribute__((weak));

    static bool use_rdma __attribute__((weak));
    static int rdma_threshold __attribute__((weak));

    static int mt_threshold __attribute__((weak));

    static bool enable_caching __attribute__((weak));
    static bool enable_workstealing __attribute__((weak));
    static int stealing_pattern __attribute__((weak));

    static bool silent __attribute__((weak));

    static bool enable_planner __attribute__((weak));
    static bool generate_statistics __attribute__((weak));
    static bool enable_budget __attribute__((weak));

    static bool enable_vattr __attribute__((weak));

    static int memstore_size_gb __attribute__((weak));
    static int est_load_factor __attribute__((weak));

    static int num_gpus __attribute__((weak));
    static int gpu_kvcache_size_gb __attribute__((weak));
    static int gpu_rbuf_size_mb __attribute__((weak));
    static int gpu_rdma_buf_size_mb __attribute__((weak));
    static int gpu_key_blk_size_mb __attribute__((weak));
    static int gpu_value_blk_size_mb __attribute__((weak));
    static bool gpu_enable_pipeline __attribute__((weak));
};


int Global::num_servers = 1;    // the number of servers
int Global::num_threads = 2;    // the number of threads per server (incl. proxy and engine)

int Global::num_proxies = 1;    // the number of proxies
int Global::num_engines = 1;    // the number of engines

string Global::input_folder;

int Global::data_port_base = 5500;
int Global::ctrl_port_base = 9576;

int Global::rdma_buf_size_mb = 64;
int Global::rdma_rbf_size_mb = 16;

bool Global::use_rdma = true;
int Global::rdma_threshold = 300;

int Global::mt_threshold = 16;

bool Global::enable_caching = true;
bool Global::enable_workstealing = false;
int Global::stealing_pattern = 0;  // 0 = pair stealing,  1 = ring stealing

bool Global::silent = true;  // don't take back results by default

bool Global::enable_planner = true;  // for planner
bool Global::generate_statistics = true;  // for planner
bool Global::enable_budget = true;  // for planner

bool Global::enable_vattr = false;  // for attr

// kvstore
int Global::memstore_size_gb = 20;
/**
 * global estimate load factor
 * when allocating buckets to segments during initialization,
 * #keys / #slots = global_est_load_factor / 100, so
 * #buckets = (#keys * 100) / (ASSOCIATIVITY * global_est_load_factor)
 */
int Global::est_load_factor = 55;

// GPU support
int Global::num_gpus = 0;
int Global::gpu_kvcache_size_gb = 10;    // key-value cache
int Global::gpu_rbuf_size_mb =  32;      // result (dual) buffer
int Global::gpu_rdma_buf_size_mb = 64;   // RDMA buffer
int Global::gpu_key_blk_size_mb = 16;
int Global::gpu_value_blk_size_mb = 4;
bool Global::gpu_enable_pipeline = true;
