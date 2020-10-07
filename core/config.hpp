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

#include <map>
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <boost/algorithm/string/predicate.hpp>

#include "global.hpp"
#include "rdma.hpp"

// utils
#include "assertion.hpp"

using namespace std;

static bool set_immutable_config(string cfg_name, string value)
{
    if (cfg_name == "global_num_proxies") {
        Global::num_proxies = atoi(value.c_str());
        ASSERT(Global::num_proxies > 0);
    } else if (cfg_name == "global_num_engines") {
        Global::num_engines = atoi(value.c_str());
        ASSERT(Global::num_engines > 0);
    } else if (cfg_name == "global_input_folder") {
        Global::input_folder = value;

        // make sure to check that the Global::input_folder is non-empty.
        if (Global::input_folder.length() == 0) {
            logstream(LOG_ERROR) << "the directory path of RDF data can not be empty!"
                                 << "You should set \"global_input_folder\" in config file." << LOG_endl;
            exit(-1);
        }

        // force a "/" at the end of Global::input_folder.
        if (Global::input_folder[Global::input_folder.length() - 1] != '/')
            Global::input_folder = Global::input_folder + "/";
    } else if (cfg_name == "global_data_port_base") {
        Global::data_port_base = atoi(value.c_str());
        ASSERT(Global::data_port_base > 0);
    } else if (cfg_name == "global_ctrl_port_base") {
        Global::ctrl_port_base = atoi(value.c_str());
        ASSERT(Global::ctrl_port_base > 0);
    } else if (cfg_name == "global_memstore_size_gb") {
        Global::memstore_size_gb = atoi(value.c_str());
        ASSERT(Global::memstore_size_gb > 0);
    } else if (cfg_name == "global_est_load_factor") {
        Global::est_load_factor = atoi(value.c_str());
        ASSERT(Global::est_load_factor > 0 && Global::est_load_factor < 100);
    } else if (cfg_name == "global_rdma_buf_size_mb") {
        if (RDMA::get_rdma().has_rdma())
            Global::rdma_buf_size_mb = atoi(value.c_str());
        else
            Global::rdma_buf_size_mb = 0;
        ASSERT(Global::rdma_buf_size_mb >= 0);
    } else if (cfg_name == "global_rdma_rbf_size_mb") {
        if (RDMA::get_rdma().has_rdma())
            Global::rdma_rbf_size_mb = atoi(value.c_str());
        else
            Global::rdma_rbf_size_mb = 0;
        ASSERT(Global::rdma_rbf_size_mb >= 0);
    } else if (cfg_name == "global_generate_statistics") {
        Global::generate_statistics = atoi(value.c_str());
    } else if (cfg_name == "global_num_gpus") {
        Global::num_gpus = atoi(value.c_str());
    } else if (cfg_name == "global_gpu_rdma_buf_size_mb") {
        if (RDMA::get_rdma().has_rdma())
            Global::gpu_rdma_buf_size_mb = atoi(value.c_str());
        else
            Global::gpu_rdma_buf_size_mb = 0;
        ASSERT(Global::gpu_rdma_buf_size_mb >= 0);
    } else if (cfg_name == "global_gpu_rbuf_size_mb") {
        Global::gpu_rbuf_size_mb = atoi(value.c_str());
    } else if (cfg_name == "global_gpu_kvcache_size_gb") {
        Global::gpu_kvcache_size_gb = atoi(value.c_str());
    } else if (cfg_name == "global_gpu_key_blk_size_mb") {
        Global::gpu_key_blk_size_mb = atoi(value.c_str());
    } else if (cfg_name == "global_gpu_value_blk_size_mb") {
        Global::gpu_value_blk_size_mb = atoi(value.c_str());
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
                logstream(LOG_ERROR) << "can't enable RDMA due to building Wukong w/o RDMA support!\n"
                                     << "HINT: please disable global_use_rdma in config file." << LOG_endl;
                Global::use_rdma = false; // disable RDMA if no RDMA device
                return true;
            }

            Global::use_rdma = true;
        } else {
            Global::use_rdma = false;
        }
    } else if (cfg_name == "global_rdma_threshold") {
        Global::rdma_threshold = atoi(value.c_str());
    } else if (cfg_name == "global_mt_threshold") {
        Global::mt_threshold = atoi(value.c_str());
        ASSERT(Global::mt_threshold > 0);
    } else if (cfg_name == "global_enable_caching") {
        Global::enable_caching = atoi(value.c_str());
    } else if (cfg_name == "global_enable_workstealing") {
        Global::enable_workstealing = atoi(value.c_str());
    } else if (cfg_name == "global_stealing_pattern") {
        Global::stealing_pattern = atoi(value.c_str());
    } else if (cfg_name == "global_silent") {
        Global::silent = atoi(value.c_str());
    } else if (cfg_name == "global_enable_planner") {
        Global::enable_planner = atoi(value.c_str());
    } else if (cfg_name == "global_enable_budget") {
        Global::enable_budget = atoi(value.c_str());
    } else if (cfg_name == "global_enable_vattr") {
        Global::enable_vattr = atoi(value.c_str());
    } else if (cfg_name == "global_gpu_enable_pipeline") {
        Global::gpu_enable_pipeline = atoi(value.c_str());
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
        logstream(LOG_ERROR) << fname << " does not exist." << LOG_endl;
        exit(0);
    }

    string line, row, val;
    while (std::getline(file, line)) {
        if (boost::starts_with(line, "#") || line.empty())
            continue; // skip comments and blank lines

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

    for (auto const &entry : items)
        set_mutable_config(entry.first, entry.second);

    // limited the number of engines
    Global::mt_threshold = max(1, min(Global::mt_threshold, Global::num_engines));

    return;
}

/**
 * load config
 */
void load_config(string fname, int nsrvs)
{
    ASSERT(nsrvs > 0);
    Global::num_servers = nsrvs;

    // load config file
    map<string, string> items;
    file2items(fname, items);

    for (auto const &entry : items) {
        if (!(set_immutable_config(entry.first, entry.second)
                || set_mutable_config(entry.first, entry.second))) {
            logstream(LOG_WARNING) << "unsupported configuration item! ("
                                   << entry.first << ")" << LOG_endl;
        }
    }

    // set the total number of threads
    Global::num_threads = Global::num_engines + Global::num_proxies;

#ifdef USE_GPU
    // each GPU card needs one (dedicated) agent thread
    Global::num_threads += Global::num_gpus;
    if (Global::num_gpus != 1) {
        logstream(LOG_ERROR) << "Wrong config: please config num_gpus with 1 to enable GPU extension."
                             << LOG_endl;
        exit(-1);
    }
#endif

    // limited the number of engines
    Global::mt_threshold = max(1, min(Global::mt_threshold, Global::num_engines));

    return;
}

/**
 * print current config
 */
void print_config(void)
{
    cout << "------ global configurations ------" << LOG_endl;

    // setting by config file
    cout << "the number of proxies: "        << Global::num_proxies           << LOG_endl;
    cout << "the number of engines: "        << Global::num_engines           << LOG_endl;
    cout << "global_input_folder: "          << Global::input_folder          << LOG_endl;
    cout << "global_memstore_size_gb: "      << Global::memstore_size_gb      << LOG_endl;
    cout << "global_est_load_factor: "       << Global::est_load_factor       << LOG_endl;
    cout << "global_data_port_base: "        << Global::data_port_base        << LOG_endl;
    cout << "global_ctrl_port_base: "        << Global::ctrl_port_base        << LOG_endl;
    cout << "global_rdma_buf_size_mb: "      << Global::rdma_buf_size_mb      << LOG_endl;
    cout << "global_rdma_rbf_size_mb: "      << Global::rdma_rbf_size_mb      << LOG_endl;
    cout << "global_use_rdma: "              << Global::use_rdma              << LOG_endl;
    cout << "global_enable_caching: "        << Global::enable_caching        << LOG_endl;
    cout << "global_enable_workstealing: "   << Global::enable_workstealing   << LOG_endl;
    cout << "global_stealing_pattern: "      << Global::stealing_pattern      << LOG_endl;
    cout << "global_rdma_threshold: "        << Global::rdma_threshold        << LOG_endl;
    cout << "global_mt_threshold: "          << Global::mt_threshold          << LOG_endl;
    cout << "global_silent: "                << Global::silent                << LOG_endl;
    cout << "global_enable_planner: "        << Global::enable_planner        << LOG_endl;
    cout << "global_generate_statistics: "   << Global::generate_statistics   << LOG_endl;
    cout << "global_enable_budget: "         << Global::enable_budget         << LOG_endl;
    cout << "global_enable_vattr: "          << Global::enable_vattr          << LOG_endl;
    cout << "global_num_gpus: "              << Global::num_gpus              << LOG_endl;
    cout << "global_gpu_rdma_buf_size_mb: "  << Global::gpu_rdma_buf_size_mb  << LOG_endl;
    cout << "global_gpu_rbuf_size_mb: "      << Global::gpu_rbuf_size_mb      << LOG_endl;
    cout << "global_gpu_kvcache_size_gb: "   << Global::gpu_kvcache_size_gb   << LOG_endl;
    cout << "global_gpu_key_blk_size_mb: "   << Global::gpu_key_blk_size_mb   << LOG_endl;
    cout << "global_gpu_value_blk_size_mb: " << Global::gpu_value_blk_size_mb << LOG_endl;
    cout << "global_gpu_enable_pipeline: "   << Global::gpu_enable_pipeline   << LOG_endl;
    cout << "--" << LOG_endl;

    // compute from other settings
    cout << "the number of servers: "        << Global::num_servers           << LOG_endl;
    cout << "the number of threads: "        << Global::num_threads           << LOG_endl;

}
