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
#include "assertion.hpp"

using namespace std;

static bool set_immutable_config(string cfg_name, string value)
{
    if (cfg_name == "global_num_proxies") {
        global_num_proxies = atoi(value.c_str());
        ASSERT(global_num_proxies > 0);
    } else if (cfg_name == "global_num_engines") {
        global_num_engines = atoi(value.c_str());
        ASSERT(global_num_engines > 0);
    } else if (cfg_name == "global_input_folder") {
        global_input_folder = value;

        // make sure to check that the global_input_folder is non-empty.
        if (global_input_folder.length() == 0) {
            logstream(LOG_ERROR) << "the directory path of RDF data can not be empty!"
                                 << "You should set \"global_input_folder\" in config file." << LOG_endl;
            exit(-1);
        }

        // force a "/" at the end of global_input_folder.
        if (global_input_folder[global_input_folder.length() - 1] != '/')
            global_input_folder = global_input_folder + "/";
    } else if (cfg_name == "global_data_port_base") {
        global_data_port_base = atoi(value.c_str());
        ASSERT(global_data_port_base > 0);
    } else if (cfg_name == "global_ctrl_port_base") {
        global_ctrl_port_base = atoi(value.c_str());
        ASSERT(global_ctrl_port_base > 0);
    } else if (cfg_name == "global_memstore_size_gb") {
        global_memstore_size_gb = atoi(value.c_str());
        ASSERT(global_memstore_size_gb > 0);
    } else if (cfg_name == "global_rdma_buf_size_mb") {
        if (RDMA::get_rdma().has_rdma())
            global_rdma_buf_size_mb = atoi(value.c_str());
        else
            global_rdma_buf_size_mb = 0;
        ASSERT(global_rdma_buf_size_mb >= 0);
    } else if (cfg_name == "global_rdma_rbf_size_mb") {
        if (RDMA::get_rdma().has_rdma())
            global_rdma_rbf_size_mb = atoi(value.c_str());
        else
            global_rdma_buf_size_mb = 0;
        ASSERT(global_rdma_rbf_size_mb >= 0);
    } else if (cfg_name == "global_generate_statistics") {
        global_generate_statistics = atoi(value.c_str());
    }
#ifdef USE_GPU
    else if (cfg_name == "global_num_gpus") {
        global_num_gpus = atoi(value.c_str());
    } else if (cfg_name == "global_gpu_rdma_buf_size_mb") {
        if (RDMA::get_rdma().has_rdma())
            global_gpu_rdma_buf_size_mb = atoi(value.c_str());
        else
            global_gpu_rdma_buf_size_mb = 0;
        ASSERT(global_gpu_rdma_buf_size_mb >= 0);
    } else if (cfg_name == "global_gpu_max_element") {
        char *tmp;
        global_gpu_max_element = strtoull(value.c_str(), &tmp, 10);
    } else if (cfg_name == "global_gpu_kvcache_size_gb") {
        global_gpu_kvcache_size_gb = atoi(value.c_str());
    } else if (cfg_name == "global_gpu_key_blk_size_mb") {
        global_gpu_key_blk_size_mb = atoi(value.c_str());
    } else if (cfg_name == "global_gpu_value_blk_size_mb") {
        global_gpu_value_blk_size_mb = atoi(value.c_str());
    }
#endif
    else
        return false;

    return true;
}

static bool set_mutable_config(string cfg_name, string value)
{
    if (cfg_name == "global_use_rdma") {
        if (atoi(value.c_str())) {
            if (!RDMA::get_rdma().has_rdma()) {
                logstream(LOG_ERROR) << "can't enable RDMA due to building Wukong w/o RDMA support!\n"
                                     << "HINT: please disable global_use_rdma in config file." << LOG_endl;
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
        ASSERT(global_mt_threshold > 0);
    } else if (cfg_name == "global_enable_caching") {
        global_enable_caching = atoi(value.c_str());
    } else if (cfg_name == "global_enable_workstealing") {
        global_enable_workstealing = atoi(value.c_str());
    } else if (cfg_name == "global_silent") {
        global_silent = atoi(value.c_str());
    } else if (cfg_name == "global_enable_planner") {
        global_enable_planner = atoi(value.c_str());
    } else if (cfg_name == "global_enable_vattr") {
        global_enable_vattr = atoi(value.c_str());
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
    global_mt_threshold = max(1, min(global_mt_threshold, global_num_engines));

    return;
}

/**
 * load config
 */
void load_config(string fname, int num_servers)
{
    global_num_servers = num_servers;
    ASSERT(num_servers > 0);

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
    logstream(LOG_INFO) << "------ global configurations ------" << LOG_endl;

    // setting by config file
    logstream(LOG_INFO) << "the number of proxies: "        << global_num_proxies           << LOG_endl;
    logstream(LOG_INFO) << "the number of engines: "        << global_num_engines           << LOG_endl;
    logstream(LOG_INFO) << "global_input_folder: "      << global_input_folder          << LOG_endl;
    logstream(LOG_INFO) << "global_data_port_base: "        << global_data_port_base        << LOG_endl;
    logstream(LOG_INFO) << "global_ctrl_port_base: "        << global_ctrl_port_base        << LOG_endl;
    logstream(LOG_INFO) << "global_memstore_size_gb: "  << global_memstore_size_gb      << LOG_endl;
    logstream(LOG_INFO) << "global_rdma_buf_size_mb: "  << global_rdma_buf_size_mb      << LOG_endl;
    logstream(LOG_INFO) << "global_rdma_rbf_size_mb: "  << global_rdma_rbf_size_mb      << LOG_endl;
    logstream(LOG_INFO) << "global_use_rdma: "          << global_use_rdma              << LOG_endl;
    logstream(LOG_INFO) << "global_enable_caching: "        << global_enable_caching        << LOG_endl;
    logstream(LOG_INFO) << "global_enable_workstealing: "   << global_enable_workstealing   << LOG_endl;
    logstream(LOG_INFO) << "global_rdma_threshold: "        << global_rdma_threshold        << LOG_endl;
    logstream(LOG_INFO) << "global_mt_threshold: "      << global_mt_threshold          << LOG_endl;
    logstream(LOG_INFO) << "global_silent: "                << global_silent                << LOG_endl;
    logstream(LOG_INFO) << "global_enable_planner: "        << global_enable_planner        << LOG_endl;
    logstream(LOG_INFO) << "global_generate_statistics: "   << global_generate_statistics   << LOG_endl;
    logstream(LOG_INFO) << "global_enable_vattr: "      << global_enable_vattr          << LOG_endl;

#ifdef USE_GPU
    logstream(LOG_INFO) << "global_num_gpus: "        << global_num_gpus        << LOG_endl;
    logstream(LOG_INFO) << "global_gpu_rdma_buf_size_mb: "  << global_gpu_rdma_buf_size_mb  << LOG_endl;
    logstream(LOG_INFO) << "global_gpu_max_element: "  << global_gpu_max_element  << LOG_endl;
    logstream(LOG_INFO) << "global_gpu_kvcache_size_gb: "  << global_gpu_kvcache_size_gb  << LOG_endl;
    logstream(LOG_INFO) << "global_gpu_key_blk_size_mb: "  << global_gpu_key_blk_size_mb  << LOG_endl;
    logstream(LOG_INFO) << "global_gpu_value_blk_size_mb: "  << global_gpu_value_blk_size_mb  << LOG_endl;
#endif

    logstream(LOG_INFO) << "--" << LOG_endl;

    // compute from other settings
    logstream(LOG_INFO) << "the number of servers: "        << global_num_servers           << LOG_endl;
    logstream(LOG_INFO) << "the number of threads: "        << global_num_threads           << LOG_endl;

}
