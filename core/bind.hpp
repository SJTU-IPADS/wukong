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

#include <hwloc.h>
#include <boost/algorithm/string/predicate.hpp>

#include "global.hpp"
#include "config.hpp"

// utils
#include "logger2.hpp"

using namespace std;

/*
 * The processor architecture of machines in our cluster (Cube0-5)
 *
 * $numactl --hardware
 * available: 2 nodes (0-1)
 * node 0 cpus: 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38
 * node 0 size: 64265 MB
 * node 0 free: 19744 MB
 * node 1 cpus: 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39
 * node 1 size: 64503 MB
 * node 1 free: 53586 MB
 * node distances:
 * node   0   1
 *   0:  10  21
 *   1:  21  10
 *
 *
 * $cat config
 * global_num_proxies 4
 * global_num_engines 16
 *
 * $cat core.bind
 * 0  1  4  5  6  7  8  9 10 11
 * 2  3 12 13 14 15 16 17 18 19
 *
 */

vector<vector<int>> cpu_topo;
int num_cores = 0;

bool enable_binding = false;
vector<int> default_bindings; // bind to core one-by-one
map<int, int> core_bindings; // user-defined core binding


void dump_node_topo(vector<vector<int>> topo)
{
    logstream(LOG_INFO) << "TOPO: " << topo.size() << "nodes" << LOG_endl;
    for (int nid = 0; nid < topo.size(); nid++) {
        logstream(LOG_INFO) << "node " << nid << " cores: ";
        for (int cid = 0; cid < topo[nid].size(); cid++)
            logstream(LOG_INFO) << topo[nid][cid] << " ";
        logstream(LOG_INFO) << LOG_endl;
    }
}

void load_node_topo(void)
{
    hwloc_topology_t topology;

    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    // Currently, nnodes may return 0 while the NUMANODEs in cpulist is 1
    // (hwloc think there is actually no numa-node).
    // Fortunately, it can detect the number of processing units (PU) correctly
    // when MT processing is on, the number of PU will be twice as #cores
    int nnodes = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE);
    if (nnodes != 0) {
        cpu_topo.resize(nnodes);
        for (int i = 0; i < nnodes; i++) {
            hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, i);
            hwloc_cpuset_t cpuset = hwloc_bitmap_dup(obj->cpuset);

            unsigned int core = 0;
            hwloc_bitmap_foreach_begin(core, cpuset);
            cpu_topo[i].push_back(core);
            default_bindings.push_back(core);
            hwloc_bitmap_foreach_end();

            hwloc_bitmap_free(cpuset);
        }
    } else {
        cpu_topo.resize(1);
        int nPUs = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
        for (int i = 0; i < nPUs; i++) {
            hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, i);
            hwloc_cpuset_t cpuset = hwloc_bitmap_dup(obj->cpuset);

            unsigned int core = 0;
            hwloc_bitmap_foreach_begin(core, cpuset);
            cpu_topo[0].push_back(core);
            default_bindings.push_back(core);
            hwloc_bitmap_foreach_end();

            hwloc_bitmap_free(cpuset);
        }
    }

    num_cores = default_bindings.size();

    dump_node_topo(cpu_topo);
}

bool load_core_binding(string fname)
{
    int nbs = 0;

    //load file of core binding
    ifstream file(fname.c_str());
    if (!file) {
        logstream(LOG_ERROR) << fname << " does not exist." << LOG_endl;
        return false;
    }

    int nnodes = cpu_topo.size(), i = 0;
    string line;
    while (std::getline(file, line)) {
        if (boost::starts_with(line, "#"))
            continue; // skip comments and blank lines

        istringstream iss(line); // one node per line
        int ncores = cpu_topo[i].size(), j = 0, tid;
        while (iss >> tid) {
            core_bindings[tid] = cpu_topo[i % nnodes][j++ % ncores];
            nbs++;
        }

        i++; // next node
    }

    if (i < nnodes)
        logstream(LOG_WARNING) << "#bindings (in \'core.bind\') deos not use all of the NUMANODEs!"
                               << LOG_endl;

    if (i > nnodes)
        logstream(LOG_WARNING) << "#bindings (in \'core.bind\') exceeds number of the NUMANODEs!"
                               << LOG_endl;

    if (nbs < Global::num_threads)
        logstream(LOG_WARNING) << "#threads (in \'config\') exceeds #bindings (in \'bind\')!"
                               << LOG_endl;

    return true;
}


/*
 * Bind the current thread to a special core (core number)
 */
void bind_to_core(size_t core)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
        logstream(LOG_ERROR) << "Failed to set affinity (core: " << core << ")" << LOG_endl;
}

/*
 * Bind the current thread to special cores (mask)
 */
void bind_to_core(cpu_set_t mask)
{
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
        logstream(LOG_ERROR) << "Fail to set affinity!" << LOG_endl;
}

/*
 * Bind the current thread to all of cores
 * It would like unbind to special cores
 * and not return the previous core binding
 */
void bind_to_all()
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int i = 0; i < default_bindings.size(); i++)
        CPU_SET(default_bindings[i], &mask);

    if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
        logstream(LOG_ERROR) << "Fail to set affinity" << LOG_endl;
}

/*
 * Return the mask of the current core binding
 */
cpu_set_t get_core_binding()
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
        logstream(LOG_ERROR) << "Fail to get affinity" << LOG_endl;
    return mask;
}

/*
 * Unbind the current thread to special cores
 * and return the preivous core binding
 */
cpu_set_t unbind_to_core()
{
    cpu_set_t mask;
    mask = get_core_binding(); // record the current core binding

    bind_to_all();
    return mask;
}
