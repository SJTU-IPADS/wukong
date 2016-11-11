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

#include "network_node.h"
#include "rdma_resource.h"


struct thread_cfg {
	int sid;    // servert id
	int nsrvs;  // #servers
	int wid;    // worker id
	int nwkrs;  // #workers on each server
	int nswkrs; // #server-workers on each server
	int ncwkrs; // #client-workers on each server

	Network_Node *node;  // communicaiton by TCP/IP
	RdmaResource *rdma;  // communicaiton by RDMA

	unsigned int seed;
	void* ptr;

	// unique global ID for SPARQL requests
	int req_gid;

	void init() {
		req_gid = nwkrs * sid + wid;
		seed = req_gid;
	}

	unsigned get_random() {
		return rand_r(&seed);
	}

	int get_inc_id() {
		int tmp = req_gid;
		req_gid += nsrvs * nwkrs;
		return tmp;
	}

	int sid_of(int gid) {
		return (gid % (nsrvs * nwkrs)) / nwkrs;
	}

	int wid_of(int gid) {
		return gid % nwkrs;
	}

	bool is_cwkr(int gid) {
		if (wid_of(gid) < ncwkrs)
			return true;
		return false;
	}
};
