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

	Network_Node* node;  // communicaiton by TCP/IP
	RdmaResource* rdma;  // communicaiton by RDMA

	unsigned int seed;
	void* ptr;

	//get_id for requests
	int inc_id;//internal

	void init() {
		inc_id = nwkrs * sid + wid;
		seed = inc_id;
	}

	unsigned get_random() {
		return rand_r(&seed);
	}

	int get_inc_id() {
		int tmp = inc_id;
		inc_id += nsrvs * nwkrs;
		return tmp;
	}

	int mid_of(int target_id) {
		return (target_id % (nsrvs * nwkrs)) / nwkrs;
	}

	int tid_of(int target_id) {
		return target_id % nwkrs;
	}

	bool is_client(int target_id) {
		if (tid_of(target_id) < ncwkrs) {
			return true;
		}
		return false;
	}
};
