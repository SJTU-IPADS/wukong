#pragma once

#include "network_node.h"
#include "rdma_resource.h"


struct thread_cfg {
	int m_id; // machine id
	int m_num; // total machine number
	int t_id; // thread id
	int t_num;  // total thread number in each machine
	int server_num;// server thread number in each machine
	int client_num;// client thread number in each machine
	Network_Node* node;
	RdmaResource* rdma;
	unsigned int seed;
	void* ptr;

	//get_id for requests
	int inc_id;//internal

	void init() {
		inc_id = t_num * m_id + t_id;
		seed = inc_id;
	}

	unsigned get_random() {
		return rand_r(&seed);
	}

	int get_inc_id() {
		int tmp = inc_id;
		inc_id += m_num * t_num;
		return tmp;
	}

	int mid_of(int target_id) {
		return (target_id % (m_num * t_num)) / t_num;
	}

	int tid_of(int target_id) {
		return target_id % t_num;
	}

	bool is_client(int target_id) {
		if (tid_of(target_id) < client_num) {
			return true;
		}
		return false;
	}
};
