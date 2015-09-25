#pragma once 
#include "network_node.h"
#include "rdma_resource.h"

struct thread_cfg{
	int m_id; // machine id
	int m_num; // total machine number
	int t_id; // thread id
	int t_num;  // total thread number in each machine
	Network_Node* node;
	RdmaResource* rdma;
	void* ptr; 
};