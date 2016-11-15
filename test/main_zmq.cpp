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

#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>

#include <iostream>
#include "utils.h"
#include "global_cfg.h"
#include "thread_cfg.h"
#include "string_server.h"
#include "distributed_graph.h"
#include "server.h"
#include "client.h"
#include "client_mode.h"

using namespace std;

int socket_0[] = {
	0, 2, 4, 6, 8, 10, 12, 14, 16, 18
};

int socket_1[] = {
	1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
};
void pin_to_core(size_t core) {
	cpu_set_t  mask;
	CPU_ZERO(&mask);
	CPU_SET(core , &mask);
	int result = sched_setaffinity(0, sizeof(mask), &mask);
}

void* Run(void *ptr) {
	struct thread_cfg *cfg = (struct thread_cfg*) ptr;
	pin_to_core(socket_1[cfg->t_id]);
	if (cfg->t_id >= cfg->client_num) {
		((server*)(cfg->ptr))->run();
	} else {
		iterative_shell(((client*)(cfg->ptr)));
	}
}

int main(int argc, char * argv[]) {
	if (argc != 3) {
		cout << "usage:./wukong config_file hostfile" << endl;
		return -1;
	}
	load_global_cfg(argv[1]);

	global_use_rbf = false;
	global_use_rdma = false;
	global_rdma_threshold = 0;


	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	uint64_t rdma_size = 1024 * 1024 * 1024;
	rdma_size = rdma_size * global_total_memory_gb;
	uint64_t msg_slot_per_thread = 1024 * 1024 * global_perslot_msg_mb;
	uint64_t rdma_slot_per_thread = 1024 * 1024 * global_perslot_rdma_mb;
	uint64_t total_size = rdma_size + rdma_slot_per_thread * global_nthrs + msg_slot_per_thread * global_nthrs;
	Network_Node *node = new Network_Node(world.rank(), global_nthrs, string(argv[2])); //[0-thread_num-1] are used
	char *buffer = (char*) malloc(total_size);
	memset(buffer, 0, total_size);
	RdmaResource *rdma = new RdmaResource(world.size(), global_nthrs,
	                                      world.rank(), buffer, total_size, rdma_slot_per_thread, msg_slot_per_thread, rdma_size);
	rdma->node = node;
	MPI_Barrier(MPI_COMM_WORLD);

	thread_cfg* cfg_array = new thread_cfg[global_nthrs];
	for (int i = 0; i < global_nthrs; i++) {
		cfg_array[i].t_id = i;
		cfg_array[i].t_num = global_nthrs;
		cfg_array[i].m_id = world.rank();
		cfg_array[i].m_num = world.size();
		cfg_array[i].client_num = global_nfewkrs;
		cfg_array[i].server_num = global_nbewkrs;
		cfg_array[i].rdma = rdma;
		cfg_array[i].node = new Network_Node(cfg_array[i].m_id, cfg_array[i].t_id, string(argv[2]));
		cfg_array[i].init();
	}

	string_server str_server(global_input_folder);
	distributed_graph graph(world, rdma, global_input_folder);
	client** client_array = new client*[global_nfewkrs];
	for (int i = 0; i < global_nfewkrs; i++) {
		client_array[i] = new client(&cfg_array[i], &str_server);
	}
	server** server_array = new server*[global_nbewkrs];
	for (int i = 0; i < global_nbewkrs; i++) {
		server_array[i] = new server(graph, &cfg_array[global_nfewkrs + i]);
	}
	for (int i = 0; i < global_nbewkrs; i++) {
		server_array[i]->set_server_array(server_array);
	}


	pthread_t     *thread  = new pthread_t[global_nthrs];
	for (size_t id = 0; id < global_nthrs; ++id) {
		if (id < global_nfewkrs) {
			cfg_array[id].ptr = client_array[id];
		} else {
			cfg_array[id].ptr = server_array[id - global_nfewkrs];
		}
		pthread_create (&(thread[id]), NULL, Run, (void *) & (cfg_array[id]));
	}
	for (size_t t = 0 ; t < global_nthrs; t++) {
		int rc = pthread_join(thread[t], NULL);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
	}

	return 0;
}
