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

/*
 * The processor architecture of our cluster (Cube0-5)
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
 * bind server-worker threads to the same socket, at most 8 each.
 *
 * TODO: it should be identify by runtime detection
 */
int cores[] = {
	1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
	0, 2, 4, 6, 8, 10, 12, 14, 16, 18
};

void
pin_to_core(size_t core)
{
	cpu_set_t  mask;
	CPU_ZERO(&mask);
	CPU_SET(core, &mask);
	int result = sched_setaffinity(0, sizeof(mask), &mask);
}

void*
worker_thread(void *arg)
{
	struct thread_cfg *cfg = (struct thread_cfg *) arg;
	pin_to_core(cores[cfg->wid]);

	// reserver ncwkrs threads to client workers
	if (cfg->wid >= cfg->ncwkrs) {
		// server-worker threads
		((server *)(cfg->ptr))->run();
	} else {
		// built-in client (by default)
		interactive_shell((client*)(cfg->ptr));
		// connected client
		//proxy_mode((client*)(cfg->ptr));
	}
}

int
main(int argc, char *argv[])
{
	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	if (argc != 3) {
		cout << "usage: ./wukong config_file hostfile" << endl;
		exit(-1);
	}
	cfg_fname = std::string(argv[1]);
	host_fname = std::string(argv[2]);

	// config global setting
	load_cfg();

	// calculate memory usage
	uint64_t rdma_size = GiB2B(global_total_memory_gb);
	uint64_t msg_slot_per_thread = MiB2B(global_perslot_msg_mb);
	uint64_t rdma_slot_per_thread = MiB2B(global_perslot_rdma_mb);
	uint64_t mem_size = rdma_size
	                    + rdma_slot_per_thread * global_num_thread
	                    + msg_slot_per_thread * global_num_thread;
	cout << "memory usage: " << B2GiB(mem_size) << "GB" << endl;


	// create an RDMA instance
	char *buffer = (char*) malloc(mem_size);
	memset(buffer, 0, mem_size);
	RdmaResource *rdma = new RdmaResource(world.size(), global_num_thread,
	                                      world.rank(), buffer, mem_size,
	                                      rdma_slot_per_thread, msg_slot_per_thread, rdma_size);
	// a special TCP/IP instance used by RDMA (wid == global_num_threads)
	rdma->node = new Network_Node(world.rank(), global_num_thread, host_fname);
	rdma->Servicing();
	rdma->Connect();

	thread_cfg *cfg_array = new thread_cfg[global_num_thread];
	for (int i = 0; i < global_num_thread; i++) {
		cfg_array[i].wid = i;
		cfg_array[i].nwkrs = global_num_thread;
		cfg_array[i].sid = world.rank();
		cfg_array[i].nsrvs = world.size();
		cfg_array[i].ncwkrs = global_num_client;
		cfg_array[i].nswkrs = global_num_server;
		cfg_array[i].rdma = rdma;
		cfg_array[i].node = new Network_Node(cfg_array[i].sid, cfg_array[i].wid, host_fname);
		cfg_array[i].init();
	}

	/*
	    bool get_back=true;
		for(int size=8;size<1000000;size*=2){
			if(world.rank()==0){
				string str;
				str.resize(size);
	            for(int i=0;i<str.size();i++){
	                str[i]=(i*i*i);
	            }
				uint64_t t1,t2,t_mid;
	            int use_rdma=0;
	            for(int i=0;i<5;i++){
	                t1=timer::get_usec();
	    			if(global_use_rbf){
	                    rdma->rbfSend(0,1, 0, str.c_str(),str.size());
	                    t_mid=timer::get_usec();
	                    if(get_back)str=rdma->rbfRecv(0);
	                    use_rdma=1;
	    			} else {
	    				cfg_array[0].node->Send(1,0,str);
	                    t_mid=timer::get_usec();
	    				if(get_back)str=cfg_array[0].node->Recv();
	    			}
	                t2=timer::get_usec();
	                cout<<t_mid-t1<<" usec for send; "<<t2-t1<<" usec for total & size= "<<size <<endl;
	            }
			} else {
	            for(int i=0;i<15;i++){
	    			if(global_use_rbf){
	                    string str = rdma->rbfRecv(0);
	                    if(get_back)rdma->rbfSend(0,0, 0, str.c_str(),str.size());
	    			} else {
	    				string str=cfg_array[0].node->Recv();
	    				if(get_back)cfg_array[0].node->Send(0,0,str);
	    			}
	            }
			}
		}
		sleep(1);
		return 0;
	*/

	string_server str_server(global_input_folder);

	distributed_graph graph(world, rdma, global_input_folder);

	client **client_array = new client *[global_num_client];
	for (int i = 0; i < global_num_client; i++) {
		client_array[i] = new client(&cfg_array[i], &str_server);
	}

	server **server_array = new server *[global_num_server];
	for (int i = 0; i < global_num_server; i++) {
		server_array[i] = new server(graph, &cfg_array[global_num_client + i]);
	}
	for (int i = 0; i < global_num_server; i++) {
		server_array[i]->set_server_array(server_array);
	}

	// spawn client and server workers
	pthread_t *thread  = new pthread_t[global_num_thread];
	for (size_t id = 0; id < global_num_thread; ++id) {
		if (id < global_num_client)
			cfg_array[id].ptr = client_array[id];
		else
			cfg_array[id].ptr = server_array[id - global_num_client];

		pthread_create(&(thread[id]), NULL, worker_thread, (void *) & (cfg_array[id]));
	}

	// wait to termination
	for (size_t t = 0; t < global_num_thread; t++) {
		int rc = pthread_join(thread[t], NULL);
		if (rc) {
			printf("ERROR: return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
	}

	return 0;
}
