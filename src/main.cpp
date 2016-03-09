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

void* Run_SendAndRecv(void *ptr) {
	struct thread_cfg *cfg = (struct thread_cfg*) ptr;
	if(cfg->t_id >= cfg->client_num){
		for(int i=0;i<cfg->m_num;i++){
			for(int j=0;j<cfg->client_num;j++){
				cfg->node->Recv();
			}
		}
		string message="("+to_string(cfg->m_id)+
						","+to_string(cfg->t_id)+
						") recv all messages";
		cout<<message<<endl;
	}else {
		for(int i=0;i<cfg->m_num;i++){
			for(int j=0;j<cfg->server_num;j++){
				cfg->node->Send(i,cfg->client_num+j,"test");
			}
		}
		string message="("+to_string(cfg->m_id)+
						","+to_string(cfg->t_id)+
						") send all messages";
		cout<<message<<endl;
	}
}

void* Run(void *ptr) {
	struct thread_cfg *cfg = (struct thread_cfg*) ptr;
	if(cfg->t_id >= cfg->client_num){
		((server*)(cfg->ptr))->run();
	}else {
		iterative_shell(((client*)(cfg->ptr)));
	}
}

int main(int argc, char * argv[]) {
	if(argc !=3) {
		cout<<"usage:./wukong config_file hostfile"<<endl;
		return -1;
	}
    load_global_cfg(argv[1]);
    boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	uint64_t rdma_size = 1024*1024*1024;
	rdma_size = rdma_size*global_total_memory_gb;
	uint64_t slot_per_thread= 1024*1024*global_perslot_msg_mb;
	uint64_t total_size=rdma_size+slot_per_thread*global_num_thread*2;
	Network_Node *node = new Network_Node(world.rank(),global_num_thread,string(argv[2]));//[0-thread_num-1] are used
	char *buffer= (char*) malloc(total_size);
	memset(buffer,0,total_size);
	RdmaResource *rdma=new RdmaResource(world.size(),global_num_thread,
				world.rank(),buffer,total_size,slot_per_thread,rdma_size);
	rdma->node = node;
	rdma->Servicing();
	rdma->Connect();

	string_server str_server(global_input_folder);
	distributed_graph graph(world,rdma,global_input_folder);
	thread_cfg* cfg_array= new thread_cfg[global_num_thread];
	for(int i=0;i<global_num_thread;i++){
		cfg_array[i].t_id=i;
		cfg_array[i].t_num=global_num_thread;
		cfg_array[i].m_id=world.rank();
		cfg_array[i].m_num=world.size();
		cfg_array[i].client_num=global_num_client;
		cfg_array[i].server_num=global_num_server;
		cfg_array[i].rdma=rdma;
		cfg_array[i].node=new Network_Node(cfg_array[i].m_id,cfg_array[i].t_id,string(argv[2]));
		cfg_array[i].init();
	}


	client** client_array=new client*[global_num_client];
	for(int i=0;i<global_num_client;i++){
		client_array[i]=new client(&cfg_array[i],&str_server);
	}
	server** server_array=new server*[global_num_server];
	for(int i=0;i<global_num_server;i++){
		server_array[i]=new server(graph,&cfg_array[global_num_client+i]);
	}


	pthread_t     *thread  = new pthread_t[global_num_thread];
	for(size_t id = 0;id < global_num_thread;++id) {
		if(id<global_num_client){
			cfg_array[id].ptr=client_array[id];
		} else {
			cfg_array[id].ptr=server_array[id-global_num_client];
		}
		pthread_create (&(thread[id]), NULL, Run, (void *) &(cfg_array[id]));
	}
	for(size_t t = 0 ; t < global_num_thread; t++) {
		int rc = pthread_join(thread[t], NULL);
		if (rc) {
			printf("ERROR; return code from pthread_join() is %d\n", rc);
			exit(-1);
		}
	}
	return 0;
}
