
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <iostream>

#include "graph.h"
#include "index_server.h"
#include "network_node.h"
#include "rdma_resource.h"
#include "thread_cfg.h"
#include "global_cfg.h"
#include <pthread.h>
#include <sstream>

#include "trinity_client.h"
#include "trinity_server.h"

int socket_0[] = {
  0,2,4,6,8,10,12,14,16,18
};
int socket_1[] = {
  1,3,5,7,9,11,13,15,17,19,0,2,4,6,8,10,12,14,16,18
};
void pin_to_core(size_t core) {
  cpu_set_t  mask;
  CPU_ZERO(&mask);
  CPU_SET(core , &mask);
  int result=sched_setaffinity(0, sizeof(mask), &mask);  
}

int server_num;
int client_num;
int thread_num;
int batch_factor;

void interactive_mode(trinity_client* is){
	while(true){
		cout<<"interactive mode:"<<endl;
		// string filename;
		// cin>>filename;
		string input_str;
		std::getline(std::cin,input_str);
		istringstream iss(input_str);
		string filename;
		iss>>filename;
		int execute_count=1;
		iss>>execute_count;
		if(execute_count<1){
			execute_count=1;
		}
		int sum=0;
		for(int i=0;i<execute_count;i++){
			ifstream file(filename);
			if(!file){
				cout<<"File "<<filename<<" not exist"<<endl;
				break;
			}
			string cmd;
			while(file>>cmd){
				if(cmd=="execute"){
					uint64_t t1=timer::get_usec();
					is->Send();
					is->Recv();
					uint64_t t2=timer::get_usec();
					sum+=t2-t1;
					break;
				} else {
					is->cmd_string.push_back(cmd);
				}
			}
		}
		cout<<"average latency "<<sum/execute_count<<" us"<<endl;
	}
}
void* Run(void *ptr) {
  struct thread_cfg *cfg = (struct thread_cfg*) ptr;
  pin_to_core(socket_1[cfg->t_id]);
  //cout<<"("<<cfg->m_id<<","<<cfg->t_id<<")"<<endl;
  if(cfg->t_id >= cfg->client_num){
  	if(cfg->t_id!=cfg->client_num){
  		return NULL;
  	}
  	cout<<"Server "<<cfg->m_id<<" started"<<endl;
  	((trinity_server*)(cfg->ptr))->run();
  } else {
  	cout<<"Client "<<cfg->m_id<<" started"<<endl;
  	while(true){
		if(cfg->m_id!=0 || cfg->t_id!=0){
			// sleep forever
			sleep(1);
		}
		else{
			interactive_mode((trinity_client*)(cfg->ptr));
		}
	}  	
  	cout<<"Finish all requests"<<endl;
  }
}


int main(int argc, char * argv[])
{
	int provided;

/*
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE)
	{
	    printf("ERROR: The MPI library does not have full thread support\n");
	    MPI_Abort(MPI_COMM_WORLD, 1);
	}
*/
	if(argc !=2)
	{
		printf("usage:./test_graph config_file\n");
		return -1;
	}
	load_global_cfg(argv[1]);
	batch_factor=global_batch_factor;
	server_num=global_num_server;
	client_num=global_num_client;
	thread_num=server_num+client_num;

	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	uint64_t rdma_size = 1024*1024*1024;  //1G
	rdma_size = rdma_size*14; //2G 
	//rdma_size = rdma_size*2; //2G 
  	
  	uint64_t slot_per_thread= 1024*1024*64;
  	uint64_t total_size=rdma_size+slot_per_thread*thread_num*2; 
	Network_Node *node = new Network_Node(world.rank(),thread_num);//[0-thread_num-1] are used
	char *buffer= (char*) malloc(total_size);
	memset(buffer,0,total_size);
	RdmaResource *rdma=new RdmaResource(world.size(),thread_num,world.rank(),buffer,total_size,slot_per_thread,rdma_size);
	rdma->node = node;
	rdma->Servicing();
	rdma->Connect();

	uint64_t *local_buffer = (uint64_t *)rdma->GetMsgAddr(0);
  	uint64_t start_addr=0;
  	
	thread_cfg* cfg_array= new thread_cfg[thread_num];
	for(int i=0;i<thread_num;i++){
		cfg_array[i].t_id=i;
		cfg_array[i].t_num=thread_num;
		cfg_array[i].m_id=world.rank();
		cfg_array[i].m_num=world.size();
		cfg_array[i].client_num=client_num;
		cfg_array[i].server_num=server_num;
		cfg_array[i].rdma=rdma;
		cfg_array[i].node=new Network_Node(cfg_array[i].m_id,cfg_array[i].t_id);
		cfg_array[i].init();
	}
	index_server is(global_input_folder.c_str());
	//init of index_server and graph shouldn't be reordered!
	//we will set the global_rdftype_id in index_server

	graph g(world,rdma,global_input_folder.c_str());
	trinity_client** client_array=new trinity_client*[client_num];
	MPI_Barrier(MPI_COMM_WORLD);
	for(int i=0;i<client_num;i++){
		client_array[i]=new trinity_client(&is,&cfg_array[i]);
	}

	trinity_server** traverser_array=new trinity_server*[server_num];
	for(int i=0;i<server_num;i++){
		traverser_array[i]=new trinity_server(g,&cfg_array[client_num+i]);
	}
	
	pthread_t     *thread  = new pthread_t[thread_num];
	for(size_t id = 0;id < thread_num;++id) {
		if(id<client_num){
			cfg_array[id].ptr=client_array[id];
		} else {
			cfg_array[id].ptr=traverser_array[id-client_num];
		}
		pthread_create (&(thread[id]), NULL, Run, (void *) &(cfg_array[id]));
    }
    for(size_t t = 0 ; t < thread_num; t++) {
      int rc = pthread_join(thread[t], NULL);
      if (rc) {
        printf("ERROR; return code from pthread_join() is %d\n", rc);
        exit(-1);
      }
    }


    return 0;
}
