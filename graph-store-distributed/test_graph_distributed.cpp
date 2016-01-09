
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <iostream>

#include "graph.h"
#include "traverser.h"
#include "index_server.h"
#include "client.h"
#include "network_node.h"
#include "rdma_resource.h"
#include "thread_cfg.h"
#include "global_cfg.h"
#include <pthread.h>
#include <sstream>
#include "latency_logger.h"
#include "batch_lubm.h"

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

void batch_execute(client* is,struct thread_cfg *cfg,string filename){
	ifstream file(filename);
	if(!file){
		cout<<"File "<<filename<<" not exist"<<endl;
		return ;
	}
	string cmd;
	file>>cmd;
	vector<uint64_t> ids=get_ids(is,cmd);
	if(ids.size()==0){
		cout<<"id set is empty..."<<endl;
		exit(0);
	}
	vector<string> cmd_chain;
	int total_request=0;
	while(file>>cmd){
		cmd_chain.push_back(cmd);
		if(cmd=="execute"){
			file>>cmd;
			total_request = atoi(cmd.c_str());
			total_request*=global_num_server;
			break;
		}
	}
	////start to send message
	latency_logger logger;
	request reply;
	for(int i=0;i<global_batch_factor;i++){
		is->lookup_id(ids[0]);
		if(!(is->parse_cmd_vector(cmd_chain))){
			cout<<"error cmd"<<endl;
			return ;
		}
		is->req.timestamp=timer::get_usec();
		is->Send();
	}
	unsigned int seed=cfg->m_id*cfg->t_num+cfg->t_id;
	uint64_t total_latency=0;
	uint64_t t1;
	uint64_t t2;
	for(int times=0;times<total_request;times++){
		reply=is->Recv();
		if(times==total_request/4){
			logger.start();
		}
		if(times==total_request/4 *3){
			logger.stop();
		}
		if(times>=total_request/4 && times<total_request/4*3){
			logger.record(reply.timestamp,timer::get_usec());
		}
		int i=rand_r(&seed) % ids.size();
		is->lookup_id(ids[i]);
		if(!(is->parse_cmd_vector(cmd_chain))){
			cout<<"error cmd"<<endl;
			return ;
		}
		is->req.timestamp=timer::get_usec();
		is->Send();
	}
	for(int i=0;i<global_batch_factor;i++){
		reply=is->Recv();
	}
	logger.print();
}
void interactive_execute(client* is,string filename,int execute_count){
	int sum=0;
	request reply;
	for(int i=0;i<execute_count;i++){
		ifstream file(filename);
		if(!file){
			cout<<"File "<<filename<<" not exist"<<endl;
			break;
		}
		string cmd;
		vector<string> cmd_vec;
		while(file>>cmd){
			cmd_vec.push_back(cmd);
		}
		if(is->parse_cmd_vector(cmd_vec)){
			uint64_t t1=timer::get_usec();
			is->Send();
			reply=is->Recv();
			uint64_t t2=timer::get_usec();
			sum+=t2-t1;
		} else {
			cout<<"error cmd"<<endl;
		}
	}
	cout<<"result size:"<<reply.row_num()<<endl;
	cout<<"average latency "<<sum/execute_count<<" us"<<endl;
}
void interactive_mode(client* is,struct thread_cfg *cfg){
	if(cfg->m_id!=0 || cfg->t_id!=0){
		return ;
	}
	while(true){
		cout<<"interactive mode:"<<endl;
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
		interactive_execute(is,filename,execute_count);
	}
}
void batch_mode(client* is,struct thread_cfg *cfg){
	get_ids(is,"get_university_id");
	get_ids(is,"get_department_id");
	get_ids(is,"get_AssistantProfessor_id");
	get_ids(is,"get_AssociateProfessor_id");
	get_ids(is,"get_GraduateCourse_id");
	while(true){
		//// Wait for user input
		MPI_Barrier(MPI_COMM_WORLD);
		string filename;
		if(cfg->m_id==0 && cfg->t_id==0){
			cout<<"batch mode:"<<endl;
			cin>>filename;
			for(int i=1;i<cfg->m_num;i++){
				cfg->node->Send(i,0,filename);
			}
  		} else {
  			filename=cfg->node->Recv();
  		}
  		////prepare for all used ids
  		MPI_Barrier(MPI_COMM_WORLD);
  		batch_execute(is,cfg,filename);
	}//while loop, never end
}
void mixmode(client* is,struct thread_cfg *cfg){
	get_ids(is,"get_university_id");
	get_ids(is,"get_department_id");
	get_ids(is,"get_AssistantProfessor_id");
	get_ids(is,"get_AssociateProfessor_id");
	get_ids(is,"get_GraduateCourse_id");
	while(true){
		//// Wait for user input
		MPI_Barrier(MPI_COMM_WORLD);
		string batch_filename;
		string iterative_filename;
		int iterative_count=1;
		if(cfg->m_id==0 && cfg->t_id==0){
			cout<<"mix mode (batch file + iterative file):"<<endl;
			std::getline(std::cin,batch_filename);
			for(int i=1;i<cfg->m_num;i++){
				cfg->node->Send(i,0,batch_filename);
			}
			string input_str;
			std::getline(std::cin,input_str);
			istringstream iss(input_str);
			iss>>iterative_filename;
			iss>>iterative_count;
			if(iterative_count<1){
				iterative_count=1;
			}
  		} else {
  			batch_filename=cfg->node->Recv();
  		}
  		////prepare for all used ids
  		//MPI_Barrier(MPI_COMM_WORLD);
  		if(cfg->m_id==0){
  			interactive_execute(is,iterative_filename,iterative_count);
  		} else {
  			batch_execute(is,cfg,batch_filename);
  		}
	}//while loop, never end
}
void* Run(void *ptr) {
  struct thread_cfg *cfg = (struct thread_cfg*) ptr;
  pin_to_core(socket_1[cfg->t_id]);
  if(cfg->t_id >= cfg->client_num){
  	cout<<"("<<cfg->m_id<<","<<cfg->t_id<<")"<<endl;
  	((traverser*)(cfg->ptr))->run();
  }else {
  	cout<<"("<<cfg->m_id<<","<<cfg->t_id<<")"<<endl;
  	if(global_interactive){
  		interactive_mode((client*)(cfg->ptr),cfg);
  	} else {
  		//batch_mode((client*)(cfg->ptr),cfg);
  		mixmode((client*)(cfg->ptr),cfg);
  	}
  }
}


int main(int argc, char * argv[])
{
	int provided;

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
	rdma_size = rdma_size*25; //25G 
	
  	uint64_t slot_per_thread= 1024*1024*512;
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
	client** client_array=new client*[client_num];
	cout<<world.rank()<<" before barrier"<<endl;
	MPI_Barrier(MPI_COMM_WORLD);
	cout<<world.rank()<<" after barrier"<<endl;
	for(int i=0;i<client_num;i++){
		client_array[i]=new client(&is,&cfg_array[i]);
	}

	traverser** traverser_array=new traverser*[server_num];
	for(int i=0;i<server_num;i++){
		traverser_array[i]=new traverser(g,&cfg_array[client_num+i]);		
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
