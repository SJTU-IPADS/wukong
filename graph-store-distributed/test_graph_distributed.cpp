
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


int two_socket[] = {
  1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
};


int socket_1[] = {
  1,3,5,0,11,13,15,2,4,6,8,10,12,14,16,18
};

// int socket_1[] = {
//   1,3,5,7,9,11,13,15,17,19,0,2,4,6,8,10,12,14,16,18
// };

int client_4_server_16[] = {
  17,19,16,18,1,3,5,7,9,11,13,15,0,2,4,6,8,10,12,14
};
int client_2_server_16[] = {
  17,16,1,3,5,7,9,11,13,15,0,2,4,6,8,10,12,14,18,19
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
	cout<<"result size:"<<reply.final_row_number<<endl;
	cout<<"average latency "<<sum/execute_count<<" us"<<endl;
}



void batch_execute(client* is,struct thread_cfg *cfg,string mix_config,latency_logger& logger){
	ifstream configfile(mix_config);
	if(!configfile){
		cout<<"File "<<mix_config<<" not exist"<<endl;
		return ;
	}
	int total_query_type;
	int total_request;
	int useless;
	configfile>>total_query_type>>total_request>>useless;
	vector<vector<uint64_t> > first_ids;
	vector<vector<string> > cmd_chains;
	vector<int > distribution;
	int distribution_sum=0;
	first_ids.resize(total_query_type);
	cmd_chains.resize(total_query_type);
	distribution.resize(total_query_type);
	logger.reserve(total_request);
	for(int i=0;i<total_query_type;i++){
		string filename;
		configfile>>filename;
		int current_dist;
		configfile>>current_dist;
		distribution_sum+=current_dist;
		distribution[i]=distribution_sum;
		ifstream file(filename);
		string cmd;
		file>>cmd;
		first_ids[i]=get_ids(is,cmd);
		if(first_ids[i].size()==0){
			cout<<"id set is empty..."<<endl;
			exit(0);
		}
		while(file>>cmd){
			cmd_chains[i].push_back(cmd);
		}
		file.close();
	}
	configfile.close();
	////start to send message
	//latency_logger logger;
	request reply;
	for(int i=0;i<global_batch_factor;i++){
		is->lookup_id(first_ids[0][0]);
		if(!(is->parse_cmd_vector(cmd_chains[0]))){
			cout<<"error cmd"<<endl;
			return ;
		}
		is->req.timestamp=timer::get_usec();
		is->req.type_id=0;
		is->Send();
	}
	unsigned int seed=cfg->m_id*cfg->t_num+cfg->t_id;
	logger.start();
	for(int times=0;times<total_request;times++){
		reply=is->Recv();
		logger.record(reply.timestamp,timer::get_usec(),reply.type_id);
		unsigned random_number=rand_r(&seed);
		//unsigned query_type=random_number%total_query_type;
		//random_number= random_number%distribution_sum;
		int i=0;
		unsigned query_type=0;
		while(true){
			assert(query_type<total_query_type);
			if( (random_number%distribution_sum) <distribution[query_type]){
				break;
			}
			query_type++;
		}
		//cout<<query_type<<endl;
		

		unsigned idx=(random_number/total_query_type) % first_ids[query_type].size();
		
		is->lookup_id(first_ids[query_type][idx]);
		if(!(is->parse_cmd_vector(cmd_chains[query_type]))){
			cout<<"error cmd"<<endl;
			return ;
		}
		is->req.timestamp=timer::get_usec();
		is->req.type_id=query_type;
		is->Send();
	}
	logger.stop();
	for(int i=0;i<global_batch_factor;i++){
		reply=is->Recv();
	}
	//logger.print();
}



void noblocking_execute(client* is,struct thread_cfg *cfg,string mix_config,latency_logger& logger){
	ifstream configfile(mix_config);
	if(!configfile){
		cout<<"File "<<mix_config<<" not exist"<<endl;
		return ;
	}
	int total_query_type;
	int total_request;
	int sleep_round=1;
	configfile>>total_query_type>>total_request>>sleep_round;
	vector<vector<uint64_t> > first_ids;
	vector<vector<string> > cmd_chains;
	vector<int > distribution;
	int distribution_sum=0;
	first_ids.resize(total_query_type);
	cmd_chains.resize(total_query_type);
	distribution.resize(total_query_type);
	logger.reserve(total_request);
	for(int i=0;i<total_query_type;i++){
		string filename;
		configfile>>filename;
		int current_dist;
		configfile>>current_dist;
		distribution_sum+=current_dist;
		distribution[i]=distribution_sum;
		ifstream file(filename);
		string cmd;
		file>>cmd;
		first_ids[i]=get_ids(is,cmd);
		if(first_ids[i].size()==0){
			cout<<"id set is empty..."<<endl;
			exit(0);
		}
		while(file>>cmd){
			cmd_chains[i].push_back(cmd);
		}
		file.close();
	}
	configfile.close();
	////start to send message
	unsigned int seed=cfg->m_id*cfg->t_num+cfg->t_id;
	logger.start();
	int waiting_request=total_request;
	
	while(total_request>0){
		//count++;
		for(int i=0;i<batch_factor;i++){
			//if(waiting_request>0 && count%100< request_per_ms)
			if(waiting_request>0 )
			{
				
				unsigned random_number=rand_r(&seed);
				unsigned query_type=0;
				while(true){
					assert(query_type<total_query_type);
					if( (random_number%distribution_sum) <distribution[query_type]){
						break;
					}
					query_type++;
				}
				unsigned idx=(random_number/total_query_type) % first_ids[query_type].size();

				is->lookup_id(first_ids[query_type][idx]);
				if(!(is->parse_cmd_vector(cmd_chains[query_type]))){
					cout<<"error cmd"<<endl;
					return ;
				}
				is->req.timestamp=timer::get_usec();
				is->req.type_id=query_type;
				//is->req.type_id=0;

				is->Send();
				waiting_request--;
			}
		}
		for(int i=0;i<sleep_round;i++){
			usleep(batch_factor);
			request reply;
			bool success=TryRecvReq(cfg,reply);
			while(success){
				total_request--;
				logger.record(reply.timestamp,timer::get_usec(),reply.type_id);
				success=TryRecvReq(cfg,reply);
			}
		}
	}
	logger.stop();
}

int global_barrier_val;
void ClientBarrier(struct thread_cfg *cfg,int last_barrier){
	//cout<<"Client "<<cfg->t_id<<" wish to enter a barrier"<<endl;
	if(cfg->t_id==0){
		MPI_Barrier(MPI_COMM_WORLD);
		__sync_fetch_and_add(&global_barrier_val,1);
	}  else {
		__sync_fetch_and_add(&global_barrier_val,1);
	}
	while(global_barrier_val !=last_barrier+cfg->client_num){
		usleep(1);
	}
	//cout<<"Client "<<cfg->t_id<<" leave the barrier"<<endl;
}
void run_client(client* is,struct thread_cfg *cfg){
	if(cfg->t_id==0){
		get_ids(is,"get_university_id");
		get_ids(is,"get_department_id");
		get_ids(is,"get_AssistantProfessor_id");
		get_ids(is,"get_AssociateProfessor_id");
		get_ids(is,"get_GraduateCourse_id");
	}
	if(cfg->m_id==0 && cfg->t_id==0){
		cout<<"switch_iterative ->	iterative mode"<<endl;
		cout<<"switch_batch 	->	batch mode"<<endl;
		cout<<"switch_mix 		->	mix mode"<<endl;
	}
	if(global_client_mode<0|| global_client_mode>=3){
		global_client_mode=0;
	}
	string mode_str[3];
	mode_str[0]="iterative mode (iterative file + [count]):";
	mode_str[1]="batch mode (batch file):";
	mode_str[2]="mix mode (batch file + iterative file + [count]):";
	int last_barrier=0;
	while(true){
		ClientBarrier(cfg,last_barrier);
		last_barrier+=cfg->client_num;

		string batch_filename;
		string iterative_filename;
		latency_logger logger;
		int iterative_count=1;
		if(cfg->m_id==0 && cfg->t_id==0){
			string input_str;
			while(true){
				cout<<mode_str[global_client_mode]<<endl;;
				std::getline(std::cin,input_str);
				if(input_str.find("switch_iterative")!=string::npos){
					global_client_mode=0;
					continue;
				} else if(input_str.find("switch_batch")!=string::npos){
					global_client_mode=1;
					continue;
				} else if(input_str.find("switch_mix")!=string::npos){
					global_client_mode=2;
					continue;
				} else {
					break;
				}
			}
			istringstream iss(input_str);	
			if(global_client_mode==0){
				iss>>iterative_filename;
				iss>>iterative_count;
				if(iterative_count<1){
					iterative_count=1;
				}
				batch_filename="NO_BATCH_FILE";
			} else if(global_client_mode==1){
				iss>>batch_filename;
			} else if(global_client_mode==2){
				iss>>batch_filename;
				iss>>iterative_filename;
				iss>>iterative_count;
				if(iterative_count<1){
					iterative_count=1;
				}
			}
			if(batch_filename==""){
				batch_filename="empty";
			}
			for(int i=0;i<cfg->m_num;i++){
				for(int j=0;j<cfg->client_num;j++){
					if(i==0 && j==0){
						continue;
					}
					cfg->rdma->rbfSend(cfg->t_id,i, j, batch_filename.c_str(),batch_filename.size()); 
					//cfg->node->Send(i,j,batch_filename);
				}
			}
  		} else {
  			//batch_filename=cfg->node->Recv();
  			batch_filename=cfg->rdma->rbfRecv(cfg->t_id);
  			if(batch_filename=="NO_BATCH_FILE"){
  				continue;
  			}
		}

  		if(cfg->m_id==0 && cfg->t_id==0 && (global_client_mode==0 || global_client_mode==2) ){
  			sleep(1);
  			interactive_execute(is,iterative_filename,iterative_count);
  		} else {
  			batch_execute(is,cfg,batch_filename,logger);
  			//noblocking_execute(is,cfg,batch_filename,logger);
  			//batch_execute(is,cfg,batch_filename);
  		}

  		if(batch_filename!="NO_BATCH_FILE"){
  			ClientBarrier(cfg,last_barrier);
  			last_barrier+=cfg->client_num;
  			if(cfg->m_id==0 && cfg->t_id==0){
  				for(int i=0;i<cfg->m_num;i++){
					for(int j=0;j<cfg->client_num;j++){
						if(i==0 && j==0){
							continue;
						}
						latency_logger r=RecvLog(cfg);
						logger.merge(r);
					}
				}
				logger.print();
			} else {
				//logger.print();
  				SendLog(cfg,0,0,logger);
  			}
  		} else {
  			cout<<"NO_BATCH_FILE"<<endl;
  		}

	}
}
void* Run(void *ptr) {
  struct thread_cfg *cfg = (struct thread_cfg*) ptr;
  // if(cfg->client_num==4 &&  cfg->server_num==16){
  // 	pin_to_core(client_4_server_16[cfg->t_id]);
  // } else if(cfg->client_num==2 &&  cfg->server_num==16){
  // 	pin_to_core(client_2_server_16[cfg->t_id]);
  // } else {
  // 	pin_to_core(socket_1[cfg->t_id]);
  // }
  pin_to_core(socket_1[cfg->t_id]);
  if(cfg->t_id >= cfg->client_num){
  	cout<<"("<<cfg->m_id<<","<<cfg->t_id<<")"<<endl;
  	((traverser*)(cfg->ptr))->run();
  }else {
  	cout<<"("<<cfg->m_id<<","<<cfg->t_id<<")"<<endl;
  	run_client((client*)(cfg->ptr),cfg);
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
	global_barrier_val=0;

	uint64_t rdma_size = 1024*1024*1024;  //1G
	//rdma_size = rdma_size*20; //25G 
	rdma_size = rdma_size*global_total_memory_gb; //25G 
	
  	uint64_t slot_per_thread= 1024*1024*global_perslot_msg_mb;
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
	per_thread_resource* res_array=new per_thread_resource[thread_num];
	traverser** traverser_array=new traverser*[server_num];
	for(int i=0;i<server_num;i++){
		traverser_array[i]=new traverser(g,res_array,&cfg_array[client_num+i],cfg_array);		
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
