
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

//const int batch_factor=100;

//query 2 is a patten matching query
//query 4 is not complete , need to read attributes of every vertex
//query 8 is not complete , need to read attributes of every vertex
//query 9 is a patten matching query

void query1(client* is); 
//shape: 1->2.52, but actually, avg neighbor=12.5,because of some filter operation 
void query3(client* is);
//shape: 1->17.5
void query4(client* is);
void query5(client* is);
void query6(client* is);
void query7(client* is);
void query8(client* is);
void query10(client* is);

void interactive_mode(client* is){
	while(true){
		cout<<"interactive mode:"<<endl;
		string filename;
		cin>>filename;
		ifstream file(filename);
		if(!file)
			cout<<"File "<<filename<<" not exist"<<endl;

		string cmd;
		while(file>>cmd){
			if(cmd=="lookup"){
				string object;
				file>>object;
				is->lookup(object);
			} else if(cmd=="neighbors"){
				string dir,object;
				file>>dir>>object;
				is->neighbors(dir,object);
			} else if(cmd=="triangle"){
				vector<string> dir_vec;
				vector<string> predict_vec;
				string type1;
				string type2;
				file>>type1>>type2;
				dir_vec.resize(3);
				predict_vec.resize(3);
				for(int i=0;i<3;i++){
					file>>dir_vec[i]>>predict_vec[i];
				}
				is->triangle(type1,type2,dir_vec,predict_vec);
			} else if(cmd=="subclass_of"){
				string object;
				file>>object;
				is->subclass_of(object);
			} else if(cmd=="get_attr"){
				string object;
				file>>object;
				is->get_attr(object);
			} else if(cmd=="execute"){
				uint64_t t1=timer::get_usec();
				is->Send();
				is->Recv();
				uint64_t t2=timer::get_usec();
				cout<<"result size:"<<is->req.path_num()<<endl;
				cout<<t2-t1<<"us"<<endl;
				// for(int i=0;i<min(5,is->req.path_num());i++){
				// 	cout<<"row "<<i<<endl;
				// 	for(int column=0;column< is->req.path_length();column++){
				// 		int id=is->req.get_node(i,column).id;
				// 		cout<<is->is->id_to_subject[id]<<"\t";
				// 	}
				// 	cout<<endl;
				// }
				break;
			} else {
				cout<<"error cmd"<<endl;
				break;
			}
		}
	}
}

void batch_mode(client* is,struct thread_cfg *cfg){
	get_ids(is,"get_university_id");
	get_ids(is,"get_department_id");
	get_ids(is,"get_AssistantProfessor_id");
	get_ids(is,"get_AssociateProfessor_id");
	get_ids(is,"get_GraduateCourse_id");
	while(true){
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
  		MPI_Barrier(MPI_COMM_WORLD);
		uint64_t t1=timer::get_usec();
		//only support one client now
		ifstream file(filename);
		if(!file){
			cout<<"File "<<filename<<" not exist"<<endl;
			continue;
		}
		string cmd;
		file>>cmd;
		vector<uint64_t> ids=get_ids(is,cmd);
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
		if(ids.size()==0){
			cout<<"id set is empty..."<<endl;
			exit(0);
		}
		uint64_t t2=timer::get_usec();	    
		cout<<"start executing in "<<(t2-t1)/1000.0<<" ms ..."<<endl;
		//if(cfg->m_id<1){
		batch_execute(is,cfg,total_request,ids,cmd_chain);
		//}
		///// batch request has two part
		
	}
}

void tuning_mode(client* is,struct thread_cfg *cfg){
	get_ids(is,"get_university_id");
	get_ids(is,"get_department_id");
	get_ids(is,"get_AssistantProfessor_id");
	get_ids(is,"get_AssociateProfessor_id");
	get_ids(is,"get_GraduateCourse_id");
	MPI_Barrier(MPI_COMM_WORLD);
	string filename;
	if(cfg->m_id==0 && cfg->t_id==0){
		cout<<"tuning mode:"<<endl;
		cin>>filename;
		for(int i=1;i<cfg->m_num;i++){
			cfg->node->Send(i,0,filename);
		}
	} else {
		filename=cfg->node->Recv();
	}
	ifstream file(filename);
	if(!file){
		cout<<"File "<<filename<<" not exist"<<endl;
		//continue;
	}
	string cmd;
	file>>cmd;
	vector<uint64_t> ids=get_ids(is,cmd);
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
	if(ids.size()==0){
		cout<<"id set is empty..."<<endl;
		exit(0);
	}
	int count=0;
	while(true){		
		//if(cfg->m_id<1){
		if(global_tuning_threshold>5000){
			global_tuning_threshold-=500;
		} else if(global_tuning_threshold>1000){
			global_tuning_threshold-=100;
		} else if(global_tuning_threshold>500){
			global_tuning_threshold-=50;
		} else if(global_tuning_threshold>20){
			global_tuning_threshold-=10;
		}
		if(global_tuning_threshold==20){
			count++;
			if(count==5)
				exit(0);
		}
		MPI_Barrier(MPI_COMM_WORLD);
		if(cfg->m_id==0){
			//cout<<global_tuning_threshold<<"  == global_tuning_threshold "<<endl;
		}
		batch_execute(is,cfg,total_request,ids,cmd_chain);
		//}
		///// batch request has two part
		
	}
}
void* Run(void *ptr) {
  struct thread_cfg *cfg = (struct thread_cfg*) ptr;
  pin_to_core(socket_1[cfg->t_id]);

  if(cfg->t_id >= cfg->client_num){
  	((traverser*)(cfg->ptr))->run();
  }else {
  	while(global_interactive){
  		if(cfg->m_id!=0 || cfg->t_id!=0){
  			// sleep forever
  			sleep(1);
  		}
  		else{
  			interactive_mode((client*)(cfg->ptr));
  		}
  	}
  	//batch_mode((client*)(cfg->ptr),cfg);
	tuning_mode((client*)(cfg->ptr),cfg);

  	void(* query_array[11])(client*);
  	for(int i=0;i<=10;i++)
  		query_array[i]=NULL;
  	query_array[1]=	query1;
  	query_array[3]=	query3;
  	query_array[4]=	query4;
  	query_array[5]=	query5;
  	query_array[6]=	query6;
  	query_array[7]=	query7;
  	query_array[8]=	query8;
  	query_array[10]=query10;
  	sleep(1);
  	if(global_query_type<1 || global_query_type >10 || query_array[global_query_type]==NULL){
  		cout<<"Error Query "<<endl;
  		assert(false);
  	}
  	
  	query_array[global_query_type]((client*)(cfg->ptr));
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
	//batch_factor=atoi(argv[2]);
	//server_num=4;
	//client_num=1;
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
	//memset(buffer,0,total_size);
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
	MPI_Barrier(MPI_COMM_WORLD);
	for(int i=0;i<client_num;i++){
		client_array[i]=new client(&is,&cfg_array[i]);
	}

	traverser** traverser_array=new traverser*[server_num];
	concurrent_request_queue crq;
	for(int i=0;i<server_num;i++){
		traverser_array[i]=new traverser(g,crq,&cfg_array[client_num+i]);
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


void query1(client* is){
	request r=is->get_subtype("<ub#Course>")
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
			.execute()
			.req;
	vector<path_node> * vec_ptr=r.last_level();
	if(vec_ptr!=NULL){
		for(int i=0;i<batch_factor;i++){
			is->lookup_id((*vec_ptr)[0].id)
				.neighbors("in","<ub#takesCourse>")
				.subclass_of("<ub#GraduateStudent>")
				.Send();
		}
		for(int times=0;times<100;times++){
			for(int i=0;i<(*vec_ptr).size();i++){
				is->Recv();
				is->lookup_id((*vec_ptr)[i].id)
					.neighbors("in","<ub#takesCourse>")
					.subclass_of("<ub#GraduateStudent>")
					.Send();
			}
		}
		for(int i=0;i<batch_factor;i++){
			is->Recv();
		}
	}
}

void query3(client* is){
//	request r=is->get_subtype("<ub#Professor>")
	request r=is->get_subtype("<ub#FullProfessor>")
//	request r=is->get_subtype("<ub#AssociateProfessor>")
//	request r=is->get_subtype("<ub#AssistantProfessor>")	//RealQuery
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
			.execute()
			.req;
	vector<path_node> * vec_ptr=r.last_level();
	if(vec_ptr!=NULL){
		for(int i=0;i<batch_factor;i++){
			is->lookup_id((*vec_ptr)[0].id)
				.neighbors("in","<ub#publicationAuthor>")
				.subclass_of("<ub#Publication>")
				.Send();
		}
		for(int times=0;times<100;times++){
			for(int i=0;i<(*vec_ptr).size();i++){
				is->Recv();
				is->lookup_id((*vec_ptr)[i].id)
					.neighbors("in","<ub#publicationAuthor>")
					.subclass_of("<ub#Publication>")
					.Send();
			}
		}
		for(int i=0;i<batch_factor;i++){
			is->Recv();
		}
	}
}

void query4(client* is){
	request r=is->get_subtype("<ub#Department>")
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
			.execute()
			.req;
	vector<path_node> * vec_ptr=r.last_level();
	if(vec_ptr!=NULL){
		for(int i=0;i<batch_factor;i++){
			is->lookup_id((*vec_ptr)[0].id)
				.neighbors("in","<ub#worksFor>")
				//.subclass_of("<ub#Professor>")
				.subclass_of("<ub#FullProfessor>")
				.get_attr("<ub#name>")
				.get_attr("<ub#emailAddress>")
				.get_attr("<ub#telephone>")
				.Send();
		}
		for(int times=0;times<100000;times++){
			for(int i=0;i<(*vec_ptr).size();i++){
				is->Recv();
				is->lookup_id((*vec_ptr)[i].id)
					.neighbors("in","<ub#worksFor>")
					//.subclass_of("<ub#Professor>")
					.subclass_of("<ub#FullProfessor>")
					.get_attr("<ub#name>")
					.get_attr("<ub#emailAddress>")
					.get_attr("<ub#telephone>")
					.Send();
			}
		}
		for(int i=0;i<batch_factor;i++){
			is->Recv();
		}
	}
}

void query5(client* is){
	request r=is->get_subtype("<ub#Department>")
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
			.execute()
			.req;
	vector<path_node> * vec_ptr=r.last_level();
	if(vec_ptr!=NULL){
		for(int i=0;i<batch_factor;i++){
			is->lookup_id((*vec_ptr)[0].id)
				.neighbors("in","<ub#memberOf>")
				.subclass_of("<ub#Person>")
				.Send();
		}
		for(int times=0;times<10000;times++){
			for(int i=0;i<(*vec_ptr).size();i++){
				is->Recv();
				is->lookup_id((*vec_ptr)[i].id)
					.neighbors("in","<ub#memberOf>")
					.subclass_of("<ub#Person>")
					.Send();
			}
		}
		for(int i=0;i<batch_factor;i++){
			is->Recv();
		}
	}
}


void query6(client* is){
	for(int i=0;i<batch_factor;i++){
		is->get_subtype("<ub#Student>")
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
			.Send();
	}
	for(int times=0;times<10000;times++){
		is->Recv();
		is->get_subtype("<ub#Student>")
		.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
		.Send();	
	}
	for(int i=0;i<batch_factor;i++){
		is->Recv();
	}
}

void query7(client* is){
//	request r=is->get_subtype("<ub#Professor>")
//	request r=is->get_subtype("<ub#FullProfessor>")
	request r=is->get_subtype("<ub#AssociateProfessor>")  //RealQuery
//	request r=is->get_subtype("<ub#AssistantProfessor>")
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
			.execute()
			.req;
	vector<path_node> * vec_ptr=r.last_level();
	if(vec_ptr!=NULL){
		for(int i=0;i<batch_factor;i++){
			is->lookup_id((*vec_ptr)[0].id)
				.neighbors("out","<ub#teacherOf>")
				//.subclass_of("<ub#Course>")
				.neighbors("in","<ub#takesCourse>")
				.subclass_of("<ub#Student>")
				.Send();
		}
		for(int times=0;times<100;times++){
			for(int i=0;i<(*vec_ptr).size();i++){
				is->Recv();
				is->lookup_id((*vec_ptr)[i].id)
					.neighbors("out","<ub#teacherOf>")
					//.subclass_of("<ub#Course>")
					.neighbors("in","<ub#takesCourse>")
					.subclass_of("<ub#Student>")
					.Send();
			}
		}
		for(int i=0;i<batch_factor;i++){
			is->Recv();
		}
	}
}


void query8(client* is){
	request r=is->get_subtype("<ub#University>")
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
			.execute()
			.req;
	vector<path_node> * vec_ptr=r.last_level();
	cout<<"University Number:"<<(*vec_ptr).size()<<endl;
	if(vec_ptr!=NULL){
		for(int i=0;i<batch_factor;i++){
			is->lookup_id((*vec_ptr)[0].id)
				.neighbors("in","<ub#subOrganizationOf>")
				.subclass_of("<ub#Department>")	
				.neighbors("in","<ub#memberOf>")
				.subclass_of("<ub#Student>")
				.get_attr("<ub#emailAddress>")
				.Send();
		}
		for(int times=0;times<100;times++){
			for(int i=0;i<(*vec_ptr).size();i++){
				is->Recv();
				is->lookup_id((*vec_ptr)[i].id)
					.neighbors("in","<ub#subOrganizationOf>")
					.subclass_of("<ub#Department>")	
					.neighbors("in","<ub#memberOf>")
					.subclass_of("<ub#Student>")
					.get_attr("<ub#emailAddress>")
					.Send();
			}
		}
		for(int i=0;i<batch_factor;i++){
			is->Recv();
		}
	}
	
}
void query10(client* is){
	request r=is->get_subtype("<ub#GraduateCourse>")
			.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
			.execute()
			.req;
	vector<path_node> * vec_ptr=r.last_level();
	if(vec_ptr!=NULL){
		for(int i=0;i<batch_factor;i++){
			is->lookup_id((*vec_ptr)[0].id)
				.neighbors("in","<ub#takesCourse>")
				.subclass_of("<ub#Student>")
				.Send();
		}
		for(int times=0;times<100;times++){
			for(int i=0;i<(*vec_ptr).size();i++){
				is->Recv();
				is->lookup_id((*vec_ptr)[i].id)
					.neighbors("in","<ub#takesCourse>")
					.subclass_of("<ub#Student>")
					.Send();
			}
		}
		for(int i=0;i<batch_factor;i++){
			is->Recv();
		}
	}
}
