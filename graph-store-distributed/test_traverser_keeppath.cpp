
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
#include <pthread.h>


int socket_0[] = {
  0,2,4,6,8,10,12,14,16,18
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
void query3(client* is);
void query5(client* is);
void query6(client* is);
void query7(client* is);
void query8(client* is);
void query10(client* is);


void* Run(void *ptr) {
  struct thread_cfg *cfg = (struct thread_cfg*) ptr;
  pin_to_core(socket_0[cfg->t_id]);

  if(cfg->t_id >= cfg->client_num){
  	((traverser*)(cfg->ptr))->run();
  }else {
  	sleep(1);	
  	query1((client*)(cfg->ptr));
  	cout<<"Finish all requests"<<endl;
  }
}


int main(int argc, char * argv[])
{
	int provided;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE)
	{
	    printf("ERROR: The MPI library does not have full thread support\n");
	    MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if(argc !=3)
	{
		printf("usage:./test_graph dir batch_factor\n");
		return -1;
	}
	batch_factor=atoi(argv[2]);
	server_num=4;
	client_num=2;
	thread_num=server_num+client_num;

	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	uint64_t rdma_size = 1024*1024*1024;  //1G
  	uint64_t slot_per_thread= 1024*1024*128;
  	//rdma_size = rdma_size*20; //20G 
  	uint64_t total_size=rdma_size+slot_per_thread*thread_num*2; 
	Network_Node *node = new Network_Node(world.rank(),thread_num);//[0-thread_num-1] are used
	char *buffer= (char*) malloc(total_size);
	RdmaResource *rdma=new RdmaResource(world.size(),thread_num,world.rank(),buffer,total_size,slot_per_thread,rdma_size);
	rdma->node = node;
	rdma->Servicing();
	rdma->Connect();

	uint64_t *local_buffer = (uint64_t *)rdma->GetMsgAddr(0);
  	uint64_t start_addr=0;
  	//rdma->RdmaRead(0,(world.rank()+1)%world.size() ,(char *)local_buffer,1024,start_addr);
  	//cout<<"Fucking OK"<<endl;
	
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
	graph g(world,rdma,argv[1]);
	index_server is(argv[1]);
	client** client_array=new client*[client_num];
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
//	request r=is->get_subtype("<ub#AssistantProfessor>")
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
				.subclass_of("<ub#Professor>")
				.Send();
		}
		for(int times=0;times<100;times++){
			for(int i=0;i<(*vec_ptr).size();i++){
				is->Recv();
				is->lookup_id((*vec_ptr)[i].id)
					.neighbors("in","<ub#worksFor>")
					.subclass_of("<ub#Professor>")
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
		for(int times=0;times<100;times++){
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
//	request r=is->get_subtype("<ub#AssociateProfessor>")
	request r=is->get_subtype("<ub#AssistantProfessor>")
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
	if(vec_ptr!=NULL){
		for(int i=0;i<batch_factor;i++){
			is->lookup_id((*vec_ptr)[0].id)
				.neighbors("in","<ub#subOrganizationOf>")
				.subclass_of("<ub#Department>")	
				.neighbors("in","<ub#memberOf>")
				.subclass_of("<ub#Student>")
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
