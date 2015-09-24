
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <iostream>

#include "graph.h"
#include "traverser.h"
#include "index_server.h"
#include "network_node.h"
#include "rdma_resource.h"
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

struct Thread_config{
  int id;
  traverser* traverser_ptr;
  index_server* index_server_ptr;
};

const int batch_factor=100;

//query 2 is a patten matching query
//query 4 is not complex , need to read attributes of every vertex
//query 8 is not complex , need to read attributes of every vertex
//query 9 is a patten matching query

void query1(index_server* is);
void query3(index_server* is);
void query5(index_server* is);
void query6(index_server* is);
void query7(index_server* is);
void query8(index_server* is);
void query10(index_server* is);

void* Run(void *ptr) {
  struct Thread_config *config = (struct Thread_config*) ptr;
  pin_to_core(socket_0[config->id]);

  if(config->id!=0){
	config->traverser_ptr->run();
  }else {
  	//if(config->index_server_ptr->world.rank()!=0)
  	//	return (void* )NULL;
  	sleep(1);	
  	query1(config->index_server_ptr);

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

	if(argc !=2)
	{
		printf("usage:./test_graph dir\n");
		return -1;
	}
	boost::mpi::environment env(argc, argv);
	boost::mpi::communicator world;

	uint64_t rdma_size = 1024*1024*1024;  //1G
  	uint64_t slot_per_thread= 1024*1024*128;
  	//rdma_size = rdma_size*20; //20G 
  	uint64_t total_size=rdma_size+slot_per_thread*THREAD_NUM*2;
	Network_Node *node = new Network_Node(world.rank(),THREAD_NUM);
	char *buffer= (char*) malloc(total_size);
	RdmaResource *rdma=new RdmaResource(world.size(),THREAD_NUM,world.rank(),buffer,total_size,slot_per_thread,rdma_size);
	rdma->node = node;
	rdma->Servicing();
	rdma->Connect();

	uint64_t *local_buffer = (uint64_t *)rdma->GetMsgAddr(0);
  	uint64_t start_addr=0;
  	//rdma->RdmaRead(0,(world.rank()+1)%world.size() ,(char *)local_buffer,1024,start_addr);
  	//cout<<"Fucking OK"<<endl;

	graph g(world,rdma,argv[1]);
	index_server is(world,argv[1],rdma,0);
	traverser* traverser_array[TRAVERSER_NUM];
	concurrent_request_queue crq;
	for(int i=0;i<TRAVERSER_NUM;i++){
		traverser_array[i]=new traverser(world,g,crq,rdma,1+i);
	}
	Thread_config *configs = new Thread_config[THREAD_NUM];
  	pthread_t     *thread  = new pthread_t[THREAD_NUM];
	for(size_t id = 0;id < THREAD_NUM;++id) {
      configs[id].id = id;
      if(id==0)
      	configs[id].index_server_ptr=&is;
      else
      	configs[id].traverser_ptr=traverser_array[id-1];
      pthread_create (&(thread[id]), NULL, Run, (void *) &(configs[id]));
    }
    for(size_t t = 0 ; t < THREAD_NUM; t++) {
      int rc = pthread_join(thread[t], NULL);
      if (rc) {
        printf("ERROR; return code from pthread_join() is %d\n", rc);
        exit(-1);
      }
    }


	// if(world.rank() < world.size()-1){
	// 	graph g(world,argv[1]);
	// 	traverser t(world,g);
	// 	t.run();
	// } else {
	// 	index_server is(world,argv[1]);

	// 	//query 1
	// 	is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
	// 		.neighbors("in","<ub#takesCourse>")
	// 		.subclass_of("<ub#GraduateStudent>")
	// 		.execute()
	// 		.print_count();

	// 	//query 1
	// 	is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
	// 		.neighbors("in","<ub#takesCourse>")
	// 		.subclass_of("<ub#GraduateStudent>")
	// 		.execute()
	// 		.print_count();
	// 	//query 1
	// 	is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
	// 		.neighbors("in","<ub#takesCourse>")
	// 		//.subclass_of("<ub#GraduateStudent>")
	// 		.execute()
	// 		.print_count();
	// 	//query 1
	// 	is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
	// 		.neighbors("in","<ub#takesCourse>")
	// 		//.subclass_of("<ub#GraduateStudent>")
	// 		.execute()
	// 		.print_count();

		//query 2
		//TODO
	// 	{
	// 		request r1=	is.get_subtype("<ub#Department>")
	// 						.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
	// 						.neighbors("out","<ub#subOrganizationOf>")
	// 						.neighbors("in","<ub#undergraduateDegreeFrom>")
	// 						.execute()
	// 						.req;
	// 		request r2=	is.get_subtype("<ub#Department>")
	// 						.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
	// 						.neighbors("in","<ub#memberOf>")
	// 						.execute()
	// 						.req;
	// 		int c1=r1.path_num();
	// 		int c2=r2.path_num();
	// 		r1.merge(r2,2);
			
	// 		cout<<c1<<"*"<<c2<<"="<<r1.path_num()<<endl;
	// 	}
	// 	//query 3
	// 	is.lookup("<http://www.Department0.University0.edu/AssistantProfessor0>")
	// 		.neighbors("in","<ub#publicationAuthor>")
	// 		.subclass_of("<ub#Publication>")
	// 		.execute()
	// 		.print_count();

	// 	//query 4
	// 	is.lookup("<http://www.Department0.University0.edu>")
	// 		.neighbors("in","<ub#worksFor>")
	// 		.subclass_of("<ub#Professor>")
	// 		.execute()
	// 		.print_count();
		
	// 	//query 5
	// 	is.lookup("<http://www.Department0.University0.edu>")
	// 		.neighbors("in","<ub#memberOf>")
	// 		.subclass_of("<ub#Person>")
	// 		.execute()
	// 		.print_count();

	// 	//query 6
	// 	is.get_subtype("<ub#Student>")
	// 		.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
	// 		.execute()
	// 		.print_count();

	// 	//query 7
	// 	is.lookup("<http://www.Department0.University0.edu/AssociateProfessor0>")
	// 			.neighbors("out","<ub#teacherOf>")
	// 			.subclass_of("<ub#Course>")
	// 			.neighbors("in","<ub#takesCourse>")
	// 			.subclass_of("<ub#Student>")
	// 			.execute()
	// 			.print_count();

	// 	//query 8
				
	// 	is.lookup("<http://www.University0.edu>")
	// 		.neighbors("in","<ub#subOrganizationOf>")
	// 		.subclass_of("<ub#Department>")	
	// 		.neighbors("in","<ub#memberOf>")
	// 		.subclass_of("<ub#Student>")
	// 		.execute()
	// 		.print_count();

	// 	//query 9
	// 	//TODO
	// 	{
	// 		cout<<"Query 9-1 :"<<endl;
	// 		cout<<"Faculty (teacherOf)-> Course <-(takesCourse) Student"<<endl;
	// 		cout<<"Faculty <-(advisor) Student"<<endl;
	// 		request r1=	is.get_subtype("<ub#Faculty>")
	// 						.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
	// 						.neighbors("out","<ub#teacherOf>")
	// 						.neighbors("in","<ub#takesCourse>")
	// 						.execute()
	// 						.req;
	// 		request r2=	is.get_subtype("<ub#Faculty>")
	// 						.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
	// 						.neighbors("in","<ub#advisor>")
	// 						.execute()
	// 						.req;
	// 		int c1=r1.path_num();
	// 		int c2=r2.path_num();
	// 		r1.merge(r2,2);
	// 		cout<<c1<<"*"<<c2<<"="<<r1.path_num()<<endl;
	// 	}
	// 	{
	// 		cout<<"Query 9-2 :"<<endl;
	// 		cout<<"Faculty <-(advisor) Student (takesCourse)-> Course"<<endl;
	// 		cout<<"Faculty (teacherOf)-> Course "<<endl;
	// 		request r1=	is.get_subtype("<ub#Faculty>")
	// 						.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
	// 						.neighbors("in","<ub#advisor>")
	// 						.neighbors("out","<ub#takesCourse>")
	// 						.execute()
	// 						.req;
	// 		request r2=	is.get_subtype("<ub#Faculty>")
	// 						.neighbors("in","<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
	// 						.neighbors("out","<ub#teacherOf>")
	// 						.execute()
	// 						.req;
	// 		int c1=r1.path_num();
	// 		int c2=r2.path_num();
	// 		r1.merge(r2,2);
	// 		cout<<c1<<"*"<<c2<<"="<<r1.path_num()<<endl;
	// 	}

	// 	//query 10
	// 	is.lookup("<http://www.Department0.University0.edu/GraduateCourse0>")
	// 		.neighbors("in","<ub#takesCourse>")
	// 		.subclass_of("<ub#Student>")
	// 		.execute()
	// 		.print_count();

	// 	cout<<"finish"<<endl;

	// }

    return 0;
}


void query1(index_server* is){
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

void query3(index_server* is){
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

void query4(index_server* is){
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

void query5(index_server* is){
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


void query6(index_server* is){
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

void query7(index_server* is){
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


void query8(index_server* is){
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
void query10(index_server* is){
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
